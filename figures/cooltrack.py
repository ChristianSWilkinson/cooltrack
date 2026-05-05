import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.spatial import cKDTree
from scipy.interpolate import BSpline

# Suppress annoying scipy optimize warnings for clean terminal output
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")


class LocalLinearRegressor:
    """
    Smooth, gradient-aware KD-Tree interpolator with Ridge Regularization.
    Fits a local linear hyperplane to the K-nearest neighbors, utilizing an L2 
    penalty to mathematically suppress high-frequency wiggles and over-fitting.
    """
    def __init__(self, n_neighbors=15, ridge_penalty=1e-3, softening=1e-2):
        self.k = n_neighbors
        self.ridge_penalty = ridge_penalty # L2 regularization term (lambda)
        self.softening = softening         # Prevents infinite weights at d=0
        self.tree = None
        self.X = None
        self.y = None
        self.weights = None

    def fit(self, X, y, weights=None):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.weights = np.asarray(weights) if weights is not None else np.ones(len(y))
        
        self.k = min(self.k, len(self.X))
        self.tree = cKDTree(self.X)
        return self

    def predict(self, X_query):
        X_query = np.asarray(X_query)
        y_pred = np.zeros(len(X_query))

        distances, indices = self.tree.query(X_query, k=self.k)
        
        if self.k < 3:
            for i, idx in enumerate(indices):
                y_pred[i] = np.average(self.y[idx], weights=self.weights[idx])
            return y_pred

        # Identity matrix for the Ridge penalty (matches the dimension of X)
        I = np.eye(self.X.shape[1])

        for i, x_q in enumerate(X_query):
            idx = indices[i]
            d_local = distances[i]
            
            X_loc = self.X[idx]
            y_loc = self.y[idx]
            w_grid = self.weights[idx]
            
            # Softened distance weight to prevent asymptotic spikes
            w_dist = 1.0 / (d_local + self.softening) 
            w_total = w_grid * w_dist
            
            X_mean = np.average(X_loc, axis=0, weights=w_total)
            y_mean = np.average(y_loc, weights=w_total)
            
            X_centered = X_loc - X_mean
            y_centered = y_loc - y_mean
            
            W = np.diag(w_total)
            XTW = X_centered.T @ W
            
            # Weighted Least Squares WITH L2 Ridge Regularization
            XTWX_ridge = (XTW @ X_centered) + (self.ridge_penalty * I)
            
            Beta = np.linalg.pinv(XTWX_ridge) @ (XTW @ y_centered)
            
            y_pred[i] = y_mean + np.dot(Beta, (x_q - X_mean))
            
        return y_pred

def softplus_piecewise(x, x0, y0, k1, k2, beta):
    """
    Piecewise softplus for Radius and Photometry.
    Beta is a fit parameter controlling the sharpness of the transition knee.
    """
    x = np.asarray(x)
    z = beta * (x - x0)
    # np.logaddexp prevents overflow for large z
    soft_term = np.logaddexp(0, z) - np.log(2) 
    return y0 + k1 * (x - x0) + ((k2 - k1) / beta) * soft_term

def weighted_r2(y_true, y_pred, weights):
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - np.average(y_true, weights=weights))**2)
    if ss_tot == 0: return 0.0
    return 1 - (ss_res / ss_tot)


def generalized_sigmoid(x, x0, y_floor, amplitude, k, nu):
    """
    Richards' Curve (Asymmetric Sigmoid) to model Boltzmann transitions in log-space.
    The 'nu' parameter allows the curve to bend asymmetrically, preventing it from 
    bowing above or below the physical data points during the transition.
    """
    x = np.asarray(x)
    # Clip the exponent to prevent overflow crashes
    z = np.clip(-k * (x - x0), -100, 100)
    return y_floor + (amplitude / ((1.0 + np.exp(z)) ** nu))

def sloped_sigmoid(x, x0, y_floor, amplitude, k, nu, m_hot):
    """
    Sloped Asymmetric Sigmoid.
    S = 0 at cold temperatures (leaving just y_floor).
    S = 1 at hot temperatures (leaving y_floor + amplitude + m_hot * delta_x).
    This perfectly captures the flat irradiation floor, the cloud transition jump, 
    and the continuous Stefan-Boltzmann flux rise at high temperatures!
    """
    x = np.asarray(x)
    z = np.clip(-k * (x - x0), -100, 100)
    
    # S scales from 0 (cold) to 1 (hot)
    S = 1.0 / ((1.0 + np.exp(z)) ** nu)
    
    # Apply slope ONLY to the hot side!
    return y_floor + S * (amplitude + m_hot * (x - x0))

def quadratic_sigmoid(x, y0, m1, m2, amp, k, x0):
    """
    Upgraded Analytical Photometry Fit.
    - m1 and m2 (Quadratic) capture the shifting Planck peak at high temperatures.
    - amp, k, and x0 (Sigmoid) capture the sharp L/T transition (cloud clearing).
    """
    # Clip the exponential to prevent math overflow warnings
    exp_arg = np.clip(-k * (x - x0), -100, 100)
    return y0 + m1 * x + m2 * (x**2) + (amp / (1.0 + np.exp(exp_arg)))

class SemiAnalyticalCoolTrack:
    """
    Core engine for generating semi-analytical planetary cooling tracks.
    Bypasses traditional ODE solvers by extracting analytical Softplus 
    and Linear surrogate models and numerically integrating them over time.
    """
    def __init__(self, grid_df: pd.DataFrame, initial_conditions_model, independent_dims: list, bandwidth: float = 0.5):
        self.independent_dims = independent_dims
        self.bandwidth = bandwidth
        self.grid_df = self._prepare_grid(grid_df)
        self.init_conds = initial_conditions_model
        
        self.scaler = StandardScaler()
        self.scaler.fit(self.grid_df[self.independent_dims])

    def _prepare_grid(self, df):
        """Cleans and calculates necessary log-space columns upon initialization."""
        work_df = df.copy()
        if 'dsdt' not in work_df.columns and 'dsdt_J_K_kg_s' in work_df.columns:
            work_df['dsdt'] = work_df['dsdt_J_K_kg_s']
            
        work_df = work_df[(work_df['T_int'] > 0) & (work_df['dsdt'] < 0) & (work_df['Req_Rj'] > 0)].copy()
        
        work_df['log10_mass_Mj'] = np.log10(work_df['mass_Mj'])
        work_df['ln_Tint'] = np.log(work_df['T_int'])
        work_df['ln_S'] = np.log(work_df['S_physical'])
        work_df['ln_tau'] = -np.log(np.abs(work_df['dsdt']))
        work_df['ln_Req'] = np.log(work_df['Req_Rj'])
        
        # Replace inf/-inf with NaN so curve_fit doesn't crash on truncated spectra
        work_df = work_df.replace([np.inf, -np.inf], np.nan)
        
        # Keep log photometry columns if they exist
        phot_cols = [c for c in work_df.columns if c.startswith('log_') and 'Flambda' in c]
        return work_df.dropna(subset=self.independent_dims + ['ln_Tint', 'ln_S', 'ln_tau', 'ln_Req'] + phot_cols).reset_index(drop=True)

    def _calculate_weights(self, target_planet: dict):
        """Calculates Gaussian proximity weights for a target planet."""
        target_copy = target_planet.copy()
        if 'mass_Mj' in target_copy:
            target_copy['log10_mass_Mj'] = np.log10(target_copy['mass_Mj'])
            
        target_df = pd.DataFrame([target_copy])[self.independent_dims]
        scaled_grid = self.scaler.transform(self.grid_df[self.independent_dims])
        scaled_target = self.scaler.transform(target_df)
        
        distances = np.linalg.norm(scaled_grid - scaled_target, axis=1)
        weights = np.exp(- (distances**2) / (2 * self.bandwidth**2))
        
        track = self.grid_df.copy()
        track['weight'] = weights
        track = track[track['weight'] > 0.05].copy() 
        
        if len(track) < 5:
            track = self.grid_df.copy()
            track['weight'] = weights
            track = track.sort_values('weight', ascending=False).head(50).copy()
            
        return track

    def fit_surrogate(self, target_planet: dict, photometry_bands: list = None, photometry_method: str = 'sigmoid'):
        """
        Extracts analytical constants for Thermodynamics, Cooling, Structure, and Photometry.
        photometry_method: 'sigmoid' (analytical) or 'locallinear' (KD-Tree fallback).
        """
        track = self._calculate_weights(target_planet)
        w_sqrt = np.sqrt(track['weight'])
        
        fits = {'target': target_planet, 'track_data': track, 'photometry': {}}

        # =========================================================
        # 1. Thermodynamics S(T_int) - STRICTLY LINEAR
        # =========================================================
        x_Tint = track['ln_Tint'].values
        y_S = track['ln_S'].values
        
        (fits['C'], fits['D']), fits['cov_S'] = np.polyfit(x_Tint, y_S, deg=1, w=w_sqrt, cov=True)
        fits['r2_S'] = weighted_r2(y_S, fits['C'] * x_Tint + fits['D'], track['weight'])

        # =========================================================
        # 2. Cooling Rate tau(T_int) - SOFTPLUS TO CAPTURE L/T PLATEAU
        # =========================================================
        x_Tint = track['ln_Tint'].values
        y_tau = track['ln_tau'].values
        
        # We need a robust initial guess. 
        # A simple linear fit provides the baseline slope.
        try:
            (A_lin, B_lin) = np.polyfit(x_Tint, y_tau, deg=1, w=w_sqrt)
            
            # 1. Define the physical bounds for the L/T Transition Knee (x0)
            # The transition typically occurs between 1000 K and 1500 K
            T_knee_min = 1000.0
            T_knee_max = 1500.0
            
            ln_T_knee_min = np.log(T_knee_min)
            ln_T_knee_max = np.log(T_knee_max)
            
            # Initial guess: [knee_location, y_offset, slope1, slope2, sharpness]
            p0_tau = [np.log(1300.0), np.median(y_tau), A_lin, A_lin * 0.8, 10.0]
            
            # Bounds: Ensure x0 strictly stays within the physical L/T window
            bounds_tau_lower = [ln_T_knee_min, np.min(y_tau)-2, -20.0, -20.0, 1.0]
            bounds_tau_upper = [ln_T_knee_max, np.max(y_tau)+2, 0.0, 0.0, 100.0]

            popt_tau, cov_tau = curve_fit(
                softplus_piecewise, x_Tint, y_tau, p0=p0_tau, 
                bounds=(bounds_tau_lower, bounds_tau_upper),
                sigma=1.0/(w_sqrt + 1e-9), method='trf', maxfev=5000
            )
            r2_tau = weighted_r2(y_tau, softplus_piecewise(x_Tint, *popt_tau), track['weight'])
            
            fits['method_tau'] = 'softplus'
            fits['popt_tau'] = popt_tau
            fits['cov_tau'] = cov_tau
            fits['r2_tau'] = r2_tau
            
        except Exception as e:
            # SANE FALLBACK: If the Softplus fails (e.g., at high masses where the 
            # transition fades and the optimizer can't find a distinct knee), 
            # gracefully fall back to the linear fit.
            (A, B), cov = np.polyfit(x_Tint, y_tau, deg=1, w=w_sqrt, cov=True)
            fits['method_tau'] = 'linear'
            fits['A'] = A
            fits['B'] = B
            fits['cov_tau'] = cov
            fits['r2_tau'] = weighted_r2(y_tau, A * x_Tint + B, track['weight'])
                
        # =========================================================
        # 3. Structural EoS Radius(S) - 5-PARAM SOFTPLUS
        # =========================================================
        x_S = track['ln_S'].values
        y_R = track['ln_Req'].values
        sigma_val = 1.0 / (w_sqrt + 1e-9)
        
        p0_R = [np.percentile(x_S, 30), np.percentile(y_R, 30), 0.01, 0.5, 10.0]
        bounds_R_lower = [np.min(x_S), np.min(y_R)-0.5, -0.1, 0.0, 1.0]
        bounds_R_upper = [np.max(x_S), np.max(y_R)+0.5, 0.5, 5.0, 50.0]
        
        try:
            popt_R, cov_R = curve_fit(
                softplus_piecewise, x_S, y_R, p0=p0_R, 
                bounds=(bounds_R_lower, bounds_R_upper),
                sigma=sigma_val, method='trf', maxfev=5000
            )
            r2_R = weighted_r2(y_R, softplus_piecewise(x_S, *popt_R), track['weight'])
        except Exception:
            try:
                (G, H), _ = np.polyfit(x_S, y_R, deg=1, w=w_sqrt, cov=True)
                popt_R, cov_R = [0, H, G, G, 10.0], np.eye(5) * 1e-4
                r2_R = weighted_r2(y_R, G * x_S + H, track['weight'])
            except Exception:
                popt_R, cov_R = [0, 0, 0, 0, 10.0], np.eye(5) * 1e-4
                r2_R = 0.0
                
        fits['popt_R'], fits['cov_R'], fits['r2_R'] = popt_R, cov_R, r2_R

        # =========================================================
        # 4. Photometry
        # =========================================================
        if photometry_bands:
            X_tree = np.vstack([track['ln_Tint'].values, track['T_irr'].values / 100.0]).T
            target_tirr = target_planet.get('T_irr', 0.0)
            
            for band in photometry_bands:
                
                # --- THE FIX: STRICT T_IRR SLICE FOR 1D ANALYTICAL FITTING ---
                # The KD-Tree is 2D and can handle mixed T_irr data.
                # 1D Analytical curves (Splines/Sigmoids) must ONLY be trained on points matching the target irradiation floor!
                if photometry_method in ['sigmoid', 'quadratic_sigmoid', 'bspline']:
                    strict_mask = (np.abs(track['T_irr'] - target_tirr) <= 25.0)
                    clean_mask = track[band].notna() & strict_mask
                else:
                    clean_mask = track[band].notna()
                    
                if clean_mask.sum() < 5:
                    fits['photometry'][band] = None
                    continue

                if photometry_method in ['sigmoid', 'quadratic_sigmoid', 'bspline']:
                    x_fit = x_Tint[clean_mask]
                    y_fit = track.loc[clean_mask, band].values
                    w_fit = np.sqrt(track.loc[clean_mask, 'weight'].values)
                    sigma_F = 1.0 / (w_fit + 1e-9)
                    
                    x_min, x_max = np.min(x_fit), np.max(x_fit)
                    
                    # =========================================================
                    # CUBIC B-SPLINE FITTER (Stable Error Propagation & Hook Tracking)
                    # =========================================================
                    from scipy.interpolate import BSpline
                    
                    # Create a fixed knot vector: 2 internal knots + clamped boundaries
                    # This yields exactly 6 control points (c0-c5)
                    internal_knots = np.linspace(x_min, x_max, 4)[1:-1] 
                    t_knots = np.concatenate(([x_min]*4, internal_knots, [x_max]*4))
                    
                    # Define a wrapper for curve_fit to optimize the 6 control points
                    def bspline_wrapper(x, c0, c1, c2, c3, c4, c5):
                        spl = BSpline(t_knots, [c0, c1, c2, c3, c4, c5], 3, extrapolate=True)
                        return spl(x)
                        
                    # A linearly increasing guess is very stable for luminosity tracks
                    p0_spl = np.linspace(np.min(y_fit), np.max(y_fit), 6)

                    # Force the optimizer to care 10x more about the hot boundary (the hook!)
                    hot_mask = x_fit > np.log(1200.0)
                    if np.any(hot_mask):
                        sigma_F[hot_mask] *= 0.1 

                    try:
                        # Splines don't strictly require bounds, making them incredibly fast & stable
                        popt_F, cov_F = curve_fit(
                            bspline_wrapper, x_fit, y_fit, p0=p0_spl, 
                            sigma=sigma_F, method='trf', maxfev=10000
                        )
                        r2_F = weighted_r2(y_fit, bspline_wrapper(x_fit, *popt_F), w_fit**2)
                    except Exception as e:
                        # --- SANE FALLBACK ---
                        popt_F = p0_spl
                        cov_F = np.eye(6) * 1e-4
                        r2_F = 0.0

                    fits['photometry'][band] = {
                        'method': 'bspline',
                        'popt': popt_F,
                        'cov': cov_F,
                        't_knots': t_knots,  # Crucial: save the knots for the evaluator
                        'r2': r2_F,
                        'variance': np.var(y_fit - bspline_wrapper(x_fit, *popt_F))
                    }

        return fits

    def evolve(self, fits: dict, start_type: int = 10, n_points: int = 500, n_draws: int = 1000, T_start_override: float = None):
        """
        Integrates the analytical surrogate model over time to produce a continuous evolutionary track.
        Outputs a DataFrame containing median evolution paths and 1-sigma confidence intervals.
        """
        if not fits.get('C'):
            raise ValueError("Invalid surrogate fits provided. Cannot evolve.")
            
        if 'mass_Mj' in fits['target']:
            mass = fits['target']['mass_Mj']
        else:
            mass = 10 ** fits['target'].get('log10_mass_Mj', 1.0)

        C, D = fits['C'], fits['D']
        
        # =========================================================
        # 1. Fetch Boundary Conditions (Entropy)
        # =========================================================
        if T_start_override is not None:
            ln_T0 = np.log(T_start_override)
            ln_S0 = C * ln_T0 + D
        elif start_type is not None:
            S0 = self.init_conds.get_starting_physical_entropy(mass_mjup=mass, bin_index=start_type)
            ln_S0 = np.log(S0)
            ln_T0 = (ln_S0 - D) / C
        else :
            raise ValueError("Engine initialized without 'initial_conditions_model'. Provide a 'T_start_override' ot start_type index to set the initial temperature.")
            
        T0 = np.exp(ln_T0)
        
        # =========================================================
        # 2. Define Integration Array (T_int Space)
        # =========================================================
        T_end = np.exp(fits['track_data']['ln_Tint'].min())
        T_int_arr = np.logspace(np.log10(T0), np.log10(T_end), n_points)
        ln_Tint = np.log(T_int_arr)
        
        # =========================================================
        # 3. Evaluate Median Thermodynamics & Cooling
        # =========================================================
        ln_S_median = C * ln_Tint + D
        S_median = np.exp(ln_S_median)
        
        # Adaptive Evaluation for Cooling Rate
        if fits.get('method_tau') == 'softplus':
            ln_tau_median = softplus_piecewise(ln_Tint, *fits['popt_tau'])
        else:
            ln_tau_median = fits['A'] * ln_Tint + fits['B']
            
        tau_median = np.exp(ln_tau_median)
        
        # --- THE FIX: INTEGRATE THE MEDIAN AGE HERE ---
        dS_median = np.diff(S_median)
        tau_mid_median = (tau_median[:-1] + tau_median[1:]) / 2.0
        dt_median = -tau_mid_median * dS_median
        
        age_yr = np.zeros(len(T_int_arr))
        age_yr[1:] = np.cumsum(dt_median) / (3600 * 24 * 365.25)
        # ----------------------------------------------
        
        # =========================================================
        # 4. Evaluate Median Structural Radius (S Space)
        # =========================================================
        analytical_Radius = np.exp(softplus_piecewise(ln_S_median, *fits['popt_R']))
        
        results = {
            'T_int': T_int_arr,
            'S_physical': S_median,
            'tau': tau_median,
            'age_yr': age_yr,
            'Req_Rj': analytical_Radius
        }
        
        # =========================================================
        # 5. Evaluate Median Photometry 
        # =========================================================
        target_tirr = fits['target'].get('T_irr', 0.0) / 100.0
        X_query = np.vstack([ln_Tint, np.full_like(ln_Tint, target_tirr)]).T

        photometry_noise = {}
        
        if 'photometry' in fits:
            for band, params in fits['photometry'].items():
                if params is None:
                    continue
                    
                if params['method'] == 'bspline': 
                    spl = BSpline(params['t_knots'], params['popt'], 3, extrapolate=True)
                    results[band] = spl(ln_Tint)
                elif params['method'] == 'sigmoid':
                    results[band] = sloped_sigmoid(ln_Tint, *params['popt']) 
                else:
                    tree_model = params['model']
                    results[band] = tree_model.predict(X_query)
                
        # =========================================================
        # 6. Monte Carlo Uncertainty Propagation
        # =========================================================
        if n_draws > 0:
            try:
                samples_S = np.random.multivariate_normal([C, D], fits['cov_S'], n_draws)
                samples_R = np.random.multivariate_normal(fits['popt_R'], fits['cov_R'], n_draws)
                
                # --- Adaptive MC Draws for Cooling Rate ---
                if fits.get('method_tau') == 'softplus':
                    samples_tau = np.random.multivariate_normal(fits['popt_tau'], fits['cov_tau'], n_draws)
                    # Clip the Softplus sharpness parameter (beta) to prevent math explosion
                    samples_tau[:, 4] = np.clip(samples_tau[:, 4], 0.5, 100.0)
                else:
                    # FIX: Fetch A and B directly from the dictionary for the linear fallback
                    samples_tau = np.random.multivariate_normal([fits['A'], fits['B']], fits['cov_tau'], n_draws)
                
                # --- Clip Radius bounds ---
                samples_R[:, 2] = np.clip(samples_R[:, 2], -0.5, 1.0)   # k1
                samples_R[:, 3] = np.clip(samples_R[:, 3], 0.0, 5.0)    # k2
                samples_R[:, 4] = np.clip(samples_R[:, 4], 0.5, 100.0)  # beta
                
                # Pre-sample analytical photometry fits
                phot_samples = {}
                all_phot_mc = {}
                for band, params in fits.get('photometry', {}).items():
                    if params is not None and params['method'] == 'bspline': 
                        try:
                            p_samp = np.random.multivariate_normal(params['popt'], params['cov'], n_draws)
                            phot_samples[band] = p_samp
                        except Exception:
                            phot_samples[band] = np.tile(params['popt'], (n_draws, 1))
                        all_phot_mc[band] = np.zeros((n_draws, n_points))

                all_ages = np.zeros((n_draws, n_points))
                all_radii = np.zeros((n_draws, n_points))
                all_S = np.zeros((n_draws, n_points))
                
                for i in range(n_draws):
                    C_i, D_i = samples_S[i]
                    
                    ln_S_i = C_i * ln_Tint + D_i
                    S_i = np.exp(ln_S_i)
                    all_S[i, :] = S_i
                    
                    # --- Adaptive MC Evaluation for Cooling Rate ---
                    if fits.get('method_tau') == 'softplus':
                        ln_tau_i = softplus_piecewise(ln_Tint, *samples_tau[i])
                    else:
                        A_i, B_i = samples_tau[i]
                        ln_tau_i = A_i * ln_Tint + B_i
                        
                    tau_i = np.exp(ln_tau_i)
                    
                    dS_i = np.diff(S_i)
                    tau_mid_i = (tau_i[:-1] + tau_i[1:]) / 2.0
                    dt_i = -tau_mid_i * dS_i
                    
                    all_ages[i, 1:] = np.cumsum(dt_i) / (3600 * 24 * 365.25)
                    all_radii[i, :] = np.exp(softplus_piecewise(ln_S_i, *samples_R[i]))
                    
                    # Evaluate Analytical Photometry
                    for band in phot_samples:
                        if fits['photometry'][band]['method'] == 'bspline':
                            spl = BSpline(fits['photometry'][band]['t_knots'], phot_samples[band][i], 3, extrapolate=True)
                            all_phot_mc[band][i, :] = spl(ln_Tint)
                    
                # Calculate Core percentiles
                results['age_yr_lower'] = np.percentile(all_ages, 16, axis=0)
                results['age_yr_upper'] = np.percentile(all_ages, 84, axis=0)
                results['Req_Rj_lower'] = np.percentile(all_radii, 16, axis=0)
                results['Req_Rj_upper'] = np.percentile(all_radii, 84, axis=0)
                results['S_physical_lower'] = np.percentile(all_S, 16, axis=0)
                results['S_physical_upper'] = np.percentile(all_S, 84, axis=0)
                
                # Apply Photometry Bounds
                for band, params in fits.get('photometry', {}).items():
                    if params is None: continue
                    
                    # Extract the empirical variance to capture true grid scatter
                    empirical_var = params.get('variance', 0.0)
                    std_dev = np.sqrt(empirical_var) if empirical_var > 0 else 0.0
                    noise = np.random.normal(loc=0.0, scale=std_dev, size=(n_draws, n_points))
                    
                    # --- THE FIX: Use actual MC draws for B-Splines and Sigmoids! ---
                    if band in all_phot_mc:
                        band_fluxes = all_phot_mc[band] + noise
                    else:
                        band_fluxes = results[band] + noise
                        
                    results[f"{band}_lower"] = np.percentile(band_fluxes, 16, axis=0)
                    results[f"{band}_upper"] = np.percentile(band_fluxes, 84, axis=0)

            except Exception as e:
                import logging
                logging.warning(f"Monte Carlo error propagation failed (Covariance matrix issue?): {e}")
                pass

        return pd.DataFrame(results)