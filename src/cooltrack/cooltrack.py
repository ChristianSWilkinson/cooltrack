import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, root_scalar
from scipy.integrate import cumulative_trapezoid
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress annoying scipy optimize warnings for clean terminal output
warnings.filterwarnings("ignore", message="Covariance of the parameters could not be estimated")

def softplus_piecewise(x, x0, y0, k1, k2):
    """Locked-beta softplus for Thermodynamics, Cooling, Radius, and Photometry."""
    beta = 10.0  
    x = np.asarray(x)
    z = beta * (x - x0)
    soft_term = np.logaddexp(0, z) - np.log(2) 
    return y0 + k1 * (x - x0) + ((k2 - k1) / beta) * soft_term

def weighted_r2(y_true, y_pred, weights):
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    ss_tot = np.sum(weights * (y_true - np.average(y_true, weights=weights))**2)
    return 1 - (ss_res / ss_tot)


class SemiAnalyticalCoolTrack:
    """
    Core engine for generating semi-analytical planetary cooling tracks.
    Bypasses traditional ODE solvers by extracting analytical Softplus 
    surrogate models and numerically integrating them over time.
    """
    def __init__(self, grid_df: pd.DataFrame, initial_conditions_model, independent_dims: list, bandwidth: float = 0.5):
        self.grid_df = self._prepare_grid(grid_df, independent_dims)
        self.init_conds = initial_conditions_model
        self.independent_dims = independent_dims
        self.bandwidth = bandwidth
        self.scaler = StandardScaler()
        
        # Pre-fit the scaler on the grid's independent dimensions
        self.scaler.fit(self.grid_df[self.independent_dims])

    def _prepare_grid(self, df, independent_dims):
        """Cleans and calculates necessary log-space columns upon initialization."""
        work_df = df.copy()
        if 'dsdt' not in work_df.columns and 'dsdt_J_K_kg_s' in work_df.columns:
            work_df['dsdt'] = work_df['dsdt_J_K_kg_s']
            
        work_df = work_df[(work_df['T_int'] > 0) & (work_df['dsdt'] < 0) & (work_df['Req_Rj'] > 0)].copy()
        work_df['ln_Tint'] = np.log(work_df['T_int'])
        work_df['ln_S'] = np.log(work_df['S_physical'])
        work_df['ln_tau'] = -np.log(np.abs(work_df['dsdt']))
        work_df['ln_Req'] = np.log(work_df['Req_Rj'])
        
        # Keep log photometry columns if they exist
        phot_cols = [c for c in work_df.columns if c.startswith('log_') and 'Flambda' in c]
        return work_df.dropna(subset=independent_dims + ['ln_Tint', 'ln_S', 'ln_tau', 'ln_Req'] + phot_cols).reset_index(drop=True)

    def _calculate_weights(self, target_planet: dict):
        """Calculates Gaussian proximity weights for a target planet."""
        target_df = pd.DataFrame([target_planet])[self.independent_dims]
        scaled_grid = self.scaler.transform(self.grid_df[self.independent_dims])
        scaled_target = self.scaler.transform(target_df)
        
        distances = np.linalg.norm(scaled_grid - scaled_target, axis=1)
        weights = np.exp(- (distances**2) / (2 * self.bandwidth**2))
        
        track = self.grid_df.copy()
        track['weight'] = weights
        track = track[track['weight'] > 0.1].copy()
        
        if len(track) < 5:
            track = self.grid_df.copy()
            track['weight'] = weights
            track = track.sort_values('weight', ascending=False).head(50).copy()
            
        return track

    def fit_surrogate(self, target_planet: dict, photometry_bands: list = None):
        """Extracts the analytical Softplus constants for Thermodynamics, Cooling, Structure, and Photometry."""
        track = self._calculate_weights(target_planet)
        w_sqrt = np.sqrt(track['weight'])
        sigma_val = 1.0 / (w_sqrt + 1e-9)
        
        fits = {'target': target_planet, 'track_data': track, 'photometry': {}}

        # =========================================================
        # 1. Thermodynamics S(T_int)
        # =========================================================
        x_S = track['ln_Tint'].values
        y_S = track['ln_S'].values
        
        (fits['C'], fits['D']), fits['cov_S_line'] = np.polyfit(x_S, y_S, deg=1, w=w_sqrt, cov=True)
        
        try:
            popt_S, cov_S = curve_fit(
                softplus_piecewise, x_S, y_S, p0=[np.median(x_S), np.median(y_S), fits['C'], fits['C']], 
                bounds=([np.min(x_S), np.min(y_S)-0.5, 0.0, 0.0], [np.max(x_S), np.max(y_S)+0.5, 5.0, 5.0]),
                sigma=sigma_val.values, method='trf', maxfev=5000
            )
        except Exception:
            popt_S, cov_S = [0, fits['D'], fits['C'], fits['C']], np.eye(4) * 1e-4
        fits['popt_S'], fits['cov_S'] = popt_S, cov_S

        # =========================================================
        # 2. Cooling Rate tau(T_int)
        # =========================================================
        x_tau = track['ln_Tint'].values
        y_tau = track['ln_tau'].values
        
        (fits['A'], fits['B']), fits['cov_tau_line'] = np.polyfit(x_tau, y_tau, deg=1, w=w_sqrt, cov=True)
        
        try:
            popt_tau, cov_tau = curve_fit(
                softplus_piecewise, x_tau, y_tau, p0=[np.median(x_tau), np.median(y_tau), fits['A'], fits['A']], 
                bounds=([np.min(x_tau), np.min(y_tau)-2.0, -15.0, -15.0], [np.max(x_tau), np.max(y_tau)+2.0, 5.0, 5.0]),
                sigma=sigma_val.values, method='trf', maxfev=5000
            )
        except Exception:
            popt_tau, cov_tau = [0, fits['B'], fits['A'], fits['A']], np.eye(4) * 1e-4
        fits['popt_tau'], fits['cov_tau'] = popt_tau, cov_tau

        # =========================================================
        # 3. Structural EoS Radius(S)
        # =========================================================
        x_R = track['ln_S'].values
        y_R = track['ln_Req'].values
        
        try:
            popt_R, cov_R = curve_fit(
                softplus_piecewise, x_R, y_R, p0=[np.percentile(x_R, 30), np.percentile(y_R, 30), 0.01, 0.5], 
                bounds=([np.min(x_R), np.min(y_R)-0.5, -0.1, 0.0], [np.max(x_R), np.max(y_R)+0.5, 0.5, 5.0]),
                sigma=sigma_val.values, method='trf', maxfev=5000
            )
        except Exception:
            (G, H), _ = np.polyfit(x_R, y_R, deg=1, w=w_sqrt, cov=True)
            popt_R, cov_R = [0, H, G, G], np.eye(4) * 1e-4
        fits['popt_R'], fits['cov_R'] = popt_R, cov_R

        # =========================================================
        # 4. Optional Photometry Fits Flux(S)
        # =========================================================
        if photometry_bands:
            for band in photometry_bands:
                y_F = track[band].values
                try:
                    popt_F, cov_F = curve_fit(
                        softplus_piecewise, x_R, y_F, p0=[np.percentile(x_R, 40), np.percentile(y_F, 40), 0.0, 5.0], 
                        bounds=([np.min(x_R), np.min(y_F)-2.0, -0.5, 0.0], [np.max(x_R), np.max(y_F)+2.0, 2.0, 20.0]),
                        sigma=sigma_val.values, method='trf', maxfev=5000
                    )
                except Exception:
                    (E, F_val), _ = np.polyfit(x_R, y_F, deg=1, w=w_sqrt, cov=True)
                    popt_F, cov_F = [0, F_val, E, E], np.eye(4) * 1e-4
                
                fits['photometry'][band] = {'popt': popt_F, 'cov': cov_F}

        return fits

    def evolve(self, fits: dict, start_type: int = 10, n_points: int = 500, n_draws: int = 1000):
        """Numerically integrates time evolution over Softplus structures."""
        SECONDS_PER_YR = 31557600.0
        track = fits['track_data']
        mass = fits['target']['mass_Mj']
        
        # 1. Fetch Boundary Conditions (Entropy)
        S0 = self.init_conds.get_starting_physical_entropy(mass_mjup=mass, bin_index=start_type)
        ln_S0 = np.log(S0)
        
        # Use root_scalar to invert S(T_int) and find starting Temperature
        def s_diff(ln_T):
            return softplus_piecewise(ln_T, *fits['popt_S']) - ln_S0
        
        try:
            res = root_scalar(s_diff, bracket=[np.log(10.0), np.log(10000.0)], method='brentq')
            T0 = np.exp(res.root)
        except Exception:
            T0 = np.exp((ln_S0 - fits['D']) / fits['C']) # Linear fallback
            
        T_min = track['T_int'].min()
        
        # 2. Create the Grid (T_int is the independent variable)
        smooth_Tint = np.geomspace(T0, T_min, n_points)
        ln_Tint = np.log(smooth_Tint)
        
        # 3. Base Arrays
        ln_S_median = softplus_piecewise(ln_Tint, *fits['popt_S'])
        smooth_S = np.exp(ln_S_median)
        
        ln_tau_median = softplus_piecewise(ln_Tint, *fits['popt_tau'])
        tau_median = np.exp(ln_tau_median)
        
        # Numerical Integration for Median Age
        age_median_sec = cumulative_trapezoid(y=-tau_median, x=smooth_S, initial=0)
        age_median = age_median_sec / SECONDS_PER_YR
        
        analytical_Radius = np.exp(softplus_piecewise(ln_S_median, *fits['popt_R']))
        
        # 4. Monte Carlo Setup
        all_ages = np.zeros((n_draws, n_points))
        all_radii = np.zeros((n_draws, n_points))
        
        # Safe Multi-Variate Draws
        try: samples_S = np.random.multivariate_normal(fits['popt_S'], fits['cov_S'], n_draws)
        except Exception: samples_S = [fits['popt_S']] * n_draws
        
        try: samples_tau = np.random.multivariate_normal(fits['popt_tau'], fits['cov_tau'], n_draws)
        except Exception: samples_tau = [fits['popt_tau']] * n_draws
            
        try: samples_R = np.random.multivariate_normal(fits['popt_R'], fits['cov_R'], n_draws)
        except Exception: samples_R = [fits['popt_R']] * n_draws

        # 5. Main MC Integration Loop
        for i in range(n_draws):
            # Compute trajectory physics for this draw
            ln_S_i = softplus_piecewise(ln_Tint, *samples_S[i])
            smooth_S_i = np.exp(ln_S_i)
            
            ln_tau_i = softplus_piecewise(ln_Tint, *samples_tau[i])
            tau_i = np.exp(ln_tau_i)
            
            # Integrate numerically over physical entropy
            age_sec_i = cumulative_trapezoid(y=-tau_i, x=smooth_S_i, initial=0)
            all_ages[i, :] = age_sec_i / SECONDS_PER_YR
            all_radii[i, :] = np.exp(softplus_piecewise(ln_S_i, *samples_R[i]))

        # 6. Compile Results
        results = {
            'T_int': smooth_Tint,
            'S_physical': smooth_S,
            'age_yr': age_median,
            'age_yr_lower': np.percentile(all_ages, 16, axis=0),
            'age_yr_upper': np.percentile(all_ages, 84, axis=0),
            'Radius_Rj': analytical_Radius,
            'Radius_lower': np.percentile(all_radii, 16, axis=0),
            'Radius_upper': np.percentile(all_radii, 84, axis=0)
        }
        
        # 7. Compile Photometry
        for band, params in fits['photometry'].items():
            results[band] = softplus_piecewise(ln_S_median, *params['popt'])
            
            try:
                samples_F = np.random.multivariate_normal(params['popt'], params['cov'], n_draws)
                # Note: Photometry Softplus evaluates against ln(S), not ln(T)
                band_fluxes = np.array([softplus_piecewise(ln_S_median, *s) for s in samples_F])
                results[f"{band}_lower"] = np.percentile(band_fluxes, 16, axis=0)
                results[f"{band}_upper"] = np.percentile(band_fluxes, 84, axis=0)
            except Exception:
                pass 
                
        return pd.DataFrame(results)