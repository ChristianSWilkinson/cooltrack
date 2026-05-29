"""
CoolTrack: Semi-analytical models for planetary thermal evolution.
Provides tools to predict and extract thermodynamic pathways, cooling rates,
structural radii, and photometry for substellar objects.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline
from scipy.optimize import curve_fit, minimize_scalar
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore", 
    message="Covariance of the parameters could not be estimated"
)


def regularize_cov(cov, max_var=0.02):
    """
    Mathematical dampener: Scales down massive covariance values (like the 
    degenerate beta parameter) to prevent Monte Carlo explosions, while 
    preserving the true correlation structure of the grid.

    Args:
        cov (array-like): Original covariance matrix.
        max_var (float, optional): Maximum allowed variance on the diagonal. 
            Defaults to 0.02.

    Returns:
        np.ndarray: Regularized covariance matrix.
    """
    try:
        cov = np.asarray(cov, dtype=float)
        diags = np.diag(cov)
        if np.all(diags <= max_var):
            return cov
        
        scaling = np.sqrt(np.minimum(diags, max_var) / np.maximum(diags, 1e-12))
        return cov * np.outer(scaling, scaling)
    except Exception:
        return np.eye(len(cov)) * 1e-4


class LocalLinearRegressor:
    """
    A local linear regression model utilizing a cKDTree for efficient
    nearest-neighbor searches and weighted ridge regression.
    """

    def __init__(self, n_neighbors=15, ridge_penalty=1e-3, softening=1e-2):
        """
        Initialize the regressor.

        Args:
            n_neighbors (int): Number of neighbors to use for local fitting.
            ridge_penalty (float): Regularization penalty for the local fit.
            softening (float): Distance softening parameter to avoid division by zero.
        """
        self.k = n_neighbors
        self.ridge_penalty = ridge_penalty
        self.softening = softening
        self.tree = None
        self.X = None
        self.y = None
        self.weights = None

    def fit(self, X, y, weights=None):
        """
        Fit the cKDTree with the training data.

        Args:
            X (array-like): Training features.
            y (array-like): Training targets.
            weights (array-like, optional): Sample weights.

        Returns:
            self: The fitted regressor instance.
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        if weights is not None:
            self.weights = np.asarray(weights)
        else:
            self.weights = np.ones(len(y))
            
        self.k = min(self.k, len(self.X))
        self.tree = cKDTree(self.X)
        return self

    def predict(self, X_query):
        """
        Predict target values for the given query points.

        Args:
            X_query (array-like): Query features.

        Returns:
            np.ndarray: Predicted target values.
        """
        X_query = np.asarray(X_query)
        y_pred = np.zeros(len(X_query))
        distances, indices = self.tree.query(X_query, k=self.k)

        if self.k < 3:
            for i, idx in enumerate(indices):
                y_pred[i] = np.average(self.y[idx], weights=self.weights[idx])
            return y_pred

        I = np.eye(self.X.shape[1])
        for i, x_q in enumerate(X_query):
            idx = indices[i]
            d_local = distances[i]
            w_dist = 1.0 / (d_local + self.softening)
            w_total = self.weights[idx] * w_dist

            X_mean = np.average(self.X[idx], axis=0, weights=w_total)
            y_mean = np.average(self.y[idx], weights=w_total)
            X_centered = self.X[idx] - X_mean
            y_centered = self.y[idx] - y_mean

            W = np.diag(w_total)
            XTW = X_centered.T @ W
            XTWX_ridge = (XTW @ X_centered) + (self.ridge_penalty * I)
            Beta = np.linalg.pinv(XTWX_ridge) @ (XTW @ y_centered)
            y_pred[i] = y_mean + np.dot(Beta, (x_q - X_mean))

        return y_pred


def softplus_piecewise(x, x0, y0, k1, k2, beta):
    """
    Piecewise linear function smoothed by a softplus transition.

    Args:
        x (array-like): Independent variable.
        x0 (float): Transition point (knee).
        y0 (float): Function value at the transition point.
        k1 (float): Slope before the transition.
        k2 (float): Slope after the transition.
        beta (float): Sharpness of the transition.

    Returns:
        np.ndarray: Evaluated function values.
    """
    x = np.asarray(x)
    z = beta * (x - x0)
    soft_term = np.logaddexp(0, z) - np.log(2)
    return y0 + k1 * (x - x0) + ((k2 - k1) / beta) * soft_term


def softplus_derivative(x, x0, y0, k1, k2, beta):
    """
    Analytical derivative of the Softplus piecewise function.

    Args:
        x (array-like): Independent variable.
        x0, y0, k1, k2, beta (float): Parameters of the softplus function.

    Returns:
        np.ndarray: Evaluated derivative values.
    """
    x = np.asarray(x)
    z = np.clip(beta * (x - x0), -100, 100)
    # Numerically stable sigmoid
    sigmoid = np.where(
        z > 0, 
        1.0 / (1.0 + np.exp(-z)), 
        np.exp(z) / (1.0 + np.exp(z))
    )
    return k1 + (k2 - k1) * sigmoid


def dual_softplus_piecewise(x, y0, x1, k1, k2, beta1, x2, k3, beta2):
    """
    Three-segment piecewise linear function smoothed by two softplus transitions.

    Args:
        x (array-like): Independent variable.
        y0 (float): Base offset value.
        x1, x2 (float): First and second transition points.
        k1, k2, k3 (float): Slopes of the three segments.
        beta1, beta2 (float): Sharpness of the first and second transitions.

    Returns:
        np.ndarray: Evaluated function values.
    """
    x = np.asarray(x)
    z1 = beta1 * (x - x1)
    z2 = beta2 * (x - x2)
    soft_term_1 = np.logaddexp(0, z1) - np.log(2)
    soft_term_2 = np.logaddexp(0, z2) - np.log(2)
    
    base = y0 + k1 * (x - x1)
    knee_1 = ((k2 - k1) / beta1) * soft_term_1
    knee_2 = ((k3 - k2) / beta2) * soft_term_2
    return base + knee_1 + knee_2


def weighted_r2(y_true, y_pred, weights):
    """
    Calculate the weighted R-squared metric.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.
        weights (array-like): Sample weights.

    Returns:
        float: Weighted R-squared value.
    """
    ss_res = np.sum(weights * (y_true - y_pred)**2)
    mean_true = np.average(y_true, weights=weights)
    ss_tot = np.sum(weights * (y_true - mean_true)**2)
    
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)


def generalized_sigmoid(x, x0, y_floor, amplitude, k, nu):
    """Generalized Richards curve (sigmoid)."""
    x = np.asarray(x)
    z = np.clip(-k * (x - x0), -100, 100)
    return y_floor + (amplitude / ((1.0 + np.exp(z)) ** nu))


def sloped_sigmoid(x, x0, y_floor, amplitude, k, nu, m_hot):
    """Generalized sigmoid with a sloped asymptote."""
    x = np.asarray(x)
    z = np.clip(-k * (x - x0), -100, 100)
    S = 1.0 / ((1.0 + np.exp(z)) ** nu)
    return y_floor + S * (amplitude + m_hot * (x - x0))


def quadratic_sigmoid(x, y0, m1, m2, amp, k, x0):
    """Sigmoid function transitioning into a quadratic curve."""
    exp_arg = np.clip(-k * (x - x0), -100, 100)
    return y0 + m1 * x + m2 * (x**2) + (amp / (1.0 + np.exp(exp_arg)))


class SemiAnalyticalCoolTrack:
    """
    A class to extract and propagate surrogate models for planetary 
    thermal evolution tracks.
    """

    def __init__(
        self, 
        grid_df: pd.DataFrame, 
        initial_conditions_model, 
        independent_dims: list, 
        bandwidth: float = 0.5
    ):
        """
        Initialize the CoolTrack model.

        Args:
            grid_df (pd.DataFrame): The raw thermal evolution grid.
            initial_conditions_model: Object containing physical starting conditions.
            independent_dims (list): List of independent dimension column names.
            bandwidth (float, optional): Gaussian kernel bandwidth. Defaults to 0.5.
        """
        self.independent_dims = independent_dims
        self.bandwidth = bandwidth
        self.init_conds = initial_conditions_model
        self.grid_df = self._prepare_grid(grid_df)

        self.scaler = StandardScaler()
        self.scaler.fit(self.grid_df[self.independent_dims])

    def _prepare_grid(self, df):
        """Clean and log-transform the necessary thermodynamic variables."""
        work_df = df.copy()
        
        if 'dsdt' not in work_df.columns and 'dsdt_J_K_kg_s' in work_df.columns:
            work_df['dsdt'] = work_df['dsdt_J_K_kg_s']

        valid_mask = (
            (work_df['T_int'] > 0) & 
            (work_df['dsdt'] < 0) & 
            (work_df['Req_Rj'] > 0)
        )
        work_df = work_df[valid_mask].copy()

        work_df['log10_mass_Mj'] = np.log10(work_df['mass_Mj'])
        work_df['ln_Tint'] = np.log(work_df['T_int'])
        work_df['ln_S'] = np.log(work_df['S_physical'])
        work_df['ln_tau'] = -np.log(np.abs(work_df['dsdt']))
        work_df['ln_Req'] = np.log(work_df['Req_Rj'])
        work_df = work_df.replace([np.inf, -np.inf], np.nan)

        phot_cols = [c for c in work_df.columns if c.startswith('log_') and 'Flambda' in c]
        
        req_cols = self.independent_dims + ['ln_Tint', 'ln_S', 'ln_tau', 'ln_Req'] + phot_cols
        return work_df.dropna(subset=req_cols).reset_index(drop=True)

    def _calculate_weights(self, target_planet: dict):
        """Calculate Gaussian proximity weights based on target planet parameters."""
        target_copy = target_planet.copy()
        if 'mass_Mj' in target_copy:
            target_copy['log10_mass_Mj'] = np.log10(target_copy['mass_Mj'])

        target_df = pd.DataFrame([target_copy])[self.independent_dims]
        scaled_grid = self.scaler.transform(self.grid_df[self.independent_dims])
        scaled_target = self.scaler.transform(target_df)

        sq_distances = np.linalg.norm(scaled_grid - scaled_target, axis=1)**2
        weights = np.exp(-sq_distances / (2 * self.bandwidth**2))

        track = self.grid_df.copy()
        track['weight'] = weights
        track = track[track['weight'] > 0.05].copy()

        if len(track) < 5:
            track = self.grid_df.copy()
            track['weight'] = weights
            track = track.sort_values('weight', ascending=False).head(50).copy()

        return track

    def fit_surrogate(
        self, 
        target_planet: dict, 
        photometry_bands: list = None, 
        photometry_method: str = 'sigmoid'
    ):
        """
        Fit surrogate models for thermodynamics, cooling rate, radius, and photometry.

        Args:
            target_planet (dict): Target planet parameters.
            photometry_bands (list, optional): List of photometric bands to fit.
            photometry_method (str, optional): Fitting method for photometry.

        Returns:
            dict: Dictionary containing optimized parameters and metadata.
        """
        track = self._calculate_weights(target_planet)
        w_sqrt = np.sqrt(track['weight'])
        fits = {'target': target_planet, 'track_data': track, 'photometry': {}}
        # Store the fit's actual T_int support range as a small explicit field,
        # so evolve() can clip the start to a safe range even if the database
        # build later strips/reduces track_data for size. Without this, the
        # clip range can collapse to a single point if track_data is reduced.
        try:
            _t = track['ln_Tint'].values
            fits['T_int_range_K'] = (float(np.exp(_t.min())), float(np.exp(_t.max())))
        except Exception:
            pass

        # 1. Thermodynamics S(T_int)
        x_Tint = track['ln_Tint'].values
        y_S = track['ln_S'].values
        sigma_weights = 1.0 / (w_sqrt + 1e-9)
        
        try:
            C_lin, D_lin = np.polyfit(x_Tint, y_S, deg=1, w=w_sqrt)
            
            if np.exp(np.min(x_Tint)) < 300.0:
                p0_S = [
                    np.median(y_S), np.log(1500.0), C_lin, C_lin * 0.8, 
                    10.0, np.log(150.0), C_lin * 1.5, 10.0
                ]
                bounds_S = (
                    [np.min(y_S) - 2, np.log(800), -5.0, -5.0, 0.5, np.log(50), -5.0, 0.5],
                    [np.max(y_S) + 2, np.log(1600), 5.0, 5.0, 100.0, np.log(400), 5.0, 100.0]
                )
                popt_S, cov_S = curve_fit(
                    dual_softplus_piecewise, x_Tint, y_S, p0=p0_S, 
                    bounds=bounds_S, sigma=sigma_weights, method='trf'
                )
                fits.update({'method_S': 'dual_softplus', 'popt_S': popt_S, 'cov_S': cov_S})
                fits['r2_S'] = weighted_r2(y_S, dual_softplus_piecewise(x_Tint, *popt_S), track['weight'])
            else:
                p0_S = [np.log(1300.0), np.median(y_S), C_lin, C_lin * 0.8, 10.0]
                bounds_S = (
                    [np.log(800), np.min(y_S) - 2, -5.0, -5.0, 0.5], 
                    [np.log(1800), np.max(y_S) + 2, 5.0, 5.0, 100.0]
                )
                popt_S, cov_S = curve_fit(
                    softplus_piecewise, x_Tint, y_S, p0=p0_S, 
                    bounds=bounds_S, sigma=sigma_weights, method='trf'
                )
                fits.update({'method_S': 'softplus', 'popt_S': popt_S, 'cov_S': cov_S})
                fits['r2_S'] = weighted_r2(y_S, softplus_piecewise(x_Tint, *popt_S), track['weight'])
                
        except Exception:
            (C, D), cov = np.polyfit(x_Tint, y_S, deg=1, w=w_sqrt, cov=True)
            fits.update({'method_S': 'linear', 'C': C, 'D': D, 'cov_S': cov})
            fits['r2_S'] = weighted_r2(y_S, C * x_Tint + D, track['weight'])

        # 2. Cooling Rate tau(T_int)
        y_tau = track['ln_tau'].values
        try:
            A_lin, B_lin = np.polyfit(x_Tint, y_tau, deg=1, w=w_sqrt)
            p0_tau = [np.log(1200.0), np.median(y_tau), A_lin, A_lin * 0.8, 10.0]
            bounds_tau = (
                [np.log(700), np.min(y_tau), -1000.0, -100.0, 10], 
                [np.log(1500), np.max(y_tau), 0.0, 0.0, 500.0]
            )
            popt_tau, cov_tau = curve_fit(
                softplus_piecewise, x_Tint, y_tau, p0=p0_tau, 
                bounds=bounds_tau, sigma=sigma_weights, method='trf'
            )
            fits.update({'method_tau': 'softplus', 'popt_tau': popt_tau, 'cov_tau': cov_tau})
            fits['r2_tau'] = weighted_r2(y_tau, softplus_piecewise(x_Tint, *popt_tau), track['weight'])
            
        except Exception:
            (A, B), cov = np.polyfit(x_Tint, y_tau, deg=1, w=w_sqrt, cov=True)
            fits.update({'method_tau': 'linear', 'A': A, 'B': B, 'cov_tau': cov})
            fits['r2_tau'] = weighted_r2(y_tau, A * x_Tint + B, track['weight'])

        # 3. Structural Radius(T_int)
        # NOTE: radius is fit DIRECTLY against ln_Tint (not ln_S). The grid's
        # R(T_eff) relation is the well-validated, mass-clean one; fitting R as a
        # function of T_int keeps the radius off the entropy surrogate entirely,
        # so evolve()'s R(T_int) reproduces that validated relation and R(age)
        # stays consistent with T_eff(age). 'radius_basis' records this so evolve
        # knows which axis to evaluate popt_R on (older .dat files default to 'S').
        x_R_axis = x_Tint                      # ln_Tint, defined in section 1
        y_R = track['ln_Req'].values
        p0_R = [np.percentile(x_R_axis, 50), np.percentile(y_R, 50), 0.05, 0.3, 4.0]
        bounds_R = (
            [np.percentile(x_R_axis, 1), np.min(y_R) - 1.0, -2.0, -2.0, 0.5],
            [np.percentile(x_R_axis, 99), np.max(y_R) + 1.0, 5.0, 5.0, 100.0]
        )

        try:
            popt_R, cov_R = curve_fit(
                softplus_piecewise, x_R_axis, y_R, p0=p0_R,
                bounds=bounds_R, sigma=sigma_weights, method='trf'
            )
            r2_R = weighted_r2(y_R, softplus_piecewise(x_R_axis, *popt_R), track['weight'])
        except Exception:
            try:
                (G, H), _ = np.polyfit(x_R_axis, y_R, deg=1, w=w_sqrt, cov=True)
                popt_R = [np.median(x_R_axis), H + G * np.median(x_R_axis), G, G, 10.0]
                cov_R = np.eye(5) * 1e-4
                r2_R = weighted_r2(y_R, G * x_R_axis + H, track['weight'])
            except Exception:
                popt_R = [np.median(x_R_axis), np.median(y_R), 0.0, 0.0, 10.0]
                cov_R = np.eye(5) * 1e-4
                r2_R = 0.0

        # Weighted residual scatter of the R fit: this is the honest, stable
        # measure of how well the surrogate pins radius, and is what evolve()
        # turns into the uncertainty band (instead of the near-degenerate
        # softplus parameter covariance, which produces runaway bands).
        try:
            resid_R = y_R - softplus_piecewise(x_R_axis, *popt_R)
            R_resid_var = float(np.average(resid_R**2, weights=track['weight'].values))
        except Exception:
            R_resid_var = 0.0

        fits.update({
            'popt_R': popt_R, 'cov_R': cov_R, 'r2_R': r2_R,
            'radius_basis': 'Tint', 'R_resid_var': R_resid_var,
        })

        # 4. Photometry
        if photometry_bands:
            for band in photometry_bands:
                target_T_irr = target_planet.get('T_irr', 0.0)
                is_adaptive_method = photometry_method in ['sigmoid', 'quadratic_sigmoid', 'bspline']
                
                if is_adaptive_method:
                    clean_mask = track[band].notna() & (np.abs(track['T_irr'] - target_T_irr) <= 50.0)
                else:
                    clean_mask = track[band].notna()
                    
                if clean_mask.sum() < 5:
                    fits['photometry'][band] = None
                    continue

                if is_adaptive_method:
                    x_fit = x_Tint[clean_mask]
                    y_fit = track.loc[clean_mask, band].values
                    w_subset = track.loc[clean_mask, 'weight'].values
                    sigma_F = 1.0 / (np.sqrt(w_subset) + 1e-9)
                    
                    t_knots = np.concatenate((
                        [np.min(x_fit)] * 4, 
                        np.linspace(np.min(x_fit), np.max(x_fit), 4)[1:-1], 
                        [np.max(x_fit)] * 4
                    ))
                    
                    def bspline_wrapper(x, c0, c1, c2, c3, c4, c5):
                        return BSpline(t_knots, [c0, c1, c2, c3, c4, c5], 3, extrapolate=True)(x)

                    try:
                        p0_F = np.linspace(np.min(y_fit), np.max(y_fit), 6)
                        popt_F, cov_F = curve_fit(
                            bspline_wrapper, x_fit, y_fit, 
                            p0=p0_F, sigma=sigma_F, method='trf'
                        )
                        y_pred_F = BSpline(t_knots, popt_F, 3, extrapolate=True)(x_fit)
                        r2_F = weighted_r2(y_fit, y_pred_F, 1 / sigma_F**2)
                    except Exception:
                        popt_F = np.linspace(np.min(y_fit), np.max(y_fit), 6)
                        cov_F = np.eye(6) * 1e-4
                        r2_F = 0.0

                    y_pred_variance = BSpline(t_knots, popt_F, 3, extrapolate=True)(x_fit)
                    fits['photometry'][band] = {
                        'method': 'bspline', 
                        'popt': popt_F, 
                        'cov': cov_F, 
                        't_knots': t_knots, 
                        'r2': r2_F, 
                        'variance': np.var(y_fit - y_pred_variance)
                    }

        return fits

    def evolve(
        self, 
        fits: dict, 
        start_type: int = 10, 
        n_points: int = 500, 
        n_draws: int = 1000, 
        T_start_override: float = None
    ):
        """
        Integrate the thermal evolution over time based on the fitted surrogate models.

        Args:
            fits (dict): Fitted parameter dictionary from `fit_surrogate`.
            start_type (int, optional): Initial condition bin index. Defaults to 10.
            n_points (int, optional): Number of integration points. Defaults to 500.
            n_draws (int, optional): Number of Monte Carlo draws for error propagation.
            T_start_override (float, optional): Manual override for initial temperature.

        Returns:
            pd.DataFrame: Evolved track with median values and confidence intervals.
        """
        if 'method_S' not in fits:
            raise ValueError("Invalid surrogate fits.")
            
        target = fits['target']
        mass = target.get('mass_Mj', 10 ** target.get('log10_mass_Mj', 1.0))

        # Safe T_int range for the start clip. Built as a robust hierarchy:
        #   1) an explicit (T_min, T_max) stored alongside the fit (preferred -
        #      added below by fit_surrogate so rebuilt .dat files carry it);
        #   2) the spread of track_data['ln_Tint'] if it has real width;
        #   3) a generous substellar safety range (50-4000 K) if the database
        #      stored a reduced single-row track_data to save space.
        # This decouples the clip from how much of the training neighbourhood
        # the .dat build chose to keep.
        rng = fits.get('T_int_range_K')
        if rng is not None and float(rng[1]) > float(rng[0]):
            ln_T_lo = float(np.log(float(rng[0])))
            ln_T_hi = float(np.log(float(rng[1])))
        else:
            _ln_lo = float(fits['track_data']['ln_Tint'].min())
            _ln_hi = float(fits['track_data']['ln_Tint'].max())
            if _ln_hi - _ln_lo > 0.1:
                ln_T_lo, ln_T_hi = _ln_lo, _ln_hi
            else:
                ln_T_lo, ln_T_hi = float(np.log(50.0)), float(np.log(4000.0))

        # 1. Boundary Conditions
        if T_start_override is not None:
            ln_T0 = np.log(T_start_override)
            if fits.get('method_S') == 'dual_softplus':
                ln_S0 = dual_softplus_piecewise(ln_T0, *fits['popt_S'])
            elif fits.get('method_S') == 'softplus':
                ln_S0 = softplus_piecewise(ln_T0, *fits['popt_S'])
            else:
                ln_S0 = fits['C'] * ln_T0 + fits['D']
        else:
            S0_phys = self.init_conds.get_starting_physical_entropy(
                mass_mjup=mass, bin_index=start_type
            )
            ln_S0 = np.log(S0_phys)
            WIDE_LO, WIDE_HI = np.log(10.0), np.log(100000.0)
            try:
                if fits.get('method_S') == 'dual_softplus':
                    res = minimize_scalar(
                        lambda x: (dual_softplus_piecewise(x, *fits['popt_S']) - ln_S0)**2, 
                        bounds=(WIDE_LO, WIDE_HI), 
                        method='bounded'
                    )
                    ln_T0 = res.x
                elif fits.get('method_S') == 'softplus':
                    res = minimize_scalar(
                        lambda x: (softplus_piecewise(x, *fits['popt_S']) - ln_S0)**2, 
                        bounds=(WIDE_LO, WIDE_HI), 
                        method='bounded'
                    )
                    ln_T0 = res.x
                else:
                    ln_T0 = (ln_S0 - fits['D']) / fits['C']
            except Exception as e:
                raise RuntimeError(f"Numerical inversion failed: {e}")

            # Fallback: if the inversion couldn't find a match anywhere in the
            # wide range (lands on a boundary), the IC entropy is on a different
            # absolute scale than the grid's S_physical (CGS vs SI, different
            # mean molecular weight, etc.) and the absolute match is meaningless.
            # In that case interpret start_type as a quantile of the grid's
            # T_int range: bin (n_bins-1) -> top of grid, bin 0 -> bottom. This
            # decouples the start from the entropy-normalization mismatch.
            boundary_tol = 0.05
            n_bins = 20
            if (ln_T0 - WIDE_LO) < boundary_tol or (WIDE_HI - ln_T0) < boundary_tol:
                frac = max(0.0, min(1.0, float(start_type) / float(n_bins - 1)))
                ln_T0 = ln_T_lo + frac * (ln_T_hi - ln_T_lo)

            # Never start outside the grid's T_int support (no surrogate extrapolation).
            ln_T0 = float(np.clip(ln_T0, ln_T_lo, ln_T_hi))

        T0 = np.exp(ln_T0)
        T_end = np.exp(fits['track_data']['ln_Tint'].min())
        T_int_arr = np.logspace(np.log10(T0), np.log10(T_end), n_points)
        ln_Tint = np.log(T_int_arr)

        # 3. Median Eval
        if fits.get('method_S') == 'dual_softplus':
            ln_S_median = dual_softplus_piecewise(ln_Tint, *fits['popt_S'])
        elif fits.get('method_S') == 'softplus':
            ln_S_median = softplus_piecewise(ln_Tint, *fits['popt_S'])
        else:
            ln_S_median = fits['C'] * ln_Tint + fits['D']

        S_median = np.exp(ln_S_median)
        
        if fits.get('method_tau') == 'softplus':
            ln_tau_median = softplus_piecewise(ln_Tint, *fits['popt_tau'])
        else:
            ln_tau_median = fits['A'] * ln_Tint + fits['B']

        tau_median = np.exp(ln_tau_median)
        age_yr = np.zeros(len(T_int_arr))
        dt_median = -((tau_median[:-1] + tau_median[1:]) / 2.0) * np.diff(S_median)
        age_yr[1:] = np.cumsum(dt_median) / (3600 * 24 * 365.25)

        # Radius is evaluated on whichever axis it was fit against. New fits use
        # 'Tint' (decoupled from the entropy surrogate); legacy .dat files fall
        # back to the old ln_S basis so they still run unchanged.
        ln_R_axis = ln_Tint if fits.get('radius_basis') == 'Tint' else ln_S_median

        results = {
            'T_int': T_int_arr, 
            'S_physical': S_median, 
            'tau': tau_median, 
            'age_yr': age_yr, 
            'Req_Rj': np.exp(softplus_piecewise(ln_R_axis, *fits['popt_R']))
        }

        # 5. Median Photometry
        if 'photometry' in fits:
            for band, params in fits['photometry'].items():
                if params is None:
                    continue
                
                if params['method'] == 'bspline':
                    results[band] = BSpline(
                        params['t_knots'], params['popt'], 3, extrapolate=True
                    )(ln_Tint)
                elif params['method'] == 'sigmoid':
                    results[band] = sloped_sigmoid(ln_Tint, *params['popt'])
                else:
                    query_stack = np.vstack([
                        ln_Tint, 
                        np.full_like(ln_Tint, target.get('T_irr', 0.0) / 100.0)
                    ]).T
                    results[band] = params['model'].predict(query_stack)

        # 6. Strict Monte Carlo Error Propagation
        if n_draws > 0:
            try:
                cov_S_reg = regularize_cov(fits['cov_S'], 1)
                cov_tau_reg = regularize_cov(fits['cov_tau'], 1)
                # Radius cov is only used by the legacy (ln_S-basis) fallback.
                # Cap it tightly: 1.0 allowed an e^1 ~ x2.7 swing in log-radius.
                cov_R_reg = regularize_cov(fits['cov_R'], 0.02)
                

                if fits.get('method_S') == 'dual_softplus':
                    samples_S = np.random.multivariate_normal(fits['popt_S'], cov_S_reg, n_draws)
                    samples_S[:, 0] = fits['popt_S'][0]
                    #samples_S[:, 2] = np.clip(samples_S[:, 2], -5.0, 5.0)
                    #samples_S[:, 3] = np.clip(samples_S[:, 3], -5.0, 5.0)
                    #samples_S[:, 6] = np.clip(samples_S[:, 6], -5.0, 5.0)
                elif fits.get('method_S') == 'softplus':
                    samples_S = np.random.multivariate_normal(fits['popt_S'], cov_S_reg, n_draws)
                    samples_S[:, 1] = fits['popt_S'][1]
                    samples_S[:, 2] = np.clip(samples_S[:, 2], 1e-3, 5.0)
                    samples_S[:, 3] = np.clip(samples_S[:, 3], 1e-3, 5.0)
                else:
                    normal_C = np.random.normal(fits['C'], np.sqrt(cov_S_reg[0, 0]), n_draws)
                    samples_S = np.column_stack((normal_C, np.full(n_draws, fits['D'])))

                if fits.get('method_tau') == 'softplus':
                    samples_tau = np.random.multivariate_normal(fits['popt_tau'], cov_tau_reg, n_draws)
                    #samples_tau[:, 2] = np.clip(samples_tau[:, 2], -10.0, 0.0)
                    #samples_tau[:, 3] = np.clip(samples_tau[:, 3], -10.0, 0.0)
                else:
                    samples_tau = np.random.multivariate_normal([fits['A'], fits['B']], cov_tau_reg, n_draws)

                samples_R = np.random.multivariate_normal(fits['popt_R'], cov_R_reg, n_draws)
                # Stabilize the legacy parametric band the same way the S/tau
                # draws are stabilized: freeze the unidentifiable knee location
                # (x0) and sharpness (beta), and clip the slopes. The residual-
                # scatter band below supersedes this when R_resid_var exists.
                samples_R[:, 0] = fits['popt_R'][0]
                samples_R[:, 4] = fits['popt_R'][4]
                samples_R[:, 2] = np.clip(samples_R[:, 2], -2.0, 5.0)
                samples_R[:, 3] = np.clip(samples_R[:, 3], -2.0, 5.0)

                phot_samples, all_phot_mc = {}, {}
                for band, params in fits.get('photometry', {}).items():
                    if params is not None and params['method'] == 'bspline':
                        try:
                            phot_cov_reg = regularize_cov(params['cov'], 0.5)
                            phot_samples[band] = np.random.multivariate_normal(
                                params['popt'], phot_cov_reg, n_draws
                            )
                        except Exception:
                            phot_samples[band] = np.tile(params['popt'], (n_draws, 1))
                        all_phot_mc[band] = np.zeros((n_draws, n_points))

                all_ages = np.zeros((n_draws, n_points))
                all_S = np.zeros((n_draws, n_points))
                all_R = np.zeros((n_draws, n_points))
                
                # ---> THE CLEVER CAP <---
                # Query the absolute maximum physical entropy for this mass (Hot Start / Bin 19)
                max_physical_S = self.init_conds.get_starting_physical_entropy(mass_mjup=mass, bin_index=19)

                for i in range(n_draws):
                    if fits.get('method_S') == 'dual_softplus':
                        ln_S_i = dual_softplus_piecewise(ln_Tint, *samples_S[i])
                    elif fits.get('method_S') == 'softplus':
                        ln_S_i = softplus_piecewise(ln_Tint, *samples_S[i])
                    else:
                        ln_S_i = samples_S[i][0] * ln_Tint + samples_S[i][1]

                    # Exponentiate the draw, but strictly clip it to the laws of thermodynamics
                    #S_i = np.clip(np.exp(ln_S_i), a_min=None, a_max=max_physical_S)
                    S_i = np.exp(ln_S_i)
                    all_S[i, :] = S_i

                    ln_R_i = softplus_piecewise(ln_R_axis, *samples_R[i])
                    all_R[i, :] = np.exp(ln_R_i)

                    if fits.get('method_tau') == 'softplus':
                        ln_tau_i = softplus_piecewise(ln_Tint, *samples_tau[i])
                    else:
                        ln_tau_i = samples_tau[i][0] * ln_Tint + samples_tau[i][1]

                    tau_i = np.exp(ln_tau_i)
                    dt_i = -((tau_i[:-1] + tau_i[1:]) / 2.0) * np.diff(S_i)
                    all_ages[i, 1:] = np.cumsum(dt_i) / (3600 * 24 * 365.25)

                    for band in phot_samples:
                        t_knots = fits['photometry'][band]['t_knots']
                        all_phot_mc[band][i, :] = BSpline(
                            t_knots, phot_samples[band][i], 3, extrapolate=True
                        )(ln_Tint)

                results['age_yr_lower'] = np.percentile(all_ages, 16, axis=0)
                results['age_yr_upper'] = np.percentile(all_ages, 84, axis=0)
                results['S_physical_lower'] = np.percentile(all_S, 16, axis=0)
                results['S_physical_upper'] = np.percentile(all_S, 84, axis=0)

                # Radius band: prefer the residual scatter of the R(T_int) fit.
                # This is the honest "how well does the surrogate know R" width
                # (a few percent), and is stable, unlike percentiles of curves
                # drawn from the near-degenerate softplus parameter covariance.
                R_resid_var = fits.get('R_resid_var', None)
                if R_resid_var is not None:
                    sigma_R = float(np.sqrt(np.clip(R_resid_var, 0.0, 0.01)))
                    ln_R_med = np.log(results['Req_Rj'])
                    R_band = np.exp(
                        ln_R_med[None, :]
                        + np.random.normal(0.0, sigma_R, size=(n_draws, n_points))
                    )
                    results['Req_Rj_lower'] = np.percentile(R_band, 16, axis=0)
                    results['Req_Rj_upper'] = np.percentile(R_band, 84, axis=0)
                else:
                    results['Req_Rj_lower'] = np.percentile(all_R, 16, axis=0)
                    results['Req_Rj_upper'] = np.percentile(all_R, 84, axis=0)

                for band, params in fits.get('photometry', {}).items():
                    if params is None:
                        continue
                    
                    var = params.get('variance', 0.0)
                    std_dev = np.sqrt(var) if var > 0 else 0.0
                    noise = np.random.normal(loc=0.0, scale=std_dev, size=(n_draws, n_points))
                    
                    if band in all_phot_mc:
                        band_fluxes = all_phot_mc[band] + noise
                    else:
                        band_fluxes = results[band] + noise
                        
                    results[f"{band}_lower"] = np.percentile(band_fluxes, 16, axis=0)
                    results[f"{band}_upper"] = np.percentile(band_fluxes, 84, axis=0)

            except Exception as e:
                print(f"MC Error propagation bypassed due to: {e}")

        return pd.DataFrame(results)