import numpy as np
from scipy.integrate import solve_ivp
from .constants import SECONDS_PER_YR, INDEPENDENT_DIMS
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

class CoolingIntegrator:
    def __init__(self, ml_models):
        self.tint_model = ml_models.tint_model
        self.dsdt_model = ml_models.dsdt_model

    def _age_ode_ml(self, s, t, fixed_params_array):
        """
        ODE for age integration. 
        fixed_params_array order matches INDEPENDENT_DIMS.
        """
        # 1. Reconstruct the state vector to predict T_int
        # feature order: INDEPENDENT_DIMS + ['S_physical']
        tint_input = np.append(fixed_params_array, s).reshape(1, -1)
        current_tint = self.tint_model.predict(tint_input)[0]
        
        # 2. Reconstruct the state vector to predict dS/dt
        # feature order: INDEPENDENT_DIMS + ['S_physical', 'T_int']
        dsdt_input = np.append(tint_input[0], current_tint).reshape(1, -1)
        log_abs_dsdt = self.dsdt_model.predict(dsdt_input)[0]
        
        abs_dsdt = 10**log_abs_dsdt
        
        if abs_dsdt < 1e-20:
            return -np.inf 
            
        # dt/ds = - 1 / |ds/dt| 
        return -1.0 / abs_dsdt

    def calculate_age(self, row, s0_initial, s_final):
        """Calculates age in years for a given planetary state."""
        if s0_initial <= s_final:
            return np.nan

        # Extract fixed parameters in the exact order of INDEPENDENT_DIMS
        fixed_params = row[INDEPENDENT_DIMS].values.astype(float)

        solution = solve_ivp(
            fun=self._age_ode_ml,
            t_span=[s0_initial, s_final],
            y0=[0],
            method='RK45',
            args=(fixed_params,)
        )

        if solution.status == 0:
            age_in_seconds = solution.y[0][-1]
            return age_in_seconds / SECONDS_PER_YR
        return np.nan
    
    def calculate_track(self, row, s0_initial, s_final, num_points=100):
        """Returns the full cooling track (ages and entropies) for plotting."""
        if s0_initial <= s_final:
            return None, None

        fixed_params = row[INDEPENDENT_DIMS].values.astype(float)
        
        # We force the solver to evaluate at specific points so we get a smooth curve
        s_eval = np.linspace(s0_initial, s_final, num_points)

        solution = solve_ivp(
            fun=self._age_ode_ml,
            t_span=[s0_initial, s_final],
            y0=[0],
            t_eval=s_eval,
            method='RK45',
            args=(fixed_params,)
        )

        if solution.status == 0:
            ages_yr = solution.y[0] / SECONDS_PER_YR
            entropies = solution.t
            return ages_yr, entropies
        return None, None