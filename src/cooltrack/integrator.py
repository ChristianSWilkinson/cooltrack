"""
ODE integration module for CoolTrack.

This module provides the numerical integrator required to calculate planetary 
ages. Since machine learning models predict the cooling rate (dS/dt) as a 
function of state, this module uses SciPy's solve_ivp to integrate the inverse 
ordinary differential equation: dt/dS = -1 / |dS/dt|.
"""

import warnings

import numpy as np
from scipy.integrate import solve_ivp

from .constants import INDEPENDENT_DIMS, SECONDS_PER_YR

# Suppress runtime warnings that may occur during edge-case ODE steps
warnings.filterwarnings('ignore', category=RuntimeWarning)


class CoolingIntegrator:
    """
    Numerically integrates planetary cooling tracks using ML surrogates.

    This class evaluates the cooling rate at arbitrary entropy steps using the 
    pre-trained XGBoost models, allowing SciPy's Runge-Kutta solver to piece 
    together a continuous time-evolution track from a hot-start initial 
    condition down to a target physical state.

    Attributes:
        tint_model (xgb.XGBRegressor): Model to predict internal temperature.
        dsdt_model (xgb.XGBRegressor): Model to predict the cooling rate.
    """

    def __init__(self, ml_models):
        """
        Initializes the integrator with the trained machine learning engine.

        Args:
            ml_models (ThermalEvolutionModels): An instance of the ML engine 
                containing the trained 'tint_model' and 'dsdt_model'.
        """
        self.tint_model = ml_models.tint_model
        self.dsdt_model = ml_models.dsdt_model

    def _age_ode_ml(self, s: float, t: float, fixed_params_array: np.ndarray):
        """
        Evaluates the differential equation dt/dS for the SciPy solver.

        Args:
            s (float): The current independent variable (physical entropy).
            t (float): The current dependent variable (age in seconds).
            fixed_params_array (np.ndarray): The fixed planetary parameters 
                in the exact order defined by INDEPENDENT_DIMS.

        Returns:
            float: The derivative dt/dS evaluated at state `s`. Returns -np.inf 
            if the cooling rate approaches zero to halt integration safely.
        """
        # 1. Reconstruct the state vector to predict T_int
        # Feature order: INDEPENDENT_DIMS + ['S_physical']
        tint_input = np.append(fixed_params_array, s).reshape(1, -1)
        current_tint = self.tint_model.predict(tint_input)[0]
        
        # 2. Reconstruct the state vector to predict dS/dt
        # Feature order: INDEPENDENT_DIMS + ['S_physical', 'T_int']
        dsdt_input = np.append(tint_input[0], current_tint).reshape(1, -1)
        log_abs_dsdt = self.dsdt_model.predict(dsdt_input)[0]
        
        # Un-log the predicted absolute cooling rate
        abs_dsdt = 10 ** log_abs_dsdt
        
        # Prevent division by zero if the planet has stopped cooling
        if abs_dsdt < 1e-20:
            return -np.inf 
            
        # The ODE: dt/ds = - 1 / |ds/dt| 
        return -1.0 / abs_dsdt

    def calculate_age(self, row, s0_initial: float, s_final: float) -> float:
        """
        Calculates the exact total age in years for a given planetary state.

        Args:
            row (pd.Series or dict): The planet's structural parameters.
            s0_initial (float): The hot-start initial physical entropy.
            s_final (float): The current target physical entropy.

        Returns:
            float: The integrated age of the planet in years. Returns np.nan 
            if the initial entropy is less than or equal to the final entropy, 
            or if the solver fails to converge.
        """
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
    
    def calculate_track(
        self, row, s0_initial: float, s_final: float, num_points: int = 100
    ):
        """
        Generates a full continuous cooling track from formation to current state.

        Forces the Runge-Kutta solver to evaluate and store the planet's age 
        at a requested number of evenly spaced entropy steps, which is ideal 
        for plotting cooling curves.

        Args:
            row (pd.Series or dict): The planet's structural parameters.
            s0_initial (float): The hot-start initial physical entropy.
            s_final (float): The target physical entropy at the end of the track.
            num_points (int, optional): The number of evaluation steps to 
                return. Defaults to 100.

        Returns:
            tuple: A tuple containing two arrays: (ages_yr, entropies). 
            Returns (None, None) if integration fails or inputs are invalid.
        """
        if s0_initial <= s_final:
            return None, None

        fixed_params = row[INDEPENDENT_DIMS].values.astype(float)
        
        # Force solver evaluation at specific points to generate a smooth curve
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