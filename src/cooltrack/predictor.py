"""
Fast prediction module for CoolTrack.

This module provides a plug-and-play interface to evaluate planetary properties
in microseconds using pre-computed analytical coefficient grids, completely 
bypassing the need to load the raw HDF5/Parquet data.
"""

import pickle
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .cooltrack import SemiAnalyticalCoolTrack
from .initial_conditions import InitialConditions
from .constants import INDEPENDENT_DIMS

class CoolTrackPredictor:
    """
    A lightweight, ultra-fast evaluator for CoolTrack surrogate models.
    Requires a pre-computed coefficient .dat file.
    """
    
    def __init__(self, dat_filepath: str, age_data_path: str):
        """
        Loads the coefficient database and initializes a hollow integration engine.
        
        Args:
            dat_filepath (str): Path to the pre-computed .dat coefficient file.
            age_data_path (str): Path to the directory containing formation entropy CSVs.
        """
        # 1. Load the pre-computed coefficients
        with open(dat_filepath, 'rb') as f:
            self.database = pickle.load(f)
            
        # 2. Build the KDTree for instant parameter matching
        self.search_matrix = np.array([fit['search_vector'] for fit in self.database])
        self.mean = self.search_matrix.mean(axis=0)
        self.std = self.search_matrix.std(axis=0) + 1e-6
        self.tree = cKDTree((self.search_matrix - self.mean) / self.std)
        
        # 3. Silently initialize the hollow engine for integration math
        self.engine = self._initialize_hollow_engine(age_data_path)

    def _initialize_hollow_engine(self, age_data_path):
        """Creates a memory-free dummy engine to access the evolve() methods."""
        init_conds = InitialConditions(age_data_path=age_data_path)
        
        # Create a tiny 1-row DataFrame to satisfy the engine's StandardScaler
        dummy_grid = pd.DataFrame({col: [0.0] for col in INDEPENDENT_DIMS})
        dummy_grid['ln_Tint'] = 0.0
        dummy_grid['ln_S'] = 0.0
        dummy_grid['ln_tau'] = 0.0
        dummy_grid['ln_Req'] = 0.0
        dummy_grid['dsdt'] = -1.0         # Required for internal filters
        dummy_grid['Req_Rj'] = 1.0        # Required for internal filters
        dummy_grid['T_int'] = 1000.0      # Required for internal filters
        dummy_grid['mass_Mj'] = 1.0       # Required for internal filters
        dummy_grid['S_physical'] = 1.0    # <--- ADD THIS LINE (Required for np.log(S_physical))
        
        return SemiAnalyticalCoolTrack(dummy_grid, init_conds, INDEPENDENT_DIMS)

    def predict(self, target_planet: dict, target_age_yr: float, start_type: int = 19, n_draws: int = 0) -> dict:
        """
        Retrieves the exact physical and photometric properties for a planet at a specific age.
        
        Args:
            target_planet (dict): Dictionary of planetary parameters.
            target_age_yr (float): The requested age of the system in years.
            start_type (int): The formation entropy bin (0=Coldest, 19=Hottest).
            
        Returns:
            dict: Interpolated properties at the exact requested age.
        """
        # Extract the query vector strictly in the order of INDEPENDENT_DIMS
        query_vec = np.array([
            target_planet['mass_Mj'], target_planet['T_irr'], target_planet['Met'], 
            target_planet['core'], target_planet['f_sed_refractory'], 
            target_planet['f_sed_volatile'], target_planet['kzz']
        ])
        
        # Find the closest pre-computed analytical fit
        scaled_query = (query_vec - self.mean) / self.std
        distance, index = self.tree.query(scaled_query)
        cached_fits = self.database[index]
        
        # Run the fast mathematical integration (~1ms)
        evol_df = self.engine.evolve(cached_fits, start_type=start_type, n_draws=n_draws)
        
        # Check boundaries
        if target_age_yr < evol_df['age_yr'].min() or target_age_yr > evol_df['age_yr'].max():
            pass # Boundary warning
            
        # 1. Standard interpolation for all median/direct columns
        output = {'age_yr': target_age_yr}
        for col in evol_df.columns:
            if col != 'age_yr':
                output[col] = float(np.interp(target_age_yr, evol_df['age_yr'].values, evol_df[col].values))
                
        # 2. Extract Temperature Uncertainty (if Monte Carlo draws were used)
        if 'age_yr_upper' in evol_df.columns and 'age_yr_lower' in evol_df.columns:
            # Slower cooling (upper age bound) means a HOTTER planet at the target age
            output['T_int_upper'] = float(np.interp(target_age_yr, evol_df['age_yr_upper'].values, evol_df['T_int'].values))
            
            # Faster cooling (lower age bound) means a COLDER planet at the target age
            output['T_int_lower'] = float(np.interp(target_age_yr, evol_df['age_yr_lower'].values, evol_df['T_int'].values))

        output['snap_distance'] = float(distance)
        
        return output