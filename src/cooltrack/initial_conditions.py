import os
import glob
import logging
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from .constants import M_J, K_B, MASS_PARTICLE_APPROX_KG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InitialConditions:
    def __init__(self, age_data_path: str):
        """
        Loads the hot/cold start CSVs and builds the initial entropy interpolators.
        """
        self.age_data_path = age_data_path
        self.S_cold_interp, self.S_hot_interp = self._build_interpolators()

    def _build_interpolators(self):
        """Reads the CSV files and builds scipy interp1d functions for S_cold and S_hot."""
        files = glob.glob(os.path.join(self.age_data_path, '*.csv'))
        if not files:
            raise FileNotFoundError(f"No CSV files found in age_data_path: {self.age_data_path}")
        
        try:
            df_list = [pd.read_csv(f) for f in files]
            init_s_df = pd.concat(df_list, ignore_index=True)
        except Exception as e:
            raise ValueError(f"Error reading CSVs: {e}")

        init_s_df['M'] = pd.to_numeric(init_s_df['M'], errors='coerce')
        init_s_df['S'] = pd.to_numeric(init_s_df['S'], errors='coerce')
        init_s_df = init_s_df.dropna(subset=['M', 'S'])

        min_mass, max_mass = init_s_df['M'].min(), init_s_df['M'].max()
        if min_mass <= 0: min_mass = 0.01 
        if max_mass <= min_mass: max_mass = min_mass + 1.0 
        
        if np.isclose(min_mass, max_mass):
            max_mass = min_mass * 1.1 

        mass_bins = np.logspace(np.log10(min_mass), np.log10(max_mass), 10)
        init_s_df['M_bin_intervals'] = pd.cut(init_s_df['M'], bins=mass_bins, right=False, include_lowest=True)
        init_s_df['M_bin'] = init_s_df['M_bin_intervals'].apply(lambda x: x.mid if pd.notna(x) else np.nan).astype(float)
        
        result = init_s_df.groupby('M_bin')['S'].agg(['min', 'max']).reset_index().dropna()
        
        if len(result) < 2:
            raise ValueError("Not enough binned data for S_cold/S_hot interpolators.")
            
        S_cold = interp1d(result['M_bin'], result['min'], fill_value='extrapolate', kind='linear')
        S_hot = interp1d(result['M_bin'], result['max'], fill_value='extrapolate', kind='linear')
        
        logging.info(f"Loaded initial condition interpolators from {len(files)} files.")
        return S_cold, S_hot

    def get_starting_physical_entropy(self, mass_mjup: float, bin_index: int = 19, n_bins: int = 20) -> float:
        """
        Calculates the exact starting physical entropy (J/K/kg) for a given mass 
        and hot/cold start bin (0 is cold, n_bins-1 is hot).
        """
        s_min_k = float(self.S_cold_interp(mass_mjup))
        s_max_k = float(self.S_hot_interp(mass_mjup))
        
        if s_min_k > s_max_k: 
            s_min_k, s_max_k = s_max_k, s_min_k
            
        if np.isclose(s_min_k, s_max_k) or n_bins <= 1:
            s0_k_val = s_min_k
        else:
            s0_k_val = float(np.linspace(s_min_k, s_max_k, n_bins)[bin_index])
            
        # Convert to total physical entropy, then specific physical entropy (J/K/kg)
        mass_kg_planet = mass_mjup * M_J
        n_particles = mass_kg_planet / MASS_PARTICLE_APPROX_KG 
        total_entropy = s0_k_val * n_particles * K_B
        
        return total_entropy / mass_kg_planet