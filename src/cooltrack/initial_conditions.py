"""
Initial boundary conditions module for CoolTrack.

This module loads theoretical hot and cold start entropy data for newly formed
gas giants and brown dwarfs. It constructs interpolators to determine the initial 
boundary condition (specific physical entropy) required for the ODE integrator 
based on a planet's mass and its presumed formation mechanism.
"""

import glob
import logging
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .constants import K_B, M_J, MASS_PARTICLE_APPROX_KG

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class InitialConditions:
    """
    Manages the physical boundary conditions for planetary cooling tracks.

    Reads foundational mass-entropy data from CSV files to build interpolation 
    functions. These functions map a planet's mass to its starting physical 
    entropy, bounded by the theoretical 'cold start' (core accretion) and 
    'hot start' (gravitational collapse) scenarios.

    Attributes:
        age_data_path (str): Directory containing the initial condition CSVs.
        s_cold_interp (interp1d): Interpolator for cold-start minimum entropy.
        s_hot_interp (interp1d): Interpolator for hot-start maximum entropy.
    """

    def __init__(self, age_data_path: str):
        """
        Initializes the object and automatically builds the interpolators.

        Args:
            age_data_path (str): The folder path containing the CSV data files.
        """
        self.age_data_path = age_data_path
        self.s_cold_interp, self.s_hot_interp = self._build_interpolators()

    def _build_interpolators(self):
        """
        Reads CSV files and builds scipy interp1d functions for initial entropy.

        Returns:
            tuple: A tuple containing (s_cold_interp, s_hot_interp), which are
            both `scipy.interpolate.interp1d` objects.

        Raises:
            FileNotFoundError: If no CSV files are found in the target directory.
            ValueError: If there is an error parsing the CSVs or if there is 
                insufficient binned data to create the interpolators.
        """
        files = glob.glob(os.path.join(self.age_data_path, '*.csv'))
        if not files:
            raise FileNotFoundError(
                f"No CSV files found in age_data_path: {self.age_data_path}"
            )
        
        try:
            df_list = [pd.read_csv(f) for f in files]
            init_s_df = pd.concat(df_list, ignore_index=True)
        except Exception as e:
            raise ValueError(f"Error reading CSVs: {e}")

        # Clean and prepare the data
        init_s_df['M'] = pd.to_numeric(init_s_df['M'], errors='coerce')
        init_s_df['S'] = pd.to_numeric(init_s_df['S'], errors='coerce')
        init_s_df = init_s_df.dropna(subset=['M', 'S'])

        # Determine bin boundaries
        min_mass, max_mass = init_s_df['M'].min(), init_s_df['M'].max()
        
        if min_mass <= 0:
            min_mass = 0.01 
        if max_mass <= min_mass:
            max_mass = min_mass + 1.0 
        if np.isclose(min_mass, max_mass):
            max_mass = min_mass * 1.1 

        mass_bins = np.logspace(np.log10(min_mass), np.log10(max_mass), 10)
        
        # Bin the data and extract midpoints
        init_s_df['M_bin_intervals'] = pd.cut(
            init_s_df['M'], 
            bins=mass_bins, 
            right=False, 
            include_lowest=True
        )
        
        init_s_df['M_bin'] = init_s_df['M_bin_intervals'].apply(
            lambda x: x.mid if pd.notna(x) else np.nan
        ).astype(float)
        
        # Aggregate to find the cold (min) and hot (max) entropy for each bin
        result = init_s_df.groupby('M_bin')['S'].agg(
            ['min', 'max']
        ).reset_index().dropna()
        
        if len(result) < 2:
            raise ValueError(
                "Not enough binned data for S_cold/S_hot interpolators."
            )
            
        # Create interpolators
        s_cold = interp1d(
            result['M_bin'], result['min'], 
            fill_value='extrapolate', kind='linear'
        )
        s_hot = interp1d(
            result['M_bin'], result['max'], 
            fill_value='extrapolate', kind='linear'
        )
        
        logging.info(
            f"Loaded initial condition interpolators from {len(files)} files."
        )
        return s_cold, s_hot

    def get_starting_physical_entropy(
        self, mass_mjup: float, bin_index: int = 19, n_bins: int = 20
    ) -> float:
        """
        Calculates the exact starting physical entropy for a target mass.

        This method extracts the dimensionless entropy limits for the given mass,
        selects a specific value based on the requested bin index (where 0 
        represents a pure cold start and n_bins-1 represents a pure hot start), 
        and converts it into specific physical entropy.

        Args:
            mass_mjup (float): The mass of the planet in Jupiter masses.
            bin_index (int, optional): The index of the starting condition to 
                use (0 = coldest, default 19 = hottest). Defaults to 19.
            n_bins (int, optional): The total number of bins to divide the 
                entropy range into. Defaults to 20.

        Returns:
            float: The starting specific physical entropy in J/K/kg.
        """
        s_min_k = float(self.s_cold_interp(mass_mjup))
        s_max_k = float(self.s_hot_interp(mass_mjup))
        
        # Ensure correct ordering
        if s_min_k > s_max_k: 
            s_min_k, s_max_k = s_max_k, s_min_k
            
        # Select the target entropy in dimensionless units
        if np.isclose(s_min_k, s_max_k) or n_bins <= 1:
            s0_k_val = s_min_k
        else:
            s0_k_val = float(np.linspace(s_min_k, s_max_k, n_bins)[bin_index])
            
        # Convert non-dimensional entropy (S/k_B per particle) to J/K/kg
        mass_kg_planet = mass_mjup * M_J
        n_particles = mass_kg_planet / MASS_PARTICLE_APPROX_KG 
        total_entropy = s0_k_val * n_particles * K_B
        
        return total_entropy / mass_kg_planet