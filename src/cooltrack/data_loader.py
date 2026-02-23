"""
Data loading and preprocessing module for CoolTrack.

This module handles the ingestion of raw planetary evolution grids (in Parquet
format), applies initial physical filters, calculates derived quantities like
logarithmic cooling rates and photometric fluxes, and removes invalid entries.
"""

import logging

import numpy as np
import pandas as pd

from .constants import INDEPENDENT_DIMS, M_J, PHOTOMETRY_BANDS, R_J

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_and_clean_grid_pandas(filepath: str) -> pd.DataFrame:
    """
    Load, filter, and preprocess the raw planetary evolution grid.

    This function reads a Parquet file and applies preliminary physical filters 
    (e.g., mass <= 20 M_J, T_int < 2000 K) directly during the read step to 
    save memory. It scales values to standard astrophysical units, computes 
    the base-10 logarithm of the absolute cooling rate and photometric fluxes, 
    and drops any rows with missing or non-physical data.

    Args:
        filepath (str): The path to the raw Parquet grid file.

    Returns:
        pd.DataFrame: A cleaned DataFrame ready for machine learning ingestion,
        containing scaled variables, log-transformed cooling rates, and 
        log-transformed photometric fluxes.
    """
    logging.info(f"Loading filtered parquet file: {filepath}...")
    
    # 1. Define the subset of columns to load from the massive grid
    raw_columns = [
        'mass', 'Req', 'T_int', 'T_irr', 'Met', 'core', 'f_sed', 'kzz', 
        'S_physical', 'dsdt'
    ] + PHOTOMETRY_BANDS
    
    # 2. Apply physical filters during read to save RAM
    mass_threshold_kg = 20.0 * M_J
    filters = [
        ('T_int', '<', 2000),
        ('mass', '<=', mass_threshold_kg)
    ]
    
    df = pd.read_parquet(
        filepath, 
        engine='pyarrow', 
        columns=raw_columns, 
        filters=filters
    )
    
    # 3. Scale units to standard Jupiter metrics
    df['mass_Mj'] = df['mass'] / M_J
    df['Req_Rj'] = df['Req'] / R_J
    df['abs_log_dsdt'] = np.log10(np.abs(df['dsdt']))
    
    # 4. Create log10 columns for all photometry bands safely
    for band in PHOTOMETRY_BANDS:
        # Filter out 0 or negative fluxes to prevent log10 domain errors
        df = df[df[band] > 0]
        df[f'log_{band}'] = np.log10(df[band])
    
    # 5. Define critical columns required for the ML engine
    log_bands = [f'log_{b}' for b in PHOTOMETRY_BANDS]
    critical_cols = INDEPENDENT_DIMS + ['S_physical', 'abs_log_dsdt'] + log_bands
    
    # 6. Drop rows with NaN values in critical features and reset the index
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    
    logging.info(f"Grid loaded successfully. Final shape: {df.shape}")
    
    return df