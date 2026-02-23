import pandas as pd
import numpy as np
import logging
from .constants import M_J, R_J, INDEPENDENT_DIMS, PHOTOMETRY_BANDS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_grid_pandas(filepath: str) -> pd.DataFrame:
    logging.info(f"Loading filtered parquet file: {filepath}...")
    
    # 1. Add PHOTOMETRY_BANDS to the raw columns
    raw_columns = [
        'mass', 'Req', 'T_int', 'T_irr', 'Met', 'core', 'f_sed', 'kzz', 
        'S_physical', 'dsdt'
    ] + PHOTOMETRY_BANDS
    
    mass_threshold_kg = 20 * M_J
    filters = [
        ('T_int', '<', 2000),
        ('mass', '<=', mass_threshold_kg)
    ]
    
    df = pd.read_parquet(
        filepath, engine='pyarrow', columns=raw_columns, filters=filters
    )
    
    df['mass_Mj'] = df['mass'] / M_J
    df['Req_Rj'] = df['Req'] / R_J
    df['abs_log_dsdt'] = np.log10(np.abs(df['dsdt']))
    
    # Create log10 columns for all photometry bands
    for band in PHOTOMETRY_BANDS:
        # Filter out 0 or negative fluxes to prevent log10 errors
        df = df[df[band] > 0]
        df[f'log_{band}'] = np.log10(df[band])
    
    critical_cols = INDEPENDENT_DIMS + ['S_physical', 'abs_log_dsdt'] + [f'log_{b}' for b in PHOTOMETRY_BANDS]
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    
    logging.info(f"Grid loaded successfully. Final shape: {df.shape}")
    return df