import pandas as pd
import numpy as np
import logging
from .constants import M_J, R_J, INDEPENDENT_DIMS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_clean_grid_pandas(filepath: str) -> pd.DataFrame:
    """
    Uses Pandas with the PyArrow engine to efficiently load and filter 
    an out-of-core Parquet file without blowing up RAM.
    """
    logging.info(f"Loading filtered parquet file: {filepath}...")
    
    # 1. Select only necessary columns
    needed_columns = INDEPENDENT_DIMS + ['mass', 'Req', 'T_int', 'S_physical', 'dsdt']
    
    # 2. Apply Predicate Pushdown filters 
    mass_threshold_kg = 20 * M_J
    
    filters = [
        ('T_int', '<', 2000),
        ('mass', '<=', mass_threshold_kg)
    ]
    
    # 3. Read directly with PyArrow engine
    df = pd.read_parquet(
        filepath, 
        engine='pyarrow',
        columns=needed_columns,
        filters=filters
    )
    
    # 4. Process the remaining data in memory
    df['mass_Mj'] = df['mass'] / M_J
    df['Req_Rj'] = df['Req'] / R_J
    df['abs_log_dsdt'] = np.log10(np.abs(df['dsdt']))
    
    # 5. Drop rows where critical variables are NaN
    critical_cols = INDEPENDENT_DIMS + ['S_physical', 'abs_log_dsdt']
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    
    logging.info(f"Grid loaded successfully. Final shape: {df.shape}")
    return df