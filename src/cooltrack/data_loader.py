"""
Data loading and preprocessing module for CoolTrack.

This module handles the ingestion of raw planetary evolution grids (in Parquet
format), applies initial physical filters, calculates derived quantities like
logarithmic cooling rates and photometric fluxes, and removes invalid entries.
"""
import h5py
import pandas as pd
import numpy as np
import json
import logging
from tqdm import tqdm
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from .constants import INDEPENDENT_DIMS, M_J, PHOTOMETRY_BANDS, R_J

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_fsed(val, cloud_key):
    """Safely extracts a specific cloud's f_sed from strings or dicts."""
    if pd.isna(val): return np.nan
    if isinstance(val, str) and val != "None":
        try:
            d = ast.literal_eval(val)
            if isinstance(d, dict): return float(d.get(cloud_key, np.nan))
        except (ValueError, SyntaxError):
            pass
    elif isinstance(val, dict):
        return float(val.get(cloud_key, np.nan))
    
    try:
        return float(val)
    except Exception:
        return np.nan


def load_and_clean_grid_pandas(filepath: str) -> pd.DataFrame:
    """
    Load, filter, and preprocess the raw planetary evolution grid.
    """
    logging.info(f"Loading filtered parquet file: {filepath}...")
    
    raw_columns = [
        'mass', 'Req', 'T_int', 'T_irr', 'Met', 'core', 'f_sed', 'kzz', 
        'S_physical', 'dsdt'
    ] + PHOTOMETRY_BANDS
    
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
    
    # --- THE FIX: Unpack the f_sed string into two numeric columns ---
    df['f_sed_refractory'] = df['f_sed'].apply(lambda x: parse_fsed(x, 'Fe'))
    df['f_sed_volatile'] = df['f_sed'].apply(lambda x: parse_fsed(x, 'H2O'))
    df = df.drop(columns=['f_sed'])
    
    # 3. Scale units to standard Jupiter metrics & ADD LOG MASS
    df['mass_Mj'] = df['mass'] / M_J
    df['log10_mass_Mj'] = np.log10(df['mass_Mj'])
    
    df['Req_Rj'] = df['Req'] / R_J
    df['abs_log_dsdt'] = np.log10(np.abs(df['dsdt']))
    
    for band in PHOTOMETRY_BANDS:
        df = df[df[band] > 0]
        df[f'log_{band}'] = np.log10(df[band])
    
    log_bands = [f'log_{b}' for b in PHOTOMETRY_BANDS]
    critical_cols = INDEPENDENT_DIMS + ['S_physical', 'abs_log_dsdt'] + log_bands
    
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    
    logging.info(f"Grid loaded successfully. Final shape: {df.shape}")
    
    return df

def _worker_extract(filepath, model_keys):
    """
    Worker function: Opens its OWN connection to the HDF5 file to avoid 
    multiprocessing crashes, and processes a chunk of models.
    """
    extracted_rows = []
    # Force HDF5 to ignore locks in workers
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    
    with h5py.File(filepath, "r") as h5f:
        for model_id in model_keys:
            model_grp = h5f[model_id]
            try:
                params = model_grp['parameters'].attrs
                mass_mjup = params.get('true_mass_Mjup', params.get('mass', np.nan))
                
                fsed_raw = params.get('f_sed')
                if isinstance(fsed_raw, str) and fsed_raw != "None":
                    try:
                        fsed_dict = json.loads(fsed_raw)
                        f_ref = float(fsed_dict.get('Fe', np.nan))
                        f_vol = float(fsed_dict.get('H2O', np.nan))
                    except Exception:
                        f_ref, f_vol = np.nan, np.nan
                else:
                    f_ref, f_vol = np.nan, np.nan
                
                row_data = {
                    'mass': mass_mjup * M_J,
                    'T_int': params.get('T_int', np.nan),
                    'T_int_input_dial' : params.get('T_int_input_dial', np.nan),
                    'T_irr': params.get('T_irr', np.nan),
                    'T_eff': params.get('T_eff', np.nan),
                    'Met': params.get('Met', np.nan),
                    'core': params.get('core_mass_earth', np.nan),
                    'f_sed_refractory': f_ref,
                    'f_sed_volatile': f_vol,
                    'kzz': params.get('kzz', np.nan),
                }
                
                int_attrs = model_grp['interior_raw'].attrs
                cool_attrs = model_grp[f"cooling_metrics"].attrs
                row_data['Req'] = int_attrs.get('R_total', np.nan)
                
                dt_ds = cool_attrs.get("dt_ds", np.nan)
                row_data['dsdt'] = 1.0 / dt_ds if dt_ds != 0 else np.nan
                
                try:
                    row_data['S_physical'] = np.max(model_grp['interior_raw']['S'][:])
                except Exception:
                    row_data['S_physical'] = np.nan
                
                if 'photometry' in model_grp and 'bands' in model_grp['photometry']:
                    h5_bands = model_grp['photometry']['bands']
                    for ct_band in PHOTOMETRY_BANDS:
                        expected_h5_key = ct_band.replace('/', '_')
                        if expected_h5_key in h5_bands:
                            row_data[ct_band] = h5_bands[expected_h5_key].attrs.get('flux_W_m2_um', np.nan)
                        else:
                            row_data[ct_band] = np.nan
                else:
                    for ct_band in PHOTOMETRY_BANDS:
                        row_data[ct_band] = np.nan
                        
                extracted_rows.append(row_data)
            except KeyError:
                continue
    return extracted_rows


def load_and_clean_exoweave_hdf5(filepath: str) -> pd.DataFrame:
    """
    Load, filter, and preprocess an Exoweave HDF5 grid in parallel.
    """
    logging.info(f"Loading Exoweave HDF5 file: {filepath}...")
    
    # 1. Grab all keys quickly
    with h5py.File(filepath, "r") as h5f:
        all_keys = list(h5f.keys())
        
    extracted_rows = []
    
    # 2. Split keys into chunks for parallel processing
    n_cores = max(1, os.cpu_count() - 1)
    chunk_size = max(1, len(all_keys) // (n_cores * 4))
    key_chunks = [all_keys[i:i + chunk_size] for i in range(0, len(all_keys), chunk_size)]
    
    logging.info(f"Spinning up {n_cores} workers to read HDF5 across network...")
    
    # 3. Read in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = {executor.submit(_worker_extract, filepath, chunk): chunk for chunk in key_chunks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading HDF5 Chunks"):
            extracted_rows.extend(future.result())

    df = pd.DataFrame(extracted_rows)
    print(f"Extracted {len(df)} models from Exoweave HDF5 file.")
    
    if df.empty:
        logging.error("No valid models were extracted from the HDF5 file!")
        return df
    
    # --- Standard Cleaning Logic ---
    mass_threshold_kg = 20.0 * M_J
    df = df[(df['T_int'] < 2000) & (df['mass'] <= mass_threshold_kg)].copy()

    df['mass_Mj'] = df['mass'] / M_J
    df['log10_mass_Mj'] = np.log10(df['mass_Mj'])
    
    df['Req_Rj'] = df['Req'] / R_J
    df['abs_log_dsdt'] = np.log10(np.abs(df['dsdt']))

    for band in PHOTOMETRY_BANDS:
        if df[band].isnull().all():
            continue
        df = df[df[band] > 0]
        df[f'log_{band}'] = np.log10(df[band])
    
    logging.info(f"✅ Exoweave Grid loaded and mapped successfully. Final shape: {df.shape}")
    return df

def load_grid(filepath: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Loads the planetary grid. If an HDF5 file is provided, it automatically
    caches the extracted DataFrame as a Parquet file for instant loading on future runs.
    """
    if filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        # Automatically generate a cache filename next to the HDF5 file
        cache_path = filepath.rsplit('.', 1)[0] + '_cache.parquet'
        
        # 1. Check for Cache
        if use_cache and os.path.exists(cache_path):
            logging.info(f"🚀 Loading instantly from Parquet cache: {cache_path}")
            return pd.read_parquet(cache_path)
            
        # 2. Extract if no cache is found
        logging.info("⏳ Parquet cache not found or disabled. Running parallel HDF5 extraction...")
        df = load_and_clean_exoweave_hdf5(filepath)
        
        # 3. Save to Cache for next time
        if use_cache and not df.empty:
            logging.info(f"💾 Saving extracted grid to cache: {cache_path}")
            df.to_parquet(cache_path)
            
        return df
        
    elif filepath.endswith('.parquet'):
        # Handles raw, uncleaned parquet grids (legacy mode)
        return load_and_clean_grid_pandas(filepath)
        
    else:
        raise ValueError("Unsupported file format. Please provide a .parquet or .h5 file.")