"""
Data loading and preprocessing module for CoolTrack.

This module handles the ingestion of raw planetary evolution grids (in Parquet
format), applies initial physical filters, calculates derived quantities like
logarithmic cooling rates and photometric fluxes, and removes invalid entries.
"""

import logging

import numpy as np
import pandas as pd
import h5py

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


def load_and_clean_exoweave_hdf5(filepath: str) -> pd.DataFrame:
    """
    Load, filter, and preprocess an Exoweave HDF5 grid for CoolTrack.
    
    Acts as an adapter to map Exoweave output formats to the standard 
    CoolTrack Pandas DataFrame structure.
    """
    logging.info(f"Loading Exoweave HDF5 file: {filepath}...")
    
    extracted_rows = []
    
    with h5py.File(filepath, "r") as h5f:
        for model_id in h5f.keys():

            model_grp = h5f[model_id]
            
            try:
                # 1. Extract Exoweave Parameters
                params = model_grp['parameters'].attrs
                
                # Exoweave saves mass in M_Jup, but CoolTrack expects raw kg initially
                # We use the true integrated mass if available, else the target dial
                mass_mjup = params.get('true_mass_Mjup', params.get('mass', np.nan))
                
                row_data = {
                    'mass': mass_mjup * M_J,
                    'T_int': params.get('T_int', np.nan),
                    'T_irr': params.get('T_irr', np.nan),
                    'Met': params.get('Met', np.nan),
                    'core': params.get('core_mass_earth', np.nan),
                    'f_sed': params.get('f_sed', np.nan),
                    'kzz': params.get('kzz', np.nan),
                }
                
                # 2. Extract State Variables from Interior
                int_attrs = model_grp['interior_raw'].attrs
                int_arrays = model_grp['interior_raw']
                
                row_data['Req'] = int_attrs.get('R_total', np.nan)
                
                # Invert dt/ds to get dS/dt
                dt_ds = int_attrs.get('dt_ds_total', np.nan)
                row_data['dsdt'] = 1.0 / dt_ds if dt_ds != 0 else np.nan
                
                # Grab the specific physical entropy at the top of the convective envelope
                try:
                    row_data['S_physical'] = int_attrs.get('S', np.nan)[-1]
                except Exception:
                    row_data['S_physical'] = np.nan
                
                # 3. Extract and Map Photometry
                if 'photometry' in model_grp and 'bands' in model_grp['photometry']:
                    h5_bands = model_grp['photometry']['bands']
                    
                    # Map the SVO HDF5 keys to the exact strings CoolTrack expects
                    for ct_band in PHOTOMETRY_BANDS:
                        matched_flux = np.nan
                        for h5_key in h5_bands.keys():
                            # Extract just the filter ID (e.g. 'F770W' from 'JWST_MIRI.F770W')
                            short_filter = h5_key.split('.')[-1]
                            
                            # If the filter ID is inside the CoolTrack target string
                            if short_filter in ct_band:
                                matched_flux = h5_bands[h5_key].attrs.get('flux_W_m2_um', np.nan)
                                break
                                
                        row_data[ct_band] = matched_flux
                else:
                    # If photometry is missing, fill with NaNs
                    for ct_band in PHOTOMETRY_BANDS:
                        row_data[ct_band] = np.nan
                        
                extracted_rows.append(row_data)
                
            except KeyError as e:
                logging.debug(f"Skipping {model_id} due to missing root data: {e}")
                continue

    # Convert the extracted dictionary list into a DataFrame
    df = pd.DataFrame(extracted_rows)
    print(f"Extracted {len(df)} models from Exoweave HDF5 file.")
    
    if df.empty:
        logging.error("No valid models were extracted from the HDF5 file!")
        return df
    
    # --- Apply Standard CoolTrack Filtering & Scaling ---
    mass_threshold_kg = 20.0 * M_J
    df = df[(df['T_int'] < 2000) & (df['mass'] <= mass_threshold_kg)].copy()
    
    df['mass_Mj'] = df['mass'] / M_J
    df['Req_Rj'] = df['Req'] / R_J
    df['abs_log_dsdt'] = np.log10(np.abs(df['dsdt']))
    
    for band in PHOTOMETRY_BANDS:
        df = df[df[band] > 0]
        df[f'log_{band}'] = np.log10(df[band])
    
    log_bands = [f'log_{b}' for b in PHOTOMETRY_BANDS]
    critical_cols = INDEPENDENT_DIMS + ['S_physical', 'abs_log_dsdt'] + log_bands

    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    
    logging.info(f"✅ Exoweave Grid loaded and mapped successfully. Final shape: {df.shape}")
    return df


def load_grid(filepath: str) -> pd.DataFrame:
    """
    Universal grid loader. Automatically detects whether the file is an 
    Exoweave HDF5 or a legacy CoolTrack Parquet file and routes accordingly.
    """
    if filepath.endswith('.h5') or filepath.endswith('.hdf5'):
        return load_and_clean_exoweave_hdf5(filepath)
    elif filepath.endswith('.parquet'):
        return load_and_clean_grid_pandas(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a .parquet or .h5 file.")