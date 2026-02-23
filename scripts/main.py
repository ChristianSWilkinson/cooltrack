"""
Master pipeline execution script for CoolTrack.

This script manages the end-to-end workflow of the CoolTrack package. It is 
responsible for loading the raw planetary grid, executing or bypassing the ML 
surrogate training phase via smart caching, and dispatching the numerical ODE 
integrations across multiple CPU cores to calculate ages for the entire grid.
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from cooltrack.constants import INDEPENDENT_DIMS, Bands
from cooltrack.data_loader import load_and_clean_grid_pandas
from cooltrack.initial_conditions import InitialConditions
from cooltrack.integrator import CoolingIntegrator
from cooltrack.models import ThermalEvolutionModels

# Configure module-level logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- CONFIGURATION ---
GRID_FILE_PATH = "../data/HADES_grid/hades_processed_grid.parquet" 
CLEAN_GRID_PATH = "../data/HADES_grid/hades_clean_grid.parquet" 
MODELS_DIR = "../data/models/" 
AGE_DATA_PATH = "../data/age_data/"
OUTPUT_FILE_PATH = "../data/HADES_grid/hades_grid_with_ages.parquet"

# Number of CPU cores to use for parallel processing (-1 means use ALL cores)
N_CORES = -1 


def compute_age(
    row_tuple: Tuple[int, pd.Series], 
    integrator: CoolingIntegrator, 
    init_cond: InitialConditions
) -> float:
    """
    Worker function to calculate the exact age for a single row in the grid.

    This function is designed to run in an isolated thread/process. It determines
    the starting entropy for a given planet's mass, handles edge cases where the 
    planet is essentially at age zero, and calls the ODE integrator.

    Args:
        row_tuple (Tuple[int, pd.Series]): A tuple containing the row index and 
            the row data (as yielded by pd.DataFrame.iterrows()).
        integrator (CoolingIntegrator): The initialized ODE solver engine.
        init_cond (InitialConditions): The hot/cold start boundary models.

    Returns:
        float: The exact age of the planet in years. Returns np.nan if the 
        integration fails to converge or encounters numerical instability.
    """
    _, row = row_tuple 
    mass = row['mass_Mj']
    s_target = row['S_physical']
    
    try:
        # Get exact hot-start boundary condition
        s_hot_start = init_cond.get_starting_physical_entropy(
            mass, bin_index=19, n_bins=20
        )
        
        # If the grid point is hotter than the boundary, it represents a state 
        # right at formation (Age ~ 0)
        if s_target >= s_hot_start:
            return 0.0
            
        # Integrate exactly down to this row's physical entropy
        ages, _ = integrator.calculate_track(
            row, s_hot_start, s_target, num_points=10
        )
        
        if ages is not None and len(ages) > 0:
            return ages[-1] 
            
        return np.nan

    except Exception:
        # Catch broad exceptions to prevent a single broken row from 
        # crashing the entire multiprocessing pool
        return np.nan


def main():
    """
    Executes the master processing pipeline.

    Workflow:
    1. Check for cached models to bypass training.
    2. Load or compute the clean DataFrame.
    3. Initialize physics boundary conditions and the ODE integrator.
    4. Spin up a multiprocessing pool to evaluate ages across all rows.
    5. Clean up failed integrations and save the final dataset to disk.
    """
    print("==================================================")
    print("🚀 COOLTRACK MASTER PIPELINE INITIALIZED 🚀")
    print("==================================================")
    
    ml_engine = ThermalEvolutionModels()
    
    # --- SMART CACHING LOGIC ---
    # Check if we already have trained models AND the clean data saved
    if os.path.exists(MODELS_DIR) and os.path.exists(CLEAN_GRID_PATH):
        logging.info("Found pre-trained models and clean grid! Skipping training...")
        ml_engine.load_models(MODELS_DIR)
        df_clean = pd.read_parquet(CLEAN_GRID_PATH, engine='pyarrow')
    else:
        logging.info("No saved models found. Loading grid and training...")
        df_raw = load_and_clean_grid_pandas(GRID_FILE_PATH)
        df_clean = ml_engine.train_models(
            df_raw, tune_hyperparameters=False, clean_outliers=True
        )
        
        # Save them so we never have to do this again
        ml_engine.save_models(MODELS_DIR)
        logging.info(f"Saving clean grid to {CLEAN_GRID_PATH}...")
        df_clean.to_parquet(CLEAN_GRID_PATH, engine='pyarrow', index=False)
        
    logging.info(f"Ready for integration with {len(df_clean):,} pristine points.")
    
    # --- PHYSICS INITIALIZATION ---
    logging.info("Initializing Initial Conditions and ODE Integrator...")
    init_cond = InitialConditions(AGE_DATA_PATH)
    integrator = CoolingIntegrator(ml_engine)
    
    # --- PARALLEL INTEGRATION ---
    logging.info(f"Firing up parallel age computation across CPU cores...")
    rows_to_process = list(df_clean.iterrows())
    
    ages_list = Parallel(n_jobs=N_CORES)(
        delayed(compute_age)(row, integrator, init_cond) 
        for row in tqdm(rows_to_process, desc="Calculating Ages", unit="planet")
    )
    
    df_clean['Age_yr'] = ages_list
    
    # --- FINAL POLISH ---
    original_len = len(df_clean)
    df_clean = df_clean.dropna(subset=['Age_yr']).reset_index(drop=True)
    failed = original_len - len(df_clean)
    
    if failed > 0:
        logging.warning(
            f"ODE Solver failed to converge for {failed} points. Dropped."
        )
        
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    logging.info(f"Saving final grid with ages to {OUTPUT_FILE_PATH}...")
    df_clean.to_parquet(OUTPUT_FILE_PATH, engine='pyarrow', index=False)
    
    print("\n==================================================")
    print("✅ PIPELINE COMPLETE! ✅")
    print(f"Total simulated planets saved: {len(df_clean):,}")
    print("==================================================")


if __name__ == "__main__":
    main()