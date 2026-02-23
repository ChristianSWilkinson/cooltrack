import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from cooltrack.constants import INDEPENDENT_DIMS, Bands
from cooltrack.data_loader import load_and_clean_grid_pandas
from cooltrack.models import ThermalEvolutionModels
from cooltrack.integrator import CoolingIntegrator
from cooltrack.initial_conditions import InitialConditions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION ---
GRID_FILE_PATH = "../data/HADES_grid/hades_processed_grid.parquet" 
CLEAN_GRID_PATH = "../data/HADES_grid/hades_clean_grid.parquet" # Saves the outlier-free grid
MODELS_DIR = "../data/models/" # Where we dump the ML models
AGE_DATA_PATH = "../data/age_data/"
OUTPUT_FILE_PATH = "../data/HADES_grid/hades_grid_with_ages.parquet"

N_CORES = -1 


def compute_age(row_tuple, integrator, init_cond):
    """Worker function to calculate the exact age for a single row in the grid."""
    _, row = row_tuple 
    mass = row['mass_Mj']
    s_target = row['S_physical']
    
    try:
        s_hot_start = init_cond.get_starting_physical_entropy(mass, bin_index=19, n_bins=20)
        
        if s_target >= s_hot_start:
            return 0.0
            
        ages, _ = integrator.calculate_track(row, s_hot_start, s_target, num_points=10)
        
        if ages is not None and len(ages) > 0:
            return ages[-1] 
        else:
            return np.nan
    except Exception as e:
        return np.nan


def main():
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
        logging.info("No saved models found. Loading raw grid and training from scratch...")
        df_raw = load_and_clean_grid_pandas(GRID_FILE_PATH)
        df_clean = ml_engine.train_models(df_raw, tune_hyperparameters=False, clean_outliers=True)
        
        # Save them so we never have to do this again!
        ml_engine.save_models(MODELS_DIR)
        logging.info(f"Saving clean grid to {CLEAN_GRID_PATH}...")
        df_clean.to_parquet(CLEAN_GRID_PATH, engine='pyarrow', index=False)
        
    logging.info(f"Ready for integration with {len(df_clean):,} pristine grid points.")
    
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
        logging.warning(f"ODE Solver failed to converge for {failed} points. Dropped.")
        
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    logging.info(f"Saving final grid with ages to {OUTPUT_FILE_PATH}...")
    df_clean.to_parquet(OUTPUT_FILE_PATH, engine='pyarrow', index=False)
    
    print("\n==================================================")
    print("✅ PIPELINE COMPLETE! ✅")
    print(f"Total simulated planets saved: {len(df_clean):,}")
    print("==================================================")

if __name__ == "__main__":
    main()