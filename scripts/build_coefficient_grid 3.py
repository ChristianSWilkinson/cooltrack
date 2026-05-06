import sys
import os

# Tell Python to look in the parent directory for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pickle
import itertools

# Your imports will now work perfectly!
from cooltrack.cooltrack import SemiAnalyticalCoolTrack
from cooltrack.initial_conditions import InitialConditions
from cooltrack.data_loader import load_grid
from cooltrack.constants import INDEPENDENT_DIMS, Bands, PHOTOMETRY_BANDS  # <-- IMPORTED PHOTOMETRY_BANDS[cite: 1]

# =====================================================================
# 1. CONFIGURATION PATHS
# =====================================================================
AGE_DATA_PATH = "../data/age_data" 
GRID_PATH = "../../exoweave/outputs/master_grid_cooltrack.h5"   # Update this to your actual raw grid file

def main():
    # =====================================================================
    # 2. INITIALIZE THE CORE ENGINE
    # =====================================================================
    print(f"Loading initial boundary conditions from {AGE_DATA_PATH}...")
    init_conds = InitialConditions(age_data_path=AGE_DATA_PATH)

    print(f"Loading raw planetary grid from {GRID_PATH}...")
    # use_cache=True allows instant loading via Parquet on subsequent runs
    df_grid = load_grid(GRID_PATH, use_cache=True)

    print("Initializing SemiAnalyticalCoolTrack engine...")
    engine = SemiAnalyticalCoolTrack(
        grid_df=df_grid, 
        initial_conditions_model=init_conds, 
        independent_dims=INDEPENDENT_DIMS, 
        bandwidth=0.5
    )

    # =====================================================================
    # 3. DEFINE THE PARAMETER SPACE TO PRE-COMPUTE
    # =====================================================================
    mass_mj = np.linspace(1.0, 15.0, 15)  # 1 to 15 Jupiter masses
    t_irr = [0.0, 100.0, 500.0]
    met = [0.0]
    core = [10.0]
    fsed_ref = [3.0, 4.0]
    fsed_vol = [3.0, 4.0]
    kzz = [8.0]

    # Create all combinations
    grid_points = list(itertools.product(mass_mj, t_irr, met, core, fsed_ref, fsed_vol, kzz))
    
    # --- UPDATED: Dynamically generate the list for ALL bands ---
    # This automatically pulls every band from your constants file and adds 'log_'[cite: 1]
    phot_bands = [f"log_{band}" for band in PHOTOMETRY_BANDS]

    precomputed_fits = []

    print(f"Pre-computing {len(grid_points)} analytical models across {len(phot_bands)} bands...")

    # =====================================================================
    # 4. RUN THE EXTRACTION LOOP
    # =====================================================================
    for i, pt in enumerate(grid_points):
        target = {
            'mass_Mj': pt[0], 
            'T_irr': pt[1], 
            'Met': pt[2], 
            'core': pt[3],
            'f_sed_refractory': pt[4], 
            'f_sed_volatile': pt[5], 
            'kzz': pt[6]
        }
        
        # Run the heavy extraction using B-splines for photometry
        fits = engine.fit_surrogate(
            target_planet=target, 
            photometry_bands=phot_bands, 
            photometry_method='bspline'
        )
        
        # --- CRITICAL OPTIMIZATION ---
        # The evolve() method only needs track_data to find the coldest grid temperature (T_end).
        # We replace the massive raw grid with a tiny 1-row dummy dataframe to save huge amounts of memory.
        min_tint = fits['track_data']['ln_Tint'].min()
        fits['track_data'] = pd.DataFrame({'ln_Tint': [min_tint]})
        
        # Store the parameters as a flat tuple for ultra-fast searching later
        fits['search_vector'] = pt 
        
        precomputed_fits.append(fits)
        
        if i % 50 == 0 and i > 0:
            print(f"Processed {i}/{len(grid_points)}")

    # =====================================================================
    # 5. SAVE TO DISK
    # =====================================================================
    output_file = "../data/cooltrack_coefficients.dat"
    with open(output_file, 'wb') as f:
        pickle.dump(precomputed_fits, f)

    print(f"✅ Coefficient grid safely saved to {output_file}!")

if __name__ == "__main__":
    main()