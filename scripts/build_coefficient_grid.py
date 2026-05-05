import sys
import os

# =====================================================================
# 0. SYSTEM CRITICAL THREAD LOCKS (Must be at the very top!)
# =====================================================================
# Prevent Numpy/Scipy from launching hidden threads inside our parallel workers.
# This prevents "CPU thrashing" and ensures maximum parallel efficiency.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Tell Python to look in the parent directory for modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pickle
import itertools
import multiprocessing as mp
from functools import partial

from cooltrack.cooltrack import SemiAnalyticalCoolTrack
from cooltrack.initial_conditions import InitialConditions
from cooltrack.data_loader import load_grid
from cooltrack.constants import INDEPENDENT_DIMS, Bands, PHOTOMETRY_BANDS  

AGE_DATA_PATH = "../data/age_data" 
GRID_PATH = "../../exoweave/outputs/master_grid_cooltrack.h5"   

# =====================================================================
# 1. THE WORKER FUNCTION (Must be top-level)
# =====================================================================
def process_grid_point(pt, engine, phot_bands):
    """Processes a single grid point. Executed independently by a CPU worker."""
    target = {
        'mass_Mj': pt[0], 
        'T_irr': pt[1], 
        'Met': pt[2], 
        'core': pt[3],
        'f_sed_refractory': pt[4], 
        'f_sed_volatile': pt[5], 
        'kzz': pt[6]
    }
    
    # Run the heavy extraction
    fits = engine.fit_surrogate(
        target_planet=target, 
        photometry_bands=phot_bands, 
        photometry_method='bspline'
    )
    
    # Memory optimization: Discard raw grid data, keep only the coldest temperature
    min_tint = fits['track_data']['ln_Tint'].min()
    fits['track_data'] = pd.DataFrame({'ln_Tint': [min_tint]})
    fits['search_vector'] = pt 
    
    return fits


def main():
    # =====================================================================
    # 2. INITIALIZE THE CORE ENGINE
    # =====================================================================
    print(f"Loading initial boundary conditions from {AGE_DATA_PATH}...")
    init_conds = InitialConditions(age_data_path=AGE_DATA_PATH)

    print(f"Loading raw planetary grid from {GRID_PATH}...")
    df_grid = load_grid(GRID_PATH, use_cache=True)

    print("Initializing SemiAnalyticalCoolTrack engine...")
    engine = SemiAnalyticalCoolTrack(
        grid_df=df_grid, 
        initial_conditions_model=init_conds, 
        independent_dims=INDEPENDENT_DIMS, 
        bandwidth=0.5
    )

    # =====================================================================
    # 3. DEFINE THE PARAMETER SPACE
    # =====================================================================
    mass_mj = np.logspace(np.log10(0.3), np.log10(15.0), 100)  
    t_irr = [0.0, 100.0, 500.0]
    met = [0.0]
    core = [10.0]
    fsed_ref = [3.0, 4.0]
    fsed_vol = [3.0, 4.0]
    kzz = [8.0]

    grid_points = list(itertools.product(mass_mj, t_irr, met, core, fsed_ref, fsed_vol, kzz))
    phot_bands = [f"log_{band}" for band in PHOTOMETRY_BANDS]

    # =====================================================================
    # 4. RUN MULTIPROCESSING
    # =====================================================================
    # Leave 1 core free so your computer doesn't completely freeze up
    num_cores = max(1, mp.cpu_count() - 1) 
    print(f"\n🚀 Launching parallel extraction across {num_cores} CPU cores...")
    print(f"Pre-computing {len(grid_points)} models across {len(phot_bands)} bands...\n")

    precomputed_fits = []
    
    # We use functools.partial to package the worker function with the engine and bands
    worker_func = partial(process_grid_point, engine=engine, phot_bands=phot_bands)

    # Launch the parallel pool
    with mp.Pool(num_cores) as pool:
        # imap_unordered is significantly faster than standard map() because it 
        # yields results as soon as they finish computing, preventing bottlenecks.
        for i, fits in enumerate(pool.imap_unordered(worker_func, grid_points)):
            precomputed_fits.append(fits)
            
            # Progress tracker
            if (i + 1) % 50 == 0 or (i + 1) == len(grid_points):
                percentage = ((i + 1) / len(grid_points)) * 100
                print(f"Processed {i + 1}/{len(grid_points)} ({percentage:.1f}%)")

    # =====================================================================
    # 5. SAVE TO DISK
    # =====================================================================
    output_file = "../data/cooltrack_coefficients.dat"
    with open(output_file, 'wb') as f:
        pickle.dump(precomputed_fits, f)

    print(f"\n✅ Coefficient grid safely saved to {output_file}!")

if __name__ == "__main__":
    main()