# CoolTrack 🪐

**CoolTrack** is a machine-learning-accelerated physics engine for calculating the thermal evolution, structural contraction, and photometric light curves of gas giant exoplanets and brown dwarfs. 

By training an ensemble of XGBoost surrogate models on the HADES grid, CoolTrack bypasses the numerical instability of traditional 1D structural evolution codes. It cleans broken grid points, learns the underlying thermodynamic laws, and uses a custom ODE solver to integrate exact planetary ages from hot-start initial conditions.

## ✨ Key Features

* **Self-Cleaning ML Engine:** Automatically identifies and removes broken physics/numerical noise from raw evolution grids before training robust XGBoost surrogate models.
* **Exact Age Integration:** Integrates the cooling rate ($dS/dt$) to calculate planetary ages using precise "hot-start" boundary conditions based on exact core masses and composition.
* **Structural & Thermal Tracking:** Simulates the continuous evolution of Internal Temperature ($T_{int}$), Physical Entropy ($S_{physical}$), and Radius ($R_J$).
* **JWST Observables:** Predicts the evolution of 15 different photometric fluxes (MIRI and NIRISS), complete with a fuzzy-finder for easy filter selection.
* **Publication-Ready Smoothing:** Includes a suite of mathematical filters (Savitzky-Golay, Splines, Gaussian) to remove ML "staircase" artifacts from simulated tracks.
* **Multiprocessed Pipeline:** Uses `joblib` to distribute ODE integration across all CPU cores, calculating ages for grids of 100,000+ planets in minutes.
* **Smart Caching:** Saves trained models and clean data to disk locally to bypass retraining times in Jupyter Notebooks.

---

## 📂 Repository Structure

```text
cooltrack/
├── data/
│   ├── HADES_grid/      # Raw and processed .parquet grid files
│   ├── age_data/        # Hot-start initial condition CSVs
│   └── models/          # Cached XGBoost .json models
├── notebooks/           # Jupyter notebooks for exploration and plotting
│   ├── explore_dsdt.ipynb
│   └── test_cooling_tracks.ipynb
├── scripts/             
│   └── main.py          # The master multiprocessing pipeline script
├── src/cooltrack/       # The core Python package
│   ├── __init__.py
│   ├── constants.py     # Physical constants and JWST Bands fuzzy-finder
│   ├── data_loader.py   # Parquet ingestion and log10 conversions
│   ├── initial_conditions.py # Hot-start boundary interpolators
│   ├── integrator.py    # SciPy ODE solver for dS/dt
│   ├── models.py        # XGBoost ensemble and outlier cleaning
│   └── smoothing.py     # Savitzky-Golay and spline filters
└── README.md
```

---

## 🚀 Quick Start

### 1. Run the Master Pipeline
To calculate the ages for the entire HADES grid in bulk, run the master script from the root directory of your project:

```bash
python scripts/main.py
```
* **First Run:** Loads the raw parquet file, cleans outliers, trains 18 XGBoost models (State, Radius, dS/dt, and 15 JWST bands), saves the models to `data/models/`, and runs the parallel integrator.
* **Subsequent Runs:** Instantly loads the cached models and clean grid, skipping directly to integration.

### 2. Using the API in Jupyter Notebooks
You can easily simulate a bespoke planet (e.g., a Jupiter analog) directly in a notebook:

```python
import pandas as pd
from cooltrack.models import ThermalEvolutionModels
from cooltrack.integrator import CoolingIntegrator
from cooltrack.initial_conditions import InitialConditions
from cooltrack.constants import Bands

# 1. Load Pre-trained Models and Clean Grid
ml_engine = ThermalEvolutionModels()
ml_engine.load_models("../../data/models/")
df_clean = pd.read_parquet("../../data/HADES_grid/hades_clean_grid.parquet")

# 2. Initialize Physics Engine
init_cond = InitialConditions("../../data/age_data/")
integrator = CoolingIntegrator(ml_engine)

# 3. Define your target planet (e.g., matching a grid row, then overriding parameters)
planet_row = df_clean.iloc[0].copy()
planet_row['mass_Mj'] = 1.0     # 1 Jupiter Mass
planet_row['T_irr'] = 150.0     # 150 K Irradiation
planet_row['Met'] = 3.0         # 3x Solar Metallicity
planet_row['core'] = 10.0       # 10 Earth-mass core

# 4. Get Boundary Conditions
s_hot_start = init_cond.get_starting_physical_entropy(planet_row['mass_Mj'])
s_cold_end = df_clean['S_physical'].min()

# 5. Integrate the track!
ages, entropies = integrator.calculate_track(planet_row, s_hot_start, s_cold_end)
```

---

## 🛠 Advanced Tools

### The JWST Fuzzy Finder
No need to memorize exact string names for photometric filters. Use `Bands.find()` to dynamically grab the correct column for MIRI or NIRISS:

```python
from cooltrack.constants import Bands

# Type it casually; the fuzzy finder will match it!
my_filter = Bands.find('miri 1000') 
print(my_filter) # Outputs: 'MIRI_F1000W_Flambda_wm2um'

# Predict the flux
log_flux = ml_engine.photo_models[my_filter].predict(input_features)
```

### Publication-Ready Smoothing
XGBoost predictions can look jagged when zoomed in. Use `TrackSmoother` to apply physically realistic smoothing algorithms to your cooling curves:

```python
from cooltrack.smoothing import TrackSmoother

# Smooth out the ML staircase artifacts using a Savitzky-Golay filter
t_ints_smooth = TrackSmoother.smooth(
    x=ages_array, 
    y=raw_temperatures, 
    method='savgol', 
    window_length=31, 
    polyorder=3
)
```

---
*Developed for robust, fast, and scalable planetary thermal evolution modeling.*