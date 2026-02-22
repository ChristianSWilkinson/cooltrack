# 🪐 CoolTrack 

**CoolTrack** is a machine learning-driven engine for calculating the thermal evolution and age estimation of gas giant planets. 

Built as a companion to `fuzzycore` and designed to ingest the massive **HADES** planetary grid, CoolTrack replaces traditional, slow, and brittle N-dimensional grid interpolators (`LinearNDInterpolator`) with blazing-fast **XGBoost** surrogate models. By learning the underlying thermodynamic derivatives, CoolTrack integrates physical cooling tracks (Age vs. Internal Temperature/Entropy) robustly across irregular grid boundaries.

---

## 🚀 Features
* **Out-of-Core Data Loading:** Safely slices and filters massive (8GB+) Parquet grids on local machines without crashing RAM, utilizing PyArrow predicate pushdown.
* **ML Surrogate Models:** Trains dual XGBoost regressors to predict internal temperature (`T_int`) and cooling rates (`dS/dt`) from physical state parameters.
* **Robust ODE Integration:** Uses `scipy.integrate.solve_ivp` to step through entropy states over time, producing smooth, physically consistent evolutionary cooling tracks.
* **Modern Packaging:** Built using the `src` layout for seamless local development and `pip` installation.

---

## 🛠️ Installation

Because CoolTrack relies on C-backed scientific libraries (like XGBoost and PyArrow), we highly recommend using a `conda` environment—especially if you are on an Apple Silicon Mac (M1/M2/M3) to avoid C-compiler conflicts.

### 1. Clone the repository
`git clone git@github.com:ChristianSWilkinson/cooltrack.git`
`cd cooltrack`

### 2. Set up your environment (Mac users)
If you are on a Mac, ensure you install the core math libraries via `conda-forge` before installing the package to prevent kernel crashes:
`conda install -c conda-forge xgboost pyarrow scipy scikit-learn pandas numpy`

### 3. Install CoolTrack (Editable Mode)
Install the package in "editable" mode so any changes you make to the source code are instantly reflected in your notebooks:
`pip install -e .`

---

## 📂 Repository Structure

```text
cooltrack/
├── pyproject.toml         # Package configuration for pip
├── .gitignore             # Ignores large data files (like the HADES grid)
├── README.md              # You are here!
├── notebooks/             # Jupyter notebooks for testing and plotting
│   └── test_cooling_tracks.ipynb
└── src/
    └── cooltrack/         # Main package source code
        ├── __init__.py
        ├── constants.py   # Physical constants and feature definitions
        ├── data_loader.py # PyArrow-backed Parquet loaders
        ├── models.py      # XGBoost training and prediction wrappers
        └── integrator.py  # ODE solvers for age and thermal tracks
```

*(Note: The `HADES` grid data is intentionally omitted from version control due to file size limits. Place your `.parquet` data files in a local directory ignored by Git, such as `../data/`)*

---

## 💻 Quick Start

Here is a minimal example of how to load the grid, train the surrogate models, and integrate a cooling track for a 1 Jupiter-mass planet.

```python
import numpy as np
from cooltrack.constants import INDEPENDENT_DIMS
from cooltrack.data_loader import load_and_clean_grid_pandas
from cooltrack.models import ThermalEvolutionModels
from cooltrack.integrator import CoolingIntegrator

# 1. Load a slice of the grid
GRID_PATH = "../data/HADES_grid/hades_processed_grid.parquet"
df = load_and_clean_grid_pandas(GRID_PATH)

# 2. Train the XGBoost Regressors
ml_engine = ThermalEvolutionModels()
ml_engine.train_models(df, tune_hyperparameters=False)

# 3. Setup the Integrator
integrator = CoolingIntegrator(ml_engine)

# 4. Define planet parameters and integrate!
# Fixed dims: ['mass_Mj', 'T_irr', 'Met', 'core', 'f_sed', 'kzz']
planet_row = df[(np.isclose(df['mass_Mj'], 1.0, atol=0.1)) & 
                (np.isclose(df['T_irr'], 500.0, atol=10))].iloc[0]

s_hot_start = df['S_physical'].max()
s_cold_end = df['S_physical'].min()

ages_yr, entropies = integrator.calculate_track(planet_row, s_hot_start, s_cold_end)

print(f"Integration complete! Final age: {ages_yr[-1]:.2e} years")
```