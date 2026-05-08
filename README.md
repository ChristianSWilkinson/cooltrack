# CoolTrack 🪐

**CoolTrack** is a semi-analytical modeling framework for planetary thermal evolution. It provides tools to predict and extract thermodynamic pathways, cooling rates, structural radii, and photometry for substellar objects (gas giants and brown dwarfs). By converting raw, heavy grids into continuous mathematical surrogate models, CoolTrack enables microsecond-scale parameter evaluations and robust Monte Carlo error propagation.

## ✨ Key Features

* **Ultra-Fast Predictor Engine:** Uses a KDTree-backed coefficient database (`CoolTrackPredictor`) to instantly evaluate evolutionary tracks without loading massive raw HDF5/Parquet data.
* **Semi-Analytical Surrogate Fits:** Employs advanced, localized curve fitting (including piecewise softplus transitions, B-splines, and sigmoids) to continuously map thermodynamics, radius, and photometry across the parameter space.
* **Robust Uncertainty Propagation:** Integrates strict Monte Carlo error propagation using regularized covariance matrices to generate scientifically accurate confidence intervals.
* **Physical Boundary Conditions:** Automatically interpolates specific starting entropies bounded by theoretical "hot start" (gravitational collapse) and "cold start" (core accretion) formation scenarios.
* **Parallel Grid Builder:** Features an automated, multi-core processing script to rapidly ingest raw grids and compile the lightweight analytical coefficient database.

---

## 📂 Repository Structure

```text
cooltrack/
├── data/
│   ├── age_data/                        # CSVs containing hot/cold start boundary condition data
│   └── cooltrack_coefficients.dat       # The compiled, lightweight analytical database
├── notebooks/                           # Jupyter notebooks for exploration and plotting
├── scripts/             
│   └── build_coefficient_grid.py        # Master parallel script to generate the .dat database
├── src/cooltrack/                       # The core Python package
│   ├── cooltrack.py                     # Core SemiAnalytical engine and mathematical fitting logic
│   ├── predictor.py                     # Ultra-fast KDTree-based evaluation API
│   ├── initial_conditions.py            # Hot/cold start entropy boundary condition logic
│   └── data_loader.py                   # HDF5/Parquet ingestion, cleaning, and caching
└── README.md
```

---

## 📏 Input Parameters & Units
When constructing a custom planet or modifying grid rows, the engine expects the independent dimensions (`INDEPENDENT_DIMS`) to be provided in the exact following units/scales:

| Parameter | Description | Units / Scale | Example |
| :--- | :--- | :--- | :--- |
| `mass_Mj` | Planet Mass | Jupiter Masses ($M_J$) | `1.0` |
| `T_irr` | Irradiation Temperature | Kelvin (K) | `150.0` |
| `Met` | Metallicity | $\log_{10}$ (relative to Solar) | `np.log10(3.0)` for 3x Solar |
| `core` | Core Mass | Earth Masses ($M_\oplus$) | `10.0` |
| `f_sed_volatile` | Volatile Cloud Sedimentation | Unitless | `6.0` |
| `f_sed_refractory`| Refractory Cloud Sedimentation | Unitless | `6.0` |
| `kzz` | Eddy Diffusion Coefficient | $\log_{10}(\text{cm}^2/\text{s})$ | `8.0` for $10^8 \text{ cm}^2/\text{s}$ |

---

## 🚀 Quick Start

### 1. Build the Analytical Database
To compile the raw planetary grid into the fast surrogate coefficient database, run the master script from the `scripts/` directory. This will automatically utilize available CPU cores to process the HDF5/Parquet files.

```bash
cd scripts
python build_coefficient_grid.py
```
ONLY DO IF YOU HAVE ACCESS TO THE UNDERLYING GRID OTHERWISE USE JOINED COEFFICIENT DTABASE PROVIDED.

*This outputs `cooltrack_coefficients.dat` into your `data/` folder.*

### 2. Using the Fast Predictor API
For everyday scientific use and pipeline integration, the `CoolTrackPredictor` is the recommended interface. It bypasses raw data loading entirely for microsecond evaluations.

```python
from cooltrack.predictor import CoolTrackPredictor

# 1. Boot up the Oracle
oracle = CoolTrackPredictor(
    dat_filepath="../data/cooltrack_coefficients.dat", 
    age_data_path="../data/age_data"
)

# 2. Define your target planet
target_planet = {
    'mass_Mj': 5.0, 
    'T_irr': 0.0, 
    'Met': 0.0, 
    'core': 10.0,
    'f_sed_refractory': 6.0, 
    'f_sed_volatile': 6.0, 
    'kzz': 8.0
}

target_age_yr = 150e6  # 150 Million Years

# 3. Evaluate instantly (n_draws=0 for median only, >0 for Monte Carlo uncertainties)
results = oracle.predict(
    target_planet=target_planet, 
    target_age_yr=target_age_yr, 
    start_type=19,     # 19 = Hottest start, 0 = Coldest start
    n_draws=250        # Calculate 1-sigma uncertainty bounds
)

print(f"Radius: {results['Req_Rj']:.2f} R_J")
print(f"T_int: {results['T_int']:.0f} K (+{results['T_int_upper'] - results['T_int']:.0f} / -{results['T_int'] - results['T_int_lower']:.0f})")
```

---
*Developed for robust, fast, and scalable planetary thermal evolution modeling.*