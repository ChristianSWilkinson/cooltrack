# CoolTrack 🪐

**CoolTrack** is an advanced semi-analytical modeling framework explicitly designed for the thermal evolution of substellar objects (gas giants and brown dwarfs). By converting dense, computationally heavy planetary grids into smooth, continuous mathematical surrogate models, CoolTrack enables microsecond-scale parameter evaluations and highly robust Monte Carlo error propagation. This tool is ideal for researchers needing rapid mass, age, and flux estimations with rigorous statistical bounds.

## ✨ Key Features

* **Ultra-Fast Predictor Engine:** Uses a localized KDTree-backed coefficient database (`CoolTrackPredictor`) to instantly evaluate multi-dimensional evolutionary tracks, entirely bypassing the need to load massive raw HDF5 or Parquet grids into memory.
* **Semi-Analytical Surrogate Fits:** Employs advanced, localized curve fitting—including piecewise softplus transitions, regularized B-splines, and sigmoids—to continuously map thermodynamic variables ($S$), structural radii ($R$), and multi-band photometry across the entire parameter space.
* **Robust Uncertainty Propagation:** Integrates strict Monte Carlo error propagation using regularized covariance matrices to generate scientifically accurate confidence intervals. The architecture guarantees mathematically stable boundaries by strictly enforcing the laws of thermodynamics during random draws.
* **Orthogonal Error Decomposition:** Intelligently separates thermodynamic uncertainty (X-axis: Age) from structural Equation of State uncertainty (Y-axis: Radius), preventing the visual double-counting of variance and ensuring clean, physically meaningful confidence intervals.
* **Physical Boundary Conditions:** Automatically interpolates specific starting entropies bounded by theoretical "hot start" (gravitational collapse) and "cold start" (core accretion) formation scenarios, cleanly teleporting cold-start models to their appropriate evolutionary phase.
* **Heteroscedastic Mass Inversion:** Includes a production-grade `DoubleShotMassInverter` module designed for direct observational data (e.g., JWST images). It uses a Predictor-Corrector algorithm to map flux maps into statistically bounded mass limit arrays, automatically handling complex instrument, surrogate, and grid-gap noise profiles.

---

## 📂 Repository Structure

```text
cooltrack/
├── data/
│   ├── age_data/                        # CSVs containing hot/cold start boundary condition data
│   └── cooltrack_coefficients.dat       # The compiled, lightweight analytical database
├── notebooks/                           # Jupyter notebooks for exploration, tutorials, and plotting
│   ├── explainer.ipynb                  # Tutorial: Drawing simple and complex evolution curves
│   └── detection_map_explainer.ipynb    # Tutorial: Advanced JWST image mass inversions
├── scripts/             
│   └── build_coefficient_grid.py        # Master parallel script to generate the .dat database from raw files
├── src/cooltrack/                       # The core Python package
│   ├── cooltrack.py                     # Core SemiAnalytical engine and mathematical surrogate fitting logic
│   ├── predictor.py                     # Ultra-fast KDTree-based evaluation API
│   ├── initial_conditions.py            # Hot/cold start entropy boundary condition logic
│   └── mass_mapper.py                   # Heteroscedastic Double-Shot Inversion pipeline for observations
└── README.md
```

---

## 📖 Tutorials & Notebooks

To get up to speed with CoolTrack's advanced capabilities, we highly recommend executing the included Jupyter notebooks in this specific order:

1.  **`notebooks/explainer.ipynb`**
    * **Start Here.** This notebook serves as the foundational tutorial. It walks you through initializing the Oracle predictor, drawing fundamental planetary thermal evolution curves, and understanding Orthogonal Error Decomposition (separating thermodynamic age uncertainty from structural radius uncertainty). It demonstrates exactly how physical ceilings and floors are managed during Monte Carlo sampling.

2.  **`notebooks/detection_map_explainer.ipynb`**
    * **Advanced Pipeline.** Once you understand the base evolution tracks, dive into this notebook to see CoolTrack deployed in a production setting. It provides a complete tutorial on performing Heteroscedastic Predictor-Corrector mass inversions—teaching you how to convert raw, noisy telescopic flux limits (like those generated from JWST FITS files) into robust, statistically bounded 2D planetary mass limit maps.

---

## 📏 Input Parameters & Units

When constructing a custom target planet or modifying the underlying grid rows, the engine strictly expects the independent dimensions (`INDEPENDENT_DIMS`) to be provided in the exact following units and scales:

| Parameter | Description | Units / Scale | Example |
| :--- | :--- | :--- | :--- |
| `mass_Mj` | Planet Mass | Jupiter Masses ($M_J$) | `1.0` |
| `T_irr` | Irradiation Temperature | Kelvin (K) | `150.0` |
| `Met` | Metallicity | $\log_{10}$ (relative to Solar) | `np.log10(3.0)` for 3x Solar |
| `core` | Core Mass | Earth Masses ($M_\oplus$) | `10.0` |
| `f_sed_volatile` | Volatile Cloud Sedimentation | Unitless Parameter | `6.0` |
| `f_sed_refractory`| Refractory Cloud Sedimentation | Unitless Parameter | `6.0` |
| `kzz` | Eddy Diffusion Coefficient | $\log_{10}(\text{cm}^2/\text{s})$ | `8.0` for $10^8 \text{ cm}^2/\text{s}$ |

---

## 🚀 Quick Start

### 1. Build the Analytical Database

To compile the massive raw planetary grid into the fast, localized surrogate coefficient database, execute the master script from the `scripts/` directory. This process will automatically utilize available CPU cores to ingest the raw HDF5/Parquet files.

```bash
cd scripts
python build_coefficient_grid.py
```

> **⚠️ NOTE:** Only execute this build script if you have access to the underlying raw physical grids on your machine. Otherwise, skip this compilation step entirely and use the pre-compiled `cooltrack_coefficients.dat` database provided natively in the `data/` folder.

### 2. Installation

To install the `cooltrack` module for use across your local environment, run the following command from the root directory:

```bash
pip install -e .
```

### 3. Using the Fast Predictor API (Forward Modeling)

For everyday scientific exploration and pipeline integration, the `CoolTrackPredictor` is the primary interface. It bypasses raw grid loading entirely, offering sub-millisecond parameter evaluations.

```python
from cooltrack.predictor import CoolTrackPredictor

# 1. Boot up the Oracle
oracle = CoolTrackPredictor(
    dat_filepath="../data/cooltrack_coefficients.dat", 
    age_data_path="../data/age_data"
)

# 2. Define your target physical parameters
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

# 3. Evaluate instantly (n_draws=0 for median only, >0 for bounded Monte Carlo uncertainties)
results = oracle.predict(
    target_planet=target_planet, 
    target_age_yr=target_age_yr, 
    start_type=19,     # 19 = Hottest Start (Collapse), 0 = Coldest Start (Core Accretion)
    n_draws=250        # Calculate strict 1-sigma uncertainty bounds
)

print(f"Radius: {results['Req_Rj']:.2f} R_J")
print(f"T_int: {results['T_int']:.0f} K (+{results['T_int_upper'] - results['T_int']:.0f} / -{results['T_int'] - results['T_int_lower']:.0f})")
```

### 4. Using the Double-Shot Inverter (Observational Inversion)

If you are working with direct imaging data and need to translate a flux limit into a planet mass, use the `DoubleShotMassInverter`. It natively handles scalar values, 1D arrays, and 2D image matrices.

```python
from cooltrack.mass_mapper import DoubleShotMassInverter

# Initialize the inverter for your specific host star system
inverter = DoubleShotMassInverter(
    oracle=oracle,
    distance_pc=22.48,
    system_age_yr=5000e6,
    observed_band='JWST/MIRI.F1500W'
)

# The inverter natively handles single pixels, 1D profiles, or entire 2D image maps!
flux_limit = 1.2e-16 # W/m^2/um
age_error = 5000e6 * 0.10 # 10% age uncertainty

median_mass, mass_error = inverter.get_mass(flux_limit, age_error, n_draws=100)

print(f"Detected Mass Limit: {median_mass:.2f} ± {mass_error:.2f} M_J")
```

---
*Developed for robust, fast, and scalable planetary thermal evolution modeling.*