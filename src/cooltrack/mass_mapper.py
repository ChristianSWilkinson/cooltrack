import time
import warnings
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, interp1d

# ==========================================
# PHYSICAL CONSTANTS
# ==========================================
RJ_TO_M = 7.1492e7   # 1 Jupiter Radius in meters
PC_TO_M = 3.0857e16  # 1 Parsec in meters

def scale_to_earth_flux(surf_flux_dex, radius_rj, distance_pc):
    """
    Scales intrinsic model surface flux to observed flux at Earth.
    (4*pi*R^2) / (4*pi*d^2) simplifies to (R/d)^2
    """
    radius_m = radius_rj * RJ_TO_M
    distance_m = distance_pc * PC_TO_M
    
    # In log space: log((R/d)^2) = 2 * log(R/d)
    geometric_scaling_dex = 2.0 * np.log10(radius_m / distance_m)
    return surf_flux_dex + geometric_scaling_dex


class DoubleShotMassInverter:
    """
    Heteroscedastic Predictor-Corrector module for planetary mass inversion.
    Calibrates to a specific star system and safely bounds thermodynamic edge cases.
    """
    
    def __init__(
        self, 
        oracle, 
        distance_pc, 
        system_age_yr, 
        observed_band='JWST/MIRI.F1500W',
        min_mass=0.3, 
        max_mass=25.0,
        fixed_params=None
    ):
        self.oracle = oracle
        self.distance_pc = distance_pc
        self.system_age_yr = system_age_yr
        self.band = observed_band
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.fixed_params = fixed_params or {
            'T_irr': 0.0, 'Met': 0, 'core': 10.0, 
            'kzz': 8.0, 'f_sed_refractory': 6.0, 'f_sed_volatile': 6.0
        }
        
        # Out-of-bounds sentinel values (for edge truncation)
        self.under_val = self.min_mass - 0.01
        self.over_val = self.max_mass + 0.01

        print(f"Calibrating Inverter for Dist: {self.distance_pc} pc | Age: {self.system_age_yr/1e6:.1f} Myr")
        t0 = time.time()
        self._build_error_profiles()
        self._build_interpolation_surface()
        print(f"Calibration complete in {time.time()-t0:.1f}s.")

    def _build_error_profiles(self):
        """Calculates surrogate fit noise (B-splines) and discrete grid gaps."""
        # 1. Earth-Scaled Grid Gap Calibrator
        log_m_cal = np.linspace(np.log10(self.min_mass), np.log10(self.max_mass), 15)
        gap_c, gap_e = [], []
        
        for i in range(len(log_m_cal) - 1):
            m_low, m_high = 10**log_m_cal[i], 10**log_m_cal[i+1]
            
            r_low = self.oracle.predict({**self.fixed_params, 'mass_Mj': m_low}, self.system_age_yr, n_draws=0)
            r_high = self.oracle.predict({**self.fixed_params, 'mass_Mj': m_high}, self.system_age_yr, n_draws=0)
            
            f_low, rad_low = r_low.get(f"log_{self.band}", np.nan), r_low.get("Req_Rj", np.nan)
            f_high, rad_high = r_high.get(f"log_{self.band}", np.nan), r_high.get("Req_Rj", np.nan)
            
            if not np.isnan(f_low) and not np.isnan(f_high) and rad_low > 0 and rad_high > 0:
                obs_f_low = scale_to_earth_flux(f_low, rad_low, self.distance_pc)
                obs_f_high = scale_to_earth_flux(f_high, rad_high, self.distance_pc)
                
                gap_c.append(10**((log_m_cal[i] + log_m_cal[i+1]) / 2.0))
                gap_e.append(min(abs(obs_f_high - obs_f_low), 2.0))

        self.gap_centers = np.array(gap_c)
        self.gap_errors = np.array(gap_e)

        # 2. Integrated B-Spline Calibrator
        prof_m = np.linspace(self.min_mass, self.max_mass, 6)
        prof_e = []
        for pm in prof_m:
            res = self.oracle.predict({**self.fixed_params, 'mass_Mj': pm}, self.system_age_yr, n_draws=250)
            f_up, f_low = res.get(f"log_{self.band}_upper", np.nan), res.get(f"log_{self.band}_lower", np.nan)
            r_up, r_low = res.get("Req_Rj_upper", np.nan), res.get("Req_Rj_lower", np.nan)
            
            if not np.isnan(f_up) and not np.isnan(r_up) and r_up > 0 and r_low > 0:
                flux_err = (f_up - f_low) / 2.0
                rad_err = 2.0 * ((np.log10(r_up) - np.log10(r_low)) / 2.0)
                prof_e.append(np.sqrt(flux_err**2 + rad_err**2))
            else:
                prof_e.append(0.80)
                
        self.prof_masses = prof_m
        self.prof_errors = np.array(prof_e)

    def _build_interpolation_surface(self):
        """Constructs the dense 2D CloughTocher surface and convex hull trackers."""
        mass_range = np.logspace(np.log10(self.min_mass), np.log10(self.max_mass), 10) 
        age_range = np.logspace(np.log10(10e6), np.log10(10e9), 10)
        fluxes, ages, masses = [], [], []

        for m in mass_range:
            for a in age_range:
                res = self.oracle.predict({**self.fixed_params, 'mass_Mj': m}, a, n_draws=0)
                f, r = res.get(f"log_{self.band}", np.nan), res.get("Req_Rj", np.nan)
                if not np.isnan(f) and not np.isnan(r) and r > 0:
                    obs_flux = scale_to_earth_flux(f, r, self.distance_pc)
                    fluxes.append(obs_flux)
                    ages.append(np.log10(a))
                    masses.append(m)

        fluxes, ages, masses = np.array(fluxes), np.array(ages), np.array(masses)
        self.inverse_spline = CloughTocher2DInterpolator(np.column_stack((fluxes, ages)), masses)

        # Build boundary trackers for safe truncation outside the convex hull
        self.actual_min_mass = np.min(masses)
        self.actual_max_mass = np.max(masses)
        
        min_mask = (masses == self.actual_min_mass)
        max_mask = (masses == self.actual_max_mass)
        
        self.lower_flux_bound = interp1d(ages[min_mask], fluxes[min_mask], bounds_error=False, fill_value="extrapolate")
        self.upper_flux_bound = interp1d(ages[max_mask], fluxes[max_mask], bounds_error=False, fill_value="extrapolate")

    def _classify_and_bound(self, draw_fluxes, draw_ages, raw_masses):
        """Categorizes out-of-hull NaNs and truncates polynomial overshoots."""
        processed = np.copy(raw_masses)
        nan_mask = np.isnan(processed)
        
        processed[nan_mask & (draw_fluxes < self.lower_flux_bound(draw_ages))] = self.under_val
        processed[nan_mask & (draw_fluxes > self.upper_flux_bound(draw_ages))] = self.over_val
        
        processed[(~np.isnan(processed)) & (processed < self.actual_min_mass)] = self.under_val
        processed[(~np.isnan(processed)) & (processed > self.actual_max_mass)] = self.over_val
        
        return processed

    def get_mass(self, flux_w_m2_um, age_err_yr, instrument_flux_err=0.05, n_draws=100):
        """
        Executes the Double-Shot inversion on the provided flux data.
        Automatically scales to handle scalars, 1D arrays, or 2D images.
        """
        t_start = time.time()
        
        # Determine input shape to rebuild it later
        is_scalar = np.isscalar(flux_w_m2_um)
        flux_array = np.atleast_1d(flux_w_m2_um)
        orig_shape = flux_array.shape
        
        # Flatten inputs and convert to log-space for the interpolator
        flat_fluxes = np.log10(flux_array.flatten())
        n_pixels = len(flat_fluxes)
        
        # --- STEP 1: THE PREDICTOR ---
        median_ages = np.full(n_pixels, np.log10(self.system_age_yr))
        raw_first_guess = self.inverse_spline(flat_fluxes, median_ages)
        first_guess_masses = self._classify_and_bound(flat_fluxes, median_ages, raw_first_guess)

        # --- STEP 2: PIXEL-BY-PIXEL LOCAL ERROR LOOKUP ---
        pixel_variances = np.full(n_pixels, instrument_flux_err) 
        valid = (~np.isnan(first_guess_masses)) & (first_guess_masses > self.actual_min_mass) & (first_guess_masses < self.actual_max_mass)

        if np.any(valid):
            local_b_errs = np.interp(first_guess_masses[valid], self.prof_masses, self.prof_errors)
            local_g_errs = np.interp(first_guess_masses[valid], self.gap_centers, self.gap_errors)
            pixel_variances[valid] = np.sqrt(instrument_flux_err**2 + local_b_errs**2 + local_g_errs**2)

        # --- STEP 3: THE CORRECTOR (MC LOOP) ---
        # Draw normally distributed ages. Clip to > 0 to prevent log(0) errors
        mc_ages = np.random.normal(self.system_age_yr, age_err_yr, n_draws)
        mc_ages = mc_ages[mc_ages > 0] 
        
        mass_cube = np.zeros((len(mc_ages), n_pixels))

        for i, age_draw in enumerate(mc_ages):
            draw_ages_array = np.full(n_pixels, np.log10(age_draw))
            perturbed_fluxes = np.random.normal(flat_fluxes, pixel_variances)
            
            raw_draw_masses = self.inverse_spline(perturbed_fluxes, draw_ages_array)
            mass_cube[i, :] = self._classify_and_bound(perturbed_fluxes, draw_ages_array, raw_draw_masses)

        # --- STEP 4: EXTRACT PERCENTILES & RESHAPE ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p50_flat = np.nanpercentile(mass_cube, 50, axis=0)
            p84_flat = np.nanpercentile(mass_cube, 84, axis=0)
            
        # Reshape back to the user's original dimensions
        p50_out = p50_flat.reshape(orig_shape)
        unc_out = (p84_flat - p50_flat).reshape(orig_shape)
        
        print(f"Processed {len(mc_ages) * n_pixels} evaluations in {time.time() - t_start:.2f}s.")
        
        if is_scalar:
            return p50_out.item(), unc_out.item()
            
        return p50_out, unc_out