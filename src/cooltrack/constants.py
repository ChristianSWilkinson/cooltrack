"""
Physical constants, model features, and photometric band definitions for CoolTrack.

This module houses all the fundamental physical constants, the features required 
by the ML engine, and a helper class to manage and retrieve JWST photometric 
bands using exact or fuzzy string matching.
"""

import difflib

# --- Planetary & Solar Constants ---
R_J = 69911000.0          # Jupiter radius (m)
M_J = 1.898e27            # Jupiter mass (kg)
R_S = 696340000.0         # Solar radius (m)
M_S = 1.989e30            # Solar mass (kg)
R_E = 6371000.0           # Earth radius (m)
M_E = 5.972e24            # Earth mass (kg)

# --- Physics Constants ---
G_U = 6.674e-11           # Gravitational constant (m^3 kg^-1 s^-2)
N_A = 6.022e23            # Avogadro's number (mol^-1)
K_B = 1.380649e-23        # Boltzmann constant (J K^-1)
R_G = 8.31446261815324    # Universal gas constant (J mol^-1 K^-1)
SIGMA = 5.67e-8           # Stefan-Boltzmann constant (W m^-2 K^-4)
MASS_PARTICLE_APPROX_KG = 1.6735e-27  # Approx mass of a proton/neutron (kg)

# --- Time Constants ---
SECONDS_PER_YR = 3600 * 24 * 365.25

# --- Features & Targets ---
INDEPENDENT_DIMS = ['mass_Mj', 'T_irr', 'Met', 'core', 'f_sed', 'kzz']


class Bands:
    """
    Helper class for easy autocomplete and fuzzy searching of photometry bands.
    
    This class stores the exact column names for JWST MIRI and NIRISS 
    filters used in the CoolTrack grid. It also provides a method to 
    fuzzy-match colloquial filter names to their exact column strings.
    """

    # JWST MIRI
    MIRI_F770W = 'MIRI_F770W_Flambda_wm2um'
    MIRI_F1000W = 'MIRI_F1000W_Flambda_wm2um'
    MIRI_F1065C = 'MIRI_F1065C_Flambda_wm2um'
    MIRI_F1140C = 'MIRI_F1140C_Flambda_wm2um'
    MIRI_F1130W = 'MIRI_F1130W_Flambda_wm2um'
    MIRI_F1280W = 'MIRI_F1280W_Flambda_wm2um'
    MIRI_F1500W = 'MIRI_F1500W_Flambda_wm2um'
    MIRI_F1550C = 'MIRI_F1550C_Flambda_wm2um'
    MIRI_F1800W = 'MIRI_F1800W_Flambda_wm2um'
    MIRI_F2100W = 'MIRI_F2100W_Flambda_wm2um'
    MIRI_F2300C = 'MIRI_F2300C_Flambda_wm2um'
    
    # JWST NIRISS
    NIRISS_F277W = 'NIRISS_F277W_Flambda_wm2um'
    NIRISS_F380M = 'NIRISS_F380M_Flambda_wm2um'
    NIRISS_F430M = 'NIRISS_F430M_Flambda_wm2um'
    NIRISS_F480M = 'NIRISS_F480M_Flambda_wm2um'

    @classmethod
    def find(cls, search_term: str) -> str:
        """
        Find the exact column name for a photometric band using fuzzy matching.
        
        Takes a casual string representation of a filter (e.g., 'miri 1000',
        'f277w') and attempts to match it against the defined class attributes.
        It prioritizes exact substring matches before falling back to
        difflib's fuzzy matching algorithm.

        Args:
            search_term (str): The colloquial or approximate name of the band.

        Returns:
            str: The exact column name matching the search term.

        Raises:
            ValueError: If no suitable match is found within the cutoff threshold.
        """
        valid_bands = {
            k: v for k, v in vars(cls).items() 
            if not k.startswith('_') and isinstance(v, str)
        }
        
        # Clean up the search term
        clean_search = str(search_term).upper().replace(' ', '_').replace('-', '_')

        # 1. Exact substring match
        for key, exact_col_name in valid_bands.items():
            if clean_search in key:
                return exact_col_name

        # 2. Fuzzy matching fallback
        possible_keys = list(valid_bands.keys())
        matches = difflib.get_close_matches(
            clean_search, possible_keys, n=1, cutoff=0.3
        )

        if matches:
            best_match = matches[0]
            print(
                f"Bands.find(): Guessed '{best_match}' "
                f"from input '{search_term}'"
            )
            return valid_bands[best_match]
        
        # 3. No match found
        raise ValueError(
            f"Could not find a photometric band matching '{search_term}'. "
            f"Available options: {possible_keys}"
        )


# Automatically generate the list of all band strings for the data loader
PHOTOMETRY_BANDS = [
    value for key, value in vars(Bands).items() 
    if not key.startswith('_') and isinstance(value, str)
]