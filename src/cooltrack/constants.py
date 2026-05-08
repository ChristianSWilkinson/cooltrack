"""
Physical constants, model features, and photometric band definitions for CoolTrack.

This module houses all the fundamental physical constants, the features required 
by the ML engine, and a helper class to manage and retrieve photometric 
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
INDEPENDENT_DIMS = ['log10_mass_Mj', 'T_irr', 'Met', 'core', 'f_sed_refractory', 'f_sed_volatile', 'kzz']


class Bands:
    """
    Helper class for easy autocomplete and fuzzy searching of photometry bands.
    Stores the exact SVO/Species column names for all filters used in the CoolTrack grid.
    """

    # --- JWST (NIRCam) ---
    NIRCAM_F070W = "JWST/NIRCam.F070W"
    NIRCAM_F090W = "JWST/NIRCam.F090W"
    NIRCAM_F115W = "JWST/NIRCam.F115W"
    NIRCAM_F140M = "JWST/NIRCam.F140M"
    NIRCAM_F150W = "JWST/NIRCam.F150W"
    NIRCAM_F182M = "JWST/NIRCam.F182M"
    NIRCAM_F200W = "JWST/NIRCam.F200W"
    NIRCAM_F210M = "JWST/NIRCam.F210M"
    NIRCAM_F250M = "JWST/NIRCam.F250M"
    NIRCAM_F277W = "JWST/NIRCam.F277W"
    NIRCAM_F300M = "JWST/NIRCam.F300M"
    NIRCAM_F335M = "JWST/NIRCam.F335M"
    NIRCAM_F356W = "JWST/NIRCam.F356W"
    NIRCAM_F410M = "JWST/NIRCam.F410M"
    NIRCAM_F430M = "JWST/NIRCam.F430M"
    NIRCAM_F444W = "JWST/NIRCam.F444W"
    NIRCAM_F460M = "JWST/NIRCam.F460M"
    NIRCAM_F480M = "JWST/NIRCam.F480M"

    # --- JWST (MIRI) ---
    MIRI_F560W = "JWST/MIRI.F560W"
    MIRI_F770W = "JWST/MIRI.F770W"
    MIRI_F1000W = "JWST/MIRI.F1000W"
    MIRI_F1130W = "JWST/MIRI.F1130W"
    MIRI_F1280W = "JWST/MIRI.F1280W"
    MIRI_F1500W = "JWST/MIRI.F1500W"
    MIRI_F1800W = "JWST/MIRI.F1800W"
    MIRI_F2100W = "JWST/MIRI.F2100W"
    MIRI_F2550W = "JWST/MIRI.F2550W"

    # --- JWST (NIRISS) ---
    NIRISS_F090W = "JWST/NIRISS.F090W"
    NIRISS_F115W = "JWST/NIRISS.F115W"
    NIRISS_F140M = "JWST/NIRISS.F140M"
    NIRISS_F150W = "JWST/NIRISS.F150W"
    NIRISS_F158M = "JWST/NIRISS.F158M"
    NIRISS_F200W = "JWST/NIRISS.F200W"
    NIRISS_F277W = "JWST/NIRISS.F277W"
    NIRISS_F380M = "JWST/NIRISS.F380M"
    NIRISS_F430M = "JWST/NIRISS.F430M"
    NIRISS_F480M = "JWST/NIRISS.F480M"

    # --- VLT (Paranal) ---
    SPHERE_IRDIS_B_Y = "Paranal/SPHERE.IRDIS_B_Y"
    SPHERE_IRDIS_B_J = "Paranal/SPHERE.IRDIS_B_J"
    SPHERE_IRDIS_B_H = "Paranal/SPHERE.IRDIS_B_H"
    SPHERE_IRDIS_B_KS = "Paranal/SPHERE.IRDIS_B_Ks"
    SPHERE_IRDIS_D_J23_2 = "Paranal/SPHERE.IRDIS_D_J23_2"
    SPHERE_IRDIS_D_J23_3 = "Paranal/SPHERE.IRDIS_D_J23_3"
    SPHERE_IRDIS_D_H23_2 = "Paranal/SPHERE.IRDIS_D_H23_2"
    SPHERE_IRDIS_D_H23_3 = "Paranal/SPHERE.IRDIS_D_H23_3"
    SPHERE_IRDIS_D_K12_1 = "Paranal/SPHERE.IRDIS_D_K12_1"
    SPHERE_IRDIS_D_K12_2 = "Paranal/SPHERE.IRDIS_D_K12_2"
    
    NACO_J = "Paranal/NACO.J"
    NACO_H = "Paranal/NACO.H"
    NACO_KS = "Paranal/NACO.Ks"
    NACO_LP = "Paranal/NACO.Lp"
    NACO_MP = "Paranal/NACO.Mp"
    
    HAWKI_J = "Paranal/HAWKI.J"
    HAWKI_H = "Paranal/HAWKI.H"
    HAWKI_KS = "Paranal/HAWKI.Ks"
    HAWKI_CH4 = "Paranal/HAWKI.CH4"
    
    VISIR_B87 = "Paranal/VISIR.B87"
    VISIR_SIV = "Paranal/VISIR.SIV"
    VISIR_PAH2 = "Paranal/VISIR.PAH2"

    # --- KECK & GEMINI ---
    NIRC2_J = "Keck/NIRC2.J"
    NIRC2_H = "Keck/NIRC2.H"
    NIRC2_KS = "Keck/NIRC2.Ks"
    NIRC2_KP = "Keck/NIRC2.Kp"
    NIRC2_LP = "Keck/NIRC2.Lp"
    NIRC2_MS = "Keck/NIRC2.Ms"
    
    NIRI_J_G0202W = "Gemini/NIRI.J-G0202w"
    NIRI_H_G0203W = "Gemini/NIRI.H-G0203w"
    NIRI_K_G0204W = "Gemini/NIRI.K-G0204w"
    NIRI_LPRIME_G0207W = "Gemini/NIRI.Lprime-G0207w"
    NIRI_MPRIME_G0208W = "Gemini/NIRI.Mprime-G0208w"

    # --- SPACE: HST, SPITZER, WISE ---
    WFC3_IR_F110W = "HST/WFC3_IR.F110W"
    WFC3_IR_F140W = "HST/WFC3_IR.F140W"
    WFC3_IR_F160W = "HST/WFC3_IR.F160W"
    WFC3_UVIS1_F606W = "HST/WFC3_UVIS1.F606W"
    WFC3_UVIS1_F814W = "HST/WFC3_UVIS1.F814W"
    
    IRAC_I1 = "Spitzer/IRAC.I1"
    IRAC_I2 = "Spitzer/IRAC.I2"
    IRAC_I3 = "Spitzer/IRAC.I3"
    IRAC_I4 = "Spitzer/IRAC.I4"
    
    WISE_W1 = "WISE/WISE.W1"
    WISE_W2 = "WISE/WISE.W2"
    WISE_W3 = "WISE/WISE.W3"
    WISE_W4 = "WISE/WISE.W4"

    # --- NEXT-GEN SPACE: ROMAN ---
    WFI_F062 = "Roman/WFI.F062"
    WFI_F087 = "Roman/WFI.F087"
    WFI_F106 = "Roman/WFI.F106"
    WFI_F129 = "Roman/WFI.F129"
    WFI_F146 = "Roman/WFI.F146"
    WFI_F158 = "Roman/WFI.F158"
    WFI_F184 = "Roman/WFI.F184"

    # --- ALL-SKY SURVEYS ---
    _2MASS_J = "2MASS/2MASS.J"
    _2MASS_H = "2MASS/2MASS.H"
    _2MASS_KS = "2MASS/2MASS.Ks"
    
    SDSS_U = "SLOAN/SDSS.u"
    SDSS_G = "SLOAN/SDSS.g"
    SDSS_R = "SLOAN/SDSS.r"
    SDSS_I = "SLOAN/SDSS.i"
    SDSS_Z = "SLOAN/SDSS.z"
    
    PS1_G = "PAN-STARRS/PS1.g"
    PS1_R = "PAN-STARRS/PS1.r"
    PS1_I = "PAN-STARRS/PS1.i"
    PS1_Z = "PAN-STARRS/PS1.z"
    PS1_Y = "PAN-STARRS/PS1.y"
    
    GAIA3_G = "GAIA/GAIA3.G"
    GAIA3_GBP = "GAIA/GAIA3.Gbp"
    GAIA3_GRP = "GAIA/GAIA3.Grp"
    
    TESS_RED = "TESS/TESS.Red"
    KEPLER_K = "Kepler/Kepler.K"

    # --- UKIRT & VISTA ---
    WFCAM_Z = "UKIRT/WFCAM.Z"
    WFCAM_Y = "UKIRT/WFCAM.Y"
    WFCAM_J = "UKIRT/WFCAM.J"
    WFCAM_H = "UKIRT/WFCAM.H"
    WFCAM_K = "UKIRT/WFCAM.K"
    
    VISTA_Z = "Paranal/VISTA.Z"
    VISTA_Y = "Paranal/VISTA.Y"
    VISTA_J = "Paranal/VISTA.J"
    VISTA_H = "Paranal/VISTA.H"
    VISTA_KS = "Paranal/VISTA.Ks"

    # --- BESSELL ---
    BESSELL_U = "Generic/Bessell.U"
    BESSELL_B = "Generic/Bessell.B"
    BESSELL_V = "Generic/Bessell.V"
    BESSELL_R = "Generic/Bessell.R"
    BESSELL_I = "Generic/Bessell.I"

    # --- MKO (NSFCam & MIRSI) ---
    MKO_NSFCAM_J = "MKO/NSFCam.J"
    MKO_NSFCAM_H = "MKO/NSFCam.H"
    MKO_NSFCAM_K = "MKO/NSFCam.K"
    MKO_NSFCAM_KP = "MKO/NSFCam.Kp"
    MKO_NSFCAM_KS = "MKO/NSFCam.Ks"
    MKO_NSFCAM_LP = "MKO/NSFCam.Lp"
    MKO_NSFCAM_MP = "MKO/NSFCam.Mp"

    MKO_MIRSI_K = "MKO/MIRSI.K"
    MKO_MIRSI_4_9 = "MKO/MIRSI.4_9"
    MKO_MIRSI_7_7 = "MKO/MIRSI.7_7"
    MKO_MIRSI_8_7 = "MKO/MIRSI.8_7"
    MKO_MIRSI_9_7 = "MKO/MIRSI.9_7"
    MKO_MIRSI_N = "MKO/MIRSI.N"
    MKO_MIRSI_11_7 = "MKO/MIRSI.11_7"
    MKO_MIRSI_12_282 = "MKO/MIRSI.12_282"
    MKO_MIRSI_12_33 = "MKO/MIRSI.12_33"
    MKO_MIRSI_18 = "MKO/MIRSI.18"
    MKO_MIRSI_20 = "MKO/MIRSI.20"
    MKO_MIRSI_24 = "MKO/MIRSI.24"


    @classmethod
    def find(cls, search_term: str) -> str:
        """
        Find the exact column name for a photometric band using fuzzy matching.
        
        Takes a casual string representation of a filter (e.g., 'miri 1000',
        'f277w', 'mko j') and attempts to match it against the defined class attributes.
        It prioritizes exact substring matches before falling back to
        difflib's fuzzy matching algorithm.

        Args:
            search_term (str): The colloquial or approximate name of the band.

        Returns:
            str: The exact column name matching the search term.

        Raises:
            ValueError: If no suitable match is found within the cutoff threshold.
        """
        import difflib
        
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
    if not key.startswith('__') and isinstance(value, str)  # Ignore dunder methods, allow _2MASS
]