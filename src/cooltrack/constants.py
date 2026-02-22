import numpy as np

# --- Planetary & Solar Constants ---
R_J = 69911000          # Jupiter radius (m)
M_J = 1.898e27          # Jupiter mass (kg)
R_S = 696340000         # Solar radius (m)
M_S = 1.989e30          # Solar mass (kg)
R_E = 6371e3            # Earth radius (m)
M_E = 5.972e24          # Earth mass (kg)

# --- Physics Constants ---
G_U = 6.674e-11         # Gravitational constant (m^3 kg^-1 s^-2)
N_A = 6.022e23          # Avogadro's number (mol^-1)
K_B = 1.380649e-23      # Boltzmann constant (J K^-1)
R_G = 8.31446261815324  # Universal gas constant (J mol^-1 K^-1)
SIGMA = 5.67e-8         # Stefan-Boltzmann constant (W m^-2 K^-4)
MASS_PARTICLE_APPROX_KG = 1.6735e-27

SECONDS_PER_YR = 3600 * 24 * 365.25

# --- Features & Targets ---
INDEPENDENT_DIMS = ['mass_Mj', 'T_irr', 'Met', 'core', 'f_sed', 'kzz']