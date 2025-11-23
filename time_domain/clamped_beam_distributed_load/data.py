from re import L
import jax.numpy as jnp
from jax import config; config.update("jax_enable_x64", True)

# Beam and material parameters

dt = 0.005
T_end = 10

L = jnp.sqrt(5)

theta_0 = jnp.atan2(1, 2)

height, width = 0.05 , 0.05 # Cross-section dimensions (m)

kappa = 1               # Shear correction factor
A = height*width        # Cross-sectional area (m²)
I = width*height**3/12  # Second moment of area (m^4)
E = 2.1e11              # Young's modulus (Pa)
nu = 0.3
G = E/(2*(1+nu))        # Shear modulus (Pa)
rho = 2710              # Density (kg/m³)
g=9.81

geometrical_props = {
    'L': L,           # Length of the beam
    'A': A,             # Cross-sectional area
    'I': I,             # Second moment of area of the beam
    'k': kappa,         # Shear coefficient for rectungular cross-section
}

material_props = {
    'E': E,      # Young's modulus
    'G': G,      # Shear modulus 
    'rho': rho,  # Density
}
