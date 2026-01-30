import jax.numpy as jnp
from jax import config; config.update("jax_enable_x64", True)

T_end = 10

dt_gyr = 0.001
dt_transf = 0.00025
E_2 = 2.1e8

# dt_gyr = 0.02
# dt_transf = 0.02
# E_2 = 2.1e11

n_elem_1 = 10
n_elem_2 = 6
n_elem_3 = 6

# Beam and material parameters
L_1 = jnp.sqrt(5)
L_2 = 1.5
L_3 = jnp.sqrt(2)

theta_1_0 = jnp.atan2(1, 2)
theta_2_0 = 0
theta_3_0 = jnp.pi - jnp.arcsin(- (L_1 * jnp.sin(theta_1_0) + L_2*jnp.sin(theta_2_0)) / L_3)

L_0 = L_1 * jnp.cos(theta_1_0) + L_2 * jnp.cos(theta_2_0) + L_3 * jnp.cos(theta_3_0)

point_A_0 = jnp.array([0, 0])
point_B_0 = jnp.array([L_1*jnp.cos(theta_1_0), 
                       L_1*jnp.sin(theta_1_0)])

point_C_0 = jnp.array([L_1*jnp.cos(theta_1_0) + L_2*jnp.cos(theta_2_0), 
                       L_1*jnp.sin(theta_1_0) + L_2*jnp.sin(theta_2_0)])

height, width = 0.05 , 0.05 # Cross-section dimensions (m)

kappa = 1               # Shear correction factor
A = height*width        # Cross-sectional area (m²)
I = width*height**3/12  # Second moment of area (m^4)
E = 2.1e11              # Young's modulus (Pa)
nu = 0.3
G = E/(2*(1+nu))        # Shear modulus (Pa)
G_2 = E_2/(2*(1+nu))   
rho = 2710              # Density (kg/m³)

geometrical_props_1 = {
    'L': L_1,           # Length of the beam
    'A': A,             # Cross-sectional area
    'I': I,             # Second moment of area of the beam
    'k': kappa,         # Shear coefficient for rectungular cross-section
}

geometrical_props_2 = {
    'L': L_2,           # Length of the beam
    'A': A,             # Cross-sectional area
    'I': I,             # Second moment of area of the beam
    'k': kappa,         # Shear coefficient for rectungular cross-section
}

geometrical_props_3 = {
    'L': L_3,           # Length of the beam
    'A': A,             # Cross-sectional area
    'I': I,             # Second moment of area of the beam
    'k': kappa,         # Shear coefficient for rectungular cross-section
}

material_props = {
    'E': E,      # Young's modulus
    'G': G,      # Shear modulus 
    'rho': rho,  # Density
}

material_props_2 = {
    'E': E_2,      # Young's modulus
    'G': G_2,      # Shear modulus 
    'rho': rho,  # Density
}
