# Beam and material parameters
beam_length = 3.0443   # Total length (m)
dt = 0.1 # Time step (s)
T_end = 5  # End time (s)

kappa = 1       # Shear correction factor
A = 1           # Cross-sectional area (m²)
I = 1          # Second moment of area (m^4)
E = 1e10          # Young's modulus (Pa)
G = 1e10         # Shear modulus (Pa)
rho = 0         # Density (kg/m³)

geometrical_props = {
    'L': beam_length,   # Length of the beam
    'A': A,  # Cross-sectional area
    'I': I,  # Second moment of area of the beam
    'k': kappa,   # Shear coefficient for rectungular cross-section
}

material_props = {
    'E': E,  # Young's modulus
    'G': G,  # Shear modulus 
    'rho': rho,  # Density
}

g = 9.8  # Gravitational acceleration (m/s²)
m_mass = 10.0  # Mass at the free end (kg)