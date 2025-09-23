# Beam and material parameters
length = 10.0   # Total length (m)

G = 1e4          # Shear modulus (Pa)
E = 1e4          # Young's modulus (Pa)
rho = 0         # Density (kg/m³)

kappa = 1      # Shear coefficient for rectangular cross-section
A = 1         # Cross-sectional area (m^2)
I = 500/E          # Second moment of area (m^4)

geometrical_props = {
    'L': length,   # Length of the beam
    'A': A,  # Cross-sectional area
    'I': I,  # Second moment of area of the beam
    'k': kappa,   # Shear coefficient for rectungular cross-section
}

material_props = {
    'E': E,  # Young's modulus
    'G': G,  # Shear modulus 
    'rho': rho,  # Density
}