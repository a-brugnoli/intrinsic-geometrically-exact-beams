# Beam and material parameters
length = 10.0   # Total length (m)

kappa = 1       # Shear correction factor
A = 1           # Cross-sectional area (m²)
I = 10          # Second moment of area (m^4)
E = 10          # Young's modulus (Pa)
G = 1e4         # Shear modulus (Pa)
rho = 1         # Density (kg/m³)

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