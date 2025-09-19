# Beam and material parameters
length = 1.0              # Total length (m)
E = 210e9            # Young's modulus (Pa)
G = 80.77e9          # Shear modulus (Pa)
rho = 7800           # Density (kg/m³)
b = 0.1             # Width (m)
h = 0.1             # Height (m)
A = b * h            # Cross-sectional area (m²)
I = b * h**3 / 12    # Second moment of area (m^4)
kappa = 5/6          # Shear correction factor

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