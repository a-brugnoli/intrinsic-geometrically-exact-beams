
dt = 0.01  # Time step
n_elements = 10  # Number of finite elements

T_end = 5

# Beam and material parameters
length = 10.0   # Total length (m)

# Stiff version
case = "stiff/"  # Options: "stiff", "soft"
G = 5*1e6          # Shear modulus (Pa)
E = 1e7          # Young's modulus (Pa)
rho = 1         # Density (kg/m³)

kappa = 1      # Shear coefficient for rectangular cross-section
A = 1         # Cross-sectional area (m^2)
I = 0.1          # Second moment of area (m^4)

# # Soft version
# case = "soft/"  # Options: "stiff", "soft"

# G = 1e5         # Shear modulus (Pa)
# E = 2*1e5          # Young's modulus (Pa)
# rho = 1         # Density (kg/m³)

# kappa = 1      # Shear coefficient for rectangular cross-section
# A = 0.5        # Cross-sectional area (m^2)
# I = 0.05       # Second moment of area (m^4)

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