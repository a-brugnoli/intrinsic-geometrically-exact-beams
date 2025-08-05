import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from data import *

def timoshenko_beam_matrices(E, G, rho, A, I, kappa, h_e):
    """
    Calculate elementary stiffness and mass matrices for Timoshenko beam element
    
    Parameters:
    -----------
    E : float
        Young's modulus (Pa)
    G : float
        Shear modulus (Pa)
    rho : float
        Density (kg/m³)
    A : float
        Cross-sectional area (m²)
    I : float
        Second moment of area (m⁴)
    kappa : float
        Shear correction factor (dimensionless, typically 5/6 for rectangular)
    h_e : float
        Element length (m)
    
    Returns:
    --------
    K : numpy.ndarray
        4x4 elementary stiffness matrix
    M : numpy.ndarray
        4x4 elementary mass matrix
    """
    
    # Calculate stiffness matrix [K]
    # DOFs: [w1, θ1, w2, θ2]
    
    # Shear stiffness parameter
    GAk = kappa * G * A
    
    # Coefficients for stiffness matrix
    alpha1 = h_e**2/4 + E*I/(GAk)
    alpha2 = h_e**2/4 - E*I/(GAk)
    
    # Stiffness matrix
    K = np.array([
        [GAk/h_e,    -GAk/2,     -GAk/h_e,    -GAk/2],
        [-GAk/2,   GAk*alpha1/h_e, GAk/2,    GAk*alpha2/h_e],
        [-GAk/h_e,   GAk/2,      GAk/h_e,     GAk/2],
        [-GAk/2,   GAk*alpha2/h_e, GAk/2,    GAk*alpha1/h_e]
    ])
    
    # Calculate mass matrix [M]
    # Translational inertia contribution
    M_trans = rho * A * h_e / 420 * np.array([
        [156,    22*h_e,   54,     -13*h_e],
        [22*h_e,   4*h_e**2, 13*h_e,   -3*h_e**2],
        [54,     13*h_e,   156,    -22*h_e],
        [-13*h_e,  -3*h_e**2, -22*h_e,  4*h_e**2]
    ])
    
    # Rotational inertia contribution
    M_rot = rho * I / (30*h_e) * np.array([
        [36,     3*h_e,    -36,    3*h_e],
        [3*h_e,    4*h_e**2, -3*h_e,   -h_e**2],
        [-36,    -3*h_e,   36,     -3*h_e],
        [3*h_e,    -h_e**2,  -3*h_e,   4*h_e**2]
    ])
    
    # Total mass matrix
    M = M_trans + M_rot
    
    return K, M

# FEM discretization
n_elements = 50                         # Number of elements
h_e = length / n_elements                     # Element length
ndof = 2 * (n_elements + 1)             # 2 DOFs per node (w, phi)

# Initialize global mass and stiffness matrices
K = np.zeros((ndof, ndof))
M = np.zeros((ndof, ndof))

# Assembly
for e in range(n_elements):
    dof = np.array([2*e, 2*e+1, 2*e+2, 2*e+3])
    Ke, Me = timoshenko_beam_matrices(E, G, rho, A, I, kappa, h_e)
    for i in range(4):
        for j in range(4):
            K[dof[i], dof[j]] += Ke[i, j]
            M[dof[i], dof[j]] += Me[i, j]


# Solve generalized eigenvalue problem
eigvals, _ = la.eig(K, M)

# Filter real and positive eigenvalues
eigvals = np.real(eigvals)
eigvals.sort()

# Convert to natural frequencies (Rad/s)
omega_vec = np.sqrt(eigvals) 

n_eigs = 10  # Number of eigenvalues to plot
# Display
for i, om in enumerate(omega_vec[:n_eigs], 1):
    print(f"Mode {i}: {om:.4f} Hz")

# Optional: plot first few frequencies
plt.plot(np.arange(1, n_eigs+1), omega_vec[:n_eigs], 'o-')
plt.xlabel('Mode number')
plt.ylabel('omega (Rad/s)')
plt.title('Natural Frequencies of Free-Free Timoshenko Beam')
plt.grid(True)
plt.show()
