from src.models.free_beam import PlanarFreeBeam
from data import geometrical_props, material_props
from src.linear_algebra.eigensolver_jax import compute_eigenvalues_and_vectors
import matplotlib.pyplot as plt
import numpy as np
import jax

n_elements = 50  # Number of finite elements
beam = PlanarFreeBeam(n_elements, geometrical_props, material_props)
# Use scipy to solve the generalized eigenvalue problem
n_eigs = 10  # Number of eigenvalues to plot
omega_vec = compute_eigenvalues_and_vectors(beam.J_linear_matrix, beam.H_matrix, n_eigs=n_eigs)
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