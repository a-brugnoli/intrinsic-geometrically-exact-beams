from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from src.models.classical_beam import ClassicalPlanarBeam
from data import geometrical_props, material_props
from src.linear_algebra.eigensolver_jax import extract_positive_imaginary_eigenvalues, extract_real_eigenvalues
import matplotlib.pyplot as plt
import numpy as np
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()

n_elements = 10  # Number of finite elements
intrinsic_beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)
J_linear_matrix = intrinsic_beam.assemble_linear_interconnection_matrix()
H_matrix = intrinsic_beam.assemble_energy_matrix()

classical_beam = ClassicalPlanarBeam(n_elements, geometrical_props, material_props)
M_matrix, K_matrix = classical_beam.assemble_matrices()

n_eigs = 10  # Number of eigenvalues to plot
omega_vec_intrinsic = extract_positive_imaginary_eigenvalues(J_linear_matrix, H_matrix, n_eigs=n_eigs)

omega_vec_squared_classical = extract_real_eigenvalues(K_matrix, M_matrix, n_eigs=n_eigs)
omega_vec_classical = np.sqrt(omega_vec_squared_classical)
# Display
for i  in range(n_eigs):
    print(f"Mode {i}: classical = {omega_vec_classical[i]:.4f}, intrinsic = {omega_vec_intrinsic[i]:.4f}")

# Optional: plot first few frequencies
plt.plot(np.arange(1, n_eigs+1), omega_vec_classical[:n_eigs], 'o', label='Classical')
plt.plot(np.arange(1, n_eigs+1), omega_vec_intrinsic[:n_eigs], '+', label='Intrinsic')
plt.xlabel('Mode number')
plt.ylabel('$\omega$ (Rad/s)')
plt.title('Free-Free Timoshenko Beam')
plt.grid(True)
plt.show()