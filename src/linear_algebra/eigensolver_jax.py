import jax
import jax.numpy as jnp


def extract_positive_imaginary_eigenvalues(A_mat, B_mat, n_eigs: int=5):
    eigenvalues, eigenvectors = jnp.linalg.eig(jnp.linalg.solve(B_mat, A_mat))
    # Process eigenvalues (find positive imaginary parts)
    pos_imag_mask = jnp.imag(eigenvalues) >= 0
    pos_imag_eigenvalues = jnp.imag(eigenvalues)
    pos_imag_eigenvalues = jnp.where(pos_imag_mask, pos_imag_eigenvalues, jnp.inf)

    # Get smallest eigenvalues
    indexes_eigs = jnp.argsort(pos_imag_eigenvalues)
    indexes_smallest_eigs = jax.lax.slice(indexes_eigs, (0,), (n_eigs,)) 
    omega_vec = pos_imag_eigenvalues[indexes_smallest_eigs]

    return omega_vec


def extract_real_eigenvalues(A_mat, B_mat, n_eigs: int=5):
    eigenvalues, eigenvectors = jnp.linalg.eig(jnp.linalg.solve(B_mat, A_mat))
    
    # Get smallest eigenvalues
    indexes_eigs = jnp.argsort(jnp.real(eigenvalues))
    indexes_smallest_eigs = jax.lax.slice(indexes_eigs, (0,), (n_eigs,)) 
    squared_omega_vec = eigenvalues[indexes_smallest_eigs]

    return squared_omega_vec
