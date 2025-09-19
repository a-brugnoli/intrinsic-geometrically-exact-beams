import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=[-1])
def extract_positive_imaginary_eigenvalues(A_mat, B_mat, n_eigs):
    eigenvalues, eigenvectors = jnp.linalg.eig(jnp.linalg.solve(B_mat, A_mat))
    # Process eigenvalues (find positive imaginary parts)
    imag_eigenvalues = jnp.imag(eigenvalues)
    pos_imag_mask = imag_eigenvalues/jnp.max(imag_eigenvalues) >= -1e-7
    
    pos_imag_eigenvalues = jnp.abs(jnp.where(pos_imag_mask, imag_eigenvalues, jnp.inf))

    # Get smallest eigenvalues
    indexes_eigs = jnp.argsort(pos_imag_eigenvalues)
    indexes_smallest_eigs = jax.lax.slice(indexes_eigs, (0,), (n_eigs,)) 
    omega_vec = pos_imag_eigenvalues[indexes_smallest_eigs]

    return omega_vec

@partial(jax.jit, static_argnums=[-1])
def extract_real_eigenvalues(A_mat, B_mat, n_eigs):
    eigenvalues, eigenvectors = jnp.linalg.eig(jnp.linalg.solve(B_mat, A_mat))    
    real_eigenvalues = jnp.real(eigenvalues)
    # Get smallest eigenvalues
    indexes_eigs = jnp.argsort(real_eigenvalues)
    indexes_smallest_eigs = jax.lax.slice(indexes_eigs, (0,), (n_eigs,)) 
    squared_omega_vec = real_eigenvalues[indexes_smallest_eigs]

    return squared_omega_vec
