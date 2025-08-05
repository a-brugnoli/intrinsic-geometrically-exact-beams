import jax
import jax.numpy as jnp
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()

def compute_eigenvalues_and_vectors(J_mixed, H_mixed, n_eigs: int=5):
    eigenvalues, eigenvectors = jnp.linalg.eig(jnp.linalg.solve(H_mixed, J_mixed))
    # Process eigenvalues (find positive imaginary parts)
    pos_imag_mask = jnp.imag(eigenvalues) >= 0
    pos_imag_eigenvalues = jnp.imag(eigenvalues)
    pos_imag_eigenvalues = jnp.where(pos_imag_mask, pos_imag_eigenvalues, jnp.inf)

    # Get smallest eigenvalues
    indexes_eigs = jnp.argsort(pos_imag_eigenvalues)
    n_eigs = jnp.asarray(n_eigs, dtype=jnp.int32)
    indexes_smallest_eigs = jax.lax.slice(indexes_eigs, (0,), (n_eigs,)) 
    omega_vec = pos_imag_eigenvalues[indexes_smallest_eigs]

    return omega_vec
