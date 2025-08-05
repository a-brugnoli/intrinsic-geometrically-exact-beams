import jax.numpy as jnp
from jax import jit

@jit
def _gauss_quadrature_one_point():
    """
    Returns the 2-point Gauss quadrature points and weights.
    
    Returns:
        points: JAX array of Gauss points
        weights: JAX array of Gauss weights
    """
    points = jnp.array([0.0])
    weights = jnp.array([2.0])  # Weight for the midpoint in [-1, 1]
    return points, weights


@jit
def _gauss_quadrature_two_point():
    """
    Returns the 2-point Gauss quadrature points and weights.
    
    Returns:
        points: JAX array of Gauss points
        weights: JAX array of Gauss weights
    """
    points = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
    weights = jnp.array([1.0, 1.0])
    return points, weights


def eval_interp_at_gauss_node(dofs, xi_q):
    """
    Evaluates the piecewise linear or constant interpolant at the two Gauss points.
    
    Args:
        dofs: JAX array of degrees of freedom (shape (2,) for P1 or (1,) for P0)
        xi_q: JAX array of Gauss quadrature points (shape (1,) or (2,))
    Returns:
        JAX array of evaluations at the two Gauss points
    """
    if dofs.ndim == 1 and dofs.shape[0] not in (1, 2):
        raise ValueError("dofs must be of shape (2,) for P1 or (1,) for P0")
    return jnp.where(
        dofs.shape[0] == 2,
        dofs[0] * 0.5 * (1 - xi_q) + dofs[1] * 0.5 * (1 + xi_q),
        jnp.full_like(xi_q, dofs[0])
    )