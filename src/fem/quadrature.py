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


