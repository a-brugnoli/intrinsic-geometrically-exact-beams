import jax.numpy as jnp
from jax import jit


@jit
def gauss_quadrature(n_points: int):
    points_table = jnp.array([
        [0.0, 0.0, 0.0, 0.0],
        [-1.0/jnp.sqrt(3),  1.0/jnp.sqrt(3), 0.0, 0.0],
        [-jnp.sqrt(3/5), 0.0, jnp.sqrt(3/5), 0.0],
        [-jnp.sqrt(3/7 + 2/7 * jnp.sqrt(6/5)), -jnp.sqrt(3/7 - 2/7 * jnp.sqrt(6/5)),
          jnp.sqrt(3/7 - 2/7 * jnp.sqrt(6/5)),  jnp.sqrt(3/7 + 2/7 * jnp.sqrt(6/5))]
    ])
    weights_table = jnp.array([
        [2.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [5/9, 8/9, 5/9, 0.0],
        [(18-jnp.sqrt(30))/36, (18+jnp.sqrt(30))/36, (18+jnp.sqrt(30))/36, (18-jnp.sqrt(30))/36]
    ])
    # m = jnp.array([1,2,3,4])[n_points-1]
    return points_table[n_points-1], weights_table[n_points-1]

