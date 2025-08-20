import jax.numpy as jnp


def rotation_matrix_2d(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], 
                      [jnp.sin(theta), jnp.cos(theta)]])