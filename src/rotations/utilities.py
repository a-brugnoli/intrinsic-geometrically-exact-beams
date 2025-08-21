import jax.numpy as jnp

def rotation_matrix_2d(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], 
                      [jnp.sin(theta), jnp.cos(theta)]])


def rotation_matrix_2d_batch(theta):
    """
    theta: shape (N,)
    Returns: batch of rotation matrices, shape (N, 2, 2)
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.stack([
        jnp.stack([c, -s], axis=-1),
        jnp.stack([s,  c], axis=-1)
    ], axis=-2)  # shape (N, 2, 2)