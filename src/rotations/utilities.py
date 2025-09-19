import jax.numpy as jnp
import jax

@jax.jit
def rotation_matrix_2d(theta):
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], 
                      [jnp.sin(theta), jnp.cos(theta)]])


@jax.jit
def rotation_matrix_2d_batch(theta):
    """
    theta: shape (N,)
    Returns: batch of rotation matrices, shape (N, 2, 2)
    """
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.stack([
        jnp.stack([c, s], axis=-1),  # First column
        jnp.stack([-s,  c], axis=-1)  # Second column
    ], axis=-1)  # shape (N, 2, 2)


if __name__ == "__main__":

    theta_vec = jnp.array([jnp.pi / 6, jnp.pi / 4])



    rotation_batch = rotation_matrix_2d_batch(theta_vec)
    for ii in range(theta_vec.shape[0]):

        print("Rotation matrix for theta:", theta_vec[ii]*180/jnp.pi, "(batch)")
        print(rotation_batch[ii])

        print("Rotation matrix for theta:", theta_vec[ii]*180/jnp.pi, "(single)")
        print(rotation_matrix_2d(theta_vec[ii]))

