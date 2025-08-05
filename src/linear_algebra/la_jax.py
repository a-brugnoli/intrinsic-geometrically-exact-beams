import jax.numpy as jnp

def skew_block_matrix(D):
    zeros_top_left = jnp.zeros((D.shape[0], D.shape[0]))
    zeros_bottom_right = jnp.zeros((D.shape[1], D.shape[1]))
    top = jnp.concatenate([zeros_top_left, D], axis=1)
    bottom = jnp.concatenate([-D.T, zeros_bottom_right], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


if __name__ == "__main__":
    # Example usage
    D = jnp.array([[1., 2.], [3., 4.]])
    S = skew_block_matrix(D)
    print(S)
