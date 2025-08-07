import jax.numpy as jnp
import jax
from jax.scipy.linalg import solve
from functools import partial

@jax.jit
def skew_block_matrix(D):
    zeros_top_left = jnp.zeros((D.shape[0], D.shape[0]))
    zeros_bottom_right = jnp.zeros((D.shape[1], D.shape[1]))
    top = jnp.concatenate([zeros_top_left, D], axis=1)
    bottom = jnp.concatenate([-D.T, zeros_bottom_right], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


@partial(jax.jit, static_argnums=0)
def newton_solver(fz_func, z0, tol=1e-8, max_iter=20):
    """
    Solves a nonlinear equation using Newton's method in JAX.

    Solves for a root of the function `fz_func(z) = 0` starting from
    an initial guess `z0`, using Newton-Raphson iteration with automatic
    differentiation for Jacobian computation.

    Parameters
    ----------
    fz_func : Callable[[Array], Array]
        The function whose root is to be found. Should accept a vector `z` and return a vector.
    z0 : Array
        Initial guess for the root.
    tol : float, optional
        Tolerance for stopping criterion based on the L2 norm of the residual. Default is 1e-8.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 20.

    Returns
    -------
    z_final : Array
        Approximate root of the equation `fz_func(z) = 0`.

    Notes
    -----
    - Uses `jax.jacfwd` to compute the Jacobian of `fz_func` at each iteration.
    - Solves the linear system `J dz = -fz_func(z)` using dense linear algebra (`jax.scipy.linalg.solve`).
    - Uses `jax.lax.while_loop` to keep the computation JIT-compatible.
    """

    def body_fn(state):
        z, i = state
        F = fz_func(z)
        J = jax.jacfwd(fz_func)(z)
        delta = solve(J, -F, assume_a='gen')  # J \ -F
        z_new = z + delta
        return (z_new, i + 1)

    def cond_fn(state):
        z, i = state
        return (jnp.linalg.norm(fz_func(z)) > tol) & (i < max_iter)

    z_final, _ = jax.lax.while_loop(cond_fn, body_fn, (z0, 0))
    return z_final

if __name__ == "__main__":
    # Example usage
    D = jnp.array([[1., 2.], [3., 4.]])
    S = skew_block_matrix(D)
    print(S)
