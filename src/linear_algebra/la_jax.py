import jax.numpy as jnp
import jax
from jax.scipy.linalg import solve
from functools import partial


@jax.jit
def skew_block_matrix(D):
    """
    Create a skew-symmetric block matrix from a given matrix D.

    Args:
        D (jnp.ndarray): Input matrix of shape (m, n).

    Returns:
        jnp.ndarray: Skew-symmetric block matrix of shape (m+n, m+n).
        J = [ 0   D ]
            [ -D^T 0 ]
    """
    zeros_top_left = jnp.zeros((D.shape[0], D.shape[0]))
    zeros_bottom_right = jnp.zeros((D.shape[1], D.shape[1]))
    top = jnp.concatenate([zeros_top_left, D], axis=1)
    bottom = jnp.concatenate([-D.T, zeros_bottom_right], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


@partial(jax.jit, static_argnums=0)
def forward_difference_jacobian(fz_func, z, eps=None):
    """
    Compute the Jacobian of fz_func at z using forward differences.
    
    Parameters
    ----------
    fz_func : Callable[[Array], Array]
        The function to differentiate.
    z : Array
        Point at which to compute the Jacobian. Shape (n,).
    eps : float, optional
        Step size for finite differences. If None, uses machine epsilon scaled appropriately.
    
    Returns
    -------
    J : Array
        Jacobian matrix with shape (m, n) where m = len(fz_func(z)), n = len(z).
    """
    # Use optimal step size based on precision if not specified
    if eps is None:
        eps = jnp.sqrt(jnp.finfo(z.dtype).eps)
    
    f0 = fz_func(z)
    n = z.shape[0]
    
    # Vectorized computation of all columns
    def col_func(i):
        z_perturbed = z.at[i].add(eps)
        f_perturbed = fz_func(z_perturbed)
        return (f_perturbed - f0) / eps
    
    # Compute all columns of the Jacobian
    J = jax.vmap(col_func)(jnp.arange(n))
    
    # Transpose to get (m, n) shape
    return J.T


@partial(jax.jit, static_argnums=0)
def central_difference_jacobian(fz_func, z, eps=None):
    """
    Compute the Jacobian of fz_func at z using central finite differences.
    
    Parameters
    ----------
    fz_func : Callable[[Array], Array]
        The function to differentiate.
    z : Array
        Point at which to compute the Jacobian. Shape (n,).
    eps : float, optional
        Step size for finite differences. If None, uses machine epsilon scaled appropriately.
    
    Returns
    -------
    J : Array
        Jacobian matrix with shape (m, n) where m = len(fz_func(z)), n = len(z).
    """
    # Use optimal step size based on precision if not specified
    if eps is None:
        # Optimal step size for central differences is the cube root of machine epsilon
        eps = jnp.cbrt(jnp.finfo(z.dtype).eps)
    
    n = z.shape[0]
    
    # Vectorized computation of all columns
    def col_func(i):
        z_plus = z.at[i].add(eps)
        z_minus = z.at[i].add(-eps)
        
        f_plus = fz_func(z_plus)
        f_minus = fz_func(z_minus)
        
        return (f_plus - f_minus) / (2.0 * eps)
    
    # Compute all columns of the Jacobian
    J = jax.vmap(col_func)(jnp.arange(n))
    
    # Transpose to get (m, n) shape
    return J.T


@partial(jax.jit, static_argnums=0)
def newton_solver_fd(fz_func, z0, tol=1e-12, max_iter=20, eps=None):
    """
    Solves a nonlinear equation using Newton's method with finite differences.
    
    Solves for a root of the function `fz_func(z) = 0` starting from
    an initial guess `z0`, using Newton-Raphson iteration with finite
    difference approximation for Jacobian computation.
    
    Parameters
    ----------
    fz_func : Callable[[Array], Array]
        The function whose root is to be found. Should accept a vector `z` and return a vector.
    z0 : Array
        Initial guess for the root (should be float64 for maximum precision).
    tol : float, optional
        Tolerance for stopping criterion based on the L2 norm of the residual. 
    max_iter : int, optional
        Maximum number of iterations allowed. 
    eps : float, optional
        Step size for finite differences. If None, uses sqrt(machine epsilon).
    
    Returns
    -------
    z_final : Array
        Approximate root of the equation `fz_func(z) = 0`.
    final_residual : float
        L2 norm of the residual at the final iteration.
    
    Notes
    -----
    - Requires JAX x64 mode enabled: `jax.config.update("jax_enable_x64", True)`
    - Uses finite differences to approximate the Jacobian.
    - Optimal step size is sqrt(eps_machine) ≈ 1.5e-8 for float64.
    - Solves the linear system `J dz = -fz_func(z)` using dense linear algebra.
    - Uses `jax.lax.while_loop` to keep the computation JIT-compatible.
    """
    def body_fn(state):
        z, i = state
        F = fz_func(z)
        J = central_difference_jacobian(fz_func, z, eps)  # or forward_difference_jacobian
        delta = solve(J, -F, assume_a='gen')  # J \ -F
        z_new = z + delta
        return (z_new, i + 1)

    def cond_fn(state):
        z, i = state
        return (jnp.linalg.norm(fz_func(z)) > tol) & (i < max_iter)

    z_final, iter_count = jax.lax.while_loop(cond_fn, body_fn, (z0, 0))

    # Compute residual norm after iteration
    final_residual = jnp.linalg.norm(fz_func(z_final))
    
    # JAX-compatible print
    jax.debug.print(
        "Newton terminated after {i} iterations: (residual = {res:.3e})",
        i=iter_count, res=final_residual
    )

    return z_final, final_residual, iter_count


@partial(jax.jit, static_argnums=0)
def newton_solver_jax(fz_func, z0, tol=1e-12, max_iter=20):
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
        Tolerance for stopping criterion based on the L2 norm of the residual.
    max_iter : int, optional
        Maximum number of iterations allowed. 
    gradient_method : str, optional
        Method to compute the Jacobian. Options are "autodiff", "forward_difference", "central_difference". 
        Default is "autodiff".

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

    z_final, iter_count = jax.lax.while_loop(cond_fn, body_fn, (z0, 0))

    # Compute residual norm after iteration
    final_residual = jnp.linalg.norm(fz_func(z_final))
    
    # JAX-compatible print
    jax.debug.print(
        "Newton terminated after {i} iterations: (residual = {res:.3e})",
        i=iter_count, res=final_residual
    )

    return z_final, final_residual, iter_count

if __name__ == "__main__":
    # Example usage
    D = jnp.array([[1., 2.], [3., 4.]])
    S = skew_block_matrix(D)
    print(S)
