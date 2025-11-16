from ast import For
from src.linear_algebra.la_jax import newton_solver_jax, newton_solver_fd
import jax.numpy as jnp
import jax
from functools import partial


@partial(jax.jit, static_argnums=-1)
def forward_euler_step_dae(xn, tn, h, E, f: callable):
    """
    Forward Euler step for DAE system.
    Solves: E * (x_{n+1} - x_n) = h * f(x_n, t_n)
    """
    return xn + h * jnp.linalg.solve(E, f(xn, tn))


@partial(jax.jit, static_argnums=-1)
def implicit_midpoint_step_dae(xn, tn, h, E, f: callable):

    def G(z):
        return jnp.dot(E, z-xn) - h * f((z + xn)/2, tn + h/2)

    z0 = xn  # initial guess 
    
    tol = 1e-8

    # xn1, _ = newton_solver_fd(G, z0, tol=tol)
    xn1, _ = newton_solver_jax(G, z0, tol=tol)

    return xn1



@partial(jax.jit, static_argnums=[2, 4, 5])
def simulate(x0, dt, num_steps, E, f: callable, method="midpoint"):
    def step_fn(xn, i):
        jax.debug.print(
        "Progress time integration: {i} / {num_steps}", i=i, num_steps=num_steps
    )
        if method=="midpoint":
            xnp1 = implicit_midpoint_step_dae(xn, i*dt, dt, E, f)
        elif method=="Euler explicit":
            xnp1 = forward_euler_step_dae(xn, i*dt, dt, E, f)
        return xnp1, xnp1

    _, xs = jax.lax.scan(step_fn, x0, jnp.arange(num_steps))

    return jnp.concatenate([x0.reshape(-1, 1), xs.T], axis=1)  


@partial(jax.jit, static_argnums=[2, 5])
def simulate_linear(x0, dt, num_steps, E, J, method="midpoint"):
    def step_fn(xn, i):
        jax.debug.print(
        "Progress time integration: {i} / {num_steps}", i=i, num_steps=num_steps
    )
        if method=="midpoint":
            xnp1 = jnp.linalg.solve(E - dt/2 * J, (E + dt/2 * J) @ xn)
            return xnp1, xnp1 
        else:
            raise NotImplementedError("Only implicit midpont for linear integrator")  

    _, xs = jax.lax.scan(step_fn, x0, jnp.arange(num_steps))

    return jnp.concatenate([x0.reshape(-1, 1), xs.T], axis=1)  


def compute_lambda0(H, dae_fn, x0, t0=0, rcond=1e-8):
    """
    Compute λ₀ for the index-2 DAE:
        H ẋ = f(x) + G(x) λ
        G(x)^T x = 0
    using one function:
        dae_fn(x, t) -> (f(x, t), G(x, t), Φ(x, t))

    PARAMETERS
    ----------
    H : (n×n) constant matrix
    dae_fn : function(x) -> (f, G, Φ)
        f : (n,)
        G : (n×m)
    x0 : initial state vector,
    t0: initial time
    rcond : tolerance for least squares fallback

    RETURNS
    -------
    lambda0 : (m,)
    info     : dict with details
    """

    # Evaluate f, G, Φ
    f0, G0 = dae_fn(x0, t0)          # f0:(n,), G0:(n×m), phi0:(m,)

    # Jacobian of constraints wrt state
    Phi = lambda x: dae_fn(x, t0)[1].T @ x
    Dphi = jax.jacobian(Phi)(x0)       # (m×n)

    # H^{-1} f via linear solve
    v_f = jnp.linalg.solve(H, f0)      # (n,)

    # H^{-1} G via linear solve for all columns
    v_G = jnp.linalg.solve(H, G0)      # (n×m)

    # Build linear system A λ = b
    A = Dphi @ v_G                     # (m×m)
    b = -Dphi @ v_f                    # (m,)

    # Solve A λ = b
    if A.shape[0] == A.shape[1]:
        # square system: try solve, fallback to lstsq
        try:
            lambda0 = jnp.linalg.solve(A, b)
        except Exception:
            lambda0, *_ = jnp.linalg.lstsq(A, b, rcond=rcond)
    else:
        # rectangular: use least squares
        lambda0, *_ = jnp.linalg.lstsq(A, b, rcond=rcond)

    return lambda0