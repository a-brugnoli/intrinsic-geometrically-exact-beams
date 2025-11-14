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
    ## Explicit Euler predictor
    # z0 = xn + h * jnp.linalg.solve(E, f(xn, tn))

    tol = 1e-8

    xn1, _ = newton_solver_fd(G, z0, tol=tol)
    # xn1, _ = newton_solver_jax(G, z0, tol=tol)

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