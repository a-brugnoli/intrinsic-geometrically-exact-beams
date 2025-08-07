from src.linear_algebra.la_jax import newton_solver
import jax.numpy as jnp
import jax

def implicit_midpoint_step_dae(xn, tn, h, E, f: callable):
    def G(z):
        return jnp.dot(E, z-xn) - h * f((z + xn)/2, tn + h/2)

    z0 = xn  # initial guess (can be changed)
    return newton_solver(G, z0)


# Simulation loop
def simulate(x0, dt, steps, E, f: callable):
    def step_fn(xn, i):
        xnp1 = implicit_midpoint_step_dae(xn, i*dt, dt, E, f)
        return xnp1, xnp1

    _, xs = jax.lax.scan(step_fn, x0, jnp.arange(steps))
    return xs