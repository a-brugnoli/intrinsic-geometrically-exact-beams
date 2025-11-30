import jax.numpy as jnp
import jax
from src.rotations.utilities import rotation_matrix_2d_batch
from src.models.intrinsic_generic_beam import IntrinsicBeam
from functools import partial


def implicit_midpoint_integrate(v, dt, x0=None):
    """
    Integrate xdot = v using the implicit midpoint method.
    
    The implicit midpoint rule: x_{n+1} = x_n + dt * v_{n+1/2}
    where v_{n+1/2} is approximated as (v_n + v_{n+1}) / 2
    
    Args:
        v: Velocity vector of shape (N, n_times)
        dt: Time step (scalar or array of shape (n_times-1,))
        x0: Initial condition of shape (N,). If None, starts at origin.
    
    Returns:
        x: Position vector of shape (N, n_times)
    """

    N, n_times = v.shape[0], v.shape[1]
    
    # Initialize position array
    if x0 is None:
        x0 = jnp.zeros(N)
    
    x = jnp.zeros((N, n_times))
    x = x.at[:, 0].set(x0)
    
    # Integration loop using implicit midpoint
    def step_fn(x_prev, i):
        # Implicit midpoint: x_{n+1} = x_n + dt * (v_n + v_{n+1}) / 2
        v_mid = (v[:, i] + v[:, i + 1]) / 2
        
        # Handle dt as scalar or array
        dt_step = dt if jnp.isscalar(dt) or dt.ndim == 0 else dt[i]
        
        x_next = x_prev + dt_step * v_mid
        return x_next, x_next
    
    # Scan over time steps
    _, x_vals = jax.lax.scan(step_fn, x0, jnp.arange(n_times - 1))
    
    # Combine initial condition with computed values
    x = x.at[:, 1:].set(x_vals.T)
    
    return x