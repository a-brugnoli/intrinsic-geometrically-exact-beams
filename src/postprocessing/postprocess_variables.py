import jax.numpy as jnp
import jax
from src.rotations.utilities import rotation_matrix_2d_batch

@jax.jit
def compute_beam_positions(theta_sol: jnp.ndarray, lin_velocity_sol: jnp.ndarray, 
                        initial_position: jnp.ndarray, time_sol: jnp.ndarray):
    
    dt_vec = time_sol[1:] - time_sol[:-1]
    n_steps = time_sol.shape[0] - 1

    n_lin_vel = lin_velocity_sol.shape[0]
    n_lin_vel_half = int(n_lin_vel / 2)

    if theta_sol.shape[0] == n_lin_vel_half+1:
        # jax.debug.print("Postprocessing Theta is nodal, velocity is DG. Converting theta to midpoint")
        theta_sol = (theta_sol[1:, :] + theta_sol[:-1, :])/2
    elif theta_sol.shape[0] == n_lin_vel_half:
        pass
    
    # Integrate position vector dynamics using the trapezoidal rule
    reshaped_velocity_sol = lin_velocity_sol.reshape(2, n_lin_vel_half, -1)

    position_sol = jnp.zeros_like(reshaped_velocity_sol)

    position_sol = position_sol.at[0, :, 0].set(initial_position[:, 0])
    position_sol = position_sol.at[1, :, 0].set(initial_position[:, 1])


    def integration_position(position_prev, ii):

        theta_mid = (theta_sol[:, ii] + theta_sol[:, ii+1]) / 2        # shape (N,)
        velocity_mid = (reshaped_velocity_sol[:, :, ii] + reshaped_velocity_sol[:, :, ii+1]) / 2  # shape (2, N)

        # Compute rotation matrices for all nodes: shape (N, 2, 2)
        R_batch = rotation_matrix_2d_batch(theta_mid)

        # Apply rotation to each velocity vector
        spatial_velocity_midpoint = jnp.einsum('nij,jn->in', R_batch, velocity_mid)  # result: (2, N)

        position_next = position_prev + dt_vec[ii] * spatial_velocity_midpoint
        return position_next, position_next

    # Scan over time steps
    # Initialize
    position_sol_init = position_sol[:, :, 0]  # shape (2, N)

    _, positions = jax.lax.scan(integration_position, position_sol_init, jnp.arange(n_steps))

    # Concatenate initial position to match original shape (2, N, n_steps+1)
    position_sol = jnp.concatenate([position_sol_init[None, :, :], positions], axis=0).transpose(1, 2, 0)

    return position_sol



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