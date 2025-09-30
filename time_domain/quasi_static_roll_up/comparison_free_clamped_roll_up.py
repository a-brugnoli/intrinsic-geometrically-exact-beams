from src.models.intrinsic_clamped_beam import IntrinsicPlanarClampedBeam
from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam

from data import geometrical_props, material_props
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from src.postprocessing.postprocess_variables import compute_positions
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
import os
from jax import config; config.update("jax_enable_x64", True)


dt = 0.2
n_elements = 8  # Number of finite elements

T_end = 1
n_steps = int(T_end/dt)
time_vec = dt * jnp.arange(n_steps + 1)

jax.jit
def spatial_torque(t):
    m_roll_up = 100*jnp.pi  # Roll-up moment
    return t*m_roll_up

# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)


clamped_beam = IntrinsicPlanarClampedBeam(n_elements, geometrical_props, material_props)
H_matrix_clamped = clamped_beam.assemble_energy_matrix()
J_linear_matrix_clamped = clamped_beam.assemble_linear_interconnection_matrix()

B_matrix_clamped_at_L = clamped_beam.assemble_control_matrix()[:, 3:]
n_lambda_clamped = B_matrix_clamped_at_L.shape[1]


jax.jit
def vector_field_clamped(state_vector, t):
    theta_vector = state_vector[:clamped_beam.n_ang_vel]
    coenergy_vector = state_vector[clamped_beam.n_ang_vel:-n_lambda_clamped]
    lagrange_multiplier = state_vector[-n_lambda_clamped:]

    vector_field_theta = coenergy_vector[clamped_beam.n_lin_vel:clamped_beam.n_vel]

    vector_field_coenergy = J_linear_matrix_clamped @ coenergy_vector \
        + clamped_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + B_matrix_clamped_at_L @ lagrange_multiplier \
        
    vector_field_constraints = - B_matrix_clamped_at_L.T @ coenergy_vector + jnp.array([0, 0, spatial_torque(t)])

    return jnp.concatenate([vector_field_theta, vector_field_coenergy, vector_field_constraints])

angle_0_clamped = jnp.zeros(clamped_beam.n_ang_vel)
coenergy_vector_0_clamped = jnp.zeros(clamped_beam.n_coenergy)
lagrange_multiplier_0_clamped = jnp.zeros(n_lambda_clamped)
state_vector_0_clamped = jnp.concatenate([angle_0_clamped, 
                                          coenergy_vector_0_clamped, lagrange_multiplier_0_clamped])

# Run simulation for clamped beam
E_matrix_clamped = jax.scipy.linalg.block_diag(jnp.eye(clamped_beam.n_ang_vel), 
                                               H_matrix_clamped, jnp.zeros((n_lambda_clamped, n_lambda_clamped)))
trajectory_clamped = simulate(state_vector_0_clamped, dt, n_steps, E_matrix_clamped, vector_field_clamped)

theta_sol_clamped = trajectory_clamped[:clamped_beam.n_ang_vel, :]
coenergy_sol_clamped = trajectory_clamped[clamped_beam.n_ang_vel:-n_lambda_clamped  , :]
lagrange_sol_clamped = trajectory_clamped[-n_lambda_clamped:, :]


lin_velocity_sol_clamped = coenergy_sol_clamped[:clamped_beam.n_lin_vel, :]
midpoints_clamped = 0.5 * (clamped_beam.x_nodes[:-1] + clamped_beam.x_nodes[1:])
initial_position_clamped = jnp.stack([midpoints_clamped,
                              jnp.zeros_like(midpoints_clamped)], axis=1)
position_sol_clamped = compute_positions(theta_sol_clamped, lin_velocity_sol_clamped, initial_position_clamped, time_vec)

idx_moment_clamped = clamped_beam.n_lin_vel + clamped_beam.n_ang_vel + clamped_beam.n_force
moment_sol_clamped = coenergy_sol_clamped[idx_moment_clamped:, :]

## Free beam
free_beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)
H_matrix_free = free_beam.assemble_energy_matrix()
J_linear_matrix_free = free_beam.assemble_linear_interconnection_matrix()

B_matrix_free_at_0 = free_beam.assemble_control_matrix()[:, :3]
n_lambda_free = B_matrix_free_at_0.shape[1]

B_torque_free_at_L = free_beam.assemble_control_matrix()[:, -1]


jax.jit
def vector_field_free(state_vector, t):
    theta_vector = state_vector[:free_beam.n_ang_vel]
    coenergy_vector = state_vector[free_beam.n_ang_vel:-n_lambda_free]
    lagrange_multiplier = state_vector[-n_lambda_free:]

    vector_field_theta = coenergy_vector[free_beam.n_lin_vel: free_beam.n_vel]
    
    vector_field_coenergy = J_linear_matrix_free @ coenergy_vector \
        + free_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + B_matrix_free_at_0 @ lagrange_multiplier \
        + B_torque_free_at_L * spatial_torque(t)

    vector_field_constraints = - B_matrix_free_at_0.T @ coenergy_vector


    return jnp.concatenate([vector_field_theta, vector_field_coenergy, vector_field_constraints])



# Compatible lagrange mutipliers are found by solving
# (B^T M^{-1} B) lambda_0 = -B^T M^{-1} (J(e_0)e_0 + f_0)
# Since e_0 = 0 and f_0 = 0 => lambda_0 = 0

angle_0_free = jnp.zeros(free_beam.n_ang_vel)
coenergy_vector_0_free = jnp.zeros(free_beam.n_coenergy)
lagrange_multiplier_0_free = jnp.zeros(n_lambda_free)
state_vector_0_free = jnp.concatenate([angle_0_free, coenergy_vector_0_free, lagrange_multiplier_0_free])

# Run simulation for free beam
E_matrix_free = jax.scipy.linalg.block_diag(jnp.eye(free_beam.n_ang_vel), \
                                            H_matrix_free, jnp.zeros((n_lambda_free, n_lambda_free)))

trajectory_free = simulate(state_vector_0_free, dt, n_steps, E_matrix_free, vector_field_free)

theta_sol_free = trajectory_free[:free_beam.n_ang_vel, :]
coenergy_sol_free = trajectory_free[free_beam.n_ang_vel:-n_lambda_free, :]
lagrange_sol_free = trajectory_free[-n_lambda_free:, :]

lin_velocity_sol_free = coenergy_sol_free[:free_beam.n_lin_vel, :]
idx_moment_free = free_beam.n_lin_vel + free_beam.n_ang_vel + free_beam.n_force
moment_sol_free = coenergy_sol_free[idx_moment_free:, :]


initial_position_free = jnp.stack([free_beam.x_nodes,
                                jnp.zeros_like(free_beam.x_nodes)], axis=1)

position_sol_free = compute_positions(theta_sol_free, lin_velocity_sol_free, initial_position_free, time_vec)

# Create and display animation
from src.postprocessing.animate import animate_overlay_beam_deformation
print("Creating animation...")
anim = animate_overlay_beam_deformation(
    time_vec, position_sol_clamped, position_sol_free,
    fps=5, interval=100,
    title="Comparison: clamped and free formulations roll-up",
    colors=('blue', 'red'),
    labels=('Clamped', 'Free'),
    lock_at_end=True,
    save_path=folder_results + "comparison_clamped_free_roll_up.gif"
)

# Show the animation
plt.show()



# time_instants = [j*1/5 for j in range(6)]
# time_indices = [j for j in range(n_steps+1) if j*dt in time_instants]

# norm_free = Normalize(vmin=jnp.min(moment_sol_free), vmax=jnp.max(moment_sol_free))
# norm_clamped = Normalize(vmin=jnp.min(moment_sol_clamped), vmax=jnp.max(moment_sol_clamped))

# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# for j in time_indices:
#     x_nodes_free = position_sol_free[0, :, j]
#     y_nodes_free = position_sol_free[1, :, j]
    
#     # Create segments between nodes
#     segments_free = jnp.array([[ [x_nodes_free[i], y_nodes_free[i]], [x_nodes_free[i+1], y_nodes_free[i+1]] ] 
#                          for i in range(x_nodes_free.size - 1)])
    
#     # Color by bending moment
#     colors_free = moment_sol_free[:, j]  # shape: (n_elements,)
    
#     lc_free = LineCollection(segments_free, array=colors_free, cmap='viridis', linewidths=2, norm=norm_free)

#     axs[0].add_collection(lc_free)

#     x_nodes_clamped = position_sol_clamped[0, :, j]
#     y_nodes_clamped = position_sol_clamped[1, :, j]
    
#     # Create segments between nodes
#     segments_clamped = jnp.array([[[x_nodes_clamped[i], y_nodes_clamped[i]], 
#                                    [x_nodes_clamped[i+1], y_nodes_clamped[i+1]] ] 
#                                    for i in range(x_nodes_clamped.size - 1)])
    
#     # Color by bending moment
#     colors_clamped = moment_sol_clamped[:, j]  # shape: (n_elements,)
    
#     lc_clamped = LineCollection(segments_clamped, array=colors_clamped, cmap='viridis', linewidths=2, norm=norm_clamped)
    
#     axs[1].add_collection(lc_clamped)

# axs[0].set_xlabel('x')
# axs[0].set_ylabel('y')
# # ax.set_title('Beam configurations over time')
# axs[0].grid(True)

# # Create a colorbar axis that matches the y-axis height
# divider_free = make_axes_locatable(axs[0])
# cax = divider_free.append_axes("right", size="5%", pad=0.1)  # 5% width, small padding
# cbar_free = fig.colorbar(lc_free, cax=cax)
# cbar_free.set_label('$M$ Free')

# axs[0].set_xlim(-2.5, 10.5)
# axs[0].set_ylim(-0.5, 8)
# axs[0].set_aspect("equal", adjustable="box")

# axs[1].set_xlabel('x')
# axs[1].set_ylabel('y')
# # ax.set_title('Beam configurations over time')
# axs[1].grid(True)

# # Create a colorbar axis that matches the y-axis height
# divider_clamped = make_axes_locatable(axs[1])
# cax = divider_clamped.append_axes("right", size="5%", pad=0.1)  # 5% width, small padding
# cbar_clamped = fig.colorbar(lc_clamped, cax=cax)
# cbar_clamped.set_label('$M$ Clamped')

# axs[1].set_xlim(-2.5, 10.5)
# axs[1].set_ylim(-0.5, 8)
# axs[1].set_aspect("equal", adjustable="box")

# fig.tight_layout()
# fig.savefig(folder_results + "configurations.pdf", format="pdf")


# plt.show()