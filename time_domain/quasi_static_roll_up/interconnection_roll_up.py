from src.models.intrinsic_cantilever_beam import IntrinsicPlanarCantileverBeam
from data import geometrical_props, material_props
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np  
from src.postprocessing.postprocess_variables import compute_positions
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()

import os
from jax import config; config.update("jax_enable_x64", True)



dt = 0.2
n_elements = 16  # Number of finite elements

T_end = 1
n_steps = int(T_end/dt)
time_vec = dt * jnp.arange(n_steps + 1)



# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

cantilever_beam = IntrinsicPlanarCantileverBeam(n_elements, geometrical_props, material_props)
clamped_half = cantilever_beam.clamped_beam
free_half = cantilever_beam.free_beam

H_matrix = cantilever_beam.assemble_energy_matrix()
J_linear_matrix = cantilever_beam.assemble_linear_interconnection_matrix()
E_matrix = jax.scipy.linalg.block_diag(jnp.eye(cantilever_beam.n_ang_vel), H_matrix)

B_torque_at_L = cantilever_beam.assemble_control_matrix()[:, 5]


jax.jit
def spatial_torque(t):
    m_roll_up = 100*jnp.pi  # Roll-up moment
    return t*m_roll_up



jax.jit
def vector_field(state_vector, t):
    # theta_vector_clamped = state_vector[:clamped_half.n_ang_vel]
    # theta_vector_free = state_vector[clamped_half.n_ang_vel: cantilever_beam.n_ang_vel]

    # theta_at_L_clamped = theta_vector_clamped[-1]
    # theta_at_0_free = theta_vector_free[0]

    coenergy_vector = state_vector[cantilever_beam.n_ang_vel:]
    coenergy_vector_clamped = coenergy_vector[:clamped_half.n_coenergy]
    coenergy_vector_free = coenergy_vector[clamped_half.n_coenergy:]

    vector_field_theta_clamped = coenergy_vector_clamped[clamped_half.n_lin_vel: clamped_half.n_vel]

    vector_field_theta_free = coenergy_vector_free[free_half.n_lin_vel:free_half.n_vel]

    vector_field_theta = jnp.concatenate([vector_field_theta_clamped, vector_field_theta_free])

    vector_field_coenergy = J_linear_matrix @ coenergy_vector \
        + cantilever_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + B_torque_at_L * spatial_torque(t) 


    return jnp.concatenate([vector_field_theta, vector_field_coenergy])



angle_0 = jnp.zeros(cantilever_beam.n_ang_vel)
coenergy_vector_0 = jnp.zeros(cantilever_beam.n_coenergy)
state_vector_0 = jnp.concatenate([angle_0, coenergy_vector_0])

# Run simulation
trajectory = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field)


theta_sol = trajectory[:cantilever_beam.n_ang_vel, :]
theta_sol_clamped = theta_sol[:clamped_half.n_ang_vel, :]
theta_sol_free = theta_sol[clamped_half.n_ang_vel:, :]

coenergy_sol = trajectory[cantilever_beam.n_ang_vel:, :]

coenergy_sol_clamped = coenergy_sol[:clamped_half.n_coenergy, :]
coenergy_sol_free = coenergy_sol[clamped_half.n_coenergy:, :]


lin_velocity_sol_clamped = coenergy_sol_clamped[:clamped_half.n_lin_vel, :]
idx_moment_clamped = clamped_half.n_lin_vel + clamped_half.n_ang_vel + clamped_half.n_force
moment_sol_clamped = coenergy_sol_clamped[idx_moment_clamped:, :]

lin_velocity_sol_free = coenergy_sol_free[:free_half.n_lin_vel, :]
idx_moment_free = free_half.n_lin_vel + free_half.n_ang_vel + free_half.n_force
moment_sol_free = coenergy_sol_free[idx_moment_free:, :]

# lin_velocity_sol = jnp.concatenate([lin_velocity_sol_clamped, lin_velocity_sol_free], axis=0)
# moment_sol = jnp.concatenate([moment_sol_clamped, moment_sol_free], axis=0)

clamped_half_x_midpoints = 0.5 * (clamped_half.x_nodes[:-1] + clamped_half.x_nodes[1:])
free_half_x_midpoints = 0.5 * (free_half.x_nodes[:-1] + free_half.x_nodes[1:])
free_half_x_midpoints = free_half_x_midpoints + clamped_half.L

# initial_position = jnp.stack([jnp.hstack((clamped_half_x_midpoints, clamped_half.L + free_half.x_nodes)),
#                               jnp.zeros(cantilever_beam.n_ang_vel)], axis=1)
# position_sol = compute_positions(theta_sol, lin_velocity_sol, initial_position, time_vec)

initial_position_clamped = jnp.stack([clamped_half_x_midpoints,
                                      jnp.zeros(clamped_half.n_ang_vel)], axis=1)
position_sol_clamped = compute_positions(theta_sol_clamped, lin_velocity_sol_clamped, initial_position_clamped, time_vec)

initial_position_free = jnp.stack([free_half.x_nodes + clamped_half.L,
                                   jnp.zeros(free_half.n_ang_vel)], axis=1)
position_sol_free = compute_positions(theta_sol_free, lin_velocity_sol_free, initial_position_free, time_vec)


# # Create and display animation
# # from src.postprocessing.animate import animate_overlay_beam_deformation
# # print("Creating animation...")
# # anim = animate_overlay_beam_deformation(
# #     time_vec, position_sol_clamped, position_sol_free,
# #     fps=5, interval=100,
# #     title="Comparison: clamped and free formulations roll-up",
# #     colors=('blue', 'red'),
# #     labels=('Clamped', 'Free'),
# #     lock_at_end=True,
# #     save_path=folder_results + "interconnection_roll_up.gif"
# # )

# # Show the animation
# # plt.show()


plt.figure(figsize=(8, 6))
plt.plot(clamped_half.x_nodes, moment_sol_clamped[:, -1], label='Clamped', color='blue')
plt.plot(free_half_x_midpoints, moment_sol_free[:, -1], label='Free', color='red')
plt.xlabel('x')
plt.ylabel('Bending Moment $M$')
plt.title('Bending Moment Distribution at Final Time')
plt.legend()
plt.grid()
plt.show()

# time_instants = [j*1/5 for j in range(6)]
# time_indices = [j for j in range(n_steps+1) if j*dt in time_instants]

# norm_clamped = Normalize(vmin=jnp.min(moment_sol_clamped), vmax=jnp.max(moment_sol_clamped))
# norm_free = Normalize(vmin=jnp.min(moment_sol_free), vmax=jnp.max(moment_sol_free))

# fig, ax = plt.subplots()
# for j in time_indices:

#     x_nodes_clamped = position_sol_clamped[0, :, j]
#     y_nodes_clamped = position_sol_clamped[1, :, j]

#     x_nodes_free = position_sol_free[0, :, j]
#     y_nodes_free = position_sol_free[1, :, j]
    
#     # Create segments between nodes
#     segments_clamped = np.array([[[x_nodes_clamped[i], y_nodes_clamped[i]], 
#                                 [x_nodes_clamped[i+1], y_nodes_clamped[i+1]] ]
#                                     for i in range(x_nodes_clamped.size - 1)])

#     segments_free = np.array([[ [x_nodes_free[i], y_nodes_free[i]], 
#                                [x_nodes_free[i+1], y_nodes_free[i+1]] ] 
#                          for i in range(x_nodes_free.size - 1)])
    
#     # Color by bending moment
#     colors_clamped = moment_sol_clamped[:, j]  # shape: (n_elements,)
#     colors_free = moment_sol_free[:, j]  # shape: (n_elements,)
    
#     lc_clamped = LineCollection(segments_clamped, array=colors_clamped, cmap='viridis', linewidths=2, norm=norm_clamped)
#     lc_free = LineCollection(segments_free, array=colors_free, cmap='viridis', linewidths=2, norm=norm_free)

#     plt.gca().add_collection(lc_clamped)
#     plt.gca().add_collection(lc_free)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# # ax.set_title('Beam configurations over time')
# ax.grid(True)

# # Create a colorbar axis that matches the y-axis height
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% width, small padding

# cbar_clamped = fig.colorbar(lc_clamped, cax=cax)
# cbar_clamped.set_label('$M$')
# cbar_free = fig.colorbar(lc_free, cax=cax)
# cbar_free.set_label('$M$')

# ax.set_xlim(-5, 12)
# ax.set_ylim(-1, 8)
# ax.set_aspect("equal", adjustable="box")

# fig.tight_layout()
# fig.savefig(folder_results + "Configurations.pdf", format="pdf")
# plt.show()