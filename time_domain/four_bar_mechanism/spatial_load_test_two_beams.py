from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from src.models.intrinsic_pinned_beam import IntrinsicPlanarPinnedBeam

from data import geometrical_props, material_props, \
    A, rho, g, theta_0, dt, T_end, n_elem
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d_batch
from src.fem.forcing import assemble_state_dependent_forcing_vector_cg, \
    assemble_state_dependent_forcing_vector_dg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.postprocess_variables import compute_beam_positions
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
import os
from jax import config; config.update("jax_enable_x64", True)

# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/spatial_load_test/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

pinned_beam = IntrinsicPlanarPinnedBeam(n_elem, geometrical_props, material_props)
H_matrix_pinned = pinned_beam.assemble_energy_matrix()
J_linear_matrix_pinned = pinned_beam.assemble_linear_interconnection_matrix()


free_beam = IntrinsicPlanarFreeBeam(n_elem, geometrical_props, material_props)
H_matrix_free = free_beam.assemble_energy_matrix()
J_linear_matrix_free = free_beam.assemble_linear_interconnection_matrix()
B_matrix_free = free_beam.assemble_control_matrix()
B_free_at_0 = B_matrix_free[:, :3]
B_free_force_at_0 = B_free_at_0[:, :2]
B_free_at_L = B_matrix_free[:, 3:]
B_free_force_at_L = B_free_at_L[:, :2]

# Simply supported Bcs
n_lambda = 4

jax.jit
def spatial_force(t):
    return jnp.array([0, -g*rho*A])

    # t_end_loading = 10*dt
    # return jnp.where(t <= t_end_loading, 
    #                  jnp.array([0, -t/t_end_loading*g*rho*A]), 
    #                  jnp.array([0, -g*rho*A]))


# Simbolic expression to be converted into a distributed force
def exp_f_material_x(t):
    return lambda angle: + jnp.cos(angle) * spatial_force(t)[0] \
                         + jnp.sin(angle) * spatial_force(t)[1]

def exp_f_material_y(t):
    return lambda angle: - jnp.sin(angle) * spatial_force(t)[0] \
                         + jnp.cos(angle) * spatial_force(t)[1]   


jax.jit
def vector_field_pinned(state_vector, t):
    theta_vector = state_vector[:pinned_beam.n_ang_vel]
    coenergy_vector = state_vector[pinned_beam.n_ang_vel:]

    load_vector = jnp.zeros((pinned_beam.n_coenergy, ))

    f_vector_x = assemble_state_dependent_forcing_vector_dg(pinned_beam.x_nodes,\
                                                            theta_vector, exp_f_material_x(t))    
    f_vector_y = assemble_state_dependent_forcing_vector_dg(pinned_beam.x_nodes,\
                                                            theta_vector, exp_f_material_y(t))
    

    # Forcing along the x direction
    load_vector = load_vector.at[:pinned_beam.n_lin_vel_x].set(f_vector_x)
    # Forcing along the y direction
    load_vector = load_vector.at[pinned_beam.n_lin_vel_x:pinned_beam.n_lin_vel].set(f_vector_y)


    vector_field_theta = coenergy_vector[pinned_beam.n_lin_vel:pinned_beam.n_vel]

    vector_field_coenergy = J_linear_matrix_pinned @ coenergy_vector \
        + pinned_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + load_vector

    return jnp.concatenate([vector_field_theta, vector_field_coenergy])


# jax.jit
# def vector_field_free(state_vector, t):
#     theta_vector = state_vector[:free_beam.n_ang_vel]
#     coenergy_vector = state_vector[free_beam.n_ang_vel:-n_lambda]
#     lagrange_multiplier = state_vector[-n_lambda:]
#     lagrange_multiplier_at_0 = lagrange_multiplier[:2]
#     lagrange_multiplier_at_L = lagrange_multiplier[2:]

#     f_vector_x = assemble_state_dependent_forcing_vector_cg(free_beam.x_nodes, \
#                                                             theta_vector, exp_f_material_x(t))
#     f_vector_y = assemble_state_dependent_forcing_vector_cg(free_beam.x_nodes, \
#                                                             theta_vector, exp_f_material_y(t))
#     # Force vector of the size of the system
#     load_vector = jnp.zeros((free_beam.n_coenergy, ))
#     # Forcing along the x direction
#     load_vector = load_vector.at[:free_beam.n_lin_vel_x].set(f_vector_x)
#     # Forcing along the y direction
#     load_vector = load_vector.at[free_beam.n_lin_vel_x:free_beam.n_lin_vel].set(f_vector_y)

#     vector_field_theta = coenergy_vector[free_beam.n_lin_vel:free_beam.n_vel]

#     vector_field_coenergy = J_linear_matrix_free @ coenergy_vector \
#         + free_beam.assemble_nonlinear_dynamics(coenergy_vector) \
#         + B_free_force_at_0 @ lagrange_multiplier_at_0 \
#         + B_free_force_at_L @ lagrange_multiplier_at_L \
#         + load_vector
    
#     vector_field_constraints_at_0 = - B_free_force_at_0.T @ coenergy_vector
#     vector_field_constraints_at_L = - B_free_force_at_L.T @ coenergy_vector

#     return jnp.concatenate([vector_field_theta, vector_field_coenergy, 
#                             vector_field_constraints_at_0,
#                             vector_field_constraints_at_L])



angle_0_pinned = theta_0 * jnp.ones(pinned_beam.n_ang_vel)
coenergy_vector_0_pinned = jnp.zeros(pinned_beam.n_coenergy)
state_vector_0_pinned = jnp.concatenate([angle_0_pinned, coenergy_vector_0_pinned])

# Run simulation for clamped beam
E_matrix_pinned = jax.scipy.linalg.block_diag(jnp.eye(pinned_beam.n_ang_vel), H_matrix_pinned)

n_steps = int(T_end/dt)
time_vec_all = dt * jnp.arange(n_steps + 1)

trajectory_pinned = simulate(state_vector_0_pinned, dt, n_steps, E_matrix_pinned, vector_field_pinned)

n_plot = 500
if n_plot<=n_steps:
    step_plot = n_steps//n_plot
else:
    step_plot=1
time_vec = time_vec_all[::step_plot]

theta_sol_pinned = trajectory_pinned[:pinned_beam.n_ang_vel, ::step_plot]
coenergy_sol_pinned = trajectory_pinned[pinned_beam.n_ang_vel:, ::step_plot]

energy_vec_pinned, kinetic_energy_vec_pinned, deformation_energy_vec_pinned = \
    pinned_beam.compute_energies(coenergy_sol_pinned)

# angle_0_free = theta_0 * jnp.ones(free_beam.n_ang_vel)
# coenergy_vector_0_free = jnp.zeros(free_beam.n_coenergy)
# lagrange_multiplier_0 = jnp.zeros(n_lambda)
# state_vector_0_free = jnp.concatenate([angle_0_free, coenergy_vector_0_free, lagrange_multiplier_0])

# # Run simulation for free beam
# E_matrix_free = jax.scipy.linalg.block_diag(jnp.eye(free_beam.n_ang_vel), \
#                                             H_matrix_free, jnp.zeros((n_lambda, n_lambda)))

# trajectory_free = simulate(state_vector_0_free, dt, n_steps, E_matrix_free, vector_field_free)

# theta_sol_free = trajectory_free[:free_beam.n_ang_vel, ::step_plot]
# coenergy_sol_free = trajectory_free[free_beam.n_ang_vel:-n_lambda, ::step_plot]
# lagrange_sol_free = trajectory_free[-n_lambda:, ::step_plot]

# energy_vec_free, kinetic_energy_vec_free, deformation_energy_vec_free = \
#     free_beam.compute_energies(coenergy_sol_free)

x_points_pinned_0 = pinned_beam.x_nodes * jnp.cos(theta_sol_pinned[:, 0])
y_points_pinned_0 = pinned_beam.x_nodes * jnp.sin(theta_sol_pinned[:, 0])
x_midpoints_pinned_0 = 0.5*(x_points_pinned_0[1:] + x_points_pinned_0[:-1])
y_midpoints_pinned_0 = 0.5*(y_points_pinned_0[1:] + y_points_pinned_0[:-1])

initial_position_pinned = jnp.stack([x_midpoints_pinned_0,
                                     y_midpoints_pinned_0], axis=1)

lin_velocity_sol_pinned = coenergy_sol_pinned[:pinned_beam.n_lin_vel, :]
position_sol_pinned = compute_beam_positions(theta_sol_pinned, lin_velocity_sol_pinned, initial_position_pinned, time_vec)

jax.debug.print("Norm position 1 {}", jnp.linalg.norm(\
    jnp.linalg.norm(position_sol_pinned, axis=0), axis=0))

# x_points_free_0 = free_beam.x_nodes * jnp.cos(theta_sol_free[:, 0])
# y_points_free_0 = free_beam.x_nodes * jnp.sin(theta_sol_free[:, 0])

# initial_position_free = jnp.stack([x_points_free_0,
#                                    y_points_free_0], axis=1)

# lin_velocity_sol_free = coenergy_sol_free[:free_beam.n_lin_vel, :]
# position_sol_free = compute_beam_positions(theta_sol_free, lin_velocity_sol_free, initial_position_free, time_vec)

time_instants = jnp.array([j*2 for j in range(6)])
time_indices = [j for j in range(len(time_vec)) if time_vec[j] in time_instants]

norm = Normalize(vmin=jnp.min(time_instants), vmax=jnp.max(time_instants))

fig, ax = plt.subplots()
for j in time_indices:

    x_nodes_1 = position_sol_pinned[0, :, j]
    y_nodes_1 = position_sol_pinned[1, :, j]

    # x_nodes_2 = position_sol_free[0, :, j]
    # y_nodes_2 = position_sol_free[1, :, j]

    # Create segments between nodes
    segments_1 = jnp.array([[[x_nodes_1[i], y_nodes_1[i]], 
                            [x_nodes_1[i+1], y_nodes_1[i+1]] ]
                            for i in range(x_nodes_1.size - 1)])

    # segments_2 = jnp.array([[[x_nodes_2[i], y_nodes_2[i]], 
    #                         [x_nodes_2[i+1], y_nodes_2[i+1]] ] 
    #                         for i in range(x_nodes_2.size - 1)])
    
    color_values = jnp.full(segments_1.shape[0], time_vec[j])
    
    lc_1 = LineCollection(segments_1, array=color_values, cmap='viridis', linewidths=2, norm=norm)
    # lc_2 = LineCollection(segments_2, array=color_values, cmap='viridis', linewidths=2, norm=norm)

    plt.gca().add_collection(lc_1)
    # plt.gca().add_collection(lc_2)

ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.set_title('Beam configurations over time')
ax.grid(True)

# Create a colorbar axis that matches the y-axis height
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% width, small padding

cbar_1 = fig.colorbar(lc_1, cax=cax)
# cbar_2 = fig.colorbar(lc_2, cax=cax)

cbar_1.set_label('$t$')
# cbar_2.set_label('$t$')

ax.set_xlim(-0.1, 3.5)
ax.set_ylim(-2.1, 1.5)

fig.tight_layout()
fig.savefig(folder_results + "Configurations.pdf", format="pdf")

# # Create and display animation
# from src.postprocessing.animate import animate_overlay_beam_deformation
# print("Creating animation...")
# anim = animate_overlay_beam_deformation(
#     time_vec, [position_sol_pinned, position_sol_free],
#     fps=20, interval=100,
#     title="Comparison: Clamped Beam under distributed Load",
#     colors=('blue', 'red'),
#     labels=('Pinned', 'Free'),
#     save_path=folder_results + f"Pinned_vs_Free.gif"
# )

plt.show()


# # # Plot of energy trend
# plt.figure()
# plt.plot(dt* jnp.arange(n_steps+1), energy_vec_pinned, label='H CC')
# plt.plot(dt* jnp.arange(n_steps+1), energy_vec_free, label='H FF')
# plt.xlabel('Time (s)')
# plt.ylabel('Energy (J)')
# plt.title('Energy vs Time')
# plt.grid(True)
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig(folder_results + f"Energies.pdf", format="pdf")


# # # Plot of energy trend
# plt.figure()
# plt.plot(dt* jnp.arange(n_steps+1), kinetic_energy_vec_pinned, label='T CC')
# plt.plot(dt* jnp.arange(n_steps+1), kinetic_energy_vec_free, label='T FF')
# plt.xlabel('Time (s)')
# plt.ylabel('Kinetic Energy (J)')
# plt.title('Kinetic Energy vs Time')
# plt.grid(True)
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig(folder_results + f"Kin_Energies.pdf", format="pdf")


# # # Plot of energy trend
# plt.figure()
# plt.plot(dt* jnp.arange(n_steps+1), deformation_energy_vec_pinned, label='V CC')
# plt.plot(dt* jnp.arange(n_steps+1), deformation_energy_vec_free, label='V FF')
# plt.xlabel('Time (s)')
# plt.ylabel('Potential Energy (J)')
# plt.title('Potential Energy vs Time')
# plt.grid(True)
# plt.legend(loc="lower right")
# plt.tight_layout()
# plt.savefig(folder_results + f"Potential_Energies.pdf", format="pdf")


# # Plot of energy difference
# plt.figure()
# plt.plot(dt* jnp.arange(n_steps), jnp.abs(jnp.diff(energy_vec_pinned)), 'r', label='CC')
# plt.plot(dt* jnp.arange(n_steps), jnp.abs(jnp.diff(energy_vec_free)), 'b', label='FF')
# plt.yscale('log')
# plt.xlabel('Time (s)')
# plt.ylabel('$|H(\mathbf{e}_{n+1}) - H(\mathbf{e}_{n})|$')
# plt.title('Delta Energy vs Time')
# plt.grid(True)
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig(folder_results + f"DeltaEnergy.pdf", format="pdf")

