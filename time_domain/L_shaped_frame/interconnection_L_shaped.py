from json import load
from re import L
from src.models.intrinsic_L_shaped_frame import IntrinsicPlanarLShapedBeam
from data import geometrical_props, material_props, dt, T_end, n_elements, case
import jax.numpy as jnp
import jax, os
from src.rotations.utilities import rotation_matrix_2d

from src.time_integration.utilities import simulate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
from jax import config; config.update("jax_enable_x64", True)

n_steps = int(T_end/dt)
time_vec = dt * jnp.arange(n_steps + 1)


# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/" + case
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

L_framed_beam = IntrinsicPlanarLShapedBeam(n_elements, geometrical_props, material_props)
free_half = L_framed_beam.free_beam
cantilever_half = L_framed_beam.cantilever_beam
cantilever_clamped_half = cantilever_half.clamped_beam
cantilever_free_half = cantilever_half.free_beam

H_matrix = L_framed_beam.assemble_energy_matrix()
J_linear_matrix = L_framed_beam.assemble_linear_interconnection_matrix()
E_matrix = jax.scipy.linalg.block_diag(jnp.eye(L_framed_beam.n_ang_vel), H_matrix)

# This is in the free part reference frame
B_force_at_A = L_framed_beam.assemble_control_matrix()[:, :2]
B_force_at_B = jnp.zeros((L_framed_beam.n_coenergy, 2))
B_at_corner_free = free_half.assemble_control_matrix()[:, 3:5]
B_force_at_B = B_force_at_B.at[:free_half.n_coenergy].set(B_at_corner_free)

# This is in the cantilever part reference frame
B_force_at_C = L_framed_beam.assemble_control_matrix()[:, 3:5]


# The loads are spatial forces
T_load = 1
load_magnitude = 175.0
@jax.jit
def spatial_force(t):
    q = jnp.where(
        t <= T_load,
        jnp.where(
            t <= T_load / 2,
            load_magnitude * (2.0 * t / T_load),          # ramp up
            load_magnitude * (2.0 * (1.0 - t / T_load))   # ramp down
        ),
        0.0                                                # zero after T_load
    )
    return q


jax.jit
def vector_field(t, state_vector):
    theta_vector = state_vector[:L_framed_beam.n_ang_vel]
    theta_vector_free = theta_vector[:free_half.n_ang_vel]
    theta_vector_cantilever = theta_vector[free_half.n_ang_vel:]

    theta_at_0_free = theta_vector_free[0]
    rot_matrix_at_0_free = rotation_matrix_2d(theta_at_0_free)
    # spatial force directed as minus y direction
    material_force_at_A = - rot_matrix_at_0_free.T[:, 1] * spatial_force(t) 

    theta_at_L_free = theta_vector_free[free_half.n_ang_vel -1]
    rot_matrix_at_L_free = rotation_matrix_2d(theta_at_L_free)
    # spatial force directed as plus x direction
    material_force_at_B = rot_matrix_at_L_free.T[:, 0] * spatial_force(t)

    theta_at_L_cantilever = theta_vector_cantilever[cantilever_half.n_ang_vel -1]
    rot_matrix_at_L_cantilever = rotation_matrix_2d(theta_at_L_cantilever)
    # spatial force directed as plus y direction
    material_force_at_C = rot_matrix_at_L_cantilever.T[:, 1] * spatial_force(t)

    coenergy_vector = state_vector[L_framed_beam.n_ang_vel:]
    coenergy_vector_free = coenergy_vector[:free_half.n_coenergy]
    coenergy_vector_cantilever = coenergy_vector[free_half.n_coenergy:]

    coenergy_vector_cantilever_clamped = coenergy_vector_cantilever[:cantilever_clamped_half.n_coenergy]
    coenergy_vector_cantilever_free = coenergy_vector_cantilever[cantilever_clamped_half.n_coenergy:]

    vector_field_theta_free = coenergy_vector_free[free_half.n_lin_vel: free_half.n_vel]
    vector_field_theta_cantilever_clamped = \
        coenergy_vector_cantilever_clamped[cantilever_clamped_half.n_lin_vel:
                                           cantilever_clamped_half.n_vel]
    
    vector_field_theta_cantilever_free = \
        coenergy_vector_cantilever_free[cantilever_free_half.n_lin_vel:
                                           cantilever_free_half.n_vel]

    vector_field_theta = jnp.concatenate([vector_field_theta_free, 
                                          vector_field_theta_cantilever_clamped,
                                          vector_field_theta_cantilever_free])

    vector_field_coenergy = J_linear_matrix @ coenergy_vector \
        + L_framed_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + B_force_at_A @ material_force_at_A \
        + B_force_at_B @ material_force_at_B \
        + B_force_at_C @ material_force_at_C

    return jnp.concatenate([vector_field_theta, vector_field_coenergy])


angle_0 = jnp.zeros(L_framed_beam.n_ang_vel)
angle_free_0 = - jnp.pi/2 * jnp.ones(free_half.n_ang_vel)

angle_0 = angle_0.at[:free_half.n_ang_vel].set(angle_free_0)

coenergy_vector_0 = jnp.zeros(L_framed_beam.n_coenergy)
state_vector_0 = jnp.concatenate([angle_0, coenergy_vector_0])

# Run simulation
trajectory = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field)


theta_sol = trajectory[:L_framed_beam.n_ang_vel, :]

theta_sol_free = theta_sol[:free_half.n_ang_vel, :]

theta_sol_cantilever = theta_sol[free_half.n_ang_vel:, :]
theta_sol_cantilever_clamped = theta_sol_cantilever[:cantilever_clamped_half.n_ang_vel]
theta_sol_cantilever_free = theta_sol_cantilever[cantilever_clamped_half.n_ang_vel:]

coenergy_sol = trajectory[L_framed_beam.n_ang_vel:, :]

coenergy_sol_free = coenergy_sol[:free_half.n_coenergy, :]
lin_velocity_sol_free = coenergy_sol_free[:free_half.n_lin_vel, :]

coenergy_sol_cantilever = coenergy_sol[free_half.n_coenergy:, :]

coenergy_sol_cantilever_clamped = coenergy_sol_cantilever[:cantilever_clamped_half.n_coenergy]
lin_velocity_sol_cantilever_clamped = coenergy_sol_cantilever_clamped[:cantilever_clamped_half.n_lin_vel, :]

coenergy_sol_cantilever_free = coenergy_sol_cantilever[cantilever_clamped_half.n_coenergy:]
lin_velocity_sol_cantilever_free = coenergy_sol_cantilever_free[:cantilever_free_half.n_lin_vel, :]

point_A = jnp.array([0, L_framed_beam.free_beam.L])
x_A, y_A = point_A

initial_position_free_spatial = jnp.concatenate([x_A + free_half.x_nodes * jnp.cos(theta_sol_free[:, 0]),
                                    y_A + free_half.x_nodes * jnp.sin(theta_sol_free[:, 0])])

position_sol_free = free_half.compute_positions(theta_sol_free, 
                                                lin_velocity_sol_free, 
                                                initial_position_free_spatial, 
                                                time_vec)

cantilever_clamped_half_x_midpoints = 0.5 * (cantilever_clamped_half.x_nodes[:-1] \
                                           + cantilever_clamped_half.x_nodes[1:])

initial_position_cantilever_clamped = jnp.concatenate([cantilever_clamped_half_x_midpoints,
                                      jnp.zeros_like(cantilever_clamped_half_x_midpoints)])
position_sol_cantilever_clamped = cantilever_clamped_half.compute_positions(theta_sol_cantilever_clamped, 
                                                           lin_velocity_sol_cantilever_clamped, 
                                                           initial_position_cantilever_clamped, 
                                                           time_vec)


initial_position_cantilever_free = jnp.concatenate([cantilever_free_half.x_nodes + cantilever_clamped_half.L,
                                   jnp.zeros_like(cantilever_free_half.x_nodes)])
position_sol_cantilever_free = cantilever_free_half.compute_positions(theta_sol_cantilever_free, 
                                                     lin_velocity_sol_cantilever_free, 
                                                     initial_position_cantilever_free, 
                                                     time_vec)


total_energy, kinetic_energy, deformation_energy = L_framed_beam.compute_energies(coenergy_sol)
plt.figure()
plt.plot(time_vec, total_energy, 'k', label='Total Energy')
plt.plot(time_vec, kinetic_energy, 'b--', label='Kinetic Energy')
plt.plot(time_vec, deformation_energy, 'r--', label='Deformation Energy')
plt.xlabel('Time (s)')
# plt.title('Energy of the L-shaped frame')
plt.grid(True)
plt.legend(loc="best")
plt.savefig(folder_results + f"Energies.pdf", format="pdf")
plt.tight_layout()


plt.figure()
plt.plot(time_vec[:-1], jnp.abs(jnp.diff(total_energy)), 'b--')
#plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('$|H(\mathbf{e}_{n+1}) - H(\mathbf{e}_{n})|$')
# plt.title('Delta Energy of the L-shaped frame')
plt.grid(True)
plt.savefig(folder_results + f"EnergyRate.pdf", format="pdf")
plt.tight_layout()


time_instants = jnp.array([j for j in range(6)])
time_indices = [j for j in range(len(time_vec)) if time_vec[j] in time_instants]

norm = Normalize(vmin=jnp.min(time_instants), vmax=jnp.max(time_instants))

fig, ax = plt.subplots()
for j in time_indices:

    x_nodes_1 = position_sol_free[0, :, j]
    y_nodes_1 = position_sol_free[1, :, j]

    x_nodes_2 = position_sol_cantilever_clamped[0, :, j]
    y_nodes_2 = position_sol_cantilever_clamped[1, :, j]

    x_nodes_3 = position_sol_cantilever_free[0, :, j]
    y_nodes_3 = position_sol_cantilever_free[1, :, j]
    
    # Create segments between nodes
    segments_1 = jnp.array([[[x_nodes_1[i], y_nodes_1[i]], 
                            [x_nodes_1[i+1], y_nodes_1[i+1]] ]
                            for i in range(x_nodes_1.size - 1)])

    segments_2 = jnp.array([[[x_nodes_2[i], y_nodes_2[i]], 
                            [x_nodes_2[i+1], y_nodes_2[i+1]] ] 
                            for i in range(x_nodes_2.size - 1)])
    
    segments_3 = jnp.array([[[x_nodes_3[i], y_nodes_3[i]], 
                            [x_nodes_3[i+1], y_nodes_3[i+1]] ] 
                            for i in range(x_nodes_3.size - 1)])
    
    color_values = jnp.full(segments_1.shape[0], time_vec[j])
    
    lc_1 = LineCollection(segments_1, array=color_values, cmap='viridis', linewidths=2, norm=norm)
    lc_2 = LineCollection(segments_2, array=color_values, cmap='viridis', linewidths=2, norm=norm)
    lc_3 = LineCollection(segments_3, array=color_values, cmap='viridis', linewidths=2, norm=norm)

    plt.gca().add_collection(lc_1)
    plt.gca().add_collection(lc_2)
    plt.gca().add_collection(lc_3)

ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.set_title('Beam configurations over time')
ax.grid(True)

# Create a colorbar axis that matches the y-axis height
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% width, small padding

cbar_1 = fig.colorbar(lc_1, cax=cax)
cbar_2 = fig.colorbar(lc_2, cax=cax)
cbar_3 = fig.colorbar(lc_3, cax=cax)

cbar_1.set_label('$t$')
cbar_2.set_label('$t$')
cbar_3.set_label('$t$')


min_x_1 = jnp.min(position_sol_free[0, :, :])
min_x_2 = jnp.min(position_sol_cantilever_clamped[0, :, :])
min_x_3 = jnp.min(position_sol_cantilever_free[0, :, :])
min_x = jnp.min(jnp.array([min_x_1, min_x_2, min_x_3]))

max_x_1 = jnp.max(position_sol_free[0, :, :])
max_x_2 = jnp.max(position_sol_cantilever_clamped[0, :, :])
max_x_3 = jnp.max(position_sol_cantilever_free[0, :, :])
max_x = jnp.max(jnp.array([max_x_1, max_x_2, max_x_3]))

ax.set_xlim(min_x - 1, max_x + 1)

min_y_1 = jnp.min(position_sol_free[1, :, :])
min_y_2 = jnp.min(position_sol_cantilever_clamped[1, :, :])
min_y_3 = jnp.min(position_sol_cantilever_free[1, :, :])
min_y = jnp.min(jnp.array([min_y_1, min_y_2, min_y_3])) 

max_y_1 = jnp.max(position_sol_free[1, :, :])
max_y_2 = jnp.max(position_sol_cantilever_clamped[1, :, :])
max_y_3 = jnp.max(position_sol_cantilever_free[1, :, :])
max_y = jnp.max(jnp.array([max_y_1, max_y_2, max_y_3]))

ax.set_ylim(min_y - 1, max_y + 1)
fig.tight_layout()
fig.savefig(folder_results + "Configurations.pdf", format="pdf")


# # Create and display animation
# from src.postprocessing.animate import animate_overlay_beam_deformation
# print("Creating animation...")
# anim = animate_overlay_beam_deformation(
#     time_vec, [position_sol_free, position_sol_cantilever_clamped, position_sol_cantilever_free],
#     title="Motion of L-shaped frame under time-varying loads",
#     colors=('blue', 'red', 'green'),
#     labels=('Free', 'Cantilever Clamped', 'Cantilever Free'),
#     lock_at_end=True,
#     save_path=folder_results + "L_frame_animation.gif"
# )

plt.show()
