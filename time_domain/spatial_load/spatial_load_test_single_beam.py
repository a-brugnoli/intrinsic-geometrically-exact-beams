from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from src.models.intrinsic_pinned_beam import IntrinsicPlanarPinnedBeam
from src.material_data.steel import material_properties, geometrical_properties, rho, A
 
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

n_elem = 20
g = 9.81  # m/s^2
theta_0 = 0.0  # initial angle
T_end = 10.0
dt = 0.01
pinned_beam = IntrinsicPlanarPinnedBeam(n_elem, geometrical_properties, 
                                        material_properties)

H_matrix_pinned = pinned_beam.assemble_energy_matrix()
J_linear_matrix_pinned = pinned_beam.assemble_linear_interconnection_matrix()

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

x_points_pinned_0 = pinned_beam.x_nodes * jnp.cos(theta_sol_pinned[:, 0])
y_points_pinned_0 = pinned_beam.x_nodes * jnp.sin(theta_sol_pinned[:, 0])
x_midpoints_pinned_0 = 0.5*(x_points_pinned_0[1:] + x_points_pinned_0[:-1])
y_midpoints_pinned_0 = 0.5*(y_points_pinned_0[1:] + y_points_pinned_0[:-1])

initial_position_pinned = jnp.stack([x_midpoints_pinned_0,
                                     y_midpoints_pinned_0], axis=1)

lin_velocity_sol_pinned = coenergy_sol_pinned[:pinned_beam.n_lin_vel, :]
position_sol_pinned = compute_beam_positions(theta_sol_pinned, lin_velocity_sol_pinned, initial_position_pinned, time_vec)

norm_position_sol = jnp.linalg.norm(position_sol_pinned, axis=(0, 1))
norm_lin_velocity_sol = jnp.linalg.norm(lin_velocity_sol_pinned, axis=(0))
norm_theta_sol = jnp.linalg.norm(theta_sol_pinned, axis=(0))

plt.figure()
plt.plot(time_vec, norm_position_sol)
plt.xlabel('Time (s)')
plt.ylabel(r'$||\mathbf{r}||$')
plt.title(' Norm position vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"NormPosition.pdf", format="pdf")


plt.figure()
plt.plot(time_vec, norm_lin_velocity_sol)
plt.xlabel('Time (s)')
plt.ylabel(r'$||\mathbf{v}||$')
plt.title(' Velocity position vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"NormVelocity.pdf", format="pdf")


plt.figure()
plt.plot(time_vec, norm_theta_sol)
plt.xlabel('Time (s)')
plt.ylabel(r'$||\theta||$')
plt.title(' Norm Theta vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"NormTheta.pdf", format="pdf")


# # Plot of energy trend
plt.figure()
plt.plot(time_vec, energy_vec_pinned)
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")


# # Plot of energy trend
plt.figure()
plt.plot(time_vec, kinetic_energy_vec_pinned)
plt.xlabel('Time (s)')
plt.ylabel('Kinetic Energy (J)')
plt.title('Kinetic Energy vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"Kin_Energies.pdf", format="pdf")


# # Plot of energy trend
plt.figure()
plt.plot(time_vec, deformation_energy_vec_pinned)
plt.xlabel('Time (s)')
plt.ylabel('Potential Energy (J)')
plt.title('Potential Energy vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"Potential_Energies.pdf", format="pdf")


# Plot of energy difference
plt.figure()
plt.plot(time_vec[1:], jnp.abs(jnp.diff(energy_vec_pinned)), 'r')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('$|H(\mathbf{e}_{n+1}) - H(\mathbf{e}_{n})|$')
plt.title('Delta Energy vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"DeltaEnergy.pdf", format="pdf")



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

time_instants = jnp.array([j*2 for j in range(6)])
time_indices = [j for j in range(len(time_vec)) if time_vec[j] in time_instants]

norm = Normalize(vmin=jnp.min(time_instants), vmax=jnp.max(time_instants))

fig, ax = plt.subplots()
for j in time_indices:

    x_nodes = position_sol_pinned[0, :, j]
    y_nodes = position_sol_pinned[1, :, j]
    segments = jnp.array([[[x_nodes[i], y_nodes[i]], 
                            [x_nodes[i+1], y_nodes[i+1]] ]
                            for i in range(x_nodes.size - 1)])

    color_values = jnp.full(segments.shape[0], time_vec[j])
    
    lc = LineCollection(segments, array=color_values, cmap='viridis', linewidths=2, norm=norm)

    plt.gca().add_collection(lc)

ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.set_title('Beam configurations over time')
ax.grid(True)

# Create a colorbar axis that matches the y-axis height
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% width, small padding

cbar = fig.colorbar(lc, cax=cax)

cbar.set_label('$t$')

ax.set_xlim(-0.1, 3.5)
ax.set_ylim(-2.1, 1.5)

fig.tight_layout()
fig.savefig(folder_results + "Configurations.pdf", format="pdf")



plt.show()


