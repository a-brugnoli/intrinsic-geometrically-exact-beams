from src.models.intrinsic_pinned_beam import IntrinsicPlanarPinnedBeam
from src.material_data.steel import material_properties, geometrical_properties
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
import os
from jax import config; config.update("jax_enable_x64", True)

# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results_single_beam_test/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

n_elem = 20
theta_0 = 0.0  # initial angle
T_end = 10.0
dt = 0.01
pinned_beam = IntrinsicPlanarPinnedBeam(n_elem, geometrical_properties, 
                                        material_properties)

H_matrix_pinned = pinned_beam.assemble_energy_matrix()
J_linear_matrix_pinned = pinned_beam.assemble_linear_interconnection_matrix()

jax.jit
def vector_field_pinned(t, state_vector):
    theta_vector = state_vector[:pinned_beam.n_ang_vel]
    coenergy_vector = state_vector[pinned_beam.n_ang_vel:]

    vector_field_theta = coenergy_vector[pinned_beam.n_lin_vel:pinned_beam.n_vel]

    vector_field_coenergy = J_linear_matrix_pinned @ coenergy_vector \
        + pinned_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + pinned_beam.assemble_gravity(theta_vector)

    return jnp.concatenate([vector_field_theta, vector_field_coenergy])


angle_0_pinned = theta_0 * jnp.ones(pinned_beam.n_ang_vel)
coenergy_vector_0_pinned = jnp.zeros(pinned_beam.n_coenergy)
state_vector_0_pinned = jnp.concatenate([angle_0_pinned, coenergy_vector_0_pinned])

# Run simulation for clamped beam
E_matrix_pinned = jax.scipy.linalg.block_diag(jnp.eye(pinned_beam.n_ang_vel), H_matrix_pinned)

n_steps = int(T_end/dt)
time_vec = dt * jnp.arange(n_steps + 1)

trajectory_pinned = simulate(state_vector_0_pinned, dt, n_steps, E_matrix_pinned, vector_field_pinned)

theta_sol_pinned = trajectory_pinned[:pinned_beam.n_ang_vel]
coenergy_sol_pinned = trajectory_pinned[pinned_beam.n_ang_vel:]

x_points_pinned_0 = pinned_beam.x_nodes * jnp.cos(theta_sol_pinned[:, 0])
y_points_pinned_0 = pinned_beam.x_nodes * jnp.sin(theta_sol_pinned[:, 0])
x_midpoints_pinned_0 = 0.5*(x_points_pinned_0[1:] + x_points_pinned_0[:-1])
y_midpoints_pinned_0 = 0.5*(y_points_pinned_0[1:] + y_points_pinned_0[:-1])

initial_position_pinned = jnp.concatenate([x_midpoints_pinned_0,
                                     y_midpoints_pinned_0])

lin_velocity_sol_pinned = coenergy_sol_pinned[:pinned_beam.n_lin_vel, :]
position_sol_pinned = pinned_beam.compute_positions(theta_sol_pinned, 
                                                         lin_velocity_sol_pinned, 
                                                         initial_position_pinned, 
                                                         time_vec)

intrinsic_energy_vec_pinned, kinetic_energy_vec_pinned, deformation_energy_vec_pinned = \
    pinned_beam.compute_energies(coenergy_sol_pinned)

potential_energy_vec_pinned = pinned_beam.compute_potential_energy(position_sol_pinned[1])
total_energy_vec_pinned = intrinsic_energy_vec_pinned + potential_energy_vec_pinned

# # Plot of energy trend
plt.figure()
plt.plot(time_vec, intrinsic_energy_vec_pinned, 'b--', label='Intrinsic Energy')
plt.plot(time_vec, potential_energy_vec_pinned, 'r-', label='Potential Energy')
plt.plot(time_vec, total_energy_vec_pinned, 'k', label='Total Energy')
plt.xlabel('Time (s)')
plt.title('Energies vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")

# Plot of energy difference
plt.figure()
plt.plot(time_vec[1:], jnp.abs(jnp.diff(total_energy_vec_pinned)), 'r')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('$|H(\mathbf{e}_{n+1}) - H(\mathbf{e}_{n})|$')
plt.title('Delta Energy vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"DeltaEnergy.pdf", format="pdf")


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
x_min = jnp.min(position_sol_pinned[0, :, :])
x_max = jnp.max(position_sol_pinned[0, :, :])
y_min = jnp.min(position_sol_pinned[1, :, :])
y_max = jnp.max(position_sol_pinned[1, :, :])
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
fig.tight_layout()
fig.savefig(folder_results + "Configurations.pdf", format="pdf")


n_plot = 500
if n_plot<=n_steps:
    step_plot = n_steps//n_plot
else:
    step_plot=1
time_vec_plot = time_vec[::step_plot]
position_plot_pinned = position_sol_pinned[:, :, ::step_plot]

# Create and display animation
from src.postprocessing.animate import animate_beam_deformation
print("Creating animation...")
anim = animate_beam_deformation(
    time_vec_plot, position_plot_pinned,
    fps=20, interval=100,
    title="Beam under distributed Load",
    save_path=folder_results + f"Pinned.gif"
)

plt.show()
