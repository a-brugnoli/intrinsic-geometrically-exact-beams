from src.models.intrinsic_pinned_beam import IntrinsicPlanarPinnedBeam
from data import geometrical_props, material_props, g, m_mass, beam_length
import jax.numpy as jnp
import jax, os
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.animate import animate_overlay_beam_deformation
from src.postprocessing.postprocess_variables import implicit_midpoint_integrate
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
from jax import config; config.update("jax_enable_x64", True)

# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

n_elements = 10  # Number of finite elements
pinned_beam = IntrinsicPlanarPinnedBeam(n_elements, geometrical_props, material_props)

# Matrices for the beam
H_matrix_beam = pinned_beam.assemble_energy_matrix()
J_linear_matrix_beam = pinned_beam.assemble_linear_interconnection_matrix()
B_matrix_beam_at_L = pinned_beam.assemble_control_matrix()[:, 3:]
# This is used for the interconnection with the end mass
B_beam_velocity = B_matrix_beam_at_L[:, 1:]


# Matrices for the end mass
H_matrix_end_mass = m_mass * jnp.eye(2)

# Energy matrix including the end mass
H_matrix = jax.scipy.linalg.block_diag(H_matrix_beam, H_matrix_end_mass)
E_matrix = jax.scipy.linalg.block_diag(jnp.eye(pinned_beam.n_ang_vel), H_matrix)

jax.jit
def vector_field(t, state_vector):
    theta_vector_beam = state_vector[:pinned_beam.n_ang_vel]

    # To convert from spatial to material the transpose of the rotation matrix has to be taken
    theta_at_L = theta_vector_beam[-1]
    rot_matrix_at_L = rotation_matrix_2d(theta_at_L)

    coenergy_vector_beam = state_vector[pinned_beam.n_ang_vel:\
                                        pinned_beam.n_ang_vel + pinned_beam.n_coenergy]
    
    inertial_velocity_end_mass = state_vector[-2:]

    vector_field_theta_beam = coenergy_vector_beam[pinned_beam.n_lin_vel:pinned_beam.n_vel]

    y_beam_inertial = rot_matrix_at_L @  B_beam_velocity.T @ coenergy_vector_beam
    y_mass_inertial = inertial_velocity_end_mass

    u_beam_inertial = + y_mass_inertial
    u_mass_inertial = - y_beam_inertial

    vector_field_coenergy_beam = J_linear_matrix_beam @ coenergy_vector_beam \
        + pinned_beam.assemble_nonlinear_dynamics(coenergy_vector_beam) \
        + B_beam_velocity @ rot_matrix_at_L.T @ u_beam_inertial
    
    vector_field_inertial_velocity_end_mass = -g * m_mass * jnp.array([0, 1]) \
                                              + u_mass_inertial

    return jnp.concatenate([vector_field_theta_beam, vector_field_coenergy_beam, \
                            vector_field_inertial_velocity_end_mass])


angle_beam_0 = jnp.zeros(pinned_beam.n_ang_vel)
coenergy_vector_beam_0 = jnp.zeros(pinned_beam.n_coenergy)
inertial_velocity_end_mass_0 = jnp.zeros(2)

state_vector_0 = jnp.concatenate([angle_beam_0, \
                                coenergy_vector_beam_0, \
                                inertial_velocity_end_mass_0])


dt = 0.1
T_end = 5
n_steps = int(T_end/dt)
time_vec = dt * jnp.arange(n_steps + 1)

trajectory = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field)

theta_beam_sol = trajectory[:pinned_beam.n_ang_vel, :]
coenergy_beam_sol = trajectory[pinned_beam.n_ang_vel:-2, :]
omega_beam_sol = coenergy_beam_sol[pinned_beam.n_lin_vel:pinned_beam.n_vel]
theta_L_sol = theta_beam_sol[-1]
omega_L_sol = omega_beam_sol[-1]

inertial_velocity_end_mass_sol = trajectory[-2:]
position_end_mass_0 = jnp.array([beam_length, 0])
position_end_mass_sol = implicit_midpoint_integrate(inertial_velocity_end_mass_sol, dt, 
                                                    position_end_mass_0)


energy_beam_vec, kinetic_energy_beam_vec, deformation_energy_beam_vec = \
    pinned_beam.compute_energies(coenergy_beam_sol)

kinetic_energy_end_mass = 0.5*m_mass*jnp.sum(inertial_velocity_end_mass_sol ** 2, axis=0)
potential_energy_end_mass = m_mass * g * position_end_mass_sol[1, :]

kinetic_energy_vec = kinetic_energy_beam_vec + kinetic_energy_end_mass
potential_energy_vec  = deformation_energy_beam_vec + potential_energy_end_mass

total_energy_vec = kinetic_energy_vec + potential_energy_vec
energy_rate = - m_mass * g * inertial_velocity_end_mass_sol[-1]

# Plot of energy trend
plt.figure()
plt.plot(time_vec, total_energy_vec, 'm', label='H')
plt.plot(time_vec, kinetic_energy_vec, 'g', label='T')
plt.plot(time_vec, potential_energy_vec, 'b', label='V')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy vs Time')
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")

# Plot of energy rate
plt.figure()
plt.plot(time_vec, energy_rate)
plt.xlabel('Time (s)')
plt.ylabel('Energy rate (W)')
plt.title('Energy rate')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"Energy_rate.pdf", format="pdf")

plt.figure()
plt.plot(time_vec, theta_L_sol + jnp.pi/2)
plt.xlabel('Time (s)')
plt.ylabel('$\\theta(L,t)$')
plt.title('End rotation vs Time')
plt.grid(True)
plt.savefig(folder_results + f"Angle.pdf", format="pdf")
plt.tight_layout()

plt.figure()
plt.plot(time_vec, omega_L_sol)
plt.xlabel('Time (s)')
plt.ylabel('$\omega(L,t)$')
plt.title('End angular velocity vs Time')
plt.grid(True)
plt.savefig(folder_results + f"Omega.pdf", format="pdf")
plt.tight_layout()

plt.figure()
plt.plot(time_vec, inertial_velocity_end_mass_sol[1])
plt.xlabel('Time (s)')
plt.ylabel('$v_y$')
plt.title('$v_y$ mass vs Time')
plt.grid(True)
plt.savefig(folder_results + f"Vel_y.pdf", format="pdf")
plt.tight_layout()


lin_velocity_sol = coenergy_beam_sol[:pinned_beam.n_lin_vel, :]
idx_moment = pinned_beam.n_lin_vel + pinned_beam.n_ang_vel + pinned_beam.n_force
moment_sol = coenergy_beam_sol[idx_moment:, :]


point_A = jnp.array([0, 0])
x_A, y_A = point_A
position_nodes_x = x_A + pinned_beam.x_nodes * jnp.cos(theta_beam_sol[:, 0])
position_midpoint_x = 0.5*(position_nodes_x[1:] + position_nodes_x[:-1])

position_nodes_y = y_A + pinned_beam.x_nodes * jnp.sin(theta_beam_sol[:, 0])
position_midpoint_y = 0.5*(position_nodes_y[1:] + position_nodes_y[:-1])

initial_position = jnp.concatenate([position_midpoint_x,
                                    position_midpoint_y])

position_beam_sol = pinned_beam.compute_positions(theta_beam_sol, 
                                                       lin_velocity_sol, 
                                                       initial_position, 
                                                       time_vec)

dual_animation = animate_overlay_beam_deformation(time_vec, [position_beam_sol, \
                                                  position_end_mass_sol.reshape(2, 1, -1)], \
                                 labels=('Beam', 'Mass'),
                                 save_path=folder_results + f"PendulumMass.gif")


# Maximum time to plot
T_max = T_end

# Determine which indices correspond to t <= T_max
time_indices = [j for j in range(n_steps+1) if j*dt <= T_max and j % 5 == 0]

norm = Normalize(vmin=0, vmax=T_max)

fig, ax = plt.subplots()
for j in time_indices:
    x_nodes = position_beam_sol[0, :, j]
    y_nodes = position_beam_sol[1, :, j]
    
    # Create segments between nodes
    segments = jnp.array([[ [x_nodes[i], y_nodes[i]], [x_nodes[i+1], y_nodes[i+1]] ] 
                         for i in range(x_nodes.size - 1)])
    
    # Color by bending moment
    # colors = moment_sol[:, j]  # shape: (n_elements,)

    colors = jnp.full(x_nodes.size - 1, j*dt)

    lc = LineCollection(segments, array=colors, cmap='viridis', linewidths=2, norm=norm)
    plt.gca().add_collection(lc)

    # --- NEW: add point mass ---
    x_m = float(position_end_mass_sol[0, j])
    y_m = float(position_end_mass_sol[1, j])
    ax.scatter(x_m, y_m, c='red', s=50, zorder=5)  # marker size/colour adjustable

ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.set_title('Beam configurations over time')
ax.grid(True)

# Create a colorbar axis that matches the y-axis height
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% width, small padding
cbar = fig.colorbar(lc, cax=cax)
# cbar.set_label('$M$')
cbar.set_label('$t$')

ax.set_xlim(-1.1*beam_length, 1.1*beam_length)
ax.set_ylim(-1.1*beam_length, 0.1*beam_length)
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(folder_results + "Configurations.pdf", format="pdf")
plt.show()