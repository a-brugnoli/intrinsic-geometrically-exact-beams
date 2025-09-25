from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from data import geometrical_props, material_props
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.postprocess_variables import compute_positions
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()

import os
from jax import config; config.update("jax_enable_x64", True)


# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

n_elements = 10  # Number of finite elements
free_beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)

# This is not consistent physically but it is the original setup by Simo and Vu-Quoc
EA = 1e4
free_beam.set_axial_compliance(1/EA)

H_matrix = free_beam.assemble_energy_matrix()
J_linear_matrix = free_beam.assemble_linear_interconnection_matrix()
E_matrix = jax.scipy.linalg.block_diag(jnp.eye(free_beam.n_ang_vel), H_matrix)

B_matrix_at_L = free_beam.assemble_control_matrix()[:, 3:]

B_force = B_matrix_at_L[:, :2]
B_torque = B_matrix_at_L[:, 2]


jax.jit
def spatial_force(t):
    return jnp.where(t <= 2.5, jnp.array([8.0, 0]), jnp.array([0, 0]))


jax.jit
def spartial_torque(t):
    # Example: sinusoidal torque
    return jnp.where(t <= 2.5, -80.0, 0.0)

jax.jit
def vector_field(state_vector, t):
    theta_vector = state_vector[:free_beam.n_ang_vel]
    coenergy_vector = state_vector[free_beam.n_ang_vel:]

    # To convert from spatial to material the transpose of the rotation matrix has to be taken
    theta_at_L = theta_vector[-1]
    rot_matrix_at_L = rotation_matrix_2d(theta_at_L)
    material_force_at_L = rot_matrix_at_L.T @ spatial_force(t)

    vector_field_theta = coenergy_vector[free_beam.n_lin_vel:free_beam.n_vel]

    vector_field_coenergy = J_linear_matrix @ coenergy_vector \
        + free_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + B_force @ material_force_at_L \
        + B_torque * spartial_torque(t) 

    return jnp.concatenate([vector_field_theta, vector_field_coenergy])


alpha = jnp.atan2(-8, 6)
angle_0 = alpha * jnp.ones(free_beam.n_ang_vel)
coenergy_vector_0 = jnp.zeros(free_beam.n_coenergy)
state_vector_0 = jnp.concatenate([angle_0, coenergy_vector_0])

dt = 0.1
T_end = 15
n_steps = int(T_end/dt)
time_vec = dt * jnp.arange(n_steps + 1)

# Run simulation
trajectory = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field)

theta_sol = trajectory[:free_beam.n_ang_vel, :]
coenergy_sol = trajectory[free_beam.n_ang_vel:, :]

energy_vec, kinetic_energy_vec, deformation_energy_vec = free_beam.compute_energies(coenergy_sol)

# Plot of energy trend
plt.figure()
plt.plot(time_vec, energy_vec, 'm', label='H')
plt.plot(time_vec, kinetic_energy_vec, 'g', label='T')
plt.plot(time_vec, deformation_energy_vec, 'b', label='V')
plt.ylim(0, 400)
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy vs Time')
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")

# Plot of energy difference
plt.figure()
plt.plot(time_vec[1:], jnp.abs(jnp.diff(energy_vec)), 'm')
plt.yscale('log')
plt.ylim(1e-14, 1e2)
plt.xlabel('Time (s)')
plt.ylabel('$|H(\mathbf{e}_{n+1}) - H(\mathbf{e}_{n})|$')
plt.title('Delta Energy vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"DeltaEnergy.pdf", format="pdf")


lin_velocity_sol = coenergy_sol[:free_beam.n_lin_vel, :]
idx_moment = free_beam.n_lin_vel + free_beam.n_ang_vel + free_beam.n_force
moment_sol = coenergy_sol[idx_moment:, :]


point_A = jnp.array([0, 8])
x_A, y_A = point_A

initial_position = jnp.stack([x_A + free_beam.x_nodes * jnp.cos(theta_sol[:, 0]),
                                    y_A + free_beam.x_nodes * jnp.sin(theta_sol[:, 0])], axis=1)

position_sol = compute_positions(theta_sol, lin_velocity_sol, initial_position, time_vec)


# Maximum time to plot
T_max = 7.5

# Determine which indices correspond to t <= T_max
time_indices = [j for j in range(n_steps+1) if j*dt <= T_max and j % 5 == 0]

norm = Normalize(vmin=jnp.min(moment_sol), vmax=jnp.max(moment_sol))

fig, ax = plt.subplots()
for j in time_indices:
    x_nodes = position_sol[0, :, j]
    y_nodes = position_sol[1, :, j]
    
    # Create segments between nodes
    segments = jnp.array([[ [x_nodes[i], y_nodes[i]], [x_nodes[i+1], y_nodes[i+1]] ] 
                         for i in range(x_nodes.size - 1)])
    
    # Color by bending moment
    colors = moment_sol[:, j]  # shape: (n_elements,)
    
    lc = LineCollection(segments, array=colors, cmap='viridis', linewidths=2, norm=norm)
    plt.gca().add_collection(lc)

ax.set_xlabel('x')
ax.set_ylabel('y')
# ax.set_title('Beam configurations over time')
ax.grid(True)

# Create a colorbar axis that matches the y-axis height
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% width, small padding
cbar = fig.colorbar(lc, cax=cax)
cbar.set_label('$M$')

ax.set_xlim(-1, 18)
ax.set_ylim(-1, 9)
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(folder_results + "Configurations.pdf", format="pdf")
plt.show()