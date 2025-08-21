from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from data import geometrical_props, material_props
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d, rotation_matrix_2d_batch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np  
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
import os
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

# print("Length: ", free_beam.L)
# print("Density per unit length: ", free_beam.density_unit_length)
# print("Rotational inertia per unit length: ", free_beam.rotational_inertia)
# print("Axial per unit length: ", free_beam.axial_compliance**(-1))
# print("Shear stiffness per unit length: ", free_beam.shear_compliance**(-1))
# print("Bending stiffness: ", free_beam.bending_compliance**(-1))

H_matrix = free_beam.assemble_energy_matrix()
J_linear_matrix = free_beam.assemble_linear_interconnection_matrix()

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

    vector_field_theta = coenergy_vector[free_beam.n_lin_vel:free_beam.n_lin_vel + free_beam.n_ang_vel]

    vector_field_coenergy = J_linear_matrix @ coenergy_vector \
        + free_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + B_force @ material_force_at_L \
        + B_torque * spartial_torque(t) 

    return jnp.concatenate([vector_field_theta, vector_field_coenergy])


alpha = jnp.atan2(-8, 6)
angle_0 = alpha * jnp.ones(free_beam.n_ang_vel)
coenergy_vector_0 = jnp.zeros(free_beam.n_dofs)
state_vector_0 = jnp.concatenate([angle_0, coenergy_vector_0])

dt = 0.1
T_end = 15
n_steps = int(T_end/dt)

# Run simulation
E_matrix = jax.scipy.linalg.block_diag(jnp.eye(free_beam.n_ang_vel), H_matrix)
trajectory = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field)

theta_sol = trajectory[:free_beam.n_ang_vel, :]
coenergy_sol = trajectory[free_beam.n_ang_vel:, :]

energy_vec, kinetic_energy_vec, deformation_energy_vec = free_beam.compute_energies(coenergy_sol)

# Plot of energy trend
plt.figure()
plt.plot(dt* jnp.arange(n_steps+1), energy_vec, 'm', label='H')
plt.plot(dt* jnp.arange(n_steps+1), kinetic_energy_vec, 'g', label='T')
plt.plot(dt* jnp.arange(n_steps+1), deformation_energy_vec, 'b', label='V')
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
plt.plot(dt* jnp.arange(n_steps), jnp.abs(jnp.diff(energy_vec)), 'm')
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
bending_moment_sol = coenergy_sol[idx_moment:, :]
# Integrate position vector dynamics using the trapezoidal rule
reshaped_velocity_sol = lin_velocity_sol.reshape(2, free_beam.n_nodes, -1)

position_sol = jnp.zeros_like(reshaped_velocity_sol)

x_local = jnp.linspace(0, free_beam.L, free_beam.n_nodes)

point_A = jnp.array([0, 8])
x_A, y_A = point_A
position_sol = position_sol.at[0, :, 0].set(x_A + x_local * jnp.cos(theta_sol[:, 0]))
position_sol = position_sol.at[1, :, 0].set(y_A + x_local * jnp.sin(theta_sol[:, 0]))

def integration_position(position_prev, ii):

    theta_mid = (theta_sol[:, ii] + theta_sol[:, ii+1]) / 2        # shape (N,)
    velocity_mid = (reshaped_velocity_sol[:, :, ii] + reshaped_velocity_sol[:, :, ii+1]) / 2  # shape (2, N)

    # Compute rotation matrices for all nodes: shape (N, 2, 2)
    R_batch = rotation_matrix_2d_batch(theta_mid)

    # Apply rotation to each velocity vector
    delta = jnp.einsum('nij,jn->in', R_batch, velocity_mid)  # result: (2, N)

    position_next = position_prev + dt * delta
    return position_next, position_next

# Scan over time steps
# Initialize
position_sol_init = position_sol[:, :, 0]  # shape (2, N)

_, positions = jax.lax.scan(integration_position, position_sol_init, jnp.arange(n_steps))

# Concatenate initial position to match original shape (2, N, n_steps+1)
position_sol = jnp.concatenate([position_sol_init[None, :, :], positions], axis=0).transpose(1, 2, 0)


# Maximum time to plot
T_max = 7.5

# Determine which indices correspond to t <= T_max
time_indices = [j for j in range(position_sol.shape[2]) if j*dt <= T_max and j % 5 == 0]

norm = Normalize(vmin=-110, vmax=110)

plt.figure(figsize=(8,6))
for j in time_indices:
    x_nodes = position_sol[0, :, j]
    y_nodes = position_sol[1, :, j]
    
    # Create segments between nodes
    segments = np.array([[ [x_nodes[i], y_nodes[i]], [x_nodes[i+1], y_nodes[i+1]] ] 
                         for i in range(x_nodes.size - 1)])
    
    # Color by bending moment
    colors = bending_moment_sol[:, j]  # shape: (n_elements,)
    
    lc = LineCollection(segments, array=colors, cmap='viridis', linewidths=2, norm=norm)
    plt.gca().add_collection(lc)

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1, 19)
plt.ylim(-1, 9)
plt.title('Beam configurations over time')
plt.grid(True)
plt.colorbar(lc, label='$M$')

plt.tight_layout()
plt.savefig(folder_results + f"Configurations.pdf", format="pdf")
plt.show()