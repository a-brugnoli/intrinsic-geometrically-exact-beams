from src.models.intrinsic_clamped_beam import IntrinsicPlanarClampedBeam
from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam

from src.material_data.unitary import geometrical_props, material_props
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d_batch
from src.fem.forcing import assemble_state_dependent_forcing_vector_cg
import matplotlib.pyplot as plt
from src.postprocessing.postprocess_variables import compute_beam_positions
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
import os
from jax import config; config.update("jax_enable_x64", True)


dt = 0.1
T_end = 15
n_steps = int(T_end/dt)
time_vec = dt * jnp.arange(n_steps + 1)

# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

n_elements = 50  # Number of finite elements
clamped_beam = IntrinsicPlanarClampedBeam(n_elements, geometrical_props, material_props)
H_matrix_clamped = clamped_beam.assemble_energy_matrix()
J_linear_matrix_clamped = clamped_beam.assemble_linear_interconnection_matrix()


jax.jit
def spatial_force(t):
    return jnp.where(t <= 2, jnp.array([0, 5*jnp.sin(jnp.pi * t)]), jnp.array([0, 0]))


jax.jit
def vector_field_clamped(state_vector, t):
    theta_vector = state_vector[:clamped_beam.n_ang_vel]
    coenergy_vector = state_vector[clamped_beam.n_ang_vel:]

    # To convert from spatial to material the transpose of the rotation matrix has to be taken
    rot_matrix_batch = rotation_matrix_2d_batch(theta_vector)
    # # Transpose only the last two dimensions
    material_distributed_force = jnp.transpose(rot_matrix_batch, (0, 2, 1)) @ spatial_force(t)  # N_theta x 2
    material_distributed_force = material_distributed_force.T.reshape((-1, ))

    # Force vector of the size of the system
    load_vector = jnp.zeros((clamped_beam.n_coenergy, ))

    load_vector = load_vector.at[:clamped_beam.n_lin_vel].set(\
        material_distributed_force * clamped_beam.h_mesh)

    vector_field_theta = coenergy_vector[clamped_beam.n_lin_vel:clamped_beam.n_vel]

    vector_field_coenergy = J_linear_matrix_clamped @ coenergy_vector \
        + clamped_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + load_vector

    return jnp.concatenate([vector_field_theta, vector_field_coenergy])


free_beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)
H_matrix_free = free_beam.assemble_energy_matrix()
J_linear_matrix_free = free_beam.assemble_linear_interconnection_matrix()
B_matrix_free = free_beam.assemble_control_matrix()
n_lambda = B_matrix_free.shape[1]


jax.jit
def vector_field_free(state_vector, t):
    theta_vector = state_vector[:free_beam.n_ang_vel]
    coenergy_vector = state_vector[free_beam.n_ang_vel:-n_lambda]
    lagrange_multiplier = state_vector[-n_lambda:]

    # Simbolic expression to be converted into a distributed force
    exp_f_body_x = lambda angle: + jnp.cos(angle) * spatial_force(t)[0] \
                                 + jnp.sin(angle) * spatial_force(t)[1]
    
    f_vector_x = assemble_state_dependent_forcing_vector_cg(free_beam.x_nodes, theta_vector, exp_f_body_x)

    exp_f_body_y = lambda angle: - jnp.sin(angle) * spatial_force(t)[0] \
                                 + jnp.cos(angle) * spatial_force(t)[1]
    
    f_vector_y = assemble_state_dependent_forcing_vector_cg(free_beam.x_nodes, theta_vector, exp_f_body_y)
    # Force vector of the size of the system
    load_vector = jnp.zeros((free_beam.n_coenergy, ))

    # Forcing along the x direction
    load_vector = load_vector.at[:free_beam.n_lin_vel_x].set(f_vector_x)
    # Forcing along the y direction
    load_vector = load_vector.at[free_beam.n_lin_vel_x:free_beam.n_lin_vel].set(f_vector_y)

    vector_field_theta = coenergy_vector[free_beam.n_lin_vel:free_beam.n_vel]

    vector_field_coenergy = J_linear_matrix_free @ coenergy_vector \
        + free_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + B_matrix_free @ lagrange_multiplier \
        + load_vector
    
    vector_field_constraints = - B_matrix_free.T @ coenergy_vector

    return jnp.concatenate([vector_field_theta, vector_field_coenergy, vector_field_constraints])


angle_0_clamped = jnp.zeros(clamped_beam.n_ang_vel)
coenergy_vector_0_clamped = jnp.zeros(clamped_beam.n_coenergy)
state_vector_0_clamped = jnp.concatenate([angle_0_clamped, coenergy_vector_0_clamped])

# Run simulation for clamped beam
E_matrix_clamped = jax.scipy.linalg.block_diag(jnp.eye(clamped_beam.n_ang_vel), H_matrix_clamped)
trajectory_clamped = simulate(state_vector_0_clamped, dt, n_steps, E_matrix_clamped, vector_field_clamped)

theta_sol_clamped = trajectory_clamped[:clamped_beam.n_ang_vel, :]
coenergy_sol_clamped = trajectory_clamped[clamped_beam.n_ang_vel:, :]

energy_vec_clamped, kinetic_energy_vec_clamped, deformation_energy_vec_clamped = \
    clamped_beam.compute_energies(coenergy_sol_clamped)

# # Compatible lagrange mutipliers are found by solving
# # (B^T M^{-1} B) lambda_0 = -B^T M^{-1} (J(e_0)e_0 + f_0)
# # Since e_0 = 0 and f_0 = 0 => lambda_0 = 0

angle_0_free = jnp.zeros(free_beam.n_ang_vel)
coenergy_vector_0_free = jnp.zeros(free_beam.n_coenergy)
lagrange_multiplier_0 = jnp.zeros(6)
state_vector_0_free = jnp.concatenate([angle_0_free, coenergy_vector_0_free, lagrange_multiplier_0])

# Run simulation for free beam
E_matrix_free = jax.scipy.linalg.block_diag(jnp.eye(free_beam.n_ang_vel), \
                                            H_matrix_free, jnp.zeros((n_lambda, n_lambda)))

trajectory_free = simulate(state_vector_0_free, dt, n_steps, E_matrix_free, vector_field_free)

theta_sol_free = trajectory_free[:free_beam.n_ang_vel, :]
coenergy_sol_free = trajectory_free[free_beam.n_ang_vel:-n_lambda, :]
lagrange_sol_free = trajectory_free[-n_lambda:, :]

energy_vec_free, kinetic_energy_vec_free, deformation_energy_vec_free = \
    free_beam.compute_energies(coenergy_sol_free)
# # Plot of energy trend
plt.figure()
plt.plot(dt* jnp.arange(n_steps+1), energy_vec_clamped, label='H CC')
plt.plot(dt* jnp.arange(n_steps+1), kinetic_energy_vec_clamped, label='T CC')
plt.plot(dt* jnp.arange(n_steps+1), deformation_energy_vec_clamped, label='V CC')
plt.plot(dt* jnp.arange(n_steps+1), energy_vec_free, label='H FF')
plt.plot(dt* jnp.arange(n_steps+1), kinetic_energy_vec_free, label='T FF')
plt.plot(dt* jnp.arange(n_steps+1), deformation_energy_vec_free, label='V FF')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy vs Time')
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")

# Plot of energy difference
plt.figure()
plt.plot(dt* jnp.arange(n_steps), jnp.abs(jnp.diff(energy_vec_clamped)), 'r', label='CC')
plt.plot(dt* jnp.arange(n_steps), jnp.abs(jnp.diff(energy_vec_free)), 'b', label='FF')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('$|H(\mathbf{e}_{n+1}) - H(\mathbf{e}_{n})|$')
plt.title('Delta Energy vs Time')
plt.grid(True)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(folder_results + f"DeltaEnergy.pdf", format="pdf")


lin_velocity_sol_clamped = coenergy_sol_clamped[:clamped_beam.n_lin_vel, :]
midpoints = 0.5 * (clamped_beam.x_nodes[:-1] + clamped_beam.x_nodes[1:])
initial_position_clamped = jnp.stack([midpoints,
                              jnp.zeros_like(midpoints)], axis=1)
position_sol_clamped = compute_beam_positions(theta_sol_clamped, lin_velocity_sol_clamped, initial_position_clamped, time_vec)

lin_velocity_sol_free = coenergy_sol_free[:free_beam.n_lin_vel, :]
initial_position_free = jnp.stack([free_beam.x_nodes,
                                jnp.zeros_like(free_beam.x_nodes)], axis=1)

position_sol_free = compute_beam_positions(theta_sol_free, lin_velocity_sol_free, initial_position_free, time_vec)

# Create and display animation
from src.postprocessing.animate import animate_overlay_beam_deformation
print("Creating animation...")
anim = animate_overlay_beam_deformation(
    time_vec, [position_sol_clamped, position_sol_free],
    fps=20, interval=100,
    title="Comparison: Clamped Beam under distributed Load",
    colors=('blue', 'red'),
    labels=('Clamped', 'Free'),
    save_path=folder_results + f"Clamped_vs_Free.gif"
)

# Show the animation
plt.show()
