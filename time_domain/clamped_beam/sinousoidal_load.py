from src.models.intrinsic_clamped_beam import IntrinsicPlanarClampedBeam
from src.material_data.unitary import geometrical_props, material_props
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d_batch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np  
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
import os
from jax import config; config.update("jax_enable_x64", True)



# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

n_elements = 50  # Number of finite elements
clamped_beam = IntrinsicPlanarClampedBeam(n_elements, geometrical_props, material_props)

free_beam = IntrinsicPlanarClampedBeam(n_elements, geometrical_props, material_props)


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
    # Transpose only the last two dimensions
    material_distributed_force = jnp.transpose(rot_matrix_batch, (0, 2, 1)) @ spatial_force(t)  # N_theta x 2

    # Force vector of the size of the system
    load_vector = jnp.zeros((clamped_beam.n_dofs, ))

    # Forcing along the x direction
    load_vector = load_vector.at[:clamped_beam.n_elem].set(\
        material_distributed_force[:, 0]*clamped_beam.h_mesh)
    
    # Forcing along the y direction
    load_vector = load_vector.at[clamped_beam.n_elem:clamped_beam.n_lin_vel].set(\
        material_distributed_force[:, 1]*clamped_beam.h_mesh)

    vector_field_theta = coenergy_vector[clamped_beam.n_lin_vel:clamped_beam.n_lin_vel + clamped_beam.n_ang_vel]

    vector_field_coenergy = J_linear_matrix_clamped @ coenergy_vector \
        + clamped_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + load_vector

    return jnp.concatenate([vector_field_theta, vector_field_coenergy])


angle_0 = jnp.zeros(clamped_beam.n_ang_vel)
coenergy_vector_0 = jnp.zeros(clamped_beam.n_dofs)
state_vector_0_clamped = jnp.concatenate([angle_0, coenergy_vector_0])

dt = 0.1
T_end = 15
n_steps = int(T_end/dt)
time_vec = dt * jnp.arange(n_steps + 1)

# Run simulation
E_matrix_clamped = jax.scipy.linalg.block_diag(jnp.eye(clamped_beam.n_ang_vel), H_matrix_clamped)
trajectory_clamped = simulate(state_vector_0_clamped, dt, n_steps, E_matrix_clamped, vector_field_clamped)

theta_sol_clamped = trajectory_clamped[:clamped_beam.n_ang_vel, :]
coenergy_sol_clamped = trajectory_clamped[clamped_beam.n_ang_vel:, :]

energy_vec_clamped, kinetic_energy_vec_clamped, deformation_energy_vec_clamped = \
    clamped_beam.compute_energies(coenergy_sol_clamped)

# Plot of energy trend
plt.figure()
plt.plot(dt* jnp.arange(n_steps+1), energy_vec_clamped, 'm', label='H')
plt.plot(dt* jnp.arange(n_steps+1), kinetic_energy_vec_clamped, 'g', label='T')
plt.plot(dt* jnp.arange(n_steps+1), deformation_energy_vec_clamped, 'b', label='V')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy vs Time')
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")

# Plot of energy difference
plt.figure()
plt.plot(dt* jnp.arange(n_steps), jnp.abs(jnp.diff(energy_vec_clamped)), 'm')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('$|H(\mathbf{e}_{n+1}) - H(\mathbf{e}_{n})|$')
plt.title('Delta Energy vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"DeltaEnergy.pdf", format="pdf")


lin_velocity_sol_clamped = coenergy_sol_clamped[:clamped_beam.n_lin_vel, :]

midpoints = 0.5 * (clamped_beam.x_nodes[:-1] + clamped_beam.x_nodes[1:])

initial_position_clamped = jnp.stack([midpoints,
                              jnp.zeros_like(midpoints)], axis=1)

position_sol_clamped = clamped_beam.compute_positions(theta_sol_clamped, lin_velocity_sol_clamped, initial_position_clamped, time_vec)

# Create and display animation
from src.postprocessing.animate import animate_beam_deformation
print("Creating animation...")
anim = animate_beam_deformation(
    time_vec, position_sol_clamped, 
    fps=20, 
    interval=100,
    title="Example: Vibrating Clamped Beam",
    save_path=None  # Set to "beam_animation.gif" to save
)

# Show the animation
plt.show()
