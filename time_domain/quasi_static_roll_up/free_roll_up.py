from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from data import geometrical_props, material_props
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.postprocess_variables import compute_positions
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()

import os
from jax import config; config.update("jax_enable_x64", True)

dt = 0.2
T_end = 1
n_steps = int(T_end/dt)
time_vec = dt * jnp.arange(n_steps + 1)

# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

n_elements = 8  # Number of finite elements
free_beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)

H_matrix = free_beam.assemble_energy_matrix()
J_linear_matrix = free_beam.assemble_linear_interconnection_matrix()

B_matrix_at_0 = free_beam.assemble_control_matrix()[:, :3]
n_lambda = B_matrix_at_0.shape[1]

B_torque_at_L = free_beam.assemble_control_matrix()[:, -1]


jax.jit
def spatial_torque(t):
    m_roll_up = 100*jnp.pi  # Roll-up moment
    return t*m_roll_up


jax.jit
def vector_field(state_vector, t):
    theta_vector = state_vector[:free_beam.n_ang_vel]
    coenergy_vector = state_vector[free_beam.n_ang_vel:-n_lambda]
    lagrange_multiplier = state_vector[-n_lambda:]

    vector_field_theta = coenergy_vector[free_beam.n_lin_vel: free_beam.n_vel]
    
    vector_field_coenergy = J_linear_matrix @ coenergy_vector \
        + free_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + B_matrix_at_0 @ lagrange_multiplier \
        + B_torque_at_L * spatial_torque(t)

    vector_field_constraints = - B_matrix_at_0.T @ coenergy_vector


    return jnp.concatenate([vector_field_theta, vector_field_coenergy, vector_field_constraints])



angle_0 = jnp.zeros(free_beam.n_ang_vel)
coenergy_vector_0 = jnp.zeros(free_beam.n_coenergy)
lagrange_multiplier_0 = jnp.zeros(3)

state_vector_0 = jnp.concatenate([angle_0, coenergy_vector_0, lagrange_multiplier_0])

# Run simulation
E_matrix = jax.scipy.linalg.block_diag(jnp.eye(free_beam.n_ang_vel), \
                                            H_matrix, jnp.zeros((n_lambda, n_lambda)))

trajectory = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field)


theta_sol = trajectory[:free_beam.n_ang_vel, :]
coenergy_sol = trajectory[free_beam.n_ang_vel:-n_lambda, :]
lagrange_sol = trajectory[-n_lambda:, :]

lin_velocity_sol = coenergy_sol[:free_beam.n_lin_vel, :]
idx_moment = free_beam.n_lin_vel + free_beam.n_ang_vel + free_beam.n_force
moment_sol = coenergy_sol[idx_moment:, :]

initial_position = jnp.stack([free_beam.x_nodes,
                                jnp.zeros_like(free_beam.x_nodes)], axis=1)

position_sol = compute_positions(theta_sol, lin_velocity_sol, initial_position, time_vec)

# # Create and display animation
# from src.postprocessing.animate import animate_beam_deformation
# print("Creating animation...")
# anim = animate_beam_deformation(
#     time_vec, position_sol,
#     fps=20, interval=1000,
#     title="Static Roll-Up of a Cantilever Beam"
# )

# # Show the animation
# plt.show()


time_instants = [j*1/5 for j in range(6)]
time_indices = [j for j in range(n_steps+1) if j*dt in time_instants]

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

ax.set_xlim(-2.5, 10.5)
ax.set_ylim(-0.5, 8)
ax.set_aspect("equal", adjustable="box")

fig.tight_layout()
fig.savefig(folder_results + "Configurations.pdf", format="pdf")
plt.show()