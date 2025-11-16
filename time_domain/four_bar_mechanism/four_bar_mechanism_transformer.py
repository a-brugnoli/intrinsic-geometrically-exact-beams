from src.models.intrinsic_pinned_beam import IntrinsicPlanarPinnedBeam
from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from data import material_props, material_props_2, \
    geometrical_props_1, geometrical_props_2, geometrical_props_3, \
    theta_1_0, theta_2_0, theta_3_0, \
    n_elem_1, n_elem_2, n_elem_3, \
    point_A_0, point_B_0, point_C_0, g
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate, compute_lambda0
from src.rotations.utilities import rotation_matrix_2d
from src.fem.forcing import assemble_state_dependent_forcing_vector_cg, \
    assemble_state_dependent_forcing_vector_dg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.animate import animate_overlay_beam_deformation
from src.postprocessing.postprocess_variables import compute_beam_positions
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
import os

# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results_transformer_soft/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

beam_1 = IntrinsicPlanarFreeBeam(n_elem_1, geometrical_props_1, material_props)
beam_2 = IntrinsicPlanarFreeBeam(n_elem_2, geometrical_props_2, material_props_2)
beam_3 = IntrinsicPlanarFreeBeam(n_elem_3, geometrical_props_3, material_props)
beam_list = [beam_1, beam_2, beam_3]

# Matrices for the pinned beam 1
H_matrix_1 = beam_1.assemble_energy_matrix()
J_linear_matrix_1 = beam_1.assemble_linear_interconnection_matrix()
B_matrix_1_at_0 = beam_1.assemble_control_matrix()[:, :3]
B_1_force_at_0 = B_matrix_1_at_0[:, :2]
B_matrix_1_at_L = beam_1.assemble_control_matrix()[:, 3:]
B_1_force_at_L = B_matrix_1_at_L[:, :2]

# Matrices for the free beam 2
H_matrix_2 = beam_2.assemble_energy_matrix()
J_linear_matrix_2 = beam_2.assemble_linear_interconnection_matrix()
B_matrix_2 = beam_2.assemble_control_matrix()
B_matrix_2_at_0 = B_matrix_2[:, :3]
B_2_force_at_0 = B_matrix_2_at_0[:, :2]
B_matrix_2_at_L = B_matrix_2[:, 3:]
B_2_force_at_L = B_matrix_2_at_L[:, :2]

# Matrices for the pinned beam 3
H_matrix_3 = beam_3.assemble_energy_matrix()  
J_linear_matrix_3 = beam_3.assemble_linear_interconnection_matrix()
B_matrix_3_at_0 = beam_3.assemble_control_matrix()[:, :3]
B_3_force_at_0 = B_matrix_3_at_0[:, :2]
B_matrix_3_at_L = beam_3.assemble_control_matrix()[:, 3:]
B_3_force_at_L = B_matrix_3_at_L[:, :2]

# Overall matrices
n_lambda = 8
H_matrix = jax.scipy.linalg.block_diag(H_matrix_1, H_matrix_2, H_matrix_3)
A_matrix = jax.scipy.linalg.block_diag(jnp.eye(beam_1.n_ang_vel), 
                                       jnp.eye(beam_2.n_ang_vel), 
                                       jnp.eye(beam_3.n_ang_vel), 
                                       H_matrix)

E_matrix = jax.scipy.linalg.block_diag(A_matrix,
                                       jnp.zeros((n_lambda, n_lambda)))

gravity_force_1 = jnp.array([0, -g*geometrical_props_1['A']*material_props['rho']])
gravity_force_2 = jnp.array([0, -g*geometrical_props_2['A']*material_props['rho']])
gravity_force_3 = jnp.array([0, -g*geometrical_props_3['A']*material_props['rho']])

gravity_forces = [gravity_force_1, gravity_force_2, gravity_force_3]

ind_theta_2 = beam_1.n_ang_vel
ind_theta_3 = beam_1.n_ang_vel + beam_2.n_ang_vel

ind_coenergy_1 = ind_theta_3 + beam_3.n_ang_vel
ind_coenergy_2 = ind_coenergy_1 + beam_1.n_coenergy
ind_coenergy_3 = ind_coenergy_2 + beam_2.n_coenergy
ind_coenergy_end = ind_coenergy_3 + beam_3.n_coenergy


def exp_gravity_x(beam_index):
    return lambda angle: + jnp.cos(angle) * gravity_forces[beam_index][0] \
                         + jnp.sin(angle) * gravity_forces[beam_index][1]

def exp_gravity_y(beam_index):
    return lambda angle: - jnp.sin(angle) * gravity_forces[beam_index][0] \
                        + jnp.cos(angle) * gravity_forces[beam_index][1]   

def assemble_gravity_vector(index_beam, theta_vec):

    beam = beam_list[index_beam]
    f_gravity_x = assemble_state_dependent_forcing_vector_cg(beam.x_nodes, theta_vec, exp_gravity_x(index_beam))
    f_gravity_y = assemble_state_dependent_forcing_vector_cg(beam.x_nodes, theta_vec, exp_gravity_y(index_beam))

    gravity_vector = jnp.zeros((beam.n_coenergy, ))
    gravity_vector = gravity_vector.at[:beam.n_lin_vel_x].set(f_gravity_x)
    gravity_vector = gravity_vector.at[beam.n_lin_vel_x:beam.n_lin_vel].set(f_gravity_y)
    
    return gravity_vector


jax.jit
def vector_field_decomposed(x_vector, t):
    """
    Given the dynamics H xdot = f(x) + G(x)lambda
    Compute f(x), G(x) separately
    """
    # state_vector does not include multiplier
    theta_vector_1 = x_vector[:beam_1.n_ang_vel]
    theta_1_at_L = theta_vector_1[-1]
    rot_matrix_1_at_L = rotation_matrix_2d(theta_1_at_L)

    theta_vector_2 = x_vector[ind_theta_2:ind_theta_2 + beam_2.n_ang_vel]
    theta_2_at_0 = theta_vector_2[0]
    rot_matrix_2_at_0 = rotation_matrix_2d(theta_2_at_0)
    theta_2_at_L = theta_vector_2[-1]
    rot_matrix_2_at_L = rotation_matrix_2d(theta_2_at_L)

    theta_vector_3 = x_vector[ind_theta_3:ind_theta_3 + beam_3.n_ang_vel]
    theta_3_at_0 = theta_vector_3[0]
    rot_matrix_3_at_0 = rotation_matrix_2d(theta_3_at_0)

    coenergy_vector_1 = x_vector[ind_coenergy_1:ind_coenergy_2]
    coenergy_vector_2 = x_vector[ind_coenergy_2:ind_coenergy_3]
    coenergy_vector_3 = x_vector[ind_coenergy_3:ind_coenergy_end]

    vector_field_theta_1 = coenergy_vector_1[beam_1.n_lin_vel:beam_1.n_vel]
    vector_field_theta_2 = coenergy_vector_2[beam_2.n_lin_vel:beam_2.n_vel]
    vector_field_theta_3 = coenergy_vector_3[beam_3.n_lin_vel:beam_3.n_vel]

    vector_field_theta = jnp.concatenate([vector_field_theta_1, \
                                          vector_field_theta_2,   \
                                          vector_field_theta_3])

    gravity_vector_1 = assemble_gravity_vector(0, theta_vector_1)
    gravity_vector_2 = assemble_gravity_vector(1, theta_vector_2)
    gravity_vector_3 = assemble_gravity_vector(2, theta_vector_3)
    
    dynamics_coenergy_1 = J_linear_matrix_1 @ coenergy_vector_1 \
        + beam_1.assemble_nonlinear_dynamics(coenergy_vector_1) \
        + gravity_vector_1 
    
    G_matrix_1 = jnp.hstack([B_1_force_at_0, 
                            B_1_force_at_L @ rot_matrix_1_at_L.T]) 
    
    dynamics_coenergy_2 = J_linear_matrix_2 @ coenergy_vector_2 \
        + beam_2.assemble_nonlinear_dynamics(coenergy_vector_2) \
        + gravity_vector_2 
    
    G_matrix_2 = jnp.hstack([-B_2_force_at_0 @ rot_matrix_2_at_0.T,
                             -B_2_force_at_L @ rot_matrix_2_at_L.T])

    dynamics_coenergy_3 = J_linear_matrix_3 @ coenergy_vector_3 \
        + beam_3.assemble_nonlinear_dynamics(coenergy_vector_3) \
        + gravity_vector_3
    
    G_matrix_3 = jnp.hstack([B_3_force_at_0 @ rot_matrix_3_at_0.T,
                            B_3_force_at_L])
        
    
    dynamics_coenergy = jnp.concatenate([dynamics_coenergy_1, \
                                         dynamics_coenergy_2, \
                                         dynamics_coenergy_3])
    
    dynamics = jnp.concatenate([vector_field_theta, dynamics_coenergy])
    
    G_matrix = jnp.zeros((len(dynamics), n_lambda))
    G_matrix = G_matrix.at[ind_coenergy_1:ind_coenergy_2, :4].set(G_matrix_1)
    G_matrix = G_matrix.at[ind_coenergy_2:ind_coenergy_3, 2:6].set(G_matrix_2)
    G_matrix = G_matrix.at[ind_coenergy_3:ind_coenergy_end, 4:8].set(G_matrix_3)

    return dynamics, G_matrix

def vector_field(state_vector, t):
    x_vec = state_vector[:ind_coenergy_end]
    lambda_vec = state_vector[ind_coenergy_end:]

    f, G = vector_field_decomposed(x_vec, t)

    vector_field_state = f + G @ lambda_vec
    constraints = G.T @ x_vec

    return jnp.concatenate([vector_field_state, constraints])


angle_beam_1_0 = theta_1_0 * jnp.ones(beam_1.n_ang_vel)
angle_beam_2_0 = theta_2_0 * jnp.ones(beam_2.n_ang_vel)
angle_beam_3_0 = theta_3_0 * jnp.ones(beam_3.n_ang_vel)

coenergy_vector_beam_1_0 = jnp.zeros(beam_1.n_coenergy)
coenergy_vector_beam_2_0 = jnp.zeros(beam_2.n_coenergy)
coenergy_vector_beam_3_0 = jnp.zeros(beam_3.n_coenergy)

x_0 = jnp.concatenate([angle_beam_1_0, 
                    angle_beam_2_0,
                    angle_beam_3_0,
                    coenergy_vector_beam_1_0,
                    coenergy_vector_beam_2_0,
                    coenergy_vector_beam_3_0])

lambda_0 =  compute_lambda0(A_matrix, vector_field_decomposed, x_0)


state_vector_0 = jnp.concatenate([x_0,
                                  lambda_0])

dt = 0.001
T_end = 1
n_steps = int(T_end/dt)
time_vec_all = dt * jnp.arange(n_steps + 1)

trajectory = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field)

n_plot = 500
if n_plot<n_steps:
    step_plot = n_steps//n_plot
else:
    step_plot=1
time_vec = time_vec_all[::step_plot]
theta_1_sol = trajectory[:beam_1.n_ang_vel, ::step_plot]
theta_2_sol = trajectory[ind_theta_2:ind_theta_2 + beam_2.n_ang_vel, ::step_plot]
theta_3_sol = trajectory[ind_theta_3:ind_theta_3 + beam_3.n_ang_vel, ::step_plot]

coenergy_1_sol = trajectory[ind_coenergy_1:ind_coenergy_2, ::step_plot]
coenergy_2_sol = trajectory[ind_coenergy_2:ind_coenergy_3, ::step_plot]
coenergy_3_sol = trajectory[ind_coenergy_3:ind_coenergy_end, ::step_plot]

lin_velocity_1_sol = coenergy_1_sol[:beam_1.n_lin_vel, ::step_plot]
lin_velocity_2_sol = coenergy_2_sol[:beam_2.n_lin_vel, ::step_plot]
lin_velocity_3_sol = coenergy_3_sol[:beam_3.n_lin_vel, ::step_plot]

x_A, y_A = point_A_0
position_nodes_1_x = x_A + beam_1.x_nodes * jnp.cos(theta_1_sol[:, 0])
position_nodes_1_y = y_A + beam_1.x_nodes * jnp.sin(theta_1_sol[:, 0])
initial_position_1 = jnp.stack([position_nodes_1_x,
                                position_nodes_1_y], axis=1)

position_1_sol = compute_beam_positions(theta_1_sol, 
                                        lin_velocity_1_sol, 
                                        initial_position_1, time_vec)

x_B, y_B = point_B_0
position_nodes_2_x = x_B + beam_2.x_nodes * jnp.cos(theta_2_sol[:, 0])
position_nodes_2_y = y_B + beam_2.x_nodes * jnp.sin(theta_2_sol[:, 0])

initial_position_2 = jnp.stack([position_nodes_2_x,
                                position_nodes_2_y], axis=1)

position_2_sol = compute_beam_positions(theta_2_sol, 
                                        lin_velocity_2_sol, 
                                        initial_position_2, time_vec)

x_C, y_C = point_C_0
position_nodes_3_x = x_C + beam_3.x_nodes * jnp.cos(theta_3_sol[:, 0])
position_nodes_3_y = y_C + beam_3.x_nodes * jnp.sin(theta_3_sol[:, 0])
initial_position_3 = jnp.stack([position_nodes_3_x,
                                position_nodes_3_y], axis=1)

position_3_sol = compute_beam_positions(theta_3_sol, 
                                        lin_velocity_3_sol, 
                                        initial_position_3, time_vec)

# Middle point of beam 2
node_mid_2 = int(n_elem_2/2)
r_mid_2 = position_2_sol[:, node_mid_2, :]

plt.figure()
plt.plot(time_vec, r_mid_2[0])
plt.xlabel('Time (s)')
plt.ylabel('$r_x^P$')
plt.grid(True)
plt.savefig(folder_results + f"r_x_P.pdf", format="pdf")
plt.tight_layout()

plt.figure()
plt.plot(time_vec, r_mid_2[1], 'r', label='1')
plt.xlabel('Time (s)')
plt.ylabel('$r_y^P$')
plt.grid(True)
plt.savefig(folder_results + f"r_y_P.pdf", format="pdf")
plt.tight_layout()


time_instants = jnp.array([j*2 for j in range(6)])
time_indices = [j for j in range(len(time_vec)) if time_vec[j] in time_instants]

norm = Normalize(vmin=jnp.min(time_instants), vmax=jnp.max(time_instants))

fig, ax = plt.subplots()
for j in time_indices:

    x_nodes_1 = position_1_sol[0, :, j]
    y_nodes_1 = position_1_sol[1, :, j]

    x_nodes_2 = position_2_sol[0, :, j]
    y_nodes_2 = position_2_sol[1, :, j]

    x_nodes_3 = position_3_sol[0, :, j]
    y_nodes_3 = position_3_sol[1, :, j]
    
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


ax.set_xlim(-0.1, 3.5)
ax.set_ylim(-2.1, 1.5)

fig.tight_layout()
fig.savefig(folder_results + "Configurations.pdf", format="pdf")

animation = animate_overlay_beam_deformation(time_vec, 
                                                [position_1_sol,
                                                position_2_sol,
                                                position_3_sol],
                                    title="Four bar mechanism",
                                    labels=('Beam 1', 'Beam 2', 'Beam 3'),
                                    save_path=folder_results + f"Four_bar_mech.gif",
                                    x_lim=[-0.1, 4],
                                    y_lim=[-2.5, 1.5])


plt.show()
