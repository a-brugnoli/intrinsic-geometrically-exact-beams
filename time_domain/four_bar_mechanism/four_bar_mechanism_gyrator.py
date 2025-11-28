from mimetypes import init
from src.models.intrinsic_pinned_beam import IntrinsicPlanarPinnedBeam
from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from data import material_props, material_props_2, \
    geometrical_props_1, geometrical_props_2, geometrical_props_3, \
    theta_1_0, theta_2_0, theta_3_0, \
    n_elem_1, n_elem_2, n_elem_3, \
    point_A_0, point_B_0, point_C_0, \
    g, A, rho, dt, T_end
import jax, diffrax, os
import numpy as np
import jax.numpy as jnp
from src.time_integration.utilities import simulate, simulate_splitting
from src.rotations.utilities import rotation_matrix_2d
from src.fem.forcing import assemble_state_dependent_forcing_vector_cg, \
    assemble_state_dependent_forcing_vector_dg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.potential_energy import compute_potential_energy_nodes, \
                                                compute_potential_energy_midpoints          
from src.postprocessing.animate import animate_overlay_beam_deformation
from src.postprocessing.postprocess_variables import compute_beam_positions, compute_consistent_beam_positions
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()

path_file = os.path.dirname(os.path.abspath(__file__))

E_2 = material_props_2['E']
if E_2 > 2.1e10:
    folder_results = path_file + "/results/gyrator_stiff/"
else:
    folder_results = path_file + "/results/gyrator_soft/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

beam_1 = IntrinsicPlanarPinnedBeam(n_elem_1, geometrical_props_1, material_props)
beam_2 = IntrinsicPlanarFreeBeam(n_elem_2, geometrical_props_2, material_props_2)
beam_3 = IntrinsicPlanarPinnedBeam(n_elem_3, geometrical_props_3, material_props)

# Matrices for the pinned beam 1
H_matrix_1 = beam_1.assemble_energy_matrix()
J_linear_matrix_1 = beam_1.assemble_linear_interconnection_matrix()
B_matrix_1_at_L = beam_1.assemble_control_matrix()[:, 3:]
B_1_velocity_at_L = B_matrix_1_at_L[:, 1:]

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
B_3_velocity_at_0 = B_matrix_3_at_0[:, 1:]

# Overall matrices
H_matrix = jax.scipy.linalg.block_diag(H_matrix_1, H_matrix_2, H_matrix_3)
H_matrix_inv = jnp.linalg.inv(H_matrix)

E_matrix = jax.scipy.linalg.block_diag(jnp.eye(beam_1.n_ang_vel), 
                                       jnp.eye(beam_2.n_ang_vel), 
                                       jnp.eye(beam_3.n_ang_vel), 
                                       H_matrix)

ind_theta_2 = beam_1.n_ang_vel
ind_theta_3 = beam_1.n_ang_vel + beam_2.n_ang_vel

ind_coenergy_1 = ind_theta_3 + beam_3.n_ang_vel
ind_coenergy_2 = ind_coenergy_1 + beam_1.n_coenergy
ind_coenergy_3 = ind_coenergy_2 + beam_2.n_coenergy
ind_coenergy_end = ind_coenergy_3 + beam_3.n_coenergy


jax.jit
def gravity_force(t):
    return jnp.array([0, -g*rho*A])
    # t_end_loading = 10
    # return jnp.where(t <= t_end_loading, 
    #                  jnp.array([0, -t/t_end_loading*g*rho*A]), 
    #                  jnp.array([0, -g*rho*A]))
    

jax.jit
def exp_f_material_x(t):
    return lambda angle: + jnp.cos(angle) * gravity_force(t)[0] \
                         + jnp.sin(angle) * gravity_force(t)[1]

jax.jit
def exp_f_material_y(t):
    return lambda angle: - jnp.sin(angle) * gravity_force(t)[0] \
                         + jnp.cos(angle) * gravity_force(t)[1]   


jax.jit
def intrinsic_dynamics(t, state_vector):
    theta_vector_1 = state_vector[:beam_1.n_ang_vel]
    theta_vector_2 = state_vector[ind_theta_2:ind_theta_3]
    theta_vector_3 = state_vector[ind_theta_3:ind_coenergy_1]

    coenergy_vector_1 = state_vector[ind_coenergy_1:ind_coenergy_2]
    coenergy_vector_2 = state_vector[ind_coenergy_2:ind_coenergy_3]
    coenergy_vector_3 = state_vector[ind_coenergy_3:ind_coenergy_end]

    vector_field_theta_1 = coenergy_vector_1[beam_1.n_lin_vel:beam_1.n_vel]
    vector_field_theta_2 = coenergy_vector_2[beam_2.n_lin_vel:beam_2.n_vel]
    vector_field_theta_3 = coenergy_vector_3[beam_3.n_lin_vel:beam_3.n_vel]

    vector_field_theta = jnp.concatenate([vector_field_theta_1, \
                                          vector_field_theta_2,   \
                                          vector_field_theta_3])
    
    gravity_vector_1 = jnp.zeros((beam_1.n_coenergy, ))
    f_vector_x_1 = assemble_state_dependent_forcing_vector_dg(beam_1.x_nodes,\
                                                            theta_vector_1, exp_f_material_x(t))    
    f_vector_y_1 = assemble_state_dependent_forcing_vector_dg(beam_1.x_nodes,\
                                                            theta_vector_1, exp_f_material_y(t))
    
    gravity_vector_1 = gravity_vector_1.at[:beam_1.n_lin_vel_x].set(f_vector_x_1)
    gravity_vector_1 = gravity_vector_1.at[beam_1.n_lin_vel_x:beam_1.n_lin_vel].set(f_vector_y_1)

    vector_field_coenergy_1 = J_linear_matrix_1 @ coenergy_vector_1 \
        + beam_1.assemble_nonlinear_dynamics(coenergy_vector_1) \
        + gravity_vector_1 
    
    gravity_vector_2 = jnp.zeros((beam_2.n_coenergy, ))    
    f_vector_x_2 = assemble_state_dependent_forcing_vector_cg(beam_2.x_nodes,\
                                                            theta_vector_2, exp_f_material_x(t))    
    f_vector_y_2 = assemble_state_dependent_forcing_vector_cg(beam_2.x_nodes,\
                                                            theta_vector_2, exp_f_material_y(t))
    gravity_vector_2 = gravity_vector_2.at[:beam_2.n_lin_vel_x].set(f_vector_x_2)
    gravity_vector_2 = gravity_vector_2.at[beam_2.n_lin_vel_x:beam_2.n_lin_vel].set(f_vector_y_2)

    vector_field_coenergy_2 = J_linear_matrix_2 @ coenergy_vector_2 \
        + beam_2.assemble_nonlinear_dynamics(coenergy_vector_2) \
        + gravity_vector_2 

    gravity_vector_3 = jnp.zeros((beam_3.n_coenergy, ))
    f_vector_x_3 = assemble_state_dependent_forcing_vector_dg(beam_3.x_nodes,\
                                                            theta_vector_3, exp_f_material_x(t))    
    f_vector_y_3 = assemble_state_dependent_forcing_vector_dg(beam_3.x_nodes,\
                                                            theta_vector_3, exp_f_material_y(t))
    gravity_vector_3 = gravity_vector_3.at[:beam_3.n_lin_vel_x].set(f_vector_x_3)
    gravity_vector_3 = gravity_vector_3.at[beam_3.n_lin_vel_x:beam_3.n_lin_vel].set(f_vector_y_3)

    vector_field_coenergy_3 = J_linear_matrix_3 @ coenergy_vector_3 \
        + beam_3.assemble_nonlinear_dynamics(coenergy_vector_3) \
        + gravity_vector_3
    
    vector_field_coenergy = jnp.concatenate([vector_field_coenergy_1, \
                                            vector_field_coenergy_2,   \
                                            vector_field_coenergy_3])

    return jnp.concatenate([vector_field_theta, vector_field_coenergy])


jax.jit
def interconnection_dynamics(t, state_vector):
    theta_vector = state_vector[:ind_coenergy_1]

    theta_vector_1 = theta_vector[:beam_1.n_ang_vel]
    theta_vector_2 = theta_vector[ind_theta_2:ind_theta_3]
    theta_vector_3 = theta_vector[ind_theta_3:ind_coenergy_1]

    theta_1_at_L = theta_vector_1[-1]
    rot_matrix_1_at_L = rotation_matrix_2d(theta_1_at_L)

    theta_2_at_0 = theta_vector_2[0]
    rot_matrix_2_at_0 = rotation_matrix_2d(theta_2_at_0)
    theta_2_at_L = theta_vector_2[-1]
    rot_matrix_2_at_L = rotation_matrix_2d(theta_2_at_L)

    theta_3_at_0 = theta_vector_3[0]
    rot_matrix_3_at_0 = rotation_matrix_2d(theta_3_at_0)

    coenergy_vector_1 = state_vector[ind_coenergy_1:ind_coenergy_2]
    coenergy_vector_2 = state_vector[ind_coenergy_2:ind_coenergy_3]
    coenergy_vector_3 = state_vector[ind_coenergy_3:]

    y_1_at_L_spatial = rot_matrix_1_at_L @ B_1_velocity_at_L.T @ coenergy_vector_1
    y_2_at_0_spatial = rot_matrix_2_at_0 @ B_2_force_at_0.T @ coenergy_vector_2
    y_2_at_L_spatial = rot_matrix_2_at_L @ B_2_force_at_L.T @ coenergy_vector_2
    y_3_at_0_spatial = rot_matrix_3_at_0 @ B_3_velocity_at_0.T @ coenergy_vector_3

    u_1_at_L_spatial = + y_2_at_0_spatial 
    u_2_at_0_spatial = - y_1_at_L_spatial 
    u_2_at_L_spatial = - y_3_at_0_spatial 
    u_3_at_0_spatial = + y_2_at_L_spatial 

    vector_field_interconnection_1 = + B_1_velocity_at_L @ rot_matrix_1_at_L.T @ u_1_at_L_spatial 
    vector_field_interconnection_2 = + B_2_force_at_0 @ rot_matrix_2_at_0.T @ u_2_at_0_spatial \
                              + B_2_force_at_L @ rot_matrix_2_at_L.T @ u_2_at_L_spatial 

    vector_field_interconnection_3 = + B_3_velocity_at_0 @ rot_matrix_3_at_0.T @ u_3_at_0_spatial
    
    vector_field_interconnection = jnp.concatenate([vector_field_interconnection_1, \
                                             vector_field_interconnection_2,   \
                                             vector_field_interconnection_3])
    
    return jnp.concatenate([jnp.zeros_like(theta_vector), vector_field_interconnection])


jax.jit
def vector_field(t, state_vector, args=None):
    return intrinsic_dynamics(t, state_vector) + interconnection_dynamics(t, state_vector)


#jax.jit
def vector_field_ode(t, state_vector, args=None):
    jax.debug.print("Progress = {perc:.2f} %", perc=t*100/T_end)
    theta_vector_1 = state_vector[:beam_1.n_ang_vel]
    theta_vector_2 = state_vector[ind_theta_2:ind_theta_3]
    theta_vector_3 = state_vector[ind_theta_3:ind_coenergy_1]

    theta_1_at_L = theta_vector_1[-1]
    rot_matrix_1_at_L = rotation_matrix_2d(theta_1_at_L)

    theta_2_at_0 = theta_vector_2[0]
    rot_matrix_2_at_0 = rotation_matrix_2d(theta_2_at_0)
    theta_2_at_L = theta_vector_2[-1]
    rot_matrix_2_at_L = rotation_matrix_2d(theta_2_at_L)

    theta_3_at_0 = theta_vector_3[0]
    rot_matrix_3_at_0 = rotation_matrix_2d(theta_3_at_0)

    coenergy_vector_1 = state_vector[ind_coenergy_1:ind_coenergy_2]
    coenergy_vector_2 = state_vector[ind_coenergy_2:ind_coenergy_3]
    coenergy_vector_3 = state_vector[ind_coenergy_3:ind_coenergy_end]

    vector_field_theta_1 = coenergy_vector_1[beam_1.n_lin_vel:beam_1.n_vel]
    vector_field_theta_2 = coenergy_vector_2[beam_2.n_lin_vel:beam_2.n_vel]
    vector_field_theta_3 = coenergy_vector_3[beam_3.n_lin_vel:beam_3.n_vel]

    vector_field_theta = jnp.concatenate([vector_field_theta_1, \
                                          vector_field_theta_2,   \
                                          vector_field_theta_3])
    
    y_1_at_L_spatial = rot_matrix_1_at_L @ B_1_velocity_at_L.T @ coenergy_vector_1
    y_2_at_0_spatial = rot_matrix_2_at_0 @ B_2_force_at_0.T @ coenergy_vector_2
    y_2_at_L_spatial = rot_matrix_2_at_L @ B_2_force_at_L.T @ coenergy_vector_2
    y_3_at_0_spatial = rot_matrix_3_at_0 @ B_3_velocity_at_0.T @ coenergy_vector_3

    u_1_at_L_spatial = + y_2_at_0_spatial 
    u_2_at_0_spatial = - y_1_at_L_spatial 
    u_2_at_L_spatial = - y_3_at_0_spatial 
    u_3_at_0_spatial = + y_2_at_L_spatial 

    gravity_vector_1 = jnp.zeros((beam_1.n_coenergy, ))
    f_vector_x_1 = assemble_state_dependent_forcing_vector_dg(beam_1.x_nodes,\
                                                            theta_vector_1, exp_f_material_x(t))    
    f_vector_y_1 = assemble_state_dependent_forcing_vector_dg(beam_1.x_nodes,\
                                                            theta_vector_1, exp_f_material_y(t))
    
    gravity_vector_1 = gravity_vector_1.at[:beam_1.n_lin_vel_x].set(f_vector_x_1)
    gravity_vector_1 = gravity_vector_1.at[beam_1.n_lin_vel_x:beam_1.n_lin_vel].set(f_vector_y_1)


    vector_field_coenergy_1 = J_linear_matrix_1 @ coenergy_vector_1 \
        + beam_1.assemble_nonlinear_dynamics(coenergy_vector_1) \
        + B_1_velocity_at_L @ rot_matrix_1_at_L.T @ u_1_at_L_spatial \
        + gravity_vector_1 
    
    gravity_vector_2 = jnp.zeros((beam_2.n_coenergy, ))    
    f_vector_x_2 = assemble_state_dependent_forcing_vector_cg(beam_2.x_nodes,\
                                                            theta_vector_2, exp_f_material_x(t))    
    f_vector_y_2 = assemble_state_dependent_forcing_vector_cg(beam_2.x_nodes,\
                                                            theta_vector_2, exp_f_material_y(t))
    gravity_vector_2 = gravity_vector_2.at[:beam_2.n_lin_vel_x].set(f_vector_x_2)
    gravity_vector_2 = gravity_vector_2.at[beam_2.n_lin_vel_x:beam_2.n_lin_vel].set(f_vector_y_2)

    vector_field_coenergy_2 = J_linear_matrix_2 @ coenergy_vector_2 \
        + beam_2.assemble_nonlinear_dynamics(coenergy_vector_2) \
        + B_2_force_at_0 @ rot_matrix_2_at_0.T @ u_2_at_0_spatial \
        + B_2_force_at_L @ rot_matrix_2_at_L.T @ u_2_at_L_spatial \
        + gravity_vector_2 

    gravity_vector_3 = jnp.zeros((beam_3.n_coenergy, ))
    f_vector_x_3 = assemble_state_dependent_forcing_vector_dg(beam_3.x_nodes,\
                                                            theta_vector_3, exp_f_material_x(t))    
    f_vector_y_3 = assemble_state_dependent_forcing_vector_dg(beam_3.x_nodes,\
                                                            theta_vector_3, exp_f_material_y(t))
    gravity_vector_3 = gravity_vector_3.at[:beam_3.n_lin_vel_x].set(f_vector_x_3)
    gravity_vector_3 = gravity_vector_3.at[beam_3.n_lin_vel_x:beam_3.n_lin_vel].set(f_vector_y_3)

    vector_field_coenergy_3 = J_linear_matrix_3 @ coenergy_vector_3 \
        + beam_3.assemble_nonlinear_dynamics(coenergy_vector_3) \
        + B_3_velocity_at_0 @ rot_matrix_3_at_0.T @ u_3_at_0_spatial \
        + gravity_vector_3
    
    vector_field_coenergy = jnp.concatenate([vector_field_coenergy_1, \
                                            vector_field_coenergy_2,   \
                                            vector_field_coenergy_3])
    
    ode_vector_field_coenergy = H_matrix_inv @ vector_field_coenergy

    return jnp.concatenate([vector_field_theta, ode_vector_field_coenergy])


angle_beam_1_0 = theta_1_0 * jnp.ones(beam_1.n_ang_vel)
angle_beam_2_0 = theta_2_0 * jnp.ones(beam_2.n_ang_vel)
angle_beam_3_0 = theta_3_0 * jnp.ones(beam_3.n_ang_vel)

coenergy_vector_beam_1_0 = jnp.zeros(beam_1.n_coenergy)
coenergy_vector_beam_2_0 = jnp.zeros(beam_2.n_coenergy)
coenergy_vector_beam_3_0 = jnp.zeros(beam_3.n_coenergy)

state_vector_0 = jnp.concatenate([angle_beam_1_0, 
                                  angle_beam_2_0,
                                  angle_beam_3_0,
                                  coenergy_vector_beam_1_0,
                                  coenergy_vector_beam_2_0,
                                  coenergy_vector_beam_3_0])

n_steps = int(T_end / dt)
time_vec_all = dt * jnp.arange(n_steps + 1)

# term = diffrax.ODETerm(vector_field_ode)
# saveat = diffrax.SaveAt(ts=time_vec_all)
# solver = diffrax.Kvaerno5()  
# stepsize_controller = diffrax.PIDController(rtol=1e-9, atol=1e-12)
# solution = diffrax.diffeqsolve(
#     term,
#     solver=solver,
#     t0=0,
#     t1=T_end,
#     dt0=0.001,          # initial guess for step size
#     y0=state_vector_0,
#     saveat=saveat,
#     stepsize_controller=stepsize_controller,
#     max_steps=1000000,
# )
# trajectory = solution.ys.T

trajectory = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field)
# trajectory = simulate_splitting(state_vector_0, dt, n_steps, E_matrix, \
#                                 intrinsic_dynamics, interconnection_dynamics)

n_plot = 500
if n_plot<=n_steps:
    step_plot = n_steps//n_plot
else:
    step_plot=1
time_vec = time_vec_all[::step_plot]

trajectory_plot = trajectory[:, ::step_plot]

np.save(folder_results + "time_vec.npy", np.array(time_vec))
np.save(folder_results + "trajectory_plot.npy", np.array(trajectory_plot))

theta_1_sol = trajectory_plot[:beam_1.n_ang_vel]
theta_2_sol = trajectory_plot[ind_theta_2:ind_theta_3]
theta_3_sol = trajectory_plot[ind_theta_3:ind_coenergy_1]

coenergy_1_sol = trajectory_plot[ind_coenergy_1:ind_coenergy_2]
coenergy_2_sol = trajectory_plot[ind_coenergy_2:ind_coenergy_3]
coenergy_3_sol = trajectory_plot[ind_coenergy_3:ind_coenergy_end]

energy_vec, kinetic_energy_vec, deformation_energy_vec = \
    beam_1.compute_energies(coenergy_1_sol)

lin_velocity_1_sol = coenergy_1_sol[:beam_1.n_lin_vel]
lin_velocity_2_sol = coenergy_2_sol[:beam_2.n_lin_vel]
lin_velocity_3_sol = coenergy_3_sol[:beam_3.n_lin_vel]

x_A, y_A = point_A_0
x_points_1_t0 = x_A + beam_1.x_nodes * jnp.cos(theta_1_sol[:, 0])
y_points_1_t0 = y_A + beam_1.x_nodes * jnp.sin(theta_1_sol[:, 0])

x_midpoints_1_t0 = 0.5*(x_points_1_t0[1:] + x_points_1_t0[:-1])
y_midpoints_1_t0 = 0.5*(y_points_1_t0[1:] + y_points_1_t0[:-1])

x_B, y_B = point_B_0
x_points_2_t0 = x_B + beam_2.x_nodes * jnp.cos(theta_2_sol[:, 0])
y_points_2_t0 = y_B + beam_2.x_nodes * jnp.sin(theta_2_sol[:, 0])

x_C, y_C = point_C_0
x_points_3_t0 = x_C + beam_3.x_nodes * jnp.cos(theta_3_sol[:, 0])
y_points_3_t0 = y_C + beam_3.x_nodes * jnp.sin(theta_3_sol[:, 0])

x_midpoints_3_t0 = 0.5*(x_points_3_t0[1:] + x_points_3_t0[:-1])
y_midpoints_3_t0 = 0.5*(y_points_3_t0[1:] + y_points_3_t0[:-1])


# positions_1_t0 = jnp.stack([x_midpoints_1_t0,
#                             y_midpoints_1_t0], axis=1)

# position_1_sol = compute_beam_positions(theta_1_sol, 
#                                         lin_velocity_1_sol, 
#                                         positions_1_t0, time_vec)

# positions_2_t0 = jnp.stack([x_points_2_t0,
#                             y_points_2_t0], axis=1)
# position_2_sol = compute_beam_positions(theta_2_sol, 
#                                         lin_velocity_2_sol, 
#                                         positions_2_t0, time_vec)

# positions_3_t0 = jnp.stack([x_midpoints_3_t0,
#                                 y_midpoints_3_t0], axis=1)
# position_3_sol = compute_beam_positions(theta_3_sol, 
#                                         lin_velocity_3_sol, 
#                                         positions_3_t0, time_vec)


positions_1_t0 = jnp.concatenate([x_midpoints_1_t0,
                             y_midpoints_1_t0])
position_1_sol = compute_consistent_beam_positions(beam_1,
                                        theta_1_sol, 
                                        lin_velocity_1_sol, 
                                        positions_1_t0, time_vec)

positions_2_t0 = jnp.concatenate([x_points_2_t0,
                            y_points_2_t0])
position_2_sol = compute_consistent_beam_positions(beam_2,
                                        theta_2_sol, 
                                        lin_velocity_2_sol, 
                                        positions_2_t0, time_vec)

positions_3_t0 = jnp.concatenate([x_midpoints_3_t0,
                                y_midpoints_3_t0])
position_3_sol = compute_consistent_beam_positions(beam_3,
                                        theta_3_sol, 
                                        lin_velocity_3_sol, 
                                        positions_3_t0, time_vec)


energy_1_vec, kinetic_energy_1_vec, deformation_energy_1_vec = \
    beam_1.compute_energies(coenergy_1_sol)

energy_2_vec, kinetic_energy_2_vec, deformation_energy_2_vec = \
    beam_2.compute_energies(coenergy_2_sol)

energy_3_vec, kinetic_energy_3_vec, deformation_energy_3_vec = \
    beam_3.compute_energies(coenergy_3_sol)

potential_energy_1_vec = compute_potential_energy_midpoints(position_1_sol, beam_1.x_nodes, rho, A)
potential_energy_2_vec = compute_potential_energy_nodes(position_2_sol, beam_2.x_nodes, rho, A)
potential_energy_3_vec = compute_potential_energy_midpoints(position_3_sol, beam_3.x_nodes, rho, A)

total_intrinsic_energy = energy_1_vec + energy_2_vec + energy_3_vec
total_potential_energy = potential_energy_1_vec + potential_energy_2_vec + potential_energy_3_vec
total_energy = total_intrinsic_energy + total_potential_energy
plt.figure()
plt.plot(time_vec, total_intrinsic_energy, 'b--', label='Intrinsic Energy')
plt.plot(time_vec, total_potential_energy, 'r--', label='Potential Energy')
plt.plot(time_vec, total_energy, 'k', label='Total Energy')
plt.xlabel('Time (s)')
plt.title('Energies of the four bar mechanism')
plt.grid(True)
plt.legend()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")
plt.tight_layout()

plt.figure()
plt.plot(time_vec[:-1], jnp.diff(total_energy), 'b--')
plt.xlabel('Time (s)')
plt.title('Delta Energy of the four bar mechanism')
plt.grid(True)
plt.savefig(folder_results + f"EnergyRate.pdf", format="pdf")
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


# # Middle point of beam 2
# node_mid_2 = int(n_elem_2/2)
# r_mid_2 = position_2_sol[:, node_mid_2, :]

# plt.figure()
# plt.plot(time_vec, r_mid_2[0])
# plt.xlabel('Time (s)')
# plt.ylabel('$r_x^P$')
# plt.grid(True)
# plt.savefig(folder_results + f"r_x_P.pdf", format="pdf")
# plt.tight_layout()

# plt.figure()
# plt.plot(time_vec, r_mid_2[1], 'r', label='1')
# plt.xlabel('Time (s)')
# plt.ylabel('$r_y^P$')
# plt.grid(True)
# plt.savefig(folder_results + f"r_y_P.pdf", format="pdf")
# plt.tight_layout()

animation = animate_overlay_beam_deformation(time_vec, 
                                                [position_1_sol,
                                                position_2_sol,
                                                position_3_sol],
                                    title="Four bar mechanism",
                                    labels=('Beam 1', 'Beam 2', 'Beam 3'),
                                    save_path=folder_results + f"Four_bar_mech.gif",
                                    x_lim=[-0.1, 4],
                                    y_lim=[-2.5, 2.5])

plt.show()
