from data import point_A_0, point_B_0, point_C_0, node_mid_2, \
    beam_1, beam_2, beam_3, \
    ind_theta_2, ind_theta_3, \
    ind_coenergy_1, ind_coenergy_2, ind_coenergy_3, ind_coenergy_end
import jax.numpy as jnp


def compute_positions(trajectory, time_vec):

    theta_1_sol = trajectory[:beam_1.n_ang_vel]
    theta_2_sol = trajectory[ind_theta_2:ind_theta_3]
    theta_3_sol = trajectory[ind_theta_3:ind_coenergy_1]

    coenergy_1_sol = trajectory[ind_coenergy_1:ind_coenergy_2]
    coenergy_2_sol = trajectory[ind_coenergy_2:ind_coenergy_3]
    coenergy_3_sol = trajectory[ind_coenergy_3:ind_coenergy_end]

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

    positions_1_t0 = jnp.concatenate([x_midpoints_1_t0,
                                    y_midpoints_1_t0])
    position_1_sol = beam_1.compute_positions(theta_1_sol, 
                                                lin_velocity_1_sol, 
                                                positions_1_t0, time_vec)

    positions_2_t0 = jnp.concatenate([x_points_2_t0,
                                    y_points_2_t0])
    position_2_sol = beam_2.compute_positions(theta_2_sol, 
                                                lin_velocity_2_sol, 
                                                positions_2_t0, time_vec)

    positions_3_t0 = jnp.concatenate([x_midpoints_3_t0,
                                    y_midpoints_3_t0])
    position_3_sol = beam_3.compute_positions(theta_3_sol, 
                                                lin_velocity_3_sol, 
                                                positions_3_t0, time_vec)

    return position_1_sol, position_2_sol, position_3_sol