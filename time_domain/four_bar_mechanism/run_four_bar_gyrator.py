import jax
import jax.numpy as jnp
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d

from data import theta_1_0, theta_2_0, theta_3_0, \
    beam_1, beam_2, beam_3, \
    ind_theta_2, ind_theta_3, \
    ind_coenergy_1, ind_coenergy_2, ind_coenergy_3, ind_coenergy_end, \
    folder_results


def run_four_bar_gyrator(dt, T_end, folder_results, save_data=False):
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
    # H_matrix_inv = jnp.linalg.inv(H_matrix)

    E_matrix = jax.scipy.linalg.block_diag(jnp.eye(beam_1.n_ang_vel), 
                                        jnp.eye(beam_2.n_ang_vel), 
                                        jnp.eye(beam_3.n_ang_vel), 
                                        H_matrix)



    jax.jit
    def vector_field(t, state_vector):
        theta_vector = state_vector[:ind_coenergy_1]

        theta_vector_1 = theta_vector[:beam_1.n_ang_vel]
        theta_vector_2 = theta_vector[ind_theta_2:ind_theta_3]
        theta_vector_3 = theta_vector[ind_theta_3:ind_coenergy_1]

        coenergy_vector_1 = state_vector[ind_coenergy_1:ind_coenergy_2]
        coenergy_vector_2 = state_vector[ind_coenergy_2:ind_coenergy_3]
        coenergy_vector_3 = state_vector[ind_coenergy_3:ind_coenergy_end]

        vector_field_theta_1 = coenergy_vector_1[beam_1.n_lin_vel:beam_1.n_vel]
        vector_field_theta_2 = coenergy_vector_2[beam_2.n_lin_vel:beam_2.n_vel]
        vector_field_theta_3 = coenergy_vector_3[beam_3.n_lin_vel:beam_3.n_vel]

        vector_field_theta = jnp.concatenate([vector_field_theta_1, \
                                            vector_field_theta_2,   \
                                            vector_field_theta_3])
        
        vector_field_coenergy_1 = J_linear_matrix_1 @ coenergy_vector_1 \
            + beam_1.assemble_nonlinear_dynamics(coenergy_vector_1) \
            + beam_1.assemble_gravity(theta_vector_1) 
        
        vector_field_coenergy_2 = J_linear_matrix_2 @ coenergy_vector_2 \
            + beam_2.assemble_nonlinear_dynamics(coenergy_vector_2) \
            + beam_2.assemble_gravity(theta_vector_2)

        vector_field_coenergy_3 = J_linear_matrix_3 @ coenergy_vector_3 \
            + beam_3.assemble_nonlinear_dynamics(coenergy_vector_3) \
            + beam_3.assemble_gravity(theta_vector_3)
        
        vector_field_intrinsic = jnp.concatenate([vector_field_coenergy_1, \
                                                vector_field_coenergy_2,   \
                                                vector_field_coenergy_3])
        
        theta_1_at_L = theta_vector_1[-1]
        rot_matrix_1_at_L = rotation_matrix_2d(theta_1_at_L)

        theta_2_at_0 = theta_vector_2[0]
        rot_matrix_2_at_0 = rotation_matrix_2d(theta_2_at_0)
        theta_2_at_L = theta_vector_2[-1]
        rot_matrix_2_at_L = rotation_matrix_2d(theta_2_at_L)

        theta_3_at_0 = theta_vector_3[0]
        rot_matrix_3_at_0 = rotation_matrix_2d(theta_3_at_0)

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
        
        vector_field_coenergy = vector_field_intrinsic + vector_field_interconnection

        return jnp.concatenate([vector_field_theta, vector_field_coenergy])


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

    n_steps = int(T_end/dt)
    time_vec = dt * jnp.arange(n_steps + 1)
    trajectory, iterations = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field, return_iterations=True)

    # Format the dt string for clean filenames (e.g., '0.1' instead of '0.1000')
    dt_str = f"{dt:g}"
    # Save the data with the dt appended to the filename

    if save_data:
        jnp.save(f"{folder_results}iterations_dt_{dt_str}.npy", iterations)
        jnp.save(f"{folder_results}time_vec_dt_{dt_str}.npy", time_vec)
        jnp.save(f"{folder_results}trajectory_dt_{dt_str}.npy", trajectory)

    return trajectory, time_vec, iterations

if __name__ == "__main__":
    dt = 0.02
    T_end = 10
    import os
    folder_results_main = folder_results + f"Tend_{T_end}/"
    if not os.path.exists(folder_results_main):
        os.makedirs(folder_results_main)

    run_four_bar_gyrator(dt, T_end, folder_results_main, save_data=True)