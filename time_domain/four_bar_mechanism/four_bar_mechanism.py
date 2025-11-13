import time
from src.models.intrinsic_pinned_beam import IntrinsicPlanarPinnedBeam
from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from data import material_props, geometrical_props_1, geometrical_props_2, geometrical_props_3, \
    L_1, L_2, L_3, theta_1_0, theta_2_0, theta_3_0, g
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d, rotation_matrix_2d_batch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.animate import animate_overlay_beam_deformation
from src.postprocessing.postprocess_variables import compute_beam_positions, implicit_midpoint_integrate
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()

import os
from jax import config; config.update("jax_enable_x64", True)

# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

n_elem_1 = 10
n_elem_2 = 6
n_elem_3 = 6

pinned_beam_1 = IntrinsicPlanarPinnedBeam(n_elem_1, geometrical_props_1, material_props)
free_beam_2 = IntrinsicPlanarFreeBeam(n_elem_2, geometrical_props_2, material_props)
pinned_beam_3 = IntrinsicPlanarPinnedBeam(n_elem_3, geometrical_props_3, material_props)

# Matrices for the pinned beam 1
H_matrix_1 = pinned_beam_1.assemble_energy_matrix()
J_linear_matrix_1 = pinned_beam_1.assemble_linear_interconnection_matrix()
B_matrix_1_at_L = pinned_beam_1.assemble_control_matrix()[:, 3:]
B_1_velocity_at_L = B_matrix_1_at_L[:, 1:]

# Matrices for the free beam 2
H_matrix_2 = free_beam_2.assemble_energy_matrix()
J_linear_matrix_2 = free_beam_2.assemble_linear_interconnection_matrix()
B_matrix_2 = free_beam_2.assemble_control_matrix()
B_matrix_2_at_0 = B_matrix_2[:, :3]
B_2_force_at_0 = B_matrix_2_at_0[:, :2]

B_matrix_2_at_L = B_matrix_2[:, 3:]
B_2_force_at_L = B_matrix_2_at_L[:, :2]

# Matrices for the pinned beam 3
H_matrix_3 = pinned_beam_3.assemble_energy_matrix()  
J_linear_matrix_3 = pinned_beam_3.assemble_linear_interconnection_matrix()
B_matrix_3_at_0 = pinned_beam_3.assemble_control_matrix()[:, :3]
B_3_velocity_at_0 = B_matrix_3_at_0[:, 1:]

gravity_force_1 = jnp.array([0, -g*geometrical_props_1['A']*material_props['rho']])
gravity_force_2 = jnp.array([0, -g*geometrical_props_2['A']*material_props['rho']])
gravity_force_3 = jnp.array([0, -g*geometrical_props_3['A']*material_props['rho']])

ind_theta_2 = pinned_beam_1.n_ang_vel
ind_theta_3 = pinned_beam_1.n_ang_vel + free_beam_2.n_ang_vel

ind_coenergy_1 = ind_theta_3 + pinned_beam_3.n_ang_vel
ind_coenergy_2 = ind_coenergy_1 + pinned_beam_1.n_coenergy
ind_coenergy_3 = ind_coenergy_2 + free_beam_2.n_coenergy

jax.jit
def vector_field(state_vector, t):
    theta_vector_1 = state_vector[:pinned_beam_1.n_ang_vel]
    theta_vector_1_mid = (theta_vector_1[:-1] + theta_vector_1[1:]) / 2

    rot_matrix_batch_1_mid = jnp.transpose(rotation_matrix_2d_batch(theta_vector_1_mid),\
                                       (0, 2, 1))

    # To convert from spatial to material the transpose of the rotation matrix has to be taken
    theta_1_at_L = theta_vector_1[-1]
    rot_matrix_1_at_L = rotation_matrix_2d(theta_1_at_L)

    theta_vector_2 = state_vector[ind_theta_2:ind_theta_2 + free_beam_2.n_ang_vel]
    rot_matrix_batch_2 = jnp.transpose(rotation_matrix_2d_batch(theta_vector_2),\
                                       (0, 2, 1))
    
    theta_2_at_0 = theta_vector_2[0]
    rot_matrix_2_at_0 = rotation_matrix_2d(theta_2_at_0)

    theta_2_at_L = theta_vector_2[-1]
    rot_matrix_2_at_L = rotation_matrix_2d(theta_2_at_L)

    theta_vector_3 = state_vector[ind_theta_3:ind_theta_3 + pinned_beam_3.n_ang_vel]
    theta_vector_3_mid = (theta_vector_3[:-1] + theta_vector_3[1:]) / 2 
    rot_matrix_batch_3_mid = jnp.transpose(rotation_matrix_2d_batch(theta_vector_3_mid),\
                                           (0, 2, 1))
    
    theta_3_at_0 = theta_vector_3[0]
    rot_matrix_3_at_0 = rotation_matrix_2d(theta_3_at_0)

    coenergy_vector_1 = state_vector[ind_coenergy_1:ind_coenergy_2]
    coenergy_vector_2 = state_vector[ind_coenergy_2:ind_coenergy_3]
    coenergy_vector_3 = state_vector[ind_coenergy_3:]

    vector_field_theta_1 = coenergy_vector_1[pinned_beam_1.n_lin_vel:pinned_beam_1.n_vel]
    vector_field_theta_2 = coenergy_vector_2[free_beam_2.n_lin_vel:free_beam_2.n_vel]
    vector_field_theta_3 = coenergy_vector_3[pinned_beam_3.n_lin_vel:pinned_beam_3.n_vel]

    vector_field_theta = jnp.concatenate([vector_field_theta_1, \
                                          vector_field_theta_2,   \
                                          vector_field_theta_3])
    
    material_gravity_1_mid = rot_matrix_batch_1_mid @ gravity_force_1
    material_gravity_2 = rot_matrix_batch_2 @ gravity_force_2
    material_gravity_3_mid = rot_matrix_batch_3_mid @ gravity_force_3

    raise NotImplementedError("Finish the implementation of the four bar mechanism interconnection.")

    # vector_field_coenergy_beam_1 = J_linear_matrix_1 @ coenergy_vector_1 \
    #     + pinned_beam_1.assemble_nonlinear_dynamics(coenergy_vector_1) 

    return 0