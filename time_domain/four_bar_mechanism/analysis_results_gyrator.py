import array
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
from src.postprocessing.postprocess_variables import compute_beam_positions
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

ind_theta_2 = beam_1.n_ang_vel
ind_theta_3 = beam_1.n_ang_vel + beam_2.n_ang_vel

ind_coenergy_1 = ind_theta_3 + beam_3.n_ang_vel
ind_coenergy_2 = ind_coenergy_1 + beam_1.n_coenergy
ind_coenergy_3 = ind_coenergy_2 + beam_2.n_coenergy
ind_coenergy_end = ind_coenergy_3 + beam_3.n_coenergy

time_vec = jnp.array(np.load(folder_results + "time_vec.npy"))
trajectory_plot = jnp.array(np.load(folder_results + "trajectory_plot.npy"))

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

print('Theta 1 shape:', theta_1_sol.shape)
print('Lin vel 1 shape:', lin_velocity_1_sol.shape)
print('Time vec shape:', time_vec.shape)

x_A, y_A = point_A_0
x_points_1_t0 = x_A + beam_1.x_nodes * jnp.cos(theta_1_sol[:, 0])
y_points_1_t0 = y_A + beam_1.x_nodes * jnp.sin(theta_1_sol[:, 0])

x_midpoints_1_t0 = 0.5*(x_points_1_t0[1:] + x_points_1_t0[:-1])
y_midpoints_1_t0 = 0.5*(y_points_1_t0[1:] + y_points_1_t0[:-1])
positions_1_t0 = jnp.stack([x_midpoints_1_t0,
                            y_midpoints_1_t0], axis=1)

position_1_sol = compute_beam_positions(theta_1_sol, 
                                        lin_velocity_1_sol, 
                                        positions_1_t0, time_vec)


norm_position_sol = jnp.linalg.norm(position_1_sol, axis=(0, 1))
norm_lin_velocity_sol = jnp.linalg.norm(lin_velocity_1_sol, axis=0)
norm_theta_sol = jnp.linalg.norm(theta_1_sol, axis=0)

x_B, y_B = point_B_0
x_points_2_t0 = x_B + beam_2.x_nodes * jnp.cos(theta_2_sol[:, 0])
y_points_2_t0 = y_B + beam_2.x_nodes * jnp.sin(theta_2_sol[:, 0])

positions_2_t0 = jnp.stack([x_points_2_t0,
                                y_points_2_t0], axis=1)

position_2_sol = compute_beam_positions(theta_2_sol, 
                                        lin_velocity_2_sol, 
                                        positions_2_t0, time_vec)

x_C, y_C = point_C_0
x_points_3_t0 = x_C + beam_3.x_nodes * jnp.cos(theta_3_sol[:, 0])
y_points_3_t0 = y_C + beam_3.x_nodes * jnp.sin(theta_3_sol[:, 0])

x_midpoints_3_t0 = 0.5*(x_points_3_t0[1:] + x_points_3_t0[:-1])
y_midpoints_3_t0 = 0.5*(y_points_3_t0[1:] + y_points_3_t0[:-1])
positions_3_t0 = jnp.stack([x_midpoints_3_t0,
                                y_midpoints_3_t0], axis=1)

position_3_sol = compute_beam_positions(theta_3_sol, 
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
plt.plot(time_vec, energy_2_vec, 'b--', label='Intrinsic Energy')
plt.plot(time_vec, potential_energy_2_vec, 'r--', label='Potential Energy')
plt.plot(time_vec, potential_energy_2_vec + energy_2_vec, 'k', label='Total Energy')
plt.xlabel('Time (s)')
plt.title('Energies of the second beam')
plt.grid(True)
plt.legend()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")
plt.tight_layout()

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


plt.show()
