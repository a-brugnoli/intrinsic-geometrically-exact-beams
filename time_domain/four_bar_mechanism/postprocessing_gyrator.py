import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from src.postprocessing.animate import animate_overlay_beam_deformation
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()  
import jax.numpy as jnp
from data import node_mid_2, \
    beam_1, beam_2, beam_3, \
    ind_coenergy_1, ind_coenergy_2, ind_coenergy_3, ind_coenergy_end, \
    folder_results
from compute_positions import compute_positions

dt = 0.02

folder_results_main = folder_results + f"Tend_10/"

trajectory = jnp.load(folder_results_main + f"trajectory_dt_{dt}.npy")
time_vec = jnp.load(folder_results_main + f"time_vec_dt_{dt}.npy")
iterations = jnp.load(folder_results_main + f"iterations_dt_{dt}.npy")

coenergy_1_sol = trajectory[ind_coenergy_1:ind_coenergy_2]
coenergy_2_sol = trajectory[ind_coenergy_2:ind_coenergy_3]
coenergy_3_sol = trajectory[ind_coenergy_3:ind_coenergy_end]

energy_1_vec, kinetic_energy_1_vec, deformation_energy_1_vec = \
    beam_1.compute_energies(coenergy_1_sol)

energy_2_vec, kinetic_energy_2_vec, deformation_energy_2_vec = \
    beam_2.compute_energies(coenergy_2_sol)

energy_3_vec, kinetic_energy_3_vec, deformation_energy_3_vec = \
    beam_3.compute_energies(coenergy_3_sol)

position_1_sol, position_2_sol, position_3_sol = compute_positions(trajectory, time_vec)
potential_energy_1_vec = beam_1.compute_potential_energy(position_1_sol[1])
potential_energy_2_vec = beam_2.compute_potential_energy(position_2_sol[1])
potential_energy_3_vec = beam_3.compute_potential_energy(position_3_sol[1])

total_intrinsic_energy = energy_1_vec + energy_2_vec + energy_3_vec
total_potential_energy = potential_energy_1_vec + potential_energy_2_vec + potential_energy_3_vec
total_energy = total_intrinsic_energy + total_potential_energy
plt.figure()
plt.plot(time_vec, total_intrinsic_energy, 'b--', label='Intrinsic Energy')
plt.plot(time_vec, total_potential_energy, 'r--', label='Potential Energy')
plt.plot(time_vec, total_energy, 'k', label='Total Energy')
plt.xlabel('Time [s]')
plt.ylabel('Energy [J]')
#plt.title('Energies of the four bar mechanism')
plt.grid(True)
plt.legend()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")
plt.tight_layout()

plt.figure()
plt.plot(time_vec[:-1], jnp.diff(total_energy), 'b--')
plt.xlabel('Time [s]')
plt.ylabel('$|H(\mathbf{e}_{n+1}) - H(\mathbf{e}_{n})|$ [J]')
#plt.title('Delta Energy of the four bar mechanism')
plt.grid(True)
plt.savefig(folder_results + f"EnergyRate.pdf", format="pdf")
plt.tight_layout()

plt.figure()
plt.plot(iterations, 'b--')
plt.xlabel('Time step')
plt.ylabel('Newton iterations')
plt.grid(True)
plt.savefig(folder_results + f"NewtonIterations.pdf", format="pdf")
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

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
# ax.set_title('Beam configurations over time')
ax.grid(True)

# Create a colorbar axis that matches the y-axis height
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # 5% width, small padding

cbar_1 = fig.colorbar(lc_1, cax=cax)
cbar_2 = fig.colorbar(lc_2, cax=cax)
cbar_3 = fig.colorbar(lc_3, cax=cax)

cbar_1.set_label('$t$ [s]')
cbar_2.set_label('$t$ [s]')
cbar_3.set_label('$t$ [s]')

ax.set_xlim(-0.1, 3.5)
ax.set_ylim(-2.1, 1.5)

fig.tight_layout()
fig.savefig(folder_results + "Configurations.pdf", format="pdf")


# Middle point of beam 2
r_mid_2 = position_2_sol[:, node_mid_2, :]

plt.figure()
plt.plot(time_vec, r_mid_2[0])
plt.xlabel('Time [s]')
plt.ylabel('$r_x^P$ [m]')
plt.grid(True)
plt.savefig(folder_results + f"r_x_P.pdf", format="pdf")
plt.tight_layout()

plt.figure()
plt.plot(time_vec, r_mid_2[1], 'r', label='1')
plt.xlabel('Time [s]')
plt.ylabel('$r_y^P$ [m]')
plt.grid(True)
plt.savefig(folder_results + f"r_y_P.pdf", format="pdf")
plt.tight_layout()


import pandas as pd

df_r_P = pd.DataFrame({
    "time": time_vec,
    "x": r_mid_2[0],
    "y": r_mid_2[1],
})

df_r_P.to_csv(folder_results + "time_signal_rP.csv", index=False)

n_steps = time_vec.shape[0] - 1

n_plot = 500
if n_plot<=n_steps:
    step_plot = n_steps//n_plot
else:
    step_plot=1

time_vec_plot = time_vec[::step_plot]
position_1_plot = position_1_sol[:, :, ::step_plot]
position_2_plot = position_2_sol[:, :, ::step_plot]
position_3_plot = position_3_sol[:, :, ::step_plot]

animation = animate_overlay_beam_deformation(time_vec_plot, 
                                    [position_1_plot,
                                    position_2_plot,
                                    position_3_plot],
                                    title="Four bar mechanism",
                                    labels=('Beam 1', 'Beam 2', 'Beam 3'),
                                    save_path=folder_results + f"Four_bar_mech.gif",
                                    x_lim=[-0.1, 4],
                                    y_lim=[-2.5, 2.5])

plt.show()
