from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from data import geometrical_props, material_props
import jax.numpy as jnp
from src.time_integration.utilities import simulate_linear
import matplotlib.pyplot as plt
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
import os

from jax import config; config.update("jax_enable_x64", True)

# Setup output directory
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)


n_elements = 10  # Number of finite elements
free_beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)


H_matrix = free_beam.assemble_energy_matrix()
J_linear_matrix = free_beam.assemble_linear_interconnection_matrix()

coenergy_vector_0 = jnp.zeros(free_beam.n_dofs)
coenergy_vector_0 = coenergy_vector_0.at[free_beam.n_nodes:free_beam.n_lin_vel].set(jnp.sin(jnp.pi*free_beam.x_nodes))

dt = 0.01
T_end = 1
n_steps = int(T_end/dt)

# Run simulation
coenergy_sol = simulate_linear(coenergy_vector_0, dt, n_steps, H_matrix, J_linear_matrix)

energy_vec, kinetic_energy_vec, deformation_energy_vec = free_beam.compute_energies(coenergy_sol)

# Plot of energy trend
plt.figure()
plt.plot(dt* jnp.arange(n_steps+1), energy_vec, 'm', label='H')
plt.plot(dt* jnp.arange(n_steps+1), kinetic_energy_vec, 'g', label='T')
plt.plot(dt* jnp.arange(n_steps+1), deformation_energy_vec, 'b', label='V')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy vs Time')
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(folder_results + f"Energies.pdf", format="pdf")

# Plot of energy difference
plt.figure()
plt.plot(dt* jnp.arange(n_steps), jnp.abs(jnp.diff(energy_vec)), 'm')
plt.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('$|H(\mathbf{e}_{n+1}) - H(\mathbf{e}_{n})|$')
plt.title('Delta Energy vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig(folder_results + f"DeltaEnergy.pdf", format="pdf")
plt.show()