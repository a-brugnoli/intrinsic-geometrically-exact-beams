from src.models.intrinsic_free_beam import IntrinsicPlanarFreeBeam
from data import geometrical_props, material_props
import jax.numpy as jnp
import jax
from src.time_integration.utilities import simulate
from src.rotations.utilities import rotation_matrix_2d
import matplotlib.pyplot as plt
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()

n_elements = 10  # Number of finite elements
free_beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)
# This is not consistent physically but it is the original setup by Simo and Vu-Quoc
EA = 1e4
free_beam.set_axial_compliance(1/EA)

H_matrix, J_linear_matrix = free_beam.assemble_matrices()
B_matrix_at_L = free_beam.assemble_control_matrix()[:, 3:]

B_force = B_matrix_at_L[:, :2]
B_torque = B_matrix_at_L[:, 2]


jax.jit
def spatial_force(t):
    return jnp.where(t <= 2.5, jnp.array([8.0, 0]), jnp.array([0, 0]))


jax.jit
def spartial_torque(t):
    # Example: sinusoidal torque
    return jnp.where(t <= 2.5, -80.0, 0.0)


jax.jit
def vector_field(state_vector, t):
    theta_vector = state_vector[:free_beam.n_ang_vel]
    coenergy_vector = state_vector[free_beam.n_ang_vel:]

    # To convert from spatial to material the transpose of the rotation matrix has to be taken
    theta_at_L = theta_vector[-1]
    rot_matrix_at_L = rotation_matrix_2d(theta_at_L)
    material_force_at_L = rot_matrix_at_L.T @ spatial_force(t)

    vector_field_theta = coenergy_vector[free_beam.n_lin_vel:free_beam.n_lin_vel + free_beam.n_ang_vel]

    vector_field_coenergy = J_linear_matrix @ coenergy_vector \
        + free_beam.assemble_nonlinear_dynamics(coenergy_vector) \
        + B_force @ material_force_at_L \
        + B_torque * spartial_torque(t)
        
    return jnp.concatenate([vector_field_theta, vector_field_coenergy])


alpha = jnp.atan2(-8, 6)
angle_0 = alpha * jnp.ones(free_beam.n_ang_vel)
coenergy_vector_0 = jnp.zeros(free_beam.n_dofs)
state_vector_0 = jnp.concatenate([angle_0, coenergy_vector_0])

dt = 0.1
T_end = 15
n_steps = int(T_end/dt)

# Run simulation
E_matrix = jax.scipy.linalg.block_diag(jnp.eye(free_beam.n_ang_vel), H_matrix)
trajectory = simulate(state_vector_0, dt, n_steps, E_matrix, vector_field)
# trajectory_np = jax.device_get(trajectory)

theta_sol = trajectory[:free_beam.n_ang_vel, :]
coenergy_sol = trajectory[free_beam.n_ang_vel:, :]

velocity_sol = coenergy_sol[:free_beam.n_lin_vel + free_beam.n_ang_vel, :]
lin_velocity_sol = velocity_sol[:free_beam.n_lin_vel, :]
stress_sol = coenergy_sol[free_beam.n_lin_vel + free_beam.n_ang_vel:, :]

energy_vec = free_beam.compute_energy(coenergy_sol)

plt.figure(figsize=(8, 5))
plt.plot(dt* jnp.arange(n_steps+1), energy_vec, 'm', label='H')
plt.plot(dt* jnp.arange(n_steps+1), free_beam.compute_kinetic_energy(velocity_sol), 'g', label='T')
plt.plot(dt* jnp.arange(n_steps+1), free_beam.compute_deformation_energy(stress_sol), 'b', label='V')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.title('Energy vs Time')
plt.grid(True)
plt.legend(loc="best")
plt.show()

