import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from src.time_integration.utilities import simulate
# Physical parameters
g = 9.81  # gravity
l = 1.0   # length of pendulum
m = 1.0
M = jnp.diag(jnp.array([1.0, m*l**2]))

def torque(t):
    # Example: sinusoidal torque
    return 0.5 * jnp.sin(2 * jnp.pi * t)

def f(x, t):
    theta, omega = x
    tau = torque(t)
    return jnp.array([omega, - g *l* jnp.sin(theta) + tau ])

# Initial condition: angle 90 degrees, no velocity
x0 = jnp.array([jnp.pi / 2, 0.0])
dt = 0.05
steps = 400

# Run simulation
trajectory = simulate(x0, dt, steps, M, f)
trajectory_np = jax.device_get(trajectory)

# Plot phase space (theta vs omega)
theta = trajectory_np[:, 0]
omega = trajectory_np[:, 1]

plt.figure(figsize=(8, 5))
plt.plot(theta, omega, lw=2)
plt.xlabel(r"$\theta$ (rad)")
plt.ylabel(r"$\omega$ (rad/s)")
plt.title("Phase Space Trajectory of Pendulum (Implicit Midpoint)")
plt.grid(True)
plt.show()
