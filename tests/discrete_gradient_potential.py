import time, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results_discrete_gradient/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)

# -----------------------------------------
# Parameters
# -----------------------------------------
m = 1.0
I = 1.0
g = 9.81
L = 1
# center of mass in body coordinates
r_c = np.array([L/2, 0.0])
g_vec = np.array([0.0, -g])
S = np.array([[0, -1], [1, 0]])

def R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

# Potential
def U(theta):
    return -m * g_vec @ (R(theta) @ r_c)

# Torque (continuous gradient)
def tau(theta):
    f = -m * (R(theta).T @ g_vec)        # body force
    return -(r_c @ (S @ f))

# Discrete gradient torque + its derivative
def tau_dg(theta_k, theta_k1):
    dθ = theta_k1 - theta_k
    U0 = U(theta_k)
    U1 = U(theta_k1)
    ΔU = U1 - U0
    τ0 = tau(theta_k)
    
    if abs(dθ) < 1e-14:
        return τ0
    
    return τ0 + (ΔU - τ0*dθ)/dθ

def tau_dg_prime(theta_k, theta_k1):
    dθ = theta_k1 - theta_k
    τ0 = tau(theta_k)
    τ1 = tau(theta_k1)
    
    if abs(dθ) < 1e-14:
        # limit → second derivative of U
        return τ1
    
    # derivative of the Gonzalez expression
    return (τ1*dθ - (U(theta_k1)-U(theta_k)-τ0*dθ)) / (dθ**2)

# -----------------------------------------
# Time integration
# -----------------------------------------
h = 0.01
steps = 100
time_vec = h * np.arange(steps+1)

theta = np.zeros(steps+1)
omega = np.zeros(steps+1)

# initial condition
theta_0 = np.pi/4
omega_0 = 0.0
theta[0] = theta_0
omega[0] = omega_0

n_iterations = 20

for k in range(steps):

    θk = theta[k]
    ωk = omega[k]

    # initial guess = previous state
    x = np.array([θk, ωk])

    for _ in range(n_iterations):   # Newton iterations
    
        θ1, ω1 = x
        θm = 0.5*(θk + θ1)
        ωm = 0.5*(ωk + ω1)
        
        # nonlinear equations
        F1 = θ1 - θk - h*ωm
        τDG = tau_dg(θk, θ1)
        F2 = ω1 - ωk - h/I * τDG
        
        F = np.array([F1, F2])
        
        # Jacobian
        dτ = tau_dg_prime(θk, θ1)
        J = np.array([
            [1.0,   -h/2],
            [-h/I * dτ,  1.0]
        ])
        
        # Newton update
        dx = np.linalg.solve(J, -F)
        x += dx
        
        if np.linalg.norm(dx) < 1e-12:
            break

    theta[k+1], omega[k+1] = x

# energy
KE = 0.5*I*omega**2
PE = np.array([U(t) for t in theta])
E = KE + PE

plt.figure()
plt.plot(time_vec, E)
plt.title("Energy (Gonzalez DG, Newton-solved midpoint)")
plt.xlabel("time")
plt.ylabel("E")


plt.figure()
plt.plot(time_vec, theta)
plt.xlabel("time")
plt.ylabel("theta")

plt.figure()
plt.plot(time_vec[1:], np.diff(E))
plt.title("Energy (Gonzalez DG, Newton-solved midpoint)")
plt.xlabel("time")
plt.ylabel("E")

# fig, ax = plt.subplots()
# ax.set_xlim(-1.2, 1.2)
# ax.set_ylim(-1.2, 1.2)
# ax.set_aspect('equal')
# line, = ax.plot([], [], lw=3)

# def update(frame):
#     th = theta[frame]
#     x2 = L * np.cos(th)
#     y2 = L * np.sin(th)
#     line.set_data([0, x2], [0, y2])
#     return line,


# anim = FuncAnimation(fig, update, frames=steps, interval=1, \
#                     blit=True, repeat=True)
# save_path = folder_results + "RotatingRod.gif"
# if save_path.endswith('.gif'):
#     anim.save(save_path, writer='pillow')
#     print(f"Overlay animation saved as GIF: {save_path}")
# elif save_path.endswith('.mp4'):
#     anim.save(save_path, writer='ffmpeg')
#     print(f"Overlay animation saved as MP4: {save_path}")

# print("Energy drift:", E[-1] - E[0])

plt.show()