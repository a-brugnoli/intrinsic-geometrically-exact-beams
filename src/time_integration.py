import numpy as np
from src.linear_algebra import solve_bcs
from scipy.sparse.linalg import spsolve

def _valid_bc_keys(dictionary):
    valid_values = {"left", "right"}
    for key in dictionary.keys():
        if key not in valid_values:
            return False
    return True

def newmark(q0, v0, M, K, dt, nt, bcs_acceleration = {}, gamma=0.5, beta=0.25):
    assert beta >= 0 and gamma >= 0.5, "The Newmark parameters are invalid"
    assert _valid_bc_keys(bcs_acceleration), "The bcs dictionary contains invalid keys"

    q_solution = np.zeros((nt+1, len(q0)))
    q_solution[0, :] = q0

    v_solution = np.zeros((nt+1, len(v0)))
    v_solution[0, :] = v0

    q_old = q0
    v_old = v0
    a_old = solve_bcs(M, - K @ q_old)
    
    A_newmark = (M + beta * dt**2 * K)
    
    for n in range(nt):
        b = - K @ (q_old + dt * v_old + (0.5 - beta) * dt**2 * a_old)
        a_new = solve_bcs(A_newmark, b)

        v_new = v_old + dt * ((1 - gamma) * a_old + gamma * a_new)
        q_new = q_old + dt * v_old + 0.5*dt**2*((1 - 2*beta)*a_old + 2*beta*a_new)

        q_solution[n+1, :] = q_new
        v_solution[n+1, :] = v_new

        q_old = q_new
        v_old = v_new
        a_old = a_new
    
    return q_solution, v_solution


def implicit_midpoint(x_0, M, A, dt, nt, bcs_essential = {}):
    x_solution = np.zeros((nt+1, len(x_0)))
    x_solution[0, :] = x_0

    x_old = x_0

    A_imp_midpoint = (M - dt/2 * A)

    for n in range(nt):
        b = (M + dt/2 * A) @ x_old
        x_new = solve_bcs(A_imp_midpoint, b)
        
        x_solution[n+1, :] = x_new

        x_old = x_new

    return x_solution


def stormer_verlet(var1_0, var2_0, M_2, A_1, A_2, \
            dt, nt, bcs_essential = {}):
    """
    Compute the solution using the Stormer Verlet Method
    The first variable is evaluated at integer time steps, 
    while the second is evaluated on half-integers.
    """
    var1_solution = np.zeros((nt+1, len(var1_0)))
    var2_solution = np.zeros((nt+1, len(var2_0)))

    var1_old = var1_0
    var2_old = var2_0

    var1_solution[0, :] = var1_old

    var2_old_midpoint = var2_old + dt/2 * solve_bcs(M_2, A_2 @ var1_old)
    var2_solution[0, :] = var2_old_midpoint

    for n in range(nt):
        var1_new = var1_old + dt * A_1 @ var2_old_midpoint
        var2_new_midpoint = var2_old_midpoint + dt * solve_bcs(M_2, A_2 @ var1_new)

        var1_solution[n+1, :] = var1_new
        var2_solution[n+1, :] = var2_new_midpoint

        var1_old = var1_new
        var2_old_midpoint = var2_new_midpoint

    
    return var1_solution, var2_solution
