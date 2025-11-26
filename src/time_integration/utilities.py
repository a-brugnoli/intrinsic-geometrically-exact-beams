from ast import For
from csv import Error
from src.linear_algebra.la_jax import newton_solver_jax, newton_solver_fd
import jax.numpy as jnp
import jax
from functools import partial


@partial(jax.jit, static_argnums=-1)
def forward_euler(xn, tn, h, E, f: callable):
    """
    Forward Euler step for DAE system.
    Solves: E * (x_{n+1} - x_n) = h * f(t_n, x_n)
    """
    return xn + h * jnp.linalg.solve(E, f(tn, xn))


@partial(jax.jit, static_argnums=-1)
def implicit_midpoint(xn, tn, h, E, f: callable):

    def residual(x_):
        return jnp.dot(E, x_-xn) - h * f(tn + h/2, (x_ + xn)/2)

    x0 = xn  # initial guess 
    xn1, _ = newton_solver_jax(residual, x0)

    return xn1


@partial(jax.jit, static_argnums=[2, 4, 5])
def simulate(x0, dt, num_steps, E, f: callable, method="midpoint"):
    def step_fn(xn, i):
        jax.debug.print(
        "Progress time integration: {i} / {num_steps}", i=i, num_steps=num_steps
    )
        if method=="midpoint":
            xnp1 = implicit_midpoint(xn, i*dt, dt, E, f)
        elif method=="Euler explicit":
            xnp1 = forward_euler(xn, i*dt, dt, E, f)
        return xnp1, xnp1

    _, xs = jax.lax.scan(step_fn, x0, jnp.arange(num_steps))

    return jnp.concatenate([x0.reshape(-1, 1), xs.T], axis=1)  


@partial(jax.jit, static_argnums=[4, 5])
def implicit_midpoint_splitting(xn, tn, h, E, f_intrinsic: callable, f_interconnection: callable):
    """
    Solves the dynamics
    E ẋ = f_intrinsic(t, x) + f_interconnection(t, x),
    using implicit midpoint rule for the intrinsic dynamics and Crank-Nicolson for the interconnection.
    """    

    def residual(x_):
        return jnp.dot(E, x_-xn) - h * (f_intrinsic(tn + h/2, (x_ + xn)/2) \
                                        + 0.5*f_interconnection(tn, xn) + 0.5*f_interconnection(tn + h, x_))

    x0 = xn  # initial guess 
    xn1, _ = newton_solver_jax(residual, x0)

    return xn1


@partial(jax.jit, static_argnums=[2, 4, 5])
def simulate_splitting(x0, dt, num_steps, E, f_intrinsic: callable, f_interconnection: callable):
    def step_fn(xn, i):
        jax.debug.print(
        "Progress time integration: {i} / {num_steps}", i=i, num_steps=num_steps
    )
        xnp1 = implicit_midpoint_splitting(xn, i*dt, dt, E, f_intrinsic, f_interconnection)
        return xnp1, xnp1

    _, xs = jax.lax.scan(step_fn, x0, jnp.arange(num_steps))

    return jnp.concatenate([x0.reshape(-1, 1), xs.T], axis=1)  


@partial(jax.jit, static_argnums=[5, 6])
def implicit_midpoint_constraints_backward_euler(xn, lambdan, tn, h, E, f: callable, G: callable):
    """
    Solves the dynamics
    E ẋ = f(t, x) + G(t, x) lambda,
    0 = - G(t, x)^T x.
    using implicit midpoint rule for the dynamics and backward Euler for the constraints.
    """
    n_state = xn.shape[0]
    # n_lambda = lambdan.shape[0]
    def residual(z_):
        x_ =  z_[:n_state]
        lambda_ = z_[n_state:]

        res_dynamics = jnp.dot(E, x_ - xn) - h * (f(tn + h/2, (x_+ xn)/2) \
                                                + G(tn + h, x_) @ lambda_)
        
        res_constraints = G(tn + h, x_).T @ x_
        return jnp.concatenate([res_dynamics, res_constraints])

    z0 = jnp.concatenate([xn, lambdan])  # initial guess
    zn1, _ = newton_solver_jax(residual, z0)

    return zn1


@partial(jax.jit, static_argnums=[5, 6])
def implicit_midpoint_constraints_crank_nicolson(xn, lambdan, tn, h, E, f: callable, G: callable):
    """
    Solves the dynamics
    E ẋ = f(t, x) + G(t, x) lambda,
    0 = - G(t, x)^T x.
    using implicit midpoint rule for the dynamics and Crank Nicolson for the constraints.
    """
    n_state = xn.shape[0]
    # n_lambda = lambdan.shape[0]
    def residual(z_):
        x_ =  z_[:n_state]
        lambda_ = z_[n_state:]

        res_dynamics = jnp.dot(E, x_ - xn) - h * (f(tn + h/2, (x_+ xn)/2) \
                                            + 0.5*G(tn + h, x_) @ lambda_ \
                                            + 0.5*G(tn, xn) @ lambdan)
        
        res_constraints = G(tn + h, x_).T @ x_ 

        return jnp.concatenate([res_dynamics, res_constraints])

    z0 = jnp.concatenate([xn, lambdan])  # initial guess
    zn1, _ = newton_solver_jax(residual, z0)

    return zn1


@partial(jax.jit, static_argnums=[3, 5, 6, 7])
def simulate_exact_constraints(x0, lambda0, dt, num_steps, E, f: callable, G: callable, method="crank_nicolson"):
    n_state = x0.shape[0]
    # n_lambda = lambda0.shape[0]
    def step_fn(zn, i):
        jax.debug.print(
        "Progress time integration: {i} / {num_steps}", i=i, num_steps=num_steps
        )
        xn = zn[:n_state]
        lambdan = zn[n_state:]
        if method=="crank_nicolson":
            znp1 = implicit_midpoint_constraints_crank_nicolson(xn, lambdan, i*dt, dt, E, f, G)
        elif method=="backward_euler":
            znp1 = implicit_midpoint_constraints_backward_euler(xn, lambdan, i*dt, dt, E, f, G)
        else:
            raise NotImplementedError("Only backward Euler and Crank-Nicolson for constraints integrator")
        
        return znp1, znp1

    z0 = jnp.concatenate([x0, lambda0])
    _, zs = jax.lax.scan(step_fn, z0, jnp.arange(num_steps))

    return jnp.concatenate([z0.reshape(-1, 1), zs.T], axis=1)  


@partial(jax.jit, static_argnums=[2, 5])
def simulate_linear(x0, dt, num_steps, E, J, method="midpoint"):
    def step_fn(xn, i):
        jax.debug.print(
        "Progress time integration: {i} / {num_steps}", i=i, num_steps=num_steps
    )
        if method=="midpoint":
            xnp1 = jnp.linalg.solve(E - dt/2 * J, (E + dt/2 * J) @ xn)
            return xnp1, xnp1 
        else:
            raise NotImplementedError("Only implicit midpont for linear integrator")  

    _, xs = jax.lax.scan(step_fn, x0, jnp.arange(num_steps))

    return jnp.concatenate([x0.reshape(-1, 1), xs.T], axis=1)  


@partial(jax.jit, static_argnums=[1, 2])
def compute_lambda0(E, f: callable, G: callable, x0, t0=0):
    """
    Compute λ₀ for the index-2 DAE:
        E ẋ = f(x) + G(x) λ
        G(x)^T x = 0
    
    PARAMETERS
    ----------
    E : (n×n) constant matrix
    f(x) : (n,)
    G(x) : (n×m)
    x0 : initial state vector,
    t0: initial time
    rcond : tolerance for least squares fallback

    RETURNS
    -------
    lambda0 : (m,)
    info     : dict with details
    """

    f0 = f(t0, x0)      
    G0 = G(t0, x0)
    # Jacobian of constraints wrt state
    Phi = lambda x: G(t0, x).T @ x
    Dphi = jax.jacobian(Phi)(x0)       # (m×n)

    # E^{-1} f via linear solve
    v_f = jnp.linalg.solve(E, f0)      # (n,)

    # E^{-1} G via linear solve for all columns
    v_G = jnp.linalg.solve(E, G0)      # (n×m)

    # Build linear system A λ = b
    A = Dphi @ v_G                     # (m×m)
    b = -Dphi @ v_f                    # (m,)

    # Solve A λ = b
    if A.shape[0] == A.shape[1]:
        # square system: try solve, fallback to lsts
        lambda0 = jnp.linalg.solve(A, b)
    else:
        raise Error("Non-square system for initial multipliers")
    return lambda0