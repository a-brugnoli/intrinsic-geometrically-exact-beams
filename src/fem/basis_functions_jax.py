import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# ref element basis functions (linear Lagrange) and evaluations
def phi_ref(i, xi):
    """Lagrange basis functions on ref element [-1,1]
    
    Args:
        i: index of the basis function (0 or 1)
        xi: JAX array of evaluation points in [-1, 1]
    Returns:
        Evaluation of the i-th basis function at points xi 
    """
    return jnp.where(i == 0, 0.5 * (1 - xi), 0.5 * (1 + xi))


def dphi_ref(i, xi):
    """Derivatives of linear Lagrange basis functions on ref [-1, 1]"""
    return jnp.where(i == 0, -0.5, 0.5)  # constant on ref element


def lagrange_piecewise_linear_1d(x_nodes: jnp.ndarray, u_nodes: jnp.ndarray) -> callable:
    """
    Builds a piecewise linear Lagrange interpolant using reference element.

    Args:
        x_nodes (jnp.ndarray): Coordinates of the nodes, shape (N,)
        u_nodes (jnp.ndarray): Values at the nodes, shape (N,)

    Returns:
        Callable: A function u_h(x) evaluating the interpolant at given x
    """
    assert x_nodes.shape[0] == u_nodes.shape[0], "x_nodes and u_nodes must have the same length"
    def find_element(x):
        # Return index i such that x in [x_nodes[i], x_nodes[i+1])
        # Clamp to last interval to include x == x_nodes[-1]
        i = jnp.searchsorted(x_nodes, x, side="right") - 1
        return jnp.clip(i, 0, x_nodes.shape[0] - 2)

    def eval_single(x):
        i = find_element(x)
        xL = x_nodes[i]
        xR = x_nodes[i + 1]
        uL = u_nodes[i]
        uR = u_nodes[i + 1]

        # Map physical x to reference xi in [-1, 1]
        xi = 2 * (x - xL) / (xR - xL) - 1.0

        # Evaluate reference basis functions
        phi0 = phi_ref(0, xi)
        phi1 = phi_ref(1, xi)

        return uL * phi0 + uR * phi1

    # Vectorized and JIT-ed version
    u_h = jax.jit(jax.vmap(eval_single))
    return lambda x: u_h(x) if x.ndim > 0 else eval_single(x)


def piecewise_constant_1d(x_nodes: jnp.ndarray, u_dofs: jnp.ndarray) -> callable:
    """
    Constructs a discontinuous piecewise constant interpolant.
    
    Args:
        x_nodes (jnp.ndarray): Mesh node coordinates, shape (N+1,)
        u_dofs (jnp.ndarray): Function values at element midpoints, shape (N,)

    Returns:
        Callable: A function u_h(x) evaluating the interpolant at given x
    """
    assert x_nodes.shape[0] == u_dofs.shape[0] + 1, "x_nodes must have one more element than u_dofs"
    # Number of elements
    N = x_nodes.shape[0] - 1

    def find_element(x):
        # Find element i such that x in [x_i, x_{i+1})
        i = jnp.searchsorted(x_nodes, x, side='right') - 1
        return jnp.clip(i, 0, N - 1)

    def eval_single(x):
        i = find_element(x)
        return u_dofs[i]

    # Vectorized and JIT-compiled version
    u_h = jax.jit(jax.vmap(eval_single))
    return lambda x: u_h(x) if x.ndim > 0 else eval_single(x)


if __name__ == "__main__":
    x_nodes = jnp.linspace(0, 1, 11)  # Example nodes
    f_nodes = jnp.sin(2 * jnp.pi * x_nodes)  # Example values at nodes
    
    x_midpoints = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    f_midpoints = jnp.sin(2 * jnp.pi * x_midpoints)  # Values at midpoints

    f_cg = lagrange_piecewise_linear_1d(x_nodes, f_nodes)
    f_dg = piecewise_constant_1d(x_nodes, f_midpoints)

    x_eval = jnp.linspace(0, 1, 51)
    y_eval_cg = f_cg(x_eval)  # CG interpolated values
    y_eval_dg = f_dg(x_eval)  # DG interpolated values

    plt.plot(x_nodes, f_nodes, 'o', label='Nodes')
    plt.plot(x_midpoints, f_midpoints, '+', label='Midpoints')
    plt.plot(x_eval, y_eval_cg, '-1', label='C1 Interpolant')
    plt.plot(x_eval, y_eval_dg, 'x', label='C0 Interpolant')

    # from src.postprocessing.plot_dg import plot_piecewise_constant
    # x_plot_dg, y_plot_dg = plot_piecewise_constant(x_nodes, f_midpoints)
    # plt.plot(x_plot_dg, y_plot_dg, 'g--', label='Piecewise Constant')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Interpolants Comparison')
    plt.legend()
    plt.grid()
    plt.show()