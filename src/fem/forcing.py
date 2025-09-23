import jax
import jax.numpy as jnp
from functools import partial
from src.fem.quadrature import gauss_quadrature
from src.fem.basis_functions import phi_ref, eval_ref_elem_interp_at_nodes


@partial(jax.jit, static_argnums=[-2])
def _element_forcing_cg(xL, xR, u_local, nonlinear_func, n_gauss_points):
    xi_q, w_q = gauss_quadrature(n_gauss_points)
    jac = (xR - xL) / 2.0
    
    # Shape functions at Gauss point
    phi_vals = jnp.stack([phi_ref(0, xi_q), phi_ref(1, xi_q)])
    u_inerpolated_at_gp = eval_ref_elem_interp_at_nodes(u_local, xi_q)
    # Integrand: J * phi_i * f(u) where f is the user-specified function
    integrand = jac * phi_vals * nonlinear_func(u_inerpolated_at_gp)
    return jnp.dot(integrand, w_q)


@partial(jax.jit, static_argnums=[2])
def assemble_state_dependent_forcing_vector_cg(x_nodes: jnp.ndarray,
                                                u_dofs: jnp.ndarray,
                                                nonlinear_func : callable,
                                                n_gauss_points: int = 3) -> jnp.ndarray:
    """
    Assemble global nonlinear forcing vector f(u) where f is a user-defined function of u.

    Args:
        x_nodes (jnp.ndarray): 1D array of mesh node coordinates, length N+1.
        u_dofs (jnp.ndarray): Degrees of freedom for u_h.
            - If length N+1, u_h is piecewise linear (P1).
            - If length N, u_h is piecewise constant (P0).
        nonlinear_func (callable): Function f(u) defining the nonlinear term ∫ φᵢ f(u) dx
        n_gauss_points (int): Number of Gauss   points per element for integration (1 to 4)

    Returns:    
        jnp.ndarray: Global forcing vector of length N+1 corresponding to the nodal test functions.
    """
    n_nodes = x_nodes.shape[0]
    n_elements = n_nodes - 1
    len_u = u_dofs.shape[0]

    assert len_u in (n_nodes, n_nodes - 1), "u_dofs must be of length N+1 or N"
    u_is_p1 = len_u == n_nodes

    form = jnp.zeros(n_nodes)

    def element_contribution(i, acc):
        """"
        Compute the local contribution to the global linear form for each element.
        """
        xL, xR = x_nodes[i], x_nodes[i + 1]
        u_local = jax.lax.dynamic_slice(u_dofs, (i,), (2,)) if u_is_p1 else jnp.array([u_dofs[i]])

        local = _element_forcing_cg(xL, xR, u_local, nonlinear_func, n_gauss_points)

        acc = acc.at[i].add(local[0])
        acc = acc.at[i + 1].add(local[1])
        return acc

    return jax.lax.fori_loop(0, n_elements, element_contribution, form)


# Test with common nonlinear functions
def test_common_functions():
    """Test various common nonlinear functions"""
    
    x_nodes = jnp.linspace(0.0, 1.0, 11)  # 10 elements
    u_dofs = jnp.linspace(-1, 1, 11)  # Linear variation from -1 to 1
    
    # Define test functions
    test_functions = {
        'cos(u)': jnp.cos,
        'sin(u)': jnp.sin,
        'exp(u)': jnp.exp,
        'tanh(u)': jnp.tanh,
    }
    
    print("Testing various nonlinear functions:")
    print("=" * 50)
    
    for name, func in test_functions.items():
        try:
            f_vector = assemble_state_dependent_forcing_vector_cg(x_nodes, u_dofs, func)
            norm = jnp.linalg.norm(f_vector)
            print(f"{name:12s}: L2 norm = {norm:10.6f}")
        except Exception as e:
            print(f"{name:12s}: Error - {str(e)}")


# Convergence test with different numbers of Gauss points
def convergence_test():
    """Test convergence with different numbers of Gauss points"""
    
    # Fixed mesh and solution
    x_nodes = jnp.linspace(0.0, 1.0, 21)  # 20 elements
    u_dofs = jnp.sin(jnp.pi * x_nodes)
    
    # Test with a more complex function
    complex_func = lambda u: jnp.sin(u)*jnp.exp(-u**2) + u**2
    
    print("Convergence test with complex function f(u) = sin(u)*exp(-u²) + u²:")
    print("=" * 70)
    
    f_reference = assemble_state_dependent_forcing_vector_cg(x_nodes, u_dofs, complex_func, n_gauss_points=4)
    for n_gp in [1, 2, 3]:
        f_gp = assemble_state_dependent_forcing_vector_cg(x_nodes, u_dofs, complex_func, n_gauss_points=n_gp)
    
        error = jnp.linalg.norm(f_gp - f_reference) / jnp.linalg.norm(f_reference)
        print(f"Gauss points: {n_gp}, L2 norm: {jnp.linalg.norm(f_gp):.6f}, "
                  f"Relative error: {error:.2e}")


if __name__ == "__main__":
    # Run examples
    test_common_functions()
    print("\n" + "="*60 + "\n")
    convergence_test()