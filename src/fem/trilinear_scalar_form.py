import jax
import jax.numpy as jnp
from src.fem.basis_functions import phi_ref
from src.fem.quadrature import _gauss_quadrature_two_point, eval_interp_at_gauss_node

@jax.jit
def _element_trilinear_scalar_form_cg(xL, xR, u_local, w_local):
    """
    Compute the element-level trilinear form ∫ phi_i * u_h * w_h dx for a given element.

    This function evaluates the contribution of an element to the global trilinear form
    using continuous (P1) basis functions and Gauss quadrature.

    Args:
        xL: Left coordinate of the element.
        xR: Right coordinate of the element.
        u_local: JAX array of degrees of freedom for the solution u_h within the element.
        w_local: JAX array of degrees of freedom for the weighting function w_h within the element.

    Returns:
        A JAX array representing the element-level contribution to the trilinear form.
    """
    xi_q, w_q = _gauss_quadrature_two_point()
    h = xR - xL
    jac = 0.5 * h

    # Shape (2,2), i.e., [phi_0, phi_1] for P1 and number of Gauss points
    phi_vals = jnp.stack([phi_ref(0, xi_q), phi_ref(1, xi_q)])

    # Shape (2,) i.e. number of gauss points
    u_vals = eval_interp_at_gauss_node(u_local, xi_q)
    w_vals = eval_interp_at_gauss_node(w_local, xi_q)
    integrands = jac * phi_vals * u_vals * w_vals
    print(f"u_vals.shape: {u_vals.shape}, w_vals.shape: {w_vals.shape}, phi_vals.shape: {phi_vals.shape}, integrands.shape: {integrands.shape}")
    return jnp.dot(integrands, w_q)


@jax.jit
def assemble_trilinear_scalar_form_cg(x_nodes: jnp.ndarray,
                                    u_dofs: jnp.ndarray,
                                    w_dofs: jnp.ndarray) -> jnp.ndarray:
    """
    Assemble the linear form L_j = ∫ u_h * w_h * ψ_j dx for continuous (C0) test functions 
    (e.g., Lagrange P1 finite elements) over a 1D mesh.

    Args:
        x_nodes (jnp.ndarray): 1D array of mesh node coordinates, length N+1.
        u_dofs (jnp.ndarray): Degrees of freedom for u_h.
            - If length N+1, u_h is piecewise linear (P1).
            - If length N, u_h is piecewise constant (P0).
        w_dofs (jnp.ndarray): Degrees of freedom for w_h.
            - Same convention as u_dofs.

    Returns:
        jnp.ndarray: Load vector of length N+1 corresponding to the nodal test functions.
    """
    n_nodes = x_nodes.shape[0]
    n_elements = n_nodes - 1
    len_u = u_dofs.shape[0]
    len_w = w_dofs.shape[0]
    assert len_u in (n_nodes, n_nodes - 1), "u_dofs must be of length N+1 or N"
    assert len_w in (n_nodes, n_nodes - 1), "w_dofs must be of length N+1 or N"

    u_is_p1 = len_u == n_nodes
    w_is_p1 = len_w == n_nodes

    form = jnp.zeros(n_nodes)

    def element_contribution(i, acc):
        """"
        Compute the local contribution to the global linear form for each element.
        """
        xL, xR = x_nodes[i], x_nodes[i + 1]
        u_local = jax.lax.dynamic_slice(u_dofs, (i,), (2,)) if u_is_p1 else jnp.array([u_dofs[i]])
        w_local = jax.lax.dynamic_slice(w_dofs, (i,), (2,)) if w_is_p1 else jnp.array([w_dofs[i]])

        local = _element_trilinear_scalar_form_cg(xL, xR, u_local, w_local)

        acc = acc.at[i].add(local[0])
        acc = acc.at[i + 1].add(local[1])
        return acc

    return jax.lax.fori_loop(0, n_elements, element_contribution, form)


def _element_trilinear_scalar_form_dg(xL, xR, u_local, w_local):
    xi_q, w_q = _gauss_quadrature_two_point()
    h = xR - xL
    jac = 0.5 * h

    # Shape (2,) i.e. number of gauss points

    u_vals = eval_interp_at_gauss_node(u_local, xi_q)
    w_vals = eval_interp_at_gauss_node(w_local, xi_q)
    integrands = jac * u_vals * w_vals
    print(f"u_vals.shape: {u_vals.shape}, w_vals.shape: {w_vals.shape}, integrands.shape: {integrands.shape}")
    return jnp.array([jnp.dot(integrands, w_q)])


@jax.jit
def assemble_trilinear_scalar_form_dg(x_nodes: jnp.ndarray,
                                       u_dofs: jnp.ndarray,
                                       w_dofs: jnp.ndarray) -> jnp.ndarray:
    """
    Assemble the linear form L_j = ∫ u_h * w_h * ψ_j dx for discontinuous (C-1) test functions 
    (e.g., piecewise constant finite elements) over a 1D mesh.

    Args:
        x_nodes (jnp.ndarray): 1D array of mesh node coordinates, length N+1.
        u_dofs (jnp.ndarray): Degrees of freedom for u_h.
            - If length N+1, u_h is piecewise linear (P1).
            - If length N, u_h is piecewise constant (P0).
        w_dofs (jnp.ndarray): Degrees of freedom for w_h.
            - Same convention as u_dofs.

    Returns:
        jnp.ndarray: Load vector of length N corresponding to the element-wise test functions.
    """
    n_nodes = x_nodes.shape[0]
    n_elements = n_nodes - 1
    len_u = u_dofs.shape[0]
    len_w = w_dofs.shape[0]
    assert len_u in (n_nodes, n_nodes - 1), "u_dofs must be of length N+1 or N"
    assert len_w in (n_nodes, n_nodes - 1), "w_dofs must be of length N+1 or N"

    u_is_p1 = len_u == n_nodes
    w_is_p1 = len_w == n_nodes

    form = jnp.zeros(n_elements)

    def element_contribution(i, acc):
        xL, xR = x_nodes[i], x_nodes[i + 1]
        u_local = jax.lax.dynamic_slice(u_dofs, (i,), (2,)) if u_is_p1 else jnp.array([u_dofs[i]])
        w_local = jax.lax.dynamic_slice(w_dofs, (i,), (2,)) if w_is_p1 else jnp.array([w_dofs[i]])

        local = _element_trilinear_scalar_form_dg(xL, xR, u_local, w_local)

        acc = acc.at[i].add(local[0])
        return acc

    return jax.lax.fori_loop(0, n_elements, element_contribution, form)


if __name__ == "__main__":
    N_els = 2
    length = 1
    x_nodes = jnp.linspace(0.0, length, N_els + 1)  # 3 elements
    x_mids = 0.5 * (x_nodes[:-1] + x_nodes[1:])

    alpha = 3
    u_dofs_p1 = alpha * x_nodes  # P1 interpolation
    u_dofs_p0 = alpha * x_mids   # P0 interpolation

    beta = 5
    w_dofs_p1 = beta * x_nodes
    w_dofs_p0 = beta * jnp.pi * x_mids

    # P1 interpolation for test
    Lcg = assemble_trilinear_scalar_form_cg(x_nodes, u_dofs_p1, w_dofs_p1)
    print("P1 Linear Form (P1 u): \n", Lcg)
    print("Analytical P1 linear form: \n", alpha * beta * length**3 * jnp.array([1/96, 7/48, 17/96]))

    # Compare with analytical result (P0 linear form)
    assert jnp.allclose(Lcg, alpha * beta * length**3 * jnp.array([1/96, 7/48, 17/96]))

    # P0 interpolation for test
    Ldg = assemble_trilinear_scalar_form_dg(x_nodes, u_dofs_p1, w_dofs_p1)
    print("P0 Linear Form (P1 u): \n", Ldg)
    print("Analytical P0 linear form: \n", alpha * beta * length**3 * jnp.array([1/24, 7/24]))

    # Compare with analytical result (P0 linear form)
    assert jnp.allclose(Ldg, alpha*beta*length**3*jnp.array([1/24, 7/24]))

    # jacobian_Lcg_u = jax.jacrev(assemble_trilinear_scalar_form_cg, argnums=1)
    # dL1_du = jacobian_Lcg_u(x_nodes, u_dofs_p1, w_dofs_p1)

    # # The convention for the Jacobian is J_{ij} = ∂L_i/∂u_j
    # print("Jacobian of P1 Linear Form w.r.t. u_dofs (P1 u): \n", dL1_du)