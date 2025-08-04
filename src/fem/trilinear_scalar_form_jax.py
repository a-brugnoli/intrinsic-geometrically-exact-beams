import jax
import jax.numpy as jnp
from src.fem.basis_functions_jax import phi_ref
from src.fem.quadrature import _gauss_quadrature_two_point

@jax.jit
def _element_trilinear_scalar_form_cg(xL, xR, u_local, w_local):
    # Same as before but returns shape (2,)
    xi_q, w_q = _gauss_quadrature_two_point()
    h = xR - xL
    jac = 0.5 * h

    phi_vals = jnp.stack([phi_ref(0, xi_q), phi_ref(1, xi_q)])

    def eval_interp(dofs):
        return jnp.where(
            dofs.shape[0] == 2,
            dofs[0] * phi_vals[0] + dofs[1] * phi_vals[1],
            jnp.full_like(xi_q, dofs[0])
        )

    u_vals = eval_interp(u_local)
    w_vals = eval_interp(w_local)
    integrands = jac * u_vals * w_vals * phi_vals
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

    def eval_interp(dofs):
        return jnp.where(
            dofs.shape[0] == 2,
            dofs[0] * 0.5 * (1 - xi_q) + dofs[1] * 0.5 * (1 + xi_q),
            jnp.full_like(xi_q, dofs[0])
        )

    u_vals = eval_interp(u_local)
    w_vals = eval_interp(w_local)
    integrands = jac * u_vals * w_vals
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

    def body(i, acc):
        xL, xR = x_nodes[i], x_nodes[i + 1]
        u_local = jax.lax.dynamic_slice(u_dofs, (i,), (2,)) if u_is_p1 else jnp.array([u_dofs[i]])
        w_local = jax.lax.dynamic_slice(w_dofs, (i,), (2,)) if w_is_p1 else jnp.array([w_dofs[i]])

        local = _element_trilinear_scalar_form_dg(xL, xR, u_local, w_local)

        acc = acc.at[i].add(local[0])
        return acc

    return jax.lax.fori_loop(0, n_elements, body, form)


if __name__ == "__main__":
    N_els = 3
    x_nodes = jnp.linspace(0.0, 1, N_els + 1)  # 3 elements
    x_mids = 0.5 * (x_nodes[:-1] + x_nodes[1:])

    u_dofs_p1 = jnp.cos(jnp.pi * x_nodes)  # P1 interpolation
    u_dofs_p0 = jnp.cos(jnp.pi * x_mids)  # P1 interpolation

    w_dofs_p1 = jnp.sin(jnp.pi * x_nodes)
    w_dofs_p0 = jnp.sin(jnp.pi * x_mids)

    # P1 interpolation for u_h
    L1 = assemble_trilinear_scalar_form_cg(x_nodes, u_dofs_p1, w_dofs_p1)
    print("P1 Linear Form (P1 u) :", L1)

    # P1 interpolation for u_h
    L0 = assemble_trilinear_scalar_form_cg(x_nodes, u_dofs_p0, w_dofs_p1)
    print("P1 Linear Form (P0 u) :", L0)


    jacobian_L1_wrt_u = jax.jacrev(assemble_trilinear_scalar_form_cg, argnums=1)
    dL1_du = jacobian_L1_wrt_u(x_nodes, u_dofs_p1, w_dofs_p1)

    # The convention for the Jacobian is J_{ij} = ∂L_i/∂u_j

    print("Jacobian of P1 Linear Form w.r.t. u_dofs (P1 u): \n", dL1_du)