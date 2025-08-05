import jax
import jax.numpy as jnp
from src.fem.trilinear_scalar_form import assemble_trilinear_scalar_form_cg, \
                                    assemble_trilinear_scalar_form_dg


@jax.jit
def assemble_trilinear_2d_vector_form_cg(x_nodes, u_dofs, w_dofs):
    """
    Assemble the global linear form ∫ (ψ_j · u_h) * w_h dx over a 1D mesh
    using continuous (P1) test functions and vector-valued u_h.
    The construction piles together the contributions from each scalar component.

    Supports:
    - u_h as either linear (P1) or piecewise constant (P0) vector-valued interpolant
    - w_h as either linear (P1) or piecewise constant (P0) scalar-valued interpolant
    - Test functions are assumed to be P1 and vector-valued

    Args:
        x_nodes: jnp.ndarray of shape (N+1,)
            Mesh node coordinates
        u_dofs: jnp.ndarray of shape (N+1, 2) or (N, 2)
            Global degrees of freedom for u_h
        w_dofs: jnp.ndarray of shape (N+1,) or (N,)
            Global degrees of freedom for w_h

    Returns:
        jnp.ndarray of shape (N+1,)
            Assembled load vector
    """
    n_nodes = x_nodes.shape[0]
    len_u = u_dofs.shape[0]
    len_w = w_dofs.shape[0]
    assert len_u in (n_nodes, n_nodes - 1), "u_dofs must be of length N+1 or N"
    assert len_w in (n_nodes, n_nodes - 1), "w_dofs must be of length N+1 or N"
    assert u_dofs.shape[1] == 2, "u_dofs must be vector-valued (shape (N+1, 2) or (N, 2))"
    assert w_dofs.ndim == 1, "w_dofs must be scalar-valued (shape (N+1,) or (N,))"

    # The load vector will have shape (2 * n_nodes,)
    # The components are piled consequtively: (l_x, l_y)
    u_dofs_x = u_dofs[:, 0]
    u_dofs_y = u_dofs[:, 1]

    form_x = assemble_trilinear_scalar_form_cg(x_nodes, u_dofs_x, w_dofs)   
    form_y = assemble_trilinear_scalar_form_cg(x_nodes, u_dofs_y, w_dofs)

    return jnp.concatenate([form_x, form_y], axis=0)


@jax.jit
def assemble_2d_cross_form_cg(x_nodes, u_dofs, w_dofs):
    """
    Assemble the global linear form ∫ (ψ_j · J u_h) * w_h dx

    J = [[0, 1], [-1, 0]] is such that
    u x v = u· J v = u_x v_y - u_y v_x

    Continuous (P1) test functions and vector-valued u_h are considered.
    The construction piles together the contributions from each scalar component.

    Supports:
    - u_h as either linear (P1) or piecewise constant (P0) vector-valued interpolant
    - w_h as either linear (P1) or piecewise constant (P0) scalar-valued interpolant
    - Test functions are assumed to be P1 and vector-valued

    Args:
        x_nodes: jnp.ndarray of shape (N+1,)
            Mesh node coordinates
        u_dofs: jnp.ndarray of shape (N+1, 2) or (N, 2)
            Global degrees of freedom for u_h
        w_dofs: jnp.ndarray of shape (N+1,) or (N,)
            Global degrees of freedom for w_h

    Returns:
        jnp.ndarray of shape (N+1,)
            Assembled load vector
    """
    n_nodes = x_nodes.shape[0]
    len_u = u_dofs.shape[0]
    len_w = w_dofs.shape[0]
    assert len_u in (n_nodes, n_nodes - 1), "u_dofs must be of length N+1 or N"
    assert len_w in (n_nodes, n_nodes - 1), "w_dofs must be of length N+1 or N"
    assert u_dofs.shape[1] == 2, "u_dofs must be vector-valued (shape (N+1, 2) or (N, 2))"
    assert w_dofs.ndim == 1, "w_dofs must be scalar-valued (shape (N+1,) or (N,))"

    # The load vector will have shape (2 * n_nodes,)
    # The components are piled consequtively: (l_x, l_y)
    u_dofs_x = u_dofs[:, 0]
    u_dofs_y = u_dofs[:, 1]

    form_x = assemble_trilinear_scalar_form_cg(x_nodes, u_dofs_x, w_dofs)   
    form_y = assemble_trilinear_scalar_form_cg(x_nodes, u_dofs_y, w_dofs)

    return jnp.concatenate([form_y, -form_x], axis=0)


@jax.jit
def assemble_trilinear_2d_vector_form_dg(x_nodes, u_dofs, w_dofs):
    """
    Assemble the global linear form ∫ (ψ_j · u_h) * w_h dx over a 1D mesh
    
    Discontinuous (P0) test functions and vector-valued u_h are considered.
    The construction piles together the contributions from each scalar component.

    Supports:
    - u_h as either linear (P1) or piecewise constant (P0) vector-valued interpolant
    - w_h as either linear (P1) or piecewise constant (P0) scalar-valued interpolant
    - Test functions are assumed to be P1 and vector-valued

    Args:
        x_nodes: jnp.ndarray of shape (N+1,)
            Mesh node coordinates
        u_dofs: jnp.ndarray of shape (N+1, 2) or (N, 2)
            Global degrees of freedom for u_h
        w_dofs: jnp.ndarray of shape (N+1,) or (N,)
            Global degrees of freedom for w_h

    Returns:
        jnp.ndarray of shape (N+1,)
            Assembled load vector
    """
    n_nodes = x_nodes.shape[0]
    len_u = u_dofs.shape[0]
    len_w = w_dofs.shape[0]
    assert len_u in (n_nodes, n_nodes - 1), "u_dofs must be of length N+1 or N"
    assert len_w in (n_nodes, n_nodes - 1), "w_dofs must be of length N+1 or N"
    assert u_dofs.shape[1] == 2, "u_dofs must be vector-valued (shape (N+1, 2) or (N, 2))"
    assert w_dofs.ndim == 1, "w_dofs must be scalar-valued (shape (N+1,) or (N,))"

    # The load vector will have shape (2 * n_nodes,)
    # The components are piled consequtively: (l_x, l_y)
    u_dofs_x = u_dofs[:, 0]
    u_dofs_y = u_dofs[:, 1]
    form_x = assemble_trilinear_scalar_form_dg(x_nodes, u_dofs_x, w_dofs)   
    form_y = assemble_trilinear_scalar_form_dg(x_nodes, u_dofs_y, w_dofs)

    return jnp.concatenate([form_x, form_y], axis=0)


@jax.jit
def assemble_2d_cross_form_dg(x_nodes, u_dofs, w_dofs):
    """
    Assemble the global linear form ∫ (ψ_j · J u_h) * w_h dx 

    J = [[0, 1], [-1, 0]] is such that
    u x v = u· J v = u_x v_y - u_y v_x

    Discontinuous (P0) test functions and vector-valued u_h are considered.
    The construction piles together the contributions from each scalar component.

    Supports:
    - u_h as either linear (P1) or piecewise constant (P0) vector-valued interpolant
    - w_h as either linear (P1) or piecewise constant (P0) scalar-valued interpolant
    - Test functions are assumed to be P1 and vector-valued

    Args:
        x_nodes: jnp.ndarray of shape (N+1,)
            Mesh node coordinates
        u_dofs: jnp.ndarray of shape (N+1, 2) or (N, 2)
            Global degrees of freedom for u_h
        w_dofs: jnp.ndarray of shape (N+1,) or (N,)
            Global degrees of freedom for w_h

    Returns:
        jnp.ndarray of shape (N+1,)
            Assembled load vector
    """
    n_nodes = x_nodes.shape[0]
    len_u = u_dofs.shape[0]
    len_w = w_dofs.shape[0]
    assert len_u in (n_nodes, n_nodes - 1), "u_dofs must be of length N+1 or N"
    assert len_w in (n_nodes, n_nodes - 1), "w_dofs must be of length N+1 or N"
    assert u_dofs.shape[1] == 2, "u_dofs must be vector-valued (shape (N+1, 2) or (N, 2))"
    assert w_dofs.ndim == 1, "w_dofs must be scalar-valued (shape (N+1,) or (N,))"

    # The load vector will have shape (2 * n_nodes,)
    # The components are piled consequtively: (l_x, l_y)
    u_dofs_x = u_dofs[:, 0]
    u_dofs_y = u_dofs[:, 1]
    form_x = assemble_trilinear_scalar_form_dg(x_nodes, u_dofs_x, w_dofs)   
    form_y = assemble_trilinear_scalar_form_dg(x_nodes, u_dofs_y, w_dofs)

    return jnp.concatenate([form_x, form_y], axis=0)

     
if __name__ == "__main__":
    n_elements = 2
    x_nodes = jnp.linspace(0.0, 1.0, n_elements + 1)

    # u_h is vector-valued P1 → shape (N+1, 2)
    u_dofs = jnp.stack([
        jnp.cos(jnp.pi * x_nodes),
        jnp.exp(jnp.pi * x_nodes)
    ], axis=1)  # shape (N+1, 2)

    # w_h is scalar P0 → shape (N,)
    x_midpoints = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    w_dofs = jnp.sin(jnp.pi * x_midpoints)  # shape (N,)

    Lcg = assemble_trilinear_2d_vector_form_cg(x_nodes, u_dofs, w_dofs)
    print("Vector valued form using CG: \n", Lcg)

    jac_Lcg_u = jax.jacrev(assemble_trilinear_2d_vector_form_cg, argnums=1)
    dL1_du = jac_Lcg_u(x_nodes, u_dofs, w_dofs)

    print("Jacobian of CG form w.r.t. u_dofs (P1 u): \n", dL1_du.shape)

    Wcg = assemble_2d_cross_form_cg(x_nodes, u_dofs, w_dofs)
    print("Cross form using CG: \n", Wcg)

    jac_Wcg_u = jax.jacrev(assemble_trilinear_2d_vector_form_cg, argnums=1)
    dL1_du = jac_Wcg_u(x_nodes, u_dofs, w_dofs)

    print("Jacobian of CG form w.r.t. u_dofs (P1 u): \n", dL1_du.shape)

    # L_dg = assemble_trilinear_2d_vector_form_dg(x_nodes, u_dofs, w_dofs)
    # print("Vector valued form using DG:", L_dg)


