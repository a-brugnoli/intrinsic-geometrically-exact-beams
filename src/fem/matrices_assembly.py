import jax.numpy as jnp
import jax
from src.fem.basis_functions import phi_ref, dphi_ref
from src.fem.quadrature import _gauss_quadrature_one_point, _gauss_quadrature_two_point

@jax.jit
def _local_mass_matrix(h_e, parameter) -> jnp.ndarray:
    """Compute 2x2 local mass matrix for element of length h_e.
    Arguments:
        h_e: length of the element
        parameter: scalar parameter for the mass matrix (may be density, compliance, etc.)
    """
    M = jnp.zeros((2, 2))
    jacobian = h_e/2
    gauss_points, gauss_weights = _gauss_quadrature_two_point()
    for i in range(2):
        for j in range(2):
            integrand = parameter * phi_ref(i, gauss_points) * phi_ref(j, gauss_points) * jacobian
            M = M.at[i, j].set(jnp.sum(gauss_weights * integrand) )
    return M


@jax.jit
def mass_matrix_lagrange(coordinates, parameter=1.0) -> jnp.ndarray:
    """Assemble global mass matrix for arbitrary 1D mesh."""
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    M_global = jnp.zeros((n_nodes, n_nodes))

    # JAX efficient for loop
    def body(M_global, e):
        x0, x1 = coordinates[e], coordinates[e + 1]
        h_e = x1 - x0
        M_loc = _local_mass_matrix(h_e, parameter)
        nodes = jnp.array([e, e + 1])

        def add_entry(M, idxs):
            i, j = idxs
            return M.at[nodes[i], nodes[j]].add(M_loc[i, j])

        # Enumerate over all (i, j) in local element
        idxs = jnp.array([[i, j] for i in range(2) for j in range(2)])
        M_global = jax.lax.fori_loop(
            0, 4, lambda k, M: add_entry(M, idxs[k]), M_global
        )

        return M_global, None

    M_global, _ = jax.lax.scan(body, M_global, jnp.arange(n_elements))

    # for e in range(n_elements):
    #     x0, x1 = coordinates[e], coordinates[e + 1]
    #     h_e = x1 - x0
    #     # To be corrected if parameter is not constant
    #     M_loc = _local_mass_matrix(h_e, parameter)
    #     nodes = jnp.array([e, e + 1])
    #     for i in range(2):
    #         for j in range(2):
    #             M_global = M_global.at[nodes[i], nodes[j]].add(M_loc[i, j])

    return M_global


@jax.jit
def _local_inner_product_cg_dg(h_e) -> jnp.ndarray:
    """Compute local vector [∫_K φ_0, ∫_K φ_1]ᵗ for element of length h_e."""
    b = jnp.zeros((2,))
    jacobian = h_e / 2  # Jacobian of affine map
    gauss_points, gauss_weights = _gauss_quadrature_one_point()
    for i in range(2):
        integrand = phi_ref(i, gauss_points) * jacobian
        b = b.at[i].set(jnp.sum(gauss_weights * integrand) )
    return b


@jax.jit
def inner_product_cg_dg(coordinates) -> jnp.ndarray:
    """
    Construct the inner product matrix between continuous piecewise linear 
    (Lagrange) and piecewise constant (DG) finite elements in 1D.

    Parameters:
    -----------
    coordinates: JAX array of node coordinates
        
    Returns:
    --------
    M : jax.numpy.ndarray
        Inner product matrix of shape (n_nodes, n_elements)
        M[i,j] = ∫ φ_i(x) * ψ_j(x) dx
        where φ_i are Lagrange basis functions and ψ_j are DG basis functions
    """
    # Grid setup
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    M_cg_dg = jnp.zeros((n_nodes, n_elements))

    def body(M_cg_dg, e):
        x0, x1 = coordinates[e], coordinates[e + 1]
        h_e = x1 - x0
        M_cg_dg_loc = _local_inner_product_cg_dg(h_e)
        nodes_cg = jnp.array([e, e + 1])

        def add_entry(M, i):
            return M.at[nodes_cg[i], e].add(M_cg_dg_loc[i])

        M_cg_dg = jax.lax.fori_loop(0, 2, lambda i, M: add_entry(M, i), M_cg_dg)
        return M_cg_dg, None

    M_cg_dg, _ = jax.lax.scan(body, M_cg_dg, jnp.arange(n_elements))

    # for e in range(n_elements):
    #     x0, x1 = coordinates[e], coordinates[e + 1]
    #     h_e = x1 - x0
    #     # To be corrected if parameter is not constant
    #     M_cg_dg_loc = _local_inner_product_cg_dg(h_e)
    #     nodes_cg = jnp.array([e, e + 1])
    #     for i in range(2):
    #         M_cg_dg = M_cg_dg.at[nodes_cg[i], e].add(M_cg_dg_loc[i])

    return M_cg_dg


@jax.jit
def _local_discrete_gradient(h_e) -> jnp.ndarray:
    """Compute vector [∫_K dφ_0, ∫_K dφ_1] for element K of length h_e"""
    b = jnp.zeros((2,))
    jacobian = h_e / 2  # Jacobian of affine map
    dxi_dx = 2.0 / h_e
    gauss_points, gauss_weights = _gauss_quadrature_one_point()
    for i in range(2):
        integrand = dphi_ref(i, gauss_points) * jacobian * dxi_dx
        b = b.at[i].set(jnp.sum(gauss_weights * integrand))
    return b


@jax.jit
def discrete_gradient(node_coords) -> jnp.ndarray:
    """
    Assemble D ∈ ℝ^{n_elements × n_nodes}, where
    D[e, j] = ∫_{K_e} φ_j'(x) dx
    """
    n_nodes = node_coords.shape[0]
    n_elements = n_nodes - 1
    D = jnp.zeros((n_elements, n_nodes))

    def body(D, e):
        x0, x1 = node_coords[e], node_coords[e + 1]
        h_e = x1 - x0
        c_loc = _local_discrete_gradient(h_e)
        nodes = jnp.array([e, e + 1])

        def add_entry(D_matrix, i):
            return D_matrix.at[e, nodes[i]].add(c_loc[i])

        # Add two entries to row e of D
        D = jax.lax.fori_loop(0, 2, lambda i, D_: add_entry(D_, i), D)
        return D, None

    D, _ = jax.lax.scan(body, D, jnp.arange(n_elements))

    # for e in range(n_elements):
    #     x0, x1 = node_coords[e], node_coords[e + 1]
    #     h_e = x1 - x0
    #     c_loc = _local_discrete_gradient(h_e)
    #     nodes = jnp.array([e, e + 1])
    #     for i in range(2):
    #         D = D.at[e, nodes[i]].add(c_loc[i])
    return D


@jax.jit
def _local_stiffness_matrix(h_e, stiffness) -> jnp.ndarray:
    """Compute 2x2 local stiffness matrix for element of length h_e."""
    K = jnp.zeros((2, 2))
    jacobian = h_e / 2  # Jacobian of affine map
    dxi_dx = 2.0 / h_e  # inverse Jacobian
    gauss_points, gauss_weights = _gauss_quadrature_two_point()

    for i in range(2):
        for j in range(2):
            integrand = stiffness * dphi_ref(i, gauss_points) * dphi_ref(j, gauss_points) * jacobian * dxi_dx**2
            K = K.at[i, j].set(jnp.sum(gauss_weights * integrand))
    return K

@jax.jit
def stiffness_matrix(coordinates, stiffness=1.0) -> jnp.ndarray:
    """Assemble global stiffness matrix for arbitrary 1D mesh."""
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    K_global = jnp.zeros((n_nodes, n_nodes))

    def body(K_global, e):
        x0, x1 = coordinates[e], coordinates[e + 1]
        h_e = x1 - x0
        K_loc = _local_stiffness_matrix(h_e, stiffness)
        nodes = jnp.array([e, e + 1])

        def add_entry(K, idxs):
            i, j = idxs
            return K.at[nodes[i], nodes[j]].add(K_loc[i, j])

        # Loop over (i, j) in the 2x2 local stiffness matrix
        idxs = jnp.array([[i, j] for i in range(2) for j in range(2)])
        K_global = jax.lax.fori_loop(
            0, 4, lambda k, K: add_entry(K, idxs[k]), K_global
        )

        return K_global, None

    K_global, _ = jax.lax.scan(body, K_global, jnp.arange(n_elements))

    # for e in range(n_elements):
    #     x0, x1 = coordinates[e], coordinates[e + 1]
    #     h_e = x1 - x0
    #     K_loc = _local_stiffness_matrix(h_e, stiffness)
    #     nodes = jnp.array([e, e + 1])
    #     for i in range(2):
    #         for j in range(2):
    #             K_global = K_global.at[nodes[i], nodes[j]].add(K_loc[i, j])
    return K_global


# Example usage and testing
if __name__ == "__main__":
    # Create sample coordinates
    coordinates = jnp.linspace(0.0, 1.0, 3) 
    
    # Compute matrices
    M = mass_matrix_lagrange(coordinates, parameter=1)
    M_cg_dg = inner_product_cg_dg(coordinates)

    D = discrete_gradient(coordinates)

    stiffness = 1
    K = stiffness_matrix(coordinates, stiffness=stiffness)

    print("Mass matrix shape:", M.shape)
    print("Mass matrix CG DG shape:", M_cg_dg.shape)
    print("Gradient matrix shape:", D.shape) 
    print("Stiffness matrix shape:", K.shape)
    
    print("\nMass matrix:")
    print(M)

    print("\nMass matrix CG-DG:")
    print(M_cg_dg)

    print("\nDiscrete gradient matrix:")
    print(D)
    
    print("\nStiffness matrix:")
    print(K)
