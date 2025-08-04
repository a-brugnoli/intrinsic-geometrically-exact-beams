import jax.numpy as jnp
from jax import jit
from basis_functions_jax import phi_ref, dphi_ref
from quadrature import _gauss_quadrature_one_point, _gauss_quadrature_two_point

@jit
def _local_mass_matrix(h_e, parameter):
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


@jit
def mass_matrix_lagrange(coordinates, parameter=1.0):
    """Assemble global mass matrix for arbitrary 1D mesh."""
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    M_global = jnp.zeros((n_nodes, n_nodes))

    for e in range(n_elements):
        x0, x1 = coordinates[e], coordinates[e + 1]
        h_e = x1 - x0
        # To be corrected if parameter is not constant
        M_loc = _local_mass_matrix(h_e, parameter)
        nodes = jnp.array([e, e + 1])
        for i in range(2):
            for j in range(2):
                M_global = M_global.at[nodes[i], nodes[j]].add(M_loc[i, j])
    return M_global


@jit
def _local_inner_product_cg_dg(h_e):
    """Compute local vector [∫_K φ_0, ∫_K φ_1]ᵗ for element of length h_e."""
    b = jnp.zeros((2,))
    jacobian = h_e / 2  # Jacobian of affine map
    gauss_points, gauss_weights = _gauss_quadrature_one_point()
    for i in range(2):
        integrand = phi_ref(i, gauss_points) * jacobian
        b = b.at[i].set(jnp.sum(gauss_weights * integrand) )
    return b


@jit
def inner_product_cg_dg(coordinates):
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

    for e in range(n_elements):
        x0, x1 = coordinates[e], coordinates[e + 1]
        h_e = x1 - x0
        # To be corrected if parameter is not constant
        M_cg_dg_loc = _local_inner_product_cg_dg(h_e)
        nodes_cg = jnp.array([e, e + 1])
        for i in range(2):
            M_cg_dg = M_cg_dg.at[nodes_cg[i], e].add(M_cg_dg_loc[i])
    return M_cg_dg


@jit
def _local_discrete_gradient(h_e):
    """Compute vector [∫_K dφ_0, ∫_K dφ_1] for element K of length h_e"""
    b = jnp.zeros((2,))
    jacobian = h_e / 2  # Jacobian of affine map
    dxi_dx = 2.0 / h_e
    gauss_points, gauss_weights = _gauss_quadrature_one_point()
    for i in range(2):
        integrand = dphi_ref(i, gauss_points) * jacobian * dxi_dx
        b = b.at[i].set(jnp.sum(gauss_weights * integrand))
    return b


@jit
def discrete_gradient(node_coords):
    """
    Assemble D ∈ ℝ^{n_elements × n_nodes}, where
    D[e, j] = ∫_{K_e} φ_j'(x) dx
    """
    n_nodes = node_coords.shape[0]
    n_elements = n_nodes - 1
    D = jnp.zeros((n_elements, n_nodes))

    for e in range(n_elements):
        x0, x1 = node_coords[e], node_coords[e + 1]
        h_e = x1 - x0
        c_loc = _local_discrete_gradient(h_e)
        nodes = jnp.array([e, e + 1])
        for i in range(2):
            D = D.at[e, nodes[i]].add(c_loc[i])
    return D


@jit
def _local_stiffness_matrix(h_e, stiffness):
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

@jit
def stiffness_matrix(coordinates, stiffness=1.0):
    """Assemble global stiffness matrix for arbitrary 1D mesh."""
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    K_global = jnp.zeros((n_nodes, n_nodes))

    for e in range(n_elements):
        x0, x1 = coordinates[e], coordinates[e + 1]
        h_e = x1 - x0
        K_loc = _local_stiffness_matrix(h_e, stiffness)
        nodes = jnp.array([e, e + 1])
        for i in range(2):
            for j in range(2):
                K_global = K_global.at[nodes[i], nodes[j]].add(K_loc[i, j])
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
