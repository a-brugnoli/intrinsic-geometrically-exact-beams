import jax.numpy as jnp
import jax
from jax import jit
import numpy as np

@jit
def gauss_quadrature_one_point():
    """
    Returns the 2-point Gauss quadrature points and weights.
    
    Returns:
        points: JAX array of Gauss points
        weights: JAX array of Gauss weights
    """
    points = jnp.array([0.0])
    weights = jnp.array([2.0])  # Weight for the midpoint in [-1, 1]
    return points, weights



@jit
def gauss_quadrature_two_point():
    """
    Returns the 2-point Gauss quadrature points and weights.
    
    Returns:
        points: JAX array of Gauss points
        weights: JAX array of Gauss weights
    """
    points = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
    weights = jnp.array([1.0, 1.0])
    return points, weights



@jit
def mass_matrix_lagrange(coordinates, parameter=1.0):
    """
    Compute the mass matrix using Lagrange elements with JAX.
    
    Args:
    coordinates: JAX array of node coordinates
    parameter: scalar parameter (default 1.0)
    
    Returns:
    Dense mass matrix as JAX array
    """
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    M = jnp.zeros((n_nodes, n_nodes))

    # Numerical quadrature (2-point Gauss quadrature)
    gauss_points, gauss_weights = gauss_quadrature_two_point()

    def element_contribution(i, M):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left  
        midpoint_element = (x_left + x_right) / 2
        
        def gauss_point_contribution(j, element_M):
            gp, gw = gauss_points[j], gauss_weights[j]
            x_gp = midpoint_element + gp * element_length / 2
            N_lagrange = 1/element_length * jnp.array([x_right - x_gp, x_gp - x_left])
            contribution = parameter * gw * element_length / 2 * jnp.outer(N_lagrange, N_lagrange)
            return element_M + contribution
        
        # Accumulate contributions from both Gauss points
        element_M = jnp.zeros((2, 2))
        for j in range(len(gauss_points)):
            element_M = gauss_point_contribution(j, element_M)
        
        # Add element matrix to global matrix
        M = M.at[i:i+2, i:i+2].add(element_M)
        return M

    # Loop over elements
    for i in range(n_elements):
        M = element_contribution(i, M)

    return M


@jit
def construct_inner_product_cg_dg(coordinates):
    """
    Construct the inner product matrix between continuous piecewise linear 
    (Lagrange) and piecewise constant (DG) finite elements in 1D using
    2-point Gaussian quadrature.
    
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
    
    # 1-point Gauss quadrature points and weights on reference element [-1,1]
    gauss_points, gauss_weights = gauss_quadrature_one_point()
    
    # Grid setup
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    M_cg_dg = jnp.zeros((n_nodes, n_elements))

    def element_contribution(i, M):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left  
        midpoint_element = (x_left + x_right) / 2
        
        def gauss_point_contribution(j, element_M):
            gp, gw = gauss_points[j], gauss_weights[j]
            x_gp = midpoint_element + gp * element_length / 2
            N_lagrange = 1/element_length * jnp.array([x_right - x_gp, x_gp - x_left])

            contribution = gw * element_length / 2 * N_lagrange
            return element_M + contribution
        
        # Accumulate contributions from both Gauss points
        element_M = jnp.zeros((2, ))
        for j in range(len(gauss_points)):
            element_M = gauss_point_contribution(j, element_M)
        
        # Add element matrix to global matrix
        M = M.at[i:i+2, i].add(element_M)
        return M

    # Loop over elements
    for i in range(n_elements):
        M_cg_dg = element_contribution(i, M_cg_dg)

    return M_cg_dg

@jit
def discrete_gradient(coordinates):
    """
    Compute the discrete gradient matrix using JAX.
    
    Args:
        coordinates: JAX array of node coordinates
    
    Returns:
        Dense gradient matrix as JAX array
    """
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1

    D = jnp.zeros((n_elements, n_nodes))

    def element_contribution(i, D):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left  
        dN = 1/element_length * jnp.array([-1.0, 1.0])
        
        row_update = element_length * dN
        D = D.at[i, i:i+2].add(row_update)
        return D

    # Loop over elements
    for i in range(n_elements):
        D = element_contribution(i, D)

    return D


@jit
def stiffness_matrix(coordinates, stiffness=1.0):
    """
    Compute the stiffness matrix using JAX.
    
    Args:
        coordinates: JAX array of node coordinates
        stiffness: scalar stiffness parameter (default 1.0)
    
    Returns:
        Dense stiffness matrix as JAX array
    """
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1

    K = jnp.zeros((n_nodes, n_nodes))

    def element_contribution(i, K):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left
        dN = 1/element_length * jnp.array([-1.0, 1.0])
        element_K = stiffness * element_length * jnp.outer(dN, dN)
        
        K = K.at[i:i+2, i:i+2].add(element_K)
        return K

    # Loop over elements
    for i in range(n_elements):
        K = element_contribution(i, K)

    return K


# Example usage and testing
if __name__ == "__main__":
    # Create sample coordinates
    coordinates = jnp.linspace(0.0, 1.0, 3) 
    
    # Compute matrices
    M = mass_matrix_lagrange(coordinates, parameter=2.0)
    M_cg_dg = construct_inner_product_cg_dg(coordinates)
    D = discrete_gradient(coordinates)
    K = stiffness_matrix(coordinates, stiffness=1.5)
    
    print("Mass matrix shape:", M.shape)
    print("Mass matrix CG DG shape:", M_cg_dg.shape)
    print("Gradient matrix shape:", D.shape) 
    print("Stiffness matrix shape:", K.shape)
    
    # Print a small portion to verify
    print("\nMass matrix:")
    print(M)

    # Print a small portion to verify
    print("\nMass matrix CG-DG:")
    print(M_cg_dg)

    print("\nDiscrete gradient matrix:")
    print(D)
    
    print("\nStiffness matrix:")
    print(K)