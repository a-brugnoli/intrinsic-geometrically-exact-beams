import jax.numpy as jnp
from jax import jit


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
    gauss_points = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
    gauss_weights = jnp.array([1.0, 1.0])

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
        for j in range(2):
            element_M = gauss_point_contribution(j, element_M)
        
        # Add element matrix to global matrix
        M = M.at[i:i+2, i:i+2].add(element_M)
        return M

    # Loop over elements
    for i in range(n_elements):
        M = element_contribution(i, M)

    return M


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
    coordinates = jnp.linspace(0.0, 1.0, 3)  # 3 nodes, 2 elements

    
    # Compute matrices
    M = mass_matrix_lagrange(coordinates, parameter=2.0)
    D = discrete_gradient(coordinates)
    K = stiffness_matrix(coordinates, stiffness=1.5)
    
    print("Mass matrix shape:", M.shape)
    print("Gradient matrix shape:", D.shape) 
    print("Stiffness matrix shape:", K.shape)
    
    # Print a small portion to verify
    print("\nMass matrix:")
    print(M)

    print("\nDiscrete gradient matrix:")
    print(D)
    
    print("\nStiffness matrix:")
    print(K)