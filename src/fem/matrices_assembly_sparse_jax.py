import jax.numpy as jnp
from jax import jit
from jax.experimental import sparse

# Spare matrices are used for efficient storage and computation
# of large matrices, especially in finite element methods.
# Jax.sparse is unfortunately only experimental and may not have all features.

@jit
def mass_matrix_lagrange(coordinates, parameter=1.0):
    """
    Compute the mass matrix using Lagrange elements with JAX sparse matrices.
    
    Args:
        coordinates: JAX array of node coordinates
        parameter: scalar parameter (default 1.0)
    
    Returns:
        Mass matrix as a sparse BCOO matrix
    """
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    
    # Initialize lists to store sparse matrix data
    data = []
    indices = []

    # Numerical quadrature (2-point Gauss quadrature)
    gauss_points = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
    gauss_weights = jnp.array([1.0, 1.0])

    for i in range(n_elements):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left  
        midpoint_element = (x_left + x_right) / 2
        
        element_M = jnp.zeros((2, 2))
        
        for j in range(2):
            gp, gw = gauss_points[j], gauss_weights[j]
            x_gp = midpoint_element + gp * element_length / 2
            N_lagrange = 1/element_length * jnp.array([x_right - x_gp, x_gp - x_left])
            contribution = parameter * gw * element_length / 2 * jnp.outer(N_lagrange, N_lagrange)
            element_M += contribution
        
        # Add element matrix contributions to global matrix data
        for local_i in range(2):
            for local_j in range(2):
                global_i = i + local_i
                global_j = i + local_j
                data.append(element_M[local_i, local_j])
                indices.append((global_i, global_j))

    # Convert to BCOO format
    indices = jnp.array(indices)
    data = jnp.array(data)
    shape = (n_nodes, n_nodes)
    
    return sparse.BCOO((data, indices), shape=shape)


@jit
def discrete_gradient(coordinates):
    """
    Compute the discrete gradient matrix using JAX sparse matrices.
    
    Args:
        coordinates: JAX array of node coordinates
    
    Returns:
        Gradient matrix as a sparse BCOO matrix
    """
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    
    data = []
    indices = []

    for i in range(n_elements):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left  
        dN = 1/element_length * jnp.array([-1.0, 1.0])
        
        # Add contributions for both nodes in the element
        for local_j in range(2):
            global_j = i + local_j
            data.append(element_length * dN[local_j])
            indices.append((i, global_j))

    # Convert to BCOO format
    indices = jnp.array(indices)
    data = jnp.array(data)
    shape = (n_elements, n_nodes)
    
    return sparse.BCOO((data, indices), shape=shape)


@jit
def stiffness_matrix(coordinates, stiffness=1.0):
    """
    Compute the stiffness matrix using JAX sparse matrices.
    
    Args:
        coordinates: JAX array of node coordinates
        stiffness: scalar stiffness parameter (default 1.0)
    
    Returns:
        Stiffness matrix as a sparse BCOO matrix
    """
    n_nodes = coordinates.shape[0]
    n_elements = n_nodes - 1
    
    data = []
    indices = []

    for i in range(n_elements):
        x_left, x_right = coordinates[i], coordinates[i+1]
        element_length = x_right - x_left
        dN = 1/element_length * jnp.array([-1.0, 1.0])
        element_K = stiffness * element_length * jnp.outer(dN, dN)
        
        # Add element matrix contributions to global matrix data
        for local_i in range(2):
            for local_j in range(2):
                global_i = i + local_i
                global_j = i + local_j
                data.append(element_K[local_i, local_j])
                indices.append((global_i, global_j))

    # Convert to BCOO format
    indices = jnp.array(indices)
    data = jnp.array(data)
    shape = (n_nodes, n_nodes)
    
    return sparse.BCOO((data, indices), shape=shape)


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
    
    # Convert to dense for printing (for small matrices)
    print("\nMass matrix (dense representation):")
    print(M.todense())

    print("\nDiscrete gradient matrix (dense representation):")
    print(D.todense())
    
    print("\nStiffness matrix (dense representation):")
    print(K.todense())