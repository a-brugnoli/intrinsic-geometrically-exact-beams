import jax
import jax.numpy as jnp

def compute_potential_energy_nodes(positions, x_nodes, rho, A, g=9.81):
    """
    Compute gravitational potential energy of a beam over time.
    
    Args:
        positions: (2, n_nodes, n_times) array where positions[1] gives y-coordinates
        x_nodes: (n_nodes,) array of node positions along beam axis
        rho: material density
        A: cross-sectional area
        g: gravitational acceleration
    
    Returns:
        PE: (n_times,) array of potential energy at each time step
    """
    n_nodes = len(x_nodes)
    y_coords = positions[1, :, :]  # (n_nodes, n_times) - vertical coordinates
    
    # Compute element lengths in reference configuration
    element_lengths = jnp.diff(x_nodes)
    
    # Compute lumped masses at each node
    masses = jnp.zeros(n_nodes)
    
    # First node (end node)
    masses = masses.at[0].set(rho * A * element_lengths[0] / 2)
    # Interior nodes
    for i in range(1, n_nodes - 1):
        masses = masses.at[i].set(rho * A * (element_lengths[i-1] + element_lengths[i]) / 2)
    # Last node (end node)
    masses = masses.at[-1].set(rho * A * element_lengths[-1] / 2)
    
    # Compute potential energy at each time step
    # masses: (n_nodes,), y_coords: (n_nodes, n_times)
    potential_energy = jnp.sum(masses[:, None] * g * y_coords, axis=0)  # (n_times,)
    
    return potential_energy


def compute_potential_energy_midpoints(positions, x_nodes, rho, A, g=9.81):
    """
    Compute gravitational potential energy of a beam over time.
    Positions are given at element midpoints.
    
    Args:
        positions: (2, n_elements, n_times) array where positions[1] gives y-coordinates
        x_nodes: (n_nodes,) array of node positions along beam axis
        rho: material density
        A: cross-sectional area
        g: gravitational acceleration
    
    Returns:
        PE: (n_times,) array of potential energy at each time step
    """
    y_midpoints = positions[1, :, :]  # (n_elements, n_times) - vertical coordinates at midpoints
    
    # Compute element lengths in reference configuration
    element_lengths = jnp.diff(x_nodes)  # (n_elements,)
    
    # Mass of each element
    element_masses = rho * A * element_lengths  # (n_elements,)
    
    # Compute potential energy at each time step
    # element_masses: (n_elements,), y_midpoints: (n_elements, n_times)
    PE = jnp.sum(element_masses[:, None] * g * y_midpoints, axis=0)  # (n_times,)
    
    return PE
