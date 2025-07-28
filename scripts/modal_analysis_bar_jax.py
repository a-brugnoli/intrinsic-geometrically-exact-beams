import jax.numpy as jnp
from jax import jit
import numpy as np  # Still needed for matplotlib and file operations
from math import pi
import matplotlib.pyplot as plt
from src.fem_jax import mass_matrix_lagrange, discrete_gradient
from scipy.linalg import eig  # For eigenvalue computation - JAX doesn't have generalized eigenvalue solver yet
import os

@jit
def create_block_diagonal(matrices):
    """Create a block diagonal matrix from a list of matrices."""
    if len(matrices) == 1:
        return matrices[0]
    
    total_rows = sum(mat.shape[0] for mat in matrices)
    total_cols = sum(mat.shape[1] for mat in matrices)
    result = jnp.zeros((total_rows, total_cols))
    
    row_offset = 0
    col_offset = 0
    for mat in matrices:
        rows, cols = mat.shape
        result = result.at[row_offset:row_offset+rows, col_offset:col_offset+cols].set(mat)
        row_offset += rows
        col_offset += cols
    
    return result


@jit
def create_block_matrix(blocks):
    """Create a matrix from a 2D list of block matrices."""
    # Assume blocks is a 2x2 list of matrices
    top_row = jnp.hstack([blocks[0][0], blocks[0][1]])
    bottom_row = jnp.hstack([blocks[1][0], blocks[1][1]])
    return jnp.vstack([top_row, bottom_row])

@jit
def setup_mixed_matrices(coordinates_left, coordinates_right, 
                        axial_compliance, density_unit_length):
    """Set up the mixed formulation matrices using JAX operations."""
    
    n_nodes_l = coordinates_left.shape[0]
    n_elements_l = n_nodes_l - 1
    n_nodes_r = coordinates_right.shape[0]
    n_elements_r = n_nodes_r - 1
    
    n_dofs_l = n_elements_l + n_nodes_l
    n_dofs_r = n_elements_r + n_nodes_r
    
    # Left side matrices
    M_left = mass_matrix_lagrange(coordinates_left, axial_compliance)
    D_left = discrete_gradient(coordinates_left)
    # Coupling vectors
    B_left = jnp.zeros((n_dofs_l, 1))
    B_left = B_left.at[n_nodes_l - 1, 0].set(1.0)

    # Right side matrices  
    M_right = mass_matrix_lagrange(coordinates_right, density_unit_length)
    D_right = discrete_gradient(coordinates_right)
    
    B_right = jnp.zeros((n_dofs_r, 1))
    B_right = B_right.at[n_elements_r, 0].set(1.0)

    # Mixed mass matrices
    mesh_sizes_left = jnp.diff(coordinates_left)
    M_mixed_left = create_block_diagonal([
        M_left, 
        density_unit_length * jnp.diag(mesh_sizes_left)
    ])
    
    mesh_sizes_right = jnp.diff(coordinates_right)
    M_mixed_right = create_block_diagonal([
        axial_compliance * jnp.diag(mesh_sizes_right),
        M_right
    ])
    
    # Mixed Jacobian matrices
    zeros_nl_nl = jnp.zeros((n_nodes_l, n_nodes_l))
    zeros_el_el = jnp.zeros((n_elements_l, n_elements_l))
    zeros_er_er = jnp.zeros((n_elements_r, n_elements_r))
    zeros_nr_nr = jnp.zeros((n_nodes_r, n_nodes_r))
    
    J_mixed_left = create_block_matrix([
        [zeros_nl_nl, -D_left.T],
        [D_left, zeros_el_el]
    ])
    
    J_mixed_right = create_block_matrix([
        [zeros_er_er, D_right],
        [-D_right.T, zeros_nr_nr]
    ])
    

    # Combine matrices
    M_mixed = create_block_diagonal([M_mixed_left, M_mixed_right])
    J_mixed = create_block_diagonal([J_mixed_left, J_mixed_right])
    
    # Add interconnection
    n_dofs_total = n_dofs_l + n_dofs_r
    J_interconnection = jnp.zeros((n_dofs_total, n_dofs_total))
    
    # Add coupling terms
    coupling_term = B_left @ B_right.T
    J_interconnection = J_interconnection.at[:n_dofs_l, n_dofs_l:].set(coupling_term)
    J_interconnection = J_interconnection.at[n_dofs_l:, :n_dofs_l].set(-coupling_term.T)
    
    J_mixed = J_mixed + J_interconnection
    
    return M_mixed, J_mixed

def main():
    # Configure matplotlib if available
    try:
        from src.plot_config import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        pass
    
    # Create results folder
    path_file = os.path.dirname(os.path.abspath(__file__))
    folder_results = path_file + "/results/"
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)
    
    # Parameters
    length = 2*pi
    cross_section = 1
    density = 1
    young_modulus = 1
    n_elements = 100
    
    # Derived parameters
    mesh_size = length / n_elements
    density_unit_length = density * cross_section
    axial_compliance = 1.0 / (young_modulus * cross_section)
    
    # Ensure even number of elements for splitting
    assert n_elements % 2 == 0
    n_elements_l = n_elements // 2
    n_elements_r = n_elements // 2
    n_nodes_l = n_elements_l + 1
    n_nodes_r = n_elements_r + 1
    
    n_dofs_l = n_elements_l + n_nodes_l
    n_dofs_r = n_elements_r + n_nodes_r

    # Create coordinate arrays as JAX arrays
    coordinates_left = jnp.linspace(0, length/2, n_nodes_l)
    coordinates_right = jnp.linspace(length/2, length, n_nodes_r)
    
    # Setup mixed matrices using JAX
    M_mixed, J_mixed = setup_mixed_matrices(
        coordinates_left, coordinates_right, axial_compliance, 
        density_unit_length)
    
    # Convert to numpy for eigenvalue computation (JAX doesn't have generalized eig yet)
    J_mixed_np = np.array(J_mixed)
    M_mixed_np = np.array(M_mixed)
    
    # Eigenvalue analysis
    n_eigs = 10
    eigenvalues, eigenvectors = eig(J_mixed_np, M_mixed_np)
    
    # Filter positive imaginary eigenvalues
    pos_imag_indexes = np.imag(eigenvalues) >= 0
    pos_imag_eigenvalues = np.imag(eigenvalues)[pos_imag_indexes]
    indexes_smallest_eigs = np.argsort(pos_imag_eigenvalues)[:n_eigs]
    omega_vec = pos_imag_eigenvalues[indexes_smallest_eigs]
    
    pos_imag_eigenvectors = eigenvectors[:, pos_imag_indexes]
    smallest_eigenvectors = pos_imag_eigenvectors[:, indexes_smallest_eigs]
    
    # Calculate midpoints for plotting
    midpoints_left = np.zeros(n_elements_l)
    coordinates_left_np = np.array(coordinates_left)
    for i in range(len(midpoints_left)):
        midpoints_left[i] = 0.5 * (coordinates_left_np[i] + coordinates_left_np[i+1])
    
    coordinates_right_np = np.array(coordinates_right)
    
    # Plot results
    for i in range(n_eigs):
        print(f"Numerical Omega {i+1}: {omega_vec[i]}")
        om_analytical = (2*i+1)*pi/(2*length)
        print(f'Analytical Omega: {om_analytical}')
        
        # Extract displacement components
        disp_l_real = np.real(smallest_eigenvectors[n_nodes_l:n_dofs_l, i])
        disp_l_imag = np.imag(smallest_eigenvectors[n_nodes_l:n_dofs_l, i])
        
        if np.linalg.norm(disp_l_real) < 1e-10:
            values_disp_l = disp_l_imag
        else:
            values_disp_l = disp_l_real
        
        disp_r_real = np.real(smallest_eigenvectors[n_dofs_l+n_elements_r:, i])
        disp_r_imag = np.imag(smallest_eigenvectors[n_dofs_l+n_elements_r:, i])
        
        if np.linalg.norm(disp_r_real) < 1e-10:
            values_disp_r = disp_r_imag
        else:
            values_disp_r = disp_r_real
        
        # Plot eigenfunction
        fig = plt.figure()
        axes = fig.add_subplot()
        plt.plot(midpoints_left, values_disp_l, label=r"$\mathbb{DG}_0$")
        plt.plot(coordinates_right_np, values_disp_r, label=r"$\mathbb{L}_1$")
        plt.legend()
        axes.set_xlabel("x")
        axes.set_title(f"Eigenfunction {i+1}, $\omega$ = {omega_vec[i]:.4f}")
        plt.tight_layout()
        plt.savefig(folder_results + f"eigenfunction_{i+1}.pdf", format="pdf")
    
    plt.show()

if __name__ == "__main__":
    main()