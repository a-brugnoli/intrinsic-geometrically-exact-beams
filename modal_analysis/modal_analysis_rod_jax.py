"""
Script Name: modal_analysis_rod_jax.py

Description:
    JAX-compatible version: Computes the eigenvalues and eigenvectors of the 1D wave equation using
    interconnection of two port-Hamiltonian systems with automatic differentiation support.

Inputs:
    - None
    
Outputs:
    - Visualizations of the eigenfunctions for the left and right domains

Dependencies:
    - jax, jaxlib, matplotlib 
Usage:

Author: Andrea Brugnoli (JAX conversion)
Date: 29 July 2025
"""
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import numpy as np  # Only for matplotlib and file operations
from math import pi
import matplotlib.pyplot as plt
import os
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
from src.fem.matrices_assembly_jax import mass_matrix_lagrange, discrete_gradient
from src.postprocessing.plot_dg import plot_piecewise_constant

# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)

@jit
def setup_mixed_matrices(coordinates_left, coordinates_right, 
                        axial_compliance, density_unit_length):
    """JAX-compiled function to setup mixed matrices"""
    
    n_nodes_l = coordinates_left.shape[0]
    n_elements_l = n_nodes_l - 1
    n_nodes_r = coordinates_right.shape[0]
    n_elements_r = n_nodes_r - 1
    
    n_dofs_l = n_elements_l + n_nodes_l
    n_dofs_r = n_elements_r + n_nodes_r

    # Mass matrices
    M_cg_left = mass_matrix_lagrange(coordinates_left, axial_compliance)
    D_left = discrete_gradient(coordinates_left)
    
    # Boundary matrices
    B_left = jnp.zeros((n_dofs_l, 1))
    B_left = B_left.at[-1, 0].set(1.0)

    M_cg_right = mass_matrix_lagrange(coordinates_right, density_unit_length)
    D_right = discrete_gradient(coordinates_right)
    
    B_right = jnp.zeros((n_dofs_r, 1))
    B_right = B_right.at[0, 0].set(1.0)

    # Mixed mass matrices
    mesh_sizes_left = jnp.diff(coordinates_left)
    M_diag_left = density_unit_length * mesh_sizes_left
    M_mixed_left_upper = jnp.diag(M_diag_left)
    M_mixed_left = jnp.block([[M_mixed_left_upper, jnp.zeros((n_elements_l, n_nodes_l))],
                              [jnp.zeros((n_nodes_l, n_elements_l)), M_cg_left]])

    mesh_sizes_right = jnp.diff(coordinates_right)
    M_diag_right = axial_compliance * mesh_sizes_right
    M_mixed_right_lower = jnp.diag(M_diag_right)
    M_mixed_right = jnp.block([[M_cg_right, jnp.zeros((n_nodes_r, n_elements_r))],
                               [jnp.zeros((n_elements_r, n_nodes_r)), M_mixed_right_lower]])

    # Interconnection matrices
    zeros_el_el = jnp.zeros((n_elements_l, n_elements_l))
    zeros_n_n_l = jnp.zeros((n_nodes_l, n_nodes_l))
    J_mixed_left = jnp.block([[zeros_el_el, D_left],
                              [-D_left.T, zeros_n_n_l]])

    zeros_n_n_r = jnp.zeros((n_nodes_r, n_nodes_r))
    zeros_er_er = jnp.zeros((n_elements_r, n_elements_r))
    J_mixed_right = jnp.block([[zeros_n_n_r, -D_right.T],
                               [D_right, zeros_er_er]])

    # Assembly
    total_dofs = n_dofs_l + n_dofs_r
    M_mixed = jnp.block([[M_mixed_left, jnp.zeros((n_dofs_l, n_dofs_r))],
                         [jnp.zeros((n_dofs_r, n_dofs_l)), M_mixed_right]])
    
    J_mixed = jnp.block([[J_mixed_left, jnp.zeros((n_dofs_l, n_dofs_r))],
                         [jnp.zeros((n_dofs_r, n_dofs_l)), J_mixed_right]])

    # Interconnection
    J_interconnection = jnp.zeros((total_dofs, total_dofs))
    
    # Add interconnection terms
    B_left_flat = B_left.reshape(-1)
    B_right_flat = B_right.reshape(-1)
    
    # Upper right block: B_left @ B_right.T
    interconnection_ur = jnp.outer(B_left_flat, B_right_flat)
    J_interconnection = J_interconnection.at[:n_dofs_l, n_dofs_l:].set(interconnection_ur)
    
    # Lower left block: -B_right @ B_left.T
    interconnection_ll = -jnp.outer(B_right_flat, B_left_flat)
    J_interconnection = J_interconnection.at[n_dofs_l:, :n_dofs_l].set(interconnection_ll)

    J_mixed = J_mixed + J_interconnection

    return M_mixed, J_mixed

@jit
def compute_eigenvalues_and_vectors(M_mixed, J_mixed):
    """JAX-compiled eigenvalue computation"""
    # Use JAX's eigenvalue solver
    eigenvalues, eigenvectors = jnp.linalg.eig(jnp.linalg.solve(M_mixed, J_mixed))
    return eigenvalues, eigenvectors

def modal_analysis_jax(folder_results, length=2*pi, n_elements=100, n_eigs=10, 
                       cross_section=1, density=1, young_modulus=1):
    """
    Main function for modal analysis with JAX
    
    Parameters can be differentiated w.r.t. using JAX autodiff
    """
    
    # Derived parameters
    density_unit_length = density * cross_section
    axial_compliance = 1./(young_modulus * cross_section)

    assert n_elements%2==0
    n_elements_l = n_elements//2
    n_elements_r = n_elements//2
    n_nodes_l = n_elements_l + 1
    n_nodes_r = n_elements_r + 1

    n_dofs_l = n_elements_l + n_nodes_l

    coordinates_left = jnp.linspace(0, length/2, n_nodes_l)
    midpoints_left = 0.5*(coordinates_left[:-1] + coordinates_left[1:])
    coordinates_right = jnp.linspace(length/2, length, n_nodes_r)

    # Setup matrices
    M_mixed, J_mixed = setup_mixed_matrices(coordinates_left, coordinates_right,
                                             axial_compliance, density_unit_length)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = compute_eigenvalues_and_vectors(M_mixed, J_mixed)
    
    # Process eigenvalues (find positive imaginary parts)
    pos_imag_mask = jnp.imag(eigenvalues) >= 0
    pos_imag_eigenvalues = jnp.imag(eigenvalues)
    pos_imag_eigenvalues = jnp.where(pos_imag_mask, pos_imag_eigenvalues, jnp.inf)
    
    # Get smallest eigenvalues
    indexes_smallest_eigs = jnp.argsort(pos_imag_eigenvalues)[:n_eigs]
    omega_vec = pos_imag_eigenvalues[indexes_smallest_eigs]
    
    # Get corresponding eigenvectors
    smallest_eigenvectors = eigenvectors[:, indexes_smallest_eigs]
    
    # Plot results
    for i in range(n_eigs):
        print(f"Numerical Omega {i+1}: {omega_vec[i]}")
        om_analytical = (2*i+1)*pi/(2*length)
        print(f'Analytical Omega: {om_analytical}')
        
        disp_l_real = np.real(smallest_eigenvectors[:n_elements_l, i])
        disp_l_imag = np.imag(smallest_eigenvectors[:n_elements_l, i])

        if np.linalg.norm(disp_l_real) < 1e-10:
            values_disp_l = disp_l_imag
        else:
            values_disp_l = disp_l_real
        
        disp_r_real = np.real(smallest_eigenvectors[n_dofs_l:n_dofs_l+n_nodes_r, i])
        disp_r_imag = np.imag(smallest_eigenvectors[n_dofs_l:n_dofs_l+n_nodes_r, i])
        if np.linalg.norm(disp_r_real) < 1e-10:
            values_disp_r = disp_r_imag
        else:
            values_disp_r = disp_r_real

        fig = plt.figure()
        axes = fig.add_subplot()

        # Use the JAX-compatible plotting function
        x_plot_left, y_plot_left = plot_piecewise_constant(coordinates_left, values_disp_l)
        plt.plot(x_plot_left, y_plot_left, label=r"$\mathbb{DG}_0$")

        # plt.plot(midpoints_left, values_disp_l, label=r"$\mathbb{DG}_0$")
        plt.plot(coordinates_right, values_disp_r, label=r"$\mathbb{L}_1$")
        plt.legend(loc="best")
        axes.set_xlabel("x")
        axes.set_title(f"Eigenfunction {i+1}, $\omega$ = {omega_vec[i]:.4f}")
        plt.tight_layout()
        plt.savefig(folder_results + f"eigenfunction_{i+1}.pdf", format="pdf")
    
    plt.show()
def main():
    """Main execution function"""
    
    # Configure matplotlib if available
    try:
        from src.postprocessing.plot_config import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        pass
    
    # Setup output directory
    path_file = os.path.dirname(os.path.abspath(__file__))
    folder_results = path_file + "/results/"
    if not os.path.exists(folder_results):
        os.makedirs(folder_results)
    modal_analysis_jax(folder_results=folder_results)

if __name__ == "__main__":
    main()
    