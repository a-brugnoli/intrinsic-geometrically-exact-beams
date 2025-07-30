"""
Script Name: my_script.py

Description:
    Computes the eigenvalues and eigenvectors of the 1D wave equation using
    interconnection of two port-Hamiltonian systems. 

Inputs:
    - None
    
Outputs:
    - Visualizations of the eigenfunctions for the left and right domains

Dependencies:
    - numpy, scipy, matplotlib 
Usage:

Author: Andrea Brugnoli
Date: 28 July 2025
"""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from math import pi
import matplotlib.pyplot as plt
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()
import os
from src.fem.matrices_assembly import mass_matrix_lagrange, discrete_gradient
from src.postprocessing.plot_dg import plot_piecewise_constant


path_file = os.path.dirname(os.path.abspath(__file__))
folder_results = path_file + "/results/"
if not os.path.exists(folder_results):
    os.makedirs(folder_results)



def setup_mixed_matrices(coordinates_left, coordinates_right, 
                        axial_compliance, density_unit_length):

    n_nodes_l = coordinates_left.shape[0]
    n_elements_l = n_nodes_l - 1
    n_nodes_r = coordinates_right.shape[0]
    n_elements_r = n_nodes_r - 1
    
    n_dofs_l = n_elements_l + n_nodes_l
    n_dofs_r = n_elements_r + n_nodes_r

    M_cg_left = mass_matrix_lagrange(coordinates_left, axial_compliance)
    D_left = discrete_gradient(coordinates_left)
    B_left = np.zeros((n_dofs_l, 1))
    B_left[-1] = 1

    M_cg_right = mass_matrix_lagrange(coordinates_right, density_unit_length)
    D_right = discrete_gradient(coordinates_right)
    B_right = np.zeros((n_dofs_r, 1))
    B_right[0] = 1

    mesh_sizes_left = np.diff(coordinates_left)
    H_mixed_left = sp.block_diag([density_unit_length * sp.diags(mesh_sizes_left, offsets=0), M_cg_left])

    mesh_sizes_right = np.diff(coordinates_right)
    H_mixed_right = sp.block_diag([M_cg_right, axial_compliance * sp.diags(mesh_sizes_right, offsets=0)])

    row1_left = sp.hstack([sp.csr_matrix((n_elements_l, n_elements_l)), D_left])
    row2_left = sp.hstack([-D_left.T, sp.csr_matrix((n_nodes_l, n_nodes_l))])
    J_mixed_left = sp.vstack([row1_left, row2_left])

    row1_right = sp.hstack([sp.csr_matrix((n_nodes_r, n_nodes_r)), -D_right.T])
    row2_right = sp.hstack([D_right, sp.csr_matrix((n_elements_r, n_elements_r))])
    J_mixed_right = sp.vstack([row1_right, row2_right])

    H_mixed = sp.block_diag([H_mixed_left, H_mixed_right])
    J_mixed = sp.block_diag([J_mixed_left, J_mixed_right])

    J_interconnection = sp.lil_matrix((n_dofs_l + n_dofs_r, n_dofs_l + n_dofs_r))
    J_interconnection[:n_dofs_l, n_dofs_l:] = B_left @ B_right.T
    J_interconnection[n_dofs_l:, :n_dofs_l] = - B_right @ B_left.T

    J_mixed += J_interconnection

    return H_mixed, J_mixed


# Parameters
length = 2*pi  # Length of the domain (m)
cross_section = 1  # Cross section area (m^2)
density = 1  # Density of the fluid (kg/m^3)
young_modulus = 1 # Young modulus (N/m^2)

n_elements = 100  # Number of spatial elements

n_nodes = n_elements + 1
# Derived parameters
density_unit_length = density * cross_section
axial_compliance = 1./(young_modulus * cross_section)

assert n_elements%2==0
n_elements_l = n_elements//2
n_elements_r = n_elements//2
n_nodes_l = n_elements_l + 1
n_nodes_r = n_elements_r + 1

n_dofs_l = n_elements_l + n_nodes_l
n_dofs_r = n_elements_r + n_nodes_r

n_dofs = n_dofs_l + n_dofs_r

coordinates_left = np.linspace(0, length/2, n_nodes_l)
midpoints_left = 0.5*(coordinates_left[:-1] + coordinates_left[1:])

coordinates_right = np.linspace(length/2, length, n_nodes_r)

H_mixed, J_mixed = setup_mixed_matrices(coordinates_left, coordinates_right,
                                         axial_compliance, density_unit_length)
n_eigs = 10
eigenvalues, eigenvectors = la.eig(J_mixed.todense(), H_mixed.todense()) 
pos_imag_indexes = np.imag(eigenvalues) >= 0
pos_imag_eigenvalues = np.imag(eigenvalues)[pos_imag_indexes]
indexes_smallest_eigs = np.argsort(pos_imag_eigenvalues)[:n_eigs]
omega_vec = pos_imag_eigenvalues[indexes_smallest_eigs]

pos_imag_eigenvectors = eigenvectors[:, pos_imag_indexes]
smallest_eigenvectors = pos_imag_eigenvectors[:, indexes_smallest_eigs]

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

    # x_plot_left, y_plot_left = plot_piecewise_constant(coordinates_left, values_disp_l)
    # plt.plot(x_plot_left, y_plot_left, label=r"$\mathbb{DG}_0$")

    plt.plot(midpoints_left, values_disp_l, label=r"$\mathbb{DG}_0$")
    plt.plot(coordinates_right, values_disp_r, label=r"$\mathbb{L}_1$")
    plt.legend(loc="best")
    axes.set_xlabel("x")
    axes.set_title(f"Eigenfunction {i+1}, $\omega$ = {omega_vec[i]:.4f}")
    plt.tight_layout()
    plt.savefig(folder_results + f"eigenfunction_{i+1}.pdf", format="pdf")
plt.show()
