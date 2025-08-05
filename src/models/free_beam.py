import numpy as np
import jax.numpy as jnp
import jax.scipy.linalg as jla
from src.fem.matrices_assembly import mass_matrix_lagrange, discrete_gradient, inner_product_cg_dg
from src.linear_algebra.la_jax import skew_block_matrix

class PlanarFreeBeam:
    """
    Represents a geometrically exact planar free beam 
    """

    def __init__(self, n_elements, geometry_props, material_props):
        """
        Initialize the beam finite element model.
        
        Parameters:
        -----------
        n_elements : int
            Number of finite elements
        material_props : dict
            Material properties: {'E': Young's modulus, 'G': shear modulus, 'rho': density}
        geometry_props : dict
            Geometric properties: {'L': length, 'A': cross-sectional area, 
                                   'I': second moment of area, 'k': shear coefficient}

        """
        self.n_elem = n_elements
        self.n_nodes = n_elements + 1
        self.total_dofs = 2 * self.n_nodes + 2 * self.n_elem

        # Geometric properties
        self.L = geometry_props['L']
        self.A = geometry_props['A']
        self.I = geometry_props['I']
        self.k = geometry_props['k']
        
        # Element length
        self.h_mesh = self.L / self.n_elem
        
        # Material properties
        self.E = material_props['E']
        self.G = material_props['G']
        self.rho = material_props['rho']

        # Physical quantities for the dynamical system
        self.density_unit_length = self.rho * self.A
        self.rotational_inertia = self.rho * self.I

        self.axial_compliance = 1 / (self.E * self.A)
        self.shear_compliance = 1 / (self.k * self.G * self.A)
        self.bending_compliance = 1 / (self.E * self.I)
        
        # Node coordinates
        self.x_nodes = jnp.linspace(0, self.L, self.n_nodes)
        
        # Initialize system matrices
        self.H_matrix = None  # Energy matrix
        self.J_linear_matrix = None  # Linear part 
        self.J_nonlinear_matrix = None  # Nonlinear part
        self.B_matrix = None

        self._assemble_energy_matrix()
        self._assemble_linear_interconnection_matrix()

    def _assemble_energy_matrix(self) -> jnp.ndarray:
        """
        Build the energy matrix for the beam.
        
        Returns:
        --------
        H : jnp.ndarray
            The energy matrix of the beam
        """
        M_lin_vel = mass_matrix_lagrange(self.x_nodes, parameter=self.density_unit_length)
        M_ang_vel = mass_matrix_lagrange(self.x_nodes, parameter=self.rotational_inertia)

        M_axial_res = self.h_mesh * self.axial_compliance * jnp.eye(self.n_elem)
        M_shear_res = self.h_mesh * self.shear_compliance * jnp.eye(self.n_elem)
        M_bending_res = self.h_mesh * self.bending_compliance * jnp.eye(self.n_elem)

        self.H_matrix = jla.block_diag(
            M_lin_vel,    # x component of velocity
            M_lin_vel,    # y component of velocity
            M_ang_vel,    # angular velocity
            M_axial_res,  # x component of resultant (axial force)
            M_shear_res,  # y component of resultant (shear force)
            M_bending_res # bending moment
        )
        return 
    
    def _assemble_linear_interconnection_matrix(self) -> jnp.ndarray:
        """
        Build the linear interconnection matrix for the beam.
        
        Returns:
        --------
        J_linear : jnp.ndarray
            The linear interconnection matrix of the beam
        """
        D = discrete_gradient(self.x_nodes)
        M_cg_dg = inner_product_cg_dg(self.x_nodes)
        zeros_n_el = jnp.zeros((self.n_nodes, self.n_elem))

        D_free = jnp.block([
            [-D.T, zeros_n_el, zeros_n_el],
            [zeros_n_el, -D.T, zeros_n_el],
            [zeros_n_el, M_cg_dg, -D.T]
        ])

        self.J_linear_matrix = skew_block_matrix(D_free)


    def _build_nonlinear_dynamics(self, state_vector) -> jnp.ndarray:

        raise NotImplementedError("Nonlinear dynamics not implemented yet.")