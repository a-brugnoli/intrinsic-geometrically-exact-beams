import jax.numpy as jnp
import jax.scipy.linalg as jla
from src.fem.matrices_assembly import mass_matrix_lagrange, discrete_gradient, inner_product_cg_dg
from src.fem.trilinear_vector_form import assemble_2d_cross_form_cg, assemble_2d_cross_form_dg, \
                                          assemble_2d_cross_skew_adjoint_form_cg
from src.linear_algebra.la_jax import skew_block_matrix
import jax
from functools import partial
from .intrinsic_generic_beam import IntrinsicBeam

class IntrinsicPlanarFreeBeam(IntrinsicBeam):
    """
    Represents a geometrically exact planar free beam 
    """

    def __init__(self, n_elements, geometrical_props, material_props):
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
        super().__init__(n_elements, geometrical_props, material_props)

        # Dimension of different partitions
        self.n_lin_vel_x = self.n_nodes
        self.n_lin_vel = 2 * self.n_nodes
        self.n_ang_vel = self.n_nodes
        self.n_vel = self.n_lin_vel + self.n_ang_vel

        self.n_force_x = self.n_elem
        self.n_force = 2 * self.n_elem
        self.n_moment = self.n_elem   

        self.n_coenergy = self.n_lin_vel + self.n_ang_vel + self.n_force + self.n_moment

        # self.H_matrix = self.assemble_energy_matrix()
        self.J_linear_matrix = self.assemble_linear_interconnection_matrix()
        # self.E_matrix = self.assemble_E_matrix()



    @partial(jax.jit, static_argnums=0)
    def assemble_energy_matrix(self) -> jnp.ndarray:
        """
        Assemble the energy and interconnection matrix for the beam
        
        Returns:
        --------
        H : jnp.ndarray
            The energy matrix of the beam
        """
        M_lin_vel = mass_matrix_lagrange(self.x_nodes, parameter=self.density_unit_length)
        M_ang_vel = mass_matrix_lagrange(self.x_nodes, parameter=self.rotational_inertia)

        M_axial_for = self.h_mesh * self.axial_compliance * jnp.eye(self.n_elem)
        M_shear_for = self.h_mesh * self.shear_compliance * jnp.eye(self.n_elem)
        M_moment = self.h_mesh * self.bending_compliance * jnp.eye(self.n_elem)

        H_matrix = jla.block_diag(
            M_lin_vel,       # x component of velocity
            M_lin_vel,       # y component of velocity
            M_ang_vel,       # angular velocity
            M_axial_for,     # x component of resultant (axial force)
            M_shear_for,     # y component of resultant (shear force)
            M_moment         # bending moment
        )

        return H_matrix

    def assemble_linear_interconnection_matrix(self) -> jnp.ndarray:
        """
        Assemble the energy and interconnection matrix for the beam
        
        Returns:
        --------
        J_linear : jnp.ndarray
            The linear interconnection matrix of the beam
        """
        D = discrete_gradient(self.x_nodes)
        M_cg_dg = inner_product_cg_dg(self.x_nodes)
        zeros_nodes_el = jnp.zeros((self.n_nodes, self.n_elem))

        D_free = jnp.block([
            [-D.T, zeros_nodes_el, zeros_nodes_el],
            [zeros_nodes_el, -D.T, zeros_nodes_el],
            [zeros_nodes_el, M_cg_dg, -D.T]
        ])

        J_linear_matrix = skew_block_matrix(D_free)

        return J_linear_matrix
    
    @partial(jax.jit, static_argnums=0)
    def assemble_nonlinear_dynamics(self, coenergy_vector) -> jnp.ndarray:

        lin_vel, ang_vel, force_resultant, _ = self.split_coenergy_vector(coenergy_vector)
        lin_momentum, _, strain, curvature = self.compute_costates(coenergy_vector)


        nonlinear_dynamics_lin_vel = + assemble_2d_cross_form_cg(self.x_nodes, lin_momentum, ang_vel) \
                                     - assemble_2d_cross_form_cg(self.x_nodes, force_resultant, curvature)

        nonlinear_dynamics_ang_vel = + assemble_2d_cross_skew_adjoint_form_cg(self.x_nodes, lin_momentum, lin_vel) \
                                     + assemble_2d_cross_skew_adjoint_form_cg(self.x_nodes, strain, force_resultant)

        nonlinear_dynamics_force = - assemble_2d_cross_form_dg(self.x_nodes, lin_vel, curvature)  \
                                    + assemble_2d_cross_form_dg(self.x_nodes, strain, ang_vel)

        nonlinear_dynamics_moment = jnp.zeros(self.n_moment)

        nonlinear_dynamics = jnp.concatenate([
            nonlinear_dynamics_lin_vel,
            nonlinear_dynamics_ang_vel,
            nonlinear_dynamics_force,
            nonlinear_dynamics_moment
        ])
        return nonlinear_dynamics
    

    @partial(jax.jit, static_argnums=0)
    def assemble_control_matrix(self):
        """
        Assemble the control operator considering the following orderning of the control vector:
        u = [-N_x(x=0), -N_y(x=0), -M_z(x=0), N_x(x=L), N_y(x=L), M_z(x=L)]
        """
        B = jnp.zeros((self.n_coenergy, 6))

        # Input at 0
        B = B.at[0, 0].set(1)                  # Axial force at 0
        B = B.at[self.n_lin_vel_x, 1].set(1)   # Shear force at 0
        B = B.at[self.n_lin_vel, 2].set(1)     # Bending moment at 0

        # Input at L
        B = B.at[self.n_lin_vel_x-1, 3].set(1) # Axial force at L
        B = B.at[self.n_lin_vel-1, 4].set(1)   # Shear force at L
        B = B.at[self.n_lin_vel + self.n_ang_vel-1, 5].set(1)   # Bending moment at L

        # return jnp.vstack([jnp.zeros((self.n_ang_vel, 6)), B]) 
        return B
