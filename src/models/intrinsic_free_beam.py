from mimetypes import init
from re import M
from turtle import pos
import jax.numpy as jnp
from numpy import r_, reshape
from src.fem.triplet_scalar_form import assemble_nontrilinear_scalar_form_cg
from src.fem.matrices_assembly import mass_matrix_lagrange, discrete_gradient, inner_product_cg_dg
from src.fem.triplet_vector_form import assemble_2d_cross_form_cg, assemble_2d_cross_form_dg, \
                                        assemble_2d_cross_skew_adjoint_form_cg
from src.fem.triplet_scalar_form import assemble_nontrilinear_scalar_form_cg
from src.linear_algebra.la_jax import skew_block_matrix
import jax
from functools import partial
from .intrinsic_generic_beam import IntrinsicBeam
from src.postprocessing.postprocess_variables import implicit_midpoint_integrate

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

        self.n_force = 2 * self.n_elem
        self.n_moment = self.n_elem   

        self.n_coenergy = self.n_lin_vel + self.n_ang_vel + self.n_force + self.n_moment
        self.n_state = self.n_ang_vel + self.n_coenergy


    @partial(jax.jit, static_argnums=0)
    def assemble_energy_matrix(self) -> jnp.ndarray:
        """
        Assemble the energy matrix for the beam
        
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

        H_matrix = jax.scipy.linalg.block_diag(
            M_lin_vel,       # x component of velocity
            M_lin_vel,       # y component of velocity
            M_ang_vel,       # angular velocity
            M_axial_for,     # x component of resultant (axial force)
            M_shear_for,     # y component of resultant (shear force)
            M_moment         # bending moment
        )

        return H_matrix


    @partial(jax.jit, static_argnums=0)
    def assemble_linear_interconnection_matrix(self) -> jnp.ndarray:
        """
        Assemble the linear interconnection matrix for the beam
        
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
        Assemble the control operator considering the following ordering:
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

        return B
    


    @partial(jax.jit, static_argnums=0)
    def assemble_gravity(self, theta_vector: jnp.ndarray, g = 9.81, n_gauss_points=2) -> jnp.ndarray:
        
        gravity_nodal_values = - self.density_unit_length * g * jnp.ones(self.n_nodes)

        material_gravity_field_x = assemble_nontrilinear_scalar_form_cg(self.x_nodes, 
                                                                        theta_vector, 
                                                                        gravity_nodal_values, 
                                                                        jnp.sin,
                                                                        n_gauss_points=n_gauss_points)
        
        material_gravity_field_y = assemble_nontrilinear_scalar_form_cg(self.x_nodes, 
                                                                        theta_vector, 
                                                                        gravity_nodal_values, 
                                                                        jnp.cos,
                                                                        n_gauss_points=n_gauss_points)
        
        gravity_vector_field = jnp.zeros(self.n_coenergy)

        gravity_vector_field = gravity_vector_field.at[:self.n_lin_vel_x].set(material_gravity_field_x)
        gravity_vector_field = gravity_vector_field.at[self.n_lin_vel_x:self.n_lin_vel].set(material_gravity_field_y)

        return gravity_vector_field
    

    @partial(jax.jit, static_argnums=0)
    def compute_potential_energy(self, y_positions, g=9.81):
        """"
        Compute the potential energy
        Parameters:
        -----------
        y_positions : jnp.ndarray
            The positions of the beam at nodes (N_nodes, N_times)
        """
        M_pos_y = mass_matrix_lagrange(self.x_nodes)
        
        return jnp.sum(self.density_unit_length * g * M_pos_y @ y_positions, axis=0)  


    @partial(jax.jit, static_argnums=0)
    def vector_field_positions(self, theta_vector: jnp.ndarray, 
                               lin_velocity_vector: jnp.ndarray, n_gauss_points=2) -> jnp.ndarray:
        """
        Compute the vector field for the spatial positions of the beam points 
        f(theta, v) = R(theta) * v
        where R(theta) is the rotation matrix from body frame to spatial frame.
        """

        vel_x = lin_velocity_vector[:self.n_lin_vel_x]
        vel_y = lin_velocity_vector[self.n_lin_vel_x:self.n_lin_vel]


        vector_field_x = + assemble_nontrilinear_scalar_form_cg(self.x_nodes, theta_vector, 
                                                                vel_x, jnp.cos, n_gauss_points=n_gauss_points) \
                         - assemble_nontrilinear_scalar_form_cg(self.x_nodes, theta_vector, 
                                                                vel_y, jnp.sin, n_gauss_points=n_gauss_points)
        
        vector_field_y = + assemble_nontrilinear_scalar_form_cg(self.x_nodes, theta_vector, 
                                                                vel_x, jnp.sin, n_gauss_points=n_gauss_points) \
                         + assemble_nontrilinear_scalar_form_cg(self.x_nodes, theta_vector, 
                                                                vel_y, jnp.cos, n_gauss_points=n_gauss_points)
        
        return jnp.concatenate([vector_field_x, vector_field_y])
    

    @partial(jax.jit, static_argnums=0)
    def mass_matrix_positions(self) -> jnp.ndarray:
        """
        Mass matrix for the spatial positions of the beam points
        """
        M_pos_x = mass_matrix_lagrange(self.x_nodes)
        M_pos_y = mass_matrix_lagrange(self.x_nodes)

        M_positions = jax.scipy.linalg.block_diag(
            M_pos_x,
            M_pos_y
        )

        return M_positions
    

    @partial(jax.jit, static_argnums=0)
    def compute_delta_angular_momentum(self,
                                position_sol: jnp.ndarray,
                                theta_sol: jnp.ndarray,
                                lin_velocity_sol: jnp.ndarray) -> jnp.ndarray:
        """
        Docstring for compute_angular_momentum
        position_sol: jnp.ndarray of shape (2, n_nodes, n_times)
            Positions of the beam points in spatial frame over time.
        theta_sol: jnp.ndarray of shape (n_nodes, n_times)
            Angular positions of the beam nodes over time.  
        lin_velocity_sol: jnp.ndarray of shape (2*n_nodes, n_times)
            Linear velocities over time in body frame.
        Returns:
        --------
        angular_momentum_sol: jnp.ndarray of shape (n_times-1,)
            Special difference of angular momentum over time. 
        """
        n_times = position_sol.shape[2]

        reshape_position_sol = position_sol.reshape(2*self.n_nodes, n_times)

        assert lin_velocity_sol.shape[0] == 2 * self.n_nodes, "Linear velocity solution has incompatible shape with position solution"

        S_matrix = skew_block_matrix(jnp.eye(self.n_nodes))

        deltaJ = jnp.zeros(n_times-1)
        
        for i in range(n_times - 1):
            position_old = reshape_position_sol[:, i]
            position_new = reshape_position_sol[:, i + 1]

            lin_vel_mid = 0.5 * (
                lin_velocity_sol[:, i] + lin_velocity_sol[:, i + 1]
            )

            theta_mid = 0.5 * (
                theta_sol[:, i] + theta_sol[:, i + 1]
            )

            deltaJ = deltaJ.at[i].set(
                (position_new - position_old).T
                @ S_matrix
                @ self.vector_field_positions(theta_mid, lin_vel_mid)
            )
    
        return deltaJ


    # @partial(jax.jit, static_argnums=0)
    # def compute_delta_angular_momentum(self,
    #                             position_sol: jnp.ndarray,
    #                             theta_sol: jnp.ndarray,
    #                             velocity_sol: jnp.ndarray) -> jnp.ndarray:
    #     """
    #     Docstring for compute_angular_momentum
    #     position_sol: jnp.ndarray of shape (2, n_nodes, n_times)
    #         Positions of the beam points in spatial frame over time.
    #     theta_sol: jnp.ndarray of shape (n_nodes, n_times)
    #         Angular positions of the beam nodes over time.  
    #     velocity_sol: jnp.ndarray of shape (2*n_nodes, n_times)
    #         Velocities over time in body frame.
    #     Returns:
    #     --------
    #     angular_momentum_sol: jnp.ndarray of shape (n_times-1,)
    #         Special difference of angular momentum over time. 
    #     """
    #     n_times = position_sol.shape[2]

    #     reshape_position_sol = position_sol.reshape(2*self.n_nodes, n_times)

    #     assert velocity_sol.shape[0] == 3 * self.n_nodes, "Velocity solution has incompatible shape with position solution"

    #     lin_velocity_sol = velocity_sol[:2*self.n_nodes, :]
    #     ang_velocity_sol = velocity_sol[2*self.n_nodes:, :]

    #     S_matrix = skew_block_matrix(jnp.eye(self.n_nodes))

    #     deltaJ = jnp.zeros(n_times-1)
        
    #     for i in range(n_times - 1):
    #         position_old = reshape_position_sol[:, i]
    #         position_new = reshape_position_sol[:, i + 1]

    #         ang_velocity_old = ang_velocity_sol[:, i]
    #         ang_velocity_new = ang_velocity_sol[:, i + 1]

    #         lin_vel_mid = 0.5 * (
    #             lin_velocity_sol[:, i] + lin_velocity_sol[:, i + 1]
    #         )

    #         theta_mid = 0.5 * (
    #             theta_sol[:, i] + theta_sol[:, i + 1]
    #         )

    #         deltaJ = deltaJ.at[i].set(
    #             (position_new - position_old).T
    #             @ S_matrix
    #             @ self.vector_field_positions(theta_mid, lin_vel_mid) 
    #             + jnp.ones(self.n_nodes).T @ 
    #             mass_matrix_lagrange(self.x_nodes, parameter=self.rotational_inertia) @ 
    #             (ang_velocity_new - ang_velocity_old)
    #         )
    
    #     return deltaJ