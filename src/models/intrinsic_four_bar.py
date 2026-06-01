import jax.numpy as jnp
import jax.scipy.linalg as jla
    
from src.linear_algebra.la_jax import skew_block_matrix
from src.rotations.utilities import rotation_matrix_2d
import jax
from functools import partial
from .intrinsic_free_beam import IntrinsicPlanarFreeBeam
from .intrinsic_pinned_beam import IntrinsicPlanarPinnedBeam


class IntrinsicFourBar():
    def __init__(self, n_elem_list, geo_params_list, mat_params_list):
        """
        Represents a four bar elements.
        
        Parameters:
            -----------
            n_elements : vector
                Number of finite elements for each beam
            material_props : dict
                Material properties: {'E': Young's modulus, 'G': shear modulus, 'rho': density}
            geometry_props : dict
                Geometric properties: {'L': length, 'A': cross-sectional area, 
                                    'I': second moment of area, 'k': shear coefficient}
        """

        n_elem_1, n_elem_2, n_elem_3 = n_elem_list
        geo_params_1, geo_params_2, geo_params_3 = geo_params_list
        mat_params_1, mat_params_2, mat_params_3 = mat_params_list
        
        self.pinned_beam_1 = IntrinsicPlanarPinnedBeam(n_elem_1, geo_params_1, mat_params_1)
        self.free_beam = IntrinsicPlanarFreeBeam(n_elem_2, geo_params_2, mat_params_2)
        self.pinned_beam_2 = IntrinsicPlanarPinnedBeam(n_elem_3, geo_params_3, mat_params_3)

        # Dimension of different partitions
        self.n_lin_vel = self.pinned_beam_1.n_lin_vel + self.free_beam.n_lin_vel + self.pinned_beam_2.n_lin_vel
        self.n_ang_vel = self.pinned_beam_1.n_ang_vel +  self.free_beam.n_ang_vel + self.pinned_beam_2.n_ang_vel
        self.n_vel = self.n_lin_vel + self.n_ang_vel

        self.n_coenergy = self.pinned_beam_1.n_coenergy + self.free_beam.n_coenergy + self.pinned_beam_2.n_coenergy
        self.n_state = self.n_ang_vel + self.n_coenergy



    @partial(jax.jit, static_argnums=0)
    def assemble_energy_matrix(self) -> jnp.ndarray:
        """
        Assemble the energy and interconnection matrix for the beam
        
        Returns:
        --------
        H : jnp.ndarray
            The energy matrix of the cantilever beam
        """
        H_1 = self.pinned_beam_1.assemble_energy_matrix()
        H_2 = self.free_beam.assemble_energy_matrix()
        H_3 = self.pinned_beam_2.assemble_energy_matrix()

        return jla.block_diag(H_1, H_2, H_3)

    @partial(jax.jit, static_argnums=0)
    def assemble_linear_interconnection_matrix(self) -> jnp.ndarray:
        """
        Assemble the interconnection matrix for the beam
        
        Returns:
        --------
        J_linear : jnp.ndarray
            The linear interconnection matrix of the beam
        """
        J_1 = self.pinned_beam_1.assemble_linear_interconnection_matrix()
        J_2 = self.free_beam.assemble_linear_interconnection_matrix()
        J_3 = self.pinned_beam_2.assemble_linear_interconnection_matrix()

        return jla.block_diag(J_1, J_2, J_3)
    

    @partial(jax.jit, static_argnums=0)
    def assemble_nonlinear_dynamics(self, coenergy_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Assemble the nonlinear part of the dynamics for the beam
        
        Parameters:
        -----------
        coenergy_vector : jnp.ndarray
            The co-energy vector of the system
        
        Returns:
        --------
        f_nl : jnp.ndarray
            The nonlinear part of the dynamics
        """
        n_coenergy_1 = self.pinned_beam_1.n_coenergy
        coenergy_1 = coenergy_vector[:n_coenergy_1]
        n_coenergy_2 = self.free_beam.n_coenergy
        coenergy_2 = coenergy_vector[:n_coenergy_2]
        n_coenergy_3 = self.pinned_beam_2.n_coenergy
        coenergy_3 = coenergy_vector[:n_coenergy_3]

        f_nl_1 = self.pinned_beam_1.assemble_nonlinear_dynamics(coenergy_1)
        f_nl_2 = self.free_beam.assemble_nonlinear_dynamics(coenergy_2)
        f_nl_3 = self.free_beam.assemble_nonlinear_dynamics(coenergy_3)

        return jnp.concatenate([f_nl_1, f_nl_2, f_nl_3])
    
    
    @partial(jax.jit, static_argnums=0)
    def assemble_control_matrix(self) -> jnp.ndarray:
        """
        Assemble the control matrix for the beam
        
        Returns:
        --------
        B : jnp.ndarray
            The control matrix of the beam
        """
        B_pinned_beam_1_left = self.pinned_beam_1.assemble_control_matrix()[:, :3]
        B_pinned_beam_2_right = self.pinned_beam_2.assemble_control_matrix[:, 3:]
        B = jax.scipy.linalg.block_diag(B_pinned_beam_1_left, B_pinned_beam_2_right)

        return B
    

    @partial(jax.jit, static_argnums=0)
    def compute_energies(self, coenergy_sol: jnp.ndarray):
        """
        Compute the energies of the beam over time
        
        Parameters:
        -----------
        coenergy_sol : jnp.ndarray
            The co-energy solution over time
        
        Returns:
        --------
        energy_vec : jnp.ndarray
            The total energy over time
        kinetic_energy_vec : jnp.ndarray
            The kinetic energy over time
        deformation_energy_vec : jnp.ndarray
            The deformation energy over time
        """
        n_coenergy_1 = self.pinned_beam_1.n_coenergy
        n_coenergy_2 = self.free_beam.n_coenergy

        coenergy_1 = coenergy_sol[:n_coenergy_1, :]
        coenergy_2 = coenergy_sol[n_coenergy_1:n_coenergy_1 + n_coenergy_2, :]
        coenergy_3 = coenergy_sol[n_coenergy_1 + n_coenergy_2:, :]

        energy_1, kinetic_energy_1, deformation_energy_1 = self.free_beam.compute_energies(coenergy_1)
        energy_2, kinetic_energy_2, deformation_energy_2 = self.free_beam.compute_energies(coenergy_2)
        energy_3, kinetic_energy_3, deformation_energy_3 = self.free_beam.compute_energies(coenergy_3)

        energy_vec = energy_1 + energy_2 + energy_3
        kinetic_energy_vec = kinetic_energy_1 + kinetic_energy_2 + kinetic_energy_3
        deformation_energy_vec = deformation_energy_1 \
            + deformation_energy_2 \
            + deformation_energy_3

        return energy_vec, kinetic_energy_vec, deformation_energy_vec
    


    @jax.jit
    def vector_field_positions(self, theta_vector: jnp.ndarray, lin_velocity_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the vector field for the spatial positions of the beam points 
        f(theta, v) = R(theta) * v
        where R(theta) is the rotation matrix from body frame to spatial frame.
        """
        n_theta_1 = self.pinned_beam_1.n_ang_vel
        n_theta_2 = self.free_beam.n_ang_vel
        theta_1 = theta_vector[:n_theta_1]
        theta_2 = theta_vector[n_theta_1:n_theta_1+n_theta_2]
        theta_3 = theta_vector[n_theta_1+n_theta_2:]

        n_lin_vel_1 = self.pinned_beam_1.n_lin_vel
        n_lin_vel_2 = self.free_beam.n_lin_vel
        
        vel_1 = lin_velocity_vector[:n_lin_vel_1]
        vel_2 = lin_velocity_vector[n_lin_vel_1:n_lin_vel_1+n_lin_vel_2]
        vel_3 = lin_velocity_vector[n_lin_vel_1+n_lin_vel_2:]

        vector_field_1 = self.pinned_beam_1.vector_field_positions(theta_1, vel_1)
        vector_field_2 = self.free_beam.vector_field_positions(theta_2, vel_2)
        vector_field_3 = self.pinned_beam_2.vector_field_positions(theta_3, vel_3)

        return jnp.concatenate([vector_field_1, vector_field_2, vector_field_3])
    


    @partial(jax.jit, static_argnums=0)
    def mass_matrix_positions(self) -> jnp.ndarray:
        """
        Mass matrix for the spatial positions of the beam points
        """
        M_pos_1 = self.pinned_beam_1.mass_matrix_positions()
        M_pos_2 = self.free_beam.mass_matrix_positions()
        M_pos_3 = self.pinned_beam_2.mass_matrix_positions()
        M_pos = jla.block_diag(M_pos_1, M_pos_2, M_pos_3)

        return M_pos