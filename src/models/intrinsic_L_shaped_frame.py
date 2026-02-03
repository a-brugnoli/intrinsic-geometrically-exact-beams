import jax.numpy as jnp
import jax.scipy.linalg as jla
    
from src.linear_algebra.la_jax import skew_block_matrix
from src.rotations.utilities import rotation_matrix_2d
import jax
from functools import partial
from .intrinsic_free_beam import IntrinsicPlanarFreeBeam
from src.models.intrinsic_cantilever_beam import IntrinsicPlanarCantileverBeam


class IntrinsicPlanarLShapedBeam():
    def __init__(self, n_elements, geometrical_props, material_props):
        """
        Represents a L shaped frame by interconnecting a free beam to a cantilever beam.
        
        Parameters:
            -----------
            n_elements : int
                Number of finite elements for each beam
            material_props : dict
                Material properties: {'E': Young's modulus, 'G': shear modulus, 'rho': density}
            geometry_props : dict
                Geometric properties: {'L': length, 'A': cross-sectional area, 
                                    'I': second moment of area, 'k': shear coefficient}
        """


        self.free_beam = IntrinsicPlanarFreeBeam(n_elements, geometrical_props, material_props)
        self.cantilever_beam = IntrinsicPlanarCantileverBeam(n_elements, geometrical_props, material_props)

        # Dimension of different partitions
        self.n_lin_vel = self.free_beam.n_lin_vel + self.cantilever_beam.n_lin_vel
        self.n_ang_vel = self.free_beam.n_ang_vel +  self.cantilever_beam.n_ang_vel
        self.n_vel = self.n_lin_vel + self.n_ang_vel

        self.n_coenergy = self.free_beam.n_coenergy + self.cantilever_beam.n_coenergy
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
        H_free = self.free_beam.assemble_energy_matrix()
        H_cantilever = self.cantilever_beam.assemble_energy_matrix()

        H_stacked = jla.block_diag(H_free, H_cantilever)

        return H_stacked

    @partial(jax.jit, static_argnums=0)
    def assemble_linear_interconnection_matrix(self) -> jnp.ndarray:
        """
        Assemble the interconnection matrix for the beam
        
        Returns:
        --------
        J_linear : jnp.ndarray
            The linear interconnection matrix of the beam
        """
        J_free = self.free_beam.assemble_linear_interconnection_matrix()

        J_cantilever = self.cantilever_beam.assemble_linear_interconnection_matrix()
    
        J_stacked = jla.block_diag(J_free, J_cantilever)

        B_free_right = self.free_beam.assemble_control_matrix()[:, 3:]
        B_cantilever_left = self.cantilever_beam.assemble_control_matrix()[:, :3]
        # The input in the clamped beam is the output of the free beam (the velocity are the same)

        angle_weld = jnp.pi / 2  # 90 degrees
        R_weld = rotation_matrix_2d(angle_weld)
        P_weld = jla.block_diag(R_weld, 1)

        C_coupling_upper_right = - B_free_right @ P_weld @ B_cantilever_left.T
        J_coupling = skew_block_matrix(C_coupling_upper_right)

        return J_stacked + J_coupling
    

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
        n_coenergy_free = self.free_beam.n_coenergy
        coenergy_free = coenergy_vector[:n_coenergy_free]
        coenergy_cantilever = coenergy_vector[n_coenergy_free:]

        f_nl_free = self.free_beam.assemble_nonlinear_dynamics(coenergy_free)
        f_nl_cantilever = self.cantilever_beam.assemble_nonlinear_dynamics(coenergy_cantilever)

        f_nl = jnp.concatenate([f_nl_free, f_nl_cantilever])

        return f_nl
    
    
    @partial(jax.jit, static_argnums=0)
    def assemble_control_matrix(self) -> jnp.ndarray:
        """
        Assemble the control matrix for the beam
        
        Returns:
        --------
        B : jnp.ndarray
            The control matrix of the beam
        """
        B_free_left = self.free_beam.assemble_control_matrix()[:, :3]
        B_free_right = self.cantilever_beam.assemble_control_matrix()[:, 3:]

        B = jax.scipy.linalg.block_diag(B_free_left, B_free_right)

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
        n_coenergy_free = self.free_beam.n_coenergy
        coenergy_free = coenergy_sol[:n_coenergy_free, :]
        coenergy_cantilever = coenergy_sol[n_coenergy_free:, :]

        energy_free, kinetic_energy_free, deformation_energy_free = self.free_beam.compute_energies(coenergy_free)
        energy_cantilever, kinetic_energy_cantilever, deformation_energy_cantilever = self.cantilever_beam.compute_energies(coenergy_cantilever)

        energy_vec = energy_free + energy_cantilever
        kinetic_energy_vec = kinetic_energy_free + kinetic_energy_cantilever
        deformation_energy_vec = deformation_energy_free + deformation_energy_cantilever

        return energy_vec, kinetic_energy_vec, deformation_energy_vec
    


    @jax.jit
    def vector_field_positions(self, theta_vector: jnp.ndarray, lin_velocity_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the vector field for the spatial positions of the beam points 
        f(theta, v) = R(theta) * v
        where R(theta) is the rotation matrix from body frame to spatial frame.
        """
        n_theta_free = self.free_beam.n_ang_vel
        theta_free = theta_vector[:n_theta_free]
        theta_cantilever = theta_vector[n_theta_free:]
        
        vel_free = lin_velocity_vector[:self.free_beam.n_lin_vel]
        vel_cantilever = lin_velocity_vector[self.free_beam.n_lin_vel:]

        vector_field_free = self.free_beam.vector_field_positions(theta_free, vel_free)
        vector_field_cantilever = self.cantilever_beam.vector_field_positions(theta_cantilever, vel_cantilever)

        return jnp.concatenate([vector_field_free, vector_field_cantilever])
    


    @partial(jax.jit, static_argnums=0)
    def mass_matrix_positions(self) -> jnp.ndarray:
        """
        Mass matrix for the spatial positions of the beam points
        """
        M_positions_free = self.free_beam.mass_matrix_positions()
        M_positions_cantilever = self.cantilever_beam.mass_matrix_positions()
        M_positions = jla.block_diag(M_positions_free, M_positions_cantilever)

        return M_positions