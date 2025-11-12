import jax.numpy as jnp
import jax.scipy.linalg as jla
    
from src.linear_algebra.la_jax import skew_block_matrix
from src.rotations.utilities import rotation_matrix_2d
import jax
from functools import partial
from .intrinsic_free_beam import IntrinsicPlanarFreeBeam
from src.models.intrinsic_clamped_beam import IntrinsicPlanarClampedBeam


class IntrinsicPlanarClampedFreeBeam():
    def __init__(self, n_elements, geometrical_props, material_props):
        """
        Represents a geometrically exact planar cantilever beam 
        by interconnecting a fixed-fixed beam (on the left) and a free-free beam (on the right).

        The interconnection point is at the midspan (L/2)


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

        n_elements_half = n_elements // 2  # Even number of elements for each half

        geometrical_props_half = geometrical_props.copy()
        geometrical_props_half['L'] = geometrical_props['L'] / 2

        self.clamped_beam = IntrinsicPlanarClampedBeam(n_elements_half, geometrical_props_half, material_props)
        self.free_beam = IntrinsicPlanarFreeBeam(n_elements_half, geometrical_props_half, material_props)

        # Dimension of different partitions
        self.n_lin_vel = self.clamped_beam.n_lin_vel + self.free_beam.n_lin_vel
        self.n_ang_vel = self.clamped_beam.n_ang_vel + self.free_beam.n_ang_vel
        self.n_vel = self.n_lin_vel + self.n_ang_vel

        self.n_coenergy = self.clamped_beam.n_coenergy + self.free_beam.n_coenergy
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
        H_clamped = self.clamped_beam.assemble_energy_matrix()
        H_free = self.free_beam.assemble_energy_matrix()

        H_stacked = jla.block_diag(H_clamped, H_free)

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
        J_clamped = self.clamped_beam.assemble_linear_interconnection_matrix()
        J_free = self.free_beam.assemble_linear_interconnection_matrix()

        J_stacked = jla.block_diag(J_clamped, J_free)

        B_clamped_right = self.clamped_beam.assemble_control_matrix()[:, 3:]
        B_free_left = self.free_beam.assemble_control_matrix()[:, :3]
        # The input in the clamped beam is the output of the free beam (the velocity are the same)
        C_coupling_upper_right = B_clamped_right @ B_free_left.T
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
        n_coenergy_left = self.clamped_beam.n_coenergy
        coenergy_left = coenergy_vector[:n_coenergy_left]
        coenergy_right = coenergy_vector[n_coenergy_left:]

        f_nl_left = self.clamped_beam.assemble_nonlinear_dynamics(coenergy_left)
        f_nl_right = self.free_beam.assemble_nonlinear_dynamics(coenergy_right)

        f_nl = jnp.concatenate([f_nl_left, f_nl_right])

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
        B_clamped_left = self.clamped_beam.assemble_control_matrix()[:, :3]
        B_free_right = self.free_beam.assemble_control_matrix()[:, 3:]

        B = jax.scipy.linalg.block_diag(B_clamped_left, B_free_right)

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
        n_coenergy_left = self.clamped_beam.n_coenergy
        coenergy_left = coenergy_sol[:n_coenergy_left, :]
        coenergy_right = coenergy_sol[n_coenergy_left:, :]

        energy_left, kinetic_energy_left, deformation_energy_left = self.clamped_beam.compute_energies(coenergy_left)
        energy_right, kinetic_energy_right, deformation_energy_right = self.free_beam.compute_energies(coenergy_right)

        energy_vec = energy_left + energy_right
        kinetic_energy_vec = kinetic_energy_left + kinetic_energy_right
        deformation_energy_vec = deformation_energy_left + deformation_energy_right

        return energy_vec, kinetic_energy_vec, deformation_energy_vec