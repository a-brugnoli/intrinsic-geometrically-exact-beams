from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
from functools import partial
from src.rotations.utilities import rotation_matrix_2d_batch


class IntrinsicBeam(ABC):
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
        self.n_elem = n_elements
        self.n_nodes = n_elements + 1

        # Geometric properties
        L = geometrical_props['L']
        A = geometrical_props['A']
        I = geometrical_props['I']
        k = geometrical_props['k']
        
        # Element length
        self.h_mesh = L / self.n_elem
        
        # Material properties
        E = material_props['E']
        G = material_props['G']
        rho = material_props['rho']

        # Physical quantities for the dynamical system
        self.density_unit_length = rho * A
        self.rotational_inertia = rho * I

        self.axial_compliance = 1 / (E * A)
        self.shear_compliance = 1 / (k * G * A)
        self.bending_compliance = 1 / (E * I)
        
        # Node coordinates
        self.x_nodes = jnp.linspace(0, L, self.n_nodes)

        # # Dimension of different partitions
        self.n_lin_vel = None
        self.n_ang_vel = None
        self.n_vel = None

        self.n_force = None
        self.n_moment = None  

        self.J_linear_matrix = None


    def set_axial_compliance(self, axial_compliance):
        """
        Set the axial compliance of the beam
        This is to allow for non consistent values of EA
        """
        self.axial_compliance = axial_compliance


    def set_shear_compliance(self, shear_compliance):
        self.shear_compliance = shear_compliance


    def set_bending_compliance(self, bending_compliance):
        self.bending_compliance = bending_compliance       

    

    @abstractmethod
    def assemble_energy_matrix(self) -> jnp.ndarray:
        pass


    @abstractmethod
    def assemble_linear_interconnection_matrix(self) -> jnp.ndarray:
        pass


    @abstractmethod
    def assemble_nonlinear_dynamics(self, coenergy_vector: jnp.ndarray) -> jnp.ndarray:
        pass


    @abstractmethod
    def assemble_control_matrix(self) -> jnp.ndarray:
        pass


    # @partial(jax.jit, static_argnums=0)
    # def assemble_E_matrix(self) -> jnp.ndarray:
    #     """
    #     Assemble the E matrix for the implicit time integration scheme
    #     """
    #     E_matrix = jax.scipy.linalg.block_diag(jnp.eye(self.n_ang_vel), self.H_matrix)
    #     return E_matrix
    

    @partial(jax.jit, static_argnums=0)
    def assemble_autonomous_vector_field(self, state_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Assemble the autonomous part of the vector field
        f_theta = omega
        f_coenergy = J(x) * dH(x)

        f(x) = [f_theta; f_coenergy]

        Parameters:
        -----------
        state_vector : jnp.ndarray
            State vector of the dynamical system

        """
        coenergy_vector = state_vector[self.n_ang_vel:]

        vector_field_theta = coenergy_vector[self.n_lin_vel:self.n_lin_vel + self.n_ang_vel]

        vector_field_coenergy = self.J_linear_matrix  @ coenergy_vector \
                            + self.assemble_nonlinear_dynamics(coenergy_vector) \
   
        return jnp.concatenate([vector_field_theta, vector_field_coenergy])
    

    @partial(jax.jit, static_argnums=0)
    def split_coenergy_vector(self, coenergy_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Split the state vector into its components
        Vector variables are arranged in a matrix: [v_x, v_y, v_z]

        Warning: this function is not vectorized and takes one vector at a time
        """
        lin_vel = jax.lax.dynamic_slice_in_dim(coenergy_vector, 0, self.n_lin_vel)
        lin_vel = lin_vel.reshape((2, -1)).T
        
        ang_vel = jax.lax.dynamic_slice_in_dim(coenergy_vector, self.n_lin_vel, self.n_ang_vel)

        idx_force = self.n_lin_vel + self.n_ang_vel
        force_resultant = jax.lax.dynamic_slice_in_dim(coenergy_vector, idx_force, self.n_force)
        force_resultant = force_resultant.reshape((2, -1)).T

        idx_moment = self.n_lin_vel + self.n_ang_vel + self.n_force
        moment = jax.lax.dynamic_slice_in_dim(coenergy_vector, idx_moment, self.n_moment)

        return lin_vel, ang_vel, force_resultant, moment


    @partial(jax.jit, static_argnums=0)
    def compute_costates(self, coenergy_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the energy variables from the co-energy variables
        """
        lin_vel, ang_vel, force_resultant, moment = self.split_coenergy_vector(coenergy_vector)

        lin_momentum = self.density_unit_length * lin_vel
        ang_momentum = self.rotational_inertia * ang_vel
        strain = force_resultant * jnp.array([self.axial_compliance, self.shear_compliance])
        curvature = self.bending_compliance * moment

        return lin_momentum, ang_momentum, strain, curvature


    @partial(jax.jit, static_argnums=0)
    def compute_energies(self, state_sol: jnp.ndarray):
        velocity_sol = state_sol[:self.n_vel, :]
        stress_sol = state_sol[self.n_vel:, :]

        H_matrix = self.assemble_energy_matrix()
        H_kinetic = H_matrix[:self.n_vel, :self.n_vel]
        H_stress = H_matrix[self.n_vel:, self.n_vel:]

        # Compute column-wise quadratic forms
        energy_vec = 0.5 * jnp.sum(state_sol *  (H_matrix @ state_sol), axis=0)  
        kinetic_energy_vec = 0.5 * jnp.sum(velocity_sol * (H_kinetic @ velocity_sol), axis=0)  
        deformation_energy_vec = 0.5 * jnp.sum(stress_sol * (H_stress @ stress_sol), axis=0)  

        return energy_vec, kinetic_energy_vec, deformation_energy_vec
    
    