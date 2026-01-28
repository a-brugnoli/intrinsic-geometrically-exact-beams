from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax
from functools import partial
from src.linear_algebra.la_jax import skew_block_matrix
from src.fem.matrices_assembly import mass_matrix_lagrange

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
        A = geometrical_props['A']
        I = geometrical_props['I']
        k = geometrical_props['k']
        
        # Element length
        self.L = geometrical_props['L']
        self.h_mesh = self.L / self.n_elem
        
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
        self.x_nodes = jnp.linspace(0, self.L, self.n_nodes)

        # # Dimension of different partitions        
        self.n_lin_vel = None
        self.n_ang_vel = None
        self.n_vel = None

        self.n_force = None
        self.n_moment = None  


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
    def compute_energies(self, coenergy_sol: jnp.ndarray):
        velocity_sol = coenergy_sol[:self.n_vel, :]
        stress_sol = coenergy_sol[self.n_vel:, :]

        H_matrix = self.assemble_energy_matrix()

        H_kinetic = H_matrix[:self.n_vel, :self.n_vel]
        H_stress = H_matrix[self.n_vel:, self.n_vel:]

        # Compute column-wise quadratic forms
        energy_vec = 0.5 * jnp.sum(coenergy_sol *  (H_matrix @ coenergy_sol), axis=0)  
        kinetic_energy_vec = 0.5 * jnp.sum(velocity_sol * (H_kinetic @ velocity_sol), axis=0)  
        deformation_energy_vec = 0.5 * jnp.sum(stress_sol * (H_stress @ stress_sol), axis=0)  

        return energy_vec, kinetic_energy_vec, deformation_energy_vec
    


    @abstractmethod
    def assemble_gravity(self, theta_vector: jnp.ndarray, g=9.81, n_gauss_points=2) -> jnp.ndarray:
        pass


    @abstractmethod
    def compute_potential_energy(self, y_positions: jnp.ndarray, g=9.81) -> jnp.ndarray:
        pass

    @abstractmethod
    def vector_field_positions(self, theta_vector: jnp.ndarray, lin_velocity_vector: jnp.ndarray) -> jnp.ndarray:
        pass


    @abstractmethod
    def mass_matrix_positions(self) -> jnp.ndarray:
        pass


    # @abstractmethod
    # def compute_angular_momentum(self, position_sol: jnp.ndarray,
    #                             theta_vector: jnp.ndarray, 
    #                             velocity_sol: jnp.ndarray,
    #                             n_gauss_points=2) -> jnp.ndarray:
    #     """
    #     Docstring for compute_angular_momentum
    #     position_sol: jnp.ndarray of shape (2, n_nodes, n_times) or (2, n_elem, n_times)
    #         Angular displacements over time.
    #     theta_sol: jnp.ndarray of shape (n_nodes, n_times) or (n_elem, n_times)
    #         Angular displacements over time.
    #     lin_velocity_sol: jnp.ndarray of shape (2*n_nodes, n_times) or (2*n_elem, n_times)
    #         Linear velocities over time in body frame.
    #     n_gauss_points: int
    #         Number of gauss points for numerical integration.
    #     Returns:
    #     --------
    #     angular_momentum_sol: jnp.ndarray of shape (n_times,)
    #         Angular momentum over time. 
    #     """
    #     pass

    @partial(jax.jit, static_argnums=0)
    def compute_positions(self, theta_sol: jnp.ndarray, lin_velocity_sol: jnp.ndarray, 
                        initial_position: jnp.ndarray, time_sol: jnp.ndarray):
        """
        Compute the spatial positions of 2D beam points or midpoints over time given
        the angular displacements and linear velocities (in body frame).

        Parameters:
        -----------
        theta_sol: jnp.ndarray of shape (n_nodes, n_times) or (n_elem, n_times)
            Angular displacements over time.
        lin_velocity_sol: jnp.ndarray of shape (2*n_nodes, n_times) or (2*n_elem, n_times)
            Linear velocities over time in body frame.
        initial_position: jnp.ndarray of shape (2*n_nodes,) or (2*n_elem,)
            Initial spatial positions of the beam points.
        time_sol: jnp.ndarray of shape (n_times,)
            Time vector.    
        """
        
        assert theta_sol.shape[-1] == lin_velocity_sol.shape[-1] and theta_sol.shape[-1]==time_sol.shape[0], \
            f"Temporal dimension mismatch: theta_sol.shape[-1]={theta_sol.shape[-1]} != lin_velocity_sol.shape[-1]={lin_velocity_sol.shape[-1]}"
        
        dt_vec = time_sol[1:] - time_sol[:-1]
        n_steps = time_sol.shape[0] - 1

        position_sol = jnp.zeros_like(lin_velocity_sol) 

        position_sol = position_sol.at[:, 0].set(initial_position)

        mass_matrix_positions = self.mass_matrix_positions()  # shape (n_pos, n_pos)

        def integration_position(position_prev, ii):

            theta_mid = (theta_sol[:, ii] + theta_sol[:, ii+1]) / 2        # shape (N,)
            velocity_mid = (lin_velocity_sol[:, ii] + lin_velocity_sol[:, ii+1]) / 2  # shape (N)

            positions_vector_field_mid = self.vector_field_positions(theta_mid, velocity_mid)

            position_next = position_prev + dt_vec[ii] * jnp.linalg.solve(mass_matrix_positions, 
                                                                        positions_vector_field_mid)

            
            return position_next, position_next

        _, positions = jax.lax.scan(integration_position, initial_position, jnp.arange(n_steps))
        # Concatenate initial position to match original shape (2, N, n_steps+1)
        position_sol = jnp.concatenate([initial_position.reshape(-1, 1), positions.T], axis=1)

        N = position_sol.shape[0] // 2

        return position_sol.reshape(2, N, position_sol.shape[1])


