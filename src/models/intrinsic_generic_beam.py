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
        self.n_dofs = 3 * self.n_nodes + 3 * self.n_elem

        # Geometric properties
        self.L = geometrical_props['L']
        self.A = geometrical_props['A']
        self.I = geometrical_props['I']
        self.k = geometrical_props['k']
        
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

        # Dimension of different partitions
        self.n_lin_vel = None
        self.n_ang_vel = None
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


    def split_coenergy_vector(self, coenergy_vector: jnp.ndarray) -> jnp.ndarray:
        """
        Split the state vector into its components
        Vector variables are arranged in a matrix: [v_x, v_y, v_z]
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
        n_dofs_vel = self.n_lin_vel + self.n_ang_vel
        velocity_sol = state_sol[:n_dofs_vel, :]
        stress_sol = state_sol[n_dofs_vel:, :]

        H_total = self.assemble_energy_matrix()
        H_kinetic = H_total[:n_dofs_vel, :n_dofs_vel]
        H_stress = H_total[n_dofs_vel:, n_dofs_vel:]

        # Compute column-wise quadratic forms
        energy_vec = 0.5 * jnp.sum(state_sol *  (H_total @ state_sol), axis=0)  
        kinetic_energy_vec = 0.5 * jnp.sum(velocity_sol * (H_kinetic @ velocity_sol), axis=0)  
        deformation_energy_vec = 0.5 * jnp.sum(stress_sol * (H_stress @ stress_sol), axis=0)  

        return energy_vec, kinetic_energy_vec, deformation_energy_vec
    
    @partial(jax.jit, static_argnums=0)
    def compute_positions(self, theta_sol: jnp.ndarray, lin_velocity_sol: jnp.ndarray, 
                           initial_position: jnp.ndarray, time_sol: jnp.ndarray):
        
        dt_vec = time_sol[1:] - time_sol[:-1]
        n_steps = time_sol.shape[0] - 1

        
        # Integrate position vector dynamics using the trapezoidal rule
        reshaped_velocity_sol = lin_velocity_sol.reshape(2, int(self.n_lin_vel / 2), -1)

        position_sol = jnp.zeros_like(reshaped_velocity_sol)

        position_sol = position_sol.at[0, :, 0].set(initial_position[:, 0])
        position_sol = position_sol.at[1, :, 0].set(initial_position[:, 1])

        def integration_position(position_prev, ii):

            theta_mid = (theta_sol[:, ii] + theta_sol[:, ii+1]) / 2        # shape (N,)
            velocity_mid = (reshaped_velocity_sol[:, :, ii] + reshaped_velocity_sol[:, :, ii+1]) / 2  # shape (2, N)

            # Compute rotation matrices for all nodes: shape (N, 2, 2)
            R_batch = rotation_matrix_2d_batch(theta_mid)

            # Apply rotation to each velocity vector
            delta = jnp.einsum('nij,jn->in', R_batch, velocity_mid)  # result: (2, N)

            position_next = position_prev + dt_vec[ii] * delta
            return position_next, position_next

        # Scan over time steps
        # Initialize
        position_sol_init = position_sol[:, :, 0]  # shape (2, N)

        _, positions = jax.lax.scan(integration_position, position_sol_init, jnp.arange(n_steps))

        # Concatenate initial position to match original shape (2, N, n_steps+1)
        position_sol = jnp.concatenate([position_sol_init[None, :, :], positions], axis=0).transpose(1, 2, 0)

        return position_sol
