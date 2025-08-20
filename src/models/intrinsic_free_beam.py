import jax.numpy as jnp
import jax.scipy.linalg as jla
from src.fem.matrices_assembly import mass_matrix_lagrange, discrete_gradient, inner_product_cg_dg
from src.fem.trilinear_vector_form import assemble_2d_cross_form_cg, assemble_2d_cross_form_dg, \
                            assemble_2d_cross_skew_adjoint_form_cg, assemble_2d_cross_skew_adjoint_form_dg
      
from src.linear_algebra.la_jax import skew_block_matrix
import jax
from functools import partial


class IntrinsicPlanarFreeBeam:
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
        self.n_lin_vel = 2 * self.n_nodes
        self.n_ang_vel = self.n_nodes
        self.n_force = 2 * self.n_elem
        self.n_moment = self.n_elem    

    def set_axial_compliance(self, axial_compliance):
        self.axial_compliance = axial_compliance

    def set_shear_compliance(self, shear_compliance):
        self.shear_compliance = shear_compliance

    def set_bending_compliance(self, bending_compliance):
        self.bending_compliance = bending_compliance        

    @partial(jax.jit, static_argnums=0)
    def assemble_matrices(self) -> jnp.ndarray:
        """
        Assemble the energy and interconnection matrix for the beam
        
        Returns:
        --------
        H : jnp.ndarray
            The energy matrix of the beam
        J_linear : jnp.ndarray
            The linear interconnection matrix of the beam
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
            M_moment    # bending moment
        )

        D = discrete_gradient(self.x_nodes)
        M_cg_dg = inner_product_cg_dg(self.x_nodes)
        zeros_n_el = jnp.zeros((self.n_nodes, self.n_elem))

        D_free = jnp.block([
            [-D.T, zeros_n_el, zeros_n_el],
            [zeros_n_el, -D.T, zeros_n_el],
            [zeros_n_el, M_cg_dg, -D.T]
        ])

        J_linear_matrix = skew_block_matrix(D_free)

        return H_matrix, J_linear_matrix
    
    @partial(jax.jit, static_argnums=0)
    def assemble_nonlinear_dynamics(self, coenergy_vector) -> jnp.ndarray:

        lin_vel = jax.lax.dynamic_slice_in_dim(coenergy_vector, 0, self.n_lin_vel)
        lin_vel = lin_vel.reshape((2, -1)).T
        lin_momentum = self.density_unit_length * lin_vel

        ang_vel = jax.lax.dynamic_slice_in_dim(coenergy_vector, self.n_lin_vel, self.n_ang_vel)
        # ang_momentum = self.rotational_inertia * ang_vel

        idx_force = self.n_lin_vel + self.n_ang_vel
        force_resultant = jax.lax.dynamic_slice_in_dim(coenergy_vector, idx_force, self.n_force)
        force_resultant = force_resultant.reshape((2, -1)).T
        strain = force_resultant * jnp.array([self.axial_compliance, self.shear_compliance])

        idx_moment = self.n_lin_vel + self.n_ang_vel + self.n_force
        moment = jax.lax.dynamic_slice_in_dim(coenergy_vector, idx_moment, self.n_moment)
        curvature = self.bending_compliance * moment


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
        u = [N_x(x=0), N_y(x=0), M_z(x=0), N_x(x=L), N_y(x=L), M_z(x=L)]
        """
        B = jnp.zeros((self.n_dofs, 6))

        # Input at 0
        B = B.at[0, 0].set(1)
        B = B.at[self.n_nodes, 1].set(1)
        B = B.at[self.n_lin_vel, 2].set(1)

        # Input at L
        B = B.at[self.n_nodes-1, 3].set(1)
        B = B.at[self.n_lin_vel-1, 4].set(1)
        B = B.at[self.n_lin_vel + self.n_ang_vel-1, 5].set(1)

        return B
    
    @partial(jax.jit, static_argnums=0)
    def compute_energy(self, coenergy_sol):
        # Apply H_matrix to all columns
        H_matrix = self.assemble_matrices()[0]
        Hx = H_matrix @ coenergy_sol  # shape (n_states, n_steps)

        # Compute column-wise quadratic forms
        energy_vec = 0.5 * jnp.sum(coenergy_sol * Hx, axis=0)  # shape (n_steps,)
        return energy_vec
    
    @partial(jax.jit, static_argnums=0)
    def compute_kinetic_energy(self, velocity_sol):
        H_matrix = self.assemble_matrices()[0]

        n_dofs_vel = self.n_lin_vel + self.n_ang_vel
        H_velocity = H_matrix[:n_dofs_vel, :n_dofs_vel]
        Hx = H_velocity @ velocity_sol 

        # Compute column-wise quadratic forms
        kinetic_energy_vec = 0.5 * jnp.sum(velocity_sol * Hx, axis=0)  # shape (n_steps,)
        return kinetic_energy_vec
    
    @partial(jax.jit, static_argnums=0)
    def compute_deformation_energy(self, stress_sol):
        # Apply H_matrix to all columns
        H_matrix = self.assemble_matrices()[0]
        n_dofs_vel = self.n_lin_vel + self.n_ang_vel
        H_stress = H_matrix[n_dofs_vel:, n_dofs_vel:]
        Hx = H_stress @ stress_sol  # shape (n_states, n_steps)

        # Compute column-wise quadratic forms
        deformation_energy_vec = 0.5 * jnp.sum(stress_sol * Hx, axis=0)  # shape (n_steps,)
        return deformation_energy_vec