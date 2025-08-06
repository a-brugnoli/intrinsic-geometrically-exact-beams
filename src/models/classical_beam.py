import jax
import jax.numpy as jnp
from jax import jit
import jax.scipy.linalg as jla
from functools import partial


class ClassicalPlanarBeam:
    """
    Represents a linear Timoshenko beam
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
        self.total_dofs = 2 * self.n_nodes + 2 * self.n_elem

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

        # Node coordinates
        self.x_nodes = jnp.linspace(0, self.L, self.n_nodes)
        

    @partial(jax.jit, static_argnums=0)
    def _elementary_matrices_rod(self):
        """
        Elementary mass and stiffness matrices for linear Lagrange elements (2-node).
        These are the exact analytical expressions for uniform rod elements.
        
        Returns:
        --------
        M_e : array (2x2)
            Elementary mass matrix
        K_e : array (2x2)
            Elementary stiffness matrix
        """

        M_e = (self.rho * self.A * self.h_mesh / 6) * jnp.array([[2, 1],
                                                                [1, 2]])

        K_e = (self.E * self.A / self.h_mesh) * jnp.array([[ 1, -1],
                                                            [-1,  1]])
        
        return M_e, K_e


    @partial(jax.jit, static_argnums=0)
    def _timoshenko_beam_elementary_matrices(self):
        """
        Calculate elementary stiffness and mass matrices for Timoshenko beam element
        
        Returns:
        --------
        K : jax.numpy.ndarray
            4x4 elementary stiffness matrix
        M : jax.numpy.ndarray
            4x4 elementary mass matrix
        """
        
        # Calculate stiffness matrix [K]
        # DOFs: [w1, θ1, w2, θ2]
        
        # Shear stiffness parameter
        
        GAk = self.k * self.G * self.A
        Le = self.h_mesh
        
        # Coefficients for stiffness matrix
        
        # Stiffness matrix
        K_bending = self.E * self.I /Le * jnp.array([
            [0, 0, 0, 0],
            [0, 1, 0, -1],
            [0, 0, 0, 0],
            [0, -1, 0, 1]
        ])
        
        # One can consider reduced integration to avoid locking
        # K_shear = GAk/Le  * jnp.array([
        #     [1,    Le/2,    -1,    Le/2],
        #     [Le/2, Le**2/3, -Le/2, Le**2/6],
        #     [-1,  -Le/2,     1, -Le/2],
        #     [Le/2, Le**2/6, -Le/2, Le**2/3]
        # ])

        # One can consider reduced integration to avoid locking
        K_shear = GAk/Le  * jnp.array([
            [1,    Le/2,    -1,    Le/2],
            [Le/2, Le**2/4, -Le/2, Le**2/4],
            [-1,  -Le/2,     1, -Le/2],
            [Le/2, Le**2/4, -Le/2, Le**2/4]
        ])

        K = K_bending + K_shear
        
        # Calculate mass matrix [M]
        # Translational inertia contribution

        M_trans = self.rho * self.A * self.h_mesh * jnp.array([
            [1/3, 0, 1/6, 0],
            [0, 0, 0, 0],
            [1/6, 0, 1/3, 0],
            [0, 0, 0, 0]
        ])
        
        # Rotational inertia contribution
        M_rot = self.rho * self.I * self.h_mesh * jnp.array([
            [0, 0, 0, 0],
            [0, 1/3, 0, 1/6],
            [0, 0, 0, 0],
            [0, 1/6, 0, 1/3]
        ])
        
        # Total mass matrix
        M = M_trans + M_rot
        
        return K, M


    @partial(jax.jit, static_argnums=0)
    def assemble_matrices(self):
        """
        Assemble global mass and stiffness matrices.
        
        Returns:
        --------
        M : array
            Global mass matrix
        K : array
            Global stiffness matrix
        """
        
        # Number of nodes
        n_dofs_rod = self.n_elem + 1

        # Initialize global matrices
        M_rod = jnp.zeros((n_dofs_rod, n_dofs_rod))
        K_rod = jnp.zeros((n_dofs_rod, n_dofs_rod))
        
        # Get elementary matrices
        Me_rod, Ke_rod = self._elementary_matrices_rod()

        # Assembly loop using JAX's functional approach
        def assemble_rod_element(carry, e):
            M_rod_curr, K_rod_curr = carry
            # Element connectivity
            dofs_e = jnp.array([e, e + 1])
            
            # Use at[] for functional updates
            M_rod_new = M_rod_curr.at[dofs_e[0], dofs_e[0]].add(Me_rod[0, 0])
            M_rod_new = M_rod_new.at[dofs_e[0], dofs_e[1]].add(Me_rod[0, 1])
            M_rod_new = M_rod_new.at[dofs_e[1], dofs_e[0]].add(Me_rod[1, 0])
            M_rod_new = M_rod_new.at[dofs_e[1], dofs_e[1]].add(Me_rod[1, 1])
            
            K_rod_new = K_rod_curr.at[dofs_e[0], dofs_e[0]].add(Ke_rod[0, 0])
            K_rod_new = K_rod_new.at[dofs_e[0], dofs_e[1]].add(Ke_rod[0, 1])
            K_rod_new = K_rod_new.at[dofs_e[1], dofs_e[0]].add(Ke_rod[1, 0])
            K_rod_new = K_rod_new.at[dofs_e[1], dofs_e[1]].add(Ke_rod[1, 1])
            
            return (M_rod_new, K_rod_new), None

        (M_rod, K_rod), _ = jax.lax.scan(assemble_rod_element, 
                                        (M_rod, K_rod), 
                                        jnp.arange(self.n_elem))

        # Timoshenko beam matrices
        n_dofs_beam = 2 * (self.n_elem + 1)  # 2 DOFs per node (w, phi)

        # Initialize global mass and stiffness matrices
        K_beam = jnp.zeros((n_dofs_beam, n_dofs_beam))
        M_beam = jnp.zeros((n_dofs_beam, n_dofs_beam))

        # Get elementary matrices
        Ke_beam, Me_beam = self._timoshenko_beam_elementary_matrices()  
        # Assembly using JAX's functional approach
        def assemble_beam_element(carry, e):
            M_beam_curr, K_beam_curr = carry
            dofs_e = jnp.array([2*e, 2*e+1, 2*e+2, 2*e+3])
            
            # Functional assembly for beam elements
            for i in range(4):
                for j in range(4):
                    K_beam_curr = K_beam_curr.at[dofs_e[i], dofs_e[j]].add(Ke_beam[i, j])
                    M_beam_curr = M_beam_curr.at[dofs_e[i], dofs_e[j]].add(Me_beam[i, j])
            
            return (M_beam_curr, K_beam_curr), None

        (M_beam, K_beam), _ = jax.lax.scan(assemble_beam_element, 
                                        (M_beam, K_beam), 
                                        jnp.arange(self.n_elem))

        # Use JAX's block_diag equivalent
        M_matrix = jla.block_diag(M_rod, M_beam)
        K_matrix = jla.block_diag(K_rod, K_beam)
        
        return M_matrix, K_matrix


# Alternative more efficient assembly function using vectorized operations
    @partial(jax.jit, static_argnums=0)
    def assemble_matrices_vectorized(self):
        """
        Vectorized assembly of global mass and stiffness matrices using JAX.
        This version is more efficient for larger systems.
        """
        

        # Rod assembly
        n_dofs_rod = self.n_elem + 1
        Me_rod, Ke_rod = self._elementary_matrices_rod()
        
        # Create indices for assembly
        rod_indices_i = jnp.repeat(jnp.arange(self.n_elem)[:, None], 2, axis=1) + jnp.array([0, 1])
        rod_indices_j = rod_indices_i.copy()
        rod_indices_i = rod_indices_i.flatten()
        rod_indices_j = rod_indices_j.flatten()
        
        # Flatten elementary matrices and repeat for all elements
        Me_rod_flat = jnp.tile(Me_rod.flatten(), self.n_elem)
        Ke_rod_flat = jnp.tile(Ke_rod.flatten(), self.n_elem)
        
        # Assemble using scatter_add
        M_rod = jnp.zeros((n_dofs_rod, n_dofs_rod)).at[rod_indices_i, rod_indices_j].add(Me_rod_flat)
        K_rod = jnp.zeros((n_dofs_rod, n_dofs_rod)).at[rod_indices_i, rod_indices_j].add(Ke_rod_flat)

        # Beam assembly
        n_dofs_beam = 2 * (self.n_elem + 1)
        Ke_beam, Me_beam = self._timoshenko_beam_elementary_matrices()
        
        # Create indices for beam assembly
        beam_indices_i = jnp.repeat(jnp.arange(self.n_elem)[:, None], 4, axis=1) * 2 + jnp.array([0, 1, 2, 3])
        beam_indices_j = beam_indices_i.copy()
        beam_indices_i = beam_indices_i.flatten()
        beam_indices_j = beam_indices_j.flatten()
        
        # Flatten elementary matrices and repeat for all elements
        Me_beam_flat = jnp.tile(Me_beam.flatten(), self.n_elem)
        Ke_beam_flat = jnp.tile(Ke_beam.flatten(), self.n_elem)
        
        # Assemble using scatter_add
        M_beam = jnp.zeros((n_dofs_beam, n_dofs_beam)).at[beam_indices_i, beam_indices_j].add(Me_beam_flat)
        K_beam = jnp.zeros((n_dofs_beam, n_dofs_beam)).at[beam_indices_i, beam_indices_j].add(Ke_beam_flat)

        M_matrix = jla.block_diag(M_rod, M_beam)
        K_matrix = jla.block_diag(K_rod, K_beam)
        
        return M_matrix, K_matrix
