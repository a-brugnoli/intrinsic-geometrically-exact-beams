import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
import numpy as np

class NonlinearFEM1D:
    def __init__(self, nodes, nonlinear_func, n_gauss_points=3):
        """
        Initialize 1D FEM with linear Lagrange elements
        
        Args:
            nodes: Array of node coordinates
            nonlinear_func: Function f(u) for the nonlinear term ∫ φᵢ f(u) dx
            n_gauss_points: Number of Gauss points per element for integration
        """
        self.nodes = jnp.array(nodes)
        self.n_nodes = len(nodes)
        self.n_elements = self.n_nodes - 1
        self.n_gauss_points = n_gauss_points
        
        # Set the nonlinear function
        self.nonlinear_func = nonlinear_func
        
        # Get Gauss quadrature points and weights for [-1, 1]
        self.gauss_points, self.gauss_weights = self._get_gauss_quadrature(n_gauss_points)
    
    def _get_gauss_quadrature(self, n_points):
        """Get Gauss-Legendre quadrature points and weights"""
        if n_points == 1:
            points = jnp.array([0.0])
            weights = jnp.array([2.0])
        elif n_points == 2:
            points = jnp.array([-1.0/jnp.sqrt(3), 1.0/jnp.sqrt(3)])
            weights = jnp.array([1.0, 1.0])
        elif n_points == 3:
            points = jnp.array([-jnp.sqrt(3/5), 0.0, jnp.sqrt(3/5)])
            weights = jnp.array([5/9, 8/9, 5/9])
        elif n_points == 4:
            p1 = jnp.sqrt(3/7 - 2/7 * jnp.sqrt(6/5))
            p2 = jnp.sqrt(3/7 + 2/7 * jnp.sqrt(6/5))
            w1 = (18 + jnp.sqrt(30)) / 36
            w2 = (18 - jnp.sqrt(30)) / 36
            points = jnp.array([-p2, -p1, p1, p2])
            weights = jnp.array([w2, w1, w1, w2])
        else:
            raise ValueError(f"Gauss quadrature with {n_points} points not implemented")
        
        return points, weights
    
    def _linear_shape_functions(self, xi):
        """
        Linear Lagrange shape functions on reference element [-1, 1]
        
        Args:
            xi: Point in reference element
            
        Returns:
            N: Shape function values [N1, N2]
        """
        N1 = 0.5 * (1 - xi)
        N2 = 0.5 * (1 + xi)
        return jnp.array([N1, N2])
    
    def _element_coordinates(self, element_idx):
        """Get physical coordinates of element nodes"""
        return jnp.array([self.nodes[element_idx], self.nodes[element_idx + 1]])
    
    def _jacobian(self, element_idx):
        """Compute Jacobian for mapping from reference to physical element"""
        x_coords = self._element_coordinates(element_idx)
        return (x_coords[1] - x_coords[0]) / 2.0
    
    def _map_to_physical(self, xi, element_idx):
        """Map point from reference element to physical element"""
        x_coords = self._element_coordinates(element_idx)
        return 0.5 * ((1 - xi) * x_coords[0] + (1 + xi) * x_coords[1])
    
    @partial(jax.jit, static_argnums=0)
    def _evaluate_u_at_gauss_point(self, xi, element_idx, u_dofs):
        """
        Evaluate solution u at a Gauss point within an element
        
        Args:
            xi: Gauss point in reference element
            element_idx: Element index
            u_dofs: Global DOF vector
            
        Returns:
            u_value: Value of u at the Gauss point
        """
        N = self._linear_shape_functions(xi)
        # Get global DOF indices for this element
        dof_indices = jnp.array([element_idx, element_idx + 1])
        u_element = u_dofs[dof_indices]
        return jnp.dot(N, u_element)
    
    @partial(jax.jit, static_argnums=0)  
    def _element_forcing_contribution(self, element_idx, u_dofs):
        """
        Compute element contribution to forcing vector f(u)
        
        Args:
            element_idx: Element index
            u_dofs: Global DOF vector
            
        Returns:
            f_element: Element contribution to forcing vector
        """
        J = self._jacobian(element_idx)
        f_element = jnp.zeros(2)
        
        # Loop over Gauss points
        for gp in range(self.n_gauss_points):
            xi = self.gauss_points[gp]
            w = self.gauss_weights[gp]
            
            # Shape functions at Gauss point
            N = self._linear_shape_functions(xi)
            
            # Evaluate u at Gauss point
            u_gp = self._evaluate_u_at_gauss_point(xi, element_idx, u_dofs)
            
            # Integrand: phi_i * f(u)
            integrand = N * self.nonlinear_func(u_gp)
            
            # Add weighted contribution
            f_element = f_element + w * integrand * J
            
        return f_element
    
    # Define vectorized function as static method to avoid self issues
    @staticmethod
    @partial(jax.jit, static_argnums=-1)
    def _element_forcing_contribution_static(element_idx, u_dofs, nodes, gauss_points, gauss_weights, nonlinear_func):
        """Static version for vectorization"""
        # Compute element coordinates
        x_coords = jnp.array([nodes[element_idx], nodes[element_idx + 1]])
        J = (x_coords[1] - x_coords[0]) / 2.0
        
        def gauss_point_contribution(gp, f_element):
            """Process single Gauss point contribution"""
            xi = gauss_points[gp]
            w = gauss_weights[gp]
            
            # Shape functions at Gauss point
            N1 = 0.5 * (1 - xi)
            N2 = 0.5 * (1 + xi)
            N = jnp.array([N1, N2])
            
            # Evaluate u at Gauss point
            dof_indices = jnp.array([element_idx, element_idx + 1])
            u_element = u_dofs[dof_indices]
            u_gp = jnp.dot(N, u_element)
            
            # Integrand: phi_i * f(u) where f is the user-specified function
            integrand = N * nonlinear_func(u_gp)
            
            # Add weighted contribution
            return f_element + w * integrand * J
        
        # Use lax.fori_loop for JAX-compatible iteration
        f_element_initial = jnp.zeros(2)
        n_gauss = len(gauss_points)
        f_element = jax.lax.fori_loop(0, n_gauss, gauss_point_contribution, f_element_initial)
        
        return f_element
    
    @partial(jax.jit, static_argnums=0)
    def assemble_forcing_vector(self, u_dofs):
        """
        Assemble global nonlinear forcing vector f(u)
        
        Args:
            u_dofs: Global DOF vector
            
        Returns:
            f_global: Global forcing vector
        """
        f_global = jnp.zeros(self.n_nodes)
        
        # Create vectorized function for all elements
        element_indices = jnp.arange(self.n_elements)
        
        # Vectorize the static function
        vmap_func = vmap(self._element_forcing_contribution_static, 
                        in_axes=(0, None, None, None, None, None), out_axes=0)
        
        # Get contributions from all elements
        f_elements = vmap_func(element_indices, u_dofs, self.nodes, 
                             self.gauss_points, self.gauss_weights, self.nonlinear_func)
        
        # Assemble into global vector
        for e in range(self.n_elements):
            # Global DOF indices for element e
            dof_indices = jnp.array([e, e + 1])
            f_global = f_global.at[dof_indices].add(f_elements[e])
            
        return f_global
    

# Example usage and testing
def example_usage():
    """Demonstrate usage of the NonlinearFEM1D class"""
    
    # Define 1D mesh
    n_elements = 10
    nodes = jnp.linspace(0.0, 1.0, n_elements + 1)
    
    # Example 1: cosine function
    print("Example 1: Using cos(u)")
    fem1 = NonlinearFEM1D(nodes, nonlinear_func=jnp.cos, n_gauss_points=3)
    
    # Example 2: Custom sine function
    print("Example 2: Using sin(u)")
    fem2 = NonlinearFEM1D(nodes, nonlinear_func=jnp.sin, n_gauss_points=3)
    
    # Example 3: Custom polynomial function
    print("Example 3: Using u³ - u")
    def poly_func(u):
        return u**3 - u
    fem3 = NonlinearFEM1D(nodes, nonlinear_func=poly_func, n_gauss_points=3)
    
    # Example 4: Custom exponential function with limiting
    print("Example 4: Using tanh(u) (bounded nonlinearity)")
    fem4 = NonlinearFEM1D(nodes, nonlinear_func=jnp.tanh, n_gauss_points=3)
    
    # Example DOF vector (some arbitrary solution)
    u_dofs = jnp.sin(jnp.pi * nodes) + 0.1 * jnp.cos(2 * jnp.pi * nodes)
    
    # Assemble forcing vectors for different functions
    f1 = fem1.assemble_forcing_vector(u_dofs)
    f2 = fem2.assemble_forcing_vector(u_dofs)
    f3 = fem3.assemble_forcing_vector(u_dofs)
    f4 = fem4.assemble_forcing_vector(u_dofs)
    
    print("Mesh information:")
    print(f"Number of nodes: {fem1.n_nodes}")
    print(f"Number of elements: {fem1.n_elements}")
    print(f"Gauss points per element: {fem1.n_gauss_points}")
    print()
    
    print("Forcing vector L2 norms:")
    print(f"cos(u):     {jnp.linalg.norm(f1):.6f}")
    print(f"sin(u):     {jnp.linalg.norm(f2):.6f}")
    print(f"u³-u:       {jnp.linalg.norm(f3):.6f}")
    print(f"tanh(u):    {jnp.linalg.norm(f4):.6f}")
    print()
    
    return fem1, fem2, fem3, fem4, u_dofs


# Test with common nonlinear functions
def test_common_functions():
    """Test various common nonlinear functions"""
    
    nodes = jnp.linspace(0.0, 1.0, 11)  # 10 elements
    u_dofs = jnp.linspace(-1, 1, 11)  # Linear variation from -1 to 1
    
    # Define test functions
    test_functions = {
        'cos(u)': jnp.cos,
        'sin(u)': jnp.sin,
        'exp(u)': jnp.exp,
        'tanh(u)': jnp.tanh,
        'u²': lambda u: u**2,
        'u³': lambda u: u**3,
        '1/(1+u²)': lambda u: 1.0 / (1.0 + u**2),
        'arctan(u)': jnp.arctan,
        'sinh(u)': jnp.sinh,
        'u*exp(-u²)': lambda u: u * jnp.exp(-u**2)
    }
    
    print("Testing various nonlinear functions:")
    print("=" * 50)
    
    for name, func in test_functions.items():
        try:
            fem = NonlinearFEM1D(nodes, nonlinear_func=func, n_gauss_points=3)
            f_vector = fem.assemble_forcing_vector(u_dofs)
            norm = jnp.linalg.norm(f_vector)
            print(f"{name:12s}: L2 norm = {norm:10.6f}")
        except Exception as e:
            print(f"{name:12s}: Error - {str(e)}")


# Convergence test with different numbers of Gauss points
def convergence_test():
    """Test convergence with different numbers of Gauss points"""
    
    # Fixed mesh and solution
    nodes = jnp.linspace(0.0, 1.0, 21)  # 20 elements
    u_dofs = jnp.sin(jnp.pi * nodes)
    
    # Test with a more complex function
    complex_func = lambda u: jnp.sin(u) * jnp.exp(-u**2) + u**2
    
    print("Convergence test with complex function f(u) = sin(u)*exp(-u²) + u²:")
    print("=" * 70)
    
    reference_fem = NonlinearFEM1D(nodes, nonlinear_func=complex_func, n_gauss_points=4) 
    f_reference = reference_fem.assemble_forcing_vector(u_dofs)
    for n_gp in [1, 2, 3]:
        fem = NonlinearFEM1D(nodes, nonlinear_func=complex_func, n_gauss_points=n_gp)
        f_gp = fem.assemble_forcing_vector(u_dofs)
    
        error = jnp.linalg.norm(f_gp - f_reference) / jnp.linalg.norm(f_reference)
        print(f"Gauss points: {n_gp}, L2 norm: {jnp.linalg.norm(f_gp):.6f}, "
                  f"Relative error: {error:.2e}")


if __name__ == "__main__":
    # Run examples
    fem1, fem2, fem3, fem4, u_dofs = example_usage()
    print("\n" + "="*60 + "\n")
    test_common_functions()
    print("\n" + "="*60 + "\n")
    convergence_test()