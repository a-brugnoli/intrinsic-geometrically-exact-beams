import numpy as np
import matplotlib.pyplot as plt

def plot_piecewise_constant(nodes, element_values):
    """
    Plot a piecewise constant function from 1D mesh nodes and element values.
    
    Parameters:
    -----------
    nodes : array-like
        x-coordinates of mesh nodes [x0, x1, x2, ..., xN] defining N elements
    element_values : array-like  
        function values for each element (length N)
        
    Returns:
    --------
    x_plot : ndarray
        x-coordinates for plotting (separate segments)
    y_plot : ndarray
        y-coordinates for plotting (separate segments)
    """
    nodes = np.array(nodes)
    element_values = np.array(element_values)
    
    # Check dimensions
    n_elements = len(nodes) - 1
    if len(element_values) != n_elements:
        raise ValueError(f"Number of element values ({len(element_values)}) "
                        f"must equal number of elements ({n_elements})")
    
    # Create plotting arrays for horizontal segments only
    x_plot = []
    y_plot = []
    
    for i in range(n_elements):
        x_left = nodes[i]
        x_right = nodes[i + 1]
        value = element_values[i]
        
        # Add horizontal segment
        x_plot.extend([x_left, x_right, np.nan])  # np.nan creates break between segments
        y_plot.extend([value, value, np.nan])
    
    return np.array(x_plot), np.array(y_plot)

def plot_mesh_and_function(nodes, element_values, show_nodes=True, show_elements=True):
    """
    Plot the piecewise constant function with optional mesh visualization.
    
    Parameters:
    -----------
    nodes : array-like
        mesh node coordinates
    element_values : array-like
        function values on elements
    show_nodes : bool
        whether to show node positions
    show_elements : bool  
        whether to show element boundaries
    """
    x_plot, y_plot = plot_piecewise_constant(nodes, element_values)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the piecewise constant function
    plt.plot(x_plot, y_plot, 'b-', linewidth=2, label='Piecewise constant function')
    
    if show_nodes:
        # Show node positions on x-axis
        plt.plot(nodes, np.zeros_like(nodes), 'ro', markersize=6, 
                label=f'Nodes ({len(nodes)} nodes)')
    
    if show_elements:
        # Show element centers and values
        element_centers = (nodes[:-1] + nodes[1:]) / 2
        plt.plot(element_centers, element_values, 'gs', markersize=8, 
                label=f'Element values ({len(element_values)} elements)')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Piecewise Constant Function on 1D Finite Element Mesh')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage
def example_usage():
    """Demonstrate with sample mesh and function data"""
    
    # Example 1: Uniform mesh
    print("Example 1: Uniform mesh")
    nodes_uniform = np.linspace(0, 5, 6)  # 5 elements
    element_values_uniform = np.array([1.0, 2.5, 1.8, 3.2, 2.1])
    
    plot_mesh_and_function(nodes_uniform, element_values_uniform)
    
    # Example 2: Non-uniform mesh
    print("\nExample 2: Non-uniform mesh")
    nodes_nonuniform = np.array([0, 0.5, 1.2, 2.0, 3.5, 5.0])  # 5 elements
    element_values_nonuniform = np.array([0.8, 2.2, 1.5, 2.8, 1.9])
    
    plot_mesh_and_function(nodes_nonuniform, element_values_nonuniform)
    
    # Example 3: Just get plotting arrays without plotting
    print("\nExample 3: Get plotting data only")
    x_data, y_data = plot_piecewise_constant(nodes_uniform, element_values_uniform)
    print(f"x_data length: {len(x_data)}")
    print(f"y_data length: {len(y_data)}")
    print(f"First few x values: {x_data[:6]}")
    print(f"First few y values: {y_data[:6]}")

def simple_plot_example():
    """Minimal plotting example"""
    # Define mesh nodes
    nodes = np.array([0, 1, 2, 3, 4, 5])
    
    # Define function values on each element
    element_values = np.array([1.5, 0.8, 2.2, 1.1, 1.9])
    
    # Get plotting data
    x, y = plot_piecewise_constant(nodes, element_values)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Piecewise Constant Function')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    example_usage()
    simple_plot_example()