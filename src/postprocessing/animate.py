import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_beam_deformation(time_vector, coordinates, fps=30, interval=50, 
                           save_path=None, title="Beam Deformation Animation"):
    """
    Create an animation of beam deformation over time.
    
    Parameters:
    -----------
    time_vector : jax.numpy.ndarray
        Time vector of size (N_time,)
    coordinates : jax.numpy.ndarray
        Array of size (2, N_space, N_time) where:
        - coordinates[0, :, :] are x-coordinates
        - coordinates[1, :, :] are y-coordinates
    fps : int
        Frames per second for the animation
    interval : int
        Delay between frames in milliseconds
    save_path : str, optional
        Path to save the animation (e.g., 'beam_animation.gif' or 'beam_animation.mp4')
    title : str
        Title for the animation
    
    Returns:
    --------
    matplotlib.animation.FuncAnimation
        Animation object
    """
    
    # Extract dimensions
    N_time = len(time_vector)
    N_space = coordinates.shape[1]
    
    # Convert JAX arrays to numpy for matplotlib compatibility
    time_np = np.array(time_vector)
    coords_np = np.array(coordinates)
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate plot limits with some padding
    x_min, x_max = coords_np[0].min(), coords_np[0].max()
    y_min, y_max = coords_np[1].min(), coords_np[1].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    
    # Set axes limits with padding
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    # Initialize the line plot
    line, = ax.plot([], [], 'b-', linewidth=2, marker='o', markersize=3)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto', adjustable='box')
    
    def animate(frame):
        """Animation function called for each frame."""
        # Get coordinates for current time step
        x_coords = coords_np[0, :, frame]
        y_coords = coords_np[1, :, frame]
        
        # Update line data
        line.set_data(x_coords, y_coords)
        
        # Update time display
        current_time = time_np[frame]
        time_text.set_text(f'Time: {current_time:.3f}')
        
        return line, time_text
    
    def init():
        """Initialize animation."""
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=N_time, interval=interval, 
                                 blit=True, repeat=True)
    
    # Save animation if path provided
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Animation saved as GIF: {save_path}")
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps)
            print(f"Animation saved as MP4: {save_path}")
        else:
            print("Unsupported format. Use .gif or .mp4")
    
    plt.tight_layout()
    return anim

# Example usage and test function
def create_example_beam_data():
    """Create example beam deformation data for testing."""
    
    # Time parameters
    N_time = 100
    T_final = 2.0
    time_vector = jnp.linspace(0, T_final, N_time)
    
    # Spatial parameters
    N_space = 50
    L = 10.0  # beam length
    x_beam = jnp.linspace(0, L, N_space)
    
    # Create a simple beam deformation (e.g., vibrating cantilever beam)
    omega = 2 * jnp.pi * 2  # frequency
    amplitude = 0.5
    
    # Initialize coordinates array
    coordinates = jnp.zeros((2, N_space, N_time))
    
    # Fill x-coordinates (constant along beam length)
    coordinates = coordinates.at[0, :, :].set(x_beam[:, None])
    
    # Fill y-coordinates (time-varying deformation)
    for i in range(N_time):
        t = time_vector[i]
        # Simple sinusoidal deformation that varies along the beam
        y_deform = amplitude * (x_beam/L)**2 * jnp.sin(omega * t)
        coordinates = coordinates.at[1, :, i].set(y_deform)
    
    return time_vector, coordinates

# Example usage
if __name__ == "__main__":
    # Create example data
    print("Creating example beam deformation data...")
    time_vec, coords = create_example_beam_data()
    
    # Create and display animation
    print("Creating animation...")
    anim = animate_beam_deformation(
        time_vec, coords, 
        fps=20, 
        interval=100,
        title="Example: Vibrating Cantilever Beam",
        save_path=None  # Set to "beam_animation.gif" to save
    )
    
    # Show the animation
    plt.show()
    