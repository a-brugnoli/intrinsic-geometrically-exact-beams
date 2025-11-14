import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_beam_deformation(time_vector, coordinates, fps=30, interval=50, 
                           save_path=None, title="Beam Animation",
                           lock_at_end=False):
    """
    Create an animation of a single beam deformation over time.
    
    For dual beam animations, use animate_dual_beam_deformation() instead.
    
    Parameters:
    -----------
    lock_at_end : bool
        If True, animation stops at the last frame. If False, animation loops.
    """
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
    # Force axes to have similar physical dimensions regardless of data scale
    ax.set_aspect('auto')
    
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
                                 blit=True, repeat=not lock_at_end)
    
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


def animate_overlay_beam_deformation(time_vector, beam_coordinates,
                                   fps=30, interval=50, save_path=None,
                                   title="Beam Animation",
                                   colors=None, labels=None,
                                   markers=None, lock_at_end=False):
    """
    Create an animation showing multiple beam deformations overlaid on the same plot.
    
    Parameters:
    -----------
    time_vector : jax.numpy.ndarray
        Time vector of size (N_time,)
    beam_coordinates : list of jax.numpy.ndarray
        List of beam coordinates, each of size (2, N_space_i, N_time)
    fps : int
        Frames per second for the animation
    interval : int
        Delay between frames in milliseconds
    save_path : str, optional
        Path to save the animation
    title : str
        Title for the animation
    colors : list of str, optional
        Colors for each beam. If None, uses default color cycle
    labels : list of str, optional
        Labels for legend. If None, uses 'Beam 1', 'Beam 2', etc.
    markers : list of str, optional
        Markers for each beam. If None, cycles through default markers
    lock_at_end : bool
        If True, animation stops at the last frame. If False, animation loops.
    
    Returns:
    --------
    matplotlib.animation.FuncAnimation
        Animation object
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    # Determine number of beams
    n_beams = len(beam_coordinates)
    
    # Set default values if not provided
    if colors is None:
        default_colors = ['blue', 'red', 'green', 'orange', 'purple', 
                         'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = [default_colors[i % len(default_colors)] for i in range(n_beams)]
    
    if labels is None:
        labels = [f'Beam {i+1}' for i in range(n_beams)]
    
    if markers is None:
        default_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        markers = [default_markers[i % len(default_markers)] for i in range(n_beams)]
    
    # Extract dimensions
    N_time = len(time_vector)
    
    # Convert JAX arrays to numpy
    time_np = np.array(time_vector)
    coords_np = [np.array(coords) for coords in beam_coordinates]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate combined plot limits across all beams
    x_min = min(coords[0].min() for coords in coords_np)
    x_max = max(coords[0].max() for coords in coords_np)
    y_min = min(coords[1].min() for coords in coords_np)
    y_max = max(coords[1].max() for coords in coords_np)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    # Initialize line plots for all beams
    lines = []
    for i in range(n_beams):
        line, = ax.plot([], [], color=colors[i], linewidth=2, 
                       marker=markers[i], markersize=3, label=labels[i])
        lines.append(line)
    
    # Time display
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Configure axis
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto')
    ax.legend()
    
    def animate(frame):
        """Animation function for overlaid beams."""
        current_time = time_np[frame]
        
        # Update all beams
        for i, (line, coords) in enumerate(zip(lines, coords_np)):
            x_coords = coords[0, :, frame]
            y_coords = coords[1, :, frame]
            line.set_data(x_coords, y_coords)
        
        time_text.set_text(f'Time: {current_time:.3f}')
        
        return lines + [time_text]
    
    def init():
        """Initialize animation."""
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        return lines + [time_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=N_time, interval=interval,
                                 blit=True, repeat=not lock_at_end)
    
    # Save animation if path provided
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Overlay animation saved as GIF: {save_path}")
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps)
            print(f"Overlay animation saved as MP4: {save_path}")
        else:
            print("Unsupported format. Use .gif or .mp4")
    
    plt.tight_layout()
    return anim


def create_dual_example_beam_data():
    """Create example data for two different beams."""
    
    # Time parameters
    N_time = 100
    T_final = 2.0
    time_vector = jnp.linspace(0, T_final, N_time)
    
    # Beam 1: Cantilever beam
    N_space1 = 50
    L1 = 10.0
    x_beam1 = jnp.linspace(0, L1, N_space1)
    omega1 = 2 * jnp.pi * 2
    amplitude1 = 0.5
    
    coordinates1 = jnp.zeros((2, N_space1, N_time))
    coordinates1 = coordinates1.at[0, :, :].set(x_beam1[:, None])
    
    for i in range(N_time):
        t = time_vector[i]
        y_deform1 = amplitude1 * (x_beam1/L1)**2 * jnp.sin(omega1 * t)
        coordinates1 = coordinates1.at[1, :, i].set(y_deform1)
    
    # Beam 2: Simply supported beam (different frequency and amplitude)
    N_space2 = 40
    L2 = 8.0
    x_beam2 = jnp.linspace(0, L2, N_space2)
    omega2 = 2 * jnp.pi * 3  # Different frequency
    amplitude2 = 0.3
    
    coordinates2 = jnp.zeros((2, N_space2, N_time))
    coordinates2 = coordinates2.at[0, :, :].set(x_beam2[:, None])
    
    for i in range(N_time):
        t = time_vector[i]
        # Different deformation pattern (sinusoidal along length)
        y_deform2 = amplitude2 * jnp.sin(jnp.pi * x_beam2/L2) * jnp.sin(omega2 * t)
        coordinates2 = coordinates2.at[1, :, i].set(y_deform2 + 2)  # Offset vertically
    
    return time_vector, coordinates1, coordinates2
    

# Example usage
if __name__ == "__main__":
    # Create example data for two beams
    print("Creating dual beam deformation data...")
    time_vec, coords1, coords2 = create_dual_example_beam_data()
    
    anim_overlay = animate_overlay_beam_deformation(
        time_vec, [coords1, coords2],
        fps=20, interval=100,
        title="Beam Comparison: Overlaid (Stops at End)",
        colors=('blue', 'red'),
        labels=('Cantilever', 'Simply Supported'),
        lock_at_end=True  # This one will stop at the last frame
    )
    plt.show()
    
    print("Animations complete!")
    print(f"Data dimensions: Time steps = {len(time_vec)}")
    print(f"Beam 1: {coords1.shape[1]} spatial points")
    print(f"Beam 2: {coords2.shape[1]} spatial points")