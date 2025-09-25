import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_beam_deformation(time_vector, coordinates, fps=30, interval=50, 
                           save_path=None, title="Beam Deformation Animation",
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

# Example usage and test function
def animate_dual_beam_deformation(time_vector, coordinates1, coordinates2, 
                                 fps=30, interval=50, save_path=None,
                                 titles=("Beam 1", "Beam 2"), 
                                 main_title="Dual Beam Deformation Animation",
                                 colors=('blue', 'red'), lock_at_end=False):
    """
    Create an animation showing two beam deformations simultaneously.
    
    Parameters:
    -----------
    time_vector : jax.numpy.ndarray
        Time vector of size (N_time,)
    coordinates1 : jax.numpy.ndarray
        First beam coordinates of size (2, N_space1, N_time)
    coordinates2 : jax.numpy.ndarray
        Second beam coordinates of size (2, N_space2, N_time)
    fps : int
        Frames per second for the animation
    interval : int
        Delay between frames in milliseconds
    save_path : str, optional
        Path to save the animation
    titles : tuple
        Titles for each beam subplot
    main_title : str
        Main title for the entire figure
    colors : tuple
        Colors for each beam ('color1', 'color2')
    lock_at_end : bool
        If True, animation stops at the last frame. If False, animation loops.
    
    Returns:
    --------
    matplotlib.animation.FuncAnimation
        Animation object
    """
    
    # Extract dimensions
    N_time = len(time_vector)
    
    # Convert JAX arrays to numpy
    time_np = np.array(time_vector)
    coords1_np = np.array(coordinates1)
    coords2_np = np.array(coordinates2)
    
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    
    # Setup for beam 1
    x1_min, x1_max = coords1_np[0].min(), coords1_np[0].max()
    y1_min, y1_max = coords1_np[1].min(), coords1_np[1].max()
    x1_range = x1_max - x1_min
    y1_range = y1_max - y1_min
    padding = 0.1
    
    ax1.set_xlim(x1_min - padding * x1_range, x1_max + padding * x1_range)
    ax1.set_ylim(y1_min - padding * y1_range, y1_max + padding * y1_range)
    
    # Setup for beam 2
    x2_min, x2_max = coords2_np[0].min(), coords2_np[0].max()
    y2_min, y2_max = coords2_np[1].min(), coords2_np[1].max()
    x2_range = x2_max - x2_min
    y2_range = y2_max - y2_min
    
    ax2.set_xlim(x2_min - padding * x2_range, x2_max + padding * x2_range)
    ax2.set_ylim(y2_min - padding * y2_range, y2_max + padding * y2_range)
    
    # Initialize line plots
    line1, = ax1.plot([], [], color=colors[0], linewidth=2, marker='o', markersize=3)
    line2, = ax2.plot([], [], color=colors[1], linewidth=2, marker='o', markersize=3)
    
    # Time displays
    time_text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    time_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Configure axes
    for ax, title in zip([ax1, ax2], titles):
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('auto')
    
    def animate(frame):
        """Animation function for dual beams."""
        current_time = time_np[frame]
        
        # Update beam 1
        x1_coords = coords1_np[0, :, frame]
        y1_coords = coords1_np[1, :, frame]
        line1.set_data(x1_coords, y1_coords)
        time_text1.set_text(f'Time: {current_time:.3f}')
        
        # Update beam 2
        x2_coords = coords2_np[0, :, frame]
        y2_coords = coords2_np[1, :, frame]
        line2.set_data(x2_coords, y2_coords)
        time_text2.set_text(f'Time: {current_time:.3f}')
        
        return line1, line2, time_text1, time_text2
    
    def init():
        """Initialize animation."""
        line1.set_data([], [])
        line2.set_data([], [])
        time_text1.set_text('')
        time_text2.set_text('')
        return line1, line2, time_text1, time_text2
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=N_time, interval=interval,
                                 blit=True, repeat=not lock_at_end)
    
    # Save animation if path provided
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Dual animation saved as GIF: {save_path}")
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=fps)
            print(f"Dual animation saved as MP4: {save_path}")
        else:
            print("Unsupported format. Use .gif or .mp4")
    
    plt.tight_layout()
    return anim


def animate_overlay_beam_deformation(time_vector, coordinates1, coordinates2,
                                   fps=30, interval=50, save_path=None,
                                   title="Overlaid Beam Deformation Animation",
                                   colors=('blue', 'red'),
                                   labels=('Beam 1', 'Beam 2'),
                                   lock_at_end=False):
    """
    Create an animation showing two beam deformations overlaid on the same plot.
    
    Parameters:
    -----------
    time_vector : jax.numpy.ndarray
        Time vector of size (N_time,)
    coordinates1 : jax.numpy.ndarray
        First beam coordinates of size (2, N_space1, N_time)
    coordinates2 : jax.numpy.ndarray
        Second beam coordinates of size (2, N_space2, N_time)
    fps : int
        Frames per second for the animation
    interval : int
        Delay between frames in milliseconds
    save_path : str, optional
        Path to save the animation
    title : str
        Title for the animation
    colors : tuple
        Colors for each beam ('color1', 'color2')
    labels : tuple
        Labels for legend ('label1', 'label2')
    lock_at_end : bool
        If True, animation stops at the last frame. If False, animation loops.
    
    Returns:
    --------
    matplotlib.animation.FuncAnimation
        Animation object
    """
    """
    Create an animation showing two beam deformations overlaid on the same plot.
    
    Parameters:
    -----------
    time_vector : jax.numpy.ndarray
        Time vector of size (N_time,)
    coordinates1 : jax.numpy.ndarray
        First beam coordinates of size (2, N_space1, N_time)
    coordinates2 : jax.numpy.ndarray
        Second beam coordinates of size (2, N_space2, N_time)
    fps : int
        Frames per second for the animation
    interval : int
        Delay between frames in milliseconds
    save_path : str, optional
        Path to save the animation
    title : str
        Title for the animation
    colors : tuple
        Colors for each beam ('color1', 'color2')
    labels : tuple
        Labels for legend ('label1', 'label2')
    
    Returns:
    --------
    matplotlib.animation.FuncAnimation
        Animation object
    """
    
    # Extract dimensions
    N_time = len(time_vector)
    
    # Convert JAX arrays to numpy
    time_np = np.array(time_vector)
    coords1_np = np.array(coordinates1)
    coords2_np = np.array(coordinates2)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate combined plot limits
    x_min = min(coords1_np[0].min(), coords2_np[0].min())
    x_max = max(coords1_np[0].max(), coords2_np[0].max())
    y_min = min(coords1_np[1].min(), coords2_np[1].min())
    y_max = max(coords1_np[1].max(), coords2_np[1].max())
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    padding = 0.1
    
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    
    # Initialize line plots
    line1, = ax.plot([], [], color=colors[0], linewidth=2, marker='o', 
                     markersize=3, label=labels[0])
    line2, = ax.plot([], [], color=colors[1], linewidth=2, marker='s', 
                     markersize=3, label=labels[1])
    
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
        
        # Update both beams
        x1_coords = coords1_np[0, :, frame]
        y1_coords = coords1_np[1, :, frame]
        line1.set_data(x1_coords, y1_coords)
        
        x2_coords = coords2_np[0, :, frame]
        y2_coords = coords2_np[1, :, frame]
        line2.set_data(x2_coords, y2_coords)
        
        time_text.set_text(f'Time: {current_time:.3f}')
        
        return line1, line2, time_text
    
    def init():
        """Initialize animation."""
        line1.set_data([], [])
        line2.set_data([], [])
        time_text.set_text('')
        return line1, line2, time_text
    
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
    # Create example data for two beams
    print("Creating dual beam deformation data...")
    time_vec, coords1, coords2 = create_dual_example_beam_data()
    
    # Option 1: Side-by-side comparison (loops continuously)
    print("Creating side-by-side animation (looping)...")
    anim_dual = animate_dual_beam_deformation(
        time_vec, coords1, coords2,
        fps=20, interval=100,
        titles=("Cantilever Beam", "Simply Supported Beam"),
        main_title="Beam Comparison: Side by Side (Looping)",
        colors=('blue', 'red'),
        lock_at_end=False
    )
    plt.show()
    
    # Option 2: Overlaid on same plot (stops at end)
    print("Creating overlay animation (stops at end)...")
    anim_overlay = animate_overlay_beam_deformation(
        time_vec, coords1, coords2,
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