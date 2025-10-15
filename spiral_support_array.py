import numpy as np
import time
from generate_ventricle import spiral_support, ellipse
from scipy.interpolate import RegularGridInterpolator

def compute_spiral_support_array(n_spirals=8, helix_angle=30, resolution=100):
    """
    Compute the spiral support SDF over a 200x200x200 array.
    
    Parameters:
    -----------
    n_spirals : int
        Number of spiral arms
    helix_angle : float
        Helix angle in degrees
    resolution : int
        Resolution of the array (default 200)
    
    Returns:
    --------
    array : numpy.ndarray
        3D array of spiral support SDF values
    x_coords, y_coords, z_coords : numpy.ndarray
        Coordinate arrays for the grid
    """
    
    # Define the range for the array
    x_range = (-13.5, 13.5)
    y_range = (-13.5, 13.5)
    z_range = (-31, 0.5)
    
    # Create coordinate arrays
    x_coords = np.linspace(x_range[0], x_range[1], resolution)
    y_coords = np.linspace(y_range[0], y_range[1], resolution)
    z_coords = np.linspace(z_range[0], z_range[1], resolution)
    
    # Initialize the 3D array
    spiral_array = np.zeros((resolution, resolution, resolution))
    
    print(f"Computing spiral support SDF over {resolution}x{resolution}x{resolution} array...")
    print(f"Range: x={x_range}, y={y_range}, z={z_range}")
    print(f"Parameters: n_spirals={n_spirals}, helix_angle={helix_angle}")
    
    start_time = time.time()
    
    # Compute spiral support at each point
    for i, x in enumerate(x_coords):
        if i % 20 == 0:  # Progress indicator
            elapsed = time.time() - start_time
            remaining = elapsed * (resolution - i) / max(i, 1)
            print(f"Progress: {i}/{resolution} ({i/resolution*100:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
        
        for j, y in enumerate(y_coords):
            for k, z in enumerate(z_coords):
                spiral_array[i, j, k] = spiral_support(x, y, z, n_spirals, helix_angle)
    
    total_time = time.time() - start_time
    print(f"Computation completed in {total_time:.2f} seconds")
    
    return spiral_array, x_coords, y_coords, z_coords

def save_spiral_array(array, x_coords, y_coords, z_coords, filename="spiral_support_array.npy"):
    """
    Save the spiral support array and coordinates to a file.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The 3D spiral support array
    x_coords, y_coords, z_coords : numpy.ndarray
        Coordinate arrays
    filename : str
        Output filename
    """
    data = {
        'array': array,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'z_coords': z_coords,
        'metadata': {
            'shape': array.shape,
            'x_range': (x_coords[0], x_coords[-1]),
            'y_range': (y_coords[0], y_coords[-1]),
            'z_range': (z_coords[0], z_coords[-1])
        }
    }
    
    np.save(filename, data)
    print(f"Array saved to {filename}")
    print(f"Array shape: {array.shape}")
    print(f"Value range: [{array.min():.3f}, {array.max():.3f}]")

def load_spiral_array(filename="spiral_support_array.npy"):
    """
    Load the spiral support array and coordinates from a file.
    
    Parameters:
    -----------
    filename : str
        Input filename
    
    Returns:
    --------
    array : numpy.ndarray
        The 3D spiral support array
    x_coords, y_coords, z_coords : numpy.ndarray
        Coordinate arrays
    metadata : dict
        Metadata about the array
    """
    data = np.load(filename, allow_pickle=True).item()
    return data['array'], data['x_coords'], data['y_coords'], data['z_coords'], data['metadata']

def interpolate_array(array, x_coords, y_coords, z_coords, target_resolution=200):
    """
    Interpolate the array to increase resolution using scipy's RegularGridInterpolator.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The 3D spiral support array
    x_coords, y_coords, z_coords : numpy.ndarray
        Original coordinate arrays
    target_resolution : int
        Target resolution for each dimension
    
    Returns:
    --------
    interpolated_array : numpy.ndarray
        Interpolated array with higher resolution
    new_x_coords, new_y_coords, new_z_coords : numpy.ndarray
        New coordinate arrays with higher resolution
    """
    print(f"Interpolating array from {array.shape} to ({target_resolution}, {target_resolution}, {target_resolution})...")
    
    # Create interpolator
    interpolator = RegularGridInterpolator((x_coords, y_coords, z_coords), array, 
                                         method='linear', bounds_error=False, fill_value=None)
    
    # Create new coordinate grids
    new_x_coords = np.linspace(x_coords[0], x_coords[-1], target_resolution)
    new_y_coords = np.linspace(y_coords[0], y_coords[-1], target_resolution)
    new_z_coords = np.linspace(z_coords[0], z_coords[-1], target_resolution)
    
    # Create meshgrid for interpolation points
    X, Y, Z = np.meshgrid(new_x_coords, new_y_coords, new_z_coords, indexing='ij')
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Interpolate
    interpolated_values = interpolator(points)
    interpolated_array = interpolated_values.reshape(target_resolution, target_resolution, target_resolution)
    
    print(f"Interpolation completed. New array shape: {interpolated_array.shape}")
    
    return interpolated_array, new_x_coords, new_y_coords, new_z_coords

def quantify_interpolation_error(original_array, interpolated_array, x_coords, y_coords, z_coords, 
                                new_x_coords, new_y_coords, new_z_coords):
    """
    Quantify the interpolation error by comparing interpolated values with exact values
    at a subset of points.
    
    Parameters:
    -----------
    original_array : numpy.ndarray
        Original array at lower resolution
    interpolated_array : numpy.ndarray
        Interpolated array at higher resolution
    x_coords, y_coords, z_coords : numpy.ndarray
        Original coordinate arrays
    new_x_coords, new_y_coords, new_z_coords : numpy.ndarray
        New coordinate arrays
    
    Returns:
    --------
    error_stats : dict
        Dictionary containing error statistics
    """
    print("Quantifying interpolation error...")
    
    # Sample a subset of points for error analysis (to avoid excessive computation)
    sample_size = min(1000, len(new_x_coords) * len(new_y_coords) * len(new_z_coords))
    
    # Randomly sample points from the interpolated array
    np.random.seed(42)  # For reproducible results
    total_points = len(new_x_coords) * len(new_y_coords) * len(new_z_coords)
    sample_indices = np.random.choice(total_points, sample_size, replace=False)
    
    # Convert linear indices to 3D indices
    i_indices, j_indices, k_indices = np.unravel_index(sample_indices, interpolated_array.shape)
    
    errors = []
    exact_values = []
    interpolated_values = []
    
    # Compute exact values at sampled points
    for idx in range(sample_size):
        i, j, k = i_indices[idx], j_indices[idx], k_indices[idx]
        x, y, z = new_x_coords[i], new_y_coords[j], new_z_coords[k]
        
        # Get interpolated value
        interp_val = interpolated_array[i, j, k]
        
        # Compute exact value using spiral_support function
        exact_val = spiral_support(x, y, z)
        
        # Calculate error
        error = abs(interp_val - exact_val)
        errors.append(error)
        exact_values.append(exact_val)
        interpolated_values.append(interp_val)
    
    # Convert to numpy arrays
    errors = np.array(errors)
    exact_values = np.array(exact_values)
    interpolated_values = np.array(interpolated_values)
    
    # Calculate error statistics
    error_stats = {
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'relative_rmse': np.sqrt(np.mean(errors**2)) / (np.max(exact_values) - np.min(exact_values)),
        'sample_size': sample_size,
        'exact_range': (np.min(exact_values), np.max(exact_values)),
        'interpolated_range': (np.min(interpolated_values), np.max(interpolated_values))
    }
    
    # Print error statistics
    print(f"Interpolation Error Analysis (based on {sample_size} sample points):")
    print(f"  Mean absolute error: {error_stats['mean_error']:.6f}")
    print(f"  Median absolute error: {error_stats['median_error']:.6f}")
    print(f"  Standard deviation of error: {error_stats['std_error']:.6f}")
    print(f"  Maximum absolute error: {error_stats['max_error']:.6f}")
    print(f"  Minimum absolute error: {error_stats['min_error']:.6f}")
    print(f"  Root Mean Square Error (RMSE): {error_stats['rmse']:.6f}")
    print(f"  Relative RMSE: {error_stats['relative_rmse']:.6f}")
    print(f"  Exact value range: [{error_stats['exact_range'][0]:.3f}, {error_stats['exact_range'][1]:.3f}]")
    print(f"  Interpolated value range: [{error_stats['interpolated_range'][0]:.3f}, {error_stats['interpolated_range'][1]:.3f}]")
    
    return error_stats

def plot_error_analysis(error_stats, exact_values, interpolated_values, errors):
    """
    Plot error analysis results.
    
    Parameters:
    -----------
    error_stats : dict
        Error statistics dictionary
    exact_values : numpy.ndarray
        Exact values at sample points
    interpolated_values : numpy.ndarray
        Interpolated values at sample points
    errors : numpy.ndarray
        Absolute errors at sample points
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Exact vs Interpolated values
    axes[0, 0].scatter(exact_values, interpolated_values, alpha=0.6, s=20)
    min_val = min(np.min(exact_values), np.min(interpolated_values))
    max_val = max(np.max(exact_values), np.max(interpolated_values))
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('Exact Values')
    axes[0, 0].set_ylabel('Interpolated Values')
    axes[0, 0].set_title('Exact vs Interpolated Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Error histogram
    axes[0, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(error_stats['mean_error'], color='red', linestyle='--', 
                       label=f'Mean: {error_stats["mean_error"]:.4f}')
    axes[0, 1].axvline(error_stats['median_error'], color='green', linestyle='--', 
                       label=f'Median: {error_stats["median_error"]:.4f}')
    axes[0, 1].set_xlabel('Absolute Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error vs Exact value
    axes[1, 0].scatter(exact_values, errors, alpha=0.6, s=20)
    axes[1, 0].set_xlabel('Exact Values')
    axes[1, 0].set_ylabel('Absolute Error')
    axes[1, 0].set_title('Error vs Exact Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Cumulative error distribution
    sorted_errors = np.sort(errors)
    cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axes[1, 1].plot(sorted_errors, cumulative_prob, linewidth=2)
    axes[1, 1].axhline(0.95, color='red', linestyle='--', label='95%')
    axes[1, 1].axhline(0.99, color='orange', linestyle='--', label='99%')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print additional statistics
    print(f"\nAdditional Error Statistics:")
    print(f"  95th percentile error: {np.percentile(errors, 95):.6f}")
    print(f"  99th percentile error: {np.percentile(errors, 99):.6f}")
    print(f"  Percentage of errors < 0.01: {np.mean(errors < 0.01) * 100:.1f}%")
    print(f"  Percentage of errors < 0.1: {np.mean(errors < 0.1) * 100:.1f}%")

def visualize_slices(array, x_coords, y_coords, z_coords, slice_indices=None, interpolate=True, target_resolution=200):
    """
    Visualize slices of the spiral support array with optional interpolation.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The 3D spiral support array
    x_coords, y_coords, z_coords : numpy.ndarray
        Coordinate arrays
    slice_indices : tuple or None
        Indices for x, y, z slices to visualize. If None, uses middle slices.
    interpolate : bool
        Whether to interpolate the array before visualization
    target_resolution : int
        Target resolution for interpolation
    """
    import matplotlib.pyplot as plt
    
    # Interpolate if requested
    if interpolate and array.shape[0] < target_resolution:
        array, x_coords, y_coords, z_coords = interpolate_array(array, x_coords, y_coords, z_coords, target_resolution)
    
    if slice_indices is None:
        slice_indices = (array.shape[0]//2, array.shape[1]//2, array.shape[2]//2)
    
    i, j, k = slice_indices
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XY slice (constant z)
    im1 = axes[0].imshow(array[:, :, k], extent=[y_coords[0], y_coords[-1], x_coords[0], x_coords[-1]], 
                         origin='lower', cmap='RdBu_r', aspect='equal')
    axes[0].set_title(f'XY slice at z={z_coords[k]:.2f}')
    axes[0].set_xlabel('Y')
    axes[0].set_ylabel('X')
    plt.colorbar(im1, ax=axes[0])
    
    # XZ slice (constant y)
    im2 = axes[1].imshow(array[:, j, :], extent=[z_coords[0], z_coords[-1], x_coords[0], x_coords[-1]], 
                         origin='lower', cmap='RdBu_r', aspect='auto')
    axes[1].set_title(f'XZ slice at y={y_coords[j]:.2f}')
    axes[1].set_xlabel('Z')
    axes[1].set_ylabel('X')
    plt.colorbar(im2, ax=axes[1])
    
    # YZ slice (constant x)
    im3 = axes[2].imshow(array[i, :, :], extent=[z_coords[0], z_coords[-1], y_coords[0], y_coords[-1]], 
                         origin='lower', cmap='RdBu_r', aspect='auto')
    axes[2].set_title(f'YZ slice at x={x_coords[i]:.2f}')
    axes[2].set_xlabel('Z')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Compute the spiral support array
    spiral_array, x_coords, y_coords, z_coords = compute_spiral_support_array()
    
    # Save the array
    save_spiral_array(spiral_array, x_coords, y_coords, z_coords)
    
    # Interpolate for visualization and error analysis
    print("\n" + "="*50)
    print("INTERPOLATION AND ERROR ANALYSIS")
    print("="*50)
    
    interpolated_array, new_x_coords, new_y_coords, new_z_coords = interpolate_array(
        spiral_array, x_coords, y_coords, z_coords, target_resolution=200
    )
    
    # Quantify interpolation error
    error_stats = quantify_interpolation_error(
        spiral_array, interpolated_array, x_coords, y_coords, z_coords,
        new_x_coords, new_y_coords, new_z_coords
    )
    
    # Plot error analysis
    print("\n" + "="*50)
    print("ERROR ANALYSIS PLOTS")
    print("="*50)
    
    # Recompute exact values for plotting (from the error analysis)
    sample_size = error_stats['sample_size']
    np.random.seed(42)
    total_points = len(new_x_coords) * len(new_y_coords) * len(new_z_coords)
    sample_indices = np.random.choice(total_points, sample_size, replace=False)
    i_indices, j_indices, k_indices = np.unravel_index(sample_indices, interpolated_array.shape)
    
    exact_values = []
    interpolated_values = []
    errors = []
    
    for idx in range(sample_size):
        i, j, k = i_indices[idx], j_indices[idx], k_indices[idx]
        x, y, z = new_x_coords[i], new_y_coords[j], new_z_coords[k]
        interp_val = interpolated_array[i, j, k]
        exact_val = spiral_support(x, y, z)
        error = abs(interp_val - exact_val)
        
        exact_values.append(exact_val)
        interpolated_values.append(interp_val)
        errors.append(error)
    
    exact_values = np.array(exact_values)
    interpolated_values = np.array(interpolated_values)
    errors = np.array(errors)
    
    plot_error_analysis(error_stats, exact_values, interpolated_values, errors)
    
    # Visualize slices with interpolated data
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    visualize_slices(interpolated_array, new_x_coords, new_y_coords, new_z_coords, interpolate=False)
