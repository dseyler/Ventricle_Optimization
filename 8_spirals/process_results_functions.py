import os
import shutil
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.interpolate import interp1d
import sys
from cycler import cycler
import warnings
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import argrelextrema
import pickle
from tabulate import tabulate
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
from scipy.integrate import trapezoid
from scipy.spatial.distance import directed_hausdorff


def get_stylized_variable_name_map():
    """
    Returns a dictionary mapping variable names to their LaTeX-formatted equivalents.
    This function centralizes all variable name mappings for consistency and maintainability.
    """
    return {
        # Volumes
        'V_LV': r'$V_\mathrm{LV}$',
        'V_RV': r'$V_\mathrm{RV}$',
        'V_LA': r'$V_\mathrm{LA}$',
        'V_RA': r'$V_\mathrm{RA}$',
        'V_AR_SYS': r'$V_\mathrm{AR}^\mathrm{SYS}$',
        'V_VEN_SYS': r'$V_\mathrm{VEN}^\mathrm{SYS}$',
        'V_AR_PUL': r'$V_\mathrm{AR}^\mathrm{PUL}$',
        'V_VEN_PUL': r'$V_\mathrm{VEN}^\mathrm{PUL}$',
        'V_tot': r'$V_\mathrm{total}$',
        
        # Pressures
        'p_LV': r'$p_\mathrm{LV}$',
        'p_RV': r'$p_\mathrm{RV}$',
        'p_LA': r'$p_\mathrm{LA}$',
        'p_RA': r'$p_\mathrm{RA}$',
        'p_AR_SYS': r'$p_\mathrm{AR}^\mathrm{SYS}$',
        'p_VEN_SYS': r'$p_\mathrm{VEN}^\mathrm{SYS}$',
        'p_AR_PUL': r'$p_\mathrm{AR}^\mathrm{PUL}$',
        'p_VEN_PUL': r'$p_\mathrm{VEN}^\mathrm{PUL}$',
        
        # Flows
        'Q_MV': r'$Q_\mathrm{MV}$',
        'Q_AV': r'$Q_\mathrm{AV}$',
        'Q_TV': r'$Q_\mathrm{TV}$',
        'Q_PV': r'$Q_\mathrm{PV}$',
        'Q_AR_SYS': r'$Q_\mathrm{AR}^\mathrm{SYS}$',
        'Q_VEN_SYS': r'$Q_\mathrm{VEN}^\mathrm{SYS}$',
        'Q_AR_PUL': r'$Q_\mathrm{AR}^\mathrm{PUL}$',
        'Q_VEN_PUL': r'$Q_\mathrm{VEN}^\mathrm{PUL}$',
        
        # Activations
        'A_LV': r'$A_\mathrm{LV}$',
        'A_RV': r'$A_\mathrm{RV}$',
        'A_LA': r'$A_\mathrm{LA}$',
        'A_RA': r'$A_\mathrm{RA}$',
        
        # Resistances
        'R_MV': r'$R_\mathrm{MV}$',
        'R_AV': r'$R_\mathrm{AV}$',
        'R_TV': r'$R_\mathrm{TV}$',
        'R_PV': r'$R_\mathrm{PV}$',
        
        # Other variables
        'fiber_stress': r'$\sigma_f$',
        'MV_plane_displacement': r'$d_\mathrm{MV}$',
        'TV_plane_displacement': r'$d_\mathrm{TV}$',
        'LV_longitudinal_length': r'$L_\mathrm{LV}$',
        'RV_longitudinal_length': r'$L_\mathrm{RV}$',
        'LV_wall_thickness': r'$h_\mathrm{LV}$',
        'RV_wall_thickness': r'$h_\mathrm{RV}$',
        'myocardial_volume': r'$V_\mathrm{myo}$',
        
        # Image variables (add _img suffix)
        'V_LV_img': r'$\hat{V}_\mathrm{LV}$',
        'V_RV_img': r'$\hat{V}_\mathrm{RV}$',
        'V_LA_img': r'$\hat{V}_\mathrm{LA}$',
        'V_RA_img': r'$\hat{V}_\mathrm{RA}$',
        'MV_plane_displacement_img': r'$\hat{d}_\mathrm{MV}$',
        'TV_plane_displacement_img': r'$\hat{d}_\mathrm{TV}$',
        'LV_longitudinal_length_img': r'$\hat{L}_\mathrm{LV}$',
        'RV_longitudinal_length_img': r'$\hat{L}_\mathrm{RV}$',
        'LV_wall_thickness_img': r'$\hat{h}_\mathrm{LV}$',
        'RV_wall_thickness_img': r'$\hat{h}_\mathrm{RV}$',
        'myocardial_volume_img': r'$\hat{V}_\mathrm{myo}$',

        # 0D variables
        'E_LV': r'$E_\mathrm{LV}$',
        'E_RV': r'$E_\mathrm{RV}$',
        'E_LA': r'$E_\mathrm{LA}$',
        'E_RA': r'$E_\mathrm{RA}$',
    }

def get_stylized_metric_name_map():
    """
    Returns a dictionary mapping metric names to their LaTeX-formatted equivalents.
    This function centralizes all error name mappings for consistency and maintainability.
    """
    return {
        # General error metrics
        'total_error': r'$total$',
        'sum_squared_error': r'$SSE$',
        
        # Volume metrics
        'V_LV_max': r'$V_\mathrm{LV}^\mathrm{max}$',
        'V_LV_min': r'$V_\mathrm{LV}^\mathrm{min}$',
        'V_LV': r'$V_\mathrm{LV}(t)$',
        'V_RV_max': r'$V_\mathrm{RV}^\mathrm{max}$',
        'V_RV_min': r'$V_\mathrm{RV}^\mathrm{min}$',
        'V_RV': r'$V_\mathrm{RV}(t)$',
        'V_LA_max': r'$V_\mathrm{LA}^\mathrm{max}$',
        'V_LA_min': r'$V_\mathrm{LA}^\mathrm{min}$',
        'V_LA': r'$V_\mathrm{LA}(t)$',
        'V_RA_max': r'$V_\mathrm{RA}^\mathrm{max}$',
        'V_RA_min': r'$V_\mathrm{RA}^\mathrm{min}$',
        'V_RA': r'$V_\mathrm{RA}(t)$',
        
        # Pressure metrics
        'P_LA_mean': r'$p_\mathrm{LA}^\mathrm{mean}$',
        'P_RA_mean': r'$p_\mathrm{RA}^\mathrm{mean}$',
        'P_AR_SYS_max': r'$p_\mathrm{AR}^\mathrm{SYS,max}$',
        'P_AR_SYS_min': r'$p_\mathrm{AR}^\mathrm{SYS,min}$',
        'P_AR_PUL_max': r'$p_\mathrm{AR}^\mathrm{PUL,max}$',
        'P_AR_PUL_min': r'$p_\mathrm{AR}^\mathrm{PUL,min}$',
        'P_VEN_SYS_mean': r'$p_\mathrm{VEN}^\mathrm{SYS,mean}$',
        'P_VEN_PUL_mean': r'$p_\mathrm{VEN}^\mathrm{PUL,mean}$',
        
        # Volume fraction metrics
        'LV_EF': r'$EF_\mathrm{LV}$',
        'RV_EF': r'$EF_\mathrm{RV}$',
        
        # Myocardial volume metrics
        'myocardial_volume': r'$V_\mathrm{myo}$'
    }

def read_fiber_stress_data(fiber_stress_file, total_sim_time):
    # Read the fiber stress data from file
    fiber_stress = pd.read_csv(fiber_stress_file, header=None, skiprows=1, names=['time', 'fiber_stress'], delimiter='\t')

    # Create periodic extension of the fiber stress data
    t_fiber_stress = np.array(fiber_stress['time'])[:-1]  # Remove last time point to get the period correct
    fiber_stress = np.array(fiber_stress['fiber_stress'])[:-1]
    period = t_fiber_stress[-1] - t_fiber_stress[0]
    num_repeat = int(np.ceil(total_sim_time / period))
    fiber_stress_extended = np.tile(fiber_stress, num_repeat)
    t_fiber_stress_extended = np.linspace(0, num_repeat*period, len(fiber_stress_extended))

    return t_fiber_stress_extended, fiber_stress_extended

def calc_radial_strain(surface, ref_surface):
    """
    Calculate the radial strain of the surface.
    """
    x = surface.points[:, 0]
    y = surface.points[:, 1]
    x_ref = ref_surface.points[:, 0]
    y_ref = ref_surface.points[:, 1]

    x_max, x_max_i = np.max(x), np.argmax(x)
    y_max = y[x_max_i]
    x_min, x_min_i = np.min(x), np.argmin(x)
    y_min = y[x_min_i]
    #y_max, y_max_i = np.max(y), np.argmax(y)
    #y_min, y_min_i = np.min(y), np.argmin(y)

    max_radius = 0.5*(np.sqrt(x_max**2 + y_max**2) + np.sqrt(x_min**2 + y_min**2)) #0.5 * ((x_max - x_min) + (y_max - y_min))

    x_ref_max = x_ref[x_max_i]
    x_ref_min = x_ref[x_min_i]
    y_ref_max = y_ref[x_max_i]
    y_ref_min = y_ref[x_min_i]
    #y_ref_max = y_ref[y_max_i]
    #y_ref_min = y_ref[y_min_i]

    ref_radius = 0.5*(np.sqrt(x_ref_max**2 + y_ref_max**2) + np.sqrt(x_ref_min**2 + y_ref_min**2)) #0.5 * ((x_ref_max - x_ref_min) + (y_ref_max - y_ref_min))

    d_R = ref_radius - max_radius
    return d_R / max_radius

def calc_longitudinal_strain(surface, ref_length):
    """
    Calculate the longitudinal strain of the surface.

    """
    z = surface.points[:, 2]
    L_final = (np.max(z) - np.min(z))
    d_L = ref_length - L_final
    return d_L / L_final

def calc_volume_3D(start_timestep, end_timestep, step, timestep_size, results_folder, reference_surface, save_intermediate_data=False, intermediate_output_folder=None):
    """
    Calculate the ventricular lumen volume at each time step from the results of 
    an svFSI struct simulation, in which a model of the myocardium is simulated

    Calculate the volume in the following steps
    1) Sample the result.vtu file onto the reference surface
    2) Warp the samples surface by the Displacement
    3) Flat fill any holes in the warped surface
    4) Calculate the volume of the warped and filled surface

    The units of volume are whatever units used in .vtu files, cubed. For example,
    if units of length in the .vtu files are microns, then the volume calculated
    here is cubic microns. 

    Args:
        start_timestep: The first svFSI result file to process
        
        end_timestep: The last svFSI result file to process
        
        step: The step in svFSI result files to process

        timestep_size: The size of the timestep in seconds
        
        results_folder: The absolute file path of the svFSI results folder 
        (usually something/something/16-procs/)
        
        reference_surface: The absolute file path of the .vtp file containing 
        the undeformed surface corresponding to the deformed surface of which 
        we want to compute the volume.

        save_intermediate_data: Whether to save intermediate data (resampled, warped, and filled surfaces)

        intermediate_output_folder: The folder to save the intermediate data in. If None, the intermediate data is saved in the same folder as the reference_surface.

    Returns: (t, vol), a tuple of lists of length number of time steps. t 
    contains the time, and vol contains the volume at that time.
    """
    
    # Create folder to contain intermediary meshes (mostly for checking for errors)
    if save_intermediate_data:
        assert intermediate_output_folder is not None, "If save_intermediate_data is True, intermediate_output_folder must be provided"
            
        # checking if the directory exists
        if not os.path.exists(intermediate_output_folder):
            # if the directory is not present then create it.
            os.makedirs(intermediate_output_folder)

    print('\n## Calculating volumes ##')

    # Load reference surface onto which we sample
    ref_surface = pv.read(f"{reference_surface}")
    
    # Initialize arrays to store time and volume
    t = []
    vol = []
    radial_strains = []
    longitudinal_strains = []

    # Make reference surface watertight
    ref_lumen = ref_surface.fill_holes(100) # 100 is the largest size of hole to fill
    
    # Recompute normals, incase the normals of the cap are opposite
    ref_lumen.compute_normals(inplace=True)

    # Save filled lumen (to check geometry and normals)
    # (Hopefully the normals on the filled cap will be consistent with the normals
    # on the rest of the surface, but you should check to make sure.)
    if save_intermediate_data:
        ref_lumen.save(os.path.join(intermediate_output_folder,  f'resampled_warped_and_filled_{0:03d}.vtp'))
    
    # Compute volume of ref_lumen
    print(f"Iteration: {0}, Volume: {ref_lumen.volume}")
    t.append(0)
    vol.append(ref_lumen.volume)
    # Compute reference length
    z_ref = ref_surface.points[:, 2]
    ref_length = np.max(z_ref) - np.min(z_ref)
    radial_strains.append(calc_radial_strain(ref_lumen, ref_lumen))
    longitudinal_strains.append(calc_longitudinal_strain(ref_lumen, ref_length))

    # If start_timestep is 0, this is a special case, as we have already computed the volume
    # of the reference surface. Also, the smallest first result file is result_001.vtu,
    # so we start at step.
    if start_timestep == 0:
        start_timestep = step
    
    # Loop through results files at each time > 0
    for k in range(start_timestep, end_timestep+1, step):
        # Load results VTU mesh
        result = pv.read(os.path.join(results_folder, f"result_{k:03d}.vtu"))

        # Sample result onto ref_lumen
        resampled_lumen = ref_lumen.sample(result)

        # Warp resampled surface by displacement (needed for current configuration 
        # normals, as well volume calculation)
        warped_lumen = resampled_lumen.warp_by_vector('Displacement')

        # Save warped and filled lumen (to check geometry and normals)
        if save_intermediate_data:
            warped_lumen.save(os.path.join(intermediate_output_folder, f'resampled_warped_and_filled_{k:03d}.vtp'))
        
        # Add time and volume to arrays
        t.append(k*timestep_size)
        vol.append(warped_lumen.volume)
        radial_strains.append(calc_radial_strain(warped_lumen, ref_lumen))
        longitudinal_strains.append(calc_longitudinal_strain(warped_lumen, ref_length))

        print(f"Iteration: {k}, Volume: {warped_lumen.volume}, Radial strain: {radial_strains[-1]}, Longitudinal strain: {longitudinal_strains[-1]}")
    
    return (t, vol, radial_strains, longitudinal_strains)

def calc_twist_angle(start_timestep, end_timestep, step, timestep_size, results_folder, reference_surface, z_levels=None, save_intermediate_data=False, intermediate_output_folder=None, tolerance=0.5):
    '''
    Calculate the twist angle of the ventricle as a function of time and z coordinate.
    
    The twist angle is calculated by:
    1. Loading the deformation gradient tensor (Def_grad) from simulation results
    2. Performing polar decomposition to extract the rotational component
    3. Computing the twist angle from the rotation matrix at different z-levels
    4. Tracking how this angle changes as the ventricle deforms
    
    Args:
        start_timestep: The first svFSI result file to process
        end_timestep: The last svFSI result file to process
        step: The step in svFSI result files to process
        timestep_size: The size of the timestep in seconds
        results_folder: The absolute file path of the svFSI results folder
        reference_surface: The absolute file path of the .vtp file containing the reference surface
        z_levels: List of z-coordinates at which to calculate twist angles. If None, uses default levels
        save_intermediate_data: Whether to save intermediate data
        intermediate_output_folder: Folder to save intermediate data
        tolerance: Tolerance for finding points near z-levels
        
    Returns:
        (t, twist_angles, z_levels): Tuple containing time points, twist angles for each z-level, and z-levels used
    '''
    
    # Load reference surface onto which we sample
    ref_surface = pv.read(f"{reference_surface}")
    
    # Define z-levels if not provided
    if z_levels is None:
        # Get the z-range of the surface
        z_min = ref_surface.bounds[4]
        z_max = ref_surface.bounds[5]
        z_levels = np.linspace(z_min + 0.02*(z_max-z_min), z_max - 0.02*(z_max-z_min), 10)
    
    print(f"Calculating twist angles at {len(z_levels)} z-levels: {z_levels}")
    
    # Initialize arrays to store time and twist angles
    t = []
    twist_angles = {z: [] for z in z_levels}
    
    def polar_decomposition(F):
        """
        Perform polar decomposition of deformation gradient tensor F = RU.
        
        Args:
            F: 3x3 deformation gradient tensor
            
        Returns:
            R: 3x3 rotation matrix
            U: 3x3 right stretch tensor
        """
        # Compute the right Cauchy-Green deformation tensor C = F^T F
        C = np.dot(F.T, F)
        
        # Compute eigenvalues and eigenvectors of C
        eigenvals, eigenvecs = np.linalg.eigh(C)
        
        # Ensure eigenvalues are positive (numerical stability)
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        # Compute U (right stretch tensor)
        U = np.dot(eigenvecs, np.dot(np.diag(np.sqrt(eigenvals)), eigenvecs.T))
        
        # Compute R (rotation matrix) = F U^(-1)
        U_inv = np.linalg.inv(U)
        R = np.dot(F, U_inv)
        
        return R, U
    
    def extract_rotation_angle(R):
        """
        Extract rotation angle from rotation matrix R.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            angle: Rotation angle in degrees around the z-axis
        """
        # Extract rotation around z-axis (assuming z is the longitudinal axis)
        # The rotation angle can be computed from the 2x2 submatrix in the xy-plane
        cos_theta = (R[0, 0] + R[1, 1]) / 2.0
        sin_theta = (R[1, 0] - R[0, 1]) / 2.0
        
        # Compute angle in radians
        angle_rad = np.arctan2(sin_theta, cos_theta)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def calculate_twist_angle_at_z_level(result_mesh, ref_surface, z_level, tolerance=0.5):
        """
        Calculate twist angle at a specific z-level using deformation gradient.
        
        Args:
            result_mesh: PyVista mesh with Def_grad data
            ref_surface: Reference surface mesh
            z_level: Z-coordinate to calculate twist at
            tolerance: Tolerance for finding points near z-level
            
        Returns:
            twist_angle: Average twist angle in degrees, or None if no points found
        """
        # Sample result onto reference surface
        resampled_surface = ref_surface.sample(result_mesh)
        
        # Get points near the z-level
        z_mask = np.abs(resampled_surface.points[:, 2] - z_level) < tolerance
        
        if not np.any(z_mask):
            return None
        
        # Get the deformation gradient data at these points
        if 'Def_grad' not in resampled_surface.point_data:
            print(f"Warning: Def_grad not found in result mesh at timestep")
            return None
        
        def_grad_data = resampled_surface.point_data['Def_grad']
        points_at_z = resampled_surface.points[z_mask]
        def_grad_at_z = def_grad_data[z_mask]
        
        if len(def_grad_at_z) == 0:
            return None
        
        # Calculate twist angles for all points at this z-level
        twist_angles_at_z = []
        
        for i, def_grad in enumerate(def_grad_at_z):
            try:
                # Reshape deformation gradient to 3x3 matrix
                # Assuming Def_grad is stored as a 9-component vector [F11, F12, F13, F21, F22, F23, F31, F32, F33]
                F = def_grad.reshape(3, 3)
                
                # Perform polar decomposition
                R, U = polar_decomposition(F)
                
                # Extract rotation angle
                angle = extract_rotation_angle(R)
                twist_angles_at_z.append(angle)
                
            except Exception as e:
                print(f"Warning: Could not process deformation gradient at point {i}: {e}")
                continue
        
        if len(twist_angles_at_z) == 0:
            return None
        
        # Return the average twist angle at this z-level
        return np.mean(twist_angles_at_z)
    
    # Calculate twist angles for reference surface (t=0)
    print("Calculating reference twist angles...")
    ref_twist_angles = {}
    for z in z_levels:
        # For reference (t=0), twist angle should be 0
        ref_twist_angles[z] = 0.0
        print(f"  Z={z:.2f}: twist angle = 0.00° (reference)")
    
    # Store reference twist angles
    t.append(0)
    for z in z_levels:
        twist_angles[z].append(ref_twist_angles[z])
    
    # Save reference surface if requested
    if save_intermediate_data and intermediate_output_folder is not None:
        if not os.path.exists(intermediate_output_folder):
            os.makedirs(intermediate_output_folder)
        ref_surface.save(os.path.join(intermediate_output_folder, 'surface_ref.vtp'))
    
    # Loop through results files at each time > 0
    for k in range(start_timestep, end_timestep+1, step):
        print(f"Processing timestep {k}...")
        
        # Load results VTU mesh
        result = pv.read(os.path.join(results_folder, f"result_{k:03d}.vtu"))
        
        # Check if Def_grad is available
        if 'Def_grad' not in result.point_data:
            print(f"Warning: Def_grad not found in result_{k:03d}.vtu")
            # Store None values for this timestep
            t.append(k * timestep_size)
            for z in z_levels:
                twist_angles[z].append(None)
            continue
        
        # Calculate twist angles at each z-level
        current_twist_angles = {}
        for z in z_levels:
            angle = calculate_twist_angle_at_z_level(result, ref_surface, z, tolerance)
            current_twist_angles[z] = angle
            if angle is not None:
                print(f"  Z={z:.2f}: twist angle = {angle:.2f}°")
            else:
                print(f"  Z={z:.2f}: no points found")
        
        # Save result mesh if requested
        if save_intermediate_data and intermediate_output_folder is not None:
            result.save(os.path.join(intermediate_output_folder, f'result_{k:03d}.vtu'))
        
        # Store results
        t.append(k * timestep_size)
        for z in z_levels:
            twist_angles[z].append(current_twist_angles[z])
    
    # Convert twist_angles dict to list format for easier plotting
    twist_angles_list = []
    for z in z_levels:
        twist_angles_list.append(twist_angles[z])
    
    return (t, twist_angles_list, z_levels)


def visualize_twist_angle_calculation(surface, z_levels, output_file=None):
    """
    Visualize the twist angle calculation by showing the points used at each z-level.
    
    Args:
        surface: PyVista surface mesh
        z_levels: List of z-coordinates to visualize
        output_file: Optional file path to save the visualization
        
    Returns:
        plotter: PyVista plotter object
    """
    plotter = pv.Plotter()
    
    # Add the surface
    plotter.add_mesh(surface, color='lightblue', opacity=0.7, show_edges=True)
    
    # Define colors for different z-levels
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_levels)))
    
    def find_opposite_points_at_z_level(surface, z_level, tolerance=0.1):
        """Helper function to find opposite points (same as in calc_twist_angle)"""
        z_mask = np.abs(surface.points[:, 2] - z_level) < tolerance
        if not np.any(z_mask):
            return None, None
            
        y_mask = np.abs(surface.points[:, 1]) < tolerance
        combined_mask = z_mask & y_mask
        
        if not np.any(combined_mask):
            return None, None
            
        points = surface.points[combined_mask]
        
        if len(points) < 2:
            return None, None
            
        x_coords = points[:, 0]
        left_idx = np.argmin(x_coords)
        right_idx = np.argmax(x_coords)
        
        left_point = points[left_idx]
        right_point = points[right_idx]
        
        return left_point, right_point
    
    # Add points and lines for each z-level
    for i, z in enumerate(z_levels):
        left_point, right_point = find_opposite_points_at_z_level(surface, z)
        
        if left_point is not None and right_point is not None:
            # Add points
            plotter.add_points(left_point, color=colors[i], point_size=10, render_points_as_spheres=True)
            plotter.add_points(right_point, color=colors[i], point_size=10, render_points_as_spheres=True)
            
            # Add line between points
            line_points = np.array([left_point, right_point])
            line = pv.lines_from_points(line_points)
            plotter.add_mesh(line, color=colors[i], line_width=3)
            
            # Add z-level label
            mid_point = (left_point + right_point) / 2
            plotter.add_point_labels([mid_point], [f'Z={z:.1f}'], font_size=12, text_color=colors[i])
    
    # Add coordinate axes
    plotter.add_axes()
    
    if output_file:
        plotter.screenshot(output_file, off_screen=True)
    
    return plotter


def plot_twist_angles_vs_time(t, twist_angles, z_levels, output_file=None, title="Twist Angle vs Time"):
    """
    Plot twist angles as a function of time for different z-levels.
    
    Args:
        t: List of time points
        twist_angles: List of twist angle arrays (one for each z-level)
        z_levels: List of z-coordinates
        output_file: Optional file path to save the plot
        title: Plot title
        
    Returns:
        fig: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for different z-levels
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_levels)))
    
    # Plot each z-level
    for i, (z, angles) in enumerate(zip(z_levels, twist_angles)):
        # Filter out None values
        valid_indices = [j for j, angle in enumerate(angles) if angle is not None]
        valid_times = [t[j] for j in valid_indices]
        valid_angles = [angles[j] for j in valid_indices]
        
        if valid_angles:
            ax.plot(valid_times, valid_angles, color=colors[i], 
                   label=f'Z={z:.1f}', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Twist Angle (degrees)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


def plot_twist_angles_vs_z(t, twist_angles, z_levels, time_indices=None, output_file=None, title="Twist Angle vs Z-coordinate"):
    """
    Plot twist angles as a function of z-coordinate at specific time points.
    
    Args:
        t: List of time points
        twist_angles: List of twist angle arrays (one for each z-level)
        z_levels: List of z-coordinates
        time_indices: List of time indices to plot. If None, plots all time points
        output_file: Optional file path to save the plot
        title: Plot title
        
    Returns:
        fig: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if time_indices is None:
        time_indices = range(len(t))
    
    # Define colors for different time points
    colors = plt.cm.plasma(np.linspace(0, 1, len(time_indices)))
    
    # Plot each time point
    for i, time_idx in enumerate(time_indices):
        if time_idx < len(t):
            # Extract angles for this time point
            angles_at_time = []
            valid_z_levels = []
            
            for j, z in enumerate(z_levels):
                if j < len(twist_angles) and time_idx < len(twist_angles[j]):
                    angle = twist_angles[j][time_idx]
                    if angle is not None:
                        angles_at_time.append(angle)
                        valid_z_levels.append(z)
            
            if angles_at_time:
                ax.plot(valid_z_levels, angles_at_time, color=colors[i], 
                       label=f't={t[time_idx]:.2f}s', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Z-coordinate', fontsize=12)
    ax.set_ylabel('Twist Angle (degrees)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig

def compute_final_twist_angle(twist_angles_list):
    """
    Compute the final twist angle as max - min at the last time step.
    Args:
        twist_angles_list (list): list of twist angles at each z-level, each of shape (n_timesteps,)
    Returns:
        float: final twist angle (max - min at last time step)
    """
    z_0_angle = twist_angles_list[0][-1]
    z_end_angle = twist_angles_list[-1][-1]

    return abs(z_end_angle - z_0_angle)


def calc_AV_plane_position(valve_ring_surface, check_normals=False):
    '''
    Calculate the position of the atrioventricular plane given the valve ring surface.
    Computes this by fitting a plane to the valve ring surface.
    
    Args:
        valve_ring_surface: PyVista PolyData object containing the valve ring surface
        check_normals: Whether to check the normals of the valve ring surface and flip them if necessary

    Returns:
        (plane, centroid, normal): Tuple containing the plane, centroid, and normal of the valve ring surface
    '''

    # Compute the best fitting plane to the valve ring surface
    plane, centroid, normal = pv.fit_plane_to_points(valve_ring_surface.points, return_meta=True)

    # Make sure that the normal is pointing in the same direction as the average normal of the valve ring surface
    if check_normals:
        valve_ring_surface.compute_normals(inplace=True)
        surface_normal = valve_ring_surface['Normals'].mean(0)
        print(f"Surface mean normal: {surface_normal}")
        print(f"Best fitting plane normal: {normal}")
        if np.dot(normal, surface_normal) < 0:
            print(f"Flipping normal of best fitting plane")
            normal = -normal
            plane['Normals'] = -plane['Normals']

    return plane, centroid, normal

def calc_AV_plane_displacement(start_timestep, end_timestep, step, timestep_size, results_folder, reference_surface, save_intermediate_data=False, intermediate_output_folder=None):
    """
    Calculate the atrioventricular plane displacement at each time step from the results of 
    an svFSI struct simulation, in which a model of the myocardium is simulated.

    Calculate the displacement in the following steps
    1) Compute the best fitting plane and centroid of the reference surface
    3) Sample the result.vtu file onto the reference surface
    4) Compute the best fitting plane and centroid of the resampled and warped surface
    5) Compute the displacement of the centroid of the resampled surface from the centroid, and project it onto the normal of the best fitting plane

    The units of displacement are whatever units used in .vtu files. For example,
    if units of length in the .vtu files are microns, then the displacement calculated
    here is in microns. 

    Args:
        start_timestep: The first svFSI result file to process
        
        end_timestep: The last svFSI result file to process
        
        step: The step in svFSI result files to process

        timestep_size: The size of the timestep in seconds
        
        results_folder: The absolute file path of the svFSI results folder 
        (usually something/something/16-procs/)
        
        reference_surface: The absolute file path of the .vtp file containing 
        the undeformed surface corresponding to the deformed surface of which 
        we want to compute the displacement.

        save_intermediate_data: Whether to save intermediate data (e.g., the best fitting plane, the centroid of the reference surface, etc.)

        intermediate_output_folder: The folder to save the intermediate data in. If None, the intermediate data is saved in the same folder as the reference_surface.

    Returns: (t, disps, ref_plane), a tuple of lists of length number of time steps. t 
    contains the time, and disps contains the displacement at that time. ref_plane
    contains the best fitting plane of the reference surface.
    """

    if save_intermediate_data:
        if intermediate_output_folder is None:
            assert intermediate_output_folder is not None, "If save_intermediate_data is True, intermediate_output_folder must be provided"

        # checking if the directory exists
        if not os.path.exists(intermediate_output_folder):
            # if the directory is not present then create it.
            os.makedirs(intermediate_output_folder)
    
    print('\n## Calculating atrioventricular plane displacement ##')

    # Load reference surface onto which we sample
    ref_surface = pv.read(f"{reference_surface}")

    # Compute the best fitting plane to the reference surface
    ref_plane, ref_centroid, ref_normal = calc_AV_plane_position(ref_surface, check_normals=True)

    # Save best fitting plane and centroid of reference surface
    if save_intermediate_data:
        ref_surface.save(os.path.join(intermediate_output_folder, 'surface_ref.vtp'))
        ref_plane.save(os.path.join(intermediate_output_folder, 'best_fitting_plane_ref.vtp'))
        pv.PolyData(ref_centroid).save(os.path.join(intermediate_output_folder, 'centroid_ref.vtp'))
    
    # Initialize arrays to store time and displacement
    t = []
    disps = []

    # Compute displacement of the centroid of the reference surface
    disp = np.dot(ref_centroid - ref_centroid, ref_normal)
    print(f"Iteration: {0}, Displacement: {disp}")
    t.append(0)
    disps.append(disp)

    # Loop through results files at each time > 0
    for k in range(start_timestep, end_timestep+1, step):
        # Load results VTU mesh
        result = pv.read(os.path.join(results_folder, f"result_{k:03d}.vtu"))

        # Sample result onto ref_surface
        resampled_surface = ref_surface.sample(result)

        # Warp resampled surface by displacement (needed for current configuration)
        warped_surface = resampled_surface.warp_by_vector('Displacement')

        # Compute plane, center and normal of the warped surface
        warped_plane, warped_centroid, warped_normal = calc_AV_plane_position(warped_surface)

        # Save warped surface and centroid
        if save_intermediate_data:
            warped_surface.save(os.path.join(intermediate_output_folder, f'surface_{k:03d}.vtp'))
            warped_plane.save(os.path.join(intermediate_output_folder, f'best_fitting_plane_{k:03d}.vtp'))
            pv.PolyData(warped_centroid).save(os.path.join(intermediate_output_folder, f'centroid_{k:03d}.vtp'))

        # Compute displacement of the centroid of the resampled surface
        disp = np.dot(warped_centroid - ref_centroid, ref_normal)
        print(f"Iteration: {k}, Displacement: {disp}")
        t.append(k*timestep_size)
        disps.append(disp)
    
    return (t, disps, ref_plane)

def calc_longitudinal_length(start_timestep, end_timestep, step, timestep_size, results_folder, reference_valve_surface, reference_endo_surface, save_intermediate_data=False, intermediate_output_folder=None):
    """
    Calculate the longitudinal length of the ventricle at each time step from the results of 
    an svFSI struct simulation, in which a model of the myocardium is simulated

    Calculate the length in the following steps
    1.a) Compute the best fitting plane and centroid of the reference valve surface
    1.b) Find point on the reference endo surface further from the reference valve surface (in the normal direction). This is the reference apex point.
    2) Sample the result.vtu file onto the reference valve surface and apex surface
    3.a) Compute the best fitting plane and centroid of the resampled and warped valve surface
    3.b) Find the position of the warped apex point.
    4) Compute the distance between the centroid of the warped valve and apex, projected onto the normal of the best fitting plane of the reference valve surface

    The units of length are whatever units used in .vtu files. For example,
    if units of length in the .vtu files are microns, then the length calculated
    here is in microns. 

    Args:
        start_timestep: The first svFSI result file to process
        
        end_timestep: The last svFSI result file to process
        
        step: The step in svFSI result files to process

        timestep_size: The size of the timestep in seconds
        
        results_folder: The absolute file path of the svFSI results folder 
        (usually something/something/16-procs/)
        
        reference_surface_valve: The absolute file path of the .vtp file containing 
        the undeformed surface corresponding to the deformed surface with which 
        we want to compute the longitudinal length.

        reference_endo_surface: The absolute file path of the .vtp file containing
        the endocardial surface corresponding to the deformed surface with which
        we want to compute the longitudinal length.

        save_intermediate_data: Whether to save intermediate data (e.g., the best fitting plane, the centroid of the reference surface, etc.)

        intermediate_output_folder: The folder to save the intermediate data in. If None, the intermediate data is saved in the same folder as the reference_surface.

    Returns: (t, lengths, ref_plane), a tuple of lists of length number of time steps. t 
    contains the time, and lengths contains the length at that time. ref_plane
    contains the best fitting plane of the reference surface.
    """

    if save_intermediate_data:
        if intermediate_output_folder is None:
            assert intermediate_output_folder is not None, "If save_intermediate_data is True, intermediate_output_folder must be provided"

        # checking if the directory exists
        if not os.path.exists(intermediate_output_folder):
            # if the directory is not present then create it.
            os.makedirs(intermediate_output_folder)
            
    print('\n## Calculating longitudinal length ##')
    
    # Load reference valve surface onto which we sample
    ref_valve_surface = pv.read(f"{reference_valve_surface}")

    # Compute the best fitting plane to the reference valve surface
    ref_valve_plane, ref_valve_centroid, ref_valve_normal = calc_AV_plane_position(ref_valve_surface, check_normals=True)

    # Save best fitting plane and centroid of reference valve surface
    if save_intermediate_data:
        ref_valve_surface.save(os.path.join(intermediate_output_folder, 'valve_surface_ref.vtp'))
        ref_valve_plane.save(os.path.join(intermediate_output_folder, 'valve_best_fitting_plane_ref.vtp'))
        pv.PolyData(ref_valve_centroid).save(os.path.join(intermediate_output_folder, 'valve_centroid_ref.vtp'))

    # Load reference endo surface 
    ref_endo_surface = pv.read(f"{reference_endo_surface}")

    # Compute the apex point of the reference endo surface
    points = ref_endo_surface.points
    projections = np.abs(np.dot(points - ref_valve_centroid, ref_valve_normal))
    max_projection_index = np.argmax(projections)
    ref_apex_point = points[max_projection_index]
    ref_apex_point_polydata = pv.PolyData(ref_apex_point) 

    # Save apex point of reference endo surface
    if save_intermediate_data:
        ref_apex_point_polydata.save(os.path.join(intermediate_output_folder, 'apex_point_ref.vtp'))

    # Initialize arrays to store time and length
    t = []
    lengths = []

    # Compute distance between centroids of reference valve and apex surfaces
    length = np.abs(np.dot(ref_apex_point - ref_valve_centroid, ref_valve_normal))
    print(f"Iteration: {0}, Length: {length}")
    t.append(0)
    lengths.append(length)

    # Loop through results files at each time > 0
    for k in range(start_timestep, end_timestep+1, step):
        # Load results VTU mesh
        result = pv.read(os.path.join(results_folder, f"result_{k:03d}.vtu"))

        # Sample result onto ref_valve_surface
        resampled_valve_surface = ref_valve_surface.sample(result)

        # Warp resampled valve surface by displacement (needed for current configuration)
        warped_valve_surface = resampled_valve_surface.warp_by_vector('Displacement')

        # Compute plane, center and normal of the warped valve surface
        warped_valve_plane, warped_valve_centroid, warped_valve_normal = calc_AV_plane_position(warped_valve_surface)

        # Save warped valve surface and centroid
        if save_intermediate_data:
            warped_valve_surface.save(os.path.join(intermediate_output_folder, f'valve_surface_{k:03d}.vtp'))
            warped_valve_plane.save(os.path.join(intermediate_output_folder, f'valve_best_fitting_plane_{k:03d}.vtp'))
            pv.PolyData(warped_valve_centroid).save(os.path.join(intermediate_output_folder, f'valve_centroid_{k:03d}.vtp'))

        # Sample result onto ref_apex_point
        resampled_apex_point_polydata = ref_apex_point_polydata.sample(result)

        # Warp resampled apex point by displacement (needed for current configuration)
        warped_apex_point_polydata = resampled_apex_point_polydata.warp_by_vector('Displacement')

        # Save warped apex surface and centroid
        if save_intermediate_data:
            warped_apex_point_polydata.save(os.path.join(intermediate_output_folder, f'apex_point_{k:03d}.vtp'))

        # Compute distance between centroids of warped valve and apex surfaces
        assert warped_apex_point_polydata.n_points == 1, "Warped apex point should have only one point"
        warped_apex_point = warped_apex_point_polydata.points[0]
        length = np.abs(np.dot(warped_apex_point - warped_valve_centroid, ref_valve_normal))
        print(f"Iteration: {k}, Length: {length}")
        t.append(k*timestep_size)
        lengths.append(length)

    return (t, lengths)

def calculate_wall_thickness(epicardial_mesh, endocardial_mesh, plot_distribution=False):
    """
    Calculate the distance from each node on the epicardial surface to the nearest
    node on the endocardial surface.

    Parameters:
    epicardial_mesh (pyvista.PolyData): The epicardial surface mesh
    endocardial_mesh (pyvista.PolyData): The endocardial surface mesh
    plot_distribution (bool): Whether to plot the distribution of shortest distances

    Returns:
    distances (np.ndarray): Array of shortest distances for each epicardial node
    indices (np.ndarray): Array of indices of the closest endocardial node for each epicardial node
    """

    def find_bimodal_means(thickness_data):
        """
        Fit a Gaussian Mixture Model to bimodal data and return the means of each mode.

        Parameters:
        thickness_data (np.ndarray): Array of thickness values (e.g., distances between surfaces)

        Returns:
        means (np.ndarray): Means of the two modes
        mode_assignments (np.ndarray): Array assigning each data point to a mode (0 or 1)
        """
        # Reshape the data for GMM (needs 2D input)
        thickness_data = thickness_data.reshape(-1, 1)

        # Fit a Gaussian Mixture Model with 2 components (bimodal)
        gmm = GaussianMixture(n_components=2)
        gmm.fit(thickness_data)

        # The means of the two modes
        means = gmm.means_.flatten()

        # Assign each data point to a mode (0 or 1)
        mode_assignments = gmm.predict(thickness_data)

        return means, mode_assignments

    # Extract the (x, y, z) coordinates of the nodes from both meshes
    epicardial_nodes = np.array(epicardial_mesh.points)
    endocardial_nodes = np.array(endocardial_mesh.points)

    # Build a KD-tree for the endocardial surface nodes for efficient distance querying
    endocardial_tree = cKDTree(endocardial_nodes)

    # For each epicardial node, find the closest endocardial node
    thickness, indices = endocardial_tree.query(epicardial_nodes)

    means, mode_assignments = find_bimodal_means(thickness)

    # Plot the data and the fitted modes
    if plot_distribution:
        plt.hist(thickness, bins=30, alpha=0.6, color='g', density=True, label='Histogram of Data')
        plt.title("Bimodal Thickness Data with Gaussian Mixture Model")
        plt.axvline(means[0], color='r', linestyle='--', label=f'Mean of Mode 1: {means[0]:.2f}')
        plt.axvline(means[1], color='b', linestyle='--', label=f'Mean of Mode 2: {means[1]:.2f}')
        plt.legend()
        plt.show()

    return np.min(means)

def calc_wall_thickening(start_timestep, end_timestep, step, timestep_size, results_folder, epicardial_surface, endocardial_surface):
    '''
    Calculate the wall thickening at each time step from the results of an svFSI struct simulation, in which a model of the myocardium is inflated.

    ARGS:
        start_timestep: The first svFSI result file to process
        end_timestep: The last svFSI result file to process
        step: The step in svFSI result files to process
        timestep_size: The size of the timestep in seconds
        results_folder: The absolute file path of the svFSI results folder (usually something/something/16-procs/)
        epicardial_surface: The absolute file path of the .vtp file containing the epicardial surface
        endocardial_surface: The absolute file path of the .vtp file containing the endocardial surface

    RETURNS:
        (t, wall_thickness): A tuple of lists of length number of time steps. t contains the time, and wall_thickness contains the wall thickness at that time.
    '''


    print('\n## Calculating wall thickening ##')

    # Load epicardial and endocardial surfaces
    epicardial_mesh = pv.read(epicardial_surface)
    endocardial_mesh = pv.read(endocardial_surface)

    # Initialize arrays to store time and wall thickening
    t = []
    wall_thickness = []

    # Compute the initial wall thickness
    initial_wall_thickness = calculate_wall_thickness(epicardial_mesh, endocardial_mesh)

    # Add initial wall thickness to arrays
    t.append(0)
    wall_thickness.append(initial_wall_thickness)

    # Loop through results files at each time > 0
    for k in range(start_timestep, end_timestep+1, step):
        # Load results VTU mesh
        result = pv.read(os.path.join(results_folder, f"result_{k:03d}.vtu"))

        # Sample result onto epicardial and endocardial surfaces
        resampled_epicardial = epicardial_mesh.sample(result)
        resampled_endocardial = endocardial_mesh.sample(result)

        # Warp resampled surfaces by displacement (needed for current configuration)
        warped_epicardial = resampled_epicardial.warp_by_vector('Displacement')
        warped_endocardial = resampled_endocardial.warp_by_vector('Displacement')

        # Compute the wall thickness
        thickness = calculate_wall_thickness(warped_epicardial, warped_endocardial)

        # Add time and wall thickness to arrays
        t.append(k*timestep_size)
        wall_thickness.append(thickness)

        print(f"Iteration: {k}, Wall Thickness: {thickness}")
    
    return (t, wall_thickness)

def calc_myocardial_volume(start_timestep, end_timestep, step, timestep_size, results_folder):
    '''
    Calculate the myocardial volume at each time step from the results of an svFSI (u)struct simulation.
    '''
    print('\n## Calculating myocardial volume ##')

    # Initialize arrays to store time and volume
    t = []
    vol = []

    # Loop through results files at each time > 0
    for k in range(start_timestep, end_timestep+1, step):
        # Load results VTU mesh
        result = pv.read(os.path.join(results_folder, f"result_{k:03d}.vtu"))

        # Extract surface of result (not strictly necessary, but avoids negative volumes when computing volume from VTU directly)
        result_surface = result.extract_surface()

        # Add initial volume to arrays if this is the first iteration
        if k == start_timestep:
            volume = result_surface.volume
            print(f"Iteration: {0}, Volume: {volume}")
            t.append(0)
            vol.append(volume)

        # Warp result_surface by displacement
        warped_result_surface = result_surface.warp_by_vector('Displacement')

        # Compute volume of result
        volume = warped_result_surface.volume
        print(f"Iteration: {k}, Volume: {volume}")
        t.append(k*timestep_size)
        vol.append(volume)

    return (t, vol)

def get_timestep(result_file):
    '''
    Extracts the timestep of a results_###.vtu file
    '''
    # If file path is provided, get the file name
    file_name = os.path.basename(result_file)

    # Get the ###.vtu part
    s = file_name.split('_')[-1]

    # Get the ### part
    s = s.split('.')[0]

    # Return as integer
    return int(s)

def get_start_end_step(results_folder):
    """
    Automatically determine the start timestep, end timestep, and step size based on all
    svFSI results file in results_folder.

    Args:
        results_folder: A string of absolute file path of folder containing results of 
        svFSI simulation. This usually ends with 16-procs/ or other 
        number.
    
    Returns:
        (start_timestep, end_timestep, step): A tuple of 3 integers, giving the first 
        timestep of results to process, the last timestep, and the step size. 

    """

    # Get list of all .vtu files in results_folder sorted by time step
    list_of_files = sorted( filter( os.path.isfile,
                            glob.glob(os.path.join(results_folder, '*.vtu')) ), key = get_timestep)

    # Get start time from the first result file (list_of_files[0])
    start_file_name = os.path.basename(list_of_files[0])
    start_timestep = int("".join([i for i in start_file_name if i.isdigit()]))
    print('Start timestep:', start_timestep)

    # Get end time from the last result file (list_of_files[-1])
    end_file_name = os.path.basename(list_of_files[-1])
    end_timestep = int("".join([i for i in end_file_name if i.isdigit()]))
    print('End timestep:', end_timestep)

    # Get step size by looking at second time step
    start_plus_one_file_name = os.path.basename(list_of_files[1])
    start_time_plus_one = int("".join([i for i in start_plus_one_file_name if i.isdigit()]))
    step = start_time_plus_one - start_timestep
    print('Step:', step)

    return (start_timestep, end_timestep, step)

def read_3D_pressure(pressure_dat_file, t):
    """
    Calculate the ventricular lumen pressure at each time step from the results of 
    an svFSI struct simulation, in which a model of the myocardium is inflated.

    Calculates pressure on the endo surface by reading the input file and 
    pressure load file. The units of pressure are whatever units are used in the
    pressure load file (usually Pa)

    Args:
        pressure_dat_file: The file containing the pressure load information for
        the simulation whose results we are now processing.

        t: List of times at which we want to output the 
        pressure.

    Returns:
        pressure: Pressure evaluated at list of times in t
    """
    print('\n## Reading pressure ##')
        
    # Initialize arrays to store time and pressure
    pressure = []
    time = []

    # Read pressure load file (skip first line)
    with open(pressure_dat_file, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            time.append(float(line.split(',')[0]))
            pressure.append(float(line.split(',')[1]))
    
    # Interpolate pressure at time steps in t
    pressure = np.interp(t, time, pressure)

    return pressure

def get_timestep_size(svfsi_inp_file):
    '''
    Extracts the timestep size from the svFSI input file

    Args:
        svfsi_inp_file: The file path to the svFSI input file

    Returns:
        timestep_size: The size of the timestep in the input file
    '''

    # Read the input file
    with open(svfsi_inp_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Time step size:' in line:
                # Split at '#' to remove the comment, then split at ':' to get the number
                timestep_size = float(line.split('#')[0].split(':')[-1])
                break
    
    return timestep_size

def get_n_timesteps(svfsi_inp_file):
    '''
    Extracts the number of timesteps from the svFSI input file

    Args:
        svfsi_inp_file: The file path to the svFSI input file

    Returns:
        num_timesteps: The number of timesteps in the input file
    '''

    # Read the input file
    with open(svfsi_inp_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Number of time steps:' in line:
                # Split at '#' to remove the comment, then split at ':' to get the number
                num_timesteps = int(line.split('#')[0].split(':')[-1])
                break
    
    return num_timesteps

def find_closest_times(t, t_ref):
    '''
    Find the indices of the closest times in t to the times in t_ref

    Args:
        t: List of times
        t_ref: List of reference times

    Returns:
        indices: List of indices of closest times in t to t_ref
        times: List of closest times in t to t_ref
    '''

    indices = []
    for t_ in t_ref:
        indices.append(np.argmin(np.abs(t - t_)))

    return indices, [t[i] for i in indices]


    
def read_image_volume():
    '''
    Read image volume from the patient_metrics.py file

    Returns: (RR_img, V_LV_img, V_RV_img, V_LA_img, V_RA_img), a tuple of lists of length number of time steps. RR_img
    contains the RR interval percentage, V_LV_img contains the left ventricular volume at that time, and
    V_RV_img contains the right ventricular volume at that time.
    '''
    # Read image volume data
    from data.patient_metrics import V_LA_img, V_RA_img, V_LV_img, V_RV_img, RR_img 

    return RR_img, V_LV_img, V_RV_img, V_LA_img, V_RA_img


def calc_image_AV_plane_displacement(morphed_surfaces_folder, results_dict, RR_ref, save_intermediate_data=False, intermediate_output_folder=None):
    '''
    Calculate AV plane displacement from morphed mesh surfaces in image data folder

    Args:
        morphed_surfaces_folder: The folder containing the morphed surface data. This should contain the RR%_ref_with_disp.vtk files.
        results_dict: The dictionary containing the simulation results
        RR_ref: The reference RR interval percentage
        save_intermediate_data: Whether to save intermediate data (e.g., the best fitting plane, the centroid of the reference surface, etc.)
        intermediate_output_folder: The folder to save the intermediate data in. If None, the intermediate data is saved in the same folder as the reference_surface.

    Returns: (RR, disps, ref_plane), a tuple of lists of length number of time steps. RR
    contains the RR interval percentage, and disps contains the displacement at that time. ref_plane
    contains the best fitting plane of the reference surface.
    '''

    if save_intermediate_data:
        if intermediate_output_folder is None:
            assert intermediate_output_folder is not None, "If save_intermediate_data is True, intermediate_output_folder must be provided"
        
        # checking if the directory exists
        if not os.path.exists(intermediate_output_folder):
            # if the directory is not present then create it.
            os.makedirs(intermediate_output_folder)

    print('\n## Calculating atrioventricular plane displacement from image data ##')

    # Initialize array to store displacements
    disps = []

    # Get the reference surface
    reference_surface = pv.read(os.path.join(morphed_surfaces_folder, f'RR{RR_ref}_ref_with_disp.vtk'))

    # Convert to Polydata if necessary
    if not isinstance(reference_surface, pv.PolyData):
        reference_surface = reference_surface.extract_surface()

    # Compute the best fitting plane to the reference surface
    ref_plane, ref_centroid, ref_normal = calc_AV_plane_position(reference_surface, check_normals=True)

    # Save best fitting plane and centroid of reference surface
    if save_intermediate_data:
        reference_surface.save(os.path.join(intermediate_output_folder, 'surface_ref.vtp'))
        ref_plane.save(os.path.join(intermediate_output_folder, 'best_fitting_plane_ref.vtp'))
        pv.PolyData(ref_centroid).save(os.path.join(intermediate_output_folder, 'centroid_ref.vtp'))

    # Loop through morphed meshes
    for k in results_dict['RR%_img']:
        # Load morphed mesh
        morphed_mesh = pv.read(os.path.join(morphed_surfaces_folder, f"RR{k}_ref_with_disp.vtk"))

        # Convert to Polydata if necessary
        if not isinstance(morphed_mesh, pv.PolyData):
            morphed_mesh = morphed_mesh.extract_surface()

        # Warp by displacement to get current configuration
        morphed_mesh = morphed_mesh.warp_by_vector('Displacement')

        # Compute the best fitting plane to the morphed surface
        plane, centroid, normal = calc_AV_plane_position(morphed_mesh, check_normals=False)

        # Save best fitting plane and centroid of morphed surface
        if save_intermediate_data:
            morphed_mesh.save(os.path.join(intermediate_output_folder, f'surface_{k}.vtp'))
            plane.save(os.path.join(intermediate_output_folder, f'best_fitting_plane_{k}.vtp'))
            pv.PolyData(centroid).save(os.path.join(intermediate_output_folder, f'centroid_{k}.vtp'))

        # Compute displacement of the centroid of the morphed surface
        disp = np.dot(centroid - ref_centroid, ref_normal)
        print(f"Iteration: {k}, Displacement: {disp}")
        disps.append(disp)

    return (results_dict['RR%_img'], disps, ref_plane)


def calc_image_longitudinal_length(morphed_valve_surfaces_folder, morphed_endo_surfaces_folder, results_dict, RR_ref, save_intermediate_data=False, intermediate_output_folder=None):
    '''
    Calculate the longitudinal length from morphed mesh surfaces in image data folder

    Args:
        morphed_valve_surfaces_folder: The folder containing the morphed valve surface data. This should contain the RR%_ref_with_disp.vtk files.  
        morphed_endo_surfaces_folder: The folder containing the morphed endocardial surface data. This should contain the RR%_ref_with_disp.vtk files.
        results_dict: The dictionary containing the simulation results
        RR_ref: The reference RR interval percentage
        intermediate_output_folder: The folder to save the intermediate data in. If None, the intermediate data is saved in the same folder as the reference_surface.

    Returns: (RR, lengths, ref_valve_plane), a tuple of lists of length number of time steps. RR
    contains the RR interval percentage, lengths contains the length at that time, and ref_valve_plane
    contains the best fitting plane of the reference valve surface.
    '''

    if save_intermediate_data:
        if intermediate_output_folder is None:
            assert intermediate_output_folder is not None, "If save_intermediate_data is True, intermediate_output_folder must be provided"

        # checking if the directory exists
        if not os.path.exists(intermediate_output_folder):
            # if the directory is not present then create it.
            os.makedirs(intermediate_output_folder)

    print('\n## Calculating longitudinal length from image data ##')

    # Initialize array to store lengths
    lengths = []

    # Load the reference valve surface
    reference_valve_surface = pv.read(os.path.join(morphed_valve_surfaces_folder, f'RR{RR_ref}_ref_with_disp.vtk'))
    # Convert to Polydata if necessary
    if not isinstance(reference_valve_surface, pv.PolyData):
        reference_valve_surface = reference_valve_surface.extract_surface()
    # Compute the best fitting plane to the reference valve surface
    ref_valve_plane, ref_valve_centroid, ref_valve_normal = calc_AV_plane_position(reference_valve_surface, check_normals=True)
    # Save best fitting plane and centroid of reference surface
    if save_intermediate_data:
        reference_valve_surface.save(os.path.join(intermediate_output_folder, 'valve_surface_ref.vtp'))
        ref_valve_plane.save(os.path.join(intermediate_output_folder, 'valve_best_fitting_plane_ref.vtp'))
        pv.PolyData(ref_valve_centroid).save(os.path.join(intermediate_output_folder, 'valve_centroid_ref.vtp'))

    # Load the reference endo surface
    reference_endo_surface = pv.read(os.path.join(morphed_endo_surfaces_folder, f'RR{RR_ref}_ref_with_disp.vtk'))
    # Convert to Polydata if necessary
    if not isinstance(reference_endo_surface, pv.PolyData):
        reference_endo_surface = reference_endo_surface.extract_surface()
    # Compute the apex point of the reference endo surface
    points = reference_endo_surface.points
    projections = np.abs(np.dot(points - ref_valve_centroid, ref_valve_normal))
    max_projection_index = np.argmax(projections)
    ref_apex_point = points[max_projection_index]
    ref_apex_point_polydata = pv.PolyData(ref_apex_point)
    # Save apex point of reference endo surface
    if save_intermediate_data:
        ref_apex_point_polydata.save(os.path.join(intermediate_output_folder, 'apex_point_ref.vtp'))

    # Loop through morphed meshes
    for k in results_dict['RR%_img']:
        # Load morphed valve surface
        morphed_valve_surface = pv.read(os.path.join(morphed_valve_surfaces_folder, f"RR{k}_ref_with_disp.vtk"))
        # Convert to Polydata if necessary
        if not isinstance(morphed_valve_surface, pv.PolyData):
            morphed_valve_surface = morphed_valve_surface.extract_surface()
        # Warp by displacement to get current configuration
        morphed_valve_surface = morphed_valve_surface.warp_by_vector('Displacement')
        # Compute the best fitting plane to the morphed valve surface
        morphed_valve_plane, morphed_valve_centroid, morphed_valve_normal = calc_AV_plane_position(morphed_valve_surface)
        # Save best fitting plane and centroid of morphed valve surface
        if save_intermediate_data:
            morphed_valve_surface.save(os.path.join(intermediate_output_folder, f'valve_surface_{k}.vtp'))
            morphed_valve_plane.save(os.path.join(intermediate_output_folder, f'valve_best_fitting_plane_{k}.vtp'))
            pv.PolyData(morphed_valve_centroid).save(os.path.join(intermediate_output_folder, f'valve_centroid_{k}.vtp'))

        # Load morphed endo surface
        morphed_endo_surface = pv.read(os.path.join(morphed_endo_surfaces_folder, f"RR{k}_ref_with_disp.vtk"))
        # Convert to Polydata if necessary
        if not isinstance(morphed_endo_surface, pv.PolyData):
            morphed_endo_surface = morphed_endo_surface.extract_surface()
        # Warp by displacement to get current configuration
        morphed_endo_surface = morphed_endo_surface.warp_by_vector('Displacement')
        # Get the apex point of the morphed endo surface
        points = morphed_endo_surface.points
        morphed_apex_point = points[max_projection_index]
        morphed_apex_point_polydata = pv.PolyData(morphed_apex_point)
        # Save apex point of morphed endo surface
        if save_intermediate_data:
            morphed_apex_point_polydata.save(os.path.join(intermediate_output_folder, f'apex_point_{k}.vtp'))

        # Compute distance between centroids of morphed valve and apex surfaces
        length = np.abs(np.dot(morphed_apex_point - ref_valve_centroid, ref_valve_normal))
        print(f"Iteration: {k}, Length: {length}")
        lengths.append(length)

    return (results_dict['RR%_img'], lengths)

def calc_image_wall_thickening(epicardial_surface_folder, endocardial_surface_folder, results_dict):
    '''
    Calculate wall thickening from morphed mesh surfaces in image data folder

    Args:
        epicardial_surface_folder: The folder containing the epicardial surface data
        endocardial_surface_folder: The folder containing the endocardial surface data
        results_dict: The dictionary containing the simulation results

    Returns: (RR, wall_thickness), a tuple of lists of length number of time steps. RR
    contains the RR interval percentage, and wall_thickness contains the wall thickness at that time.
    '''

    print('\n## Calculating wall thickening from image data ##')

    # Initialize array to store wall thickness
    wall_thickness = []

    # Loop through morphed meshes
    for k in results_dict['RR%_img']:
        # Load epicardial and endocardial surfaces
        epicardial_mesh = pv.read(os.path.join(epicardial_surface_folder, f"RR{k}_ref_with_disp.vtk"))
        endocardial_mesh = pv.read(os.path.join(endocardial_surface_folder, f"RR{k}_ref_with_disp.vtk"))

        # Warp by displacement to get current configuration
        epicardial_mesh = epicardial_mesh.warp_by_vector('Displacement')
        endocardial_mesh = endocardial_mesh.warp_by_vector('Displacement')

        # Compute the wall thickness
        thickness = calculate_wall_thickness(epicardial_mesh, endocardial_mesh)

        # Add wall thickness to array
        print(f"Iteration: {k}, Wall Thickness: {thickness}")
        wall_thickness.append(thickness)

    return (results_dict['RR%_img'], wall_thickness)

def calc_image_myocardial_volume(exterior_surface_folder, results_dict):
    '''
    Calculate the myocardial volume from morphed mesh surfaces in image data folder

    Args:
        exterior_surface_folder: The folder containing the complete exterior surface mesh
        results_dict: The dictionary containing the simulation results
    '''
    print('\n## Calculating myocardial volume from image data ##')

    # Initialize array to store volumes
    volumes = []

    # Loop through morphed meshes
    for k in results_dict['RR%_img']:
        # Load morphed mesh
        morphed_mesh = pv.read(os.path.join(exterior_surface_folder, f"RR{k}_ref_with_disp.vtk"))

        # Extract surface to convert to polydata
        morphed_mesh = morphed_mesh.extract_surface()

        # Warp by displacement to get current configuration
        morphed_mesh = morphed_mesh.warp_by_vector('Displacement')

        # Compute volume of morphed mesh
        volume = morphed_mesh.volume

        # Add volume to array
        print(f"Iteration: {k}, Volume: {volume}")
        volumes.append(volume)

    return (results_dict['RR%_img'], volumes)

def compute_sim_to_image_surface_error(sim_results_folder, sim_file_pattern, sim_reference_surface_path, image_morphed_surface_folder, image_morphed_file_pattern, rr_timepoints, output_folder):
    """
    Compute the shortest distance from each node on the simulation-deformed surface to the nearest node
    on the image-deformed surface for each RR time point.

    Args:
        sim_results_folder (str): Path to the folder containing simulation endocardial surfaces at RR% time points.
        sim_file_pattern (str): Pattern for simulation files, with {rr} as placeholder.
        sim_reference_surface_path (str): Path to the surface in the reference mesh.
        image_morphed_surface_folder (str): Path to the folder containing image-derived morphed surfaces at RR% time points.
        image_morphed_file_pattern (str): Pattern for morphed files, with {rr} as placeholder.
        rr_timepoints (list): List of RR time points (e.g., [0, 10, 20, ..., 100]).
        output_folder (str): Path to the folder to save the output. If None, the output is not saved.
    Returns:
        dict: Mapping RR time point to a tuple of (distances, indices), where distances is an array of shortest
              distances for each node on the simulation surface, and indices are the indices of the closest
              nodes on the image surface.
    """

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize dictionary to store distance from simulation to image-derived morphed surface
    sim_to_image_surface_error = {}


    # Load the reference surface
    sim_reference_surface = pv.read(sim_reference_surface_path)

    for rr in rr_timepoints:
        print(f"Processing RR{rr}...")
        # Load the simulation result for the current RR% time point
        sim_file = os.path.join(sim_results_folder, sim_file_pattern.format(rr=rr))
        print(f"Sim file: {sim_file}")
        sim_mesh = pv.read(sim_file)

        # Sample the displacement field from the simulation result to the reference endocardial surface
        sim_resampled_surface = sim_reference_surface.sample(sim_mesh)

        # Warp the reference surface by the displacement field
        sim_morphed_surface = sim_resampled_surface.warp_by_vector('Displacement')

        # Load the image-morphed surface and warp by displacement
        image_morphed_file = os.path.join(image_morphed_surface_folder, image_morphed_file_pattern.format(rr=rr))
        print(f"Morphed file: {image_morphed_file}")
        image_morphed_surface = pv.read(image_morphed_file)
        image_morphed_surface = image_morphed_surface.warp_by_vector('Displacement')

        # Compute the distance from each node on the simulation-morphed surface to the nearest node on the image-morphed surface
        distances, indices = cKDTree(image_morphed_surface.points).query(sim_morphed_surface.points)

        # Add the distances to the simulation-morphed surface mesh and save
        sim_morphed_surface['Distance_to_image_morphed_surface'] = distances
        sim_morphed_surface.save(os.path.join(output_folder, f'sim_morphed_surface_RR{rr}.vtp'))

        # Save the image-morphed surface too for debugging
        image_morphed_surface.save(os.path.join(output_folder, f'image_morphed_surface_RR{rr}.vtk'))

        # Save the surface distances
        sim_to_image_surface_error[rr] = distances

    # Plot the surface error as a violin plot
    plt.figure()
    plt.violinplot(list(sim_to_image_surface_error.values()), showmeans=True, showmedians=True)
    plt.xticks(range(1, len(rr_timepoints)+1), rr_timepoints)
    plt.xlabel('RR%')
    plt.ylabel('Min Nodal Separation (cm)')
    plt.title('Error between simulation and image-derived morphed surfaces')
    plt.savefig(os.path.join(output_folder, f'sim_to_image_surface_error_violinplot.png'))
    plt.close()

    return sim_to_image_surface_error

def create_lv_centered_coordinate_system(lv_endo, rv_endo, long_axis):
    '''
    Create a coordinate system centered on the LV.

    Args:
        lv_endo (pv.PolyData): The endocardial surface of the LV.
        rv_endo (pv.PolyData): The endocardial surface of the RV.
        long_axis (pv.PolyData): The long axis of the heart.

    Returns:
        np.array: The coordinate system.
    '''

    # Get the long axis vector
    long_axis_vector = np.array(long_axis.points[1]) - np.array(long_axis.points[0])
    long_axis_vector = long_axis_vector / np.linalg.norm(long_axis_vector)

    # Calculate centroids by averaging all vertex positions
    lv_centroid = np.mean(lv_endo.points, axis=0)
    rv_centroid = np.mean(rv_endo.points, axis=0)

    # Compute septal-to-lateral vector as the vector from the rv_endo_centroid to the lv_endo_centroid, orthogonal to the long axis vector
    septal_to_lateral_vector = lv_centroid - rv_centroid
    septal_to_lateral_vector = septal_to_lateral_vector - np.dot(septal_to_lateral_vector, long_axis_vector) * long_axis_vector
    septal_to_lateral_vector = septal_to_lateral_vector / np.linalg.norm(septal_to_lateral_vector)

    # Compute the posterior-to-anterior vector as the cross product of the septal-to-lateral vector and the long axis vector
    posterior_to_anterior_vector = np.cross( septal_to_lateral_vector, long_axis_vector)
    posterior_to_anterior_vector = posterior_to_anterior_vector / np.linalg.norm(posterior_to_anterior_vector)

    # Compute the basis vectors of the coordinate system
    basis_vectors = np.array([septal_to_lateral_vector, posterior_to_anterior_vector, long_axis_vector])

    # Use lv_centroid as the origin of the coordinate system
    origin = lv_centroid

    return origin, basis_vectors

def create_mesh_regions_by_threshold(mesh, params):
    '''
    Split the mesh into regions based on the values of point scalar data

    Args:
        mesh (pv.PolyData): The mesh to split into regions.
        params (dict): Dictionary containing the parameters.
            'Regions' (list): List of dictionaries containing the regions.
                'Name' (str): Name of the region.
                'Threshold' (dict): Dictionary containing the point scalar data name and min/max value to include the region.
                    'Point_scalar_data_name' (str): Name of the point scalar data.
                    'Threshold' (list): Min/max value of the point scalar data to include the region.
    '''


    # Convert point data to cell data
    mesh = mesh.point_data_to_cell_data()

    # Initialize dictionary to store the regions
    mesh_regions = {}

    # Threshold the Laplace mesh by the regions 
    for region in params['Regions']:
        mesh_region = mesh.copy()
        print(f"Creating region {region['Name']}...")
        for point_scalar_data_name, threshold_values in region['Threshold'].items():    
            print(f"Thresholding mesh by {point_scalar_data_name} between {threshold_values[0]} and {threshold_values[1]}")
            mesh_region = mesh_region.threshold(scalars=point_scalar_data_name, value=threshold_values)
        mesh_regions[region['Name']] = mesh_region


    return mesh_regions

def split_mesh_into_regions(laplace_mesh_file, long_axis_mesh_file):
    '''
    Split the mesh into regions based on the values of point scalar data. This function
    creates the following regions:  
    - LV_endo
    - LV_mid
    - LV_epi
    - RV_endo
    - RV_mid
    - RV_epi
    - septum

    Args:
        laplace_mesh_file (str): Path to the volume mesh with Laplace field defined.
        long_axis_mesh_file (str): Path to the volume mesh with long axis coordinates defined.

    Returns:
        dict: Mapping region name to mesh.
    '''
    # Load the volume meshes with Laplace field and long axis coordinates defined
    laplace_mesh = pv.read(laplace_mesh_file)
    long_axis_mesh = pv.read(long_axis_mesh_file)

    # Create a mesh with combined data
    combined_mesh = laplace_mesh.copy()
    combined_mesh.point_data['long_axis_coordinates'] = long_axis_mesh.point_data['long_axis_coordinates']

    # Create endo, mid, and epi regions for LV and RV
    params = {
        'Regions': [
                    {'Name': 'LV_endo', 'Threshold': {'Phi_BiV_L2R': [0.75, 1], 'Phi_BiV_EPI': [0, 0.33]}}, 
                    {'Name': 'LV_mid', 'Threshold': {'Phi_BiV_L2R': [0.75, 1], 'Phi_BiV_EPI': [0.33, 0.66]}}, 
                    {'Name': 'LV_epi', 'Threshold': {'Phi_BiV_L2R': [0.75, 1], 'Phi_BiV_EPI': [0.66, 1.0]}},
                    {'Name': 'RV_endo', 'Threshold': {'Phi_BiV_L2R': [-1, -0.75], 'Phi_BiV_EPI': [0, 0.33]}}, 
                    {'Name': 'RV_mid', 'Threshold': {'Phi_BiV_L2R': [-1, -0.75], 'Phi_BiV_EPI': [0.33, 0.66]}}, 
                    {'Name': 'RV_epi', 'Threshold': {'Phi_BiV_L2R': [-1, -0.75], 'Phi_BiV_EPI': [0.66, 1.0]}},
                    {'Name': 'septum', 'Threshold': {'Phi_BiV_L2R': [-0.75, 0.75]}}
                    ]
    }
    mesh_regions = create_mesh_regions_by_threshold(combined_mesh, params)

    return mesh_regions

def extract_result_data_by_region(sim_results_folder, sim_file_pattern, rr_timepoints, array_names, mesh_regions):
    """
    Extract data from simulation results meshes by region for each RR time point.

    Args:
        sim_results_folder (str): Path to the folder containing simulation results.
        sim_file_pattern (str): Pattern for simulation files, with {rr} as placeholder.
        rr_timepoints (list): List of RR time points (e.g., [0, 10, 20, ..., 100]).
        array_names (list): List of array names to plot.
        mesh_regions (dict): Dictionary containing the mesh regions, with the region name as the key and the mesh as the value.

    Returns:
        dict: Dictionary, region name -> array name -> list of data at each RR time point.
    """

    # Initialize dictionary to store data by region
    data_by_region = {} # Dictionary, region name -> array name -> list of data at each RR time point
    for region_name in mesh_regions.keys():
        data_by_region[region_name] = {}
        for array_name in array_names:
            data_by_region[region_name][array_name] = {}

    # Loop through each RR time point
    for rr in rr_timepoints:
        print(f"Processing RR{rr}...")
        # Load the simulation result for the current RR% time point
        sim_file = os.path.join(sim_results_folder, sim_file_pattern.format(rr=rr))
        print(f"Sim file: {sim_file}")
        sim_mesh = pv.read(sim_file)

        # Sample data from sim_mesh onto each region
        for region_name in mesh_regions.keys():
            mesh_regions[region_name] = mesh_regions[region_name].sample(sim_mesh)

        # Save the data by region
        for region_name, region_mesh in mesh_regions.items():
            for array_name in array_names:
                data_by_region[region_name][array_name][rr] = region_mesh.point_data[array_name]

    return data_by_region


def plot_result_data_by_region(data_by_region, rr_timepoints, output_folder):
    '''
    Plot the data from simulation results by region as violin plots over RR time points.

    Args:
        data_by_region (dict): Dictionary, region name -> array name -> list of data at each RR time point.
        output_folder (str): Path to the folder to save the output. If None, the output is not saved.   
    '''

    # Plot the data by region
    for region_name in data_by_region.keys():
        for array_name in data_by_region[region_name].keys():
            plt.figure()
            plt.violinplot(list(data_by_region[region_name][array_name].values()), showmeans=True, showmedians=True, showextrema=True)
            plt.title(f'{region_name} {array_name}')
            plt.xticks(range(1, len(rr_timepoints)+1), rr_timepoints)
            plt.xlabel('RR%')
            plt.ylabel(array_name)
            plt.savefig(os.path.join(output_folder, f'{region_name}_{array_name}_violinplot.png'))
            plt.close()

    return


def find_first_local_minimum(rr_percent):
    # Compute the indices of the local minima
    local_minima_indices = argrelextrema(rr_percent, np.less)

    # If there are any local minima, get the first one
    if local_minima_indices[0].size > 0:
        first_local_minimum_index = local_minima_indices[0][0]
        first_local_minimum = rr_percent[first_local_minimum_index]
        return first_local_minimum, first_local_minimum_index
    else:
        print("No local minima found.")
        return None
    
def find_first_local_maximum(rr_percent):
    # Compute the indices of the local maxima
    local_maxima_indices = argrelextrema(rr_percent, np.greater)

    # If there are any local maxima, get the first one
    if local_maxima_indices[0].size > 0:
        first_local_maximum_index = local_maxima_indices[0][0]
        first_local_maximum = rr_percent[first_local_maximum_index]
        return first_local_maximum, first_local_maximum_index
    else:
        print("No local maxima found.")
        return None
    
def calc_cardiac_phases(results_dict):
    '''
    Calculate timings of cardiac phases from results.

    Also, calculates RR interval percentage from time and PR_interval
    '''

    # Timestep size
    dt = np.diff(results_dict['time'])[0]

    # Compute cardiac cycle duration (compute from LV volume results)
    V_LV_hat = fft(results_dict['V_LV'] - np.mean(results_dict['V_LV'])) # Compute FFT of LV volume minus mean
    freq = fftfreq(len(results_dict['time']), d=results_dict['time'][1]-results_dict['time'][0]) # Compute frequency axis
    f_max = freq[np.argmax(np.abs(V_LV_hat))] # Find frequency of maximum power
    results_dict['T_HB_fft'] = 1. / f_max # seconds

    # Use T_HB from parameters if available, otherwise use from fft calculation
    if 'T_HB' in results_dict['parameters']:
        results_dict['T_HB'] = results_dict['parameters']['T_HB']
        print("Using T_HB found in parameters.")
        print(f"T_HB from parameters: {results_dict['T_HB']}")
        print(f"T_HB from FFT: {results_dict['T_HB_fft']}")
    else:
        results_dict['T_HB'] = results_dict['T_HB_fft']

    # Compute heart rate
    results_dict['HR'] = 1. / results_dict['T_HB'] * 60 # Beats per minute


    # Plot FFT of LV volume
    #N = len(results_dict['time'])
    #plt.plot(freq[0:N//2], np.abs(V_hat[0:N//2]))
    #plt.show()

    # Compute number of cardiac cycles in results
    n_cycles = int(np.floor(len(results_dict['time']) / (results_dict['T_HB'] / dt)))
    results_dict['n_cycles'] = n_cycles

    # Compute time derivatives of valve resistances
    dR_MV = np.gradient(results_dict['R_MV'], dt)
    dR_AV = np.gradient(results_dict['R_AV'], dt) 
    dR_TV = np.gradient(results_dict['R_TV'], dt)
    dR_PV = np.gradient(results_dict['R_PV'], dt) 

    # Plot valve resistances and their time derivatives
    # plt.figure()
    # plt.plot(results_dict['time'], results_dict['R_MV'], label='R_MV')
    # plt.plot(results_dict['time'], results_dict['R_AV'], label='R_AV')
    # plt.plot(results_dict['time'], results_dict['R_TV'], label='R_TV')
    # plt.plot(results_dict['time'], results_dict['R_PV'], label='R_PV')
    # plt.legend()
    # plt.savefig('valve_resistances.png')

    # Find timesteps and time of max/min dR/dt
    n_MV_close, _ = find_peaks(dR_MV)
    n_MV_open, _ = find_peaks(-dR_MV)
    t_MV_close = results_dict['time'][n_MV_close]
    t_MV_open = results_dict['time'][n_MV_open]

    n_AV_close, _ = find_peaks(dR_AV)
    n_AV_open, _ = find_peaks(-dR_AV)
    t_AV_close = results_dict['time'][n_AV_close]
    t_AV_open= results_dict['time'][n_AV_open]

    n_TV_close, _ = find_peaks(dR_TV)
    n_TV_open, _ = find_peaks(-dR_TV)
    t_TV_close= results_dict['time'][n_TV_close]
    t_TV_open= results_dict['time'][n_TV_open]

    n_PV_close, _ = find_peaks(dR_PV)
    n_PV_open, _ = find_peaks(-dR_PV)
    t_PV_close = results_dict['time'][n_PV_close]
    t_PV_open = results_dict['time'][n_PV_open]


    # Find timestep of start of atrial contraction. Start of atrial 
    # contraction marked by maximum in second derivative of atrial activation
    # (either elastance or active pressure)

    # Compute second derivative of atrial activation
    d2A_LA = np.gradient(np.gradient(results_dict['A_LA'], dt), dt)
    d2A_RA = np.gradient(np.gradient(results_dict['A_RA'], dt), dt)

    # Plot second derivative of atrial activation
    #plt.figure()
    #plt.plot(results_dict['time'], results_dict['A_LA'], label='A_LA')
    #plt.plot(results_dict['time'], d2A_LA/1e4, label='d2A_LA/10^4')
    #plt.legend()
    #plt.savefig('atrial_activation.png')

    # Find timesteps and time of max d2A_LA and d2A_RA.
    n_C_LA, _ = find_peaks(d2A_LA)  # Find peaks in signal
    prominences = peak_prominences(d2A_LA, n_C_LA)[0]   # Compute prominences of peaks
    n_C_LA = n_C_LA[prominences.argsort()[-n_cycles:]]  # Extract the first n_cycle peaks with the highest prominences
    n_C_LA = np.sort(n_C_LA)                            # Sort the peaks in ascending order
    t_C_LA = results_dict['time'][n_C_LA]

    n_C_RA, _ = find_peaks(d2A_RA)  # Find peaks in signal
    prominences = peak_prominences(d2A_RA, n_C_RA)[0]   # Compute prominences of peaks
    n_C_RA = n_C_RA[prominences.argsort()[-n_cycles:]]  # Extract the first n_cycle peaks with the highest prominences
    n_C_RA = np.sort(n_C_RA)                            # Sort the peaks in ascending order
    t_C_RA = results_dict['time'][n_C_RA]

    # Save results in results_dict
    results_dict['n_MV_close'] = n_MV_close
    results_dict['n_MV_open'] = n_MV_open
    results_dict['t_MV_close'] = t_MV_close
    results_dict['t_MV_open'] = t_MV_open

    results_dict['n_AV_close'] = n_AV_close
    results_dict['n_AV_open'] = n_AV_open
    results_dict['t_AV_close'] = t_AV_close
    results_dict['t_AV_open'] = t_AV_open

    results_dict['n_TV_close'] = n_TV_close
    results_dict['n_TV_open'] = n_TV_open
    results_dict['t_TV_close'] = t_TV_close
    results_dict['t_TV_open'] = t_TV_open

    results_dict['n_PV_close'] = n_PV_close
    results_dict['n_PV_open'] = n_PV_open
    results_dict['t_PV_close'] = t_PV_close
    results_dict['t_PV_open'] = t_PV_open

    results_dict['n_C_LA'] = n_C_LA
    results_dict['n_C_RA'] = n_C_RA
    results_dict['t_C_LA'] = t_C_LA
    results_dict['t_C_RA'] = t_C_RA

    # Calculate RR interval percentage
    t_cyc = np.mod(results_dict['time'], results_dict['T_HB']) # Time in cardiac cycle
    t_R_cyc = results_dict['parameters']['PR_interval'] # Time of R wave in cardiac cycle
    results_dict['RR%'] = np.mod(t_cyc - t_R_cyc, results_dict['T_HB']) / results_dict['T_HB'] * 100 # Percentage of RR interval

    # Construct interpolators from time to RR% and vice versa. For RR%_to_time,
    # use only the first full cardiac cycle (RR0% to RR100%) to interpolate the time. Otherwise, the
    # interpolation will not be one to one.
    results_dict['time_to_RR%'] = interp1d(results_dict['time'], results_dict['RR%'], fill_value='extrapolate')
    try:
        # Find first local min in RR%.
        first_local_min, first_local_min_index = find_first_local_minimum(results_dict['RR%'])
        # Find next local max in RR% after first local min.
        next_local_max, next_local_max_index = find_first_local_maximum(results_dict['RR%'][first_local_min_index:])
        # Construct interpolator from RR% to time using only the first full cardiac cycle.
        results_dict['RR%_to_time'] = interp1d(results_dict['RR%'][first_local_min_index:next_local_max_index], results_dict['time'][first_local_min_index:next_local_max_index], fill_value='extrapolate')
    except:
        results_dict['RR%_to_time'] = interp1d(results_dict['RR%'], np.zeros_like(results_dict['RR%']), fill_value='extrapolate')

    # Plot RR interval percentage
    # plt.figure()
    # plt.plot(results_dict['time'], results_dict['RR%'], label='RR%', linewidth=6)
    # plt.plot(results_dict['time'], results_dict['time_to_RR%'](results_dict['time']), label='time_to_RR%', linewidth=4)
    # plt.plot(results_dict['RR%_to_time'](results_dict['RR%']), results_dict['RR%'], label='RR%_to_time', linewidth=2)
    # plt.title('RR interval percentage')
    # plt.xlabel('Time (s)')
    # plt.ylabel('RR interval (%)')
    # plt.legend()
    # plt.show()


def compute_clinical_metrics(results_dict):
    '''
    Compute clinical metrics from simulation results.

    ARGS:
        - results_dict: Dictionary containing simulation results.
    '''

    # Extract relevant simulation metrics
    clinical_metrics = {}

    # Heart rate (beats per minute) and cardiac cycle duration (s)
    calc_cardiac_phases(results_dict)
    clinical_metrics['HR'] = {'Value': results_dict['HR'], 'Units': 'BPM'}
    clinical_metrics['T_HB'] = {'Value': results_dict['T_HB'], 'Units': 's'}

    # Aortic blood pressures (mmHg)
    clinical_metrics['P_AR_SYS_max'] = {'Value': np.max(results_dict['p_AR_SYS']),'Units': 'mmHg'}
    clinical_metrics['P_AR_SYS_min'] = {'Value': np.min(results_dict['p_AR_SYS']), 'Units': 'mmHg'}

    # Pulmonary blood pressures (mmHg)
    clinical_metrics['P_AR_PUL_max'] = {'Value': np.max(results_dict['p_AR_PUL']), 'Units': 'mmHg'}
    clinical_metrics['P_AR_PUL_min'] = {'Value': np.min(results_dict['p_AR_PUL']), 'Units': 'mmHg'}

    # Mean systemic arterial pressure (mmHg)
    clinical_metrics['P_AR_SYS_mean'] = {'Value': np.mean(results_dict['p_AR_SYS']), 'Units': 'mmHg'}

    # Mean pulmonary arterial pressure (mmHg)
    clinical_metrics['P_AR_PUL_mean'] = {'Value': np.mean(results_dict['p_AR_PUL']), 'Units': 'mmHg'}

    # Central venous pressure (mean right atrial pressure) (mmHg)
    clinical_metrics['P_RA_mean'] = {'Value': np.mean(results_dict['p_RA']), 'Units': 'mmHg'}

    # Pulmonary arterial wedge pressure (mean left atrial pressure) (mmHg)
    clinical_metrics['P_LA_mean'] = {'Value': np.mean(results_dict['p_LA']), 'Units': 'mmHg'}

    # Peripheral venous pressure (mean systemic venous pressure) (mmHg)
    clinical_metrics['P_VEN_SYS_mean'] = {'Value': np.mean(results_dict['p_VEN_SYS']), 'Units': 'mmHg'}

     # Cardiac volumes (mL). Interpolate values to image time points.
    clinical_metrics['V_LV'] = {'Value': np.interp(results_dict['time_img'], results_dict['time'], results_dict['V_LV']), 'Units': 'mL'}
    clinical_metrics['V_RV'] = {'Value': np.interp(results_dict['time_img'], results_dict['time'], results_dict['V_RV']), 'Units': 'mL'}
    clinical_metrics['V_LA'] = {'Value': np.interp(results_dict['time_img'], results_dict['time'], results_dict['V_LA']), 'Units': 'mL'}
    clinical_metrics['V_RA'] = {'Value': np.interp(results_dict['time_img'], results_dict['time'], results_dict['V_RA']), 'Units': 'mL'}
    
    # Cardiac min and max volumes (mL)
    clinical_metrics['V_LV_max'] = {'Value': np.max(results_dict['V_LV']), 'Units': 'mL'}
    clinical_metrics['V_LV_min'] = {'Value': np.min(results_dict['V_LV']), 'Units': 'mL'}
    clinical_metrics['V_RV_max'] = {'Value': np.max(results_dict['V_RV']), 'Units': 'mL'}
    clinical_metrics['V_RV_min'] = {'Value': np.min(results_dict['V_RV']), 'Units': 'mL'}
    clinical_metrics['V_LA_max'] = {'Value': np.max(results_dict['V_LA']), 'Units': 'mL'}
    clinical_metrics['V_LA_min'] = {'Value': np.min(results_dict['V_LA']), 'Units': 'mL'}
    clinical_metrics['V_RA_max'] = {'Value': np.max(results_dict['V_RA']), 'Units': 'mL'}
    clinical_metrics['V_RA_min'] = {'Value': np.min(results_dict['V_RA']), 'Units': 'mL'}

    # Stroke volumes (mL)
    clinical_metrics['LVSV'] = {'Value': clinical_metrics['V_LV_max']['Value'] - clinical_metrics['V_LV_min']['Value'], 'Units': 'mL'}
    clinical_metrics['RVSV'] = {'Value': clinical_metrics['V_RV_max']['Value'] - clinical_metrics['V_RV_min']['Value'], 'Units': 'mL'}
    clinical_metrics['LASV'] = {'Value': clinical_metrics['V_LA_max']['Value'] - clinical_metrics['V_LA_min']['Value'], 'Units': 'mL'}
    clinical_metrics['RASV'] = {'Value': clinical_metrics['V_RA_max']['Value'] - clinical_metrics['V_RA_min']['Value'], 'Units': 'mL'}
    
    # Ejection fraction (no units)
    clinical_metrics['LVEF'] = {'Value': clinical_metrics['LVSV']['Value'] / clinical_metrics['V_LV_max']['Value'], 'Units': '[]'}
    clinical_metrics['RVEF'] = {'Value': clinical_metrics['RVSV']['Value'] / clinical_metrics['V_RV_max']['Value'], 'Units': '[]'}

    # Cardiac output (L/min)
    clinical_metrics['CO'] = {'Value': clinical_metrics['HR']['Value'] * clinical_metrics['LVSV']['Value'] / 1000, 'Units': 'L/min'}

    # Systemic and pulmonary vascular resistances (mmHg/(L/min))
    clinical_metrics['SVR'] = {'Value': (clinical_metrics['P_AR_SYS_mean']['Value'] - clinical_metrics['P_RA_mean']['Value']) / (clinical_metrics['CO']['Value']), 'Units': 'mmHg/(L/min)'}
    clinical_metrics['PVR'] = {'Value': (clinical_metrics['P_AR_PUL_mean']['Value'] - clinical_metrics['P_LA_mean']['Value']) / (clinical_metrics['CO']['Value']), 'Units': 'mmHg/(L/min)'}

    # Stroke work (mmHg*mL)
    clinical_metrics['LVSW'] = {'Value': -trapezoid(results_dict['p_LV'], results_dict['V_LV']) / results_dict['n_cycles'], 'Units': 'mmHg*mL/cycle'}
    clinical_metrics['RVSW'] = {'Value': -trapezoid(results_dict['p_RV'], results_dict['V_RV']) / results_dict['n_cycles'], 'Units': 'mmHg*mL/cycle'}

    # Save metrics back to results_dict
    results_dict['clinical_metrics'] = clinical_metrics

def save_results(results_dict, output_dir):
    '''
    Save results to .txt files and as pickled file

    ARGS:
        - results_dict: Dictionary containing simulation results.
        - output_dir: Directory to save results.
    '''
    os.makedirs(output_dir, exist_ok=True)

    # Save results dictionary as pickle
    with open(os.path.join(output_dir, 'results_dict.pkl'), 'wb') as f:
        pickle.dump(results_dict, f)

    # Save all pressures to text file
    df_pressures = pd.DataFrame({
        'time': results_dict['time'],
        'p_LA': results_dict['p_LA'],
        'p_LV': results_dict['p_LV'],
        'p_RA': results_dict['p_RA'],
        'p_RV': results_dict['p_RV'],
        'p_AR_SYS': results_dict['p_AR_SYS'],
        'p_VEN_SYS': results_dict['p_VEN_SYS'],
        'p_AR_PUL': results_dict['p_AR_PUL'],
        'p_VEN_PUL': results_dict['p_VEN_PUL'],
    })
    with open(os.path.join(output_dir, 'pressures.txt'), 'w') as f:
        f.write(tabulate(df_pressures, headers='keys', tablefmt='plain', showindex=False))
    
    # Save all volumes to text file
    df_volume = pd.DataFrame({
        'time': results_dict['time'],
        'V_LA': results_dict['V_LA'],
        'V_LV': results_dict['V_LV'],
        'V_RA': results_dict['V_RA'],
        'V_RV': results_dict['V_RV'],
        'V_AR_SYS': results_dict['V_AR_SYS'],
        'V_VEN_SYS': results_dict['V_VEN_SYS'],
        'V_AR_PUL': results_dict['V_AR_PUL'],
        'V_VEN_PUL': results_dict['V_VEN_PUL'],
        'V_tot': results_dict['V_tot'],
    })
    with open(os.path.join(output_dir, 'volumes.txt'), 'w') as f:
        f.write(tabulate(df_volume, headers='keys', tablefmt='plain', showindex=False))

    # Save all flows to text file
    df_flows = pd.DataFrame({
        'time': results_dict['time'],
        'Q_MV': results_dict['Q_MV'],
        'Q_AV': results_dict['Q_AV'],
        'Q_TV': results_dict['Q_TV'],
        'Q_PV': results_dict['Q_PV'],
        'Q_AR_SYS': results_dict['Q_AR_SYS'],
        'Q_VEN_SYS': results_dict['Q_VEN_SYS'],
        'Q_AR_PUL': results_dict['Q_AR_PUL'],
        'Q_VEN_PUL': results_dict['Q_VEN_PUL'],
    })
    with open(os.path.join(output_dir, 'flows.txt'), 'w') as f:
        f.write(tabulate(df_flows, headers='keys', tablefmt='plain', showindex=False))

    # Save cardiac activations to text file
    df_activations = pd.DataFrame({
        'time': results_dict['time'],
        'A_LA': results_dict['A_LA'],
        'A_LV': results_dict['A_LV'],
        'A_RA': results_dict['A_RA'],
        'A_RV': results_dict['A_RV'],
    })
    with open(os.path.join(output_dir, 'activations.txt'), 'w') as f:
        f.write(tabulate(df_activations, headers='keys', tablefmt='plain', showindex=False))

    # Save valve resistances to text file
    df_resistances = pd.DataFrame({
        'time': results_dict['time'],
        'R_MV': results_dict['R_MV'],
        'R_AV': results_dict['R_AV'],
        'R_TV': results_dict['R_TV'],
        'R_PV': results_dict['R_PV'],
    })
    with open(os.path.join(output_dir, 'resistances.txt'), 'w') as f:
        f.write(tabulate(df_resistances, headers='keys', tablefmt='plain', showindex=False))

    # Save cardiac pressure and volume to text file
    df_cardiac_PV = pd.DataFrame({
        'time': results_dict['time'],
        'RR%': results_dict['RR%'],
        'V_LV': results_dict['V_LV'],
        'p_LV': results_dict['p_LV'],
        'V_LA': results_dict['V_LA'],
        'p_LA': results_dict['p_LA'],
        'V_RV': results_dict['V_RV'],
        'p_RV': results_dict['p_RV'],
        'V_RA': results_dict['V_RA'],
        'p_RA': results_dict['p_RA'],
    })
    with open(os.path.join(output_dir, 'cardiac_PV.txt'), 'w') as f:
        f.write(tabulate(df_cardiac_PV, headers='keys', tablefmt='plain', showindex=False))

    # Save clinical metrics to text file. Print value and units
    with open(os.path.join(output_dir, 'clinical_metrics.txt'), 'w') as f:
        for key, value in results_dict['clinical_metrics'].items():
            f.write(f"{key}: {value['Value']} {value['Units']}\n")

  
def plot_variables_vs_time(results_dict, variables, title, y_label, output_dir, output_file, subplots=False, phase_transitions = None, font_size=12, set_ylim_zero=False, image_variables=[]):
    '''
    Function to plot results vs time and save to file.

    ARGS:
        - results_dict: Dictionary containing simulation results.
        - variables: List of variables to plot.
        - title: Title of plot.
        - y_label: Label of y-axis.
        - output_dir: Directory to save plots.
        - output_file: Name of output file.
        - subplots: If True, plot each variable in separate subplot.
        - phase_transitions: If present, plot vertical lines at cardiac phase transitions.
        - font_size: Font size of plot labels.
        - set_ylim_zero: If True, set y-axis lower limit to 0.
        - image_variables: List of image variables to plot.

    RETURNS:
        - None
    '''

    # Store the original font size
    original_font_size = plt.rcParams['font.size']

    # Increase the font size
    plt.rcParams.update({'font.size': font_size})

    # Plot quantities vs time
    if subplots:
        fig, axs = plt.subplots(len(variables), 1, figsize=(8, 3*len(variables)))

        # Simulation variables
        for i, var in enumerate(variables):
            axs[i].plot(results_dict['time'], results_dict[var])
            axs[i].set_title(var)
            axs[i].set_ylabel(y_label)
            if set_ylim_zero:
                axs[i].set_ylim(bottom=0)

        # Image variables
        for i, var in enumerate(image_variables):
            axs[i].set_prop_cycle(None)
            axs[i].scatter(results_dict['time_img'], results_dict[var], label=var)

        # Set x-axis limits to time range
        for ax in axs:
            ax.set_xlim([results_dict['time'][0], results_dict['time'][-1]])

        axs[-1].set_xlabel('Time (s)')

    else:
        plt.figure(figsize=(8, 8))

        # Simulation variables
        for var in variables:
            plt.plot(results_dict['time'], results_dict[var], label=var)
        
        # Image variables
        plt.gca().set_prop_cycle(None)
        for var in image_variables:
            plt.scatter(results_dict['time_img'], results_dict[var], label=var)

        plt.title(title)
        plt.xlim([results_dict['time'][0], results_dict['time'][-1]])
        plt.xlabel('Time (s)')
        plt.ylabel(y_label)
        plt.legend(loc='upper right')
        if set_ylim_zero:
            plt.ylim(bottom=0)

    # Shade cardiac phases
    if phase_transitions:
        for phase_transition in phase_transitions:
            n_phase_transition = results_dict[phase_transition]
            for n in n_phase_transition:
                plt.axvline(x=results_dict['time'][n], color='grey', linestyle='--', linewidth=0.5, alpha=0.5)


    # Plot RR interval percentage as second x-axis
    RR_tick_start = results_dict['RR%_to_time'](0)
    RR_tick_end = results_dict['RR%_to_time'](100)
    RR_tick_locations = np.linspace(RR_tick_start, RR_tick_end, 11)

    xlim = plt.gca().get_xlim()
    ax2 = plt.gca().twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(RR_tick_locations)
    ax2.set_xticklabels(np.linspace(0, 100, 11).astype(int))
    ax2.set_xlabel('RR interval (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, output_file))
    plt.close()


    # Restore the original font size
    plt.rcParams.update({'font.size': original_font_size})

def remove_subsequent_pairs(arr):
    '''
    Remove subsequent pairs of elements in an array.

    Example:
    arr = [0, 2, 3, 4, 6, 7, 10]
    remove_subsequent_pairs(arr) -> [0, 4, 10]
    The pairs (2, 3) and (6, 7) are removed.

    ARGS:
        - arr: Numpy array
    
    RETURNS:
        - arr: Numpy array with subsequent pairs removed.
    '''
    i = 0
    while i < len(arr) - 1:
        if arr[i+1] - arr[i] == 1:
            arr = np.delete(arr, [i, i+1])
        else:
            i += 1
    return arr

def plot_pv_loops_with_phases(chamber, results_dict, arrows = False, phase_transition_markers = False, font_size=12, linestyle='-', marker=False, linewidth=2.0, alpha=1.0):
    '''
    Plot and save ventricular pressure volume loops with cardiac phases marked.
    '''

    # Store the original font size
    original_font_size = plt.rcParams['font.size']

    # Increase the font size
    plt.rcParams.update({'font.size': font_size})

    # Construct string based on chamber
    V = 'V_' + chamber
    p = 'p_' + chamber

    # Set color cycle
    colors = ['yellowgreen', 'gold', 'indianred', 'mediumpurple', 'deepskyblue']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
    
    # Label valve opening and closing points
    if chamber in ['LV', 'LA']:
        # Extract valve opening and closing timesteps and atrial contraction timesteps
        n_MV_close = results_dict['n_MV_close']
        n_MV_open = results_dict['n_MV_open']
        n_AV_close = results_dict['n_AV_close']
        n_AV_open = results_dict['n_AV_open']
        n_C_LA = results_dict['n_C_LA']

        # Create lists of MV and AV close and open timesteps
        n_MV_close_open = np.sort(np.concatenate((n_MV_close, n_MV_open)))
        n_AV_close_open = np.sort(np.concatenate((n_AV_close, n_AV_open)))

        # Check if number of transitions timesteps are equal (should be equal to number of cycles in results)
        condition = len(n_MV_close) == len(n_MV_open) == len(n_AV_close) == len(n_AV_open) == len(n_C_LA)
        if not condition:
            print(f"Number of left heart cardiac phase timesteps not equal.\n"
                            f"MV_close: {n_MV_close}, MV_open: {n_MV_open}, "
                            f"AV_close: {n_AV_close}, AV_open: {n_AV_open}, C_LA: {n_C_LA}"
                            )
            # Sometimes valves can close and open in subsequent timesteps. This is 
            # a numerical issue and messes up the coloring of PV loops, so it should
            # be ignored.
            # Look for valve openings and closings in subsequent timesteps and remove
            # the pair of timesteps
            n_MV_close_open = remove_subsequent_pairs(n_MV_close_open)
            n_AV_close_open = remove_subsequent_pairs(n_AV_close_open)

            print(f"Removed subsequent pairs of valve opening and closing timesteps.\n"
                    f"MV_close_open: {n_MV_close_open}, AV_close_open: {n_AV_close_open}\n")


        # Combine cardiac phase timesteps into single array and sort
        n_phases = np.sort(np.concatenate((n_MV_close_open, n_AV_close_open, n_C_LA)))

        # Remove all timesteps less than n_C_LA[0], so that colored plotting starts at atrial contraction
        n_phases = n_phases[n_phases >= n_C_LA[0]]

        # Add last timestep to n_phases to ensure all timesteps are included
        n_phases = np.append(n_phases, -2) # -2 because we add 1 to the index in the loop below

        # Plot entire PV loop in transparent gray first
        plt.plot(results_dict[V], results_dict[p], color='gray', alpha=0.25, linestyle=linestyle, marker=marker, markevery=0.1, linewidth=linewidth)

        # Loop over cardiac phase timesteps
        for i in range(len(n_phases)-1):
            # Reset color cycle after 5 iteration
            if i % 5 == 0:
                plt.gca().set_prop_cycle(None)
            
            # Plot cardiac phases of PV loop in different colors
            plt.plot(results_dict[V][n_phases[i]:n_phases[i+1]+1], results_dict[p][n_phases[i]:n_phases[i+1]+1], linestyle=linestyle, marker=marker, markevery=0.1, linewidth=linewidth, alpha=alpha)

            # Draw an arrow at the midpoint of each cardiac phase
            if arrows:
                n_mid = (n_phases[i] + n_phases[i+1]) // 2
                arrow_direction_x = results_dict[V][n_mid+1] - results_dict[V][n_mid]
                arrow_direction_y = results_dict[p][n_mid+1] - results_dict[p][n_mid]

                plt.arrow(results_dict[V][n_mid], results_dict[p][n_mid], 
                        arrow_direction_x, arrow_direction_y, 
                        head_width=2.0, fc='k', ec='k')

        # Plot markers at cardiac phase transitions
        if phase_transition_markers:
            plt.plot(results_dict[V][results_dict['n_MV_open']], results_dict[p][results_dict['n_MV_open']], 'ko', fillstyle='none', label='MV open')
            plt.plot(results_dict[V][results_dict['n_C_LA']], results_dict[p][results_dict['n_C_LA']], 'k*', fillstyle='none', label='Atr. cont.')
            plt.plot(results_dict[V][results_dict['n_MV_close']], results_dict[p][results_dict['n_MV_close']], 'kx', fillstyle='none', label='MV close')
            plt.plot(results_dict[V][results_dict['n_AV_open']], results_dict[p][results_dict['n_AV_open']], 'ks', fillstyle='none', label='AV open')
            plt.plot(results_dict[V][results_dict['n_AV_close']], results_dict[p][results_dict['n_AV_close']], 'k+', fillstyle='none', label='AV close')

    
    elif chamber in ['RV', 'RA']:
        # Extract valve opening and closing timesteps
        n_TV_close = results_dict['n_TV_close']
        n_TV_open = results_dict['n_TV_open']
        n_PV_close = results_dict['n_PV_close']
        n_PV_open = results_dict['n_PV_open']
        n_C_RA = results_dict['n_C_RA']

        # Create lists of TV and PV close and open timesteps
        n_TV_close_open = np.sort(np.concatenate((n_TV_close, n_TV_open)))
        n_PV_close_open = np.sort(np.concatenate((n_PV_close, n_PV_open)))

        # Check if number of transitions timesteps are equal (should be equal to number of cycles in results)
        condition = len(n_TV_close) == len(n_TV_open) == len(n_PV_close) == len(n_PV_open) == len(n_C_RA)
        if not condition:
            print(f"Number of right heart cardiac phase timesteps not equal.\n"
                            f"TV_close: {n_TV_close}, TV_open: {n_TV_open}, "
                            f"PV_close: {n_PV_close}, PV_open: {n_PV_open}, C_RA: {n_C_RA}"
                            )
            
            # Sometimes valves can close and open in subsequent timesteps. This is 
            # a numerical issue and messes up the coloring of PV loops, so it should
            # be ignored.
            # Look for valve openings and closings in subsequent timesteps and remove
            # the pair of timesteps
            n_TV_close_open = remove_subsequent_pairs(n_TV_close_open)
            n_PV_close_open = remove_subsequent_pairs(n_PV_close_open)

            print(f"Removed subsequent pairs of valve opening and closing timesteps.\n"
                    f"TV_close_open: {n_TV_close_open}, PV_close_open: {n_PV_close_open}\n")
    
        # Combine valve opening and closing timesteps into single array and sort
        n_phases = np.sort(np.concatenate((n_TV_close_open, n_PV_close_open, n_C_RA)))

        # Remove all timesteps less than n_C_RA[0], so that colored plotting starts at atrial contraction
        n_phases = n_phases[n_phases >= n_C_RA[0]]

        # Add last timestep to n_phases to ensure all timesteps are included
        n_phases = np.append(n_phases, -2) # -2 because we add 1 to the index in the loop below

        # Plot entire PV loop in transparent gray first
        plt.plot(results_dict[V], results_dict[p], color='gray', alpha=0.25, linestyle=linestyle, marker=marker, markevery=0.1, linewidth=linewidth)

        # Loop over cardiac phase timesteps
        for i in range(len(n_phases)-1):
            # Reset color cycle after 5 iteration
            if i % 5 == 0:
                plt.gca().set_prop_cycle(None)

            # Plot cardiac phases of PV loop in different colors
            plt.plot(results_dict[V][n_phases[i]:n_phases[i+1]+1], results_dict[p][n_phases[i]:n_phases[i+1]+1], linestyle=linestyle, marker=marker, markevery=0.1, linewidth=linewidth, alpha=alpha)

            if arrows:
                # Draw an arrow at the midpoint of each cardiac phase
                n_mid = (n_phases[i] + n_phases[i+1]) // 2
                arrow_direction_x = results_dict[V][n_mid+1] - results_dict[V][n_mid]
                arrow_direction_y = results_dict[p][n_mid+1] - results_dict[p][n_mid]

                plt.arrow(results_dict[V][n_mid], results_dict[p][n_mid], 
                        arrow_direction_x, arrow_direction_y, 
                        head_width=2.0, fc='k', ec='k')

        # Plot markers at cardiac phase transitions
        if phase_transition_markers:
            plt.plot(results_dict[V][results_dict['n_TV_open']], results_dict[p][results_dict['n_TV_open']], 'bo', fillstyle='none', label='TV open')
            plt.plot(results_dict[V][results_dict['n_C_RA']], results_dict[p][results_dict['n_C_RA']], 'b*', fillstyle='none', label='Atr. cont.')
            plt.plot(results_dict[V][results_dict['n_TV_close']], results_dict[p][results_dict['n_TV_close']], 'bx', fillstyle='none', label='TV close')
            plt.plot(results_dict[V][results_dict['n_PV_open']], results_dict[p][results_dict['n_PV_open']], 'bs', fillstyle='none', label='PV open')
            plt.plot(results_dict[V][results_dict['n_PV_close']], results_dict[p][results_dict['n_PV_close']], 'b+', fillstyle='none', label='PV close')

    # Reset color cycle to default
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

    # Restore the original font size
    plt.rcParams.update({'font.size': original_font_size})

def plot_results(results_dict, output_dir, font_size=12):
    '''
    Plot and save results.

    ARGS:
        - results_dict: Dictionary containing simulation results.
        - output_dir: Directory to save plots.
    '''
    os.makedirs(output_dir, exist_ok=True)

    
    calc_cardiac_phases(results_dict)

    # Cardiac volumes
    plot_variables_vs_time(results_dict, ['V_LV', 'V_RV', 'V_LA', 'V_RA'], 
                        'Cardiac Volumes', 
                        'Volume (mL)', 
                        output_dir, 
                        'card_volumes.png', 
                        phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close',
                                        'n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
                        font_size=font_size,
                        image_variables=['V_LV_img', 'V_RV_img', 'V_LA_img', 'V_RA_img'])

    # Cardiac pressures
    plot_variables_vs_time(results_dict, ['p_LV', 'p_RV', 'p_LA', 'p_RA'], 
                        'Cardiac Pressures', 
                        'Pressure (mmHg)', 
                        output_dir, 
                        'card_pressures.png',
                        phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close',
                                        'n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
                        font_size=font_size)

    # Systemic pressures
    plot_variables_vs_time(results_dict, ['p_LA', 'p_LV','p_AR_SYS', 'p_VEN_SYS'], 
                        'Systemic Pressures', 
                        'Pressure (mmHg)', 
                        output_dir, 
                        'sys_pressures.png',
                        phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close'],
                        font_size=font_size)
    
    # Pulmonary pressures
    plot_variables_vs_time(results_dict, ['p_RA', 'p_RV','p_AR_PUL', 'p_VEN_PUL'], 
                        'Pulmonary Pressures', 
                        'Pressure (mmHg)', 
                        output_dir, 
                        'pul_pressures.png',
                        phase_transitions=['n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
                        font_size=font_size)
    
    # Systemic flows
    plot_variables_vs_time(results_dict, ['Q_MV', 'Q_AV', 'Q_AR_SYS', 'Q_VEN_SYS'], 
                        'Systemic Flows', 
                        'Flow (mL/s)', 
                        output_dir, 
                        'sys_flows.png', 
                        phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close'],
                        font_size=font_size)

    # Pulmonary flows
    plot_variables_vs_time(results_dict, ['Q_TV', 'Q_PV', 'Q_AR_PUL', 'Q_VEN_PUL'], 
                        'Pulmonary Flows', 
                        'Flow (mL/s)', 
                        output_dir, 
                        'pul_flows.png',
                        phase_transitions=['n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
                        font_size=font_size)
    
    # Cardiac activations
    plot_variables_vs_time(results_dict, ['A_LV', 'A_RV', 'A_LA', 'A_RA'],
                        'Cardiac Activations',
                        'Activation',
                        output_dir,
                        'card_activations.png',
                        phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close',
                                        'n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
                        font_size=font_size,
                        subplots=True)
    
        
    # Plot valve resistances
    plot_variables_vs_time(results_dict, ['R_MV', 'R_AV', 'R_TV', 'R_PV'],
                        'Valve Resistances',
                        'Resistance (mmHg/mL/s)',
                        output_dir,
                        'valve_resistances.png',
                        subplots=True,
                        font_size=font_size)
    
    # Plot all volumes
    plot_variables_vs_time(results_dict, ['V_LA', 'V_LV', 'V_RA', 'V_RV', 'V_AR_SYS', 'V_VEN_SYS', 'V_AR_PUL', 'V_VEN_PUL', 'V_tot'],
                        'Volumes',
                        'Volume (mL)',
                        output_dir,
                        'all_volumes.png',
                        font_size=font_size)
    
    # Fiber stress
    plot_variables_vs_time(results_dict, ['fiber_stress'],
                        'Fiber Stress',
                        'Stress (Pa)',
                        output_dir,
                        'fiber_stress.png',
                        font_size=font_size)


    # Plot ventricular pressure volume loops
    plt.figure(figsize=(8, 8))
    plot_pv_loops_with_phases('LV', results_dict, font_size=font_size, linestyle='--')
    plot_pv_loops_with_phases('RV', results_dict, font_size=font_size, linestyle='-.')
    plt.title('Ventricular Pressure-Volume Loops', fontsize=font_size)
    plt.xlabel('Volume (mL)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend(fontsize=font_size)
    plt.savefig(os.path.join(output_dir, 'ventricular_pv_loops.png'))
    #plt.show()
    plt.close()

    # Plot LV pressure volume loop
    plt.figure(figsize=(8, 8))
    plot_pv_loops_with_phases('LV', results_dict, font_size=font_size)
    plt.title('LV Pressure-Volume Loop', fontsize=font_size)
    plt.xlabel('Volume (mL)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend(fontsize=font_size)
    plt.savefig(os.path.join(output_dir, 'lv_pv_loop.png'))
    plt.close()


    # Plot atrial pressure volume loops
    plt.figure(figsize=(8, 8))
    plot_pv_loops_with_phases('LA', results_dict, font_size=font_size, linestyle='--')
    plot_pv_loops_with_phases('RA', results_dict, font_size=font_size, linestyle='-.')
    plt.title('Atrial Pressure-Volume Loops', fontsize=font_size)
    plt.xlabel('Volume (mL)')
    plt.ylabel('Pressure (mmHg)')
    plt.legend(fontsize=font_size)
    plt.savefig(os.path.join(output_dir, 'atrial_pv_loops.png'))
    plt.close()

    # Plot the 3D vs. 0D volumes
    plt.figure()
    V_0D_keys = ['V_LV_0D', 'V_RV_0D']
    V_0D_keys = [key for key in V_0D_keys if key in results_dict]
    V_keys = ['V_LV', 'V_RV']
    V_keys = [key for key in V_keys if key in results_dict]
    for key in V_0D_keys:
        plt.plot(results_dict['time'], results_dict[key], label=key)
    # Reset color cycle to default
    plt.gca().set_prop_cycle(None)
    for key in V_keys:
        plt.plot(results_dict['time'], results_dict[key], label=key, linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Volume (cm^3)')
    plt.title('LV and RV volumes')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, '3D_vs_0D_LVRV_volumes.png'))
    plt.close()


    # Plot left heart Wiggers-like diagram in subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(results_dict['time'], results_dict['p_LV'], label='p_LV')
    axs[0].plot(results_dict['time'], results_dict['p_LA'], label='p_LA')
    axs[0].plot(results_dict['time'], results_dict['p_AR_SYS'], label='p_AR_SYS')
    # Plot patient pressure targets
    p_sys_val = results_dict['patient_metrics']['P_AR_SYS_max']['Value']
    p_dias_val = results_dict['patient_metrics']['P_AR_SYS_min']['Value']
    p_sys_tolerance = results_dict['patient_metrics']['P_AR_SYS_max']['Relative range'] * p_sys_val
    p_dias_tolerance = results_dict['patient_metrics']['P_AR_SYS_min']['Relative range'] * p_dias_val
    axs[0].axhline(y=p_sys_val, linestyle=':', color = 'tab:green', label='Sys./Dias. target')
    axs[0].fill_between(results_dict['time'], p_sys_val-p_sys_tolerance, p_sys_val+p_sys_tolerance, color='tab:green', alpha=0.1)
    axs[0].axhline(y=p_dias_val, linestyle=':', color = 'tab:green')
    axs[0].fill_between(results_dict['time'], p_dias_val-p_dias_tolerance, p_dias_val+p_dias_tolerance, color='tab:green', alpha=0.1)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Pressure (mmHg)')
    axs[0].set_title('Left heart pressures')
    axs[0].legend(loc = 'upper right')
    axs[0].grid()

    axs[1].plot(results_dict['time'], results_dict['V_LV'], label='V_LV')
    axs[1].plot(results_dict['time'], results_dict['V_LA'], label='V_LA')
    # Plot image LV and LA volume
    axs[1].set_prop_cycle(None)
    axs[1].scatter(results_dict['time_img'], results_dict['V_LV_img'], label='V_LV_img')
    axs[1].scatter(results_dict['time_img'], results_dict['V_LA_img'], label='V_LA_img')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Volume (cm^3)')
    axs[1].set_title('Left heart volumes')
    axs[1].legend(loc = 'upper right')
    axs[1].grid()

    axs[2].plot(results_dict['time'], results_dict['fiber_stress'])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Fiber stress (Pa)')
    axs[2].set_title('Fiber stress')
    axs[2].grid()

    # Set x-limits to be the time range for all subplots
    for ax in axs:
        ax.set_xlim([results_dict['time'][0], results_dict['time'][-1]])

    fig.suptitle('Left heart Wiggers-like diagram')
    plt.savefig(os.path.join(output_dir, 'wiggers_left.png'))

    # Plot right heart Wiggers-like diagram in subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(results_dict['time'], results_dict['p_RV'], label='p_RV')
    axs[0].plot(results_dict['time'], results_dict['p_RA'], label='p_RA')
    axs[0].plot(results_dict['time'], results_dict['p_AR_PUL'], label='p_AR_PUL')
    # Plot patient pressure targets
    p_sys_pul_val = results_dict['patient_metrics']['P_AR_PUL_max']['Value']
    p_dias_pul_val = results_dict['patient_metrics']['P_AR_PUL_min']['Value']
    p_sys_pul_tolerance = results_dict['patient_metrics']['P_AR_PUL_max']['Relative range'] * p_sys_pul_val
    p_dias_pul_tolerance = results_dict['patient_metrics']['P_AR_PUL_min']['Relative range'] * p_dias_pul_val
    axs[0].axhline(y=p_sys_pul_val, linestyle=':', color = 'tab:green', label='Sys./Dias. target')
    axs[0].fill_between(results_dict['time'], p_sys_pul_val-p_sys_pul_tolerance, p_sys_pul_val+p_sys_pul_tolerance, color='tab:green', alpha=0.1)
    axs[0].axhline(y=p_dias_pul_val, linestyle=':', color = 'tab:green')
    axs[0].fill_between(results_dict['time'], p_dias_pul_val-p_dias_pul_tolerance, p_dias_pul_val+p_dias_pul_tolerance, color='tab:green', alpha=0.1)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Pressure (mmHg)')
    axs[0].set_title('Right heart pressures')
    axs[0].legend(loc = 'upper right')
    axs[0].grid()

    axs[1].plot(results_dict['time'], results_dict['V_RV'], label='V_RV')
    axs[1].plot(results_dict['time'], results_dict['V_RA'], label='V_RA')
    # Plot image RV and RA volume
    axs[1].set_prop_cycle(None)
    axs[1].scatter(results_dict['time_img'], results_dict['V_RV_img'], label='V_RV_img')
    axs[1].scatter(results_dict['time_img'], results_dict['V_RA_img'], label='V_RA_img')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Volume (cm^3)')
    axs[1].set_title('Right heart volumes')
    axs[1].legend(loc = 'upper right')
    axs[1].grid()

    axs[2].plot(results_dict['time'], results_dict['fiber_stress'])
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Fiber stress (Pa)')
    axs[2].set_title('Fiber stress')
    axs[2].grid()

    # Set x-limits to be the time range for all subplots
    for ax in axs:
        ax.set_xlim([results_dict['time'][0], results_dict['time'][-1]])

    fig.suptitle('Right heart Wiggers-like diagram')
    plt.savefig(os.path.join(output_dir, 'wiggers_right.png'))

    # Plot MV, TV, and top plane displacements (whichever exists)
    keys = ['MV_plane_displacement', 'TV_plane_displacement', 'top_plane_displacement']
    keys = [key for key in keys if key in results_dict]
    keys_img = [key + '_img' for key in keys]
    plot_variables_vs_time(results_dict, keys,
                        'Valve Plane Displacements', 
                        'Displacement (cm)', 
                        output_dir, 
                        'valve_plane_displacements.png',
                        phase_transitions=['n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
                        font_size=font_size,
                        image_variables=keys_img)

    # Plot LV and RV longitudinal length
    keys = ['LV_longitudinal_length', 'RV_longitudinal_length']
    keys = [key for key in keys if key in results_dict]
    keys_img = [key + '_img' for key in keys]
    plot_variables_vs_time(results_dict, keys,
                        'Longitudinal Lengths', 
                        'Length (cm)', 
                        output_dir, 
                        'longitudinal_lengths.png',
                        phase_transitions=['n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
                        font_size=font_size,
                        image_variables=keys_img)
    
    # Plot LV and RV wall thickness
    keys = ['LV_wall_thickness', 'RV_wall_thickness']
    keys = [key for key in keys if key in results_dict]
    keys_img = [key + '_img' for key in keys]
    plot_variables_vs_time(results_dict, keys,
                        'Wall Thickness', 
                        'Thickness (cm)', 
                        output_dir, 
                        'wall_thickness.png',
                        phase_transitions=['n_MV_open', 'n_C_LA', 'n_MV_close', 'n_AV_open', 'n_AV_close',
                                        'n_TV_open', 'n_C_RA', 'n_TV_close', 'n_PV_open', 'n_PV_close'],
                        font_size=font_size,
                        set_ylim_zero=True,
                        image_variables=keys_img)
    
    # Plot myocardial volume
    plot_variables_vs_time(results_dict, ['myocardial_volume'],
                        'Myocardial Volume', 
                        'Volume (cm^3)', 
                        output_dir, 
                        'myocardial_volume.png',
                        font_size=font_size,
                        image_variables=['myocardial_volume_img'])

def plot_metrics(pat_metrics, sim_metrics, file_path, font_size=10):
    """
    Plot simulation and patient-specific metrics using a bar plot.
    Note: keys in pat_metrics must be a subset of key in sim_metrics.

    Parameters:
    - pat_metrics: dict, patient-specific metrics with labels
    - sim_metrics: dict, metrics
    """

    # Store the original font size
    original_font_size = plt.rcParams['font.size']

    # Increase the font size
    plt.rcParams.update({'font.size': font_size})

    pat_labels = list(pat_metrics.keys())

    # Separate scalar and vector values in the dictionary, since we will plot them differently
    pat_labels_scalars = [label for label in pat_labels if np.isscalar(pat_metrics[label]['Value'])]
    pat_labels_vectors = [label for label in pat_labels if not np.isscalar(pat_metrics[label]['Value'])]

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios': [4, 1]})

    # Scalar values bar plot
    width = 0.35  # Width of the bars
    x = np.arange(len(pat_labels_scalars))  # Generate x-coordinates for each label
    pat_values_scalars = np.array([pat_metrics[labels]['Value'] for labels in pat_labels_scalars])
    sim_values_scalars = np.array([sim_metrics[labels]['Value'] for labels in pat_labels_scalars])
    axs[0].bar(x - width/2, pat_values_scalars, width, label='Patient', alpha=0.7)
    axs[0].bar(x + width/2, sim_values_scalars, width, label='Simulation', alpha=0.7)
    axs[0].set_title('Patient vs Simulation Metrics')
    axs[0].set_xlabel('Metrics')
    axs[0].set_ylabel('Values')

    # Get metric name mapping to use in x-axis labels
    metric_name_map = get_stylized_metric_name_map()

    axs[0].set_xticks(x)  # Set the x-axis labels
    axs[0].set_xticklabels([metric_name_map.get(label, label) for label in pat_labels_scalars])
    axs[0].legend()
    axs[0].grid(visible=True, linewidth=0.5, alpha=0.5)

    # Alternate the labels up and down
    for i, label in enumerate(axs[0].get_xticklabels()):
        if i % 2 == 0:
            label.set_verticalalignment('bottom')
        else:
            label.set_verticalalignment('top')
        label.set_y(label.get_position()[1] - 0.1)  # Move label slightly down

    # Plot RMS error for vector values
    width = 0.35  # Width of the bars
    x = np.arange(len(pat_labels_vectors))  # Generate x-coordinates for each label
    RMS_error = np.array([np.sqrt(np.mean((sim_metrics[label]['Value'] - pat_metrics[label]['Value'])**2)) for label in pat_labels_vectors])
    axs[1].bar(x, RMS_error, width, label='RMS Error', alpha=0.7, color='red')
    axs[1].set_title('Time Series Error')
    axs[1].set_xlabel('Metrics')
    axs[1].set_ylabel('RMS Error')
    axs[1].set_xticks(x)  # Set the x-axis labels
    axs[1].set_xticklabels([metric_name_map.get(label, label) for label in pat_labels_vectors])
    axs[1].grid(visible=True, linewidth=0.5, alpha=0.5)

    # Move labels down
    for i, label in enumerate(axs[1].get_xticklabels()):
        label.set_y(label.get_position()[1] - 0.1)  # Move label slightly down

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    # Reset the font size
    plt.rcParams.update({'font.size': original_font_size})

def plot_metrics_normalized(pat_metrics, sim_metrics, file_path, font_size=10):
    """
    Plot simulation and patient-specific metrics using a bar plot, normalized
    by the patient-specific values.

    Parameters:
    - pat_metrics: dict, patient-specific metrics with labels
    - sim_metrics: dict, metrics from simulation with labels
    """

    # Store the original font size
    original_font_size = plt.rcParams['font.size']

    # Increase the font size
    plt.rcParams.update({'font.size': font_size})

    pat_labels = list(pat_metrics.keys())
    
    # Separate scalar and vector values in the dictionary, since we will plot them differently
    pat_labels_scalars = [label for label in pat_labels if np.isscalar(pat_metrics[label]['Value'])]
    pat_labels_vectors = [label for label in pat_labels if not np.isscalar(pat_metrics[label]['Value'])]

    # Get metric name mapping to use in x-axis labels
    metric_name_map = get_stylized_metric_name_map()

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(6, 3), gridspec_kw={'width_ratios': [4, 1]})

    # Scalar values bar plot
    width = 0.35  # Width of the bars
    x = np.arange(len(pat_labels_scalars))  # Generate x-coordinates for each label
    pat_values_scalars = np.array([pat_metrics[labels]['Value'] for labels in pat_labels_scalars])
    sim_values_scalars = np.array([sim_metrics[labels]['Value'] for labels in pat_labels_scalars])
    axs[0].bar(x - width/2, pat_values_scalars / pat_values_scalars * 100, width, 
            label='Patient', alpha=0.7)
    axs[0].bar(x + width/2, sim_values_scalars / pat_values_scalars * 100, width, 
            label='Simulation', alpha=0.7)   
    axs[0].set_title('Patient vs Simulation Metrics')
    axs[0].set_xlabel('Metrics')
    axs[0].set_ylabel("% of Patient Value")
    axs[0].set_xticks(x)  # Set the x-axis labels
    axs[0].set_xticklabels([metric_name_map.get(label, label) for label in pat_labels_scalars])
    axs[0].legend(loc='lower right')

    # Alternate the labels up and down
    for i, label in enumerate(axs[0].get_xticklabels()):
        if i % 2 == 0:
            label.set_verticalalignment('bottom')
        else:
            label.set_verticalalignment('top')
        label.set_y(label.get_position()[1] - 0.1)  # Move label slightly down

    # Plot normalized RMS error for vector values
    width = 0.35  # Width of the bars
    x = np.arange(len(pat_labels_vectors))  # Generate x-coordinates for each label
    normalized_RMS_error = np.array([np.sqrt(np.mean((sim_metrics[label]['Value'] - pat_metrics[label]['Value'])**2)) / (np.max(pat_metrics[label]['Value']) - np.min(pat_metrics[label]['Value'])) for label in pat_labels_vectors])
    axs[1].bar(x, normalized_RMS_error * 100, width, label='Normalized RMS Error', alpha=0.7, color='red')
    axs[1].set_title('Time Series Error')
    axs[1].set_xlabel('Metrics')
    axs[1].set_ylabel('% Normalized RMS Error')
    axs[1].set_xticks(x)  # Set the x-axis labels
    axs[1].set_xticklabels([metric_name_map.get(label, label) for label in pat_labels_vectors])

    # Move labels down
    for i, label in enumerate(axs[1].get_xticklabels()):
        label.set_y(label.get_position()[1] - 0.1)  # Move label slightly down

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    # Reset the font size
    plt.rcParams.update({'font.size': original_font_size})

def copy_sim_results(results_dict, results_folder, processed_results_folder):
    """
    Copy simulation results.vtu files corresponding to imaged times for comparison.

    Parameters:
    - results_dict: Dictionary containing the results
    - results_folder: Folder containing the simulation results
    - processed_results_folder: Folder to save the processed results
    """
    t_3D = results_dict['t_3D']
    timestep_size = results_dict['dt']
    start_timestep = results_dict['start_timestep']
    _, times_to_save = find_closest_times(t_3D, results_dict['time_img'])
    timesteps_to_save = [int(time / timestep_size) for time in times_to_save]
    results_at_imaged_times_folder = os.path.join(processed_results_folder, 'results_at_imaged_times')
    if not os.path.exists(results_at_imaged_times_folder):
        os.makedirs(results_at_imaged_times_folder)
    for i, timestep in enumerate(timesteps_to_save):
        print(f"Timestep: {timestep}, time: {times_to_save[i]}, RR%: {results_dict['RR%_img'][i]}")
        print(f"\tCopying result_{timestep:03}.vtu to result_RR{results_dict['RR%_img'][i]}.vtu.")
        result_filename = os.path.join(results_folder, f'result_{timestep:03}.vtu')
        if os.path.exists(result_filename):
            shutil.copy(result_filename, os.path.join(results_at_imaged_times_folder, f'result_RR{results_dict["RR%_img"][i]}.vtu'))
        else:
            print(f"\tFile {result_filename} does not exist.")
            default_result_filename = os.path.join(results_folder, f'result_{start_timestep:03}.vtu')
            print(f"\tCopying default result_{start_timestep:03}.vtu to result_RR{results_dict['RR%_img'][i]}.vtu.")
            shutil.copy(default_result_filename, os.path.join(results_at_imaged_times_folder, f'result_RR{results_dict["RR%_img"][i]}.vtu'))

    return results_at_imaged_times_folder

def read_results_dict(results_dict_path):
    '''
    Reads the processed results from a simulation.
    '''

    with open(results_dict_path, 'rb') as f:
        results = pickle.load(f)
    
    return results

def get_last_n_cardiac_cycles(results_dict, n):
    '''
    Keep only the last n cardiac cycles of the results.
    '''

    # Calculate the number of timesteps from the end of the simulation to process
    T_HB = results_dict['parameters']['T_HB'] # Heartbeat period
    timestep_size = results_dict['time'][1] - results_dict['time'][0] # Timestep size
    if n > 0:
        process_last_n_timesteps = int(n * T_HB / timestep_size) # Number of timesteps to process
    else:
        # Process all timesteps
        return results_dict 

    # Loop over keys and only keep the last process_last_n_timesteps
    for key in results_dict.keys():
        if isinstance(results_dict[key], (list, tuple, np.ndarray)) and key != 'time':
            if len(results_dict[key]) == len(results_dict['time']):
                results_dict[key] = results_dict[key][-process_last_n_timesteps:] 
    # Finally, process the time array
    results_dict['time'] = results_dict['time'][-process_last_n_timesteps:]

    return results_dict

def compute_rel_error(pat_metrics, sim_metrics):
    """
    Compute the relative error between patient-specific metrics and simulation metrics.
    Note: keys in pat_metrics must be a subset of key in sim_metrics.

    Parameters:
    - pat_metrics: dict, patient-specific metrics with labels
    - sim_metrics: dict, metrics from simulation with labels

    Returns:
    - error: dict, computed relative error between simulation and experimental metrics
    - tot_error: Sum of all relative errors
    """

    # Check units match between sim_metrics and pat_metrics for each metrics
    for label in pat_metrics:
        assert sim_metrics[label]['Units'] == pat_metrics[label]['Units'], f"Units do not match for {label}. {sim_metrics[label]['Units']} != {pat_metrics[label]['Units']}"

    # Compute relative error for each metric
    error = {}
    for label in pat_metrics:
        if np.isscalar(pat_metrics[label]['Value']):
            # For scalar values, compute relative error
            e = np.abs(sim_metrics[label]['Value'] - pat_metrics[label]['Value'])/pat_metrics[label]['Value']
        else:
            # For array values, compute normalized RMS error
            e = np.sqrt(np.mean((sim_metrics[label]['Value'] - pat_metrics[label]['Value'])**2)) / (np.max(pat_metrics[label]['Value']) - np.min(pat_metrics[label]['Value']))
        error[label] = e

    # Compute relative error greater than alpha
    error_rel_alpha = {}
    for label in error:
        alpha = pat_metrics[label]['Relative range'] # Relative error threshold
        if error[label] > alpha:
            error_rel_alpha[label] = error[label] - alpha
        elif np.isnan(error[label]):
            error_rel_alpha[label] = 1e10
        else:
            error_rel_alpha[label] = 0.

    # Compute total relative error greater than alpha
    tot_error = np.sum(list(error_rel_alpha.values()))

    # Compute sum squared relative error greater than alpha
    sum_squared_error = np.sum([error_rel_alpha[label]**2 for label in error])

    return error, tot_error, sum_squared_error

def load_and_process_results(results_dict_to_load_path, reprocess_last_n_cardiac_cycles, processed_results_folder, results_folder):
    '''
    Load and process results.
    '''

    # Load the results_dict
    results_dict = read_results_dict(results_dict_to_load_path)
    
    # Keep only the last n cardiac cycles of the results
    results_dict = get_last_n_cardiac_cycles(results_dict, reprocess_last_n_cardiac_cycles)
    
    # Put processed results in a new folder
    processed_results_folder = f"{processed_results_folder}_last_{reprocess_last_n_cardiac_cycles}_cardiac_cycles"
    
    # Process and plot the results
    calc_cardiac_phases(results_dict) # To regenerate RR%_to_time interpolator
    results_dict['time_img'] = results_dict['RR%_to_time'](results_dict['RR%_img'])
    compute_clinical_metrics(results_dict)
    plot_results(results_dict, processed_results_folder)
    sim_metrics = results_dict['clinical_metrics']
    from data.patient_metrics import pat_metrics
    plot_metrics(pat_metrics, sim_metrics, os.path.join(processed_results_folder, 'metric_comparison.png'), font_size=20)
    plot_metrics_normalized(pat_metrics, sim_metrics, os.path.join(processed_results_folder, 'metric_comparison_normalized.png'), font_size=20)

    # Compute relative error between patient-specific metrics and simulation metrics and save the results as a text file
    error, total_error, sum_squared_error = compute_rel_error(pat_metrics, sim_metrics)
    results_dict['error'] = error
    results_dict['error']['total_error'] = total_error
    results_dict['error']['sum_squared_error'] = sum_squared_error
    with open(os.path.join(processed_results_folder, 'error.txt'), 'w') as f:
        for label in results_dict['error']:
            f.write(f"{label}: {results_dict['error'][label]}\n")

    return processed_results_folder, results_dict

