import numpy as np
import mcubes
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import time
import pyvista as pv
import os
from convert_obj_to_vtp import convert_obj_to_vtp
import trimesh
from scipy.optimize import minimize
import concurrent.futures

def spiral_support(x, y, z, n_spirals=8, helix_angle=30, on_surface=False):
    tube_radius = 0.45 * np.sqrt(2) # to preserve cross-sectional area
    a = 12.2  # x, y radius of ellipsoid path
    z_start = -0.8 - 2*tube_radius # where the spiral starts
    z_end = -30.5 + 5 # where the spiral ends
    top = 0 # top of ellipsoid path
    bottom = -29.7 # bottom of ellipsoid path

    #if z < z_end:
    #    return z_end - z
    
    # Early exit if point is too far from spiral region
    outer_ellipse = ellipse(x, y, z, a + 2*tube_radius, -29.7 - 2*tube_radius)
    center_ellipse = ellipse(x, y, z, a - 2*tube_radius, -29.7 + 2*tube_radius)
    if outer_ellipse > 0 or z > top or z < bottom or center_ellipse < 0:
        return 2*tube_radius #outside the ventricle, no spiral support
    
    def z_to_xy(z, theta_start, on_surface=on_surface):
        z_norm = (z - top) / (bottom - top)
        if on_surface:
            theta = np.arcsin(z_norm)* (bottom - top) / (a*np.tan(np.deg2rad(helix_angle))) + theta_start
        else:
            theta = z_norm * (bottom - top) / (a*np.tan(np.deg2rad(helix_angle))) + theta_start
        spiral_x = a * np.sqrt(1 - z_norm**2) * np.cos(theta)
        spiral_y = a * np.sqrt(1 - z_norm**2) * np.sin(theta)
        return spiral_x, spiral_y
    
    def distance_to_spiral_at_z(x, y, z, spiral_z, theta_start):
        """Calculate distance from point (x,y,z) to spiral at given spiral_z position"""
        spiral_x, spiral_y = z_to_xy(spiral_z, theta_start, on_surface)
        
        # Return signed distance (negative inside tube)
        dist = np.sqrt((x - spiral_x)**2 + (y - spiral_y)**2 + (z - spiral_z)**2)
        return dist - tube_radius

    def iterative_refinement_minimum(x, y, z, theta_start, z_min, z_max, tolerance=1e-5, max_iters = 10, points_per_iteration=20):
        """Find minimum distance using iterative refinement until convergence"""
        current_z_min = z_min
        current_z_max = z_max
        prev_min_distance = None
        refinement = 0
        
        while refinement < max_iters:
            # Create points in current range
            z_points = np.linspace(current_z_min, current_z_max, points_per_iteration)
            
            # Evaluate distances at all points
            distances = distance_to_spiral_at_z(x, y, z, z_points, theta_start)
            
            # Find minimum
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            # Print difference from previous iteration
            if prev_min_distance is not None:
                diff = abs(min_distance - prev_min_distance)
                #print(f"Refinement {refinement}: min_distance = {min_distance:.6f}, diff from prev = {diff:.6f}")
                
                # Check if we've converged
                if diff < tolerance:
                    #print(f"Converged after {refinement + 1} refinements")
                    return min_distance
                
            prev_min_distance = min_distance
            refinement += 1
            
            if min_idx == 0 or min_idx == len(z_points) - 1:
                return min_distance
            
            # Refine range around minimum
            # Find the points below and above the minimum
            z_below = z_points[min_idx - 1]
            z_above = z_points[min_idx + 1]
            current_z_min = z_below
            current_z_max = z_above
        
        return min_distance

    val = np.inf
    
    for i in range(n_spirals):
        theta_start = 2 * np.pi * i / n_spirals
        
        # Define search range around current z position
        z_search_min = max(z_end, z - 2*tube_radius)
        z_search_max = min(z_start, z + 2*tube_radius)
        
        # Use iterative refinement to find minimum distance
        min_dist = iterative_refinement_minimum(x, y, z, theta_start, z_search_min, z_search_max)

        # if the minimum distance is already close to a tube, return it
        if min_dist < tube_radius:
            return min_dist
        
        val = min(val, min_dist)

    # Signed distance function (negative inside the tube)
    return val

def ellipse(x, y, z, r_xy, r_z):
    # signed distance function
    # Calculate the distance to the ellipsoid surface
    # For a point (x,y,z), the signed distance is:
    # - distance to surface if inside (negative)
    # + distance to surface if outside (positive)
    
    # First, find the closest point on the ellipsoid surface
    # This is done by scaling the point to the surface
    scale = np.sqrt(x**2/r_xy**2 + y**2/r_xy**2 + z**2/r_z**2)
    
    if scale == 0:
        # Point is at origin, distance is -r_xy (inside)
        return -r_xy
    
    # Closest point on surface
    closest_x = x / scale
    closest_y = y / scale
    closest_z = z / scale
    
    # Distance to closest point
    dist = np.sqrt((x - closest_x)**2 + (y - closest_y)**2 + (z - closest_z)**2)
    
    # Return signed distance: negative if inside, positive if outside
    if scale < 1:
        return -dist  # Inside ellipsoid
    else:
        return dist   # Outside ellipsoid

def f(x, y, z, n_spirals=8, helix_angle=30, on_surface=False):
    # Shell function with roof: 0 isosurface is between two ellipses and below z=0, 
    # plus a roof between z=0 and z=-0.8
    # Returns negative values inside the shell, positive outside, zero on shell surface
    r_zo = 30.5
    r_xyo = 13
    r_zi = 29.7
    r_xyi = 12.2
    
    # Define the two ellipsoids
    outer_ellipse = ellipse(x, y, z, r_xyo, r_zo)
    inner_ellipse = ellipse(x, y, z, r_xyi, r_zi)

    #tube cylinder of r= 2mm
    tube = x**2 + y**2 - 1**2
    
    # Define the z planes
    z_top = z  # z=0 plane
    z_bottom = -0.8 - z  # z=-0.8 plane
    
    # Check if we're in the roof region (between z=0 and z=-0.8)
    in_roof_region = (z <= 0) and (z >= -0.8)
    
    if z >= -0.8:
        # In roof region: use the ellipsoid shell logic
        val = max(outer_ellipse, z_top, -tube, z_bottom)
    else:
        # Outside roof region: use original shell logic (below z=0)
        if inner_ellipse <= 0:
            val = min(-inner_ellipse, z_bottom)
            if z > -10: #computing tube below 10 messes up apex
                val = max(val, -tube)
        else:
            val = max(outer_ellipse, -inner_ellipse)

        #to speed up computation, don't compute spiral support for the center
        center_r_xy = 10.5
        center_r_z = 28
        center_ellipse = ellipse(x, y, z, center_r_xy, center_r_z)

        if outer_ellipse < 0 and center_ellipse > 0:
            spiral = spiral_support(x, y, z, n_spirals=n_spirals, helix_angle=helix_angle, on_surface=on_surface)
            val = min(val, spiral)
    
    return val
'''
def f_normal(x, y, z):
    #Compute the normal vector to the surface of the ventricle
    r_zo = 30.5
    r_xyo = 13
    r_zi = 29.7
    r_xyi = 12.2
    
    # Define the two ellipsoids
    outer_ellipse = ellipse(x, y, z, r_xyo, r_zo)
    inner_ellipse = ellipse(x, y, z, r_xyi, r_zi)
    
    # Compute the normal vector
    outer_ellipse_normal = np.array([2*x/r_xyo**2, 2*y/r_xyo**2, 2*z/r_zo**2])
    inner_ellipse_normal = -np.array([2*x/r_xyi**2, 2*y/r_xyi**2, 2*z/r_zi**2])
    tube_normal = np.array([-x, -y, 0])
    roof_normal_up = np.array([0, 0, 1])
    roof_normal_down = np.array([0, 0, -1])
    
    if outer_ellipse < 0:
        return outer_ellipse_normal
    elif inner_ellipse < 0:
'''
def plot_contour(f, x_range, z_range, level, resolution=400, normal=False):
    x = np.linspace(x_range[0], x_range[1], resolution)
    z = np.linspace(z_range[0], z_range[1], resolution)
    if normal:
        F = np.zeros((resolution, resolution, 3))
    else:
        F = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            if normal:
                # f_normal returns a V3 object, extract all components
                normal_vec = f(x[i], 0, z[j])
                F[j,i,0] = normal_vec.x
                F[j,i,1] = normal_vec.y
                F[j,i,2] = normal_vec.z
            else:
                F[j,i] = f(x[i], 0, z[j])
    
    # Handle invalid values
    if normal:
        for k in range(3):
            F[:,:,k] = np.nan_to_num(F[:,:,k], nan=0.0, posinf=1.0, neginf=-1.0)
    else:
        F = np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=-1.0)
    
    if normal:
        # Create three side-by-side plots for normal vector components
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        component_names = ['X', 'Y', 'Z']
        
        for k in range(3):
            ax = axes[k]
            vmin = np.min(F[:,:,k])
            vmax = np.max(F[:,:,k])
            
            # Create a custom colormap: blue for negative, red for positive
            cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap
            
            # Plot filled contours
            contourf = ax.contourf(x, z, F[:,:,k], levels=50, cmap=cmap, alpha=0.7, 
                                  vmin=vmin, vmax=vmax)
            
            # Add the black contour line at the specified level
            ax.contour(x, z, F[:,:,k], levels=[level], colors='black', linewidths=2)
            
            # Set equal aspect ratio to preserve actual proportions
            ax.set_aspect('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title(f'Normal vector {component_names[k]}-component at y=0')
            ax.grid(True, alpha=0.3)
            
            # Set the plot limits to match the input ranges
            ax.set_xlim(x_range)
            ax.set_ylim(z_range)
            
            # Add colorbar
            try:
                plt.colorbar(contourf, ax=ax, label=f'{component_names[k]}-component')
            except:
                print(f"Warning: Could not create colorbar for {component_names[k]}-component")
        
        plt.tight_layout()
        plt.show()
    else:
        # Original single plot for scalar function
        plt.figure(figsize=(10, 8))
        
        # Create a custom colormap: blue for negative, red for positive
        cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap
        
        # Plot filled contours with color coding, centered at 0
        # Find the range of values to center the colormap
        vmin = np.min(F)
        vmax = np.max(F)
        contourf = plt.contourf(x, z, F, levels=50, cmap=cmap, alpha=0.7, 
                               vmin=vmin, vmax=vmax)
        
        # Add the black contour line at the specified level
        plt.contour(x, z, F, levels=[level], colors='black', linewidths=2)
        
        # Set equal aspect ratio to preserve actual proportions
        plt.axis('equal')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.title('Cross-section at y=0 (Blue=negative, Red=positive)')
        plt.grid(True, alpha=0.3)
        
        # Set the plot limits to match the input ranges
        plt.xlim(x_range)
        plt.ylim(z_range)
        
        # Add colorbar with error handling
        try:
            plt.colorbar(contourf, label='Function Value')
        except:
            print("Warning: Could not create colorbar due to data issues")
        
        plt.show()

def set_group_ids(mesh):
    '''
    Set groupIds for each face
    #GroupId 1: inner wall - Surface of inner ellipsoid except for the spiral support
    #GroupId 2: outer wall - Surface of outer ellipsoid
    #GroupId 3: tube - Surface of tube
    #GroupId 4: roof - Surface of roof
    #GroupId 5: spiral support - Surface of spiral support
    '''
    # Get face centers and normals
    faces = mesh.faces
    face_centers = mesh.cell_centers()
    normals = mesh.cell_normals
   # face_normals = mesh.cell_normals()
    
    # Initialize GroupIds array
    group_ids = np.zeros(mesh.n_cells, dtype=int)
    
    # Parameters for classification
    r_zo = 30.5
    r_xyo = 13
    r_zi = 29.7
    r_xyi = 12.2
    tube_radius = 1.0

    tolerance = 5e-2
    # Print max and min face lengths
    
    
    for i in range(mesh.n_cells):
        center = face_centers.points[i]
        #normal = face_normals.points[i]
        x, y, z = center
        
        # Calculate distances to different surfaces
        outer_dist = ellipse(x, y, z, r_xyo, r_zo)
        inner_dist = ellipse(x, y, z, r_xyi, r_zi)
        mid_dist = ellipse(x, y, z, (r_xyo+r_xyi)/2, (r_zo+r_zi)/2)
        bottom = z+0.4
        endo_dist = max(mid_dist, bottom)
        tube_dist = np.sqrt(x**2 + y**2) - tube_radius
        
        # Check if roof (z = 0)
        in_roof_region = abs(z) < tolerance and outer_dist < tolerance
        
        # Check if near spiral support region
        center_r_xy = 10.5
        center_r_z = 28
        center_ellipse = ellipse(x, y, z, center_r_xy, center_r_z)
        in_spiral_region = (inner_dist < 0) and (center_ellipse > 0)
        
        # Classify based on proximity to different surfaces  

        if endo_dist < 0:
            # Inner wall (not spiral support)
            if tube_dist < tolerance and abs(normals[i][2]) < 0.7:
                group_ids[i] = 3
            else:
                group_ids[i] = 1
        elif endo_dist >= 0:
            if tube_dist < tolerance and abs(normals[i][2]) < 0.7:
                group_ids[i] = 3
            else:
                group_ids[i] = 2
        '''       
        elif abs(outer_dist) > tolerance and abs(z+0.8) < tolerance and normals[i][2] > 0:
            # Inner wall top surface
            group_ids[i] = 1
        elif z > -0.4 and normals[i][2] <= -0.3:  
            # Roof region
            group_ids[i] = 2       
        elif abs(outer_dist) < tolerance*1e-1:
            # Outer wall
            group_ids[i] = 2
        elif tube_dist < tolerance and z> -1.0 and abs(normals[i][2]) < 0.3:
            # Tube surface
            group_ids[i] = 3
        elif inner_dist < tolerance and z < -0.8:
            # Spiral support
            group_ids[i] = 1
        else:
            if inner_dist < tolerance*10 and outer_dist < tolerance*10 and tube_dist < tolerance*10:
                # assign to smallest distance
                if abs(inner_dist) < abs(outer_dist) and abs(inner_dist) < abs(tube_dist):
                    group_ids[i] = 1
                elif abs(outer_dist) < abs(inner_dist) and abs(outer_dist) < abs(tube_dist):
                    group_ids[i] = 2
                else:
                    group_ids[i] = 3
            elif z > -0.5:
                group_ids[i] = 2
            elif z < -0.5 and z > -0.9:
                group_ids[i] = 1
            else:
                group_ids[i] = 1
            '''

    # Plot group_ids with z on the x-axis and z normal on the y-axis. color by group_ids
    '''plt.figure()
    centers = face_centers.points
    zs = centers[:,2]
    normals = mesh.cell_normals
    z_normals = normals[:,2]
    plt.scatter(zs, z_normals, c=group_ids)
    plt.show()
    '''

    # Add GroupIds to mesh
    mesh.cell_data['GroupIds'] = group_ids
    
    # Print statistics
    unique_ids, counts = np.unique(group_ids, return_counts=True)
    print("Group ID distribution:")
    for group_id, count in zip(unique_ids, counts):
        group_names = {0: "Unassigned", 1: "Inner wall", 2: "Outer wall", 3: "Tube", 4: "Roof", 5: "Spiral support"}
        print(f"  Group {group_id} ({group_names.get(group_id, 'Unknown')}): {count} faces")
    
    return mesh

def sample_model_face_id(source_mesh, target_mesh, field_name='ModelFaceID'):
    """
    Sample ModelFaceID from source mesh to target mesh using nearest neighbor interpolation
    
    Parameters:
    -----------
    source_mesh : pyvista.PolyData
        Source mesh with ModelFaceID data
    target_mesh : pyvista.PolyData
        Target mesh to receive the sampled data
    field_name : str
        Name of the field to sample (default: 'ModelFaceID')
    
    Returns:
    --------
    target_mesh : pyvista.PolyData
        Target mesh with sampled ModelFaceID data
    """
    print(f"Sampling {field_name} from source mesh ({source_mesh.n_cells} cells) to target mesh ({target_mesh.n_cells} cells)...")
    
    # Check if source mesh has the required field
    if field_name not in source_mesh.cell_data:
        print(f"Warning: {field_name} not found in source mesh cell data")
        print(f"Available cell data fields: {list(source_mesh.cell_data.keys())}")
        return target_mesh
    
    # Get source data
    source_data = source_mesh.cell_data[field_name]
    
    # Get cell centers for both meshes
    source_centers = source_mesh.cell_centers()
    target_centers = target_mesh.cell_centers()
    
    # Use nearest neighbor interpolation
    from scipy.spatial import cKDTree
    
    # Build KD-tree from source centers
    tree = cKDTree(source_centers.points)
    
    # Find nearest neighbors for target centers
    distances, indices = tree.query(target_centers.points, k=1)
    
    # Sample the data
    sampled_data = source_data[indices]
    
    # Add the sampled data to target mesh
    target_mesh.cell_data[field_name] = sampled_data
    
    # Print statistics
    unique_values, counts = np.unique(sampled_data, return_counts=True)
    print(f"Sampled {field_name} distribution:")
    for value, count in zip(unique_values, counts):
        print(f"  {field_name} {value}: {count} cells")
    
    return target_mesh

def create_volume_mesh(surface_mesh, max_edge_size, output_file):
    """
    Create a volume mesh from surface mesh with specified max edge size
    
    Parameters:
    -----------
    surface_mesh : pyvista.PolyData
        Surface mesh to create volume mesh from
    max_edge_size : float
        Maximum edge size for the volume mesh
    output_file : str
        Output file path for the .vtu file
    
    Returns:
    --------
    volume_mesh : pyvista.UnstructuredGrid
        The created volume mesh
    """
    print(f"Creating volume mesh with max edge size: {max_edge_size}")
    print(f"Surface mesh has {surface_mesh.n_cells} cells and {surface_mesh.n_points} points")
    
    try:
        # Use PyVista's tetgen wrapper to create volume mesh
        volume_mesh = surface_mesh.tetrahedralize(max_edge_size=max_edge_size)
        
        print(f"Volume mesh created with {volume_mesh.n_cells} cells and {volume_mesh.n_points} points")
        
        # Save as .vtu file
        print(f"Saving volume mesh to: {output_file}")
        volume_mesh.save(output_file)
        
        # Print mesh quality statistics
        print("Volume mesh statistics:")
        print(f"  Number of cells: {volume_mesh.n_cells}")
        print(f"  Number of points: {volume_mesh.n_points}")
        print(f"  Cell types: {volume_mesh.celltypes}")
        
        return volume_mesh
        
    except Exception as e:
        print(f"Error creating volume mesh: {e}")
        print("Trying alternative method with gmsh...")
        
        try:
            # Alternative method using gmsh if available
            import gmsh
            gmsh.initialize()
            
            # Convert PyVista mesh to gmsh format
            # This is a simplified approach - you might need to adjust based on your gmsh setup
            print("Gmsh method not fully implemented. Please install tetgen or use alternative meshing tools.")
            return None
            
        except ImportError:
            print("Gmsh not available. Please install tetgen or gmsh for volume mesh generation.")
            return None

def process_case(case):
    mesh_basename, n_spirals, helix_angle, resolution, on_surface = case
    print(f"Processing case: {mesh_basename}")
    def f_wrapper(x, y, z):
        return f(x, y, z, n_spirals=n_spirals, helix_angle=helix_angle, on_surface=on_surface)
                # Extract the 0-isosurface using sequential processing
                #print("Starting sequential marching cubes...")

    x_range = np.linspace(-13.5, 13.5, resolution)
    y_range = np.linspace(-13.5, 13.5, resolution)
    z_range = np.linspace(-31.5, 0.5, resolution)
    SDF_array = np.zeros((resolution, resolution, resolution))
    for i in range(resolution):
        print(f"Processing row {i} of {resolution}")
        for j in range(resolution):
            for k in range(resolution):
                x = x_range[i]
                y = y_range[j]
                z = z_range[k]
                SDF_array[i, j, k] = f_wrapper(x, y, z)


    vertices, triangles = mcubes.marching_cubes_func(
        (-13.5,-13.5,-31), (13.5,13.5,0.5),
        resolution, resolution, resolution, f_wrapper, 0
    )

    # Export to OBJ file
    #print("Exporting to OBJ file...")
    mcubes.export_obj(vertices, triangles, 'meshes/all_cases/' + mesh_basename + '.obj', flip_normals=True)
    #print("Mesh generation complete!")

def main():
    # ===== FLAGS =====
    compute_mesh_flag = True
    compute_group_ids_flag = False
    show_contour_flag = False
    sample_model_face_id_flag = False
    create_volume_mesh_flag = False
    
    # ===== PARAMETERS =====
    mesh_generator = 'MarchingCubes' # MarchingCubes or DualContour:
    # MarchingCubes: Fast, doesn't fit normals, fewer intersecting faces
    # DualContour: Slow, fits normals, many intersecting faces
    resolution = 200
    
    # Spiral support parameters
    n_spirals = [8, 9, 10]
    helix_angle = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]  # degrees
    on_surface = [True]
    cases = []
    for n in n_spirals:
        for h in helix_angle:
            for o in on_surface:
                if o:
                    basename = f'ventricle_{n}_{h}_ellipsoidal'
                else:
                    basename = f'ventricle_{n}_{h}_cartesian'
                cases.append((basename, n, h, resolution, o))

    # For DualContour
    XMIN = -13.5
    XMAX = 13.5
    YMIN = -13.5
    YMAX = 13.5
    ZMIN = -31
    ZMAX = 0.5
    CELL_SIZE = (XMAX - XMIN) / resolution
    EPS = 1e-8
    ADAPTIVE = True

    # In dual contouring, if true, crudely force the selected vertex to belong in the cell
    CLIP = False
    # In dual contouring, if true, apply boundaries to the minimization process finding the vertex for each cell
    BOUNDARY = True
    # In dual contouring, if true, apply extra penalties to encourage the vertex to stay within the cell
    BIAS = True
    # Strength of the above bias, relative to 1.0 strength for the input gradients
    BIAS_STRENGTH = 0.01
    
    n_processes = 1
    decimation_factor = 0.95
    max_edge_size = 0.5  # Maximum edge size for volume mesh
    out_dir = 'meshes'
    if mesh_generator == 'MarchingCubes':
        mesh_basename = 'ventricle_MC_' + str(resolution) + '_' + str(n_spirals) + '_' + str(helix_angle)
    elif mesh_generator == 'DualContour':
        mesh_basename = 'ventricle_DC_' + str(resolution) + '_' + str(n_spirals) + '_' + str(helix_angle)
    
    # ===== MAIN ROUTINE =====

    # Parallelize over cases
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_case, case
            )
            for case in cases
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'Case generated an exception: {exc}')
    
    # Load mesh
    print("Loading mesh...")
    mesh = pv.read('meshes/' + mesh_basename + '.obj')
    # flip normals
    #mesh.flip_normals()
    
    # Set up output file paths
    output_file_name = os.path.join(out_dir, mesh_basename + '.vtp')
    output_file_name_decimated = os.path.join(out_dir, mesh_basename + '_decimated.vtp')
    
    # Process GroupIds if requested
    if compute_group_ids_flag:
        print("Computing GroupIds...")
        # Apply GroupIds to the mesh
        mesh = set_group_ids(mesh)
        # Save mesh with GroupIds
        mesh.save(os.path.join(out_dir, mesh_basename + '.obj'))

        # Visualize with GroupIds coloring
        plotter = pv.Plotter()
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for group_id in range(6):
            mask = mesh.cell_data['GroupIds'] == group_id
            if np.any(mask):
                subset = mesh.extract_cells(mask)
                plotter.add_mesh(subset, color=colors[group_id], label=f'Group {group_id}')
        plotter.add_legend()
        plotter.show()

        # -------- Check for unintended edges --------
        #mask = mesh.edge_mask(90)
        #print(f'Number of edges: {mask.sum()}')
        #mesh.plot(show_edges=False, scalars=mask)

        # -------- Create ModelFaceID array --------
        # ModelFaceID array is created based on GroupsIds array
        group_id_name = 'GroupIds'
        mesh_vtp = convert_obj_to_vtp(mesh, output_file_name, group_id_name)
    else:
        # Load existing VTP file
        print(f"Loading existing VTP file: {output_file_name}")
        mesh_vtp = pv.read(output_file_name)

    # Clean mesh
    faces = mesh_vtp.faces.reshape((-1, 4))[:, 1:]
    vertices = mesh_vtp.points

    mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # --- Report initial mesh status
    print("Initial mesh:")
    print(f"  Vertices: {len(mesh_tri.vertices)}")
    print(f"  Faces: {len(mesh_tri.faces)}")
    print(f"  Is watertight? {mesh_tri.is_watertight}")
    #print(f"  Self-intersections: {len(mesh_tri.face_adjacency_self_intersecting)}")

    # --- Optional: Print details of self-intersecting face pairs
    #for f1, f2 in mesh_tri.face_adjacency_self_intersecting:
    #    print(f"Self-intersecting pair: {f1} <-> {f2}")

    # --- Clean the mesh
    mesh_tri.remove_duplicate_faces()
    mesh_tri.remove_degenerate_faces()
    mesh_tri.remove_unreferenced_vertices()
    mesh_tri.process(validate=True)

    # Save cleaned mesh
    output_file_name_cleaned = os.path.join(out_dir, mesh_basename + '_cleaned.vtp')
    print(f"Saving cleaned mesh to: {output_file_name_cleaned}")
    mesh_tri.export(output_file_name_cleaned)

    # Count number of free edges
    print("Counting number of free edges...")
    free_edges = mesh_vtp.extract_feature_edges(boundary_edges=False, manifold_edges=False)
    print(f"Number of free edges: {free_edges.n_cells}")
        
    # Visualize sampled ModelFaceID if requested
    if sample_model_face_id_flag and 'ModelFaceID' in mesh_vtp.cell_data:
        print("Visualizing sampled ModelFaceID on decimated mesh...")
        plotter = pv.Plotter()
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        model_face_ids = mesh_vtp.cell_data['ModelFaceID']
        unique_ids = np.unique(model_face_ids)
        
        for i, group_id in enumerate(unique_ids):
            mask = model_face_ids == group_id
            if np.any(mask):
                subset = mesh_vtp.extract_cells(mask)
                color = colors[i % len(colors)]
                plotter.add_mesh(subset, color=color, label=f'ModelFaceID {group_id}')
        
        plotter.add_legend()
        plotter.show()
    
    print("Processing complete!")

if __name__ == "__main__":
    main()






