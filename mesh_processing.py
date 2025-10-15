#!/usr/bin/env python3
"""
Mesh processing script using PyMeshLab for remeshing with specified edge size.
"""

import pymeshlab
import pyvista as pv
import numpy as np
import os
import argparse
from generate_ventricle import set_group_ids
from convert_obj_to_vtp import convert_obj_to_vtp
#import tetgen

def remesh_with_pymeshlab(input_obj, output_dir, max_edge_length=0.2):
    """
    Remesh a surface mesh using PyMeshLab with specified max edge length.
    After remeshing, compute GroupIds and convert to VTP.
    """
    print(f"Loading mesh from: {input_obj}")
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_obj)
    print(f"Original mesh: {ms.current_mesh().vertex_number()} vertices, {ms.current_mesh().face_number()} faces")
    print(f"Remeshing with max edge length: {max_edge_length}")
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PureValue(max_edge_length),
        featuredeg=50,
        iterations=10,

    )
    print("Repairing mesh to ensure manifoldness and watertightness...")
    ms.meshing_repair_non_manifold_edges()
    #ms.meshing_close_holes(maxholesize=100)
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_repair_non_manifold_vertices()
    print(f"Remeshed mesh: {ms.current_mesh().vertex_number()} vertices, {ms.current_mesh().face_number()} faces")

    os.makedirs(output_dir, exist_ok=True)
    output_obj = os.path.join(output_dir, 'mesh-complete.obj')
    ms.save_current_mesh(output_obj)
    print(f"Saved remeshed mesh to: {output_obj}")

    # --- Compute GroupIds and convert to VTP ---
    print("Loading remeshed OBJ with PyVista...")
    mesh = pv.read(output_obj)
    print("Computing GroupIds...")
    mesh = set_group_ids(mesh)
    print("Converting OBJ to VTP...")
    vtp_path = os.path.join(output_dir, 'mesh-complete.exterior.vtp') #os.path.splitext(output_obj)[0] + '.vtp'
    group_id_name = 'GroupIds'
    mesh_vtp = convert_obj_to_vtp(mesh, vtp_path, group_id_name)
    print(f"Saved VTP mesh to: {vtp_path}")
    return mesh_vtp, vtp_path

def tetrahedralize(surf, output_dir, max_volume=0.025):
    """
    Tetrahedralize a surface mesh using TetGen and save as .vtu file.
    Args:
        input_surface (str): Path to surface mesh (OBJ, STL, VTP, etc)
        output_vtu (str): Output .vtu file path
        max_volume (float, optional): Maximum tetrahedron volume for TetGen
    """
    import tetgen

    print(f"Surface mesh: {surf.n_points} points, {surf.n_faces} faces")
    # Convert to TetGen input
    tgen = tetgen.TetGen(surf)
    print("Running TetGen...")
    tgen.make_manifold()
    tgen.tetrahedralize(order=1, mindihedral=10.0, minratio=1.4, maxvolume = max_volume, plc=True,
                    facet_separate_ang_tol=180.0,
                    facet_small_ang_tol=0.0,
                    collinear_ang_tol=180.0, refine=0.0, coarsen=1.0, coarsen_percent=0.5)
                    #quality=False,
                    #refine=0.0,
                    #coarsen=1.0,
                    #coarsen_percent=0.5)
    grid = tgen.grid
    # Assign GlobalNodeID and GlobalElementID
    grid.point_data['GlobalNodeID'] = np.arange(grid.n_points)
    grid.cell_data['GlobalElementID'] = np.arange(grid.n_cells)
    print(f"Tetrahedral mesh: {grid.n_points} points, {grid.n_cells} cells")
    output_vtu = os.path.join(output_dir, 'mesh-complete.mesh.vtu')
    print(f"Saving tetrahedral mesh to: {output_vtu}")
    grid.save(output_vtu)
    print("Done.")

def split_faces(case_dir):
    '''Load vtu and vtp meshes. Assign global node ids to vtu mesh and corresponding nodes on vtp mesh.'''
    vtu_mesh = pv.read(os.path.join(case_dir, 'mesh-complete.mesh.vtu'))
    vtp_mesh = pv.read(os.path.join(case_dir, 'mesh-complete.exterior.vtp'))

    # Split vtp into endo, epi, and tube
    # ModelFaceID is off by 1 for some reason
    endo_mesh = vtp_mesh[vtp_mesh.cell_data['ModelFaceID'] == 2]
    epi_mesh = vtp_mesh[vtp_mesh.cell_data['ModelFaceID'] == 3]
    tube_mesh = vtp_mesh[vtp_mesh.cell_data['ModelFaceID'] == 4]
    
    # Sample vtu GlobalNodeIds at vtp points
    endo_mesh = endo_mesh.sample(vtu_mesh)
    epi_mesh = epi_mesh.sample(vtu_mesh)
    tube_mesh = tube_mesh.sample(vtu_mesh)

    # Add ModelFaceID back to vtp meshes
    endo_mesh.cell_data['ModelFaceID'] = 2
    epi_mesh.cell_data['ModelFaceID'] = 3
    tube_mesh.cell_data['ModelFaceID'] = 4

    face_dir = os.path.join(case_dir, 'mesh-surfaces')
    os.makedirs(face_dir, exist_ok=True)

    # Save vtu and vtp meshes
    endo_mesh.save(os.path.join(face_dir, 'endo.vtp'))
    epi_mesh.save(os.path.join(face_dir, 'epi.vtp'))
    tube_mesh.save(os.path.join(face_dir, 'tube.vtp'))

def plot_mesh_comparison(original_obj, remeshed_obj):
    """
    Plot original and remeshed meshes side by side for comparison.
    
    Args:
        original_obj (str): Path to original .obj file
        remeshed_obj (str): Path to remeshed .obj file
    """
    # Load meshes
    original_mesh = pv.read(original_obj)
    remeshed_mesh = pv.read(remeshed_obj)
    
    # Create plotter
    plotter = pv.Plotter(shape=(1, 2))
    
    # Plot original mesh
    plotter.subplot(0, 0)
    plotter.add_mesh(original_mesh, color='lightblue', show_edges=True, edge_color='black')
    plotter.add_title(f'Original Mesh\n{original_mesh.n_points} vertices, {original_mesh.n_cells} faces')
    
    # Plot remeshed mesh
    plotter.subplot(0, 1)
    plotter.add_mesh(remeshed_mesh, color='lightgreen', show_edges=True, edge_color='black')
    plotter.add_title(f'Remeshed Mesh\n{remeshed_mesh.n_points} vertices, {remeshed_mesh.n_cells} faces')
    
    plotter.show()

def plot_single_mesh(obj_file, title="Mesh"):
    """
    Plot a single mesh.
    
    Args:
        obj_file (str): Path to .obj file
        title (str): Title for the plot
    """
    mesh = pv.read(obj_file)
    
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', show_edges=True, edge_color='black')
    plotter.add_title(f'{title}\n{mesh.n_points} vertices, {mesh.n_cells} faces')
    plotter.show()

def process_directory(input_dir, output_dir, max_edge_length=0.2, plot_results=False, tetgen=False, tetgen_max_volume=0.025):
    """
    Process all .obj files in a directory and save remeshed results to output directory.
    
    Args:
        input_dir (str): Directory containing input .obj files
        output_dir (str): Directory to save remeshed .obj files
        max_edge_length (float): Maximum edge length for remeshing
        plot_results (bool): Whether to plot results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .obj files in input directory
    obj_files = [f for f in os.listdir(input_dir) if f.endswith('.obj')]
    
    if not obj_files:
        print(f"No .obj files found in {input_dir}")
        return
    
    print(f"Found {len(obj_files)} .obj files to process")
    
    # Process each file
    for i, obj_file in enumerate(obj_files, 1):
        input_path = os.path.join(input_dir, obj_file)
        case_dir = os.path.join(output_dir, os.path.splitext(obj_file)[0] + '-mesh-complete')
        os.makedirs(case_dir, exist_ok=True)
        
        print(f"\nProcessing {i}/{len(obj_files)}: {obj_file}")
        
        try:
            mesh_vtp, vtp_path = remesh_with_pymeshlab(input_path, case_dir, max_edge_length)
            if tetgen:
                tetrahedralize(mesh_vtp, case_dir, max_volume=tetgen_max_volume)
                split_faces(case_dir)
            
            if plot_results:
                # Plot the remeshed result
                plot_single_mesh(vtp_path, f"Remeshed: {obj_file}")
                
        except Exception as e:
            print(f"Error processing {obj_file}: {e}")
            continue
    
    print(f"\nProcessing complete! Remeshed files saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Remesh surface meshes using PyMeshLab and optionally tetrahedralize with TetGen.')
    parser.add_argument('input', help='Path to input .obj file or directory')
    parser.add_argument('--output', '-o', help='Output .obj file path or directory')
    parser.add_argument('--max-edge-length', '-m', type=float, default=0.2, 
                       help='Maximum edge length for remeshing (default: 0.2)')
    parser.add_argument('--plot-comparison', '-c', action='store_true',
                       help='Plot original and remeshed meshes side by side')
    parser.add_argument('--plot-only', '-p', action='store_true',
                       help='Only plot the mesh without remeshing')
    parser.add_argument('--plot-results', action='store_true',
                       help='Plot remeshed results when processing directory')
    parser.add_argument('--tetgen', action='store_true',
                       help='Tetrahedralize the input surface and save as .vtu (requires tetgen)')
    parser.add_argument('--tetgen-max-volume', type=float, default=0.025,
                       help='Maximum volume for TetGen (default: 0.025)')
    args = parser.parse_args()

    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single file processing
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' not found.")
            return
        
        # Set output filename if not provided
        if args.output is None:
            base_name = os.path.splitext(args.input)[0]
            args.output = f"{base_name}-mesh-complete"
        
        if args.plot_only:
            # Just plot the original mesh
            plot_single_mesh(args.input, "Original Mesh")
        else:
            # Remesh the mesh
            try:
                mesh_vtp, vtp_path = remesh_with_pymeshlab(args.input, args.output, args.max_edge_length)
                if args.tetgen:
                    case_dir = args.output
                    tetrahedralize(mesh_vtp, case_dir, max_volume=args.tetgen_max_volume)
                    split_faces(case_dir)
                
                if args.plot_comparison:
                    # Plot both original and remeshed
                    plot_mesh_comparison(args.input, mesh_vtp)
                else:
                    # Plot only the remeshed mesh
                    plot_single_mesh(mesh_vtp, "Remeshed Mesh")
                    
            except Exception as e:
                print(f"Error during remeshing: {e}")
                return
                
    elif os.path.isdir(args.input):
        # Directory processing
        if args.output is None:
            args.output = args.input + "_remeshed"
        
        process_directory(args.input, args.output, args.max_edge_length, args.plot_results, args.tetgen, args.tetgen_max_volume)
        
    else:
        print(f"Error: Input '{args.input}' is neither a file nor a directory.")
        return

if __name__ == "__main__":
    main()
