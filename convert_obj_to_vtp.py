'''
Given a tagged model in .obj format, performs the following
- Check for unintended boundaries in the mesh
- Renames tags with ModelFaceID array
- Converts the mesh to .vtp format
'''

import pyvista as pv
import numpy as np
import os
import vtk

def convert_obj_to_vtp(surface, output_file_name, group_ids_name='GroupIds', write_flag=True):
    '''Convert a surface model defined in a .obj file into a .vtp file.

    The faces grouped by ID are converted into the 'ModelFaceID' cell array needed 
    by SimVascular when importing a solid model.

    Somehow, this outputs the mesh in a format that can be read by SimVascular.
    Other file write approaches, like PyVista .save(), cause SimVascular to crash
    when trying to load the model.

    Parameters:
    input_file_name (str): The path to the .obj file to be converted.
    output_file_name (str): The path for the output .vtp file.
    group_ids_name (str): The name of the array in the .obj file that contains the
        group IDs for the faces.
    '''

    num_cells = surface.GetNumberOfCells()

    # Find face group IDs.
    group_ids = surface.GetCellData().GetArray(group_ids_name)
    group_ids_range = 2*[0]
    group_ids.GetRange(group_ids_range, 0)
    min_id = int(group_ids_range[0])
    max_id = int(group_ids_range[1])
    print("Face IDs range: {0:d} {1:d}".format(min_id, max_id))

    # Add ModelFaceID array.
    face_ids = vtk.vtkIntArray()
    face_ids.SetNumberOfValues(num_cells)
    face_ids.SetName('ModelFaceID')
    for i in range(num_cells):
        face_id = int(group_ids.GetValue(i)) + 1
        face_ids.SetValue(i, face_id)

    surface.GetCellData().AddArray(face_ids)
    surface.Modified()
    surface.GetCellData().RemoveArray(group_ids_name)

    if write_flag:
    # Write vtp file.
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(output_file_name)
        writer.SetInputData(surface)
        writer.Write()
    return surface

if __name__ == "__main__":
    # Input mesh
    mesh_path = '/Users/dseyler/Documents/Marsden_Lab/Ventricle_optimization/meshes/ventricle_DualContour_100_meshmixer.obj'

    # Output directory
    out_dir = os.path.dirname(mesh_path)

    # Get basename of mesh
    mesh_basename = os.path.basename(mesh_path).split('.')[0]

    # Load mesh
    mesh = pv.read(mesh_path)

    # -------- Check for unintended edges --------
    mask = mesh.edge_mask(90)
    print(f'Number of edges: {mask.sum()}')
    mesh.plot(show_edges=False, scalars=mask)


    # -------- Create ModelFaceID array --------
    # ModelFaceID array is created based on GroupsIds array
    group_id_name = 'GroupIds'
    output_file_name = os.path.join(out_dir, mesh_basename + '.vtp')
    surface = pv.read(mesh_path)
    convert_obj_to_vtp(surface, output_file_name, group_id_name)


