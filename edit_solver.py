import argparse
import os
import xml.etree.ElementTree as ET


def update_solver_xml(xml_path, mesh_folder, time_step_size, num_time_steps, output_path=None):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update GeneralSimulationParameters
    gsp = root.find('GeneralSimulationParameters')
    if gsp is not None:
        for child in gsp:
            if child.tag == 'Time_step_size':
                child.text = str(time_step_size)
            elif child.tag == 'Number_of_time_steps':
                child.text = str(num_time_steps)

    # Update mesh and face paths
    mesh_tag = root.find('Add_mesh')
    if mesh_tag is not None:
        # Main mesh
        mesh_file = os.path.join(mesh_folder, 'mesh-complete.mesh.vtu')
        mesh_tag.find('Mesh_file_path').text = mesh_file
        # Domain file
        domain_file = os.path.join(mesh_folder, 'domain_ids.dat')
        mesh_tag.find('Domain_file_path').text = domain_file
        # Faces
        for face in mesh_tag.findall('Add_face'):
            face_name = face.attrib.get('name')
            face_file = os.path.join(mesh_folder, 'mesh-surfaces', f'{face_name}.vtp')
            face.find('Face_file_path').text = face_file

    # Save the updated XML
    if output_path is None:
        output_path = xml_path
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f'Updated solver XML saved to: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Edit solver.xml with new mesh folder and simulation parameters.')
    parser.add_argument('xml_file', help='Path to the solver.xml file to edit')
    parser.add_argument('mesh_folder', help='Path to the mesh folder (e.g. meshes/ventricle_MC_200_8_30_5d5mm-mesh-complete)')
    parser.add_argument('Time_step_size', type=float, help='Time step size (e.g. 1e-3)')
    parser.add_argument('Number_of_time_steps', type=int, help='Number of time steps (e.g. 30)')
    parser.add_argument('--output', '-o', help='Output XML file (default: overwrite input)')
    args = parser.parse_args()

    update_solver_xml(args.xml_file, args.mesh_folder, args.Time_step_size, args.Number_of_time_steps, args.output)

if __name__ == '__main__':
    main()
