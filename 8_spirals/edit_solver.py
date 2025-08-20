import argparse
import os
import xml.etree.ElementTree as ET


def update_solver_xml(xml_path, mesh_folder, time_step_size, num_time_steps, spiral_stiffness=None, roof_stiffness=None, pressure_file=None, output_path=None):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    mesh_str = mesh_folder.split("/")[-1].split("-")[0]
    results_folder = "results_" + mesh_str

    if pressure_file is not None:
        pressure_str = 'p' + pressure_file.split("_")[-1]
        pressure_str = pressure_str.replace('.dat', '')
        results_folder = results_folder + "_" + pressure_str

    # Update GeneralSimulationParameters
    gsp = root.find('GeneralSimulationParameters')
    if gsp is not None:
        for child in gsp:
            if child.tag == 'Time_step_size':
                child.text = str(time_step_size)
            elif child.tag == 'Number_of_time_steps':
                child.text = str(num_time_steps)
            elif child.tag == 'Save_results_in_folder':
                child.text = results_folder

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

    # Update equation parameters and domain stiffness
    equation_tag = root.find('Add_equation')
    if equation_tag is not None:
        # Update domain stiffness if provided
        for domain in equation_tag.findall('Domain'):
            domain_id = domain.attrib.get('id')
            if domain_id == "2" and roof_stiffness is not None:
                # Update roof stiffness (Domain id="2")
                elasticity_mod = domain.find('Elasticity_modulus')
                if elasticity_mod is not None:
                    elasticity_mod.text = str(roof_stiffness)
                    print(f'Updated roof (Domain id="2") Elasticity_modulus to: {roof_stiffness}')
            elif domain_id == "3" and spiral_stiffness is not None:
                # Update spiral stiffness (Domain id="3")
                elasticity_mod = domain.find('Elasticity_modulus')
                if elasticity_mod is not None:
                    elasticity_mod.text = str(spiral_stiffness)
                    print(f'Updated spiral (Domain id="3") Elasticity_modulus to: {spiral_stiffness}')
        
        # Update pressure file path if provided
        if pressure_file is not None:
            for bc in equation_tag.findall('Add_BC'):
                if bc.attrib.get('name') == 'endo':
                    temporal_values = bc.find('Temporal_values_file_path')
                    if temporal_values is not None:
                        temporal_values.text = pressure_file
                        print(f'Updated pressure file path to: {pressure_file}')

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
    parser.add_argument('--spiral', type=float, help='Elasticity modulus for spiral domain (Domain id="3")')
    parser.add_argument('--roof', type=float, help='Elasticity modulus for roof domain (Domain id="2")')
    parser.add_argument('--pressure-file', help='Path to pressure.dat file (default: pressure.dat)')
    parser.add_argument('--output', '-o', help='Output XML file (default: overwrite input)')
    args = parser.parse_args()

    update_solver_xml(args.xml_file, args.mesh_folder, args.Time_step_size, args.Number_of_time_steps, 
                     args.spiral, args.roof, args.pressure_file, args.output)

if __name__ == '__main__':
    main()
