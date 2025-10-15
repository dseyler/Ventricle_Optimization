import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from generate_ventricle import ellipse, spiral_support, circumferential_support
import os
import argparse

'''
This script generates a .dat file with the domain ids for a synthetic ventricle.
Domain_id = 1: shell
Domain_id = 2: roof
Domain_id = 3: support (spiral or circumferential)
'''

roof_thickness = 2.0

def generate_domain_ids(filename):
    '''
    This function generates a .dat file with the domain ids for a synthetic ventricle.
    Domain_id = 1: shell
    Domain_id = 2: roof
    Domain_id = 3: support (spiral or circumferential)
    '''

    # Load .vtu mesh of the ventricle
    mesh = pv.read(filename)

    # Extract parameters from parent folder name
    file_string = filename.split('/')[-2]
    try:
        parts = file_string.split('_')
        # Expected patterns: ventricle_<n>_<angle>_<tag>
        # where <tag> in {c, e, cartesian, ellipsoidal}
        if len(parts) >= 4:
            _, n_spirals, helix_angle, surf_tag = parts[:4]
        else:
            raise ValueError('Unexpected folder name format')
        n_spirals = int(n_spirals)
        helix_angle = int(helix_angle)
        surf_tag = surf_tag.lower()
        on_surface = surf_tag.startswith('e')  # 'e' or 'ellipsoidal' => True; 'c' or 'cartesian' => False
    except Exception as e:
        raise ValueError(f'Could not parse parameters from folder name: {file_string}') from e
    print(f'n_spirals: {n_spirals}\nhelix_angle: {helix_angle}\non_surface: {on_surface}')

    # get GlobalElementIds:
    global_element_ids = mesh.cell_data['GlobalElementID']
    print(min(global_element_ids), max(global_element_ids))
    domain_ids = np.zeros(len(global_element_ids), dtype=np.int32)

    r_zi = 29.7
    r_xyi = 12.2

    # if cell center is above z = 0.8, set domain_id = 2
    for id in global_element_ids:
        if id % 10000 == 0:  # print every 1000 elements
            print(id, 'of ', len(global_element_ids))
            #print domain_id count
            print(np.unique(domain_ids, return_counts=True))
        #extract element with global_element_ids == id
        element = mesh.extract_cells(global_element_ids == id)
        #assert len(element) == 1, "Multiple elements found for global_element_ids = " + str(id)
        center = element.center
        if center[2] >= -roof_thickness:
            domain_ids[id-1] = 2
        elif ellipse(center[0], center[1], center[2], r_xyi, r_zi) >= 0.05:
            domain_ids[id-1] = 1
        else:
            ellipse_distance = -ellipse(center[0], center[1], center[2], r_xyi, r_zi)
            spiral_support_distance = spiral_support(center[0], center[1], center[2], n_spirals=n_spirals, helix_angle=helix_angle, on_surface=on_surface)
            #circumferential_support_distance = circumferential_support(center[0], center[1], center[2])
            support_distance = spiral_support_distance #min(spiral_support_distance)#, circumferential_support_distance)
            bottom_distance = -roof_thickness - center[2]
            # Assign support (Domain 3) if either support is the closest structure
            if support_distance < ellipse_distance + 0.1 and support_distance < bottom_distance:
                domain_ids[id-1] = 3
            elif ellipse_distance < bottom_distance:
                domain_ids[id-1] = 1
            else:
                domain_ids[id-1] = 2
    # assert that no domain ids are 0
    # return the number of domain ids that are 0
    if np.sum(domain_ids == 0) > 0:
        print(f'Warning: {np.sum(domain_ids == 0)} unassigned ids are 0 for {filename}')

    # save domain_ids to .dat file
    domain_ids_file = os.path.join(os.path.dirname(filename), 'domain_ids.dat')
    print(f'Saving domain_ids to {domain_ids_file}')
    np.savetxt(domain_ids_file, domain_ids, fmt='%d')

    # add domain_ids to mesh
    mesh.cell_data['DOMAIN_ID'] = domain_ids
    # save mesh to .vtu file
    mesh.save(filename)

def find_vtu_files(folder):
    vtu_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.vtu'):
                vtu_files.append(os.path.join(root, file))
    return vtu_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate domain IDs for all .vtu files in a folder and its subfolders.')
    parser.add_argument('folder', help='Path to the folder containing .vtu files (recursively)')
    args = parser.parse_args()

    vtu_files = find_vtu_files(args.folder)
    print(f'Found {len(vtu_files)} .vtu files.')
    for i, vtu_file in enumerate(vtu_files, 1):
        print(f'[{i}/{len(vtu_files)}] Processing {vtu_file}')
        try:
            generate_domain_ids(vtu_file)
        except Exception as e:
            print(f'Error processing {vtu_file}: {e}')
    print('Done.')