import pyvista as pv
import numpy as np

# Load mesh
mesh = pv.read('meshes/ventricle_meshmixer.vtp')

# Define the two facets
facet_1 = np.array([29494, 29503, 27415])
facet_2 = np.array([29494, 29503, 29124])

# Create a plotter
plotter = pv.Plotter()

# Add the main mesh (semi-transparent)
plotter.add_mesh(mesh, opacity=0.2, color='lightblue', show_edges=False)

# Extract and plot facet 1
facet_1_mesh = mesh.extract_cells(facet_1)
plotter.add_mesh(facet_1_mesh, color='red', line_width=5, show_edges=True, edge_color='darkred')

# Extract and plot facet 2
facet_2_mesh = mesh.extract_cells(facet_2)
plotter.add_mesh(facet_2_mesh, color='green', line_width=5, show_edges=True, edge_color='darkgreen')

# Add labels for the facets
plotter.add_text("Facet 1 (Red)", position=(0.1, 0.9), font_size=12, color='red')
plotter.add_text("Facet 2 (Green)", position=(0.1, 0.85), font_size=12, color='green')

# Set camera position for better view
plotter.camera_position = 'xy'
plotter.show()

print("Facet 1 vertices:", facet_1)
print("Facet 2 vertices:", facet_2)
print("Mesh loaded successfully with", mesh.n_cells, "cells and", mesh.n_points, "points")
