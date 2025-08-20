# Ventricular Optimization Project

A comprehensive pipeline for generating, processing, simulating, and analyzing ventricular meshes with spiral support parameters.

## Overview

This project provides tools for:
- **Mesh Generation**: Creating complex 3D ventricular geometries with ellipsoidal shells, roofs, and spiral supports
- **Mesh Processing**: Remeshing, tetrahedralization, and mesh repair using PyMeshLab and TetGen
- **Simulation Setup**: Dynamic configuration of svMultiPhysics solver files
- **Results Analysis**: Computing twist angles, strains, and volumes from simulation outputs
- **Cluster Computing**: SLURM job submission for parallel processing on Sherlock cluster

## Project Structure

```
Ventricle_optimization/
├── 8_spirals/                    # Analysis scripts and examples
│   ├── analyze_twist_angles.py   # Main analysis script
│   ├── process_results_functions.py  # Analysis functions library
│   ├── plot_summary_metrics.py   # Summary plotting script
│   ├── edit_solver.py           # Solver configuration script
│   └── solver.xml               # Example solver configuration
├── generate_ventricle.py         # Main mesh generation script
├── mesh_processing.py           # Mesh processing utilities
├── generate_pressure.py         # Pressure file generation
├── generate_fibers_LV_Bayer_cells.py  # Fiber direction generation
├── Sherlock/                    # Cluster computing scripts
│   └── svFSI_job.sh            # SLURM job submission script
├── meshes/                      # Generated mesh files
├── ventricle_dragonskin10_061925_LR.m  # Original MATLAB script
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.7+
- MATLAB (for original script)
- svMultiPhysics (for simulations)
- Access to Sherlock cluster (for cluster computing)

### Python Dependencies

```bash
pip install numpy matplotlib pyvista pymeshlab tetgen scipy
```

### Additional Software

- **PyMeshLab**: For mesh processing and repair
- **TetGen**: For tetrahedralization
- **svMultiPhysics**: For finite element simulations

## Usage

### 1. Mesh Generation

Generate ventricular meshes with spiral supports:

```bash
python generate_ventricle.py
```

This creates meshes with different spiral configurations (number of spirals, helix angles) in parallel.

### 2. Mesh Processing

Process generated meshes for simulation:

```bash
# Process single mesh
python mesh_processing.py meshes/all_cases/ventricle_8_20.obj

# Process multiple meshes
python mesh_processing.py meshes/all_cases/
```

### 3. Simulation Setup

Configure solver files for different cases:

```bash
python 8_spirals/edit_solver.py --mesh-folder meshes/ventricle_8_20_c-mesh-complete \
    --time-step-size 0.0005 --num-time-steps 1000 \
    --roof-elasticity 1000 --spiral-elasticity 5000 \
    --pressure-file pressure_20.dat
```

### 4. Results Analysis

Analyze simulation results:

```bash
# Basic analysis
python 8_spirals/analyze_twist_angles.py meshes/ventricle_8_20_c-mesh-complete/mesh-surfaces/epi.vtp

# With pressure file
python 8_spirals/analyze_twist_angles.py meshes/ventricle_8_20_c-mesh-complete/mesh-surfaces/epi.vtp \
    --pressure-file pressure_20.dat

# Plot summary metrics
python 8_spirals/plot_summary_metrics.py 8_spirals/all_summaries_8_spirals
```

### 5. Cluster Computing

Submit jobs to Sherlock cluster:

```bash
# Submit to cluster
sbatch Sherlock/svFSI_job.sh
```

## Key Features

### Mesh Generation
- **Implicit Functions**: Uses signed distance functions for complex geometries
- **Marching Cubes**: Efficient isosurface extraction
- **Spiral Supports**: Configurable spiral tube parameters
- **Parallel Processing**: Multi-core mesh generation

### Analysis Capabilities
- **Twist Angle Calculation**: Using polar decomposition of deformation gradient
- **Strain Analysis**: Radial and longitudinal strain computation
- **Volume Tracking**: Enclosed volume calculation
- **Time Series Analysis**: Complete cardiac cycle analysis

### Cluster Integration
- **SLURM Integration**: Automated job submission
- **Multiple Cases**: Parallel processing of parameter combinations
- **Results Organization**: Structured output directories

## File Formats

- **Input**: `.obj` (mesh files), `.xml` (solver configs)
- **Output**: `.vtu` (volume meshes), `.vtp` (surface meshes), `.dat` (pressure/stress files)
- **Analysis**: `.png` (plots), `.txt` (summary statistics)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for research purposes. Please cite appropriately if used in publications.

## Contact

For questions or issues, please contact the development team. 