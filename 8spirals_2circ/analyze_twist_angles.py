#!/usr/bin/env python3
"""
Script to analyze twist angles in ventricular simulations.

This script demonstrates how to use the twist angle calculation functions
from process_results_functions.py to analyze ventricular twist during cardiac cycles.

Author: Assistant
Date: 2024
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from process_results_functions import (
    calc_twist_angle, 
    calc_volume_3D,
    visualize_twist_angle_calculation,
    plot_twist_angles_vs_time,
    plot_twist_angles_vs_z,
    get_start_end_step,
    get_timestep_size,
    compute_final_twist_angle
)

def analyze_twist_angles_example(reference_surface=None, pressure_file=None):
    """
    Example function demonstrating how to use the twist angle analysis functions.
    
    Args:
        reference_surface: Path to the reference surface file. If None, uses default.
        pressure_file: Path to the pressure file. If provided, appends 'pXX' to output directory.
    """
    
    # ===== CONFIGURATION =====
    # Update these paths to match your simulation setup
     # Append pressure file identifier if provided
    if pressure_file is not None:
        pressure_str = 'p' + pressure_file.split("_")[-1]
        pressure_str = pressure_str.replace('.dat', '')
        pressure_str = '_' + pressure_str
    else:
        pressure_str = ''
    
    # Simulation results folder (usually contains result_XXX.vtu files)
    results_folder = "results_" + reference_surface.split("/")[-3].split("-")[0] + pressure_str  # Update this path
    
    # Reference surface file (undeformed surface)
    if reference_surface is None:
        reference_surface = "meshes/ventricle_8_30_e-mesh-complete/mesh-surfaces/epi.vtp"  # Default path
    
    # Output directory for plots and intermediate data
    output_dir = "twist_analysis_" + reference_surface.split("/")[-3].split("-")[0] + pressure_str
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ===== PARAMETERS =====
    # Define z-levels for twist analysis (in mm)
    # These should be within the z-range of your ventricle
    z_levels = None #np.array([-25, -20, -15, -10, -5, 0])
    
    # Tolerance for finding points near the xz plane and z-levels
    tolerance = 0.5  # mm
    
    # Whether to save intermediate data
    save_intermediate_data = True
    intermediate_output_folder = os.path.join(output_dir, "intermediate_data")
    
    print("=== Ventricular Twist Angle Analysis ===")
    print(f"Results folder: {results_folder}")
    print(f"Reference surface: {reference_surface}")
    if pressure_file:
        print(f"Pressure file: {pressure_file}")
    print(f"Z-levels: {z_levels}")
    print(f"Output directory: {output_dir}")
    
    # ===== CHECK FILES EXIST =====
    if not os.path.exists(results_folder):
        print(f"ERROR: Results folder not found: {results_folder}")
        return
    
    if not os.path.exists(reference_surface):
        print(f"ERROR: Reference surface not found: {reference_surface}")
        return
    
    # ===== GET SIMULATION PARAMETERS =====
    try:
        # Get start, end, and step from results folder
        start_timestep, end_timestep, step = get_start_end_step(results_folder)
        print(f"Found timesteps: {start_timestep} to {end_timestep} with step {step}")
        
        # Get timestep size (you may need to provide this manually)
        # timestep_size = get_timestep_size("solver.xml")  # Uncomment if you have solver.xml
        timestep_size = 0.0005  # seconds - update this to match your simulation
        
    except Exception as e:
        print(f"Warning: Could not automatically determine simulation parameters: {e}")
        print("Using default values. Please update manually if needed.")
        start_timestep = 10
        end_timestep = 100
        step = 5
        timestep_size = 0.0005
    
    # ===== CALCULATE TWIST ANGLES =====
    print("\n=== Calculating Twist Angles ===")
    
    try:
        t, twist_angles, z_levels_used = calc_twist_angle(
            start_timestep=start_timestep,
            end_timestep=end_timestep,
            step=step,
            timestep_size=timestep_size,
            results_folder=results_folder,
            reference_surface=reference_surface,
            z_levels=z_levels,
            save_intermediate_data=False,
            intermediate_output_folder=None,
            tolerance=tolerance
        )

        final_twist_angle = compute_final_twist_angle(twist_angles)
        
        print(f"Successfully calculated twist angles for {len(t)} time points")
        print(f"Z-levels used: {z_levels_used}")
        print(f"Final twist angle: {final_twist_angle}")
    except Exception as e:
        print(f"ERROR: Failed to calculate twist angles: {e}")
        return
    
    # ===== VISUALIZE THE CALCULATION =====
    print("\n=== Creating Visualization ===")
    
    try:
        # Load reference surface for visualization
        ref_surface = pv.read(reference_surface)
        
        # Create visualization of how twist angles are calculated
        plotter = visualize_twist_angle_calculation(
            surface=ref_surface,
            z_levels=z_levels_used,
            output_file=os.path.join(output_dir, "twist_calculation_visualization.png")
        )
        
        print("Visualization saved to: twist_calculation_visualization.png")
        
        # Uncomment to show the plot interactively
        # plotter.show()
        
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
    
    # ===== EXPORT PRINCIPAL STRAIN FIELDS OVER TIME =====
    print("\n=== Exporting Principal Strain Fields Over Time ===")
    try:
        principal_strain_dir = os.path.join(output_dir, "principal_strain")
        if not os.path.exists(principal_strain_dir):
            os.makedirs(principal_strain_dir)

        # Prepare reference lumen once (watertight) for consistent sampling
        ref_surf_full = pv.read(reference_surface)
        ref_lumen = ref_surf_full.fill_holes(100)
        ref_lumen.compute_normals(inplace=True)

        # Helper to ensure Def_grad is point data on the result dataset
        def ensure_def_grad_point_data(result_ds):
            has_point = ('Def_grad' in result_ds.point_data)
            has_cell = ('Def_grad' in result_ds.cell_data)
            if not has_point and has_cell:
                try:
                    result_ds = result_ds.cell_data_to_point_data(pass_cell_data=True)
                except Exception:
                    pass
            return result_ds

        # Loop through timesteps and save per-timestep VTP with fields
        for k in range(start_timestep, end_timestep + 1, step):
            result_path = os.path.join(results_folder, f"result_{k:03d}.vtu")
            if not os.path.exists(result_path):
                continue
            try:
                result_ds = pv.read(result_path)
                result_ds = ensure_def_grad_point_data(result_ds)

                # Sample result onto reference lumen and warp by displacement
                resampled = ref_lumen.sample(result_ds)
                warped = resampled.warp_by_vector('Displacement')

                # Compute principal Green-Lagrange strain (max) and direction at each point
                if 'Def_grad' not in resampled.point_data:
                    print(f"  [k={k}] Skipping: 'Def_grad' not present after sampling")
                    continue
                F_flat = np.asarray(resampled.point_data['Def_grad'])  # shape (N, 9)
                if F_flat.ndim != 2 or F_flat.shape[1] != 9:
                    print(f"  [k={k}] Skipping: 'Def_grad' has unexpected shape {F_flat.shape}")
                    continue

                # Reshape to (N, 3, 3)
                F = F_flat.reshape((-1, 3, 3))
                # Right Cauchy-Green tensor C = F^T F
                C = np.einsum('...ji,...jk->...ik', F, F)
                # Eigen-decomposition of C
                # For numerical stability, symmetrize C
                C_sym = 0.5 * (C + np.swapaxes(C, -1, -2))
                # Use numpy.linalg.eigh which assumes symmetric input
                vals, vecs = np.linalg.eigh(C_sym)  # vals ascending per point
                # Principal Green-Lagrange strains: E_i = 0.5*(lambda_i - 1), where lambda_i are eigenvalues of C
                E_principal = 0.5 * (vals - 1.0)
                # Take max principal strain and corresponding direction
                max_idx = np.argmax(E_principal, axis=1)
                max_strain = E_principal[np.arange(E_principal.shape[0]), max_idx]
                # Directions: eigenvectors in reference configuration
                # vecs shape: (N, 3, 3) with vecs[..., i] the eigenvector for vals[..., i]
                dirs = np.zeros_like(vecs)
                for i in range(vecs.shape[0]):
                    dirs[i] = vecs[i]
                max_dir = dirs[np.arange(dirs.shape[0]), :, max_idx]
                # Normalize direction vectors
                norms = np.linalg.norm(max_dir, axis=1, keepdims=True) + 1e-12
                max_dir_unit = max_dir / norms

                # Attach fields to warped surface (current configuration geometry), using resampled points mapping
                warped.point_data['MaxPrincipalStrain'] = max_strain.astype(np.float64)
                warped.point_data['MaxPrincipalStrainDir'] = max_dir_unit.astype(np.float64)
                # Displacement should already exist; ensure present
                if 'Displacement' in resampled.point_data and 'Displacement' not in warped.point_data:
                    warped.point_data['Displacement'] = np.asarray(resampled.point_data['Displacement'])

                out_path = os.path.join(principal_strain_dir, f"principal_strain_{k:03d}.vtp")
                warped.save(out_path)
                print(f"  [k={k}] Saved principal strain surface: {out_path}")
            except Exception as e:
                print(f"  [k={k}] Warning: Failed principal strain export: {e}")
    except Exception as e:
        print(f"Warning: Principal strain export failed to initialize: {e}")

    # ===== CREATE PLOTS =====
    print("\n=== Creating Plots ===")
    
    try:
        # Plot 1: Twist angles vs time for different z-levels
        fig1 = plot_twist_angles_vs_time(
            t=t,
            twist_angles=twist_angles,
            z_levels=z_levels_used,
            output_file=os.path.join(output_dir, "twist_angles_vs_time.png"),
            title="Ventricular Twist Angles vs Time"
        )
        
        # Plot 2: Twist angles vs z-coordinate at different time points
        # Select a few representative time points
        time_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, len(t)-1]
        fig2 = plot_twist_angles_vs_z(
            t=t,
            twist_angles=twist_angles,
            z_levels=z_levels_used,
            time_indices=time_indices,
            output_file=os.path.join(output_dir, "twist_angles_vs_z.png"),
            title="Twist Angles vs Z-coordinate at Different Times"
        )
        
        print("Plots saved to:")
        print("  - twist_angles_vs_time.png")
        print("  - twist_angles_vs_z.png")
        
        # ===== CALCULATE VOLUME AND CREATE VOLUME PLOT =====
        print("\n=== Calculating Volume and Creating Volume Plot ===")
        
        try:
            # Calculate volume using the same parameters
            t_vol, volumes, radial_strains, longitudinal_strains = calc_volume_3D(
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                step=step,
                timestep_size=timestep_size,
                results_folder=results_folder,
                reference_surface=reference_surface,
                save_intermediate_data=save_intermediate_data,
                intermediate_output_folder=intermediate_output_folder
            )
            
            print(f"Volume calculation complete. Time points: {len(t_vol)}, Volume points: {len(volumes)}")
            
            # Create twist angle vs volume plot
            fig3 = plot_twist_angles_vs_volume(
                t=t,
                twist_angles=twist_angles,
                volumes=volumes,
                z_levels=z_levels_used,
                output_file=os.path.join(output_dir, "twist_angles_vs_volume.png"),
                title="Twist Angles vs Ventricular Volume"
            )

            #Create twist angle vs radial strain plot
            fig4 = plot_twist_angles_vs_radial_strain(
                twist_angles=twist_angles,
                radial_strains=radial_strains,
                z_levels=z_levels_used,
                output_file=os.path.join(output_dir, "twist_angles_vs_radial_strain.png"),
            )

            #Create twist angle vs longitudinal strain plot
            fig5 = plot_twist_angles_vs_longitudinal_strain(
                twist_angles=twist_angles,
                longitudinal_strains=longitudinal_strains,
                z_levels=z_levels_used,
                output_file=os.path.join(output_dir, "twist_angles_vs_longitudinal_strain.png"),
            )

            # New: volume vs (circ strain, long strain, twist range)
            twist_range = compute_twist_range_time_series(twist_angles)
            v0 = float(volumes[0]) if len(volumes) > 0 else 0.0
            three_panel_path = os.path.join(output_dir, "volume_vs_strains_and_twist.png")
            plot_volume_vs_strain_and_twist(volumes, radial_strains, longitudinal_strains, twist_range, v0, three_panel_path)
            
            print("  - twist_angles_vs_volume.png")
            print("  - volume_vs_strains_and_twist.png")
            
            # Update data file to include volume
            data_file = os.path.join(output_dir, "twist_angles_data.npz")
            np.savez(data_file, 
                     time=t, 
                     twist_angles=twist_angles, 
                     z_levels=z_levels_used,
                     volumes=volumes,
                     radial_strains=radial_strains,
                     longitudinal_strains=longitudinal_strains)
            
        except Exception as e:
            print(f"Warning: Could not calculate volume or create volume plot: {e}")
        
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
    
    # ===== ANALYZE RESULTS =====
    print("\n=== Analyzing Results ===")
    
    try:
        # Calculate some basic statistics
        print("Twist angle statistics:")
        print(f"Final twist angle: {final_twist_angle}")
        for i, z in enumerate(z_levels_used):
            angles = [angle for angle in twist_angles[i] if angle is not None]
            if angles:
                mean_angle = np.mean(angles)
                max_angle = np.max(angles)
                min_angle = np.min(angles)
                angle_range = max_angle - min_angle
                print(f"  Z={z:.1f}: mean={mean_angle:.1f}°, range={angle_range:.1f}°, "
                      f"min={min_angle:.1f}°, max={max_angle:.1f}°")
            else:
                print(f"  Z={z:.1f}: No valid data")
        
        # Find maximum twist at each z-level
        max_twist_by_z = []
        for i, z in enumerate(z_levels_used):
            angles = [angle for angle in twist_angles[i] if angle is not None]
            if angles:
                max_twist_by_z.append((z, np.max(angles)))
        
        if max_twist_by_z:
            max_twist_z, max_twist_angle = max(max_twist_by_z, key=lambda x: x[1])
            print(f"\nMaximum twist: {max_twist_angle:.1f}° at Z={max_twist_z:.1f}")
        
    except Exception as e:
        print(f"Warning: Could not analyze results: {e}")
    
    # ===== SAVE DATA =====
    print("\n=== Saving Data ===")
    
    try:
        # Save twist angle data to numpy file
        data_file = os.path.join(output_dir, "twist_angles_data.npz")
        np.savez(data_file, 
                 time=t, 
                 twist_angles=twist_angles, 
                 z_levels=z_levels_used)
        print(f"Data saved to: {data_file}")
        
        # Save summary to text file
        summary_file = os.path.join(output_dir, "twist_analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("Ventricular Twist Angle Analysis Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Results folder: {results_folder}\n")
            f.write(f"Reference surface: {reference_surface}\n")
            f.write(f"Time points: {len(t)}\n")
            f.write(f"Z-levels: {z_levels_used}\n")
            f.write(f"Time range: {t[0]:.3f}s to {t[-1]:.3f}s\n\n")
            
            f.write("Twist angle statistics:\n")
            for i, z in enumerate(z_levels_used):
                angles = [angle for angle in twist_angles[i] if angle is not None]
                if angles:
                    mean_angle = np.mean(angles)
                    max_angle = np.max(angles)
                    min_angle = np.min(angles)
                    angle_range = max_angle - min_angle
                    f.write(f"  Z={z:.1f}: mean={mean_angle:.1f}°, range={angle_range:.1f}°, "
                           f"min={min_angle:.1f}°, max={max_angle:.1f}°\n")
                    
            # For all time steps
            f.write("Circumferential strain statistics:\n")
            mean_strain = np.mean(radial_strains)
            max_strain = np.max(radial_strains)
            min_strain = np.min(radial_strains)
            strain_range = max_strain - min_strain
            f.write(f"  mean={mean_strain:.4f}, range={strain_range:.4f}, min={min_strain:.4f}, max={max_strain:.4f}\n\n")

            f.write("Longitudinal strain statistics:\n")
            mean_strain = np.mean(longitudinal_strains)
            max_strain = np.max(longitudinal_strains)
            min_strain = np.min(longitudinal_strains)
            strain_range = max_strain - min_strain
            f.write(f"  mean={mean_strain:.4f}, range={strain_range:.4f}, min={min_strain:.4f}, max={max_strain:.4f}\n")

            # Compute final volume
            final_volume = volumes[-1] if isinstance(volumes, (list, np.ndarray)) else volumes
            ratio_twist_volume = final_twist_angle / final_volume if final_volume != 0 else np.nan
            ratio_twist_radial = final_twist_angle / radial_strains[-1] if radial_strains[-1] != 0 else np.nan
            ratio_twist_longitudinal = final_twist_angle / longitudinal_strains[-1] if longitudinal_strains[-1] != 0 else np.nan

            # Write additional ratios to summary file
            f.write(f"Final Volume: {final_volume}\n")
            f.write(f"Twist/Volume Ratio: {ratio_twist_volume}\n")
            f.write(f"Twist/Final Circumferential Strain Ratio: {ratio_twist_radial}\n")
            f.write(f"Twist/Final Longitudinal Strain Ratio: {ratio_twist_longitudinal}\n\n")

            f.write("Final Metrics:\n")
            f.write(f"Final Circumferential Strain: {radial_strains[-1]}\n")
            f.write(f"Final Longitudinal Strain: {longitudinal_strains[-1]}\n")
            f.write(f"Final Twist Angle: {final_twist_angle}\n")
        
        print(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Warning: Could not save data: {e}")
    
    print("\n=== Analysis Complete ===")
    print(f"All outputs saved to: {output_dir}")


def analyze_multiple_simulations():
    """
    Example function to analyze twist angles from multiple simulations.
    Useful for comparing different ventricular geometries or parameters.
    """
    
    # Define simulation configurations to compare
    simulations = [
        {
            'name': '8_20',
            'results_folder': 'results_8_20_p10',
            'reference_surface': 'meshes/ventricle_MC_200_8_20_0d6mm-mesh-complete/mesh-surfaces/epi.vtp',
            'color': 'blue'
        },
        {
            'name': '8_30',
            'results_folder': 'results_8_40_p10',
            'reference_surface': 'meshes/ventricle_MC_200_8_30_5d5mm-mesh-complete/mesh-surfaces/epi.vtp',
            'color': 'red'
        }
    ]
    
    z_levels = None #np.array([-25, -20, -15, -10, -5, 0])
    timestep_size = 0.001
    
    # Collect data from all simulations
    all_data = {}
    
    for sim in simulations:
        print(f"\n=== Analyzing {sim['name']} ===")
        
        if not os.path.exists(sim['results_folder']):
            print(f"Warning: Results folder not found: {sim['results_folder']}")
            continue
        
        try:
            start_timestep, end_timestep, step = get_start_end_step(sim['results_folder'])
            
            t, twist_angles, z_levels_used = calc_twist_angle(
                start_timestep=start_timestep,
                end_timestep=end_timestep,
                step=step,
                timestep_size=timestep_size,
                results_folder=sim['results_folder'],
                reference_surface=sim['reference_surface'],
                z_levels=z_levels
            )
            
            all_data[sim['name']] = {
                't': t,
                'twist_angles': twist_angles,
                'z_levels': z_levels_used,
                'color': sim['color']
            }
            
            print(f"Successfully analyzed {sim['name']}")
            
        except Exception as e:
            print(f"Error analyzing {sim['name']}: {e}")
    
    # Create comparison plots
    if len(all_data) > 1:
        print("\n=== Creating Comparison Plots ===")
        
        # Plot twist angles vs time for comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for sim_name, data in all_data.items():
            for i, z in enumerate(data['z_levels']):
                angles = data['twist_angles'][i]
                valid_indices = [j for j, angle in enumerate(angles) if angle is not None]
                valid_times = [data['t'][j] for j in valid_indices]
                valid_angles = [angles[j] for j in valid_indices]
                
                if valid_angles:
                    ax.plot(valid_times, valid_angles, 
                           color=data['color'], alpha=0.7,
                           label=f'{sim_name} Z={z:.1f}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Twist Angle (degrees)', fontsize=12)
        ax.set_title('Twist Angle Comparison Across Simulations', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('twist_comparison.png', dpi=300, bbox_inches='tight')
        print("Comparison plot saved to: twist_comparison.png")


def plot_twist_angles_vs_volume(t, twist_angles, volumes, z_levels, output_file=None, title="Twist Angle vs Volume"):
    """
    Plot twist angles as a function of volume for different z-levels.
    
    Args:
        t: List of time points
        twist_angles: List of twist angle arrays (one for each z-level)
        volumes: List of volumes at each time point
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
        # Filter out None values and create valid data pairs
        valid_data = []
        for j, (angle, vol) in enumerate(zip(angles, volumes)):
            if angle is not None and vol is not None:
                valid_data.append((vol, angle))
        
        if valid_data:
            volumes_valid, angles_valid = zip(*valid_data)
            ax.plot(volumes_valid, angles_valid, color=colors[i], 
                   label=f'Z={z:.1f}', linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Volume (mm³)', fontsize=12)
    ax.set_ylabel('Twist Angle (degrees)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig

def compute_twist_range_time_series(twist_angles):
    """
    Compute, at each time index, the difference between the maximum and minimum
    twist angle across z-levels (ignoring None entries).
    Returns a list with length equal to the number of time points.
    """
    if not twist_angles:
        return []
    num_times = max(len(series) for series in twist_angles)
    twist_range = []
    for j in range(num_times):
        vals = []
        for series in twist_angles:
            if j < len(series):
                v = series[j]
                if v is not None:
                    vals.append(v)
        if len(vals) >= 2:
            twist_range.append(max(vals) - min(vals))
        else:
            twist_range.append(None)
    return twist_range

def plot_volume_vs_strain_and_twist(volumes, radial_strains, longitudinal_strains, twist_range, v0, output_file=None):
    """
    Create three stacked subplots vs volume:
    - Circumferential (radial) strain
    - Longitudinal strain
    - Twist angle range (max-min across z-levels)
    X-axis limited to [v0, 2*v0].
    """
    import matplotlib.pyplot as plt
    n = min(len(volumes), len(radial_strains), len(longitudinal_strains), len(twist_range))
    V = np.array(volumes[:n], dtype=float)
    Er = np.array(radial_strains[:n], dtype=float)
    Ez = np.array(longitudinal_strains[:n], dtype=float)
    Tw = np.array([x if x is not None else np.nan for x in twist_range[:n]], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(V, Er, marker='o', linewidth=1.5)
    axes[0].set_ylabel('Circumferential strain', fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.5, 0)
    # Reference line and label
    axes[0].axhline(-0.3, linestyle=':', color='k', linewidth=1)
    try:
        x_label_pos = np.nanmax(V) if np.isfinite(np.nanmax(V)) else (V[-1] if len(V) else 1.0)
    except Exception:
        x_label_pos = V[-1] if len(V) else 1.0
    axes[0].text(x_label_pos, -0.3, 'physiological target', ha='right', va='bottom', fontsize=10, color='k')

    axes[1].plot(V, Ez, marker='o', linewidth=1.5)
    axes[1].set_ylabel('Longitudinal strain', fontsize=16)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.5, 0)
    # Reference line and label
    axes[1].axhline(-0.25, linestyle=':', color='k', linewidth=1)
    try:
        x_label_pos = np.nanmax(V) if np.isfinite(np.nanmax(V)) else (V[-1] if len(V) else 1.0)
    except Exception:
        x_label_pos = V[-1] if len(V) else 1.0
    axes[1].text(x_label_pos, -0.25, 'physiological target', ha='right', va='bottom', fontsize=10, color='k')

    axes[2].plot(V, Tw, marker='o', linewidth=1.5)
    axes[2].set_ylabel('Twist angle (deg)', fontsize=16)
    axes[2].set_xlabel('Volume (mm³)', fontsize=16)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 30)
    # Reference line for twist angle target
    axes[2].axhline(17.0, linestyle=':', color='k', linewidth=1)
    if v0 > 0:
        axes[2].set_xlim([v0, 1.75 * v0])

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_twist_angles_vs_radial_strain(twist_angles, radial_strains, z_levels, output_file=None, title="Twist Angle vs Radial Strain"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_levels)))
    for i, (z, angles) in enumerate(zip(z_levels, twist_angles)):
        valid_data = []
        for j, (angle, strain) in enumerate(zip(angles, radial_strains)):
            if angle is not None and strain is not None:
                valid_data.append((strain, angle))
        if valid_data:
            strains_valid, angles_valid = zip(*valid_data)
            ax.plot(strains_valid, angles_valid, color=colors[i], label=f'Z={z:.1f}', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Radial Strain')
    ax.set_ylabel('Twist Angle (degrees)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig

def plot_twist_angles_vs_longitudinal_strain(twist_angles, longitudinal_strains, z_levels, output_file=None, title="Twist Angle vs Longitudinal Strain"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_levels)))
    for i, (z, angles) in enumerate(zip(z_levels, twist_angles)):
        valid_data = []
        for j, (angle, strain) in enumerate(zip(angles, longitudinal_strains)):
            if angle is not None and strain is not None:
                valid_data.append((strain, angle))
        if valid_data:
            strains_valid, angles_valid = zip(*valid_data)
            ax.plot(strains_valid, angles_valid, color=colors[i], label=f'Z={z:.1f}', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Longitudinal Strain')
    ax.set_ylabel('Twist Angle (degrees)')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    """
    Main execution block.
    """
    parser = argparse.ArgumentParser(description='Analyze ventricular twist angles from simulation results.')
    parser.add_argument('reference_surface', nargs='?', default=None, 
                       help='Path to reference surface file (e.g., mesh-surfaces/epi.vtp)')
    parser.add_argument('--pressure-file', help='Path to pressure file (e.g., pressure_XX.dat)')
    parser.add_argument('--compare', action='store_true', help='Run multiple simulation comparison')
    args = parser.parse_args()
    
    print("Ventricular Twist Angle Analysis Script")
    print("=" * 40)
    
    if args.compare:
            print("Running multiple simulation comparison...")
            analyze_multiple_simulations()
    else:
        print("Running single simulation analysis...")
        analyze_twist_angles_example(args.reference_surface, args.pressure_file) 