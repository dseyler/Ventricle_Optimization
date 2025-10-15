#!/bin/bash

# Name of your job
#SBATCH --job-name=auto_stop

# Name of partition
#SBATCH --partition=amarsden

# Specify the name of the output file. The %j specifies the job ID
#SBATCH --output=job.o%j

# Specify a name of the error file. The %j specifies the job ID
#SBATCH --error=job.e%j

# The walltime you require for your simulation
#SBATCH --time=24:00:00

# Job priority. Leave as normal for now.
#SBATCH --qos=normal

# Number of nodes you are requesting for your job. You can have 16 processors per node, so plan accordingly
#SBATCH --nodes=1

# Amount of memory you require per node. The default is 4000 MB (or 4 GB) per node
#SBATCH --mem=20000

# Number of processors per node
#SBATCH --ntasks-per-node=24

# Send an email to this address when you job starts and finishes
#SBATCH --mail-user=dseyler@stanford.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=fail
#SBATCH --mail-type=end

# Clean simulation directory
make clean

module purge
# for process_results.py
module load viz
module load py-matplotlib/3.8.3_py312
module load py-scipy/1.12.0_py312
module load py-seaborn/0.13.2_py312

python3 -u generate_pressure.py

MESH_BASE="meshes"

for mesh_dir in "$MESH_BASE"/*8_c-mesh-complete; do
    if [ -d "$mesh_dir" ]; then
        for pressure_file in pressure_*.dat; do
            echo "Processing $mesh_dir with $pressure_file"
            python3 -u edit_solver.py solver.xml $mesh_dir 0.001 500 --pressure-file $pressure_file
            # Start NaN watcher for histor.dat; on NaN kill solver and mark to skip
            rm -f .nan_detected 2>/dev/null || true
            case_base=$(basename "$mesh_dir" | cut -d'-' -f1)
            pressure_suffix=$(basename "$pressure_file" .dat | awk -F'_' '{print $2}')
            results_glob="results_${case_base}_p${pressure_suffix}*"
            python3 -u watchers.py nan --results-glob "$results_glob" --marker .nan_detected &
            WATCHER_PID=$!

            # Start Volume watcher: terminate if volume exceeds 3x undeformed volume
            rm -f .volume_exceeded 2>/dev/null || true
            # Volume watcher logs per-case volumes to a dedicated log file
            vol_log="volume_${case_base}_p${pressure_suffix}.log"
            python3 -u watchers.py volume --ref-surface "$mesh_dir/mesh-surfaces/epi.vtp" \
                                          --results-glob "$results_glob" --threshold-factor 3.0 \
                                          --marker .volume_exceeded --log-file "$vol_log" &
            VOL_WATCHER_PID=$!

            # MPI run the executable svFSI with svFSI.xml
            singularity exec $SCRATCH/solver_latest.sif bash -c "mpirun -n 23 $HOME/solver/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics solver.xml"

            # Ensure watchers are stopped after solver exits
            kill $WATCHER_PID 2>/dev/null || true
            kill $VOL_WATCHER_PID 2>/dev/null || true

            # If NaN or volume exceeded detected, skip post-processing and continue
            if [ -f .nan_detected ] || [ -f .volume_exceeded ]; then
                echo "Skipping post-processing due to $( [ -f .nan_detected ] && echo NaN || echo volume threshold ) for $mesh_dir with $pressure_file"
                rm -f .nan_detected .volume_exceeded
                continue
            fi

            # Run post-processing after simulation completes
            python3 -u analyze_twist_angles.py $mesh_dir/mesh-surfaces/epi.vtp --pressure-file $pressure_file
        done
    fi
done


# Submit job with 
# sbatch ./svFSI_job.sh
# Check status with
# squeue -u $USER