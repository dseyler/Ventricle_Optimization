# Use commands for managing an svFSI simulation and results

# Define a list of required files and folders
REQUIRED_FILES = \
    data \
	fiber_stress \
    *-mesh-complete* \
    genBC_* \
	iFEA \
	variable_Robin_mesh \
	make_paraview_animations_job.sh \
    Makefile \
    process_results_job.sh \
    process_results.py \
    README.md \
    submit_sim_to_sherlock.sh \
    svFSI_job.sh \
    svFSI.inp \
    sync_sim_from_sherlock.sh \
	sync_to_gdrive_job.sh

clean:
	rm -f -r 24-procs 4-procs results AllData GenBC.int InitialData
	rm -f -r processed_results* animations
	rm log.txt

clean_all: clean
	rm -f -r *_job.o* *_job.e*

run:
	mpiexec -np 4 /Users/aaronbrown/GitHub/svFSI_aabrown100-git/build/svFSI-build/bin/svFSI svFSI.inp

run_vvedula22:
	mpiexec -np 4 /Users/aaronbrown/GitHub/svFSI_vvedula22/build/svFSI-build/bin/svFSI 

run_prestress:
	mpiexec -np 4 /Users/aaronbrown/GitHub/svFSI_aabrown100/build/svFSI-build/bin/svFSI svFSI_prestress.inp

run_sherlock:
	sbatch ./svFSI_job.sh

run_sherlock_prestress:
	sbatch ./svFSI_prestress_job.sh

process_results:
	python3 process_results.py ./

stop_sim:
	(cd *-procs/ && touch STOP_SIM)

copy_folder:
	current_folder_base=$$(basename "$$(pwd)"); \
    parent_folder=$$(dirname "$$(pwd)"); \
	name=$${name:-$${current_folder_base}_copy}; \
    new_folder=$${parent_folder}/$${name}; \
    mkdir -p "$${new_folder}"; \
	rsync -av --progress $(REQUIRED_FILES) "$${new_folder}"; \
	echo "Copied to $${new_folder}"

copy_folder_sherlock:
	current_folder_base=$$(basename "$$(pwd)"); \
	name=$${name:-$${current_folder_base}_copy}; \
	remote_user="abrown97"; \
	remote_server="login.sherlock.stanford.edu"; \
	remote_folder="/scratch/users/abrown97/sandbox/personalized_cardiac_mechanics_paper_final_sims/$${current_folder_base}"; \
	remote_parent_folder=$$(dirname "$${remote_folder}"); \
	remote_new_folder=$${remote_parent_folder}/$${name}; \
	ssh -t $${remote_user}@$${remote_server} "cd $${remote_folder} && make copy_folder name=$${name} && cd $${remote_new_folder} && make sync_to_gdrive"

sync_to_gdrive:
	sbatch ./sync_to_gdrive_job.sh

sync_sherlock_to_gdrive:
	name=$${name:-$$(basename "$$(pwd)")}; \
	remote_user="abrown97"; \
	remote_server="login.sherlock.stanford.edu"; \
	remote_folder="/scratch/users/abrown97/sandbox/personalized_cardiac_mechanics_paper_final_sims/$${name}"; \
	ssh -t $${remote_user}@$${remote_server} "cd $${remote_folder} && make sync_to_gdrive"

check_files:
	@for file in $(REQUIRED_FILES); do \
		if [ ! -e "$$file" ]; then \
			echo "Error: Required file or folder not found:"; \
			echo "  $$file"; \
			exit 1; \
		fi; \
	done
	echo "All required files and folders found."

paraview_animations:
	pvpython ../paraview_files/make_paraview_animations.py . results processed_results_last_2_cardiac_cycles data 50 2