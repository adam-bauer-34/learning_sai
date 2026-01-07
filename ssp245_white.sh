#!/bin/bash

#SBATCH --job-name=ssp245-white
#SBATCH --output=logs/ssp245-white.out
#SBATCH --error=logs/ssp245-white.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=caslake

#SBATCH --time=24:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=4000

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ambauer@rcc.uchicago.edu

# the actual code
# ensure the logs directory actually exists
mkdir -p logs

# name and make the outfile
OUTFILE = "logs/${SLURM_JOB_ID}.txt"
touch "${OUTFILE}"

{
	echo "Job ID: $SLURM_JOB_ID" 
	echo "Job name: $SLURM_JOB_NAME" 
	echo "N tasks: $SLURM_ARRAY_TASK_COUNT" 
	echo "N cores: $SLURM_CPUS_ON_NODE" 
	echo "N threads per code: $SLURM_THREADS_PER_CORE" 
	echo "Minimum memory required per CPU: $SLURM_MEM_PER_CPU" 
	echo "Requested memory per GPU: $SLURM_MEM_PER_GPU" 
} > "${OUTFILE}"

# the python code
# activate env
module load python/miniforge-25.3.0
source activate /project/bbcael/ambauer/learning_sai/.env

# run simulations
# for now, no ECS != 3.0 simulations, while i figure out beta calibrations for those
# python -m model.main_margobs_pco2geowc_ws ssp245 2025 1 10 2.0 0.1 50 1 500 1 
# python -m model.main_margobs_pco2geowc_ws ssp245 2025 1 18 2.0 0.1 50 1 500 1 
# python -m model.main_margobs_pco2geowc_ws ssp245 2025 1 25 2.0 0.1 50 1 500 1 

python -m model.main_margobs_pco2geowc_ws ssp245 2025 0 9 3.0 0.1 50 1 500 1 
python -m model.main_margobs_pco2geowc_ws ssp245 2025 0 12 3.0 0.1 50 1 500 1 
python -m model.main_margobs_pco2geowc_ws ssp245 2025 0 14 3.0 0.1 50 1 500 1 
python -m model.main_margobs_pco2geowc_ws ssp245 2025 0 17 3.0 0.1 50 1 500 1 
python -m model.main_margobs_pco2geowc_ws ssp245 2025 0 21 3.0 0.1 50 1 500 1 

# python -m model.main_margobs_pco2geowc_ws ssp245 2025 1 10 4.5 0.1 50 1 500 1 
# python -m model.main_margobs_pco2geowc_ws ssp245 2025 1 18 4.5 0.1 50 1 500 1 
# python -m model.main_margobs_pco2geowc_ws ssp245 2025 1 25 4.5 0.1 50 1 500 1 
