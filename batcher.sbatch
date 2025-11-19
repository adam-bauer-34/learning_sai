#!/bin/bash

#SBATCH --job-name=hi-angle
#SBATCH --output=hi-angle.out
#SBATCH --error=hi-angle.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=broadwl

#SBATCH --time=24:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=28

#SBATCH --mail-type=ALL
#SBATCH --mail-user=ambauer@uchicago.edu

# the actual code
touch $SLURM_JOB_ID.txt

echo "Job ID: $SLURM_JOB_ID" >> $SLURM_JOB_ID.txt
echo "Job name: $SLURM_JOB_NAME" >> $SLURM_JOB_ID.txt
echo "N tasks: $SLURM_ARRAY_TASK_COUNT" >> $SLURM_JOB_ID.txt
echo "N cores: $SLURM_CPUS_ON_NODE" >> $SLURM_JOB_ID.txt
echo "N threads per code: $SLURM_THREADS_PER_CORE" >> $SLURM_JOB_ID.txt
echo "Minimum memory required per CPU: $SLURM_MEM_PER_CPU" >> $SLURM_JOB_ID.txt
echo "Requested memory per GPU: $SLURM_MEM_PER_GPU" >> $SLURM_JOB_ID.txt

# the python code
# activate env
module load python/miniforge-25.3.0
source activate /scratch/midway3/ambauer/conda_envs

# run simulations
python -m model.main_margobs_pco2geowc_ws ssp245 2020 1 0.1 50 9 250 1
