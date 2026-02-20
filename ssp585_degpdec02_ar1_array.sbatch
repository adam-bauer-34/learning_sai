#!/bin/bash

#SBATCH --job-name=ssp585-higeo-ar1
#SBATCH --output=logs/ssp585-higeo-ar1_%A_%a.out  # %A=MasterJobID, %a=ArrayIndex
#SBATCH --error=logs/ssp585-higeo-ar1_%A_%a.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=caslake

#SBATCH --time=24:00:00        

# --- ARRAY CONFIGURATION ---
#SBATCH --array=0-5            
#SBATCH --nodes=1
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G              
# ---------------------------

#SBATCH --mail-type=FAIL,END   
#SBATCH --mail-user=ambauer@rcc.uchicago.edu

# Ensure logs directory exists
mkdir -p logs

# 1. Define the parameter array
PARAMS=(9 12 14 17 21 33)
CURRENT_PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# --------------------------------
# Metadata header
# --------------------------------

echo "======================================"
echo " Job name        : $SLURM_JOB_NAME"
echo " Master job ID   : $SLURM_ARRAY_JOB_ID"
echo " Array task ID  : $SLURM_ARRAY_TASK_ID"
echo " Parameter value: $CURRENT_PARAM"
echo " Node list      : $SLURM_JOB_NODELIST"
echo " Cores allocated: $SLURM_CPUS_PER_TASK"
echo " Memory (MB)    : $SLURM_MEM_PER_NODE"
echo " Start time     : $(date)"
echo "======================================"
echo ""

# Load environment
module load python/miniforge-25.3.0
source activate /project/bbcael/ambauer/learning_sai/.env

# Prevent libraries from spawning their own threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting Array Task $SLURM_ARRAY_TASK_ID with Param $CURRENT_PARAM"

# 3. Run the python command
python -m model.main_margobs_pco2geowc_ws ssp585 2025 1 $CURRENT_PARAM 3.0 0.2 50 2 500 1 

EXIT_CODE=$?

echo ""
echo "--------------------------------------"
echo " End time  : $(date)"
echo " Exit code : $EXIT_CODE"
echo "--------------------------------------"

exit $EXIT_CODE
