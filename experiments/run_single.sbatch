#!/bin/bash

#SBATCH --job-name=ssp245-logeo-ar1-single
#SBATCH --output=logs/ssp245-logeo-ar1_single_%j.out
#SBATCH --error=logs/ssp245-logeo-ar1_single_%j.err

#SBATCH --account=pi-bbcael
#SBATCH --partition=caslake

#SBATCH --time=1:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G

#SBATCH --mail-user=ambauer@rcc.uchicago.edu

# Ensure logs directory exists
cd /project/bbcael/ambauer/learning_sai
mkdir -p logs

# --------------------------------
# Argument parsing (with defaults)
# --------------------------------

MODEL="pco2geowc"
SCENARIO="ssp245"
NOISE_MODEL="AR1"
TMIN=2025
WINDOWING="original"
ECS=3.0
DEG_P_DEC=0.1
N_YRS_RAMP=50
N_ENS=500
THETA=15

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2";        shift 2 ;;
        --scenario)     SCENARIO="$2";     shift 2 ;;
        --noise_model)  NOISE_MODEL="$2";  shift 2 ;;
        --tmin)         TMIN="$2";         shift 2 ;;
        --windowing)    WINDOWING="$2";    shift 2 ;;
        --ecs)          ECS="$2";          shift 2 ;;
        --deg_p_dec)    DEG_P_DEC="$2";    shift 2 ;;
        --n_yrs_ramp)   N_YRS_RAMP="$2";   shift 2 ;;
        --n_ens)        N_ENS="$2";        shift 2 ;;
        --theta)        THETA="$2";        shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# --------------------------------
# Metadata header
# --------------------------------

echo "======================================"
echo " Job name        : $SLURM_JOB_NAME"
echo " Job ID          : $SLURM_JOB_ID"
echo " Parameter value : $THETA"
echo " Node list       : $SLURM_JOB_NODELIST"
echo " Cores allocated : $SLURM_CPUS_PER_TASK"
echo " Memory (MB)     : $SLURM_MEM_PER_NODE"
echo " Start time      : $(date)"
echo "--------------------------------------"
echo " model           : $MODEL"
echo " scenario        : $SCENARIO"
echo " noise_model     : $NOISE_MODEL"
echo " tmin            : $TMIN"
echo " windowing       : $WINDOWING"
echo " ecs             : $ECS"
echo " deg_p_dec       : $DEG_P_DEC"
echo " n_yrs_ramp      : $N_YRS_RAMP"
echo " n_ens           : $N_ENS"
echo "======================================"
echo ""

# Load environment
module load python/miniforge-25.3.0
source activate /project/bbcael/ambauer/envs/var_assim_sai

# Prevent libraries from spawning their own threads
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting single run with theta=$THETA"

var-assim \
    --model        "$MODEL"       \
    --scenario     "$SCENARIO"    \
    --noise_model  "$NOISE_MODEL" \
    \
    --tmin         "$TMIN"        \
    --windowing    "$WINDOWING"   \
    \
    --theta        "$THETA"       \
    --ecs          "$ECS"         \
    \
    --deg_p_dec    "$DEG_P_DEC"   \
    --n_yrs_ramp   "$N_YRS_RAMP"  \
    \
    --n_ens        "$N_ENS"       \
    --save_output

EXIT_CODE=$?

echo ""
echo "--------------------------------------"
echo " End time  : $(date)"
echo " Exit code : $EXIT_CODE"
echo "--------------------------------------"

exit $EXIT_CODE