#!/bin/bash
# submit.sh — wrapper around script.sbatch
# Usage: ./submit.sh [--scenario ssp585] [--model pco2geowc] [--noise_model AR1] ...
# Any args not provided fall back to the defaults from script.sbatch.

# --------------------------------
# Defaults (must match script.sbatch)
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

# --------------------------------
# Parse arguments
# --------------------------------
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
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# --------------------------------
# Build job name from all params
# --------------------------------
JOB_NAME="${SCENARIO}-${MODEL}-${NOISE_MODEL}-${WINDOWING}-ecs${ECS}-dpdec${DEG_P_DEC}-ens${N_ENS}"
mkdir -p logs

# --------------------------------
# Submit
# --------------------------------
echo "Submitting job: $JOB_NAME"

sbatch \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}_%A_%a.out" \
    --error="logs/${JOB_NAME}_%A_%a.err" \
    experiments/runner.sbatch \
        --model        "$MODEL"       \
        --scenario     "$SCENARIO"    \
        --noise_model  "$NOISE_MODEL" \
        --tmin         "$TMIN"        \
        --windowing    "$WINDOWING"   \
        --ecs          "$ECS"         \
        --deg_p_dec    "$DEG_P_DEC"   \
        --n_yrs_ramp   "$N_YRS_RAMP"  \
        --n_ens        "$N_ENS"