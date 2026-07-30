#!/bin/bash
# submit.sh — wrapper around script.sbatch

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
TIME="2:00:00"
REG_NOISE=false
SAI_RAMP="linear"

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
        --time)         TIME="$2";         shift 2 ;;
        --reg_noise)    REG_NOISE=true;    shift ;;
        --sai_ramp)     SAI_RAMP="$2";     shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# --------------------------------
# Resource allocation (model-dependent)
# --------------------------------
CPUS_PER_TASK=16   # default, matches runner.sbatch fallback
MEM="80G"          # default, matches runner.sbatch fallback

if [[ "$MODEL" == "pco2geowc3_reg" ]]; then
    CPUS_PER_TASK=40
    MEM="100G"
fi

# --------------------------------
# Build optional flags string
# --------------------------------
OPTIONAL_FLAGS=""                                         
[[ "$REG_NOISE" == true ]] && OPTIONAL_FLAGS="--reg_noise"

# --------------------------------
# Build job name from all params
# --------------------------------
JOB_NAME="${SCENARIO}-${MODEL}-${NOISE_MODEL}-${WINDOWING}-ecs${ECS}-dpdec${DEG_P_DEC}-ens${N_ENS}-ramp${SAI_RAMP}"
mkdir -p logs

# --------------------------------
# Submit
# --------------------------------
echo "Submitting job: $JOB_NAME (cpus-per-task=$CPUS_PER_TASK, mem=$MEM)"

sbatch \
    --time="$TIME" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
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
        --sai_ramp     "$SAI_RAMP"    \
        --n_ens        "$N_ENS"       \
        --time         "$TIME"        \
        $OPTIONAL_FLAGS