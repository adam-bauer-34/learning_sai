#!/bin/bash
# submit_smip7_jobs.sh — ScenarioMIP7 CO2 pathway sweep for the two weak-constraint
# regional models (16 jobs). AR1, default --conc_stat (median)

set -euo pipefail

# Run from the repo root so submit_job.sh sees the same paths as a manual call
cd "$(dirname "$0")/.."

S=./experiments/submit_job.sh
C="--n_ens 1000 --windowing gradual --tmin 2025 --noise_model AR1"
R="--model pco2geowc_reg_noic --reg_noise"
R3="--time 5:00:00 --model pco2geowc3_reg_noic --reg_noise"

for M in "$R" "$R3"; do
  # --------------------------------
  # M-ext, all three ramps x DEGpDEC 0.1/0.2 (6 per model)
  # --------------------------------
  for rp in fast linear slow; do for dp in 0.1 0.2; do
    $S $M $C --scenario M-ext --sai_ramp $rp --deg_p_dec $dp
  done; done

  # --------------------------------
  # VLHO-ext and H-ext, linear ramp, DEGpDEC 0.1 (2 per model)
  # --------------------------------
  for sc in VLHO-ext H-ext; do
    $S $M $C --scenario $sc --sai_ramp linear --deg_p_dec 0.1
  done
done
