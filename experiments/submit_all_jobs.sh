#!/bin/bash
# submit_all_jobs.sh — resubmit the full gradual-windowing ensemble sweep (32 jobs)

set -euo pipefail

# Run from the repo root so submit_job.sh sees the same paths as a manual call
cd "$(dirname "$0")/.."

S=./experiments/submit_job.sh
C="--n_ens 1000 --windowing gradual --tmin 2025"
R="--model pco2geowc_reg_noic --reg_noise"
R3="--time 5:00:00 --model pco2geowc3_reg_noic --reg_noise"

# --------------------------------
# pco2geowc_reg_noic, linear ramp (8)
# --------------------------------
for sc in ssp245 ssp585; do for nm in AR0 AR1; do for dp in 0.1 0.2; do
  $S $R $C --scenario $sc --noise_model $nm --deg_p_dec $dp
done; done; done

# --------------------------------
# pco2geowc_reg_noic, fast/slow ramps, AR1 (6)
# --------------------------------
for rp in fast slow; do
  $S $R $C --scenario ssp245 --noise_model AR1 --deg_p_dec 0.1 --sai_ramp $rp
  $S $R $C --scenario ssp245 --noise_model AR1 --deg_p_dec 0.2 --sai_ramp $rp
  $S $R $C --scenario ssp585 --noise_model AR1 --deg_p_dec 0.1 --sai_ramp $rp
done

# --------------------------------
# pco2geowc3_reg_noic, linear ramp, 5 h wall time (8)
# --------------------------------
for sc in ssp245 ssp585; do for nm in AR0 AR1; do for dp in 0.1 0.2; do
  $S $R3 $C --scenario $sc --noise_model $nm --deg_p_dec $dp
done; done; done

# --------------------------------
# pco2geowc_nn and pco2geowc3_nn, linear ramp (8)
# --------------------------------
for m in pco2geowc_nn pco2geowc3_nn; do for sc in ssp245 ssp585; do for dp in 0.1 0.2; do
  $S --model $m --noise_model nn $C --scenario $sc --deg_p_dec $dp
done; done; done

# --------------------------------
# pco2geowc_nn, fast/slow ramps, ssp245, deg_p_dec 0.1 (2)
# --------------------------------
for rp in fast slow; do
  $S --model pco2geowc_nn --noise_model nn $C --scenario ssp245 --deg_p_dec 0.1 --sai_ramp $rp
done
