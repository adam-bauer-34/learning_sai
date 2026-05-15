# Learning SAI: Variational Data Assimilation for Stratospheric Aerosol Injection

By: Adam Michael Bauer (ambauer [at] uchicago [dot] edu)

A Python research codebase for running ensemble 4DVAR experiments on a simplified climate-energy balance model.

This repository implements variational data assimilation experiments designed to estimate the decline in uncertainty for climate metrics under Stratospheric Aerosol Injection (SAI) and different socioeconomic pathways. It supports multiple reduced-order climate models, noise parametrizations, and configurable assimilation windows.

## Key features

- Ensemble 4DVAR assimilation for climate model parameter estimation
- Three supported model variants:
  - `pco2geowc`: two-region climate model with internal variability
  - `pco2geowc3`: three-region climate model
  - `pco2geowc_nn`: no noise model (two-region)
- Configurable observation noise models: `AR1`, `AR0`, and `nn`
- Regional noise support for regional temperature observations
- Warm start and ensemble-based initialization
- Dask-based parallel execution for ensemble evaluation
- Output organization with `data/`, `analysis/`, and `logs/`

## Repository structure

- `pyproject.toml` - package metadata and `var-assim` console entry point
- `src/var_assim/` - main package code
  - `main.py` - CLI entry point and experiment orchestration
  - `config.py` - CLI parser and YAML-backed configuration
  - `models/` - model definitions and assimilation runners
  - `calibration/` - truth, priors, noise, and windowing setup
  - `postprocessing.py`, `stats/`, `warm_start.py` - analysis and preconditioning
- `config/` - YAML configuration files for data paths, truth parameters, priors, noise, and assimilation windows
- `data/` - input/output data storage locations
- `analysis/` - notebooks, scripts, and figures for post-analysis
- `experiments/` - Slurm batch script examples for running jobs
- `logs/` - example runtime logs and error outputs

## Installation

Recommended installation method is editable install from the repository root:

```bash
cd /project/bbcael/ambauer/learning_sai
pip install -e .
```

This installs the package and the `var-assim` console script.

## Usage

Run the main variational assimilation experiment from the package entry point:

```bash
var-assim --model pco2geowc --scenario ssp245 --tmin 2025 --noise_model AR1 \
  --theta 14 --ecs 3.0 --deg_p_dec 0.1 --n_yrs_ramp 50 --n_ens 500
```

Or execute the module directly:

```bash
python -m var_assim.main --model pco2geowc --scenario ssp245 --noise_model AR1 \
  --theta 14 --ecs 3.0 --deg_p_dec 0.1 --n_yrs_ramp 50 --n_ens 500
```

### Common CLI options

- `--model` : One of `pco2geowc`, `pco2geowc3`, `pco2geowc_nn`
- `--scenario` : One of `ssp245`, `ssp585`
- `--tmin` : Experiment start year (default `2025`)
- `--noise_model` : One of `AR1`, `AR0`, `nn`
- `--theta` : True SAI angle parameter (`9`, `12`, `14`, `17`, `21`, or `33`)
- `--ecs` : Equilibrium climate sensitivity (default `3.0`)
- `--deg_p_dec` : SAI cooling in degrees Celsius per decade
- `--n_yrs_ramp` : Years to ramp SAI output
- `--n_ens` : Number of ensemble members
- `--save_output` : Save output data to disk
- `--reg_noise` : Enable regional observation noise
- `--check_components` : Run tangent linear / adjoint / gradient checks
- `--debug` : Enable verbose logging for debugging

## Configuration

The package loads additional settings from YAML files under `config/`:

- `config/dirs.yaml` - data and figure paths
- `config/truth.yaml` - true climate model parameters
- `config/priors.yaml` - prior parameter distributions
- `config/noise.yaml` - noise model and observation error settings
- `config/windowing.yaml` - assimilation window definitions

## Experiment workflow

The main pipeline performs:

1. CLI argument parsing and configuration validation
2. Warm start initialization
3. Ensemble generation and prior sampling
4. Observation synthesis and noise modeling
5. Dask-parallelized 4DVAR optimization
6. Postprocessing into datasets for analysis

## Notes

- The package logs execution metadata, including Git commit hash, for reproducibility.
- `analysis/` contains notebooks and scripts for visualizing experiment results.
- `experiments/` contains Slurm batch templates for running large parameter scans.

## License

This repository is licensed under the terms in `LICENSE`.

