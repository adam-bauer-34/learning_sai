"""Postprocessing module for variational data assimilation experiments.

Adam Michael Bauer
UChicago
Feb 2026
"""

import argparse
import logging

import xarray as xr
from datatree import DataTree

from var_assim.config import DATA_DIR


def process_simulation_iteration(logger, args, results_dict, Priors, Truth, Noise):
    
    if not args.debug:
        # save output
        x = 0

    else:
        # print output
        logging.info(dt)


def make_final_datatree(logger, args, dt_dict):
    
    # make datatree for final storage
    dt = DataTree.from_dict(dt_dict, 'TMAX')

    # save model output if desired
    if args.save_output:
        if not args.reg_noise:
            path = DATA_DIR / 'output' / args.model / (
                f'margobs_ws_{args.scenario}_{args.model}_TMIN{args.tmin}'
                f'_{args.noise_model}_THETA{args.theta}_ECS{args.ecs}'
                f'_DEGpDEC{args.deg_p_dec}_NYRSRAMP{args.n_yrs_ramp}_{args.windowing}_Nens{args.n_ens}'
            )
        
        else:
            path = DATA_DIR / 'output' / args.model / (
                f'margobs_ws_{args.scenario}_{args.model}_TMIN{args.tmin}'
                f'_{args.noise_model}+reg_THETA{args.theta}_ECS{args.ecs}'
                f'_DEGpDEC{args.deg_p_dec}_NYRSRAMP{args.n_yrs_ramp}_{args.windowing}_Nens{args.n_ens}'
            )

        # report status and save to netcdf file
        logger.info(f"Output saved to: {path}")
        dt.to_netcdf(filepath=path, mode='w', format='NETCDF4', engine='netcdf4')

    else:
        # print output
        logger.info(f"Model simulation results:\n{dt}")