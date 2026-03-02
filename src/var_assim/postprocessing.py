"""Postprocessing module for variational data assimilation experiments.

Adam Michael Bauer
UChicago
Feb 2026
"""

import sys
import argparse
import logging

import xarray as xr
import numpy as np

from datatree import DataTree
from var_assim.config import DATA_DIR


def process_simulation_window(args: argparse.Namespace,
                              TMAX: int,
                              opt_ensmems,
                              obs: np.ndarray,
                              data_tr_p: np.ndarray,
                              controls_tr: np.ndarray,
                              opt_chars: dict,
                              RUNTIME: float) -> xr.Dataset:
    """Postprocess results from one assimilation window.

    Function unpacks the `opt_ensmems` list and makes a dataset at the end.

    Parameters
    ----------
    args: argparse.ArgumentParser
        command line arguments namespace

    TMAX: int
        upper bound of current assimilation window

    opt_ensmems: list
        list of optimized ensemble members using variational data assimilation
    
    obs: np.ndarray
        array of observations used to fit model to
    
    data_tr_p: np.ndarray
        the true path (before adding any observational noise to system)
    
    controls_tr: np.ndarray
        true values of control variable vector
    
    opt_chars: dict
        dictionary of optimization characteristics
    
    RUNTIME: float
        total runtime of assimilation window

    Returns
    -------
    ds: xr.Dataset
        dataset containing all model results
    """
    
    # make times list
    times = np.arange(args.tmin, TMAX + 1, 1)

    # parse optimal results into individual arrays
    data = np.array([m.data for m in opt_ensmems])
    controls = np.array([m.control for m in opt_ensmems])
    costs = np.array([m.cost for m in opt_ensmems])
    l2s = np.array([m.l2 for m in opt_ensmems])

    # history of data, controls, cost functions, and parameter L2 norms
    data_hist = np.array([m.data_hist for m in opt_ensmems])
    controls_hist = np.array([m.controls_hist for m in
                                opt_ensmems])
    cost_hist = np.array([m.cost_hist for m in opt_ensmems])
    l2s_hist = np.array([m.l2s_hist for m in opt_ensmems])

    # store flags
    flags = np.array([m.flag for m in opt_ensmems])

    # make dataset for this assimilation window and save to dictionary that
    # we'll use to make a datatree later

    # set variable names for saving based on which model equations we use
    if args.model != 'pco2geowc3':
        names = np.hstack([['T1', 'T2', 'Q', 'T_R1', 'T_R2', 'L', 'G', 'EPS', 'C1', 'C2', 'F1_CO2',
                            'ALPHA_R1', 'ALPHA_R2', 'BETA_R1', 'BETA_R2'],
                            ['q' + str(i) for i in range(len(times))]])
    
    else:
        names = np.hstack([['T1', 'T2', 'Q', 'T_R1', 'T_R2', 'L', 'G', 'EPS', 'C1', 'C2', 'F1_CO2',
                            'ALPHA_R1', 'ALPHA_R2', 'BETA_R1', 'BETA_R2', 'ALPHA_R3', 'BETA_R3'],
                            ['q' + str(i) for i in range(len(times))]])

    # make attributes dictionary for clear dataset metadata
    cli_dict = {k: str(v) if isinstance(v, bool) else v for k, v in vars(args).items()}
    attrs = (
        opt_chars | {'run_time': RUNTIME}
        | cli_dict
        | {'command': " ".join(sys.argv)}
    )

    # make dataset with simulation results and return
    ds = xr.Dataset(data_vars={'data_final': (['ens_mem', 'vari', 'time'],
                                                data),
                                'l2s': (['ens_mem'], l2s),
                                'costs': (['ens_mem'], costs),
                                'controls': (['ens_mem', 'vari'], controls),
                                'data_hist': (['ens_mem', 'vari', 'iter',
                                                'time'], data_hist),
                                'l2_hist': (['ens_mem', 'iter'], l2s_hist),
                                'cost_hist': (['ens_mem', 'iter'],
                                                cost_hist),
                                'controls_hist': (['ens_mem', 'vari',
                                                    'iter'],
                                                    controls_hist),
                                'flag': (['ens_mem'], flags),
                                'obs': (['obs_var', 'time'], obs),
                                'data_truth': (['vari', 'time'], data_tr_p),
                                'controls_truth': (['vari'], controls_tr)},
                    coords={'time': (['time'], times),
                            'iter': (['iter'], np.arange(0, opt_chars['max_iter'] + 1,
                                                            1)),
                            'vari': (['vari'], names),
                            'ens_mem': (['ens_mem'], np.arange(0, args.n_ens,
                                                                1)),
                            'obs_var': (['obs_var'], ['T1', 'Q', 'T_R1', 'T_R2'])},
                    attrs=attrs)
    
    return ds


def make_master_datatree(logger: logging.Logger, 
                        args: argparse.Namespace,
                        dt_dict: dict):
    """Make master datatree object to store simulation results.

    Parameters
    ----------
    logger: logging.Logger
        logging object

    args: argparse.ArgumentParser
        command line arguments
    
    dt_dict: dictionary
        dictionary of model results
        keys should be time upper bound of assimilation window
    """
    
    # make datatree for final storage
    dt = DataTree.from_dict(dt_dict, 'TMAX')

    # save model output if desired
    if args.save_output:
        if not args.reg_noise:
            path = DATA_DIR / 'output' / args.model / (
                f'var-assim-output_{args.scenario}_{args.model}_{args.windowing}_TMIN{args.tmin}'
                f'_{args.noise_model}_THETA{args.theta}_ECS{args.ecs}'
                f'_DEGpDEC{args.deg_p_dec}_NYRSRAMP{args.n_yrs_ramp}_{args.windowing}_Nens{args.n_ens}.nc'
            )
        
        else:
            path = DATA_DIR / 'output' / args.model / (
                f'var-assim-output_{args.scenario}_{args.model}_{args.windowing}_TMIN{args.tmin}'
                f'_{args.noise_model}+reg_THETA{args.theta}_ECS{args.ecs}'
                f'_DEGpDEC{args.deg_p_dec}_NYRSRAMP{args.n_yrs_ramp}_Nens{args.n_ens}.nc'
            )

        # report status and save to netcdf file
        logger.info(f"    > Output saved to: {path}")
        dt.to_netcdf(filepath=path, mode='w', format='NETCDF4', engine='netcdf4')

    else:
        # print output
        logger.info(f"    > Model simulation results:\n{dt}")