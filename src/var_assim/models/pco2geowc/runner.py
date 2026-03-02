"""Two layer model 4DVAR with internal variability.

Adam Michael Bauer
University of Illinois Urbana Champaign
8.23.2024
"""

import sys
import time
import warnings 
import logging
import argparse

# filter out runtime warnings which clog log files
# (they are natural in the scipy.minimize call)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

from var_assim.dask_setup import start_dask
from var_assim.warm_start import warm_start_simulation
from var_assim.emis import EmissionsBaseline 
from var_assim.model_errors import gen_noise_ts
from var_assim.tlm_adj_checks import run_component_checks
from var_assim.stats.covar import get_covar_white
from var_assim.stats.draws import get_prior_draws
from var_assim.postprocessing import process_simulation_window, make_master_datatree
from var_assim.config import opt_config, DATA_DIR

from var_assim.models.pco2geowc.dynamics import get_nonlin_path
from var_assim.models.pco2geowc.obs import get_obs_from_dynamics
from var_assim.models.pco2geowc.parallelization import EnsembleMember, runner_4dvar


def run_var_assim_experiment(logger: logging.Logger, args: argparse.Namespace,
                             Prior: object, Truth: object,
                             Noise: object, Windowing: object):
    # start dask
    c = start_dask(logger)
    logger.info(f"    > {c}")

    # set time discretization (always 1.0)
    DT = 1.0

    """WARM START MODULE.
    """
    logger.info("    > Starting warm start module")
    warm_start_simulation(logger, args, Truth, Prior, get_nonlin_path)
    logger.info("    > Warm start complete")

    """ASSIMILATION MODULE
    """
    # dictionary to make datatree out of later
    results_dict = {}

    for (TMIN, TMAX) in Windowing.windows:
        logger.info(f"    > Carrying out data assimilation for window {TMIN}-{TMAX}")

        # set seed so we get same draws for each assimilation window
        np.random.seed(1000)

        # make emissions baseline
        e = EmissionsBaseline(logger, args, TMIN, TMAX,
                              geo=True, Prior=Prior, Truth=Truth,
                              T_START=TMIN, T_END=TMIN + args.n_yrs_ramp,
                              print_level=2)

        # make model errors and their covariance matrix
        mod_errors, mod_error_covar = gen_noise_ts(Noise, len(e.conc['CO2']))

        # true vector of controls for this window
        controls_tr = Truth.get_augmented_truth_vector(mod_errors)

        if args.debug:
            logger.info(f"        >> (DEBUG) Controls vector: {controls_tr}")
            logger.info(f"        >> (DEBUG) Controls vector length: {len(controls_tr)}")
        
        # central value of priors on each parameter
        controls_cen = Prior.get_augmented_cen_vector(np.zeros_like(mod_errors))

        # make true data path over this time window
        data_tr_p, times = get_nonlin_path(e, controls_tr, TMIN, TMAX, DT=1.0)

        # make prior stds vector
        prior_stds = Prior.get_augmented_std_vector(np.ones(len(mod_errors)))

        # make inverse covariance matrices for white noise
        inv_covar_prior = get_covar_white(prior_stds,
                                          len(prior_stds),
                                          inv=True)

        # add in inverse covarianace matrix of model errors (which may not be
        # white, like the other parameters)
        inv_covar_prior[-len(mod_errors):,
                        -len(mod_errors):] = np.linalg.inv(mod_error_covar)

        # make observation error covariance matrices
        # global temp
        inv_covar_T1_obs = get_covar_white(np.array([Noise.OBS_T1_STD] *
                                                    len(times)),
                                           len(times), inv=True)

        # ocean heat content
        inv_covar_Q_obs = get_covar_white(np.array([Noise.OBS_Q_STD] *
                                                   len(times)), len(times),
                                          inv=True)
        
        # regions (in this case, 2)
        inv_covar_T_R1_obs, inv_covar_T_R2_obs = [
                get_covar_white(np.array([OBS_T_REGx_STD] * len(times)),
                                len(times), inv=True)
                                for OBS_T_REGx_STD in Noise.OBS_T_REG_STD
        ]

        # if there is regional noise, add it to the observations here
        if args.reg_noise:
            covar_T_R1_obs, covar_T_R2_obs = [
                get_covar_white(np.array([OBS_T_REGx_STD] * len(times)),
                                len(times), inv=False)
                                for OBS_T_REGx_STD in Noise.OBS_T_REG_STD
            ]
            
            # make observations from true data
            obs = get_obs_from_dynamics(data_tr_p, noise=True,
                                        noise_params=[(None, None),
                                                      (None, None),
                                                      (0.0, covar_T_R1_obs),
                                                      (0.0, covar_T_R2_obs)])
        
        else:
            # make observations from true data without any additional noise
            obs = get_obs_from_dynamics(data_tr_p, noise=False)

        # If flagged, check components of variational data assimilation module
        if args.check_components:
            logger.info("        >> (FLAGGED) Checking TLM, ADJ, and cost function gradient accuracy")
            run_component_checks(args, e, controls_tr, TMIN, TMAX,
                                 cost_args=[controls_tr,
                                            inv_covar_prior,
                                            inv_covar_T1_obs,
                                            inv_covar_Q_obs,
                                            inv_covar_T_R1_obs,
                                            inv_covar_T_R2_obs,
                                            obs,
                                            e,
                                            TMIN,
                                            TMAX,
                                            DT])
            logger.info(f"        >> (FLAGGED) Checks complete")
            logger.info(f"        >> (FLAGGED) .csv files saved to {DATA_DIR}/checks/{args.model}")
            
        # optimization parameters
        tol = opt_config['tol']
        max_iter = opt_config['max_iter']

        # give first guess at initial conditions
        theta_prior = get_prior_draws(controls_cen,
                                      np.linalg.inv(inv_covar_prior),
                                      args.n_ens)
        
        # Check on object sizes
        if args.debug:
            logger.info(f"        >> (DEBUG) Emissions object is of size: {sys.getsizeof(e) / 1e6} MB")

        # scatter emissions baseline class and true observations to each
        # dask worker 
        e_scat = c.scatter(e, broadcast=True)

        # make list of ensemble members
        ensemble_members = [EnsembleMember(theta_p,
                                           -1, tol, max_iter,
                                           TMIN, TMAX, DT, controls_tr,
                                           inv_covar_prior, inv_covar_T1_obs,
                                           inv_covar_Q_obs, inv_covar_T_R1_obs,
                                           inv_covar_T_R2_obs, obs, times)
                            for theta_p in theta_prior]

        if args.debug:
            m = ensemble_members[0]

            total = 0
            for v in vars(m).values():
                if hasattr(v, "nbytes"):
                    total += v.nbytes
                else:
                    total += sys.getsizeof(v)

            logger.info(f"        >> (DEBUG) Estimated total of on ensemble member: {total / 1e6} MB")
            logger.info(f"        >> (DEBUG) Memory overhead for entire ensemble: {total * args.n_ens / 1e6} MB")
        
        # solve the assimilation using dask
        t0_assim = time.time()

        # do dask evaluation of runner
        logger.info(f"        >> (TIME INTENSIVE) Carrying out inner and outer loops of variational data assimilation")

        # map and compute
        futures = [c.submit(runner_4dvar, m, e_scat)
                   for m in ensemble_members]
        
        # gather results
        opt_ensmems = c.gather(futures)

        t1_assim = time.time()

        RUNTIME = t1_assim - t0_assim
        logger.info(f"        >> Ensemble data assimilation solved in {RUNTIME} s")

        logger.info(f"        >> Processing simulation output")
        # process simulation output into 
        ds = process_simulation_window(args, TMAX,
                                       opt_ensmems, obs, data_tr_p,
                                       controls_tr, opt_config, RUNTIME)

        results_dict[str(TMAX)] = ds

        logger.info(f"        >> Process for window {TMIN}-{TMAX} complete")

    # synthesize datasets from each window into a datatree object
    make_master_datatree(logger, args, results_dict)
    