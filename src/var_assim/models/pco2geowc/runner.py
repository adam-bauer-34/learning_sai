"""Two layer model 4DVAR with internal variability.

Adam Michael Bauer
University of Illinois Urbana Champaign
8.23.2024

To run:
    python main_margobs_pco2sulwc.py [scenario] [P] [L or F] [SIGMA]
        [N_windows] [N_ENS] [SAVE_OUTPUT]
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

from var_assim.calibration.noise import ClimateModelNoise
from var_assim.calibration.priors import ClimateModelPriors
from var_assim.calibration.truth import ClimateModelTruth
from var_assim.calibration.windowing import AssimilationWindowing

from var_assim.dask_setup import start_dask
from var_assim.emis import EmissionsBaseline 
from var_assim.tlm_adj_checks import *
from var_assim.model_errors import gen_noise_ts
from var_assim.postprocessing import process_simulation_window, make_master_datatree
from var_assim.stats.covar import get_covar_white
from var_assim.stats.draws import get_prior_draws
from var_assim.warm_start import warm_start_simulation

from var_assim.models.pco2geowc.dynamics import get_nonlin_path
from var_assim.models.pco2geowc.obs import get_obs_from_dynamics
from var_assim.models.pco2geowc.parallelization import EnsembleMember, runner_4dvar


def run_var_assim_experiment():
    pass


def run_var_assim_experiment_wip(logger: logging.Logger, args: argparse.ArgumentParser,
                                 Prior: ClimateModelPriors, Truth: ClimateModelTruth,
                                 Noise: ClimateModelNoise, Windowing: AssimilationWindowing):
    # start dask
    c = start_dask(logger)
    logger.info(c)

    """WARM START MODULE.
    """
    logger.info("    Starting warm start module")
    warm_start_simulation(logger, args, Truth, Prior, get_nonlin_path)
    logger.info("    Warm start complete")

    """ASSIMILATION MODULE
    """
    # dictionary to make datatree out of later
    results_dict = {}

    for TMAX in Windowing.windows:
        logger.info(f"    Carrying out data assimilation for window {args.tmin}-{TMAX}")

        # set seed so we get same draws for each assimilation window
        np.random.seed(1000)

        # make emissions baseline
        e = EmissionsBaseline(SCENARIO, TMIN, TMAX,
                              geo=True, DEG_PER_DEC=DEG_PER_DEC,
                              LAMBDA=L_CEN, GAMMA=G_CEN, EPSILON=EPS_CEN, F_EFF_GEO=F_EFF_GEO_TR,
                              T_START=TMIN, T_END=TMIN + N_YEARS_RAMP)

        # make model errors and their covariance matrix
        mod_errors, mod_error_covar = gen_noise_ts(Noise, len(e.conc['CO2']))

        # true vector of controls for this window
        controls_tr = Truth.get_augmented_truth_vector(mod_errors)
        
        # central value of priors on each parameter
        controls_cen = Prior.get_augmented_cen_vector(np.zeros_like(mod_errors))

        # make true data path over this time window
        data_tr_p, times = get_nonlin_path(e, controls_tr, args.tmin, TMAX, DT=1.0)

        # note stds of priors and obs
        OBS_T1_STD = 1.0  # observation noise in measuring T1/T2
        OBS_Q_STD = 1.0  # observation noise in measuring ocean heat content
        OBS_T_R1_STD = 1.0  # observation noise in measuring regional temperature in R1
        OBS_T_R2_STD = 1.0  # observation noise in measuring regional temperature in R2

        T_IC_STD = 0.2  # initial condition std for t1 and t2 (roughly the size of internal variability)
        EPS_STD = 0.128  # pattern effect standard deviation (cummins, 2020)
        F1_STD = 0.519   # f1_co2 std, from zelinka et al. (2020)
        PRIOR_STD_FACTOR = 0.3  # implies X% std for prior for other less constrained parameters

        Q_STD = C1_TR * T_IC_STD + C2_TR * T_IC_STD  # std for OHC
        T_R1_STD = ALPHA_R1_TR * T_IC_STD  # std for temp in r1
        T_R2_STD = ALPHA_R2_TR * T_IC_STD  # std for temp in r2
        
        # make prior stds vector
        prior_stds = np.hstack([np.array([T_IC_STD, T_IC_STD,
                                          Q_STD, T_R1_STD, T_R2_STD]),
                                np.abs(theta_prior_cent[5:7]) * PRIOR_STD_FACTOR,
                                EPS_STD,
                                np.abs(theta_prior_cent[8:10]) * PRIOR_STD_FACTOR,
                                F1_STD, ALPHA_R1_STD, ALPHA_R2_STD, BETA_R1_STD, BETA_R2_STD,
                                np.ones(len(mod_errors))])

        # make inverse covariance matrices for white noise
        inv_covar_prior = get_covar_white(prior_stds,
                                          len(prior_stds),
                                          inv=True)

        # add in inverse covarianace matrix of model errors (which may not be
        # white, like the other parameters)
        inv_covar_prior[-len(mod_errors):,
                        -len(mod_errors):] = np.linalg.inv(mod_error_covar)

        # make observation error covariance matrices
        inv_covar_T1_obs = get_covar_white(np.array([OBS_T1_STD] *
                                                    len(times)),
                                           len(times), inv=True)

        inv_covar_Q_obs = get_covar_white(np.array([OBS_Q_STD] *
                                                   len(times)), len(times),
                                          inv=True)

        inv_covar_T_R1_obs = get_covar_white(np.array([OBS_T_R1_STD] *
                                                   len(times)), len(times),
                                            inv=True)

        inv_covar_T_R2_obs = get_covar_white(np.array([OBS_T_R2_STD] *
                                                   len(times)), len(times),
                                            inv=True)

        # make observations from true data
        obs = get_obs_from_dynamics(data_tr_p)

        # -----------------------------------------------
        # If desired, check tangent linear model accuracy
        # -----------------------------------------------
        if CHECK_TLM:
            # set (small) integration horizon and min/max perturbation sizes
            ALPHA_MIN = 1e-16
            ALPHA_MAX = 1.

            # check tlm and save output of that procedure
            _ = get_tlm_check(e, controls_tr, TMIN, TMAX, DT, ALPHA_MIN,
                              ALPHA_MAX, SAVE_RESULTS=True)

        
        # -----------------------------------------------
        # If desired, check adjoint accuracy
        # -----------------------------------------------
        if CHECK_ADJ:
            # Check 1: Adjoint Identity

            # do first check
            _ = get_adj_id_check(e, controls_tr, TMIN, TMAX, DT,
                                 SAVE_RESULTS=True)

            # Check 2: Gradient of Cost Function
            ALPHA_MIN = 1e-16
            ALPHA_MAX = 1.0

            # run check function
            _ = get_cost_grad_check(control=controls_tr * 1.1,
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
                                               DT],
                                    ALPHA_MIN=ALPHA_MIN,
                                    ALPHA_MAX=ALPHA_MAX,
                                    SAVE_RESULTS=True)

        
        # ----------------------------------------
        # Set up optimization
        # ----------------------------------------
        max_iter = 100  # maximum iterations
        tol = 0.001  # tolerance for convergence in 4DVAR

        # give first guess at initial conditions
        theta_prior = get_prior_draws(theta_prior_cent,
                                      np.linalg.inv(inv_covar_prior),
                                      N_ENS)
        
        # Check on object sizes
        #print("emissions object is:")
        #print(sys.getsizeof(e))
        #print(asizeof.asizeof(e) / 1e6)

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

        m = ensemble_members[0]

        print("Python object overhead:", sys.getsizeof(m) / 1e6, "MB")

        total = 0
        for v in vars(m).values():
            if hasattr(v, "nbytes"):
                total += v.nbytes
            else:
                total += sys.getsizeof(v)

        print("Estimated total size:", total / 1e6, "MB")
        
        #for i, ee in enumerate(ensemble_members):
        #    print(i, asizeof.asizeof(ee) / 1e6, " MB")

        # solve the assimilation using dask
        t0 = time.time()

        # do dask evaluation of runner
        print("Solving 4DVAR using DASK...")

        # map and compute
        futures = [c.submit(runner_4dvar, m, e_scat)
                   for m in ensemble_members]
        
        # gather results
        opt_ensmems = c.gather(futures)

        t1 = time.time()

        RUNTIME = t1 - t0
        # print(t1 - t0)

        # process simulation output into 
        ds = process_simulation_window(logger, args, TMAX,
                                       opt_ensmems, obs, data_tr_p,
                                       controls_tr, opt_chars, RUNTIME)

        results_dict[str(TMAX)] = ds

    # synthesize datasets from each window into a datatree object
    make_master_datatree(logger, args, results_dict)
    