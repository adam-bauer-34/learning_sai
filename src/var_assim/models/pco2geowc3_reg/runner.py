"""Two layer model 4DVAR with internal variability.

Adam Michael Bauer
University of Illinois Urbana Champaign
8.23.2024
"""

import os
import sys
import time
import warnings
import logging
import argparse

# filter out runtime warnings which clog log files
# (they are natural in the scipy.minimize call)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from dask.distributed import performance_report

from var_assim.dask import start_dask, run_ensemble
from var_assim.warm_start import warm_start_simulation
from var_assim.emis import EmissionsBaseline
from var_assim.model_errors import gen_noise_ts
from var_assim.tlm_adj_checks import run_component_checks
from var_assim.stats.covar import get_covar_white
from var_assim.stats.draws import get_prior_draws
from var_assim.postprocessing import process_simulation_window, make_master_datatree
from var_assim.config import (
    opt_config,
    DATA_DIR,
    PERF_REPS_PATH,
    PRIOR_SEED,
    MOD_ERROR_SEED,
    REG1_NOISE_SEED,
    REG2_NOISE_SEED,
    REG3_NOISE_SEED,
)

from var_assim.models.pco2geowc3_reg.dynamics import get_nonlin_path
from var_assim.models.pco2geowc3_reg.obs import get_obs_from_dynamics
from var_assim.models.pco2geowc3_reg.parallelization import EnsembleMember, runner_4dvar

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")


def run_var_assim_experiment(
    logger: logging.Logger,
    args: argparse.Namespace,
    Prior: object,
    Truth: object,
    Noise: object,
    Windowing: object,
):
    # start dask, unless the assimilation is being skipped: the client is
    # only touched by the optimization block, so a checks-only run has no
    # reason to pay for cluster startup and teardown
    if args.no_opt:
        c = None
        logger.info("    > Skipping dask cluster startup (--no_opt)")
    else:
        c = start_dask(logger)
        logger.info(f"    > {c}")

    # set time discretization (always 1.0)
    DT = 1.0

    """WARM START MODULE.
    """
    logger.info("    > Starting warm start module")
    t0 = time.time()
    warm_start_simulation(logger, args, Truth, Prior, get_nonlin_path)
    logger.debug(f"    ! Warm start took {time.time() - t0} s")
    logger.info("    > Warm start complete")

    """ASSIMILATION MODULE
    """
    # dictionary to make datatree out of later
    results_dict = {}

    for TMIN, TMAX in Windowing.windows:
        logger.info(f"    > Carrying out data assimilation for window {TMIN}-{TMAX}")

        # make emissions baseline
        e = EmissionsBaseline(
            logger,
            args,
            TMIN,
            TMAX,
            geo=True,
            Prior=Prior,
            Truth=Truth,
            T_START=TMIN,
            T_END=TMIN + args.n_yrs_ramp,
            print_level=2,
        )

        N_timesteps = len(e.conc["CO2"])

        t0 = time.time()
        # make global model errors and their covariance matrix
        mod_errors_rng = np.random.default_rng(seed=MOD_ERROR_SEED)
        mod_errors, mod_error_covar = gen_noise_ts(
            Noise, N_timesteps, rng=mod_errors_rng
        )

        # make regional covariance matrices
        r1_rng = np.random.default_rng(seed=REG1_NOISE_SEED)
        r2_rng = np.random.default_rng(seed=REG2_NOISE_SEED)
        r3_rng = np.random.default_rng(seed=REG3_NOISE_SEED)

        # regions (in this case, 2)
        R1_mod_error_covar, R2_mod_error_covar, R3_mod_error_covar = [
            get_covar_white(np.array([INT_T_REGx_STD] * N_timesteps), N_timesteps)
            for INT_T_REGx_STD in Noise.INT_T_REG_STD
        ]

        # make model error time series
        mod_errors_r1 = r1_rng.multivariate_normal(
            np.array([0.0] * N_timesteps), R1_mod_error_covar
        )

        mod_errors_r2 = r2_rng.multivariate_normal(
            np.array([0.0] * N_timesteps), R2_mod_error_covar
        )

        mod_errors_r3 = r3_rng.multivariate_normal(
            np.array([0.0] * N_timesteps), R3_mod_error_covar
        )

        logger.debug(f"    ! region 1 model errors std: {np.std(mod_errors_r1)}")
        logger.debug(f"    ! region 2 model errors std: {np.std(mod_errors_r2)}")
        logger.debug(f"    ! region 3 model errors std: {np.std(mod_errors_r3)}")

        # combine all model errors into one long vector
        all_mod_errors = np.hstack(
            [mod_errors, mod_errors_r1, mod_errors_r2, mod_errors_r3]
        )

        # true vector of controls for this window
        controls_tr = Truth.get_augmented_truth_vector(all_mod_errors)

        if args.debug:
            logger.info(f"        >> (DEBUG) Controls vector: {controls_tr}")
            logger.info(
                f"        >> (DEBUG) Controls vector length: {len(controls_tr)}"
            )

        # central value of priors on each parameter
        controls_cen = Prior.get_augmented_cen_vector(np.zeros_like(all_mod_errors))

        # make true data path over this time window
        data_tr_p, times = get_nonlin_path(e, controls_tr, TMIN, TMAX, DT=1.0)

        # make prior stds vector
        regional_stds = np.hstack(
            [[INT_T_REGx_STD] * N_timesteps for INT_T_REGx_STD in Noise.INT_T_REG_STD]
        )
        # NOTE: mod errors can be red, so insert dummy here an insert their inverse cov
        # matrix later
        full_std_vector = np.hstack([np.ones_like(mod_errors), regional_stds])

        # make full stds vector
        prior_stds = Prior.get_augmented_std_vector(full_std_vector)
        logger.debug(f"    ! full std vector for prior: {prior_stds}")

        # make inverse covariance matrices for white noise
        inv_covar_prior = get_covar_white(prior_stds, len(prior_stds), inv=True)

        # add in inverse covarianace matrix of model errors (which may not be
        # white, like the other parameters)
        inv_covar_prior[18 : 18 + len(mod_errors), 18 : 18 + len(mod_errors)] = (
            np.linalg.inv(mod_error_covar)
        )
        logger.debug(f"    ! inverse covariance matrix for prior: {inv_covar_prior}")

        # make observation error covariance matrices
        # global temp
        inv_covar_T1_obs = get_covar_white(
            np.array([Noise.OBS_T1_STD] * len(times)), len(times), inv=True
        )

        # ocean heat content
        inv_covar_Q_obs = get_covar_white(
            np.array([Noise.OBS_Q_STD] * len(times)), len(times), inv=True
        )

        # regions (in this case, 2)
        inv_covar_T_R1_obs, inv_covar_T_R2_obs, inv_covar_T_R3_obs = [
            get_covar_white(
                np.array([OBS_T_REGx_STD] * len(times)), len(times), inv=True
            )
            for OBS_T_REGx_STD in Noise.OBS_T_REG_STD
        ]
        logger.debug(
            f"    ! noise setup and covariance matrix formulation took {time.time() - t0} s"
        )

        # make observations from true data without any additional noise
        obs = get_obs_from_dynamics(data_tr_p, noise=False)

        # If flagged, check components of variational data assimilation module
        if args.check_components and TMAX == 2100:
            logger.info(
                "        >> (FLAGGED) Checking TLM, ADJ, and cost function gradient accuracy"
            )
            run_component_checks(
                logger,
                args,
                e,
                controls_tr,
                TMIN,
                TMAX,
                cost_args=[
                    controls_tr,
                    inv_covar_prior,
                    inv_covar_T1_obs,
                    inv_covar_Q_obs,
                    inv_covar_T_R1_obs,
                    inv_covar_T_R2_obs,
                    inv_covar_T_R3_obs,
                    obs,
                    e,
                    TMIN,
                    TMAX,
                    DT,
                ],
            )
            logger.info(f"            >>> (FLAGGED) Checks complete")
            logger.info(
                f"            >>> (FLAGGED) .csv files saved to {DATA_DIR}/checks/{args.model}"
            )

        # if flagged, skip the optimization entirely. this has to skip per
        # window rather than exiting after the checks, because the checks are
        # gated on the final window and everything before it would otherwise
        # still be assimilated.
        if args.no_opt:
            logger.info(
                "        >> (FLAGGED) --no_opt set; skipping assimilation for"
                f" window {TMIN}-{TMAX}"
            )
            continue

        # optimization parameters
        tol = opt_config["tol"]
        max_iter = opt_config["max_iter"]

        # give first guess at initial conditions
        t0 = time.time()
        prior_rng = np.random.default_rng(seed=PRIOR_SEED)
        theta_prior = get_prior_draws(
            args.model,
            controls_cen,
            np.linalg.inv(inv_covar_prior),
            args.n_ens,
            rng=prior_rng,
        )
        logger.debug(f"    ! generating prior draws took {time.time() - t0} s")

        logger.debug(
            f"    ! mean of parameter prior: {np.mean(theta_prior[:, :18], axis=0)}"
        )
        logger.debug(
            f"    ! median of parameter prior: {np.median(theta_prior[:, :18], axis=0)}"
        )
        logger.debug(
            f"    ! std of parameter prior: {np.std(theta_prior[:, :18], axis=0)}"
        )

        # Check on object sizes
        if args.debug:
            logger.info(
                f"        >> (DEBUG) Emissions object is of size: {sys.getsizeof(e) / 1e6} MB"
            )

        # scatter emissions baseline class and true observations to each
        # dask worker
        e_scat = c.scatter(e, broadcast=True)

        # make list of ensemble members
        ensemble_members = [
            EnsembleMember(
                theta_p,
                -1,
                tol,
                max_iter,
                TMIN,
                TMAX,
                DT,
                controls_tr,
                inv_covar_prior,
                inv_covar_T1_obs,
                inv_covar_Q_obs,
                inv_covar_T_R1_obs,
                inv_covar_T_R2_obs,
                inv_covar_T_R3_obs,
                obs,
                times,
            )
            for theta_p in theta_prior
        ]

        if args.debug:
            m = ensemble_members[0]

            total = 0
            for v in vars(m).values():
                if hasattr(v, "nbytes"):
                    total += v.nbytes
                else:
                    total += sys.getsizeof(v)

            logger.debug(
                f"        >> (DEBUG) Estimated total of on ensemble member: {total / 1e6} MB"
            )
            logger.debug(
                f"        >> (DEBUG) Memory overhead for entire ensemble: {total * args.n_ens / 1e6} MB"
            )

        # solve the assimilation using dask
        t0_assim = time.time()

        # do dask evaluation of runner
        logger.info(
            f"        >> (TIME INTENSIVE) Carrying out inner and outer loops of variational data assimilation"
        )

        with performance_report(
            filename=f"{PERF_REPS_PATH}/perf_report_{SLURM_JOB_ID}.html"
        ):
            opt_ensmems = run_ensemble(c, ensemble_members, e_scat, args, runner_4dvar)

        t1_assim = time.time()

        RUNTIME = t1_assim - t0_assim
        logger.info(f"        >> Ensemble data assimilation solved in {RUNTIME} s")

        logger.info(f"        >> Processing simulation output")

        # cancel scattered emissions
        c.cancel(e_scat)

        # make variable names list for saving
        var_names = np.hstack(
            [
                [
                    "T1",
                    "T2",
                    "Q",
                    "T_R1",
                    "T_R2",
                    "T_R3",
                    "L",
                    "G",
                    "EPS",
                    "C1",
                    "C2",
                    "F1_CO2",
                    "ALPHA_R1",
                    "ALPHA_R2",
                    "ALPHA_R3",
                    "BETA_R1",
                    "BETA_R2",
                    "BETA_R3",
                ],
                ["qAT_" + str(i) for i in range(len(times))],
                ["qR1_" + str(i) for i in range(len(times))],
                ["qR2_" + str(i) for i in range(len(times))],
                ["qR3_" + str(i) for i in range(len(times))],
            ]
        )

        obs_names = ["T1", "Q", "T_R1", "T_R2", "T_R3"]

        # process simulation output into
        ds = process_simulation_window(
            args,
            var_names,
            obs_names,
            TMAX,
            opt_ensmems,
            obs,
            data_tr_p,
            controls_tr,
            opt_config,
            RUNTIME,
        )

        logger.debug(
            f"    ! mean estimate of lambda: {ds.controls.sel(vari='L').mean('ens_mem').values}"
        )
        logger.debug(
            f"        !! distance from truth: {ds.controls.sel(vari='L').mean('ens_mem').values - ds.controls_truth.sel(vari='L').values}"
        )
        logger.debug(
            f"    ! mean estimate of gamma: {ds.controls.sel(vari='G').mean('ens_mem').values}"
        )
        logger.debug(
            f"        !! distance from truth: {ds.controls.sel(vari='G').mean('ens_mem').values - ds.controls_truth.sel(vari='G').values}"
        )
        logger.debug(
            f"    ! mean estimate of epsilon: {ds.controls.sel(vari='EPS').mean('ens_mem').values}"
        )
        logger.debug(
            f"        !! distance from truth: {ds.controls.sel(vari='EPS').mean('ens_mem').values - ds.controls_truth.sel(vari='EPS').values}"
        )
        logger.debug(
            f"    ! mean estimate of alpha r1: {ds.controls.sel(vari='ALPHA_R1').mean('ens_mem').values}"
        )
        logger.debug(
            f"        !! distance from truth: {ds.controls.sel(vari='ALPHA_R1').mean('ens_mem').values - ds.controls_truth.sel(vari='ALPHA_R1').values}"
        )
        logger.debug(
            f"    ! mean estimate of alpha r2: {ds.controls.sel(vari='ALPHA_R2').mean('ens_mem').values}"
        )
        logger.debug(
            f"        !! distance from truth: {ds.controls.sel(vari='ALPHA_R2').mean('ens_mem').values - ds.controls_truth.sel(vari='ALPHA_R2').values}"
        )
        logger.debug(
            f"    ! mean estimate of alpha r3: {ds.controls.sel(vari='ALPHA_R3').mean('ens_mem').values}"
        )
        logger.debug(
            f"        !! distance from truth: {ds.controls.sel(vari='ALPHA_R3').mean('ens_mem').values - ds.controls_truth.sel(vari='ALPHA_R3').values}"
        )
        logger.debug(
            f"    ! mean estimate of beta r1: {ds.controls.sel(vari='BETA_R1').mean('ens_mem').values}"
        )
        logger.debug(
            f"        !! distance from truth: {ds.controls.sel(vari='BETA_R1').mean('ens_mem').values - ds.controls_truth.sel(vari='BETA_R1').values}"
        )
        logger.debug(
            f"    ! mean estimate of beta r2: {ds.controls.sel(vari='BETA_R2').mean('ens_mem').values}"
        )
        logger.debug(
            f"        !! distance from truth: {ds.controls.sel(vari='BETA_R2').mean('ens_mem').values - ds.controls_truth.sel(vari='BETA_R2').values}"
        )
        logger.debug(
            f"    ! mean estimate of beta r3: {ds.controls.sel(vari='BETA_R3').mean('ens_mem').values}"
        )
        logger.debug(
            f"        !! distance from truth: {ds.controls.sel(vari='BETA_R3').mean('ens_mem').values - ds.controls_truth.sel(vari='BETA_R3').values}"
        )

        results_dict[str(TMAX)] = ds

        logger.info(f"        >> Process for window {TMIN}-{TMAX} complete")

    # synthesize datasets from each window into a datatree object
    # (nothing to synthesize if the assimilation was skipped)
    if results_dict:
        make_master_datatree(logger, args, results_dict)
    else:
        logger.info("    > No assimilation output to save (--no_opt)")
