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
from var_assim.model_errors import (
    gen_noise_ts,
    get_window_max_timesteps,
    get_window_prefix_inds,
)
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
)

from var_assim.models.pco2geowc3.dynamics import get_nonlin_path
from var_assim.models.pco2geowc3.obs import get_obs_from_dynamics
from var_assim.models.pco2geowc3.parallelization import EnsembleMember, runner_4dvar

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
    warm_start_simulation(logger, args, Truth, Prior, get_nonlin_path)
    logger.info("    > Warm start complete")

    """ASSIMILATION MODULE
    """
    # dictionary to make datatree out of later
    results_dict = {}

    # The truth's model errors and the prior ensemble are drawn ONCE here, at the
    # length of the longest window, and each window takes a prefix of the block.
    # Drawing them inside the window loop does not reproduce the same values even
    # with the RNG re-seeded, because the dimension of the draw grows with the
    # window -- see get_window_prefix_inds for the mechanism and the measurements.
    # Slicing is exact, not approximate, because the AR(1) covariance is Toeplitz.
    N_MAX = get_window_max_timesteps(Windowing.windows, DT)
    N_BLOCKS = 1  # global model errors only; no regional noise in this model
    N_FIXED = 18

    mod_errors_rng = np.random.default_rng(seed=MOD_ERROR_SEED)
    mod_errors_full, mod_error_covar_full = gen_noise_ts(Noise, N_MAX, rng=mod_errors_rng)

    # full-length prior moments, needed only to draw the full-length ensemble
    controls_cen_full = Prior.get_augmented_cen_vector(np.zeros_like(mod_errors_full))
    prior_stds_full = Prior.get_augmented_std_vector(np.ones(len(mod_errors_full)))
    inv_covar_prior_full = get_covar_white(
        prior_stds_full, len(prior_stds_full), inv=True
    )
    inv_covar_prior_full[-N_MAX:, -N_MAX:] = np.linalg.inv(mod_error_covar_full)

    prior_rng = np.random.default_rng(seed=PRIOR_SEED)
    theta_prior_full = get_prior_draws(
        args.model,
        controls_cen_full,
        np.linalg.inv(inv_covar_prior_full),
        args.n_ens,
        rng=prior_rng,
    )

    for TMIN, TMAX in Windowing.windows:
        logger.info(f"    > Carrying out data assimilation for window {TMIN}-{TMAX}")

        # set seed so we get same draws for each assimilation window
        np.random.seed(1000)

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

        # take this window's prefix of the full-length draws made above, so that a
        # given ensemble member keeps the same parameters and initial conditions in
        # every window and the truth stays fixed on the overlapping span
        N_timesteps = len(e.conc["CO2"])
        prefix_inds = get_window_prefix_inds(N_FIXED, N_BLOCKS, N_MAX, N_timesteps)

        mod_errors = mod_errors_full[:N_timesteps]
        mod_error_covar = mod_error_covar_full[:N_timesteps, :N_timesteps]

        # true vector of controls for this window
        controls_tr = Truth.get_augmented_truth_vector(mod_errors)

        if args.debug:
            logger.info(f"        >> (DEBUG) Controls vector: {controls_tr}")
            logger.info(
                f"        >> (DEBUG) Controls vector length: {len(controls_tr)}"
            )

        # central value of priors on each parameter
        controls_cen = Prior.get_augmented_cen_vector(np.zeros_like(mod_errors))

        # make true data path over this time window
        data_tr_p, times = get_nonlin_path(e, controls_tr, TMIN, TMAX, DT=1.0)

        # make prior stds vector
        prior_stds = Prior.get_augmented_std_vector(np.ones(len(mod_errors)))

        # make inverse covariance matrices for white noise
        inv_covar_prior = get_covar_white(prior_stds, len(prior_stds), inv=True)

        # add in inverse covarianace matrix of model errors (which may not be
        # white, like the other parameters)
        inv_covar_prior[-len(mod_errors) :, -len(mod_errors) :] = np.linalg.inv(
            mod_error_covar
        )

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

        # if there is regional noise, add it to the observations here
        if args.reg_noise:
            covar_T_R1_obs, covar_T_R2_obs, covar_T_R3_obs = [
                get_covar_white(
                    np.array([OBS_T_REGx_STD] * len(times)), len(times), inv=False
                )
                for OBS_T_REGx_STD in Noise.OBS_T_REG_STD
            ]

            # make observations from true data
            obs = get_obs_from_dynamics(
                data_tr_p,
                noise=True,
                noise_params=[
                    (None, None),
                    (None, None),
                    (0.0, covar_T_R1_obs),
                    (0.0, covar_T_R2_obs),
                    (0.0, covar_T_R3_obs),
                ],
            )

        else:
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
            logger.info(f"        >> (FLAGGED) Checks complete")
            logger.info(
                f"        >> (FLAGGED) .csv files saved to {DATA_DIR}/checks/{args.model}"
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

        # give first guess at initial conditions. this is the prefix of the
        # full-length ensemble drawn before the window loop, so ensemble member i
        # has the same parameters and initial conditions in every window
        theta_prior = theta_prior_full[:, prefix_inds]

        # Check on object sizes
        if args.debug:
            logger.info(
                f"        >> (DEBUG) Emissions object is of size: {sys.getsizeof(e) / 1e6} MB"
            )

        logger.debug(
            f"    ! mean of parameter prior: {np.mean(theta_prior[:, :15], axis=0)}"
        )
        logger.debug(
            f"    ! median of parameter prior: {np.median(theta_prior[:, :15], axis=0)}"
        )
        logger.debug(
            f"    ! std of parameter prior: {np.std(theta_prior[:, :15], axis=0)}"
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

            logger.info(
                f"        >> (DEBUG) Estimated total of on ensemble member: {total / 1e6} MB"
            )
            logger.info(
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
                ["q" + str(i) for i in range(len(times))],
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

        results_dict[str(TMAX)] = ds

        logger.info(f"        >> Process for window {TMIN}-{TMAX} complete")

    # synthesize datasets from each window into a datatree object
    # (nothing to synthesize if the assimilation was skipped)
    if results_dict:
        make_master_datatree(logger, args, results_dict)
    else:
        logger.info("    > No assimilation output to save (--no_opt)")
