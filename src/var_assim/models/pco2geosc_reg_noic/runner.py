"""Strong-constraint counterpart of pco2geowc_reg_noic.

The random draws are made exactly as pco2geowc_reg_noic makes them: the truth's
model errors from the same seeds, and the full-length joint prior ensemble
[theta, q] from the full weak-constraint prior covariance. That gives member i
the same background theta_f in both models, and makes its background noise
draw xi_f the one weak member i started from. Only the use differs. Here xi_f
perturbs the observations and the noise covariance enters S, rather than q being
estimated.

With --sc_noise flux and the truth's noise, the pseudo-observations are
bit-identical to the weak run's.

Adam Michael Bauer
UChicago
"""

import os
import time
import warnings
import logging
import argparse

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import yaml
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
    NOISE_PATH,
    PERF_REPS_PATH,
    PRIOR_SEED,
    MOD_ERROR_SEED,
    REG1_NOISE_SEED,
    REG2_NOISE_SEED,
    SC_MEAS_NOISE_SEED,
)

# the weak model's forward model is used for the warm start and for the flux
# design's truth, so both are identical to the weak run
from var_assim.models.pco2geowc_reg_noic.dynamics import (
    get_nonlin_path as get_nonlin_path_wc,
)

from .cost import SCContext
from .dynamics import N_FIXED, get_forced_path
from .noise_response import OBS_VARS as NOISE_OBS_VARS, NoiseModel
from .obs import get_obs_from_dynamics
from .parallelization import EnsembleMember, runner_4dvar
from . import checks

SLURM_JOB_ID = os.environ.get("SLURM_JOB_ID", "local")

SC_DEFAULTS = {"sc_noise": "flux", "sc_covar": "marginal", "sc_obs_pert": "match_weak"}

VAR_NAMES = [
    "T1",
    "T2",
    "Q",
    "T_R1",
    "T_R2",
    "L",
    "G",
    "EPS",
    "C1",
    "C2",
    "F1_CO2",
    "ALPHA_R1",
    "ALPHA_R2",
    "BETA_R1",
    "BETA_R2",
]
OBS_NAMES = ["T1", "Q", "T_R1", "T_R2"]


def resolve_sc_args(args):
    """Fill in SC flag defaults and write them back to args.

    The flags are defined with default=argparse.SUPPRESS, so they are absent from
    the namespace of every other model. Their run metadata, and thus
    their output files, is therefore unchanged. They are set here, so this
    run's output attrs and filename record the resolved values.
    """

    for k, v in SC_DEFAULTS.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def load_t1_noise_std():
    """Innovation std of the direct temperature noise (--sc_noise temp), in K.

    Read from the `strong_constraint` block of noise.yaml rather than through
    ClimateModelNoise. ClimateModelNoise splats its noise_model block into a
    dataclass, so a new key there would break every other model.
    """

    with open(NOISE_PATH, "r") as f:
        data = yaml.safe_load(f)
    try:
        return float(data["strong_constraint"]["T1_NOISE_STD"])
    except KeyError as err:
        raise KeyError(
            "config/noise.yaml has no strong_constraint.T1_NOISE_STD, which "
            "--sc_noise temp needs"
        ) from err


def global_noise_scale(args, Noise):
    """Factor mapping the weak model's heat-flux noise draw onto this design's noise.

    flux: 1. temp: T1_NOISE_STD / INT_VAR_STD. The temperature noise has the same
    AR(1) correlation and a rescaled innovation std, so its covariance is exactly
    scale^2 times the weak one, and the scaled draw has the right distribution.
    """

    if args.sc_noise == "flux":
        return 1.0
    return load_t1_noise_std() / Noise.INT_VAR_STD


def noise_names(design, n_times):
    """Names for the noise vector [global, q_R1, q_R2], steps 1..n_times-1."""

    glob = "qAT_" if design == "flux" else "epsT_"
    return np.hstack(
        [
            [glob + str(i) for i in range(1, n_times)],
            ["qR1_" + str(i) for i in range(1, n_times)],
            ["qR2_" + str(i) for i in range(1, n_times)],
        ]
    )


def make_shared_draws(args, Prior, Noise, Windowing, DT=1.0):
    """The truth's model errors and the full-length joint prior ensemble.

    Line for line what pco2geowc_reg_noic/runner.py draws (same seeds, same
    full-length covariance, same order of RNG calls), so member i here starts
    from exactly the theta_f and noise background that weak member i did. Must be
    called after the warm start, which sets the prior centers the ensemble is
    drawn around.

    Returns
    -------
    draws: dict
        N_MAX, N_BLOCKS, mod_errors_full, mod_error_covar_full,
        mod_errors_r1_full, mod_errors_r2_full, theta_prior_full
    """

    N_MAX = get_window_max_timesteps(Windowing.windows, DT)
    N_BLOCKS = 1 + len(Noise.INT_T_REG_STD)

    mod_errors_rng = np.random.default_rng(seed=MOD_ERROR_SEED)
    mod_errors_full, mod_error_covar_full = gen_noise_ts(Noise, N_MAX, rng=mod_errors_rng)

    r1_rng = np.random.default_rng(seed=REG1_NOISE_SEED)
    r2_rng = np.random.default_rng(seed=REG2_NOISE_SEED)
    reg_covars_full = [
        get_covar_white(np.array([INT_T_REGx_STD] * N_MAX), N_MAX)
        for INT_T_REGx_STD in Noise.INT_T_REG_STD
    ]
    mod_errors_r1_full, mod_errors_r2_full = [
        rng.multivariate_normal(np.array([0.0] * N_MAX), covar)
        for rng, covar in zip((r1_rng, r2_rng), reg_covars_full)
    ]

    all_mod_errors_full = np.hstack(
        [mod_errors_full, mod_errors_r1_full, mod_errors_r2_full]
    )
    controls_cen_full = Prior.get_augmented_cen_vector(
        np.zeros_like(all_mod_errors_full)
    )
    prior_stds_full = Prior.get_augmented_std_vector(
        np.hstack(
            [
                np.ones_like(mod_errors_full),
                np.hstack(
                    [
                        [INT_T_REGx_STD] * N_MAX
                        for INT_T_REGx_STD in Noise.INT_T_REG_STD
                    ]
                ),
            ]
        )
    )
    inv_covar_prior_full = get_covar_white(
        prior_stds_full, len(prior_stds_full), inv=True
    )
    inv_covar_prior_full[N_FIXED : N_FIXED + N_MAX, N_FIXED : N_FIXED + N_MAX] = (
        np.linalg.inv(mod_error_covar_full)
    )

    prior_rng = np.random.default_rng(seed=PRIOR_SEED)
    theta_prior_full = get_prior_draws(
        args.model,
        controls_cen_full,
        np.linalg.inv(inv_covar_prior_full),
        args.n_ens,
        rng=prior_rng,
    )

    return {
        "N_MAX": N_MAX,
        "N_BLOCKS": N_BLOCKS,
        "mod_errors_full": mod_errors_full,
        "mod_error_covar_full": mod_error_covar_full,
        "mod_errors_r1_full": mod_errors_r1_full,
        "mod_errors_r2_full": mod_errors_r2_full,
        "theta_prior_full": theta_prior_full,
    }


def run_var_assim_experiment(
    logger: logging.Logger,
    args: argparse.Namespace,
    Prior: object,
    Truth: object,
    Noise: object,
    Windowing: object,
):
    resolve_sc_args(args)
    design, mode, pert_kind = args.sc_noise, args.sc_covar, args.sc_obs_pert
    logger.info(
        f"    > Strong constraint: noise design = {design}, covariance mode = "
        f"{mode}, obs perturbation = {pert_kind}"
    )

    if args.no_opt:
        c = None
        logger.info("    > Skipping dask cluster startup (--no_opt)")
    else:
        c = start_dask(logger)
        logger.info(f"    > {c}")

    DT = 1.0

    """WARM START MODULE.
    """
    logger.info("    > Starting warm start module")
    warm_start_simulation(logger, args, Truth, Prior, get_nonlin_path_wc)
    logger.info("    > Warm start complete")

    """ASSIMILATION MODULE
    """
    results_dict = {}

    # draws identical to pco2geowc_reg_noic/runner.py: once, at full length
    draws = make_shared_draws(args, Prior, Noise, Windowing, DT)
    N_MAX, N_BLOCKS = draws["N_MAX"], draws["N_BLOCKS"]
    mod_errors_full = draws["mod_errors_full"]
    mod_error_covar_full = draws["mod_error_covar_full"]
    mod_errors_r1_full = draws["mod_errors_r1_full"]
    mod_errors_r2_full = draws["mod_errors_r2_full"]
    theta_prior_full = draws["theta_prior_full"]

    # measurement-noise perturbations, drawn once at full length so every window
    # sees the same perturbation on the years they share
    if pert_kind == "plus_meas":
        meas_rng = np.random.default_rng(seed=SC_MEAS_NOISE_SEED)
        eta_full = meas_rng.standard_normal((args.n_ens, 4, N_MAX))
    else:
        eta_full = None

    scale = global_noise_scale(args, Noise)
    sig_obs = np.array(
        [Noise.OBS_T1_STD, Noise.OBS_Q_STD, *Noise.OBS_T_REG_STD], dtype=float
    )
    sig_reg = np.array(Noise.INT_T_REG_STD, dtype=float)

    # prior on the 15 fixed controls: the head block of the weak prior
    inv_covar_prior = get_covar_white(Prior.controls_std, N_FIXED, inv=True)

    # the prior center (after the warm start); where L is evaluated for the fixed
    # perturbation and, in "fixed" mode, for S
    theta_ref = np.asarray(Prior.controls_cen, dtype=float).copy()

    logger.info(
        f"        >> obs error stds ({Noise.OBS_WEIGHTING}): "
        f"T1={Noise.OBS_T1_STD:.4g}  Q={Noise.OBS_Q_STD:.4g}  "
        f"T_REG={np.round(Noise.OBS_T_REG_STD, 4).tolist()}"
    )
    logger.info(
        f"        >> global noise innovation std = {Noise.INT_VAR_STD * scale:.4g} "
        f"({'heat flux' if design == 'flux' else 'K'}), AR(1) corr = "
        f"{Noise.AUTO_CORR}; regional noise std = {sig_reg.tolist()} K"
    )

    for TMIN, TMAX in Windowing.windows:
        logger.info(f"    > Carrying out data assimilation for window {TMIN}-{TMAX}")

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
        M = N_timesteps - 1

        prefix_inds = get_window_prefix_inds(
            N_FIXED, N_BLOCKS, N_MAX, N_timesteps, q_offset=1
        )

        # truth noise for this window, sliced exactly as the weak model does
        q_AT_tr = mod_errors_full[1:N_timesteps]
        q_R1_tr = mod_errors_r1_full[1:N_timesteps]
        q_R2_tr = mod_errors_r2_full[1:N_timesteps]
        xi_tr = np.hstack([scale * q_AT_tr, q_R1_tr, q_R2_tr])

        controls_tr = np.asarray(Truth.controls_tr, dtype=float).copy()

        # truth path and pseudo-observations
        if design == "flux":
            path_wc, times = get_nonlin_path_wc(
                e,
                Truth.get_augmented_truth_vector(np.hstack([q_AT_tr, q_R1_tr, q_R2_tr])),
                TMIN,
                TMAX,
                DT=DT,
            )
            data_tr_p = path_wc[:N_FIXED]
        else:
            data_tr_p, times = get_forced_path(
                e, controls_tr, xi_tr, design, TMIN, TMAX, DT
            )
        obs = get_obs_from_dynamics(data_tr_p, noise=False)

        # noise covariance for this window. the AR(1) covariance is Toeplitz, so
        # this slice is exactly the length-M covariance, as in the weak model
        Q_g = mod_error_covar_full[1:N_timesteps, 1:N_timesteps] * scale**2
        noise = NoiseModel(design, sig_obs, sig_reg, Q_g, N_timesteps, DT)
        ref_factor = noise.factor(noise.L_g(theta_ref))

        # the prefix of the full-length ensemble: theta_f and xi_f of member i are
        # the ones weak member i started from (xi_f rescaled for "temp")
        theta_prior = theta_prior_full[:, prefix_inds]
        theta_f = theta_prior[:, :N_FIXED]
        xi_f = theta_prior[:, N_FIXED:].copy()
        xi_f[:, :M] *= scale

        if eta_full is None:
            meas = np.zeros((args.n_ens, 4 * N_timesteps))
        else:
            meas = (sig_obs[None, :, None] * eta_full[:, :, :N_timesteps]).reshape(
                args.n_ens, 4 * N_timesteps
            )

        # component checks on the final window, as in the weak model
        if args.check_components and TMAX == 2100:
            logger.info(
                "        >> (FLAGGED) Checking TLM, ADJ, and cost function gradient accuracy"
            )
            ctx0 = SCContext(noise, mode, xi_f[0], meas[0], ref_factor)

            # the harness stamps its own output directory; find it afterwards so
            # the model-specific results land next to the generic ones
            check_root = DATA_DIR / "checks" / args.model
            before = set(check_root.glob("*")) if check_root.exists() else set()
            run_component_checks(
                logger,
                args,
                e,
                controls_tr,
                TMIN,
                TMAX,
                cost_args=[
                    theta_f[0],
                    inv_covar_prior,
                    ctx0,
                    obs,
                    e,
                    TMIN,
                    TMAX,
                    DT,
                ],
            )
            new_dirs = sorted(set(check_root.glob("*")) - before)
            checks.run_sc_checks(
                logger,
                args,
                e,
                TMIN,
                TMAX,
                DT,
                Prior,
                Noise,
                mod_error_covar_full[1:N_timesteps, 1:N_timesteps],
                q_tr=np.hstack([q_AT_tr, q_R1_tr, q_R2_tr]),
                controls_tr=controls_tr,
                obs_wc=obs if design == "flux" else None,
                check_dir=new_dirs[-1] if len(new_dirs) == 1 else None,
            )
            logger.info("        >> (FLAGGED) Checks complete")
            logger.info(
                f"        >> (FLAGGED) .csv files saved to {DATA_DIR}/checks/{args.model}"
            )

        if args.no_opt:
            logger.info(
                "        >> (FLAGGED) --no_opt set; skipping assimilation for"
                f" window {TMIN}-{TMAX}"
            )
            continue

        tol = opt_config["tol"]
        max_iter = opt_config["max_iter"]

        e_scat = c.scatter(e, broadcast=True)

        ensemble_members = [
            EnsembleMember(
                theta_f[i],
                -1,
                tol,
                max_iter,
                TMIN,
                TMAX,
                DT,
                controls_tr,
                inv_covar_prior,
                SCContext(noise, mode, xi_f[i], meas[i], ref_factor),
                obs,
                times,
            )
            for i in range(args.n_ens)
        ]

        t0_assim = time.time()
        logger.info(
            "        >> (TIME INTENSIVE) Carrying out inner and outer loops of "
            "variational data assimilation"
        )
        with performance_report(
            filename=f"{PERF_REPS_PATH}/perf_report_{SLURM_JOB_ID}.html"
        ):
            opt_ensmems = run_ensemble(c, ensemble_members, e_scat, args, runner_4dvar)
        RUNTIME = time.time() - t0_assim
        logger.info(f"        >> Ensemble data assimilation solved in {RUNTIME} s")

        c.cancel(e_scat)

        logger.info("        >> Processing simulation output")

        ds = process_simulation_window(
            args,
            np.array(VAR_NAMES),
            OBS_NAMES,
            TMAX,
            opt_ensmems,
            obs,
            data_tr_p,
            controls_tr,
            opt_config,
            RUNTIME,
        )
        ds = attach_sc_output(
            ds,
            opt_ensmems,
            design,
            times,
            xi_tr,
            ref_factor,
            theta_ref,
            Noise,
            scale,
        )

        results_dict[str(TMAX)] = ds
        logger.info(f"        >> Process for window {TMIN}-{TMAX} complete")

    if results_dict:
        # a new model has no output directory yet, and the writer won't make one
        if args.save_output:
            (DATA_DIR / "output" / args.model).mkdir(parents=True, exist_ok=True)
        make_master_datatree(logger, args, results_dict)
    else:
        logger.info("    > No assimilation output to save (--no_opt)")


def attach_sc_output(ds, opt_ensmems, design, times, xi_tr, ref_factor, theta_ref,
                     Noise, scale):
    """Add the strong-constraint variables to a window's dataset.

    The estimated noise is kept out of `vari` on purpose: `vari` is the 15 real
    controls, and tools that assume weak-constraint q blocks would silently
    misread a run that pretended otherwise.
    """

    N = len(times)
    qnames = noise_names(design, N)
    obs_time = np.array([f"{v}_{int(t)}" for v in NOISE_OBS_VARS for t in times])

    ds = ds.assign_coords(
        qvar=("qvar", qnames),
        obs_time_a=("obs_time_a", obs_time),
        obs_time_b=("obs_time_b", obs_time),
    )
    ds["obs_pert"] = (
        ("ens_mem", "obs_var", "time"),
        np.array([m.obs_pert.reshape(4, N) for m in opt_ensmems]),
    )
    ds["model_error_hat"] = (
        ("ens_mem", "qvar"),
        np.array([m.xi_star for m in opt_ensmems]),
    )
    ds["model_error_truth"] = (("qvar",), xi_tr)
    ds["noise_covar_ref"] = (("obs_time_a", "obs_time_b"), ref_factor.dense())
    ds["controls_ref"] = (("vari",), theta_ref)

    ds["obs_pert"].attrs["description"] = (
        "perturbation added to the observations for each member (data - obs)"
    )
    ds["model_error_hat"].attrs["description"] = (
        "profiled noise xi* at the solution, in the design's units"
    )
    ds["noise_covar_ref"].attrs["description"] = (
        "observation-error covariance S = R_o + L Q L^T at the prior center, "
        "observable-major (T1, Q, T_R1, T_R2) x time"
    )
    ds["controls_ref"].attrs["description"] = "prior center theta_ref"

    ds.attrs["sc_global_noise_std"] = float(Noise.INT_VAR_STD * scale)
    ds.attrs["sc_global_noise_units"] = "heat flux" if design == "flux" else "K"
    ds.attrs["sc_regional_noise_std"] = list(map(float, Noise.INT_T_REG_STD))
    ds.attrs["sc_noise_auto_corr"] = float(Noise.AUTO_CORR)
    ds.attrs["obs_std"] = [
        float(Noise.OBS_T1_STD),
        float(Noise.OBS_Q_STD),
        *map(float, Noise.OBS_T_REG_STD),
    ]
    ds.attrs["OBS_WEIGHTING"] = Noise.OBS_WEIGHTING
    return ds
