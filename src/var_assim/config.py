"""Config file that loads paths from YAML.

Adam Bauer
UChicago
Jan 2026
"""

import yaml
import argparse
import warnings

from pathlib import Path

# setup directories for data, saving, and other configurations
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "dirs.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = Path(CONFIG["DATA_DIR"])
DATA_DIR_ABS = Path(CONFIG["DATA_DIR_ABS"])
FIGS_DIR = Path(CONFIG["FIGS_DIR"])
PERF_REPS_PATH = Path(CONFIG["PERF_REPS_PATH"])

TRUTH_PATH = CONFIG_PATH.parent.parent / Path(CONFIG["TRUTH_PATH"])
PRIOR_PATH = CONFIG_PATH.parent.parent / Path(CONFIG["PRIOR_PATH"])
NOISE_PATH = CONFIG_PATH.parent.parent / Path(CONFIG["NOISE_PATH"])
WINDOW_PATH = CONFIG_PATH.parent.parent / Path(CONFIG["WINDOW_PATH"])

OPT_CHAR_PATH = Path(__file__).parent.parent.parent / "config" / "optimization.yaml"

PRIOR_SEED = 7
MOD_ERROR_SEED = 11
REG_NOISE_SEED = 39  # will depreciate soon
REG1_NOISE_SEED = 1998
REG2_NOISE_SEED = 2
REG3_NOISE_SEED = 34
SC_MEAS_NOISE_SEED = 2718  # measurement-noise obs perturbations, pco2geosc_reg_noic --sc_obs_pert plus_meas

# ScenarioMIP7 CO2 concentration pathways (data/input/scenariomip7_conc_paths.csv),
# named by the file's scenario_ext column. CO2 only, so only these models can use them
SMIP7_SCENARIOS = ("H-ext", "H-ext-OS", "M-ext", "ML-ext", "L-ext", "VLLO-ext", "VLHO-ext")
SMIP7_CONC_STATS = ("mean", "median")
SMIP7_MODELS = ("pco2geowc_reg_noic", "pco2geowc3_reg_noic")

with open(OPT_CHAR_PATH, "r") as f:
    opt_config = yaml.safe_load(f)


def parse_args():
    """Parse command line arguments for each experiment."""

    parser = argparse.ArgumentParser(
        description="Estimating future climate uncertainty using pseudo-observations and ensemble variational data assimilation."
    )

    # model (must have this argument to avoid mix ups)
    parser.add_argument(
        "--model",
        type=str,
        default="pco2geowc",
        required=True,
        choices=[
            "pco2geowc",
            "pco2geowc3",
            "pco2geowc_nn",
            "pco2geowc_reg",
            "pco2geowc_reg_noic",
            "pco2geosc_reg_noic",
            "pco2geowc3_reg",
            "pco2geowc3_reg_noic",
            "pco2geowc3_nn",
        ],
        help="The model equations to use",
    )

    # emissions scenario
    parser.add_argument(
        "--scenario",
        type=str,
        default="ssp245",
        choices=["ssp245", "ssp585", *SMIP7_SCENARIOS],
        help="The CO2 concentrations scenario (RCMIP ssp* or ScenarioMIP7 *-ext)",
    )

    # ScenarioMIP7 scenarios only. default=SUPPRESS keeps it out of RCMIP runs'
    # namespace; check_config_compatability fills in the default for ScenarioMIP7
    parser.add_argument(
        "--conc_stat",
        type=str,
        default=argparse.SUPPRESS,
        choices=SMIP7_CONC_STATS,
        help=(
            "ScenarioMIP7 scenarios: which statistic of the CO2 concentration "
            "path to use (default: median)"
        ),
    )

    # start time of model
    parser.add_argument(
        "--tmin", type=int, default=2025, help="The start time of the experiment"
    )

    # noise model
    parser.add_argument(
        "--noise_model",
        type=str,
        default="AR1",
        choices=["AR1", "AR0", "nn"],
        help="The noise model to use",
    )

    # true value of angle parameter
    parser.add_argument(
        "--theta",
        type=int,
        default=15,
        help="The true SAI angle parameter to estimate",
    )

    # true value of ECS
    parser.add_argument(
        "--ecs", type=float, default=3.0, help="Equilibrium climate sensitivity"
    )

    # SAI cooling per decade
    parser.add_argument(
        "--deg_p_dec",
        type=float,
        default=0.1,
        help="Degrees per decade of cooling from SAI",
    )

    # number of years ramp up for SAI
    parser.add_argument(
        "--n_yrs_ramp",
        type=int,
        default=50,
        help="The number of years to linearly ramp up SAI",
    )

    parser.add_argument(
        "--windowing",
        type=str,
        default="original",
        choices=[
            "original",
            "debug",
            "fine_grad_coarse",
            "four",
            "ws_gradual",
            "gradual",
        ],
        help="Assimilation window name; config pulled from config/windowing.yaml.",
    )

    # number of ensemble member
    parser.add_argument(
        "--n_ens", type=int, default=500, help="Number of ensemble members"
    )

    parser.add_argument(
        "--save_output", action="store_true", default=False, help="Save output to disk?"
    )

    # regional noise in simluations?
    parser.add_argument(
        "--reg_noise",
        action="store_true",
        default=False,
        help="Is there nonzero noise in regional temperatures?",
    )

    parser.add_argument(
        "--sai_ramp",
        type=str,
        default="linear",
        choices=["linear", "fast", "slow"],
        help="Rate of SAI ramp up (linear = linear, slow = cubic, fast = t^1/3)",
    )

    # strong-constraint model only (pco2geosc_reg_noic). default=SUPPRESS keeps
    # these out of every other model's namespace, so their run metadata and
    # output are unchanged; the model fills in its own defaults
    parser.add_argument(
        "--sc_noise",
        type=str,
        default=argparse.SUPPRESS,
        choices=["flux", "temp"],
        help=(
            "pco2geosc_reg_noic: how internal variability enters T1 -- 'flux' "
            "(heat-flux kick, as in the weak model; default) or 'temp' (direct "
            "temperature perturbation, strong_constraint.T1_NOISE_STD in noise.yaml)"
        ),
    )

    parser.add_argument(
        "--sc_covar",
        type=str,
        default=argparse.SUPPRESS,
        choices=["marginal", "profile", "fixed"],
        help=(
            "pco2geosc_reg_noic: treatment of the parameter-dependent noise "
            "covariance S(theta) -- 'marginal' (with log det S; default), "
            "'profile' (reproduces the weak model exactly), or 'fixed' (S at "
            "the prior center)"
        ),
    )

    parser.add_argument(
        "--sc_obs_pert",
        type=str,
        default=argparse.SUPPRESS,
        choices=["match_weak", "plus_meas"],
        help=(
            "pco2geosc_reg_noic: per-member observation perturbation -- "
            "'match_weak' (from the weak model's noise background draws; "
            "default) or 'plus_meas' (also add measurement noise)"
        ),
    )

    parser.add_argument(
        "--check_components",
        action="store_true",
        default=False,
        help="Check components of variational assimilation model (TLM, ADJ, grad J)?",
    )

    parser.add_argument(
        "--no_opt",
        action="store_true",
        default=False,
        help=(
            "Skip the optimization (inner and outer loops) in every window. "
            "Use with --check_components to verify model components without "
            "paying for an assimilation; pair with --windowing debug so there "
            "is only one window."
        ),
    )

    # add debugging mode
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Toggle default mode (adds a lot of print statements)",
    )

    return parser.parse_args()


def check_config_compatability(args):
    """Check if dependent CLI inputs are compatable

    Parameters
    ----------
    args: argparse.Namespace
        CLI args
    """

    if args.model == "pco2geowc_nn" and args.reg_noise:
        raise ValueError(f"{args.model} is incompatable with regional noise turned on.")

    if args.model == "pco2geowc3_nn" and args.reg_noise:
        raise ValueError(f"{args.model} is incompatable with regional noise turned on.")

    if args.model == "pco2geowc_nn" and args.noise_model != "nn":
        raise ValueError(
            f"{args.model} is not compatabile with any noise model other than 'nn'."
        )

    if args.model == "pco2geowc3_nn" and args.noise_model != "nn":
        raise ValueError(
            f"{args.model} is not compatabile with any noise model other than 'nn'."
        )

    if (
        args.model != "pco2geowc_nn" and args.model != "pco2geowc3_nn"
    ) and args.noise_model == "nn":
        raise ValueError(
            f"{args.model} has internal variability, but no noise model ({args.noise_model}) was passed. Use 'AR1' or 'AR0'."
        )

    if args.windowing == "ws_gradual" and args.tmin != 2023:
        raise ValueError(
            f"Windowing scheme {args.windowing} is designed for a warm start beginning in 2023. Please set --tmin to 2023."
        )

    if args.windowing == "gradual":
        if args.tmin >= 2030:
            raise ValueError(
                f"Windowing scheme {args.windowing} has its first window ending in 2030, so --tmin must be before 2030 (got {args.tmin}). It is designed for --tmin 2025."
            )
        if args.tmin != 2025:
            warnings.warn(
                f"Windowing scheme {args.windowing} is designed for --tmin 2025 (got {args.tmin}); the first window length will differ from the intended 5 years."
            )

    if (
        args.model in ("pco2geowc_reg", "pco2geowc_reg_noic", "pco2geosc_reg_noic")
        and not args.reg_noise
    ):
        raise ValueError(f"{args.model} cannot be run without --reg_noise flag")

    if args.model in ("pco2geowc3_reg", "pco2geowc3_reg_noic") and not args.reg_noise:
        raise ValueError(f"{args.model} cannot be run without --reg_noise flag")

    if args.scenario in SMIP7_SCENARIOS:
        if args.model not in SMIP7_MODELS:
            raise ValueError(
                f"ScenarioMIP7 scenario {args.scenario} only provides CO2 "
                f"concentrations, so it is limited to {', '.join(SMIP7_MODELS)} "
                f"(got {args.model})."
            )
        # record the statistic actually used in the run metadata and filename
        if not hasattr(args, "conc_stat"):
            args.conc_stat = "median"

    elif hasattr(args, "conc_stat"):
        raise ValueError(
            f"--conc_stat only applies to ScenarioMIP7 scenarios, not {args.scenario}."
        )

    sc_flags = [f for f in ("sc_noise", "sc_covar", "sc_obs_pert") if hasattr(args, f)]
    if sc_flags and args.model != "pco2geosc_reg_noic":
        raise ValueError(
            f"{', '.join('--' + f for f in sc_flags)} only apply to "
            f"pco2geosc_reg_noic, not {args.model}."
        )

    if args.no_opt and not args.check_components:
        raise ValueError(
            "--no_opt skips the assimilation, and without --check_components "
            "nothing would be computed. Pass --check_components too."
        )


# quick test
if __name__ == "__main__":
    args = parse_args()
    print(args)

    print(opt_config)
