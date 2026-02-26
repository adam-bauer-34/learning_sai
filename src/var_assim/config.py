"""Config file that loads paths from YAML.

Adam Bauer
UChicago
Jan 2026
"""

import yaml
import argparse 

from pathlib import Path

# setup directories for data, saving, and other configurations
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "dirs.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = Path(CONFIG["DATA_DIR"])
FIGS_DIR = Path(CONFIG["FIGS_DIR"])

TRUTH_PATH = Path(CONFIG["TRUTH_PATH"])
PRIOR_PATH = Path(CONFIG["PRIOR_PATH"])
NOISE_PATH = Path(CONFIG["NOISE_PATH"])
WINDOW_PATH = Path(CONFIG["WINDOW_PATH"])


def parse_args():
    """Parse command line arguments for each experiment.
    """

    parser = argparse.ArgumentParser(
        description="Estimating future climate uncertainty using pseudo-observations and ensemble variational data assimilation."
    )

    # model (must have this argument to avoid mix ups)
    parser.add_argument(
        "--model",
        type=str,
        default="pco2geowc",
        required=True,
        choices=['pco2geowc', 'pco2geowc3', 'pco2geowc_nn'],
        help="The model equations to use"
    )

    # emissions scenario
    parser.add_argument(
        "--scenario",
        type=str,
        default="ssp245",
        choices=['ssp245', 'ssp585'],
        help='The CO2 concentrations scenario'
    )

    # start time of model
    parser.add_argument(
        "--tmin",
        type=int,
        default=2025,
        help='The start time of the experiment'
    )

    # noise model
    parser.add_argument(
        "--noise_model",
        type=str,
        default="AR1",
        choices=["AR1", "AR0", "nn"],
        help='The noise model to use'
    )

    # true value of angle parameter
    parser.add_argument(
        "--theta",
        type=int,
        default=14,
        choices=[9, 12, 14, 17, 21, 33],
        help='The true SAI angle parameter to estimate'
    )

    # true value of ECS
    parser.add_argument(
        "--ecs",
        type=float,
        default=3.0,
        help='Equilibrium climate sensitivity'
    )

    # SAI cooling per decade
    parser.add_argument(
        "--deg_p_dec",
        type=float,
        default=0.1,
        help='Degrees per decade of cooling from SAI'
    )

    # number of years ramp up for SAI
    parser.add_argument(
        "--n_yrs_ramp",
        type=int,
        default=50,
        help='The number of years to linearly ramp up SAI'
    )

    parser.add_argument(
        '--windowing',
        type=str,
        default='original',
        choices=['original', 'debug', 'fine_grad_coarse', 'four'],
        help='Assimilation window name; config pulled from config/windowing.yaml.'
    )

    # number of ensemble member
    parser.add_argument(
        "--n_ens",
        type=int,
        default=500,
        help='Number of ensemble members'
    )

    parser.add_argument(
        "--save_output",
        action='store_true',
        default=False,
        help='Save output to disk?'
    )

    # regional noise in simluations?
    parser.add_argument(
        "--reg_noise",
        action='store_true',
        default=False,
        help='Is there nonzero noise in regional temperatures?'
    )

    # add debugging mode
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help='Toggle default mode (adds a lot of print statements)'
    )

    return parser.parse_args()


# quick test
if __name__ == "__main__":
    args = parse_args()
    print(args)