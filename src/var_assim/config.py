"""Config file that loads paths from YAML.

Adam Bauer
UChicago
Jan 2026
"""

import yaml
import argparse 

from pathlib import Path

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "dirs.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = Path(CONFIG["DATA_DIR"])
FIGS_DIR = Path(CONFIG["FIGS_DIR"])

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
        help='The noise model to use'
    )

    # true value of angle parameter
    parser.add_argument(
        "--theta",
        type=int,
        default=14,
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
        "--dep_p_dec",
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

    # number of ensemble members
    parser.add_argument(
        "--n_ens",
        type=int,
        default=500,
        help='Number of ensemble members'
    )

    # manual override of assimilation windows?
    parser.add_argument(
        "--override_windows",
        action='store_true',
        help='Override automatically generated assimilation windows'
    )

    # add debugging mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help='Debug mode'
    )

    return parser.parse_args()


# quick test
if __name__ == "__main__":
    args = parse_args()
    print(args)