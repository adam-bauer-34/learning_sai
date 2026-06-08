"""Main file for variational data assimilation experiments.

Adam Bauer
UChicago
Feb 2026
"""

import time

from var_assim.config import (
    parse_args,
    check_config_compatability,
    TRUTH_PATH,
    PRIOR_PATH,
    NOISE_PATH,
    WINDOW_PATH,
)
from var_assim.logging_utils import setup_logger, get_git_hash
from var_assim.models import MODEL_REGISTRY
from var_assim.calibration.truth import ClimateModelTruth
from var_assim.calibration.priors import ClimateModelPriors
from var_assim.calibration.noise import ClimateModelNoise
from var_assim.calibration.windowing import AssimilationWindowing


def main():
    """Main function for simulation setup and running."""

    # parse arguments and check compatability
    args = parse_args()
    check_config_compatability(args)

    # setup logging
    logger = setup_logger(args.debug)

    t0 = time.time()

    # print statement for logging and reproducibility
    logger.info(f"Git hash: {get_git_hash()}")
    logger.info(f"Running experiment with config:")
    logger.info(f"    > Model equations: {args.model}")
    logger.info(f"    > Windowing config: {args.windowing}")
    logger.info(f"    > Socio-economic pathway: {args.scenario}")
    logger.info(f"    > Initial assimilation year: {args.tmin}")
    logger.info(f"    > Noise model for internal variability: {args.noise_model}")
    logger.info(f"    > True value of SAI angle parameter: {args.theta}")
    logger.info(f"    > True value of ECS: {args.ecs}")
    if args.ecs != 3.0:
        logger.warning(
            "        > For ECS != 3.0, the F2x parameter, NOT λ, is altered."
        )
    logger.info(
        f"    > SAI offsets {args.deg_p_dec} deg C / decade and is ramped up over {args.n_yrs_ramp} years."
    )
    logger.info(f"    > Number of ensemble members: {args.n_ens}")
    if args.reg_noise:
        logger.info(f"    > There is regional noise in this model run.")
    else:
        logger.info(f"    > There is no regional noise in this model run.")

    # setup prior, truth, noise, and windowing dataclasses from CLI and config/
    Noise = ClimateModelNoise.from_cli_and_yaml(args, NOISE_PATH)
    Truth = ClimateModelTruth.from_cli_and_yaml(args, TRUTH_PATH)
    Priors = ClimateModelPriors.from_cli_and_yaml_and_noise(args, PRIOR_PATH, Noise)
    Windowing = AssimilationWindowing.from_cli_and_yaml(args, WINDOW_PATH)

    # run variational data assimilation experiment with passed model
    try:
        logger.info("Running main assimilation experiment")
        run_var_assim_experiment = MODEL_REGISTRY[args.model]["runner"]
    except KeyError:
        raise ValueError(
            f"Model {args.model} doesn't exist in model registry:\n{MODEL_REGISTRY}"
        )

    run_var_assim_experiment(logger, args, Priors, Truth, Noise, Windowing)

    t1 = time.time()

    # log finish
    logger.info("Experiment complete.")
    logger.info(f"Total runtime: {t1 - t0:.2f}s.")


if __name__ == "__main__":
    main()
