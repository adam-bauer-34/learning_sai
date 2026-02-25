"""Main file for variational data assimilation experiments.

Adam Bauer
UChicago
Feb 2026
"""

import time

from var_assim.config import parse_args
from var_assim.logging_utils import setup_logger, get_git_hash
from var_assim.models import MODEL_REGISTRY


def main():
    # parse arguments and setup logging
    args = parse_args()
    logger = setup_logger(args.debug)

    t0 = time.time()

    logger.info(f"Git hash: {get_git_hash()}")
    logger.info(f"Running experiment with config: {args}")

    # run variational data assimilation experiment with passed model
    try:
        run_var_assim_experiment = MODEL_REGISTRY[args.model]
    except KeyError:
        raise ValueError(f"Model {args.model} doesn't exist in model registry:\n{MODEL_REGISTRY}")

    run_var_assim_experiment(args, logger)

    t1 = time.time()

    # log finish
    logger.info("Experiment complete.")
    logger.info(f"Total runtime: {t1 - t0:.2f}s.")

if __name__ == "__main__":
    main()