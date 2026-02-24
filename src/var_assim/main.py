"""Main file for variational data assimilation experiments.

Adam Bauer
UChicago
Feb 2026
"""

import time

from config import parse_args, get_model
from logging_utils import setup_logger, get_git_hash
from runner import run_experiment


def main():
    args = parse_args()
    model = get_model(args.model)
    logger = setup_logger(args.debug)

    t0 = time.time()

    logger.info(f"Git hash: {get_git_hash()}")
    logger.info(f"Running experiment with config: {args}")

    from [import model framework] import run_var_assim_experiment
    run_var_assim_experiment(args, logger))

    t1 = time.time()

    logger.info("Experiment complete.")
    logger.info(f"Total runtime: {t1 - t0:.2f}s.")


if __name__ == "__main__":
    main()