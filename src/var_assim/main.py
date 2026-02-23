"""Main file for variational data assimilation experiments.

Adam Bauer
UChicago
Feb 2026
"""

import time

from config import parse_args
from logging_utils import setup_logger, get_git_hash
from runner import run_experiment
from model_reg import model_reg


def main():
    args = parse_args()
    logger = setup_logger(args.debug)

    t0 = time.time()

    logger.info(f"Git hash: {get_git_hash()}")
    logger.info(f"Running experiment with config: {args}")

    t1 = time.time()

    logger.info("Experiment complete.")
    logger.info(f"Total runtime: {t1 - t0:.2f}s.")


if __name__ == "__main__":
    main()