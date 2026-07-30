"""Logging utilities for experiments.

Adam Bauer
UChicago
Feb 2026
"""

import logging
import sys
import subprocess


def get_git_hash(short=False):
    """Get the current git hash for reproducibility.

    Parameters
    ----------
    short: bool
        return the abbreviated (7 character) hash instead of the full one
    """
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        git_hash = "unknown"

    if short:
        return git_hash[:7]

    return git_hash


def is_git_dirty():
    """Are there uncommitted changes to tracked files?

    Used to stamp check output: a git hash alone doesn't identify the code that
    ran if the working tree has been edited on top of it. Untracked files are
    ignored, since they can't change model behaviour.
    """
    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"]
            )
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        return False

    return len(status) > 0


def setup_logger(debug=False):
    """Set up a logger for the experiment."""

    logger = logging.getLogger("var_assim")

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    return logger


if __name__ == "__main__":
    logger = setup_logger(debug=True)
    logger.info(f"Git: {get_git_hash()}")
    logger.warning("Yo!")
    logger.error("Oh no!")
