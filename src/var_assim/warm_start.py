"""Warm start module for variational data assimilation simulations.

Adam Michael Bauer
UChicago
Feb 2026
"""

import logging
import argparse

import numpy as np

from typing import Callable
from var_assim.emis import EmissionsBaseline


def warm_start_simulation(
    logger: logging.Logger,
    args: argparse.Namespace,
    Truth: object,
    Prior: object,
    nonlin_path: Callable,
):

    # make warm start emissions baseline
    e = EmissionsBaseline(
        logger,
        args,
        1850,
        args.tmin,
        geo=False,
        Prior=Prior,
        Truth=Truth,
        print_level=2,
    )

    if "_nn" in args.model:
        logger.debug("       >> Running no noise warm start...")
        controls_tr_aug = Truth.controls_tr.copy()

    elif "_reg" in args.model and "3_reg" not in args.model:
        # if it's a two region regional model, you need three strings of model errors to warm start the model
        logger.debug("       >> Running 2 regional noise warm start...")
        controls_tr_aug = Truth.get_augmented_truth_vector(
            np.hstack([np.zeros_like(e.conc["CO2"])] * 3)
        )

    elif "3_reg" in args.model:
        # if it's a three region regional model, you need 4
        logger.debug("       >> Running 3 regional noise warm start...")
        controls_tr_aug = Truth.get_augmented_truth_vector(
            np.hstack([np.zeros_like(e.conc["CO2"])] * 4)
        )

    else:
        # get true controls vector for model simulation
        logger.debug("       >> Running no regional model errors warm start...")
        controls_tr_aug = Truth.get_augmented_truth_vector(np.zeros_like(e.conc["CO2"]))

    # simulate model equations over warm start period
    paths_ws, _ = nonlin_path(e, controls_tr_aug, 1850, args.tmin, DT=1.0)

    # set true values and central values according to internal functions of
    # `Truth` and `Priors`
    Truth.set_state_truth_from_warmstart(paths_ws[:, -1])
    Prior.set_state_priors_from_warmstart(paths_ws[:, -1])
