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
from var_assim.calibration.truth import ClimateModelTruth
from var_assim.calibration.priors import ClimateModelPriors


def warm_start_simulation(logger: logging.Logger,
                          args: argparse.ArgumentParser,
                          Truth: ClimateModelTruth,
                          Prior: ClimateModelPriors,
                          nonlin_path: Callable) -> np.ndarray:
    

    # make warm start emissions baseline
    e = EmissionsBaseline(logger, args.scenario, 1850, args.tmin)

    # get true controls vector for model simulation
    controls_tr_aug = Truth.get_augmented_truth_vector(np.zeros_like(e.conc['CO2']))

    # simulate model equations over warm start period
    paths_ws, _ = nonlin_path(e, controls_tr_aug, 1850,
                              args.tmin, DT=1.0)

    # set true values and central values according to internal functions of
    # `Truth` and `Priors`
    Truth.set_state_truth_from_warmstart(paths_ws[:, -1])
    Prior.set_state_priors_from_warmstart(paths_ws[:, -1])

    return paths_ws