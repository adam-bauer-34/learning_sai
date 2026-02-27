"""Truth dataclass.

Adam Michael Bauer
UChicago
Feb 2026
"""

import yaml
import argparse

import numpy as np

from pathlib import Path
from dataclasses import dataclass
from var_assim.models import MODEL_REGISTRY


@dataclass
class ClimateModelTruth:
    """True values for our climate model parameters that data assimilation will try to learn.
    """

    # global climate parameters
    ECS_TR: float
    L_TR: float  # set by ECS_TR and F1_CO2_TR on initialization
    G_TR: float
    C1_TR: float
    C2_TR: float
    F1_CO2_TR: float
    EPS_TR: float
    F_EFF_GEO_TR: float
        
    # regional climate parameters
    ALPHA_TR: list[float]
    BETA_TR: list[float]
    THETA_TR: int

    # state initial conditions (filled in from warm start module)
    T1_TR: float
    T2_TR: float
    Q_TR: float
    T_REG_TR: list[float]

    # vector of true values
    # must be in this order:
    # T1, T2, Q, T_REG, L, G, EPS, C1, C2, F1_CO2, ALPHA, BETA
    controls_tr: np.ndarray
    

    @classmethod
    def from_cli_and_yaml(cls, cli_args: argparse.Namespace, truth_path: Path) -> 'ClimateModelTruth':
        """Make dataclass from CLI and yaml.

        Parameters
        ----------
        cli_args: `argparse.Namespace`
            command line arguements for main file
        
        truth_path: Path
            path to truth.yaml

        Returns
        -------
        cls: ClimateModelTruth
            dataclass for true values based on given arguments
        """

        param_dict = {}
        param_dict['THETA_TR'] = cli_args.theta

        with open(truth_path, 'r') as f:
            truth_data = yaml.safe_load(f)

        # merge true value dictionaries
        param_dict = (
            param_dict
            | truth_data['global']
            | {'T_REG_TR': truth_data[MODEL_REGISTRY[args.model]['N_regions']]['T_REG_TR']}
            | truth_data[MODEL_REGISTRY[args.model]['N_regions']]['theta'][cli_args.theta]
        )

        # set climate feedback truth based on central value of ECS and F1_CO2
        param_dict['L_TR'] =  param_dict['F1_CO2_TR'] * np.log(2) / param_dict['ECS_TR']

        # overwrite ECS and then F1 based on CLI args
        param_dict['ECS_TR'] = cli_args.ecs
        param_dict['F1_CO2_TR'] = param_dict['L_TR'] * param_dict['ECS_TR'] / np.log(2)

        # make vector of true controls
        param_dict['controls_tr'] = np.array([
            param_dict['T1_TR'], param_dict['T2_TR'], param_dict['Q_TR'],
            *param_dict['T_REG_TR'], param_dict['L_TR'], param_dict['G_TR'],
            param_dict['EPS_TR'], param_dict['C1_TR'], param_dict['C2_TR'],
            param_dict['F1_CO2_TR'], *param_dict['ALPHA_TR'], *param_dict['BETA_TR']
        ])

        return cls(**param_dict)
    
    def get_augmented_truth_vector(self, aug_vector: np.ndarray) -> np.ndarray:
        """Returns an augmeneted truth vector. Useful for appending
        a list of arbitrary size, usually model errors, to the list of true parameter
        values.

        Parameters
        ----------
        aug_vector: np.ndarray
            the vector to augment to controls_tr

        Returns
        -------
        np.hstack(controls_tr, aug_vector)
            stacked vectors
        """

        return np.hstack([self.controls_tr, aug_vector])

    def set_state_truth_from_warmstart(self, ws_results):
        """Set state initial condition truth values from warm start results.
        """

        # pull results from warm start to update true initial conditions
        self.T1_TR = ws_results[0]
        self.T2_TR = ws_results[1]
        self.Q_TR = self.C1_TR * self.T1_TR + self.C2_TR * self.T2_TR

        # last entries are regional initial conditions
        self.T_REG_TR = ws_results[3:]

        # update true control vector with new ICs
        self._update_controls_tr()
    
    def _update_controls_tr(self):
        """Update controls vector based on new parameter values set elsewhere.
        """

        self.controls_tr = np.array([
            self.T1_TR, self.T2_TR, self.Q_TR, *self.T_REG_TR,
            self.L_TR, self.G_TR, self.EPS_TR, self.C1_TR, self.C2_TR,
            self.F1_CO2_TR, *self.ALPHA_TR, *self.BETA_TR
        ])


# quick test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)

    parser.add_argument(
        '--ecs', type=float, default=3.0
    )
    parser.add_argument(
        '--theta', type=int, default=14
    )
    args = parser.parse_args()

    Truth = ClimateModelTruth.from_cli_and_yaml(args, Path('config/truth.yaml'))

    print(Truth)

    print(Truth.get_augmented_truth_vector(np.array([100, 200, 300])))

    Truth.set_state_truth_from_warmstart([1.1, 0.6, 20, 2.2, 1.6])

    print(Truth.controls_tr)

    print(Truth.get_augmented_truth_vector(np.array([100, 200, 300])))