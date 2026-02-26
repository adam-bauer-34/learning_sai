"""Truth dataclass.

Adam Michael Bauer
UChicago
Feb 2026
"""

import yaml

import numpy as np

from dataclasses import dataclass, fields


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
    ALPHA_R1_TR: float
    ALPHA_R2_TR: float
    ALPHA_R3_TR: float
    BETA_R1_TR: float
    BETA_R2_TR: float
    BETA_R3_TR: float
    THETA_TR: int

    # state initial conditions (filled in from warm start module)
    T1_TR: float
    T2_TR: float
    Q_TR: float
    T_R1_TR: float
    T_R2_TR: float
    T_R3_TR: float

    @classmethod
    def from_cli_and_yaml(cls, cli_args: str, truth_path: str) -> 'ClimateModelTruth':
        """Make dataclass from CLI and yaml.

        Parameters
        ----------
        cli_args: `argparse.Namespace`
            command line arguements for main file
        
        truth_path: str
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
        param_dict = param_dict | truth_data['global'] | truth_data['theta'][cli_args.theta]

        # set climate feedback truth based on central value of ECS and F1_CO2
        param_dict['L_TR'] =  param_dict['F1_CO2_TR'] * np.log(2) / param_dict['ECS_TR']

        # overwrite ECS and then F1 based on CLI args
        param_dict['ECS_TR'] = cli_args.ecs
        param_dict['F1_CO2_TR'] = param_dict['L_TR'] * param_dict['ECS_TR'] / np.log(2)

        return cls(**param_dict)

    def set_state_truth_from_warmstart(self, ws_results):
        """Set state initial condition truth values from warm start results.
        """

        self.T1_TR = 0.0
        self.T2_TR = 0.0
        self.Q_TR = 0.0

        self.T_R1_TR = 0.0
        self.T_R2_TR = 0.0
        self.T_R3_TR = 0.0


# quick test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ecs', type=float, default=3.0
    )
    parser.add_argument(
        '--theta', type=int, default=14
    )
    args = parser.parse_args()

    print(type(args))
    
    Truth = ClimateModelTruth.from_cli_and_yaml(args, 'config/truth.yaml')

    print(Truth)