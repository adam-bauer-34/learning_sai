"""Priors dataclass.

Adam Michael Bauer
UChicago
Feb 2026
"""

import yaml

import numpy as np

from dataclasses import dataclass, fields


@dataclass
class ClimateModelPriors:
    """Describes priors on all model parameters.
    """

    # Set central values
    # global climate values
    ECS_CEN: float  # equilibrium climate sensitivity
    L_CEN: float  # climate feedback parameter
    G_CEN: float  # layer transfer coefficient
    C1_CEN: float  # heat capacity of surface layer
    C2_CEN: float  # heat capacity of ocean layer
    F1_CO2_CEN: float  # forcing from log term in CO2
    EPS_CEN: float  # pattern effect

    # regional values
    # pattern scaling parameters for global temp
    ALPHA_R1_CEN: float
    ALPHA_R2_CEN: float
    ALPHA_R3_CEN: float

    # sai adjustment to pattern scaling
    BETA_R1_CEN: float
    BETA_R2_CEN: float
    BETA_R3_CEN: float

    # Set standard deviations
    # first set standard deviation factor for setting prior variances
    PRIOR_STD_FACTOR: float 
    factor_vars: list[tuple]

    # global parameters (no ECS bc distribution is skewed)
    G_STD: float
    L_STD: float
    C1_STD: float
    C2_STD: float
    F1_CO2_STD: float
    EPS_STD: float

    # regional parameters
    # pattern scaling parameters for global temp
    ALPHA_R1_STD: float
    ALPHA_R2_STD: float
    ALPHA_R3_STD: float

    # sai adjustment to pattern scaling
    BETA_R1_STD: float
    BETA_R2_STD: float
    BETA_R3_STD: float

    # attributes from simulation warm start
    # surface temperature initial condition
    T1_CEN: float
    T1_STD: float

    # deep ocean temperature initial conditin
    T2_CEN: float
    T2_STD: float

    # ocean heat content
    Q_CEN: float
    Q_STD: float

    # regional temperatures
    T_R1_CEN: float 
    T_R1_STD: float

    T_R2_CEN: float 
    T_R2_STD: float

    T_R3_CEN: float 
    T_R3_STD: float

    @classmethod
    def from_yaml(cls, prior_cal_path: str) -> 'ClimateModelPriors':
        """Create a Priors object from a .yaml file.
        
        Parameters
        ----------
        prior_cal_path: str
            path to prior data

        Returns
        -------
        cls: ClimateModelPriors
            Priors object
        """

        # load the yaml file
        with open(prior_cal_path, 'r') as f:
            prior_data = yaml.safe_load(f)

        all_dc_fields = {f.name for f in fields(cls)}
        yaml_matched_fields = {k: v for k, v in prior_data['priors'].items() if k in all_dc_fields}

        # set climate feedback based on ECS and forcing
        yaml_matched_fields['L_CEN'] = yaml_matched_fields['F1_CO2_CEN'] * np.log(2) / yaml_matched_fields['ECS_CEN']

        # loop through variables that use factor-based methods to compute prior variance
        for var in yaml_matched_fields['factor_vars']:
            cen, std = var  # extract central and standard deviation names

            # set standard deviation equal to central value * PRIOR_STD_FACTOR
            yaml_matched_fields[std] = yaml_matched_fields[cen] * yaml_matched_fields['PRIOR_STD_FACTOR']

        return cls(**yaml_matched_fields)

    def set_state_priors_from_warmstart(self, ws_results):
        """Set dataclass attributes for state variables based on warm start results.

        Parameters
        ----------
        ws_results: ?
            maybe dict with relevant warm start characteristics?
        """
        
        self.T1_CEN = 0.0
        self.T1_STD = 0.0
        
        self.T2_CEN = 0.0
        self.T2_STD = 0.0

        self.Q_CEN = 0.0
        self.Q_STD = 0.0

        self.T_R1_CEN = 0.0
        self.T_R1_STD = 0.0

        self.T_R2_CEN = 0.0
        self.T_R2_STD = 0.0

        self.T_R3_CEN = 0.0
        self.T_R3_STD = 0.0


if __name__ == '__main__':
    Priors = ClimateModelPriors.from_yaml('config/priors.yaml')
    Priors.set_state_priors_from_warmstart(0)

    print(Priors)