"""Priors dataclass.

Adam Michael Bauer
UChicago
Feb 2026
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from var_assim.config import DATA_DIR

@dataclass
class Priors:
    """Describes priors on all model parameters.
    """

    # Set central values
    # global climate values
    ECS_CEN: float  # equilibrium climate sensitivity
    L_CEN: float  # climate feedback parameter
    G_GEN: float  # layer transfer coefficient
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
    def from_yaml(cls, yaml_path):
        pass

    def set_state_priors_from_warmstart(self, ws_results):
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
    class tester():
        def __init__(self):
            self.theta = 14
            self.model = 'pco2geowc'
    
    t = tester()

    dc = Priors()
    dc.set_regional_parameters(args=t)

    print(dc.BETA_R1_CEN)