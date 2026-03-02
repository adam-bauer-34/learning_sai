"""Priors dataclass.

Adam Michael Bauer
UChicago
Feb 2026
"""

import yaml
import argparse

import numpy as np

from dataclasses import dataclass
from pathlib import Path
from var_assim.calibration.noise import ClimateModelNoise
from var_assim.models import MODEL_REGISTRY


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
    ALPHA_CEN: list[float]  # pattern scaling parameters for global temp
    BETA_CEN: list[float]  # sai adjustment to pattern scaling

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
    ALPHA_STD: list[float]

    # sai adjustment to pattern scaling
    BETA_STD: list[float]

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
    T_REG_CEN: list[float]
    T_REG_STD: np.ndarray

    # vector of parameter prior central values
    controls_cen: np.ndarray

    # vector of parameter prior variance values
    controls_std: np.ndarray


    @classmethod
    def from_cli_and_yaml_and_noise(cls, cli_args: argparse.Namespace, prior_cal_path: Path, Noise: ClimateModelNoise) -> 'ClimateModelPriors':
        """Create a Priors object from a .yaml file.
        
        Parameters
        ----------
        cli_args: argparse.ArgumentParser
            parsing object for CLI arguments

        prior_cal_path: Path
            path to prior data

        Noise: ClimateModelNoise object
            noise attributes

        Returns
        -------
        cls: ClimateModelPriors
            Priors object
        """

        param_dict = {}  # final dictionary of parameters for initialization

        # load the yaml file
        with open(prior_cal_path, 'r') as f:
            prior_data = yaml.safe_load(f)

        param_dict = (
            prior_data['global']
            | prior_data[MODEL_REGISTRY[cli_args.model]['N_regions']]
        )

        # set climate feedback based on ECS and forcing
        param_dict['L_CEN'] = param_dict['F1_CO2_CEN'] * np.log(2) / param_dict['ECS_CEN']

        # loop through variables that use factor-based methods to compute prior variance
        for var in param_dict['factor_vars']:
            cen, std = var  # extract central and standard deviation names

            # set standard deviation equal to central value * PRIOR_STD_FACTOR
            param_dict[std] = param_dict[cen] * param_dict['PRIOR_STD_FACTOR']

        # set initial conditions stuff
        param_dict['T1_STD'] = Noise.INT_VAR_STD
        param_dict['T2_STD'] = Noise.INT_VAR_STD
        param_dict['Q_STD'] = (param_dict['C1_CEN'] + param_dict['C2_CEN']) * Noise.INT_VAR_STD

        # regional temperature vector
        if cli_args.reg_noise:
            # initial condition std adjusted to internal variability + regional variability
            param_dict['T_REG_STD'] = np.array(param_dict['ALPHA_CEN']) * Noise.INT_VAR_STD + Noise.OBS_T_REG_STD

        else:
            # if no regional noise, just internal variability adjusts regional ICs
            param_dict['T_REG_STD'] = np.array(param_dict['ALPHA_CEN']) * Noise.INT_VAR_STD

        param_dict['controls_cen'] = np.array([
            param_dict['T1_CEN'], param_dict['T2_CEN'], param_dict['Q_CEN'],
            *param_dict['T_REG_CEN'], param_dict['L_CEN'], param_dict['G_CEN'],
            param_dict['EPS_CEN'], param_dict['C1_CEN'], param_dict['C2_CEN'],
            param_dict['F1_CO2_CEN'], *param_dict['ALPHA_CEN'], *param_dict['BETA_CEN']
        ])

        param_dict['controls_std'] = np.array([
            param_dict['T1_STD'], param_dict['T2_STD'], param_dict['Q_STD'],
            *param_dict['T_REG_STD'], param_dict['L_STD'], param_dict['G_STD'],
            param_dict['EPS_STD'], param_dict['C1_STD'], param_dict['C2_STD'],
            param_dict['F1_CO2_STD'], *param_dict['ALPHA_STD'], *param_dict['BETA_STD']
        ])

        return cls(**param_dict)


    def get_augmented_cen_vector(self, aug_vector: list | np.ndarray) -> np.ndarray:
        """Augment central control vector, usually used in 
        conjunction with model errors.
        """
        return np.hstack([self.controls_cen, aug_vector])
    

    def get_augmented_std_vector(self, aug_vector: list | np.ndarray) -> np.ndarray:
        """Augment central standard deviation vector, usually used in 
        conjunction with model errors.
        """
        return np.hstack([self.controls_std, aug_vector])


    def set_state_priors_from_warmstart(self, ws_results):
        """Set dataclass attributes for state variables based on warm start results.

        Parameters
        ----------
        ws_results: ?
            maybe dict with relevant warm start characteristics?
        """

        # set global climate central values
        self.T1_CEN = ws_results[0]
        self.T2_CEN = ws_results[1]
        self.Q_CEN = self.C1_CEN * self.T1_CEN + self.C2_CEN * self.T2_CEN
        
        # set retional climate central values
        self.T_REG_CEN = ws_results[3:3+len(self.T_REG_STD)]

        # update master control vector
        self._update_controls_vectors()

    def _update_controls_vectors(self):
        self.controls_cen = np.array([
            self.T1_CEN, self.T2_CEN, self.Q_CEN, *self.T_REG_CEN,
            self.L_CEN, self.G_CEN, self.EPS_CEN, self.C1_CEN, self.C2_CEN,
            self.F1_CO2_CEN, *self.ALPHA_CEN, *self.BETA_CEN
        ])
        
        self.controls_std = np.array([
            self.T1_STD, self.T2_STD, self.Q_STD, *self.T_REG_STD,
            self.L_STD, self.G_STD, self.EPS_STD, self.C1_STD, self.C2_STD,
            self.F1_CO2_STD, *self.ALPHA_STD, *self.BETA_STD
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--noise_model", default='AR1')
    parser.add_argument('--reg_noise', action='store_true', default=False)
    args = parser.parse_args()

    Noise = ClimateModelNoise.from_cli_and_yaml(args, Path('config/noise.yaml'))
    Priors = ClimateModelPriors.from_cli_and_yaml_and_noise(args, Path('config/priors.yaml'), Noise)

    print(Priors)

    print(Priors.get_augmented_cen_vector(np.array([100, 200, 300])))
    print(Priors.get_augmented_std_vector(np.array([100, 200, 300])))

    Priors.set_state_priors_from_warmstart([1.1, 0.6, 20, 2.2, 1.6])

    print(Priors.controls_cen)
    print(Priors.controls_std)