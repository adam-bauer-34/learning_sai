"""Noise dataclass.

Adam Michael Bauer
UChicago
Feb 2026
"""

import yaml
import argparse

import numpy as np

from pathlib import Path
from dataclasses import dataclass, field
from var_assim.models import MODEL_REGISTRY


@dataclass
class ClimateModelNoise:
    """Parameters for our noise model.
    """

    # noise model and boolean regional on/off
    NOISE_MODEL: str
    REGIONAL: bool

    # global climate
    INT_VAR_STD: float
    AUTO_CORR: float

    # regional noise
    OBS_T_REG_STD: list[float]

    # observation noise
    OBS_T1_STD: float = 1.0
    OBS_Q_STD: float = 1.0


    @classmethod
    def from_cli_and_yaml(cls, cli_args: argparse.Namespace, noise_path: Path) -> 'ClimateModelNoise':
        """Make dataclass from CLI and yaml.

        Parameters
        ----------
        cli_args: `argparse.Namespace`
            command line arguements for main file
        
        truth_path: Path
            path to noise.yaml

        Returns
        -------
        cls: ClimateModelNoise
            dataclass for true values based on given arguments
        """

        param_dict = {}
        
        # note config
        param_dict['NOISE_MODEL'] = cli_args.noise_model
        param_dict['REGIONAL'] = cli_args.reg_noise

        with open(noise_path, 'r') as f:
            noise_data = yaml.safe_load(f)

        # merge noise model based parameters
        param_dict = param_dict | noise_data['noise_model'][cli_args.noise_model]

        # if regional noise, add that
        if cli_args.reg_noise:
            N_regions = MODEL_REGISTRY[cli_args.model]['N_regions']
            param_dict['OBS_T_REG_STD'] = noise_data['noise_model']['Regional'][N_regions]['OBS_T_REG_STD']

        # if no noise, just add ones
        else:
            if MODEL_REGISTRY[cli_args.model]['N_regions'] == 'two_region':
                param_dict['OBS_T_REG_STD'] = [1.0, 1.0]
            elif MODEL_REGISTRY[cli_args.model]['N_regions'] == 'three_region':
                param_dict['OBS_T_REG_STD'] = [1.0, 1.0, 1.0]
            else:
                raise ValueError("Number of regions not supported.")

        return cls(**param_dict)


# quick test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='pco2geowc'
    )
    parser.add_argument(
        '--noise_model', type=str, default='AR1', choices=['AR1', 'AR0']
    )
    parser.add_argument(
        '--reg_noise', action='store_true', default=False
    )
    args = parser.parse_args()

    Noise = ClimateModelNoise.from_cli_and_yaml(args, Path('config/noise.yaml'))

    print(Noise)