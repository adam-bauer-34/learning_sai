"""Noise dataclass.

Adam Michael Bauer
UChicago
Feb 2026
"""

import yaml
import argparse

from pathlib import Path
from dataclasses import dataclass


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
    OBS_T1_STD: float = 1.0
    OBS_Q_STD: float = 1.0
    OBS_T_R1_STD: float = 1.0
    OBS_T_R2_STD: float = 1.0
    OBS_T_R3_STD: float = 1.0

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
        param_dict['NOISE_MODEL'] = cli_args.noise_model
        param_dict['REGIONAL'] = cli_args.reg_noise

        with open(noise_path, 'r') as f:
            noise_data = yaml.safe_load(f)

        # merge dictionaries
        # if we have regional noise, include that part of the yaml, if not, exclude and allow defaults
        if cli_args.reg_noise:
            param_dict = param_dict | noise_data['noise_model'][param_dict['NOISE_MODEL']] | noise_data['noise_model']['Regional']

        else:
            param_dict = param_dict | noise_data['noise_model'][param_dict['NOISE_MODEL']]

        return cls(**param_dict)


# quick test
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--noise_model', type=str, default='AR1', choices=['AR1', 'AR0']
    )
    parser.add_argument(
        '--reg_noise', action='store_true', default=False
    )
    args = parser.parse_args()

    Noise = ClimateModelNoise.from_cli_and_yaml(args, Path('config/noise.yaml'))

    print(Noise)