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
class AssimilationWindowing:
    """Defines assimilation windows."""

    NAME: str  # name of windowing scheme
    windows: list[int]  # list of upper bounds for windows
    N_WINDS: int  # number of windows

    @classmethod
    def from_cli_and_yaml(
        cls, cli_args: argparse.Namespace, windowing_path: Path
    ) -> "AssimilationWindowing":
        """Make dataclass from yaml.

         Parameters
         ----------
        cli_args: `argparse.Namespace`
             command line arguements for main file

         windowing_path: Path
             path to windowing.yaml

         Returns
         -------
         cls: AssimilationWindowing
             dataclass for assimilation windows
        """

        param_dict = {}

        with open(windowing_path, "r") as f:
            noise_data = yaml.safe_load(f)

        # check if valid windows and tmin combination
        if noise_data[cli_args.windowing]["windows"][0] < cli_args.tmin:
            raise ValueError(
                (
                    f"Invalid windowing setup given tmin:\n"
                    f"    Windows specified: {param_dict['windows']}\n"
                    f"    TMIN: {cli_args.tmin}"
                )
            )

        windows = []
        for w in noise_data[cli_args.windowing]["windows"]:
            windows.append((cli_args.tmin, w))

        # set parameter dictionary and make into class
        param_dict = {
            "NAME": cli_args.windowing,
            "windows": windows,
            "N_WINDS": len(noise_data[cli_args.windowing]["windows"]),
        }

        return cls(**param_dict)


# quick test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--windowing", type=str, default="original")
    parser.add_argument("--tmin", type=int, default=2025)

    args = parser.parse_args()

    Windows = AssimilationWindowing.from_cli_and_yaml(
        args, Path("config/windowing.yaml")
    )

    print(Windows)
