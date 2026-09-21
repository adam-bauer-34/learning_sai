"""Noise dataclass.

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


def derive_obs_stds(spec: dict, model: str, prior_path: Path = None) -> dict:
    """Derive observation-error stds from assumptions about T1 and T2.

    See the `obs_weighting` block in noise.yaml for the motivation and the full
    statement of the assumptions. In brief:

        sigma(T1)   = spec["OBS_T1_STD"]
        sigma(T_Rj) = ALPHA_CEN[j] * sigma(T1)
        sigma(Q)    = sqrt((C1_CEN * sigma(T1))^2 + (C2_CEN * sigma(T2))^2)

    T2 is not observed; its assumed error enters only by propagating into Q via
    Q = C1 * T1 + C2 * T2, with the two errors treated as independent.

    ALPHA_CEN, C1_CEN and C2_CEN are read straight from priors.yaml rather than
    through ClimateModelPriors, because Priors is constructed *from* Noise and
    the reverse dependency would be circular. Reading the yaml keeps the derived
    values from going stale if the calibration is ever changed.

    Parameters
    ----------
    spec: dict
        the model's entry in noise.yaml's `obs_weighting` block; must supply
        OBS_T1_STD and OBS_T2_STD

    model: str
        model name, used to pick two_region vs three_region ALPHA_CEN

    prior_path: Path, optional
        path to priors.yaml; defaults to var_assim.config.PRIOR_PATH

    Returns
    -------
    dict
        OBS_T1_STD, OBS_Q_STD and OBS_T_REG_STD, ready to merge into the
        dataclass' parameter dict
    """

    # function-level import: var_assim.config pulls in nothing from
    # var_assim.calibration, but keeping it local avoids adding another
    # module-level edge to an import graph that already has a cycle through
    # var_assim.models
    if prior_path is None:
        from var_assim.config import PRIOR_PATH as prior_path

    missing = {"OBS_T1_STD", "OBS_T2_STD"} - set(spec)
    if missing:
        raise KeyError(
            f"obs_weighting entry for '{model}' is missing {sorted(missing)}. "
            "Both are required to derive the observation error covariances."
        )

    with open(prior_path, "r") as f:
        prior_data = yaml.safe_load(f)

    N_regions = MODEL_REGISTRY[model]["N_regions"]
    ALPHA_CEN = np.asarray(prior_data[N_regions]["ALPHA_CEN"], dtype=float)
    C1_CEN = float(prior_data["global"]["C1_CEN"])
    C2_CEN = float(prior_data["global"]["C2_CEN"])

    sigma_T1 = float(spec["OBS_T1_STD"])
    sigma_T2 = float(spec["OBS_T2_STD"])

    if sigma_T1 <= 0.0 or sigma_T2 <= 0.0:
        raise ValueError(
            f"obs_weighting for '{model}': OBS_T1_STD and OBS_T2_STD must be "
            f"positive, got {sigma_T1} and {sigma_T2}. A zero would make the "
            "observation term singular."
        )

    return {
        "OBS_T1_STD": sigma_T1,
        "OBS_Q_STD": float(
            np.sqrt((C1_CEN * sigma_T1) ** 2 + (C2_CEN * sigma_T2) ** 2)
        ),
        "OBS_T_REG_STD": (ALPHA_CEN * sigma_T1).tolist(),
    }


@dataclass
class ClimateModelNoise:
    """Parameters for our noise model."""

    # noise model and boolean regional on/off
    NOISE_MODEL: str
    REGIONAL: bool

    # global climate
    INT_VAR_STD: float
    AUTO_CORR: float

    # regional noise
    OBS_T_REG_STD: list[float]
    INT_T_REG_STD: list[float]

    # observation noise
    OBS_T1_STD: float = 1.0
    OBS_Q_STD: float = 1.0

    # provenance: "legacy" if the flat 1.0 weighting from noise.yaml was used,
    # "derived" if this model has an obs_weighting entry. Recorded so a run's
    # log states which weighting produced its results.
    OBS_WEIGHTING: str = "legacy"

    @classmethod
    def from_cli_and_yaml(
        cls, cli_args: argparse.Namespace, noise_path: Path
    ) -> "ClimateModelNoise":
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
        param_dict["NOISE_MODEL"] = cli_args.noise_model
        param_dict["REGIONAL"] = cli_args.reg_noise

        with open(noise_path, "r") as f:
            noise_data = yaml.safe_load(f)

        # merge noise model based parameters
        param_dict = param_dict | noise_data["noise_model"][cli_args.noise_model]

        # if regional noise, add that
        if cli_args.reg_noise:
            N_regions = MODEL_REGISTRY[cli_args.model]["N_regions"]
            param_dict["OBS_T_REG_STD"] = noise_data["noise_model"]["Regional"][
                N_regions
            ]["OBS_T_REG_STD"]
            param_dict["INT_T_REG_STD"] = noise_data["noise_model"]["Regional"][
                N_regions
            ]["INT_T_REG_STD"]

        # if no noise, just add ones
        else:
            if MODEL_REGISTRY[cli_args.model]["N_regions"] == "two_region":
                param_dict["OBS_T_REG_STD"] = [1.0, 1.0]
                param_dict["INT_T_REG_STD"] = [1.0, 1.0]
            elif MODEL_REGISTRY[cli_args.model]["N_regions"] == "three_region":
                param_dict["OBS_T_REG_STD"] = [1.0, 1.0, 1.0]
                param_dict["INT_T_REG_STD"] = [1.0, 1.0, 1.0]
            else:
                raise ValueError("Number of regions not supported.")

        # per-model observation-error weighting. applied last so it overrides
        # both the flat OBS_T1_STD/OBS_Q_STD merged from the noise_model block
        # and the OBS_T_REG_STD taken from the Regional block above.
        obs_weighting = noise_data.get("obs_weighting") or {}
        if cli_args.model in obs_weighting:
            param_dict.update(
                derive_obs_stds(obs_weighting[cli_args.model], cli_args.model)
            )
            param_dict["OBS_WEIGHTING"] = "derived"
        else:
            param_dict["OBS_WEIGHTING"] = "legacy"

        return cls(**param_dict)


# quick test
if __name__ == "__main__":
    import argparse
    from pprint import pprint
    from dataclasses import asdict

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pco2geowc")
    parser.add_argument(
        "--noise_model", type=str, default="AR1", choices=["AR1", "AR0"]
    )
    parser.add_argument("--reg_noise", action="store_true", default=False)
    args = parser.parse_args()

    Noise = ClimateModelNoise.from_cli_and_yaml(args, Path("config/noise.yaml"))

    pprint(asdict(Noise), sort_dicts=False)
