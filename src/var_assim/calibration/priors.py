"""Priors dataclass.

Adam Michael Bauer
UChicago
Feb 2026
"""

import yaml
import argparse

import numpy as np

from dataclasses import dataclass, field
from pathlib import Path
from var_assim.calibration.noise import ClimateModelNoise
from var_assim.models import MODEL_REGISTRY


@dataclass
class ClimateModelPriors:
    """Describes priors on all model parameters."""

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

    # parameters constrained to be nonnegative; stored as a list of *_CEN key
    # names (e.g. "L_CEN"). loaded from yaml, used to derive nonneg_inds.
    nonneg_vars: list[str] = field(default_factory=list)

    # indices into controls_cen / controls_std for nonneg parameters.
    # derived automatically from nonneg_vars — do not set manually.
    nonneg_inds: list[int] = field(default_factory=list)

    @classmethod
    def from_cli_and_yaml_and_noise(
        cls,
        cli_args: argparse.Namespace,
        prior_cal_path: Path,
        Noise: ClimateModelNoise,
    ) -> "ClimateModelPriors":
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
        with open(prior_cal_path, "r") as f:
            prior_data = yaml.safe_load(f)

        param_dict = (
            prior_data["global"]
            | prior_data[MODEL_REGISTRY[cli_args.model]["N_regions"]]
        )

        # set climate feedback based on ECS and forcing
        param_dict["L_CEN"] = (
            param_dict["F1_CO2_CEN"] * np.log(2) / param_dict["ECS_CEN"]
        )

        # loop through variables that use factor-based methods to compute prior variance
        for var in param_dict["factor_vars"]:
            cen, std = var  # extract central and standard deviation names

            # set standard deviation equal to central value * PRIOR_STD_FACTOR
            param_dict[std] = param_dict[cen] * param_dict["PRIOR_STD_FACTOR"]

        # set initial conditions stuff
        # for T1 and T2 initial conditions, set to internal variability size if the model has noise
        # else, just set to something small (it can't be zero or the covariance matrix will be singular)
        param_dict["T1_STD"] = (
            Noise.INT_VAR_STD if cli_args.model != "pco2geowc_nn" else 0.1
        )
        param_dict["T2_STD"] = (
            Noise.INT_VAR_STD if cli_args.model != "pco2geowc_nn" else 0.1
        )
        param_dict["Q_STD"] = (
            (param_dict["C1_CEN"] + param_dict["C2_CEN"]) * Noise.INT_VAR_STD
            if cli_args.model != "pco2geowc_nn"
            else 0.1 * (param_dict["C1_CEN"] + param_dict["C2_CEN"])
        )

        # regional temperature vector
        if cli_args.reg_noise:
            # initial condition std adjusted to internal variability + regional variability
            param_dict["T_REG_STD"] = (
                np.array(param_dict["ALPHA_CEN"]) * Noise.INT_VAR_STD
                + Noise.OBS_T_REG_STD
            )

        else:
            # if no regional noise, just internal variability adjusts regional ICs
            param_dict["T_REG_STD"] = (
                np.array(param_dict["ALPHA_CEN"]) * Noise.INT_VAR_STD
                if cli_args.model != "pco2geowc_nn"
                else np.array(param_dict["ALPHA_CEN"]) * 0.1
            )

        param_dict["controls_cen"] = np.array(
            [
                param_dict["T1_CEN"],
                param_dict["T2_CEN"],
                param_dict["Q_CEN"],
                *param_dict["T_REG_CEN"],
                param_dict["L_CEN"],
                param_dict["G_CEN"],
                param_dict["EPS_CEN"],
                param_dict["C1_CEN"],
                param_dict["C2_CEN"],
                param_dict["F1_CO2_CEN"],
                *param_dict["ALPHA_CEN"],
                *param_dict["BETA_CEN"],
            ]
        )

        param_dict["controls_std"] = np.array(
            [
                param_dict["T1_STD"],
                param_dict["T2_STD"],
                param_dict["Q_STD"],
                *param_dict["T_REG_STD"],
                param_dict["L_STD"],
                param_dict["G_STD"],
                param_dict["EPS_STD"],
                param_dict["C1_STD"],
                param_dict["C2_STD"],
                param_dict["F1_CO2_STD"],
                *param_dict["ALPHA_STD"],
                *param_dict["BETA_STD"],
            ]
        )

        # derive nonneg indices from nonneg_vars + regional block sizes
        param_dict["nonneg_inds"] = cls._get_nonneg_inds(
            param_dict.get("nonneg_vars", []),
            n_reg=len(param_dict["T_REG_CEN"]),
            n_alpha=len(param_dict["ALPHA_CEN"]),
            n_beta=len(param_dict["BETA_CEN"]),
        )

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
        ws_results: list
            list of t_final results from warm start
        """

        # set global climate central values
        self.T1_CEN = ws_results[0]
        self.T2_CEN = ws_results[1]
        self.Q_CEN = self.C1_CEN * self.T1_CEN + self.C2_CEN * self.T2_CEN

        # set retional climate central values
        self.T_REG_CEN = ws_results[3 : 3 + len(self.T_REG_STD)]

        # update master control vector
        self._update_controls_vectors()

    def _update_controls_vectors(self):
        self.controls_cen = np.array(
            [
                self.T1_CEN,
                self.T2_CEN,
                self.Q_CEN,
                *self.T_REG_CEN,
                self.L_CEN,
                self.G_CEN,
                self.EPS_CEN,
                self.C1_CEN,
                self.C2_CEN,
                self.F1_CO2_CEN,
                *self.ALPHA_CEN,
                *self.BETA_CEN,
            ]
        )

        self.controls_std = np.array(
            [
                self.T1_STD,
                self.T2_STD,
                self.Q_STD,
                *self.T_REG_STD,
                self.L_STD,
                self.G_STD,
                self.EPS_STD,
                self.C1_STD,
                self.C2_STD,
                self.F1_CO2_STD,
                *self.ALPHA_STD,
                *self.BETA_STD,
            ]
        )

    def gen_parameter_prior(self, N_ens: int) -> None:
        """Generate parameter prior ensemble, with nonnegativity enforced.

        Draws N_ens samples from a multivariate normal defined by controls_cen
        and controls_std. Any draw that goes negative for a physically
        constrained parameter (L, G, C1, C2, F1_CO2) is snapped back to the
        prior mean for that parameter, consistent with the original behavior.

        Parameters
        ----------
        N_ens: int
            number of ensemble members
        """
        prior_vec = np.random.multivariate_normal(
            self.controls_cen, np.diag(self.controls_std), size=N_ens
        )

        # enforce nonnegativity: snap unphysical draws back to the prior mean
        # for ind in self.nonneg_inds:
        #    prior_vec[:, ind] = np.where(
        #        prior_vec[:, ind] < 0, self.controls_cen[ind], prior_vec[:, ind]
        #    )

        self.param_prior = prior_vec

    def get_augmented_parameter_prior(self, aug_dist: np.ndarray) -> np.ndarray:
        """Augment the parameter prior ensemble with additional columns.

        This is usually used to make the prior for the simulation where aug_dist
        is the prior for model errors

        Parameters
        ----------
        aug_dist: np.ndarray, shape (N_ens, x)
            array to append to the parameter prior column-wise

        Returns
        -------
        np.ndarray, shape (N_ens, len(controls_cen) + X)
            horizontal concatenation of param_prior and x
        """
        return np.hstack([self.param_prior, aug_dist])

    @staticmethod
    def _get_nonneg_inds(
        nonneg_vars: list[str],
        n_reg: int,
        n_alpha: int,
        n_beta: int,
    ) -> list[int]:
        """Map nonneg_vars parameter names to their indices in controls_cen.

        The controls vector has this layout:
            [T1, T2, Q, *T_REG (n_reg), L, G, EPS, C1, C2, F1_CO2,
            *ALPHA (n_alpha), *BETA (n_beta)]

        Parameters
        ----------
        nonneg_vars: list[str]
            parameter names (e.g. ["L_CEN", "G_CEN"]) from the yaml config

        n_reg: int
            number of regional temperature entries (len of T_REG_CEN)

        n_alpha: int
            number of alpha entries

        n_beta: int
            number of beta entries

        Returns
        -------
        list[int]
            sorted indices into controls_cen for the nonneg parameters
        """
        # build the full ordered name list, mirroring controls_cen construction
        names = [
            "T1_CEN",
            "T2_CEN",
            "Q_CEN",
            *[f"T_REG_CEN_{i}" for i in range(n_reg)],
            "L_CEN",
            "G_CEN",
            "EPS_CEN",
            "C1_CEN",
            "C2_CEN",
            "F1_CO2_CEN",
            *[f"ALPHA_CEN_{i}" for i in range(n_alpha)],
            *[f"BETA_CEN_{i}" for i in range(n_beta)],
        ]
        name_to_ind = {name: i for i, name in enumerate(names)}

        return sorted(name_to_ind[var] for var in nonneg_vars if var in name_to_ind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--noise_model", default="AR1")
    parser.add_argument("--reg_noise", action="store_true", default=False)
    parser.add_argument("--n_ens", type=int, default=2)
    args = parser.parse_args()

    Noise = ClimateModelNoise.from_cli_and_yaml(args, Path("config/noise.yaml"))
    Priors = ClimateModelPriors.from_cli_and_yaml_and_noise(
        args, Path("config/priors.yaml"), Noise
    )

    print(Priors)

    np.random.seed(200)

    print(Priors.get_augmented_cen_vector(np.array([100, 200, 300])))
    print(Priors.get_augmented_std_vector(np.array([100, 200, 300])))

    dummy_ws_results = (
        [1.1, 0.6, 20, 2.2, 1.6]
        if args.model != "pco2geowc3"
        else [1.1, 0.6, 20, 2.2, 1.6, 1.4]
    )
    Priors.set_state_priors_from_warmstart(dummy_ws_results)

    print(Priors.controls_cen)
    print(Priors.controls_std)

    Priors.gen_parameter_prior(args.n_ens)

    print(Priors.nonneg_inds, Priors.nonneg_vars)
