"""Emissions class for FaIR model.

Adam Michael Bauer
University of Illinois Urbana-Champaign
5.21.2024
"""

import pooch

import numpy as np
import pandas as pd

from var_assim.config import DATA_DIR_ABS
from logging import Logger
from argparse import Namespace


class EmissionsBaseline:
    """Emissions baseline class.

    Parameters
    ----------
    scenario: string
        scenario that is the emissions baseline

    t_min: int
        minimum tim for time series

    t_max: int
        maximum time for time series
    """

    def __init__(
        self,
        logger: Logger,
        args: Namespace,
        t_min: int,
        t_max: int,
        geo: bool = False,
        Prior: object = None,
        Truth: object = None,
        T_START: int = 2020,
        T_END: int = 2070,
        print_level: int = 1,
    ):

        self.logger = logger
        self.scenario = args.scenario
        self.sai_ramp = args.sai_ramp

        self.t_min = int(t_min)
        self.t_max = int(t_max)

        self.geo = (
            geo  # whether or not this class should have geoengineering attributes
        )
        self.DEG_PER_DEC = args.deg_p_dec  # degrees C offset by geo per decade
        self.L = Prior.L_CEN
        self.G = Prior.G_CEN
        self.EPS = Prior.EPS_CEN
        self.F_EFF_GEO = Truth.F_EFF_GEO_TR

        self.T_START = int(T_START)  # year SAI begins
        self.T_END = int(T_END)  # year SAI levels out
        self.TOTAL_TEMP_OFFSET = (
            self.DEG_PER_DEC * (self.T_END - self.T_START) / 10.0
        )  # total temperature offset for geo program

        # set time bounds for time series
        self.times = np.arange(self.t_min, self.t_max, 1)  # time range
        self.times_ext = np.arange(self.t_min, self.t_max + 1, 1)  # time range extended

        # step 1: import .csv containing emissions data
        self._import_emissions_timeseries()

        # step 1a: check if i passed a valid scenario
        if np.all(self.scenario != self.df_emis["Scenario"].unique()):
            raise ValueError(
                "Invalid scenario. Valid scenarios are:\n{}.".format(
                    self.df_emis["Scenario"].unique()
                )
            )

        # step 2: parse the big dataframe into individual gas time series
        self._parse_species()

        # step 3: if we care about geoengineering, include those emissions
        if self.geo:
            self._make_geo_time_series()

        else:
            self.emis["geo"] = np.zeros_like(self.times_ext)  # no geoengineering
            self.forcing["geo"] = np.zeros_like(self.times_ext)  # no geo

        if print_level == 1:
            self.logger.info(f"    > Emissions baseline for {self.scenario} created")

        elif print_level == 2:
            self.logger.info(
                f"        >> Emissions baseline for {self.scenario} created"
            )

        else:
            self.logger.info(f"Emissions baseline for {self.scenario} created")

    def _import_emissions_timeseries(self):
        """Import time series of emissions for each gas species."""

        # get current working directory and set path
        EMIS_DATA_PATH = DATA_DIR_ABS / "input" / "rcmip_emissions_data.csv"
        CONC_DATA_PATH = DATA_DIR_ABS / "input" / "rcmip_conc_data.csv"

        # try to import data. if it doesn't exist, we accept a file not found
        # error, and download the file from Zenodo
        try:
            self.df_emis = pd.read_csv(EMIS_DATA_PATH, na_values=np.nan)
            self.df_conc = pd.read_csv(CONC_DATA_PATH, na_values=np.nan)

        except FileNotFoundError:
            # if you don't find the file, download it and save it
            self.logger.info(
                "    ⚠️ Did not find data for emissions and/or concentrations, fetching..."
            )
            emis_file = pooch.retrieve(
                url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
                known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
            )

            conc_file = pooch.retrieve(
                url="doi:10.5281/zenodo.4589756/rcmip-concentrations-annual-means-v5-1-0.csv",
                known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
            )

            # open using pandas, and save to local directories
            self.df_emis = pd.read_csv(emis_file)
            self.df_conc = pd.read_csv(conc_file)

            # save the downloaded file to .csv
            self.logger.info(
                f"        Saved harmonized emissions data to: {EMIS_DATA_PATH}"
            )
            self.logger.info(
                f"        Saved harmonized concentrations data to: {CONC_DATA_PATH}"
            )
            self.df_emis.to_csv(EMIS_DATA_PATH)
            self.df_conc.to_csv(CONC_DATA_PATH)

    def _parse_species(self):
        """Parse dataframe to only have emissions for gases we care about."""

        # passed as an init, but for now, i'm leaving it here
        # emis keywords are needed for aerosols, concentrations are for
        # greenhouse gases
        self.emis_keywords = [
            "Emissions|BC",
            "Emissions|Sulfur",
            "Emissions|OC",
            "Emissions|CO2",
        ]

        # truncate Emissions| bit from each gas label in the RCMIP file
        self.emis_keywords_trunc = [
            i.replace("Emissions|", "") for i in self.emis_keywords
        ]

        self.conc_keywords = [
            "Atmospheric Concentrations|N2O",
            "Atmospheric Concentrations|CO2",
            "Atmospheric Concentrations|CH4",
        ]

        # truncate Emissions| bit from each gas label in the RCMIP file
        self.conc_keywords_trunc = [
            i.replace("Atmospheric Concentrations|", "") for i in self.conc_keywords
        ]

        # now make dictionary of gas time series for our scenario
        self.emis = {}
        self.ref_emis = {}
        self.conc = {}
        self.forcing = {}  # only filled in when self.geo = True above

        # loop through emissions gases and make time series
        for i_spec in range(len(self.emis_keywords)):
            # specify labels
            tmp_spec = self.emis_keywords[i_spec]
            tmp_spec_trunc = self.emis_keywords_trunc[i_spec]

            # pull time series of gas
            # NOTE: we're interested in World emissions for CMIP6
            tmp_df = self.df_emis.loc[
                (self.df_emis["Scenario"] == self.scenario)
                & (self.df_emis["Region"] == "World")
                & (self.df_emis["Mip_Era"] == "CMIP6")
                & (self.df_emis["Variable"] == tmp_spec)
            ]

            # extract time values between 1750 and 2100
            # NOTE 1: for future emissions, we're not given annual emissions,
            # but rather emissions on ten year intervals. so we interpolate
            # over the NaNs in the selected time range linearly.

            # NOTE 2: N2O is in kt N2O / yr, so we change it to Mt N2O / yr to
            # match all the other species

            if tmp_spec != "Emissions|N2O":
                tmp_df_vals = (
                    tmp_df.loc[:, str(self.t_min) : str(self.t_max)]
                    .interpolate(axis=1)
                    .values[0]
                )
                self.ref_emis[tmp_spec_trunc] = np.mean(
                    tmp_df.loc[:, str(1750) : str(1850)].values
                )

            else:
                tmp_df_vals = tmp_df.loc[
                    :, str(self.t_min) : str(self.t_max)
                ].interpolate(axis=1).values[0] * (1 / 1000.0)

            # check if there are nans in this interpolated time series. if there are, shift the t_min window backwards
            # by the number of nan years, reinterpolate, and subselect relevant years of data
            if np.any(np.isnan(tmp_df_vals)):
                N_nans = len(np.where(np.isnan(tmp_df_vals))[0])
                tmp_df_vals = (
                    tmp_df.loc[:, str(self.t_min - N_nans) : str(self.t_max)]
                    .interpolate(axis=1)
                    .values[0]
                )
                tmp_df_vals = tmp_df_vals[N_nans:]

            # save to dictionary of species
            self.emis[tmp_spec_trunc] = tmp_df_vals

        # loop through concentrations gases and make time series
        for i_spec in range(len(self.conc_keywords)):
            # specify labels
            tmp_spec = self.conc_keywords[i_spec]
            tmp_spec_trunc = self.conc_keywords_trunc[i_spec]

            # pull time series of gas
            # NOTE: we're interested in World emissions for CMIP6
            tmp_df = self.df_conc.loc[
                (self.df_conc["Scenario"] == self.scenario)
                & (self.df_conc["Region"] == "World")
                & (self.df_conc["Mip_Era"] == "CMIP6")
                & (self.df_conc["Variable"] == tmp_spec)
            ]

            # NOTE 1: for future emissions, we're not given annual emissions,
            # but rather emissions on ten year intervals. so we interpolate
            # over the NaNs in the selected time range linearly.
            tmp_df_vals = (
                tmp_df.loc[:, str(self.t_min) : str(self.t_max)]
                .interpolate(axis=1)
                .values[0]
            )
            # print(tmp_df_vals)

            # check if there are nans in this interpolated time series. if there are, shift the t_min window backwards
            # by the number of nan years, reinterpolate, and subselect relevant years of data
            if np.any(np.isnan(tmp_df_vals)):
                N_nans = len(np.where(np.isnan(tmp_df_vals))[0])
                tmp_df_vals = (
                    tmp_df.loc[:, str(self.t_min - N_nans) : str(self.t_max)]
                    .interpolate(axis=1)
                    .values[0]
                )
                tmp_df_vals = tmp_df_vals[N_nans:]

            # save to dictionary of species
            self.conc[tmp_spec_trunc] = tmp_df_vals

    def _make_geo_time_series(self):
        # add sulfur emissions from geoengineering
        # times where geoengineering is ramped up
        geo_ramp_up_times = self.times_ext[
            (self.times_ext >= self.T_START) & (self.times_ext <= self.T_END)
        ]
        # times where SAI is held constant
        geo_constant_times = self.times_ext[self.times_ext > self.T_END]

        # make SAI ramp up
        if self.sai_ramp == "linear":
            geo_ramp = (
                self.TOTAL_TEMP_OFFSET * (self.L + self.G * self.EPS) / self.F_EFF_GEO
            ) * ((geo_ramp_up_times - self.T_START) / (self.T_END - self.T_START))

        elif self.sai_ramp == "fast":
            geo_ramp = (
                self.TOTAL_TEMP_OFFSET * (self.L + self.G * self.EPS) / self.F_EFF_GEO
            ) * ((geo_ramp_up_times - self.T_START) / (self.T_END - self.T_START)) ** (
                1 / 3
            )

        elif self.sai_ramp == "slow":
            geo_ramp = (
                self.TOTAL_TEMP_OFFSET * (self.L + self.G * self.EPS) / self.F_EFF_GEO
            ) * ((geo_ramp_up_times - self.T_START) / (self.T_END - self.T_START)) ** 3

        # set remaining years of SAI to final t levels
        geo_constant = np.ones(len(geo_constant_times), dtype=float) * geo_ramp[-1]

        # stack the arrays together and store in class attributes
        geo_emissions = np.hstack([geo_ramp, geo_constant])
        self.emis["geo"] = geo_emissions
        self.forcing["geo"] = -self.F_EFF_GEO * self.emis["geo"]


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    from var_assim.logging_utils import setup_logger
    from var_assim.config import FIGS_DIR

    log = setup_logger()

    SAI_RAMPS = ["linear", "fast", "slow"]
    RAMP_LABELS = {
        "linear": "linear ($t$)",
        "fast": "fast ($t^{1/3}$)",
        "slow": "slow ($t^3$)",
    }

    class Prior:
        def __init__(self):
            self.L_CEN = 1.06
            self.G_CEN = 0.7
            self.EPS_CEN = 1.58

    class Truth:
        def __init__(self):
            self.F_EFF_GEO_TR = 0.09

    p = Prior()
    t = Truth()

    t_min, t_max = 2025, 2100
    T_START, T_END = 2025, 2075

    # --- checks ---
    forcings = {}
    for ramp in SAI_RAMPS:
        args = argparse.Namespace(scenario="ssp245", sai_ramp=ramp, deg_p_dec=0.1)
        e = EmissionsBaseline(
            log,
            args,
            t_min,
            t_max,
            geo=True,
            Prior=p,
            Truth=t,
            T_START=T_START,
            T_END=T_END,
        )
        forcings[ramp] = e.forcing["geo"]

        # forcing should be monotonically non-increasing (more negative) during ramp
        ramp_mask = (e.times_ext >= T_START) & (e.times_ext < T_END)
        assert np.all(
            np.diff(e.forcing["geo"][ramp_mask]) <= 0
        ), f"FAIL: '{ramp}' forcing is not monotonically decreasing during ramp-up"

        # forcing should be constant after T_END
        plateau_mask = e.times_ext >= T_END
        plateau_vals = e.forcing["geo"][plateau_mask]
        assert np.allclose(
            plateau_vals, plateau_vals[0]
        ), f"FAIL: '{ramp}' forcing is not constant after T_END"

        print(f"  [OK] {ramp}: plateau forcing = {plateau_vals[0]:.4f} W m⁻²")

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ramp in SAI_RAMPS:
        axes[0].plot(e.times_ext, forcings[ramp], label=RAMP_LABELS[ramp])

        # normalized shape (0 → 1) to make ramp profile differences visible
        f_min = forcings[ramp].min()
        axes[1].plot(e.times_ext, forcings[ramp] / f_min, label=RAMP_LABELS[ramp])

    for ax in axes:
        ax.axvline(T_START, color="k", linestyle="--", linewidth=0.8, label="SAI start")
        ax.axvline(T_END, color="k", linestyle=":", linewidth=0.8, label="SAI plateau")
        ax.set_xlabel("Year")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("SAI forcing (W m$^{-2}$)")
    axes[0].set_title("Forcing by ramp type")
    axes[1].set_ylabel("Normalized forcing (0 → 1)")
    axes[1].set_title("Ramp shape comparison")

    fig.tight_layout()
    figpath = FIGS_DIR / "checks" / "sai_ramp_check.png"
    fig.savefig(figpath, dpi=400, bbox_inches="tight")
    print(f"\nSAI ramp check figure saved to:\n  {figpath}")
