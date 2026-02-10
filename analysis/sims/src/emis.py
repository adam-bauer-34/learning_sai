"""Emissions class for FaIR model.

Adam Michael Bauer
University of Illinois Urbana-Champaign
5.21.2024
"""

import os
import pooch

import numpy as np
import pandas as pd 

class EmissionsBaseline():
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

    def __init__(self, scenario, t_min, t_max,
                 geo=False, DEG_PER_DEC=0.1, LAMBDA=1.0, GAMMA=0.7, EPSILON=1.58, F_EFF_GEO=0.0,
                 T_START=2020, T_END=2070):
        self.scenario = scenario
        self.geo = geo  # whether or not this class should have geoengineering attributes
        self.DEG_PER_DEC = DEG_PER_DEC  # degrees C offset by geo per decade
        self.LAMBDA = LAMBDA  # climate feedback parameter, used to determine SAI rate
        self.GAMMA = GAMMA  # ocean heat uptake efficiency
        self.EPSILON = EPSILON  # pattern effect factor
        self.F_EFF_GEO = F_EFF_GEO
        self.T_START = int(T_START)  # year SAI begins
        self.T_END = int(T_END)  # year SAI levels out
        self.TOTAL_TEMP_OFFSET = self.DEG_PER_DEC * (self.T_END - self.T_START) / 10.  # total temperature offset for geo program

        # set time bounds for time series
        self.t_min = int(t_min)
        self.t_max = int(t_max)
        self.times = np.arange(self.t_min, self.t_max, 1)  # time range
        self.times_ext = np.arange(self.t_min, self.t_max + 1, 1)  # time range

        # step 1: import .csv containing emissions data
        self._import_emissions_timeseries()

        # step 1a: check if i passed a valid scenario
        if np.all(self.scenario != self.df_emis['Scenario'].unique()):
            raise ValueError("Invalid scenario. Valid scenarios are:\n{}."
                             .format(self.df_emis['Scenario'].unique()))

        # step 2: parse the big dataframe into individual gas time series
        self._parse_species()

        # step 3: if we care about geoengineering, include those emissions
        if self.geo:
            self._make_geo_time_series()
        
        else:
            self.emis['geo'] = np.zeros_like(self.times_ext)  # no geoengineering
            self.forcing['geo'] = np.zeros_like(self.times_ext)  # no geo

        print("\n------------------------------------------------------------------")
        print("Emissions baseline for scenario {} created successfully.".format(self.scenario))
        if self.geo:
            print("This scenario has geoengineering beginning in {} that offsets {} deg C per decade".format(self.T_START, self.DEG_PER_DEC))
        else:
            print("This scenario does not include geoengineering.")
        print("------------------------------------------------------------------\n")

    def _import_emissions_timeseries(self):
        """Import time series of emissions for each gas species.
        """

        # get current working directory and set path
        cwd = os.getcwd()
        # print(cwd)
        EMIS_DATA_PATH = cwd + '/data/rcmip_emissions_data.csv'
        CONC_DATA_PATH = cwd + '/data/rcmip_conc_data.csv'

        # print(EMIS_DATA_PATH)

        # try to import data. if it doesn't exist, we accept a file not found
        # error, and download the file from Zenodo
        try:
            self.df_emis = pd.read_csv(EMIS_DATA_PATH, na_values=np.nan)
            self.df_conc = pd.read_csv(CONC_DATA_PATH, na_values=np.nan)

        except FileNotFoundError:
            # if you don't find the file, download it and save it
            print("Did not find data for emissions and/or concentrations, downloading it now...")
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
            print("\nSaved harmonized emissions data to:\n{}".format(EMIS_DATA_PATH))
            print("\nSaved harmonized concentrations data to:\n{}".format(CONC_DATA_PATH))
            self.df_emis.to_csv(EMIS_DATA_PATH)
            self.df_conc.to_csv(CONC_DATA_PATH)

    def _parse_species(self):
        """Parse dataframe to only have emissions for gases we care about.
        """

        # passed as an init, but for now, i'm leaving it here
        # emis keywords are needed for aerosols, concentrations are for
        # greenhouse gases
        self.emis_keywords = ['Emissions|BC', 'Emissions|Sulfur',
                             'Emissions|OC', 'Emissions|CO2']

        # truncate Emissions| bit from each gas label in the RCMIP file
        self.emis_keywords_trunc = [i.replace('Emissions|', '') for i in
                                    self.emis_keywords]

        self.conc_keywords = ['Atmospheric Concentrations|N2O',
                              'Atmospheric Concentrations|CO2',
                              'Atmospheric Concentrations|CH4']

        # truncate Emissions| bit from each gas label in the RCMIP file
        self.conc_keywords_trunc = [i.replace('Atmospheric Concentrations|',
                                              '')
                                    for i in self.conc_keywords]

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
            tmp_df = self.df_emis.loc[(self.df_emis['Scenario'] ==
                                       self.scenario)
                                      & (self.df_emis['Region'] == 'World')
                                      & (self.df_emis['Mip_Era'] == 'CMIP6')
                                      & (self.df_emis['Variable'] == tmp_spec)]

            # extract time values between 1750 and 2100
            # NOTE 1: for future emissions, we're not given annual emissions, 
            # but rather emissions on ten year intervals. so we interpolate
            # over the NaNs in the selected time range linearly.

            # NOTE 2: N2O is in kt N2O / yr, so we change it to Mt N2O / yr to
            # match all the other species 

            if tmp_spec != 'Emissions|N2O':
                tmp_df_vals = tmp_df.loc[:,
                                         str(self.t_min):str(self.t_max)].interpolate(axis=1).values[0]
                self.ref_emis[tmp_spec_trunc] = np.mean(tmp_df.loc[:, str(1750):str(1850)].values)

            else:
                tmp_df_vals = tmp_df.loc[:,
                                         str(self.t_min):str(self.t_max)].interpolate(axis=1).values[0]\
                                * (1/1000.)

            # save to dictionary of species
            self.emis[tmp_spec_trunc] = tmp_df_vals

        # loop through concentrations gases and make time series
        for i_spec in range(len(self.conc_keywords)):
            # specify labels
            tmp_spec = self.conc_keywords[i_spec]
            tmp_spec_trunc = self.conc_keywords_trunc[i_spec]

            # pull time series of gas
            # NOTE: we're interested in World emissions for CMIP6
            tmp_df = self.df_conc.loc[(self.df_conc['Scenario'] ==
                                       self.scenario)
                                      & (self.df_conc['Region'] == 'World')
                                      & (self.df_conc['Mip_Era'] == 'CMIP6')
                                      & (self.df_conc['Variable'] == tmp_spec)]

            # NOTE 1: for future emissions, we're not given annual emissions, 
            # but rather emissions on ten year intervals. so we interpolate
            # over the NaNs in the selected time range linearly.
            tmp_df_vals = tmp_df.loc[:,
                                     str(self.t_min):str(self.t_max)].interpolate(axis=1).values[0]

            # save to dictionary of species
            self.conc[tmp_spec_trunc] = tmp_df_vals

    def _make_geo_time_series(self):
        # add sulfur emissions from geoengineering
        # times where geoengineering is ramped up
        geo_ramp_up_times = self.times_ext[(self.times_ext >= self.T_START) 
                                            & (self.times_ext < self.T_END)]
        # times where SAI is held constant 
        geo_constant_times = self.times_ext[self.times_ext >= self.T_END]
        
        # make SAI ramp up 
        geo_ramp = (self.TOTAL_TEMP_OFFSET * (self.LAMBDA + self.GAMMA * self.EPSILON) / self.F_EFF_GEO) * (
            (geo_ramp_up_times - self.T_START) / (self.T_END - self.T_START)
            )
        
        # set remaining years of SAI to final t levels
        geo_constant = np.ones(len(geo_constant_times), dtype=float) * geo_ramp[-1]
        
        # stack the arrays together and store in class attributes
        geo_emissions = np.hstack([geo_ramp, geo_constant])
        self.emis['geo'] = geo_emissions
        self.forcing['geo'] = - self.F_EFF_GEO * self.emis['geo']

if __name__ == '__main__':
    # small test script to verify geoengineering forcing is being generated
    # correctly
    import matplotlib.pyplot as plt
    from model.globals import FIGS_DIR
    plt.style.use('ambpy')

    scenario = 'ssp245'
    t_min = 2020
    t_max = 2100
    geo = True
    degs_per_dec = [0.0, 0.05, 0.1, 0.2]
    LAM = 4.58 * np.log(2) / 3.0  # feedback for ECS = 3.0
    print(LAM)
    GAM = 0.7  # use central value
    EPS = 1.58  # use central value
    F_EFF_GEO = 0.09
    ts = 2020
    tf = 2070
    geo_ts_emis = []
    geo_ts_force = []

    for deg in degs_per_dec:
        e = EmissionsBaseline(scenario, t_min, t_max,
                    geo=geo, DEG_PER_DEC=deg, LAMBDA=LAM, GAMMA=GAM, EPSILON=EPS, F_EFF_GEO=F_EFF_GEO,
                    T_START=ts, T_END=tf)
        geo_ts_emis.append(e.emis['geo'])
        geo_ts_force.append(e.forcing['geo'])

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for i in range(len(degs_per_dec)):
        ax[0].plot(e.times_ext, geo_ts_emis[i], label=str(degs_per_dec[i]) + ' deg C per decade')
        ax[1].plot(e.times_ext, geo_ts_force[i], label=str(degs_per_dec[i]) + ' deg C per decade')

    ax[0].set_ylabel("Emissions (MtSO$_2$)")
    ax[1].set_ylabel("Forcing (W m$^{-2}$)")

    ax[1].legend()
    
    figpath = FIGS_DIR + 'checks/geo_emis.png'
    fig.savefig(figpath, dpi=400)
    print("Emissions baseline check figure saved to:\n {}".format(figpath))

