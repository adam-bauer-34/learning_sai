"""Two layer model 4DVAR with perfect observations.

Adam Michael Bauer
University of Illinois Urbana Champaign
7.9.2024

To run: python main_co2sul_perf.py [scenario] [check_tlm] [check_adj] [solve_assim]
[save_output]
"""

import sys
import os
import time

import numpy as np
import xarray as xr
import dask.bag as db

from src.emis import EmissionsBaseline
from src.pco2sul.dynamics import get_nonlin_path
from src.pco2sul.obs import get_obs_from_dynamics
from src.pco2sul.parallelization import EnsembleMember, runner_4dvar
from src.stats.covar import get_inv_covar_white
from src.stats.draws import get_prior_draws
from dask.distributed import Client
from datatree import DataTree

if __name__ == '__main__':
    # initiate DASK client
    c = Client()
    print(c)

    # parse command line stuff
    SCENARIO = sys.argv[1]
    N_windows = int(sys.argv[2])
    N_ENS = int(sys.argv[3])
    SAVE_OUTPUT = int(sys.argv[4])

    # ----------------------------------------
    # Initialize the problem
    # ----------------------------------------
    # tmin is constant in each assimilation, as is DT. tmax will change.
    TMIN = 1850
    DT = 1.0

    # make set of assimilation windows
    tmax_assims = np.linspace(2000, 2100, N_windows, dtype=int)

    # true parameters
    L_TR = 1.4  # sensitivity
    G_TR = 0.7  # layer transfer coefficient
    C1_TR = 8  # heat capacity of surface layer
    C2_TR = 100  # heat capacity of ocean layer
    F1_CO2_TR = 4.57  # forcing from log term in CO2
    F3_CO2_TR = 0.086  # forcing from sqrt term in CO2
    B_SO2_TR = -0.955789  # true linear radiative sensitivity of SO2
    A_SO2_TR = -0.00474  # true log radiative sensitivity of SO2
    C_SO2_TR = 170.6  # shape parameter for SO2

    # true initial conditions
    T1_tr = 0.0  # surface temperatureg at TMIN
    T2_tr = 0.0  # ocean layer temperature at TMIN

    # true vector of controls
    theta_tr = np.array([T1_tr, T2_tr,
                         C1_TR * T1_tr + C2_TR * T2_tr,
                         L_TR, G_TR, C1_TR, C2_TR, F1_CO2_TR, F3_CO2_TR,
                         A_SO2_TR, B_SO2_TR, C_SO2_TR])

    # errors (assumed constant for now)
    VAR_T1_OBS = 1.  # observation error in temperature
    VAR_Q_OBS = 1.  # observation error in heat content
    # VAR_Q_OBS = C1_TR * VAR_T1_OBS + C2_TR * VAR_T1_OBS  # heat cont obs er

    # dictionary to make datatree out of later
    datatree_dict = {}

    for TMAX in tmax_assims:
        print("--------------------------------------")
        print("WE ARE ON TMAX = {}".format(TMAX))
        print("--------------------------------------")
        # make emissions baseline
        e = EmissionsBaseline(SCENARIO, TMIN, TMAX)

        # make true data path over this time window
        data_tr_p, times = get_nonlin_path(e, theta_tr, TMIN, TMAX, DT)

        # note stds of priors and data
        OBS_T1_STD = 0.1  # 0.1 K variance in white noise driving temperatures
        PRIOR_STD_FACTOR = 0.4  # implies 10% std for prior
        prior_stds = np.hstack([np.array([0.1, 0.1, 0.1 * (C1_TR + C2_TR)]),
                               np.abs(theta_tr[3:]) * PRIOR_STD_FACTOR])

        # make inverse covariance matrices for white noise
        inv_covar_prior = get_inv_covar_white(prior_stds,
                                              len(prior_stds))

        inv_covar_T1_obs = get_inv_covar_white(np.array([OBS_T1_STD] *
                                                        len(times)),
                                               len(times))

        inv_covar_Q_obs = get_inv_covar_white(np.array([VAR_Q_OBS] *
                                                       len(times)), len(times))

        # make observations from true data
        obs = get_obs_from_dynamics(data_tr_p)

        # ----------------------------------------
        # Set up optimization
        # ----------------------------------------
        max_iter = 60  # maximum iterations
        tol = 0.005  # tolerance for convergence in 4DVAR

        # set seed to keep prior draws the same across assimilation windows
        np.random.seed(1000)

        # give first guess at initial conditions
        theta_prior = get_prior_draws(theta_tr, np.linalg.inv(inv_covar_prior),
                                      N_ENS)

        # make list of ensemble members
        ensemble_members = [EnsembleMember(theta_p,
                                           -1, tol, max_iter,
                                           e, TMIN, TMAX, DT, theta_tr,
                                           inv_covar_prior, inv_covar_T1_obs,
                                           inv_covar_Q_obs, obs, times)
                            for theta_p in theta_prior]

        # solve the assimilation using dask
        t0 = time.time()

        # do dask evaluation of runner
        print("Solving 4DVAR using DASK...")
        if N_ENS > 100:
            part_size = 100
        else:
            part_size = len(ensemble_members)

        bag_ens = db.from_sequence(ensemble_members,
                                   npartitions=part_size).map(runner_4dvar)
        # map and compute
        opt_ensmems = bag_ens.compute()
        t1 = time.time()
        print("Done!")

        print("Processing output...")
        # store all the data
        data = np.array([m.data for m in opt_ensmems])
        controls = np.array([m.control for m in opt_ensmems])
        costs = np.array([m.cost for m in opt_ensmems])
        l2s = np.array([m.l2 for m in opt_ensmems])

        data_hist = np.array([m.data_hist for m in opt_ensmems])
        controls_hist = np.array([m.controls_hist for m in
                                  opt_ensmems])
        cost_hist = np.array([m.cost_hist for m in opt_ensmems])
        l2s_hist = np.array([m.l2s_hist for m in opt_ensmems])

        flags = np.array([m.flag for m in opt_ensmems])
        print("Done processing!\n")

        # ---------------------------------------------------------------------
        # make dataset for this assimilation window and save to dictionary that
        # we'll use to make a datatree later
        # ---------------------------------------------------------------------
        names = ['T1', 'T2', 'Q', 'L', 'G', 'C1', 'C2', 'F1_CO2', 'F3_CO2',
                 'A_SO2', 'B_SO2', 'C_SO2']

        ds = xr.Dataset(data_vars={'data_final': (['ens_mem', 'vari', 'time'],
                                                  data),
                                   'l2s': (['ens_mem'], l2s),
                                   'costs': (['ens_mem'], costs),
                                   'controls': (['ens_mem', 'vari'], controls),
                                   'data_hist': (['ens_mem', 'vari', 'iter',
                                                  'time'], data_hist),
                                   'l2_hist': (['ens_mem', 'iter'], l2s_hist),
                                   'cost_hist': (['ens_mem', 'iter'],
                                                 cost_hist),
                                   'controls_hist': (['ens_mem', 'vari',
                                                      'iter'],
                                                     controls_hist),
                                   'flag': (['ens_mem'], flags),
                                   'obs': (['obs_var', 'time'], obs),
                                   'data_truth': (['vari', 'time'], data_tr_p),
                                   'controls_truth': (['vari'], theta_tr)},
                        coords={'time': (['time'], times),
                                'iter': (['iter'], np.arange(0, max_iter + 1,
                                                             1)),
                                'vari': (['vari'], names),
                                'ens_mem': (['ens_mem'], np.arange(0, N_ENS,
                                                                   1)),
                                'obs_var': (['obs_var'], ['T1', 'Q'])},
                        attrs={'TMIN': TMIN,
                               'TMAX': TMAX,
                               'DT': DT,
                               'max_iter': max_iter,
                               'tol': tol,
                               'run_time': t1 - t0})

        datatree_dict[str(TMAX)] = ds

    dt = DataTree.from_dict(datatree_dict, 'TMAX')

    if SAVE_OUTPUT:
        # get current directory and save 
        cwd = os.getcwd()
        path = cwd + "/data/output/margobs_" + SCENARIO + "_pco2sul"\
            + "_Nwinds"\
            + str(N_windows) + "_N" + str(N_ENS) + "_perf.nc"
        dt.to_netcdf(filepath=path, mode='w', format='NETCDF4',
                     engine='netcdf4')

        print("\nOutput successfully saved to:\n{}\n".format(path))

    else:
        print(dt)
