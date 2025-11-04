"""Two layer model 4DVAR with internal variability.

Adam Michael Bauer
University of Illinois Urbana Champaign
8.23.2024

To run:
    python main_margobs_pco2sulwc.py [scenario] [P] [L or F] [SIGMA]
        [N_windows] [N_ENS] [SAVE_OUTPUT]
"""

import sys
import os
import time

import numpy as np
import xarray as xr
import dask.bag as db

from src.emis import EmissionsBaseline
from src.pco2sulwc.dynamics import get_nonlin_path
from src.pco2sulwc.obs import get_obs_from_dynamics
from src.pco2sulwc.parallelization import EnsembleMember, runner_4dvar
from src.pco2sulwc.model_errors import gen_noise_ts
from src.stats.covar import get_covar_white
from src.stats.draws import get_prior_draws
from dask.distributed import Client
from datatree import DataTree

if __name__ == '__main__':
    # initiate DASK client
    c = Client()
    print(c)

    # parse command line stuff
    SCENARIO = sys.argv[1]
    TMIN = int(sys.argv[2])
    AR_P = int(sys.argv[3])
    L_or_F = sys.argv[4]
    SIGMA = float(sys.argv[5])
    N_windows = int(sys.argv[6])
    N_ENS = int(sys.argv[7])
    SAVE_OUTPUT = int(sys.argv[8])

    # ----------------------------------------
    # Initialize the problem
    # ----------------------------------------
    # set the time discretization
    DT = 1.0

    # make set of assimilation windows
    tmax_assims = np.linspace(TMIN, 2100, N_windows, dtype=int)[1:]

    # central values of priors on parameters
    L_CEN = 1.258  # overall climate sensitivity
    G_CEN = 0.7  # layer transfer coefficient
    C1_CEN = 8  # heat capacity of surface layer
    C2_CEN = 100  # heat capacity of ocean layer
    F1_CO2_CEN = 4.57  # forcing from log term in CO2
    F3_CO2_CEN = 0.086  # forcing from sqrt term in CO2
    B_SO2_CEN = -0.955789  # true linear radiative sensitivity of SO2
    A_SO2_CEN = -0.00474  # true log radiative sensitivity of SO2
    C_SO2_CEN = 170.6  # shape parameter for SO2

    # true parameters used to make observations
    L_TR = 1.258  # sensitivity
    G_TR = 0.7  # layer transfer coefficient
    C1_TR = 8  # heat capacity of surface layer
    C2_TR = 100  # heat capacity of ocean layer
    F1_CO2_TR = 4.57  # forcing from log term in CO2
    F3_CO2_TR = 0.086  # forcing from sqrt term in CO2
    B_SO2_TR = -0.955789  # true linear radiative sensitivity of SO2
    A_SO2_TR = -0.00474  # true log radiative sensitivity of SO2
    C_SO2_TR = 170.6  # shape parameter for SO2

    # shift lamnda or F1/F3 for different ECS values
    if L_or_F == 'L':
        L_TR = 1.258 / SIGMA  # sensitivity
        F1_CO2_TR = 4.57  # forcing from log term in CO2
        F3_CO2_TR = 0.086  # forcing from sqrt term in CO2

    else:
        L_TR = 1.258  # sensitivity
        F1_CO2_TR = 4.57 * SIGMA  # forcing from log term in CO2
        F3_CO2_TR = 0.086 * SIGMA  # forcing from sqrt term in CO2

    ECS_TR = (F1_CO2_TR * np.log(2) + F3_CO2_TR * (np.sqrt(2) - 1) *
              np.sqrt(278.3)) / L_TR

    """WARM START MODULE
    """
    # "warm start" model to get the central estimates of the initial conditions
    # for temperature in our start year
    e_ws = EmissionsBaseline(SCENARIO, 1850, TMIN)

    # make vector of parameters for warm start
    theta_ws = np.hstack([np.array([0.0, 0.0, 0.0,
                         L_TR, G_TR, C1_TR, C2_TR, F1_CO2_TR, F3_CO2_TR,
                         A_SO2_TR, B_SO2_TR, C_SO2_TR]),
                          np.zeros_like(e_ws.conc['CO2'])])

    # make "warm start" to get true initial conditions
    data_ws, _ = get_nonlin_path(e_ws, theta_ws, 1850, TMIN, DT)

    # true initial conditions
    T1_tr = data_ws[0, -1] # surface temperature at TMIN
    T1_CEN = data_ws[0, -1] # central estimate is the truth
    T2_tr = data_ws[1, -1]  # ocean layer temperature at TMIN
    T2_CEN = data_ws[1, -1]  # central estimate is the truth

    # internal variability standard deviation
    INT_VAR_STD = 0.4  # from Proistosescu and Huybers, Sci Adv, 2017

    # dictionary to make datatree out of later
    datatree_dict = {}

    print("==================================================================")
    print("Simulation attributes:")
    print("------------------------------------------------------------------")
    print("Socio-economic pathway: {}".format(SCENARIO))
    print("The initial time is: {}".format(TMIN))
    print("Temperature is forced with AR({}) noise.".format(AR_P))
    if L_or_F == 'L':
        print("Lambda is being adjusted to change ECS.")

    else:
        print("Forcing sensitivities are being adjusted to change ECS.")

    print("ECS = {} x 3 deg C (3 deg C is the average).".format(SIGMA))
    print("ECS = {}.".format(ECS_TR))
    print("There are {} assimilation windows, starting in {} and ending in 2100 (this implies adding one window adds {} years of observations).".format(N_windows - 1, TMIN, (2100 - TMIN)/len(tmax_assims)))
    print("The 4DVAR ensemble has {} members.".format(N_ENS))
    print("==================================================================")


    for TMAX in tmax_assims:
        print("--------------------------------------")
        print("WE ARE ON TMAX = {}".format(TMAX))
        print("--------------------------------------")

        # set seed so we get same draws for each assimilation window
        np.random.seed(1000)

        # make emissions baseline
        e = EmissionsBaseline(SCENARIO, TMIN, TMAX)

        # make model errors and their covariance matrix
        mod_errors, mod_error_covar = gen_noise_ts(AR_P, len(e.conc['CO2']),
                                                   INT_VAR_STD,
                                                   CORR_COEFFS=[0.2])

        # true vector of controls: initial conditions, parameters, and model
        # errors
        theta_tr = np.hstack([np.array([T1_tr, T2_tr,
                             C1_TR * T1_tr + C2_TR * T2_tr,
                             L_TR, G_TR, C1_TR, C2_TR, F1_CO2_TR, F3_CO2_TR,
                             A_SO2_TR, B_SO2_TR, C_SO2_TR]), mod_errors])

        # central value of priors on each parameter
        theta_prior_cent = np.hstack([np.array([T1_CEN, T2_CEN,
                                     C1_CEN * T1_CEN + C2_CEN * T2_CEN,
                                     L_CEN, G_CEN, C1_CEN, C2_CEN, F1_CO2_CEN,
                                     F3_CO2_CEN, A_SO2_CEN, B_SO2_CEN,
                                                C_SO2_CEN]), mod_errors])

        # make true data path over this time window
        data_tr_p, times = get_nonlin_path(e, theta_tr, TMIN, TMAX, DT)

        # note stds of priors and obs
        OBS_T1_STD = 1.0  # observation noise in measuring T1/T2
        OBS_Q_STD = 1.0  # observation noise in measuring ocean heat content
        T_IC_STD = 0.2  # initial condition std for t1 and t2
        PRIOR_STD_FACTOR = 0.3  # implies X% std for prior
        prior_stds = np.hstack([np.array([T_IC_STD, T_IC_STD,
                                          C1_TR * T_IC_STD + C2_TR *
                                          T_IC_STD]),
                                np.abs(theta_prior_cent[3:12]) * PRIOR_STD_FACTOR,
                                np.ones(len(mod_errors))])

        # make inverse covariance matrices for white noise
        inv_covar_prior = get_covar_white(prior_stds,
                                          len(prior_stds),
                                          inv=True)

        # add in inverse covarianace matrix of model errors (which may not be
        # white, like the other parameters)
        inv_covar_prior[-len(mod_errors):,
                        -len(mod_errors):] = np.linalg.inv(mod_error_covar)

        # make observation error covariance matrices
        inv_covar_T1_obs = get_covar_white(np.array([OBS_T1_STD] *
                                                    len(times)),
                                           len(times), inv=True)

        inv_covar_Q_obs = get_covar_white(np.array([OBS_Q_STD] *
                                                   len(times)), len(times),
                                          inv=True)

        # make observations from true data
        obs = get_obs_from_dynamics(data_tr_p)

        # ----------------------------------------
        # Set up optimization
        # ----------------------------------------
        max_iter = 100  # maximum iterations
        tol = 0.001  # tolerance for convergence in 4DVAR

        # give first guess at initial conditions
        theta_prior = get_prior_draws(theta_prior_cent,
                                      np.linalg.inv(inv_covar_prior),
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
        part_size = 20  # twenty members per thread in Dask

        # make dask bag for evaluation
        bag_ens = db.from_sequence(ensemble_members,
                                   partition_size=part_size).map(runner_4dvar)

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
        names = np.hstack([['T1', 'T2', 'Q', 'L', 'G', 'C1', 'C2', 'F1_CO2',
                            'F3_CO2', 'A_SO2', 'B_SO2', 'C_SO2'],
                           ['q' + str(i) for i in range(len(times))]])

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
                               'run_time': t1 - t0,
                               'ECS': ECS_TR,
                               'internal_variability_std': INT_VAR_STD})

        datatree_dict[str(TMAX)] = ds

    dt = DataTree.from_dict(datatree_dict, 'TMAX')

    if SAVE_OUTPUT:
        # get current directory and save
        cwd = os.getcwd()
        sim_type = 'pco2sulwc'
        path = '~/a/two-layer-4dvar/data/output/pco2sulwc/margobs_ws_highint_'\
            + SCENARIO + "_"\
            + sim_type + "_"\
            + "TMIN" + str(TMIN) + "_"\
            + "AR" + str(AR_P) + "_"\
            + "d" + L_or_F + "_"\
            + "sig" + str(SIGMA) + "_"\
            + "Nwinds" + str(N_windows) + "_"\
            + "Nens" + str(N_ENS) + ".nc"
        dt.to_netcdf(filepath=path, mode='w', format='NETCDF4',
                     engine='netcdf4')

        print("\nOutput successfully saved to:\n{}\n".format(path))

    else:
        print(dt)
