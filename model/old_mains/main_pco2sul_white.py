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
from src.pco2sul.checks import get_tlm_check, get_adj_id_check, get_cost_grad_check
from src.pco2sul.dynamics import get_nonlin_path
from src.pco2sul.cost import cost, grad
from src.pco2sul.obs import get_obs_from_dynamics
from src.pco2sul.parallelization import EnsembleMember, runner_4dvar
from src.stats.covar import get_inv_covar_white
from src.stats.draws import get_prior_draws
from dask.distributed import Client

if __name__ == '__main__':
    # initiate DASK client
    c = Client()
    print(c)

    # parse command line stuff
    SCENARIO = sys.argv[1]
    CHECK_TLM = int(sys.argv[2])
    CHECK_ADJ = int(sys.argv[3])
    SOLVE_ASSIM = int(sys.argv[4])
    SAVE_OUTPUT = int(sys.argv[5])

    # ----------------------------------------
    # Initialize the problem
    # ----------------------------------------
    # times
    TMIN = 1900
    TMAX = 2100
    DT = 1.0

    # emissions baseline
    e = EmissionsBaseline(SCENARIO, TMIN, TMAX)

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
    T1_tr = 0.1  # surface temperatureg at TMIN
    T2_tr = 0.05  # ocean layer temperature at TMIN

    # true vector of controls
    theta_tr = np.array([T1_tr, T2_tr,
                         C1_TR * T1_tr + C2_TR * T2_tr,
                         L_TR, G_TR, C1_TR, C2_TR, F1_CO2_TR, F3_CO2_TR,
                         A_SO2_TR, B_SO2_TR, C_SO2_TR])

    # errors (assumed constant for now)
    STD_T1_OBS = 0.2  # observation error
    STD_Q_OBS = C1_TR * STD_T1_OBS + C2_TR * STD_T1_OBS  # heat content obs error

    data_tr_p, times = get_nonlin_path(e, theta_tr, TMIN, TMAX, DT)

    # note stds of priors and data
    PRIOR_STD_FACTOR = 0.1  # implies 10% std for prior
    prior_stds = np.hstack([np.array([STD_T1_OBS, STD_T1_OBS, STD_Q_OBS]),
                           np.abs(theta_tr[3:]) * PRIOR_STD_FACTOR])

    # make inverse covariance matrices for white noise
    inv_covar_prior = get_inv_covar_white(prior_stds,
                                          len(prior_stds))

    inv_covar_T1_obs = get_inv_covar_white(np.array([STD_T1_OBS] * len(times)),
                                           len(times))

    inv_covar_Q_obs = get_inv_covar_white(np.array(
        [(C1_TR + C2_TR) * STD_T1_OBS] * len(times)), len(times))

    # make observations
    obs = get_obs_from_dynamics(data_tr_p, noise=True,
                                noise_params=[[0.0, 0.0],
                                              [np.linalg.inv(inv_covar_T1_obs),
                                              np.linalg.inv(inv_covar_Q_obs)]])

    # -----------------------------------------------
    # If desired, check tangent linear model accuracy
    # -----------------------------------------------
    if CHECK_TLM:
        # set (small) integration horizon and min/max perturbation sizes
        T0_CHECK = 2020
        T1_CHECK = 2040
        ALPHA_MIN = 1e-16
        ALPHA_MAX = 0.1

        # check tlm and save output of that procedure
        _ = get_tlm_check(e, theta_tr, T0_CHECK, T1_CHECK, DT, ALPHA_MIN,
                          ALPHA_MAX, SAVE_RESULTS=True)

    # -----------------------------------------------
    # If desired, check adjoint accuracy
    # -----------------------------------------------
    if CHECK_ADJ:
        # Check 1: Adjoint Identity
        T_CHECK_MAX = 10

        # do first check
        _ = get_adj_id_check(e, theta_tr, TMIN, TMIN + T_CHECK_MAX, DT,
                             SAVE_RESULTS=True)

        # Check 2: Gradient of Cost Function
        ALPHA_MIN = 1e-16
        ALPHA_MAX = 0.1

        path_true_tr, times_tr = get_nonlin_path(e, theta_tr, TMIN,
                                                 TMIN + T_CHECK_MAX, DT)

        # make inverse covariance matrices for white noise
        inv_covar_prior_tr = get_inv_covar_white(prior_stds,
                                                 len(prior_stds))

        inv_covar_T1_obs_tr = get_inv_covar_white(np.array([OBS_T1_STD] *
                                                           len(times_tr)),
                                                  len(times_tr))

        inv_covar_Q_obs_tr = get_inv_covar_white(np.array(
            [(C1_TR + C2_TR) * OBS_T1_STD] * len(times_tr)), len(times_tr))

        # make observations
        obs_true_tr = get_obs_from_dynamics(path_true_tr)

        # run check function
        _ = get_cost_grad_check(control=np.hstack([theta_tr]) * 1.1,
                                cost_args=[np.hstack([theta_tr]),
                                           inv_covar_prior_tr,
                                           inv_covar_T1_obs_tr,
                                           inv_covar_Q_obs_tr,
                                           obs_true_tr,
                                           e,
                                           TMIN,
                                           TMIN + T_CHECK_MAX,
                                           DT],
                                ALPHA_MIN=ALPHA_MIN,
                                ALPHA_MAX=ALPHA_MAX,
                                SAVE_RESULTS=True)

    # ----------------------------------------
    # Set up optimization
    # ----------------------------------------
    max_iter = 60  # maximum iterations
    N_ENS = 300  # size of ensemble
    tol = 0.005  # tolerance for convergence in 4DVAR

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
    if SOLVE_ASSIM:
        t0 = time.time()
        # do dask evaluation of runner
        print("Solving 4DVAR using DASK...")
        if N_ENS > 100000:
            part_size = 100
        else:
            part_size = len(ensemble_members)

        bag_ens = db.from_sequence(ensemble_members,
                                   npartitions=part_size).map(runner_4dvar)
        # map and compute
        opt_ensmems = bag_ens.compute()
        t1 = time.time()

    # setup data arrays
    data = np.zeros((N_ENS, len(theta_tr), len(times)))
    controls = np.zeros((N_ENS, len(theta_tr)))
    costs = np.zeros((N_ENS))
    l2s = np.zeros((N_ENS))

    # make history arrays
    data_hist = np.zeros((N_ENS, len(theta_tr), max_iter + 1, len(times)))
    controls_hist = np.zeros((N_ENS, len(theta_tr), max_iter + 1))
    cost_hist = np.zeros((N_ENS, max_iter + 1))
    l2s_hist = np.zeros((N_ENS, max_iter + 1))

    # track flags for optimization
    flags = np.zeros(N_ENS)

    mc = 0
    for m in opt_ensmems:
        data[mc] = m.data
        controls[mc] = m.control
        costs[mc] = m.cost
        l2s[mc] = m.l2

        data_hist[mc] = m.data_hist
        controls_hist[mc] = m.controls_hist
        cost_hist[mc] = m.cost_hist
        l2s_hist[mc] = m.l2s_hist

        flags[mc] = m.flag

        mc += 1

    # -----------------------------------------------
    # If desired, save the output of the optimization
    # -----------------------------------------------
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
                               'cost_hist': (['ens_mem', 'iter'], cost_hist),
                               'controls_hist': (['ens_mem', 'vari', 'iter'],
                                                 controls_hist),
                               'flag': (['ens_mem'], flags),
                               'obs': (['obs_var', 'time'], obs),
                               'data_truth': (['vari', 'time'], data_tr_p),
                               'controls_truth': (['vari'], theta_tr)},
                    coords={'time': (['time'], times),
                            'iter': (['iter'], np.arange(0, max_iter + 1, 1)),
                            'vari': (['vari'], names),
                            'ens_mem': (['ens_mem'], np.arange(0, N_ENS, 1)),
                            'obs_var': (['obs_var'], ['T1', 'Q'])},
                    attrs={'TMIN': TMIN,
                           'TMAX': TMAX,
                           'DT': DT,
                           'max_iter': max_iter,
                           'tol': tol,
                           'run_time': t1 - t0})

    if SAVE_OUTPUT:
        # get current directory and save 
        cwd = os.getcwd()
        path = cwd + "/data/output/pco2sul_wind" + str(TMIN) + "_" + str(TMAX)\
            + "_N" + str(N_ENS) + "_white.nc"
        ds.to_netcdf(path=path, mode='w', format='NETCDF4', engine='netcdf4')

        print("\nOutput successfully saved to:\n{}\n".format(path))

    else:
        print(ds.run_time)
