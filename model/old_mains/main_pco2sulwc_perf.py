"""Two layer model WC-4DVAR with perfect observations.

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
from src.pco2sulwc.checks import get_tlm_check, get_adj_id_check, get_cost_grad_check
from src.pco2sulwc.dynamics import get_nonlin_path
from src.pco2sulwc.obs import get_obs_from_dynamics
from src.pco2sulwc.parallelization import EnsembleMember, runner_4dvar
from src.stats.covar import get_covar_white
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

    # set seed for consistency's sake
    np.random.seed(4)

    # ----------------------------------------
    # Initialize the problem
    # ----------------------------------------
    # times
    TMIN = 1850
    TMAX = 2000
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
    T1_tr = 0.0  # surface temperature at TMIN
    T2_tr = 0.0  # ocean layer temperature at TMIN

    # make true model errors
    INT_VAR_STD = 0.2
    mod_error_covar = get_covar_white(np.array([INT_VAR_STD] *
                                               len(e.conc['CO2'])),
                                      len(e.conc['CO2']),
                                      inv=False)

    mod_errors = np.random.multivariate_normal([0.0] * len(e.conc['CO2']),
                                               mod_error_covar)

    # true vector of controls: initial conditions, parameters, and model errors
    theta_tr = np.hstack([np.array([T1_tr, T2_tr,
                         C1_TR * T1_tr + C2_TR * T2_tr,
                         L_TR, G_TR, C1_TR, C2_TR, F1_CO2_TR, F3_CO2_TR,
                         A_SO2_TR, B_SO2_TR, C_SO2_TR]), mod_errors])

    # get true paths
    data_tr_p, times = get_nonlin_path(e, theta_tr, TMIN, TMAX, DT)

    # note stds of priors and obs
    OBS_T1_STD = 1.0  # observation noise in measuring T1/T2
    T_IC_STD = 0.1  # initial condition std for t1 and t2
    PRIOR_STD_FACTOR = 0.2  # implies 10% std for prior
    prior_stds = np.hstack([np.array([T_IC_STD, T_IC_STD,
                                      C1_TR * T_IC_STD + C2_TR * T_IC_STD]),
                            np.abs(theta_tr[3:12]) * PRIOR_STD_FACTOR,
                            np.ones(len(mod_errors))])

    # make inverse covariance matrices for white noise
    inv_covar_prior = get_covar_white(prior_stds,
                                      len(prior_stds), inv=True)

    inv_covar_prior[-len(mod_errors):,
                    -len(mod_errors):] = np.linalg.inv(mod_error_covar)

    inv_covar_T1_obs = get_covar_white(np.array([OBS_T1_STD] * len(times)),
                                       len(times), inv=True)

    inv_covar_Q_obs = get_covar_white(np.array(
        [(C1_TR + C2_TR) * OBS_T1_STD] * len(times)), len(times), inv=True)

    # make observations
    obs = get_obs_from_dynamics(data_tr_p)

    # -----------------------------------------------
    # If desired, check tangent linear model accuracy
    # -----------------------------------------------
    if CHECK_TLM:
        # set (small) integration horizon and min/max perturbation sizes
        ALPHA_MIN = 1e-16
        ALPHA_MAX = 1.

        # check tlm and save output of that procedure
        _ = get_tlm_check(e, theta_tr, TMIN, TMAX, DT, ALPHA_MIN,
                          ALPHA_MAX, SAVE_RESULTS=True)

    # -----------------------------------------------
    # If desired, check adjoint accuracy
    # -----------------------------------------------
    if CHECK_ADJ:
        # Check 1: Adjoint Identity

        # do first check
        # _ = get_adj_id_check(e, theta_tr, TMIN, TMAX, DT,
        #                      SAVE_RESULTS=True)

        # Check 2: Gradient of Cost Function
        ALPHA_MIN = 1e-16
        ALPHA_MAX = 1.0

        # run check function
        _ = get_cost_grad_check(control=theta_tr * 1.1,
                                cost_args=[theta_tr,
                                           inv_covar_prior,
                                           inv_covar_T1_obs,
                                           inv_covar_Q_obs,
                                           obs,
                                           e,
                                           TMIN,
                                           TMAX,
                                           DT],
                                ALPHA_MIN=ALPHA_MIN,
                                ALPHA_MAX=ALPHA_MAX,
                                SAVE_RESULTS=True)

    # ----------------------------------------
    # Set up optimization
    # ----------------------------------------
    max_iter = 60  # maximum iterations
    N_ENS = 2  # size of ensemble
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
        if N_ENS > 100:
            part_size = 100
        else:
            part_size = len(ensemble_members)

        bag_ens = db.from_sequence(ensemble_members,
                                   npartitions=part_size).map(runner_4dvar)

        # opt_ensmems = []
        # for mem in ensemble_members:
        #    opt_ensmems.append(runner_4dvar(mem))

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

        # store output
        names = np.hstack([['T1', 'T2', 'Q', 'L', 'G', 'C1', 'C2', 'F1_CO2', 'F3_CO2',
                 'A_SO2', 'B_SO2', 'C_SO2'],
                           ['q' + str(i) for i in range(len(times))]])

        ds = xr.Dataset(data_vars={'data_final': (['ens_mem', 'vari', 'time'],
                                                  data),
                                   'l2s': (['ens_mem'], l2s),
                                   'costs': (['ens_mem'], costs),
                                   'controls': (['ens_mem', 'vari'], controls),
                                   'data_hist': (['ens_mem', 'vari', 'iters',
                                                  'time'], data_hist),
                                   'l2_hist': (['ens_mem', 'iters'], l2s_hist),
                                   'cost_hist': (['ens_mem', 'iters'],
                                                 cost_hist),
                                   'controls_hist': (['ens_mem', 'vari',
                                                      'iters'],
                                                     controls_hist),
                                   'flag': (['ens_mem'], flags),
                                   'obs': (['obs_var', 'time'], obs),
                                   'data_truth': (['vari', 'time'], data_tr_p),
                                   'controls_truth': (['vari'], theta_tr)},
                        coords={'time': (['time'], times),
                                'iters': (['iters'], np.arange(0, max_iter + 1,
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

    # -----------------------------------------------
    # If desired, save the output of the optimization
    # -----------------------------------------------
    if SAVE_OUTPUT:
        # get current directory and save 
        cwd = os.getcwd()
        path = cwd + "/data/output/pco2sulwc_wind" + str(TMIN) + "_"\
            + str(TMAX) + "_N" + str(N_ENS) + "_perf.nc"
        ds.to_netcdf(path=path, mode='w', format='NETCDF4', engine='netcdf4')

        print("\nOutput successfully saved to:\n{}\n".format(path))

    elif SOLVE_ASSIM:
        print(ds.run_time)
