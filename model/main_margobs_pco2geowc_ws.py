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
import warnings 

import numpy as np
import xarray as xr
import dask.bag as db

from model.src.emis import EmissionsBaseline
from model.src.pco2geowc.dynamics import get_nonlin_path
from model.src.pco2geowc.checks import *
from model.src.pco2geowc.obs import get_obs_from_dynamics
from model.src.pco2geowc.parallelization import EnsembleMember, runner_4dvar
from model.src.pco2geowc.model_errors import gen_noise_ts
from model.src.stats.covar import get_covar_white
from model.src.stats.draws import get_prior_draws
from dask.distributed import Client
from datatree import DataTree
from model import DATA_DIR

if __name__ == '__main__':
    # initiate DASK client
    c = Client()
    print(c)

    # parse command line stuff
    SCENARIO = sys.argv[1]
    TMIN = int(sys.argv[2])
    AR_P = int(sys.argv[3])
    DEG_PER_DEC = float(sys.argv[4])
    N_YEARS_RAMP = int(sys.argv[5])
    N_windows = int(sys.argv[6])
    N_ENS = int(sys.argv[7])
    SAVE_OUTPUT = int(sys.argv[8])

    # turn on if you want to test the TLM and ADJ
    CHECK_TLM = False
    CHECK_ADJ = False

    # filter out runtime warnings which clog log files
    # (they are natural in the scipy.minimize call)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ----------------------------------------
    # Initialize the problem
    # ----------------------------------------
    # set the time discretization
    DT = 1.0

    # make set of assimilation windows
    tmax_assims = np.linspace(TMIN, 2100, N_windows, dtype=int)[1:]

    # central values of priors on parameters
    L_CEN = 1.06  # overall climate sensitivity
    G_CEN = 0.7  # layer transfer coefficient
    C1_CEN = 8  # heat capacity of surface layer
    C2_CEN = 100  # heat capacity of ocean layer
    F1_CO2_CEN = 4.58  # forcing from log term in CO2
    EPS_CEN = 1.58  # pattern effect
    ALPHA_R1_CEN = 1.1  # region 1 pattern scaling parameter (global T)
    ALPHA_R2_CEN = 2.0  # region 2 pattern scaling parameter (global T)
    BETA_R1_CEN = 0.7  # region 1 pattern scaling parameter (geoengeineering)
    BETA_R2_CEN = 0.7  # region 2 pattern scaling parameter (geoengineering)

    # true parameters used to make observations
    L_TR = 1.06  # sensitivity
    G_TR = 0.7  # layer transfer coefficient
    C1_TR = 8  # heat capacity of surface layer
    C2_TR = 100  # heat capacity of ocean layer
    F1_CO2_TR = 4.58  # forcing from log term in CO2
    EPS_TR = 1.58  # pattern effect
    ALPHA_R1_TR = 1.1  # region 1 pattern scaling parameter (global T)
    ALPHA_R2_TR = 2.0  # region 2 pattern scaling parameter (global T)
    BETA_R1_TR = 0.7  # region 1 pattern scaling parameter (geoengeineering)
    BETA_R2_TR = 0.4239  # region 2 pattern scaling parameter (geoengineering)
    
    # other parameters of interest that are either diagnostic (i.e., are functions of other things)
    # or treated as known
    ANGLE_TR = np.arccos((ALPHA_R1_TR * BETA_R1_TR + ALPHA_R2_TR * BETA_R2_TR)
                         / np.sqrt((ALPHA_R1_TR**2 + ALPHA_R2_TR**2)
                                   * (BETA_R1_TR**2 + BETA_R2_TR**2))) * 180 / np.pi  # angle between impact vectors
    F_EFF_GEO_TR = 0.09  # W / m2 per TgS / yr of geoengineering (forcing efficacy)
    ECS_TR = F1_CO2_TR * np.log(2) / L_TR  # equilibrium climate sensitivity
    INT_VAR_STD = 0.27  # internal variability standard deviation from Proistosescu and Huybers, Sci Adv, 2017

    """WARM START MODULE.
    """
    print("Warm starting model to get initial conditions for temperature, ocean heat content, and regional temperature...")
    # "warm start" model to get the central estimates of the initial conditions
    # for temperature in our start year
    e_ws = EmissionsBaseline(SCENARIO, 1850, TMIN)

    # make vector of parameters for warm start
    # NOTE: the first five entries are the initial conditions of global mean temperature,
    # global mean ocean temperature, ocean heat content, and two regional temperatures
    # that we use pattern scaling to find: Tri = alpha_ri * T1 - beta_ri * geo_level.
    # since geo_level = 0 in the warm start, these are all zero (since T1 = 0 at 1850).
    theta_ws = np.hstack([np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                         L_TR, G_TR, EPS_TR, C1_TR, C2_TR, F1_CO2_TR,
                         ALPHA_R1_TR, ALPHA_R2_TR, BETA_R1_TR, BETA_R2_TR]),
                         np.zeros_like(e_ws.conc['CO2'])])

    # make "warm start" to get true initial conditions
    data_ws, _ = get_nonlin_path(e_ws, theta_ws, 1850, TMIN, DT)

    # true initial conditions
    T1_TR = data_ws[0, -1] # surface temperature at TMIN
    T1_CEN = data_ws[0, -1] # central estimate is the truth

    T2_TR = data_ws[1, -1]  # ocean layer temperature at TMIN
    T2_CEN = data_ws[1, -1]  # central estimate is the truth

    Q_TR = C1_TR * T1_TR + C2_TR * T2_TR  # OHC true value
    Q_CEN = C1_CEN * T1_CEN + C2_CEN * T2_CEN  # OHC central estimate

    T_R1_TR = ALPHA_R1_TR * T1_TR  # temperature in region 1 true value
    T_R1_CEN = ALPHA_R1_CEN * T1_CEN  # central estimate, temperature in region 1

    T_R2_TR = ALPHA_R2_TR * T1_TR  # temperature in region 2 true value
    T_R2_CEN = ALPHA_R2_CEN * T1_CEN  # central estimate, temperature in region 2
    
    print("Warm start complete!")

    print(T1_TR, T2_TR, Q_TR, T_R1_TR, T_R2_TR)


    print("==================================================================")
    print("Simulation attributes:")
    print("------------------------------------------------------------------")
    print("Socio-economic pathway: {}".format(SCENARIO))
    print("Temperature offset by SAI per decade: {} deg C".format(DEG_PER_DEC))
    print("The SAI ramp-up occurs over {} years".format(N_YEARS_RAMP))
    print("The initial time is: {}".format(TMIN))
    print("Temperature is forced with AR({}) noise.".format(AR_P))
    print("ECS = {}.".format(ECS_TR))
    print("The angle is {} degrees between temperature and geoengineering.".format(ANGLE_TR))
    print("There are {} assimilation windows, starting in {} and ending in 2100 (this implies adding one window adds {} years of observations).".format(N_windows - 1, TMIN, (2100 - TMIN)/len(tmax_assims)))
    print("The 4DVAR ensemble has {} members.".format(N_ENS))
    print("==================================================================")

    # dictionary to make datatree out of later
    datatree_dict = {}

    for TMAX in tmax_assims:
        print("--------------------------------------")
        print("WE ARE ON TMAX = {}".format(TMAX))
        print("--------------------------------------")

        # set seed so we get same draws for each assimilation window
        np.random.seed(1000)

        # make emissions baseline
        e = EmissionsBaseline(SCENARIO, TMIN, TMAX,
                              geo=True, DEG_PER_DEC=DEG_PER_DEC,
                              LAMBDA=L_CEN, F_EFF_GEO=F_EFF_GEO_TR,
                              T_START=TMIN, T_END=TMIN + N_YEARS_RAMP)

        # make model errors and their covariance matrix
        mod_errors, mod_error_covar = gen_noise_ts(AR_P, len(e.conc['CO2']),
                                                   INT_VAR_STD,
                                                   CORR_COEFFS=[0.2])

        # true vector of controls: initial conditions, parameters, and model
        # errors
        theta_tr = np.hstack([np.array([T1_TR, T2_TR,
                             Q_TR, T_R1_TR, T_R2_TR,
                             L_TR, G_TR, EPS_TR, C1_TR, C2_TR, F1_CO2_TR,
                             ALPHA_R1_TR, ALPHA_R2_TR, BETA_R1_TR, BETA_R2_TR]), mod_errors])

        # central value of priors on each parameter
        theta_prior_cent = np.hstack([np.array([T1_CEN, T2_CEN,
                                     Q_CEN, T_R1_CEN, T_R2_CEN,
                                     L_CEN, G_CEN, EPS_CEN, C1_CEN, C2_CEN, F1_CO2_CEN,
                                     ALPHA_R1_CEN, ALPHA_R2_CEN, BETA_R1_CEN, BETA_R2_CEN]), mod_errors])

        # make true data path over this time window
        data_tr_p, times = get_nonlin_path(e, theta_tr, TMIN, TMAX, DT)

        # note stds of priors and obs
        OBS_T1_STD = 1.0  # observation noise in measuring T1/T2
        OBS_Q_STD = 1.0  # observation noise in measuring ocean heat content
        OBS_T_R1_STD = 1.0  # observation noise in measuring regional temperature in R1
        OBS_T_R2_STD = 1.0  # observation noise in measuring regional temperature in R2

        T_IC_STD = 0.2  # initial condition std for t1 and t2 (roughly the size of internal variability)
        EPS_STD = 0.128  # pattern effect standard deviation (cummins, 2020)
        F1_STD = 0.519   # f1_co2 std, from zelinka et al. (2020)
        PRIOR_STD_FACTOR = 0.3  # implies X% std for prior for other less constrained parameters

        Q_STD = C1_TR * T_IC_STD + C2_TR * T_IC_STD  # std for OHC
        T_R1_STD = ALPHA_R1_TR * T_IC_STD  # std for temp in r1
        T_R2_STD = ALPHA_R2_TR * T_IC_STD  # std for temp in r2
        
        # make prior stds vector
        prior_stds = np.hstack([np.array([T_IC_STD, T_IC_STD,
                                          Q_STD, T_R1_STD, T_R2_STD]),
                                np.abs(theta_prior_cent[5:7]) * PRIOR_STD_FACTOR,
                                EPS_STD,
                                np.abs(theta_prior_cent[8:10]) * PRIOR_STD_FACTOR,
                                F1_STD,
                                np.abs(theta_prior_cent[11:15]) * PRIOR_STD_FACTOR,
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

        inv_covar_T_R1_obs = get_covar_white(np.array([OBS_T_R1_STD] *
                                                   len(times)), len(times),
                                            inv=True)

        inv_covar_T_R2_obs = get_covar_white(np.array([OBS_T_R2_STD] *
                                                   len(times)), len(times),
                                            inv=True)

        # make observations from true data
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
            _ = get_adj_id_check(e, theta_tr, TMIN, TMAX, DT,
                                 SAVE_RESULTS=True)

            # Check 2: Gradient of Cost Function
            ALPHA_MIN = 1e-16
            ALPHA_MAX = 1.0

            # run check function
            _ = get_cost_grad_check(control=theta_tr * 1.1,
                                    cost_args=[theta_tr,
                                               inv_covar_prior,
                                               inv_covar_T1_obs,
                                               inv_covar_Q_obs,
                                               inv_covar_T_R1_obs,
                                               inv_covar_T_R2_obs,
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
                                           inv_covar_Q_obs, inv_covar_T_R1_obs,
                                           inv_covar_T_R2_obs, obs, times)
                            for theta_p in theta_prior]

        # solve the assimilation using dask
        t0 = time.time()

        # do dask evaluation of runner
        print("Solving 4DVAR using DASK...")
        part_size = int(N_ENS / 10)  # twenty members per thread in Dask

        # make dask bag for evaluation
        bag_ens = db.from_sequence(ensemble_members,
                                   partition_size=part_size).map(runner_4dvar)

        # map and compute
        opt_ensmems = bag_ens.compute()
        t1 = time.time()
        print(t1 - t0)
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
        names = np.hstack([['T1', 'T2', 'Q', 'T_R1', 'T_R2', 'L', 'G', 'EPS', 'C1', 'C2', 'F1_CO2',
                            'ALPHA_R1', 'ALPHA_R2', 'BETA_R1', 'BETA_R2'],
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
                                'obs_var': (['obs_var'], ['T1', 'Q', 'T_R1', 'T_R2'])},
                        attrs={'TMIN': TMIN,
                               'TMAX': TMAX,
                               'DT': DT,
                               'max_iter': max_iter,
                               'tol': tol,
                               'run_time': t1 - t0,
                               'ECS': ECS_TR,
                               'ANGLE': ANGLE_TR,
                               'internal_variability_std': INT_VAR_STD})

        datatree_dict[str(TMAX)] = ds

    dt = DataTree.from_dict(datatree_dict, 'TMAX')

    if SAVE_OUTPUT:
        # get current directory and save
        sim_type = 'pco2geowc'
        path = DATA_DIR + '/output/' + sim_type\
            + '/margobs_ws_angle10_'\
            + SCENARIO + "_"\
            + sim_type + "_"\
            + "TMIN" + str(TMIN) + "_"\
            + "AR" + str(AR_P) + "_"\
            + "DEGpDEC" + str(DEG_PER_DEC) + "_"\
            + "NYRSRAMP" + str(N_YEARS_RAMP) + "_"\
            + "Nwinds" + str(N_windows) + "_"\
            + "Nens" + str(N_ENS) + ".nc"
        dt.to_netcdf(filepath=path, mode='w', format='NETCDF4',
                     engine='netcdf4')

        print("\nOutput successfully saved to:\n{}\n".format(path))

    else:
        print(dt)
