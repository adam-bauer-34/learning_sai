"""Two layer model 4DVAR with internal variability.

Adam Michael Bauer
University of Illinois Urbana Champaign
8.23.2024

To run:
    python main_margobs_pco2sulwc.py [scenario] [P] [L or F] [SIGMA]
        [N_windows] [N_ENS] [SAVE_OUTPUT]
"""

import sys
import time
import warnings 

# filter out runtime warnings which clog log files
# (they are natural in the scipy.minimize call)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import xarray as xr

from var_assim.dask_setup import start_dask
from var_assim.emis import EmissionsBaseline 
from var_assim.tlm_adj_checks import *
from var_assim.model_errors import gen_noise_ts
from var_assim.postprocessing import process_simulation_window, make_master_datatree
from var_assim.stats.covar import get_covar_white
from var_assim.stats.draws import get_prior_draws
from var_assim.warm_start import warm_start_simulation

from var_assim.models.pco2geowc.dynamics import get_nonlin_path
from var_assim.models.pco2geowc.obs import get_obs_from_dynamics
from var_assim.models.pco2geowc.parallelization import EnsembleMember, runner_4dvar


def run_var_assim_experiment():
    pass

def run_var_assim_experiment_wip(logger, args, Prior, Truth, Noise, Windowing):
    # start dask
    c = start_dask(logger)
    logger.info(c)

    # raise ECS warning
    if ECS_TR != 3.0:
        print("WARNING: Chaning ECS changes the forcing sensitivity to CO2 concentrations, NOT the feedback \lambda, to keep the prior on the SAI angle consistent between simulations.")

    # binary variables that are pre-set
    CHECK_TLM = False  # check the tangent linear model?
    CHECK_ADJ = False  # check the adjoint model and the cost function gradient?
    MANUAL_WINDOWING = True  # set assimilation windows manually?

    # ----------------------------------------
    # Initialize the problem
    # ----------------------------------------
    # set the time discretization
    DT = 1.0

    # make set of assimilation windows
    if not MANUAL_WINDOWING:
        tmax_assims = np.linspace(TMIN, 2100, N_windows, dtype=int)[1:]
    
    else:
        fine = np.arange(TMIN, 2050, 2)  # fine grained early on
        tmax_assims = np.hstack([fine, [2075, 2100]])[1:]  # add two larger ones later, ignore TMIN

    # GLOBAL ENERGY BALANCE MODEL PARAMETERS
    # central values of priors on global parameters
    ECS_CEN = 3.0  # central value of equilibrium climate sensitivity
    G_CEN = 0.7  # layer transfer coefficient
    C1_CEN = 8  # heat capacity of surface layer
    C2_CEN = 100  # heat capacity of ocean layer
    F1_CO2_CEN = 4.58  # forcing from log term in CO2
    L_CEN = F1_CO2_CEN * np.log(2) / ECS_CEN  # central value of climate feedback
    EPS_CEN = 1.58  # pattern effect

    # true parameters used to make observations
    L_TR = L_CEN  # sensitivity
    G_TR = 0.7  # layer transfer coefficient
    C1_TR = 8  # heat capacity of surface layer
    C2_TR = 100  # heat capacity of ocean layer
    F1_CO2_TR = L_TR * ECS_TR / np.log(2)  # forcing from log term in CO2
    EPS_TR = 1.58  # pattern effect

    F_EFF_GEO_TR = 0.09  # W / m2 per TgS / yr of geoengineering (forcing efficacy)
    INT_VAR_STD = 0.27  # internal variability standard deviation from Proistosescu and Huybers, Sci Adv, 2017

    # REGIONAL PATTERN SCALING MODEL PARAMETERS
    # central value and standard deviations of regional variables
    df = pd.read_csv(DATA_DIR + '/input/regional_calibration_parameters.csv',
                     delimiter=',', header=0, index_col='THETA')
    
    # print(F1_CO2_CEN, F1_CO2_TR, ECS_TR, L_TR, L_CEN)

    # global temperature related parameters
    ALPHA_R1_CEN = df.ALPHA_R1_CEN[THETA]  # region 1 pattern scaling parameter (global T)
    ALPHA_R2_CEN = df.ALPHA_R2_CEN[THETA]  # region 2 pattern scaling parameter (global T)
    ALPHA_R1_STD = df.ALPHA_R1_STD[THETA]  # standard deviation of alpha 1 prior
    ALPHA_R2_STD = df.ALPHA_R2_STD[THETA]  # standard deviation of alpha 2 prior

    # geoengineering related parameters
    BETA_R1_CEN = df.BETA_R1_CEN[THETA]  # region 1 pattern scaling parameter (geoengeineering)
    BETA_R2_CEN = df.BETA_R2_CEN[THETA]  # region 2 pattern scaling parameter (geoengeineering)
    BETA_R1_STD = df.BETA_R1_STD[THETA]  # region 1 pattern scaling parameter (geoengeineering)
    BETA_R2_STD = df.BETA_R2_STD[THETA]  # region 2 pattern scaling parameter (geoengeineering)

    # true values used to make observations
    ALPHA_R1_TR = df.ALPHA_R1_TR[THETA]  # region 1 pattern scaling parameter (global T)
    ALPHA_R2_TR = df.ALPHA_R2_TR[THETA]  # region 2 pattern scaling parameter (global T)
    BETA_R1_TR = df.BETA_R1_TR[THETA]  # region 1 pattern scaling parameter (geoengeineering)
    BETA_R2_TR = df.BETA_R2_TR[THETA]  # region 2 pattern scaling parameter (geoengeineering)

    """WARM START MODULE.
    """
    logger.info("Starting warm start module")
    
    warm_start_simulation(logger, args, Truth, Prior, get_nonlin_path)

    print("==================================================================")
    print("Simulation attributes:")
    print("------------------------------------------------------------------")
    print("Socio-economic pathway: {}".format(SCENARIO))
    print("Temperature offset by SAI per decade: {} deg C".format(DEG_PER_DEC))
    print("The SAI ramp-up occurs over {} years".format(N_YEARS_RAMP))
    print("The initial time is: {}".format(TMIN))
    print("Temperature is forced with AR({}) noise.".format(AR_P))
    print("ECS = {}.".format(ECS_TR))
    print("The angle is {} degrees between temperature and geoengineering.".format(THETA))
    if not MANUAL_WINDOWING:
        print("There are {} (auto-generated) assimilation windows, starting in {} and ending in 2100 (this implies adding one window adds {} years of observations).".format(N_windows - 1, TMIN, (2100 - TMIN)/len(tmax_assims)))
    else:
        print("There are {} assimilation windows that were manually specified, which are {}.".format(len(tmax_assims), tmax_assims))
    print("The 4DVAR ensemble has {} members.".format(N_ENS))
    print("==================================================================")

    """ASSIMILATION MODULE
    """
    # dictionary to make datatree out of later
    results_dict = {}

    for TMAX in tmax_assims:
        print("--------------------------------------")
        print("WE ARE ON TMAX = {}".format(TMAX))
        print("--------------------------------------")

        # set seed so we get same draws for each assimilation window
        np.random.seed(1000)

        # make emissions baseline
        e = EmissionsBaseline(SCENARIO, TMIN, TMAX,
                              geo=True, DEG_PER_DEC=DEG_PER_DEC,
                              LAMBDA=L_CEN, GAMMA=G_CEN, EPSILON=EPS_CEN, F_EFF_GEO=F_EFF_GEO_TR,
                              T_START=TMIN, T_END=TMIN + N_YEARS_RAMP)

        # make model errors and their covariance matrix
        mod_errors, mod_error_covar = gen_noise_ts(AR_P, len(e.conc['CO2']),
                                                   INT_VAR_STD,
                                                   CORR_COEFFS=[0.2])

        # true vector of controls: initial conditions, parameters, and model
        # errors
        controls_tr = np.hstack([np.array([T1_TR, T2_TR,
                             Q_TR, T_R1_TR, T_R2_TR,
                             L_TR, G_TR, EPS_TR, C1_TR, C2_TR, F1_CO2_TR,
                             ALPHA_R1_TR, ALPHA_R2_TR, BETA_R1_TR, BETA_R2_TR]), mod_errors])
        
        # central value of priors on each parameter
        theta_prior_cent = np.hstack([np.array([T1_CEN, T2_CEN,
                                     Q_CEN, T_R1_CEN, T_R2_CEN,
                                     L_CEN, G_CEN, EPS_CEN, C1_CEN, C2_CEN, F1_CO2_CEN,
                                     ALPHA_R1_CEN, ALPHA_R2_CEN, BETA_R1_CEN, BETA_R2_CEN]),
                                     np.zeros_like(mod_errors)])

        # make true data path over this time window
        data_tr_p, times = get_nonlin_path(e, controls_tr, TMIN, TMAX, DT)

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
                                F1_STD, ALPHA_R1_STD, ALPHA_R2_STD, BETA_R1_STD, BETA_R2_STD,
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
            _ = get_tlm_check(e, controls_tr, TMIN, TMAX, DT, ALPHA_MIN,
                              ALPHA_MAX, SAVE_RESULTS=True)

        
        # -----------------------------------------------
        # If desired, check adjoint accuracy
        # -----------------------------------------------
        if CHECK_ADJ:
            # Check 1: Adjoint Identity

            # do first check
            _ = get_adj_id_check(e, controls_tr, TMIN, TMAX, DT,
                                 SAVE_RESULTS=True)

            # Check 2: Gradient of Cost Function
            ALPHA_MIN = 1e-16
            ALPHA_MAX = 1.0

            # run check function
            _ = get_cost_grad_check(control=controls_tr * 1.1,
                                    cost_args=[controls_tr,
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
        
        # Check on object sizes
        #print("emissions object is:")
        #print(sys.getsizeof(e))
        #print(asizeof.asizeof(e) / 1e6)

        # scatter emissions baseline class and true observations to each
        # dask worker 
        e_scat = c.scatter(e, broadcast=True)

        # make list of ensemble members
        ensemble_members = [EnsembleMember(theta_p,
                                           -1, tol, max_iter,
                                           TMIN, TMAX, DT, controls_tr,
                                           inv_covar_prior, inv_covar_T1_obs,
                                           inv_covar_Q_obs, inv_covar_T_R1_obs,
                                           inv_covar_T_R2_obs, obs, times)
                            for theta_p in theta_prior]

        m = ensemble_members[0]

        print("Python object overhead:", sys.getsizeof(m) / 1e6, "MB")

        total = 0
        for v in vars(m).values():
            if hasattr(v, "nbytes"):
                total += v.nbytes
            else:
                total += sys.getsizeof(v)

        print("Estimated total size:", total / 1e6, "MB")
        
        #for i, ee in enumerate(ensemble_members):
        #    print(i, asizeof.asizeof(ee) / 1e6, " MB")

        # solve the assimilation using dask
        t0 = time.time()

        # do dask evaluation of runner
        print("Solving 4DVAR using DASK...")

        # map and compute
        futures = [c.submit(runner_4dvar, m, e_scat)
                   for m in ensemble_members]
        
        # gather results
        opt_ensmems = c.gather(futures)

        t1 = time.time()

        RUNTIME = t1 - t0
        # print(t1 - t0)

        # process simulation output into 
        ds = process_simulation_window(logger, args, TMAX,
                                       opt_ensmems, obs, data_tr_p,
                                       controls_tr, opt_chars, RUNTIME)

        results_dict[str(TMAX)] = ds

    # synthesize datasets from each window into a datatree object
    make_master_datatree(logger, args, results_dict)
    