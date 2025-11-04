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

from src.emis import EmissionsBaseline
from src.co2sul.checks import get_tlm_check, get_adj_id_check, get_cost_grad_check
from src.co2sul.dynamics import get_nonlin_path
from src.co2sul.cost import cost, grad
from src.co2sul.obs import get_obs_from_dynamics
from src.stats.covar import get_inv_covar_white
from src.stats.draws import get_prior_draws
from scipy.optimize import minimize

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
VAR_T1_OBS = 0.1  # observation error
VAR_Q_OBS = C1_TR * VAR_T1_OBS + C2_TR * VAR_T1_OBS  # heat content obs error

data_tr_p, times = get_nonlin_path(e, theta_tr, TMIN, TMAX, DT)

# note stds of priors and data
OBS_T1_STD = 0.1  # 0.1 K variance in white noise driving temperatures
PRIOR_STD_FACTOR = 0.1  # implies 10% std for prior
prior_stds = np.hstack([np.array([0.1, 0.1, 0.1 * (C1_TR + C2_TR)]),
                       np.abs(theta_tr[3:]) * PRIOR_STD_FACTOR])

# make inverse covariance matrices for white noise
inv_covar_prior = get_inv_covar_white(prior_stds,
                                      len(prior_stds))

inv_covar_T1_obs = get_inv_covar_white(np.array([OBS_T1_STD] * len(times)),
                                       len(times))

inv_covar_Q_obs = get_inv_covar_white(np.array(
    [(C1_TR + C2_TR) * OBS_T1_STD] * len(times)), len(times))

# make observations
obs = get_obs_from_dynamics(data_tr_p)

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
N_ENS = 2  # size of ensemble
tol = 0.005  # tolerance for convergence in 4DVAR

# give first guess at initial conditions
theta_prior = get_prior_draws(theta_tr, np.linalg.inv(inv_covar_prior),
                              N_ENS)

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

# if you want print, turn on
PRINT_STUFF = False

# ----------------------------------------
# If desired, carry out optimization
# ----------------------------------------
if SOLVE_ASSIM:
    t0 = time.time()
    for n in range(N_ENS):
        print("------------------------------------------------------------")
        print("WE ARE ON ENSEMBLE MEMBER {}.".format(n))
        print("------------------------------------------------------------")

        # set prior guess and store
        theta_p = theta_prior[n]
        controls_hist[n, :, 0] = theta_p

        flag = -1  # initialize flag for status of optimization
        flags[n] = flag  # will be -1 if exited optimally, other vals if else

        # difference between true value and optimal value; before loop, just
        # set to be bigger than epsilon
        l2 = tol + 1

        iter_ = 1  # count the iterations

        # generate time series for first guess
        prior_p, times = get_nonlin_path(e, theta_p, TMIN, TMAX, DT)

        # store in history
        data_hist[n, :, 0] = prior_p

        # generate initial cost function for storage
        J0 = cost(theta_p,
                  args=[np.hstack([theta_tr]),
                        inv_covar_prior,
                        inv_covar_T1_obs,
                        inv_covar_Q_obs,
                        obs,
                        e,
                        TMIN,
                        TMAX,
                        DT])

        # store initial cost value
        cost_hist[n, 0] = J0

        # generate initial L2 norm and store
        l2s_hist[n, 0] = np.sqrt(np.sum((np.array(theta_p) - theta_tr)**2))

        while l2 > tol:
            # print("\nWe are on iteration {}".format(iter_))

            # optimize cost function
            sol = minimize(cost, x0=theta_p,
                           args=[np.hstack([theta_p]),
                                 inv_covar_prior,
                                 inv_covar_T1_obs,
                                 inv_covar_Q_obs,
                                 obs,
                                 e,
                                 TMIN,
                                 TMAX,
                                 DT],
                           bounds=[(0, np.inf), (0, np.inf),
                                   (0, np.inf), (0, np.inf),
                                   (0, np.inf), (0, np.inf),
                                   (0, np.inf), (0, np.inf),
                                   (0, np.inf), (-np.inf, np.inf),
                                   (-np.inf, np.inf), (0, np.inf)],
                           method='SLSQP',
                           jac=grad)
 
            # set new x_0 value
            new_theta = sol.x

            # store new cost value
            cost_hist[n, iter_] = sol.fun

            # compute L2 norm between truth and optimal value and store
            l2 = np.sqrt(np.sum((new_theta - theta_tr)**2))
            l2s_hist[n, iter_] = l2

            # print new estimate and cost function and L2
            if PRINT_STUFF:
                print("Updated x_0 value: {}.".format(new_theta))
                print("Cost function value: {}.".format(sol.fun))
                print("The L2 norm is: {}".format(l2))

            # store trajectory
            new_p, times = get_nonlin_path(e, new_theta, TMIN, TMAX, DT)
            data_hist[n, :, iter_] = new_p

            # if diff > tol, set the first guess as the optimal solution and
            # try again
            theta_p = new_theta
            controls_hist[n, :, iter_] = theta_p

            # index the iteration counter
            iter_ += 1

            if iter_ > max_iter:
                # we've exceeded maximum iterations, and we end the 4DVAR
                # process
                flags[n] = 1
                break

        if PRINT_STUFF:
            if flag == 1:
                print("Max iterations reached during optimization.")
                print("The final value of x_0 is: {}.".format(new_theta))
                print("The final value of the cost is: {}.".format(sol.fun))

            else:
                print("\n4DVAR procedure exited optimally.")
                print("The final value of x_0 is: {}.".format(new_theta))
                print("The final value of the cost is: {}.".format(sol.fun))

        costs[n] = sol.fun
        data[n] = new_p
        l2s[n] = l2
        controls[n] = new_theta

t1 = time.time()

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
    path = cwd + "/data/output/co2sul_wind" + str(TMIN) + "_" + str(TMAX)\
        + "_N" + str(N_ENS) + "_perf.nc"
    ds.to_netcdf(path=path, mode='w', format='NETCDF4', engine='netcdf4')

    print("\nOutput successfully saved to:\n{}\n".format(path))

else:
    print(ds.run_time)
