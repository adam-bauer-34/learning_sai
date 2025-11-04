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
from src.co2sul.dynamics import get_nonlin_path
from src.co2sul.cost import cost, grad
from src.co2sul.obs import get_obs_from_dynamics
from src.stats.covar import get_inv_covar_white

# parse command line stuff
SCENARIO = sys.argv[1]
SAVE_OUTPUT = int(sys.argv[2])

# ----------------------------------------
# Initialize the problem
# ----------------------------------------
# times
TMIN = 1850
TMAX = 2000
DT = 1.0

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
T1_tr = 0.05  # surface temperatureg at TMIN
T2_tr = 0.02  # ocean layer temperature at TMIN

# true vector of controls
theta_tr = np.array([T1_tr, T2_tr,
                     C1_TR * T1_tr + C2_TR * T2_tr,
                     L_TR, G_TR, C1_TR, C2_TR, F1_CO2_TR, F3_CO2_TR,
                     A_SO2_TR, B_SO2_TR, C_SO2_TR])

data_tr_p, times = get_nonlin_path(e, theta_tr, TMIN, TMAX, DT)

# errors (assumed constant for now)
VAR_T1_OBS = 0.1  # observation error
VAR_Q_OBS = C1_TR * VAR_T1_OBS + C2_TR * VAR_T1_OBS  # heat content obs error

# note stds of priors and data
OBS_T1_STD = 0.1  # 0.1 K variance in white noise driving temperatures
PRIOR_STD_FACTOR = 0.1  # implies 10% std for prior
prior_stds = np.hstack([np.array([0.1, 0.1, 0.1 * (C1_TR + C2_TR)]),
                       np.abs(theta_tr[3:]) * PRIOR_STD_FACTOR])


N = 100

t0_range = np.linspace(T1_tr - OBS_T1_STD * 3, T1_tr + OBS_T1_STD * 3, N)
t1_range = np.linspace(T2_tr - OBS_T1_STD * 3, T2_tr + OBS_T1_STD * 3, N)

lam_range = np.linspace(L_TR - PRIOR_STD_FACTOR * L_TR * 3, L_TR +
                        PRIOR_STD_FACTOR * L_TR * 3, N)
f1_range = np.linspace(F1_CO2_TR - PRIOR_STD_FACTOR * F1_CO2_TR * 3, F1_CO2_TR
                       + PRIOR_STD_FACTOR * F1_CO2_TR * 3, N)

a_range = np.linspace(A_SO2_TR - PRIOR_STD_FACTOR * A_SO2_TR * 3, A_SO2_TR +
                      PRIOR_STD_FACTOR * A_SO2_TR * 3, N)
b_range = np.linspace(B_SO2_TR - PRIOR_STD_FACTOR * B_SO2_TR * 3, B_SO2_TR +
                      PRIOR_STD_FACTOR * B_SO2_TR * 3, N)

t_ranges = [[1850, 2000],
            [1850, 2050],
            [1850, 2100]]

costs_Ts = np.zeros((len(t_ranges), len(t0_range), len(t1_range)))
costs_LF = np.zeros_like(costs_Ts)
costs_aero = np.zeros_like(costs_Ts)

for t in range(len(t_ranges)):
    # extract t range
    TMIN, TMAX = t_ranges[t]
    TMIN = int(TMIN)
    TMAX = int(TMAX)
    DT = 1.0

    # emissions baseline
    e = EmissionsBaseline(SCENARIO, TMIN, TMAX)

    # gen obs
    data_tr_p, times = get_nonlin_path(e, theta_tr, TMIN, TMAX, DT)

    # make observations
    obs = get_obs_from_dynamics(data_tr_p)

    # make inverse covariance matrices for white noise
    inv_covar_prior = get_inv_covar_white(prior_stds,
                                          len(prior_stds))

    inv_covar_T1_obs = get_inv_covar_white(np.array([OBS_T1_STD] * len(times)),
                                           len(times))

    inv_covar_Q_obs = get_inv_covar_white(np.array(
        [(C1_TR + C2_TR) * OBS_T1_STD] * len(times)), len(times))

    tmp_theta_t = theta_tr.copy()
    for t0 in range(len(t0_range)):
        tmp_theta_t[0] = t0_range[t0]

        for t1 in range(len(t1_range)):
            tmp_theta_t[1] = t1_range[t1]

            # generate initial cost function for storage
            J0 = cost(tmp_theta_t,
                      args=[np.hstack([theta_tr]),
                            inv_covar_prior,
                            inv_covar_T1_obs,
                            inv_covar_Q_obs,
                            obs,
                            e,
                            TMIN,
                            TMAX,
                            DT,
                            False, 0])

            costs_Ts[t, t0, t1] = J0

    tmp_theta_lf = theta_tr.copy()
    for l in range(len(lam_range)):
        tmp_theta_lf[3] = lam_range[l]

        for f in range(len(f1_range)):
            tmp_theta_lf[7] = f1_range[f]

            # generate initial cost function for storage
            J0 = cost(tmp_theta_lf,
                      args=[np.hstack([theta_tr]),
                            inv_covar_prior,
                            inv_covar_T1_obs,
                            inv_covar_Q_obs,
                            obs,
                            e,
                            TMIN,
                            TMAX,
                            DT,
                            False,
                            0])

            costs_LF[t, l, f] = J0

    tmp_theta_aero = theta_tr.copy()
    for a in range(len(a_range)):
        tmp_theta_aero[-3] = a_range[a]

        for b in range(len(b_range)):
            tmp_theta_aero[-2] = b_range[b]

            # generate initial cost function for storage
            J0 = cost(tmp_theta_aero,
                      args=[np.hstack([theta_tr]),
                            inv_covar_prior,
                            inv_covar_T1_obs,
                            inv_covar_Q_obs,
                            obs,
                            e,
                            TMIN,
                            TMAX,
                            DT,
                            False, 0])

            costs_aero[t, a, b] = J0

# -----------------------------------------------
# If desired, save the output 
# -----------------------------------------------
names = ['T1', 'T2', 'Q', 'L', 'G', 'C1', 'C2', 'F1_CO2', 'F3_CO2',
         'A_SO2', 'B_SO2', 'C_SO2']
ds = xr.Dataset(data_vars={'cost_ts': (['t_range', 't1', 't2'], costs_Ts),
                           'cost_lf': (['t_range', 'l', 'f1'], costs_LF),
                           'cost_aero': (['t_range', 'a', 'b'], costs_aero),
                           'controls_truth': (['vari'], theta_tr),
                           'tis': (['t_range'], [1850] * 3),
                           'tfs': (['t_range'], [2000, 2050, 2100])},
                coords={'vari': (['vari'], names),
                        't1': (['t1'], t0_range),
                        't2': (['t2'], t1_range),
                        'l': (['l'], lam_range),
                        'f1': (['f1'], f1_range),
                        'a': (['a'], a_range),
                        'b': (['b'], b_range),
                        't_range': (['t_range'], ['1850-2000', '1850-2050',
                                                  '1850-2100'])
                        })

if SAVE_OUTPUT:
    # get current directory and save 
    cwd = os.getcwd()
    path = cwd + "/data/output/cost_struc_co2sul_perf.nc"
    ds.to_netcdf(path=path, mode='w', format='NETCDF4', engine='netcdf4')

    print("\nOutput successfully saved to:\n{}\n".format(path))

else:
    print(ds.run_time)
