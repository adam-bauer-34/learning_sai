"""Cost function.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.15.2024
"""

import warnings

import numpy as np

from .dynamics import get_nonlin_path
from .adjoint import get_adj_path
from .obs import get_obs_from_dynamics

# filter out RuntimeWarning because they clog the log files
# and don't impact the results
warnings.filterwarnings("ignore", category=RuntimeWarning)


def cost(control, args):
    """Cost function for 4DVAR.

    Parameters
    ----------
    control: (11,) list
        control variables, all of the parameters and the initial conditions

    args: list
        arguments for finding the cost function:
            - x_f: (11,) list
                first guess for control variables
            - inv_covar_prior: (11, 11) matrix
                inverse covariance matrix between priors
            - inv_covar_T1: (T, T) matrix
                inverse covariance matrix between temperature observations
            - inv_covar_Q: (T, T) matrix
                inverse covariance matrix between ocean heat content
                observations
            - data: (2, T) vector
                observations for surface temperature and ocean heat content
            - T_MIN: int
                year to start simulations
            - T_MAX: int
                year to stop simulations
            - DT: float
                time discretization
    """

    # unpack lists
    control = np.array(control)
    (
        x_f,
        inv_covar_prior,
        inv_covar_T1,
        inv_covar_Q,
        inv_covar_T_R1,
        inv_covar_T_R2,
        inv_covar_T_R3,
        obs,
        e,
        T_MIN,
        T_MAX,
        DT,
    ) = args

    # unpack data
    T1_d, Q_d, T_R1_d, T_R2_d, T_R3_d = obs

    # initial piece that only relies on the priors
    init_piece = 0.5 * (control - x_f).T @ inv_covar_prior @ (control - x_f)

    # get nonlinear path based on control
    paths, _ = get_nonlin_path(e, control, T_MIN, T_MAX, DT)

    # get obs based on paths
    T1_obs_p, Q_obs_p, T_R1_obs_p, T_R2_obs_p, T_R3_obs_p = get_obs_from_dynamics(paths)

    # data piece
    data_piece = 0.5 * (
        (T1_obs_p - T1_d).T @ inv_covar_T1 @ (T1_obs_p - T1_d)
        + (Q_obs_p - Q_d).T @ inv_covar_Q @ (Q_obs_p - Q_d)
        + (T_R1_obs_p - T_R1_d).T @ inv_covar_T_R1 @ (T_R1_obs_p - T_R1_d)
        + (T_R2_obs_p - T_R2_d).T @ inv_covar_T_R2 @ (T_R2_obs_p - T_R2_d)
        + (T_R3_obs_p - T_R3_d).T @ inv_covar_T_R3 @ (T_R3_obs_p - T_R3_d)
    )

    cost = init_piece + data_piece
    return cost


def grad(control, args):
    # unpack lists
    control = np.array(control)
    (
        x_f,
        inv_covar_prior,
        inv_covar_T1,
        inv_covar_Q,
        inv_covar_T_R1,
        inv_covar_T_R2,
        inv_covar_T_R3,
        obs,
        e,
        T_MIN,
        T_MAX,
        DT,
    ) = args

    # initial piece that only relies on the priors
    init_piece = inv_covar_prior @ (control - x_f)

    # nonlin path
    nl_path, _ = get_nonlin_path(e, control, T_MIN, T_MAX, DT)

    # get adjoint path based on control
    adj_p = get_adj_path(
        e,
        theta=control,
        TMIN=T_MIN,
        TMAX=T_MAX,
        DT=DT,
        nl_path=nl_path,
        covars=[
            inv_covar_T1,
            inv_covar_Q,
            inv_covar_T_R1,
            inv_covar_T_R2,
            inv_covar_T_R3,
        ],
        obs=obs,
        id_check=False,
    )

    # initial values of adjoint are desired gradients of the observations-bit
    # of the cost function
    obs_piece = adj_p[:, 0].copy()

    # chain rule through the initial condition. get_nonlin_path does not set the
    # initial state to the control vector verbatim -- it sets
    # paths[0, 0] = T10 + q_AT[0], paths[3, 0] = TR10 + q_R1[0], and so on. so
    # each block's q[0] reaches the cost function by two routes, and dJ/dq[0]
    # picks up the corresponding dJ/dx0 on top of its own stationary row. no TLM
    # step maps into time index 0 (the TLM built at local time t writes column
    # q[t + 1], so t + 1 >= 1 always), which is why these terms are missing from
    # the adjoint and have to be added here.
    N_times = nl_path.shape[1]
    for blk, ic in enumerate((0, 3, 4, 5)):
        obs_piece[18 + blk * N_times] += obs_piece[ic]

    # gradient is the sum of init_piece and obs_piece
    grad = init_piece + obs_piece

    return grad
