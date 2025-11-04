"""Cost function.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.15.2024
"""

import numpy as np

from .dynamics import get_nonlin_path
from .adjoint import get_adj_path
from .obs import get_obs_from_dynamics


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
            - noise: bool
                is there a noise model for our observations?
            - noise_means: (N_obs,) list
                average of observation noise
    """

    # unpack lists
    control = np.array(control)
    x_f, inv_covar_prior, inv_covar_T1, inv_covar_Q, obs, e, T_MIN, T_MAX, DT, noise, noise_means = args

    # unpack data
    T1_d, Q_d = obs

    # initial piece that only relies on the priors
    init_piece = 0.5 * (control - x_f).T @ inv_covar_prior @ (control - x_f)

    # get nonlinear path based on control
    paths, _ = get_nonlin_path(e, control, T_MIN, T_MAX, DT)

    # get obs based on paths
    T1_obs_p, Q_obs_p = get_obs_from_dynamics(paths, noise,
                                              [noise_means,
                                               [np.linalg.inv(inv_covar_T1),
                                                np.linalg.inv(inv_covar_Q)]])

    # data piece
    data_piece = 0.5 * ((T1_obs_p - T1_d).T @ inv_covar_T1 @ (T1_obs_p - T1_d)
                        + (Q_obs_p - Q_d).T @ inv_covar_Q @ (Q_obs_p - Q_d))

    cost = init_piece + data_piece
    return cost


def grad(control, args):
    # unpack lists
    control = np.array(control)
    x_f, inv_covar_prior, inv_covar_T1, inv_covar_Q, obs, e, T_MIN, T_MAX, DT, noise, noise_means = args

    # initial piece that only relies on the priors
    init_piece = inv_covar_prior @ (control - x_f)

    # nonlin path
    nl_path, _ = get_nonlin_path(e, control, T_MIN, T_MAX, DT)

    # get adjoint path based on control
    adj_p = get_adj_path(e,
                         theta=control,
                         TMIN=T_MIN,
                         TMAX=T_MAX,
                         DT=DT,
                         nl_path=nl_path,
                         covars=[inv_covar_T1, inv_covar_Q],
                         obs=obs,
                         id_check=False)

    # initial values of adjoint are desired gradients of the observations-bit
    # of the cost function
    obs_piece = adj_p[:, 0]

    # gradient is the sum of init_piece and obs_piece
    grad = init_piece + obs_piece

    return grad
