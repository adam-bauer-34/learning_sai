"""Observation operator related functions."""

import numpy as np

from var_assim.config import REG_NOISE_SEED


def get_obs_from_dynamics(paths, noise=False, noise_params=None):
    """Observation operator."""

    # cast paths into observed quantities
    T1 = paths[0]
    Q = paths[2]
    T_R1 = paths[3]
    T_R2 = paths[4]

    obs = np.array([T1, Q, T_R1, T_R2])

    if noise:
        # apply noise to observations
        rng = np.random.default_rng(seed=REG_NOISE_SEED)
        means, covs = noise_params
        T1_noise = rng.multivariate_normal([means[0]] * len(T1), covs[0])
        Q_noise = rng.multivariate_normal([means[1]] * len(Q), covs[1])

        obs[0] += T1_noise
        obs[1] += Q_noise

    return obs


def get_obs_jac(nl_path, t_ind):
    """Get the Jacobian of the observation operator

    Parameters
    ----------
    nl_path: (N_states, N_times)
        path of nonlinear model evolution based on control variable ICs

    t_ind: int
        time index we're evaluating the Jacobian at

    Returns
    -------
    jac: (N_obs, N_states)
        Jacobian of observation operator
        Since there are N_obs observables, we are taking N_states derivatives
        of N_obs functions, resulting in an (N_obs, N_states) matrix
    """

    # initialize empty jacobian
    jac = np.zeros((4, np.shape(nl_path)[0]))

    # just manually fill in the nonzero bits
    # for surface temperature, there is only one component, and for ocean heat
    # capacity, there is only one component now
    jac[0, 0] = 1.0  # global temp
    jac[1, 2] = 1.0  # ohc
    jac[2, 3] = 1.0  # regional temp 1
    jac[3, 4] = 1.0  # regional temp 2

    # return the jacobian
    return jac
