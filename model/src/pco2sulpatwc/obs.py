"""Observation operator related functions.
"""

import numpy as np


def get_obs_from_dynamics(paths, noise=False, noise_params=None):
    """Observation operator.
    """

    # cast paths into observed quantities
    T1 = paths[0]
    Q = paths[2]

    obs = np.array([T1, Q])

    if noise:
        # apply noise to observations
        means, covs = noise_params
        T1_noise = np.random.multivariate_normal([means[0]] * len(T1),
                                                 covs[0])
        Q_noise = np.random.multivariate_normal([means[1]] * len(Q),
                                                covs[1])

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
    jac = np.zeros((2, np.shape(nl_path)[0]))

    # just manually fill in the nonzero bits
    # for surface temperature, there is only one component, and for ocean heat
    # capacity, there is only one component now
    jac[0, 0] = 1.0
    jac[1, 2] = 1.0

    # return the jacobian
    return jac
