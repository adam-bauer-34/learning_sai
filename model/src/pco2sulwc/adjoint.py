"""Adjoint for solo parameter model.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.12.2024
"""

import numpy as np

from .dynamics import get_TLM_matrix, get_tlm_path
from .obs import get_obs_from_dynamics, get_obs_jac


def get_adj_path(e, theta, TMIN, TMAX, DT, nl_path, covars=None,
                 obs=None, id_check=True):
    """Adjoint path.
    """

    # initalize adjoint path
    adj_path = np.zeros_like(nl_path, dtype=float)

    if id_check:
        # initiate at the final point of the TLM path
        tlm_path = get_tlm_path(e, theta, TMIN, TMAX, DT, nl_path)

        # print(tlm_path)
        adj_path[:, -1] = tlm_path[:, -1]

        # loop backwards and integrate
        for i in range(TMAX-TMIN, 0, -1):
            # extract adjoint vector for this time point
            dAdj = adj_path[:, i]

            # generate TLM matrix
            TLM_matrix = get_TLM_matrix(e, i-1, nl_path, DT)

            # take transpose to get adjoint
            ADJ_matrix = TLM_matrix.T

            # dot into adjoint vector
            adj_path[:, i-1] = ADJ_matrix @ dAdj

    else:
        # in this case, we're using the adjoint to compute the gradient of a
        # cost function with respect to a set of control variables

        # compute observations based on model states
        my_obs = get_obs_from_dynamics(nl_path)

        # set final timepoint of adjoint equal to forcing at final timepoint
        adj_path[:, -1] = get_adj_forcing(nl_path, my_obs, obs, covars,
                                          TMAX-TMIN)

        for t in range(TMAX-TMIN, 0, -1):
            # compute forcing at this timestep
            forcing = get_adj_forcing(nl_path, my_obs, obs, covars,
                                      t-1)

            # extract adjoint vector for this time point
            dAdj = adj_path[:, t]

            # generate TLM matrix
            TLM_matrix = get_TLM_matrix(e, t-1, nl_path, DT)

            # take transpose to get ADJ
            ADJ_matrix = TLM_matrix.T

            # dot with adjoint vector plus forcing
            adj_path[:, t-1] = ADJ_matrix @ dAdj + forcing

    return adj_path


def get_adj_forcing(nl_path, my_obs, obs, covars, t_ind):
    """Get adjoint "forcing" at t=t_ind.

    Parameters
    ----------
    nl_path: (N_states, N_times)
        nonlinear path based on control variable ICs

    my_obs: (N_obs, N_times)
        nl_path mapped to observables through the observation operator

    obs: (N_obs, N_times)
        observations or "truth"

    covars: (N_obs, N_times, N_times)
        matrix of covariances for each observable

    t_ind: int
        the time index we are getting the forcing for

    Returns
    -------
    forcing: (N_states)
        the vector of forcing for each model state's adjoint equation
    """

    # set empty matrix equal to the number of states
    forcing = np.zeros(np.shape(nl_path)[0])

    # compute the observation operator jacobian at the time of interest
    # this is a (N_obs, N_states) matrix
    obs_op_jac = get_obs_jac(nl_path, t_ind)

    # compute the misfits 
    # this is a (N_obs) vector
    misfit = get_misfit_t(my_obs, obs, covars, t_ind)

    # the forcing is equal to misfit . jacobian
    # the result is an (N_states) vector
    forcing = misfit @ obs_op_jac
 
    return forcing


def get_misfit_t(my_obs, obs, covars, t_ind):
    """Get misfit vector at the t=t_ind

    Parameters
    ----------
    my_obs: (N_obs, N_times)
        nl_path mapped to observables through the observation operator

    obs: (N_obs, N_times)
        observations or "truth"

    covars: (N_obs, N_times, N_times)
        matrix of covariances for each observable

    t_ind: int
        the time index we are getting the forcing for

    Returns
    -------
    misfit_t: (N_obs)
        the vector measuring misfit between my_obs and the observations or
        "truth"
    """

    if t_ind < 0:
        raise ValueError("Time index must be positive.")

    # establish empty list of misfits
    # this is an (N_obs) vector
    misfit_t = np.zeros(len(covars))

    # unpack covariances
    inv_covar_T1, inv_covar_Q = covars

    # compute all the misfits for all times
    misfits = my_obs - obs

    # compute diagonal pieces of the misfit
    mis_TT = inv_covar_T1[t_ind, t_ind] * misfits[0, t_ind]
    mis_QQ = inv_covar_Q[t_ind, t_ind] * misfits[1, t_ind]

    # set temporary variables for off-diagonal misfit pieces
    tmp_mis_T = 0
    tmp_mis_Q = 0

    # compute off-diagonal bits
    for t in range(len(inv_covar_T1[0, :])):
        # we exclude t = t_ind from the off-diagonal pieces
        if t != t_ind:
            tmp_mis_T += inv_covar_T1[t_ind, t] * misfits[0, t]
            tmp_mis_Q += inv_covar_Q[t_ind, t] * misfits[1, t]

    # add diagonal and off diagonal pieces and return
    misfit_t[0] = mis_TT + tmp_mis_T
    misfit_t[1] = mis_QQ + tmp_mis_Q

    return misfit_t
