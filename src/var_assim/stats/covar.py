"""Covariance generation functions.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.15.2024
"""

import numpy as np


def get_covar_white(stds, N, inv=False):
    """Get inverse covariance matrix for white noise.

    Parameters
    ----------
    stds: (N,) list
        std of white noises for each prior

    N: int
        number of entries
        NOTE: this can be either the number of control variables or the number
        of timesteps, depending on what we're doing

    inv: bool
        return inverse?

    Returns
    -------
    inv_covar: (N, N)
        inverse covariance matrix for priors
    """

    covar = stds[:, None] ** 2 * np.identity(N)

    if inv:
        return np.linalg.inv(covar)

    else:
        return covar


def get_covar_ar1(std, corr_coeff, N, inv=False):
    """Get covariance matrix for AR(1) process.

    Compute inverse covariance matrix of an AR(1) process:
        y_{t+1} = corr_coeff * y_{t} + x_{t+1}

    where
        x_{t+1} ~ N(0, std**2).

    Parameters
    ----------
    std: float
        std of white noise that drives AR(1) model

    N: int
        number of entries
        NOTE: this can be either the number of control variables or the number
        of timesteps, depending on what we're doing

    inv: bool
        return inverse?

    Returns
    -------
    inv_covar: (N, N)
        inverse covariance matrix for priors
    """

    covar = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            covar[i, j] = corr_coeff ** (abs(i - j)) * std**2 / (1 - corr_coeff**2)

    if inv:
        return np.linalg.inv(covar)

    return covar
