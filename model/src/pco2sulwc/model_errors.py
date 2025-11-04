"""Generate model error time series.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.23.2024
"""

import numpy as np

from ..stats.covar import get_covar_white, get_covar_ar1


def gen_noise_ts(AR_P, N_times, STD, CORR_COEFFS=None):
    """Generate a time series of model errors to force the model.

    Parameters
    ----------
    AR_P: int
        the "P" in AR(P) process. tells the code what type of forcing

    N_times: int
        number of time steps we have; tells code how long of a vector to return

    STD: float
        standard deviation of Gaussian white noise that forces AR(P) process

    CORR_COEFF: (AR_P,) list
        Default: None
        correlation coefficients for AR(P) process

    Returns
    -------
    model_errors: (N_times,) array
        list of model errors

    model_error_covar: (N_times, N_times) matrix
        model error covariance matrix
    """

    if AR_P == 0:
        model_error_covar = get_covar_white(np.array([STD] * N_times),
                                            N_times)

    elif AR_P == 1:
        model_error_covar = get_covar_ar1(STD, CORR_COEFFS[0], N_times)

    else:
        raise ValueError("Invalid noise model. Only AR(0) and AR(1) are currently implemented")

    model_errors = np.random.multivariate_normal(np.array([0.0] * N_times),
                                                 model_error_covar)

    return model_errors, model_error_covar
