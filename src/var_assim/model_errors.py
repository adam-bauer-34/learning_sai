"""Generate model error time series.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.23.2024
"""

import numpy as np

from var_assim.stats.covar import get_covar_white, get_covar_ar1


def gen_noise_ts(Noise, N_times, rng=None, reg=False):
    """Generate a time series of model errors to force the model.

    Parameters
    ----------
    Noise: `ClimateModelNoise` dataclass
        contains all the noise attributes for our model

    N_times: int
        number of time steps we have; tells code how long of a vector to return

    rng: np.random.default_rng object
        the rng to use in making the draws

    reg: bool
        if these are regional model errors, use AR(0) process

    Returns
    -------
    model_errors: (N_times,) array
        list of model errors

    model_error_covar: (N_times, N_times) matrix
        model error covariance matrix
    """

    if rng is None:
        rng = np.random.default_rng()

    if Noise.NOISE_MODEL == "AR0":
        model_error_covar = get_covar_white(
            np.array([Noise.INT_VAR_STD] * N_times), N_times
        )

    elif Noise.NOISE_MODEL == "AR1":
        model_error_covar = get_covar_ar1(Noise.INT_VAR_STD, Noise.AUTO_CORR, N_times)

    else:
        raise ValueError(
            "Invalid noise model. Only AR(0) and AR(1) are currently implemented."
        )

    model_errors = rng.multivariate_normal(np.array([0.0] * N_times), model_error_covar)

    return model_errors, model_error_covar
