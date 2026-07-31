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


def get_window_max_timesteps(windows, DT):
    """Number of timesteps in the longest assimilation window.

    Parameters
    ----------
    windows: list of (int, int)
        (TMIN, TMAX) pairs, as held by `AssimilationWindowing.windows`

    DT: float
        time discretization

    Returns
    -------
    N_max: int
        timestep count of the longest window
    """

    return max(len(np.arange(t_0, t_1 + DT, DT)) for t_0, t_1 in windows)


def get_window_prefix_inds(n_fixed, n_blocks, N_max, N_times):
    """Map a full-length control vector onto a single assimilation window.

    The control vector is [n_fixed initial conditions and parameters] followed by
    `n_blocks` model-error blocks of one entry per timestep. Because the block
    length grows with the window, the offset of every block after the first moves
    too, so a window's control vector is not a prefix of the full one -- it is the
    fixed head plus the leading `N_times` entries of each block.

    Why this exists: drawing the truth's model errors and the prior ensemble
    separately in each window does not reproduce the same values, even with the
    RNG re-seeded, because the dimension of the draw changes. numpy factors the
    covariance by SVD and applies it to a stream of standard normals; for the
    dense AR(1) block that factor depends on the dimension, so the same normals
    are mixed differently. Drawing `size=n_ens` samples is worse still, since
    `standard_normal((n_ens, d))` is filled row-major and every member's slice of
    the stream shifts when `d` changes. Measured on a four-window run, the true
    global model errors differed by up to 0.85 K on the shared span against a
    truth std of 0.24 K, and 0 of 100 prior members were reproduced.

    So both are drawn once at the longest window and sliced with these indices.
    That is exact rather than approximate: the AR(1) covariance is Toeplitz, so
    `get_covar_ar1(s, r, N_max)[:n, :n] == get_covar_ar1(s, r, n)`, and the white
    blocks are diagonal. Taking a prefix of each block therefore marginalises the
    long draw onto precisely the shorter window's distribution.

    Parameters
    ----------
    n_fixed: int
        number of initial conditions plus parameters, i.e. 3 + n_reg + 6 + 2 * n_reg

    n_blocks: int
        number of model-error blocks (1 for global noise only, 1 + n_reg with
        regional noise, 0 for a deterministic model)

    N_max: int
        block length in the full-length vector

    N_times: int
        block length for this window; must not exceed N_max

    Returns
    -------
    inds: (n_fixed + n_blocks * N_times,) int array
        columns of the full-length vector belonging to this window
    """

    if N_times > N_max:
        raise ValueError(
            f"Window has {N_times} timesteps, more than the {N_max} the "
            "full-length draws were made with. N_max must come from the longest "
            "window in the windowing scheme."
        )

    return np.hstack(
        [np.arange(n_fixed)]
        + [
            n_fixed + b * N_max + np.arange(N_times, dtype=int)
            for b in range(n_blocks)
        ]
    ).astype(int)
