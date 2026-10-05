"""Adjoint for the strong-constraint model.

`get_forced_adj` integrates the adjoint of the noise-forced trajectory backwards,
forced by a pre-weighted innovation v = S^{-1} (model - data) of shape (4, N):

    lam_{N-1} = H^T v_{N-1}
    lam_{t-1} = TLM_{t-1}^T lam_t + H^T v_{t-1},   t = N-1 ... 1

so lam_0 is the gradient of 1/2 d^T S^{-1} d with respect to the 15 controls, for
S held fixed. The t = 0 forcing is essential: the regional and OHC initial
conditions reach the cost only through their own t = 0 observation, so for those
controls it is the entire gradient.

The sign convention is the weak model's: the forcing is S^{-1}(model - data).

Adam Michael Bauer
UChicago
"""

import numpy as np

from .dynamics import N_FIXED, get_forced_TLM_matrix, get_TLM_matrix, get_tlm_path

# state rows that the observation vector [T1, Q, T_R1, T_R2] selects
OBS_ROWS = (0, 2, 3, 4)


def _obs_forcing(v, t):
    """H^T v_t: scatter the four weighted innovations onto their state rows."""

    f = np.zeros(N_FIXED)
    f[list(OBS_ROWS)] = v[:, t]
    return f


def get_forced_adj(e, path, g, design, v, DT):
    """Adjoint of the forced trajectory.

    Parameters
    ----------
    path: (15, N) array
        forced trajectory the adjoint is linearized about

    g: (N - 1,) array
        global noise block that produced `path`

    design: str
        "flux" or "temp"

    v: (4, N) array
        weighted innovation S^{-1}(model - data), observable-major rows

    Returns
    -------
    lam: (15, N) array
        adjoint path; lam[:, 0] is the gradient
    """

    N = path.shape[1]
    lam = np.zeros((N_FIXED, N))
    lam[:, N - 1] = _obs_forcing(v, N - 1)

    for t in range(N - 1, 0, -1):
        TLM = get_forced_TLM_matrix(e, t - 1, path, g, design, DT)
        lam[:, t - 1] = TLM.T @ lam[:, t] + _obs_forcing(v, t - 1)

    return lam


def get_adj_path(
    e, theta, TMIN, TMAX, DT, nl_path, covars=None, obs=None, id_check=True
):
    """Noiseless adjoint, with the signature tlm_adj_checks imports.

    Only the identity-check mode is supported: the gradient of this model's cost
    is assembled in cost.grad from get_forced_adj, because the forcing is the
    dense S^{-1}(model - data) rather than per-observable covariances.
    """

    if not id_check:
        raise NotImplementedError(
            "pco2geosc_reg_noic computes its gradient in cost.grad via "
            "get_forced_adj; get_adj_path supports only id_check=True."
        )

    adj_path = np.zeros_like(nl_path, dtype=float)
    tlm_path = get_tlm_path(e, theta, TMIN, TMAX, DT, nl_path)
    adj_path[:, -1] = tlm_path[:, -1]

    for i in range(nl_path.shape[1] - 1, 0, -1):
        adj_path[:, i - 1] = get_TLM_matrix(e, i - 1, nl_path, DT).T @ adj_path[:, i]

    return adj_path
