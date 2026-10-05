"""Dynamics for the strong-constraint model.

The control vector is the 15 fixed controls only,

    T1, T2, Q, T_R1, T_R2, L, G, EPS, C1, C2, F1_CO2, ALPHA_R1, ALPHA_R2,
    BETA_R1, BETA_R2,

and the model is deterministic. Internal variability is not a control here, but
it is still needed as a *known, exogenous* forcing: the envelope identity in
cost.grad differentiates the trajectory driven by the profiled noise xi*, and
the truth for --sc_noise temp is generated with it. So the core routines take an
optional noise vector xi = [global, q_R1, q_R2] (each block N - 1 long, no noise
at t = 0, as in pco2geowc_reg_noic) and a design:

    flux  T1 <- T1 + dt/C1 (F - L T1 + G EPS (T2 - T1)) + dt/C1 q_AT
    temp  T1 <- T1 + dt/C1 (F - L T1 + G EPS (T2 - T1)) + eps_T

T2 gets no noise term in either design.

`get_nonlin_path`, `get_tlm_path` and `get_TLM_matrix` are the noiseless versions,
with the signatures tlm_adj_checks imports by model name.

Adam Michael Bauer
UChicago
"""

import warnings

import numpy as np

from var_assim.models.pco2geowc_reg_noic.dynamics import get_forcing

from .noise_response import check_design

# filter out overflow errors that happen during optimization, as the weak model does
warnings.filterwarnings("ignore", category=RuntimeWarning)

N_FIXED = 15


def _split_noise(xi, M):
    """Split xi into (global, q_R1, q_R2); None means no noise."""

    if xi is None:
        z = np.zeros(M)
        return z, z, z
    xi = np.asarray(xi, dtype=float)
    if len(xi) != 3 * M:
        raise ValueError(f"Expected 3 noise blocks of length {M}, got {len(xi)} entries.")
    return xi[:M], xi[M : 2 * M], xi[2 * M :]


def get_forced_path(e, theta, xi, design, TMIN, TMAX, DT):
    """Trajectory driven by a known noise realization xi.

    Parameters
    ----------
    theta: (15,) array
        fixed controls; any extra entries are ignored

    xi: (3 (N - 1),) array or None
        noise [global, q_R1, q_R2]; the global block is in the design's units
        (heat flux for "flux", K for "temp")

    design: str
        "flux" or "temp"

    Returns
    -------
    paths: (15, N) array
    times: (N,) array
    """

    check_design(design)
    (
        T10,
        T20,
        Q0,
        TR10,
        TR20,
        L,
        G,
        EPS,
        C1,
        C2,
        F1_CO2,
        ALPHA_R1,
        ALPHA_R2,
        BETA_R1,
        BETA_R2,
    ) = theta[:N_FIXED]

    times = np.arange(TMIN, TMAX + DT, DT)
    N = len(times)
    g, r1, r2 = _split_noise(xi, N - 1)

    paths = np.zeros((N_FIXED, N))

    # the initial condition is the control vector verbatim, with no noise
    paths[:, 0] = theta[:N_FIXED]

    for t in range(1, N):
        T1, T2 = paths[:2, t - 1]
        F = get_forcing(e, F1_CO2, t - 1)

        # same term order as pco2geowc_reg_noic, so the flux design with the truth's
        # noise reproduces the weak model's truth path bit-for-bit
        if design == "flux":
            paths[0, t] = (
                (1 - DT * (L + G * EPS) * C1 ** (-1)) * T1
                + DT * G * EPS * C1 ** (-1) * T2
                + DT * C1 ** (-1) * F
                + DT * C1 ** (-1) * g[t - 1]
            )
        else:
            paths[0, t] = (
                (1 - DT * (L + G * EPS) * C1 ** (-1)) * T1
                + DT * G * EPS * C1 ** (-1) * T2
                + DT * C1 ** (-1) * F
                + g[t - 1]
            )

        # T2 has no noise term; it sees T1's noise only through G (T1 - T2)
        paths[1, t] = (1 - DT * G * C2 ** (-1)) * T2 + DT * G * C2 ** (-1) * T1

        paths[2, t] = paths[0, t] * C1 + paths[1, t] * C2

        paths[3, t] = ALPHA_R1 * paths[0, t] + BETA_R1 * e.emis["geo"][t] + r1[t - 1]
        paths[4, t] = ALPHA_R2 * paths[0, t] + BETA_R2 * e.emis["geo"][t] + r2[t - 1]

    # parameters are stationary states
    paths[5:] = theta[5:N_FIXED, None]

    return paths, times


def get_forced_TLM_matrix(e, t, path, g, design, DT):
    """15 x 15 tangent linear model of the forced step t -> t + 1.

    The noise is a known forcing here, not a control, so it has no column; it
    shows up only inside the theta-columns. Rows 0 (T1) and 1 (T2) are written
    out; rows 2-4 (Q, T_R1, T_R2) follow by the chain rule from the diagnostic
    relations Q = C1 T1 + C2 T2 and T_Rr = ALPHA_r T1 + BETA_r geo + q_R, using
    the forced next-step values. That keeps both designs right automatically:
    for "flux" the noise cancels out of the Q row, but for "temp" dQ/dC1 picks
    up + eps_T.

    For "flux" this equals pco2geowc_reg_noic.dynamics.get_TLM_matrix(...)[:15, :15]
    on the same noise-bearing path, to round-off.

    Parameters
    ----------
    t: int
        step index; maps the state at t to t + 1

    path: (15, N) array
        forced trajectory (get_forced_path)

    g: (N - 1,) array
        global noise block used to build `path`
    """

    (
        T1,
        T2,
        Q,
        TR1,
        TR2,
        L,
        G,
        EPS,
        C1,
        C2,
        F1_CO2,
        ALPHA_R1,
        ALPHA_R2,
        BETA_R1,
        BETA_R2,
    ) = path[:N_FIXED, t]
    T1n, T2n = path[0, t + 1], path[1, t + 1]
    F = get_forcing(e, F1_CO2, t)

    TLM = np.zeros((N_FIXED, N_FIXED))

    # row 0: T1
    TLM[0, 0] = 1 - (DT * (G * EPS + L) * C1 ** (-1))
    TLM[0, 1] = DT * G * EPS / C1
    TLM[0, 5] = -DT * T1 / C1
    TLM[0, 6] = DT * EPS * (T2 - T1) / C1
    TLM[0, 7] = DT * (T2 - T1) * G / C1
    TLM[0, 8] = (T1 * (G * EPS + L) - T2 * G * EPS - F) * DT * C1 ** (-2)
    if design == "flux":
        # the heat-flux kick is divided by C1, so it carries a C1 derivative
        TLM[0, 8] += -g[t] * DT * C1 ** (-2)
    TLM[0, 10] = DT * np.log(e.conc["CO2"][t] / 278.3) / C1

    # row 1: T2 (no noise term, so identical in both designs)
    TLM[1, 0] = DT * G / C2
    TLM[1, 1] = 1 - (DT * G / C2)
    TLM[1, 6] = DT * (T1 - T2) / C2
    TLM[1, 9] = G * DT * (T2 - T1) / C2**2

    # row 2: Q_{t+1} = C1 T1_{t+1} + C2 T2_{t+1}
    TLM[2] = C1 * TLM[0] + C2 * TLM[1]
    TLM[2, 8] += T1n
    TLM[2, 9] += T2n

    # rows 3, 4: T_Rr,{t+1} = ALPHA_r T1_{t+1} + BETA_r geo_{t+1} + q_Rr
    TLM[3] = ALPHA_R1 * TLM[0]
    TLM[3, 11] += T1n
    TLM[3, 13] += e.emis["geo"][t + 1]

    TLM[4] = ALPHA_R2 * TLM[0]
    TLM[4, 12] += T1n
    TLM[4, 14] += e.emis["geo"][t + 1]

    # the parameters are stationary
    TLM[5:, 5:] = np.identity(N_FIXED - 5)

    return TLM


# -- noiseless versions, with the signatures tlm_adj_checks expects ---------------


def get_nonlin_path(e, theta, TMIN, TMAX, DT):
    """Noiseless trajectory of the 15 fixed controls.

    The design does not matter without noise, so "flux" is passed only to satisfy
    the signature of get_forced_path.
    """

    return get_forced_path(e, np.asarray(theta, dtype=float), None, "flux", TMIN, TMAX, DT)


def get_TLM_matrix(e, t, nl_path, DT, CHECK_TLM=False):
    """Noiseless TLM (step t -> t + 1)."""

    TLM = get_forced_TLM_matrix(e, t, nl_path, np.zeros(nl_path.shape[1] - 1), "flux", DT)
    if CHECK_TLM:
        print(TLM[0])
    return TLM


def get_tlm_path(e, theta, TMIN, TMAX, DT, nl_path):
    """Noiseless tangent linear path, starting from the perturbation theta."""

    times = np.arange(TMIN, TMAX + DT, DT)
    paths = np.zeros_like(nl_path, dtype=float)
    paths[:, 0] = np.asarray(theta, dtype=float)[:N_FIXED]
    for t in range(1, len(times)):
        paths[:, t] = get_TLM_matrix(e, t - 1, nl_path, DT) @ paths[:, t - 1]
    return paths
