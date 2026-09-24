"""Dynamics functions.

Variant of pco2geowc3_reg in which the initial condition is the control vector
verbatim (no T0 + q[0] offset) and the model-error blocks have no t = 0 entry,
so each block has length N_times - 1.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.10.2024
"""

import warnings

import numpy as np

# filter out overflow errors that happen during optimization
# (they never impact the answer)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_nonlin_path(e, theta, TMIN, TMAX, DT):
    """Get nonlinear path for our model."""

    # unpack parameters
    params = theta[:18]
    (
        T10,
        T20,
        Q0,
        TR10,
        TR20,
        TR30,
        L,
        G,
        EPS,
        C1,
        C2,
        F1_CO2,
        ALPHA_R1,
        ALPHA_R2,
        ALPHA_R3,
        BETA_R1,
        BETA_R2,
        BETA_R3,
    ) = params
    Qs = theta[18:]  # model errors

    # make time list and bare paths list
    times = np.arange(TMIN, TMAX + DT, DT)
    paths = np.zeros((len(theta), len(times)))

    # extract model errors for gmst and each region. there is no model error at
    # t = 0, so each block has one entry per *step*, M = N_times - 1, and block
    # index j is the error applied in the step t = j -> t = j + 1
    #
    # only a too-short vector is rejected: the adjoint identity check in
    # tlm_adj_checks integrates progressively shorter sub-windows with the full
    # window's control vector, which (as in pco2geowc3_reg) is tolerated
    M = len(times) - 1
    if len(Qs) < 4 * M:
        raise ValueError(
            f"Expected 4 model-error blocks of length {M} (N_times - 1), got "
            f"{len(Qs)} model errors."
        )
    q_AT, q_R1, q_R2, q_R3 = (Qs[i * M : (i + 1) * M] for i in range(4))

    # set ics. the initial condition is the control vector verbatim, with no
    # model-error offset
    paths[:, 0] = theta

    # paths[0] = T1, paths[1] = T2
    # iterate through time and integrate two layer model. the step into time t
    # uses q[t - 1], because q has no t = 0 entry
    for t in range(1, len(times)):
        T1, T2 = paths[:2, t - 1]
        F = get_forcing(e, F1_CO2, t - 1)
        paths[0, t] = (
            (1 - DT * (L + G * EPS) * C1 ** (-1)) * T1
            + DT * G * EPS * C1 ** (-1) * T2
            + DT * C1 ** (-1) * F
            + DT * C1 ** (-1) * q_AT[t - 1]
        )
        paths[1, t] = (1 - DT * G * C2 ** (-1)) * T2 + DT * G * C2 ** (-1) * T1

        # Q = T1 * C1 + T2 * C2
        paths[2, t] = paths[0, t] * C1 + paths[1, t] * C2

        # Tr = alpha_r * T1 - beta_r * geo_level + regional_noise
        paths[3, t] = (
            ALPHA_R1 * paths[0, t] + BETA_R1 * e.emis["geo"][t] + q_R1[t - 1]
        )

        paths[4, t] = (
            ALPHA_R2 * paths[0, t] + BETA_R2 * e.emis["geo"][t] + q_R2[t - 1]
        )

        paths[5, t] = (
            ALPHA_R3 * paths[0, t] + BETA_R3 * e.emis["geo"][t] + q_R3[t - 1]
        )

    # make stationary paths for parameters and model errors
    paths[6:] = theta[6:, None]

    # return paths and times
    return paths, times


def get_forcing(e, F1_CO2, t):
    """Compute forcing."""

    F_CO2 = F1_CO2 * np.log(e.conc["CO2"][t] / 278.3)
    F_GEO = e.forcing["geo"][t]
    return F_CO2 + F_GEO


def get_tlm_path(e, theta, TMIN, TMAX, DT, nl_path):
    """Get tangent linear path for solo parameter in 2 layer model."""

    # make time list and bare paths list
    times = np.arange(TMIN, TMAX + DT, DT)
    paths = np.zeros_like(nl_path, dtype=float)

    # unpack parameters
    params = theta[:18]
    (
        T10,
        T20,
        Q0,
        TR10,
        TR20,
        TR30,
        L,
        G,
        EPS,
        C1,
        C2,
        F1_CO2,
        ALPHA_R1,
        ALPHA_R2,
        ALPHA_R3,
        BETA_R1,
        BETA_R2,
        BETA_R3,
    ) = params
    Qs = theta[18:]  # model errors

    # set ics. the initial condition is the control vector verbatim, with no
    # model-error offset, so the initial perturbation is theta itself
    paths[:, 0] = theta

    # paths[0] = T1, paths[1] = T2
    # iterate through time and integrate two layer model
    for t in range(1, len(times)):
        # extract previous timesteps dPath
        dPath = paths[:, t - 1]

        # compute nontrivial components of TLM
        TLM_matrix = get_TLM_matrix(e, t - 1, nl_path, DT, False)

        # compute TLM paths
        paths[:, t] = TLM_matrix @ dPath

    # return paths and times
    return paths


def get_TLM_matrix(e, t, nl_path, DT, CHECK_TLM=False):
    """Make TLM matrix."""

    # unpack current nonlinear path
    (
        T1,
        T2,
        Q,
        TR1,
        TR2,
        TR3,
        L,
        G,
        EPS,
        C1,
        C2,
        F1_CO2,
        ALPHA_R1,
        ALPHA_R2,
        ALPHA_R3,
        BETA_R1,
        BETA_R2,
        BETA_R3,
    ) = nl_path[:18, t]
    Qs = nl_path[18:, t]

    N_times = np.shape(nl_path)[1]  # number of timesteps in this nl_path
    M = N_times - 1  # length of each model-error block (no t = 0 entry)

    q_AT, q_R1, q_R2, q_R3 = (Qs[i * M : (i + 1) * M] for i in range(4))

    # this TLM maps the state at index t -> t+1 (get_tlm_path calls this with
    # t-1). q has no t = 0 entry, so block index j is the error applied in the
    # step j -> j+1, and get_nonlin_path applies q[t] in this update. the
    # model-error columns and q values are therefore indexed at t, while the
    # other time-dependent terms below (e.g. e.emis["geo"][t + 1]) still index
    # the full time axis at t + 1.
    tq = t

    # initialize empty TLM
    TLM_matrix = np.zeros((np.shape(nl_path)[0], np.shape(nl_path)[0]))

    # get forcing
    F = get_forcing(e, F1_CO2, t)

    # first row
    # this is the most complicated, since it's the ODE with the forcing
    # equation in it
    TLM_matrix[0, 0] = 1 - (DT * (G * EPS + L) * C1 ** (-1))
    TLM_matrix[0, 1] = DT * G * EPS / C1
    TLM_matrix[0, 2] = 0.0
    TLM_matrix[0, 3] = 0.0
    TLM_matrix[0, 4] = 0.0
    TLM_matrix[0, 5] = 0.0
    TLM_matrix[0, 6] = -DT * T1 / C1
    TLM_matrix[0, 7] = DT * EPS * (T2 - T1) / C1
    TLM_matrix[0, 8] = DT * (T2 - T1) * G / C1
    TLM_matrix[0, 9] = (
        (T1 * (G * EPS + L) - T2 * G * EPS - F - q_AT[tq]) * DT * C1 ** (-2)
    )
    TLM_matrix[0, 10] = 0.0
    TLM_matrix[0, 11] = DT * np.log(e.conc["CO2"][t] / 278.3) / C1
    TLM_matrix[0, 12] = 0.0
    TLM_matrix[0, 13] = 0.0
    TLM_matrix[0, 14] = 0.0
    TLM_matrix[0, 15] = 0.0
    TLM_matrix[0, 16] = 0.0
    TLM_matrix[0, 17] = 0.0
    TLM_matrix[0, 18 + tq] = DT / C1  # this bit is for model errors

    # second row
    # this one is a bit simpler, as there is no forcing term in it
    TLM_matrix[1, 0] = DT * G / C2
    TLM_matrix[1, 1] = 1 - (DT * G / C2)
    TLM_matrix[1, 2] = 0.0
    TLM_matrix[1, 3] = 0.0
    TLM_matrix[1, 4] = 0.0
    TLM_matrix[1, 5] = 0.0
    TLM_matrix[1, 6] = 0.0
    TLM_matrix[1, 7] = DT * (T1 - T2) / C2
    TLM_matrix[1, 8] = 0.0
    TLM_matrix[1, 9] = 0.0
    TLM_matrix[1, 10] = G * DT * (T2 - T1) / C2**2
    TLM_matrix[1, 11] = 0.0
    TLM_matrix[1, 12] = 0.0
    TLM_matrix[1, 13] = 0.0
    TLM_matrix[1, 14] = 0.0
    TLM_matrix[1, 15] = 0.0
    TLM_matrix[1, 16] = 0.0
    TLM_matrix[1, 17] = 0.0
    TLM_matrix[1, 18 + tq] = 0.0

    # third row
    # for ocean heat content
    TLM_matrix[2, 0] = C1 - DT * (G * (EPS - 1) + L)
    TLM_matrix[2, 1] = C2 + DT * G * (EPS - 1)
    TLM_matrix[2, 2] = 0.0
    TLM_matrix[2, 3] = 0.0
    TLM_matrix[2, 4] = 0.0
    TLM_matrix[2, 5] = 0.0
    TLM_matrix[2, 6] = -DT * T1
    TLM_matrix[2, 7] = DT * (EPS - 1) * (T2 - T1)
    TLM_matrix[2, 8] = DT * G * (T2 - T1)
    TLM_matrix[2, 9] = T1
    TLM_matrix[2, 10] = T2
    TLM_matrix[2, 11] = DT * np.log(e.conc["CO2"][t] / 278.3)
    TLM_matrix[2, 12] = 0.0
    TLM_matrix[2, 13] = 0.0
    TLM_matrix[2, 14] = 0.0
    TLM_matrix[2, 15] = 0.0
    TLM_matrix[2, 16] = 0.0
    TLM_matrix[2, 17] = 0.0
    TLM_matrix[2, 18 + tq] = DT

    # fourth row
    # this is complicated because of the T1_t+1 dependence, so forcing will be included
    # regional temperature 1
    TLM_matrix[3, 0] = ALPHA_R1 - DT * ALPHA_R1 * (G * EPS + L) / C1
    TLM_matrix[3, 1] = DT * ALPHA_R1 * G * EPS / C1
    TLM_matrix[3, 2] = 0.0
    TLM_matrix[3, 3] = 0.0
    TLM_matrix[3, 4] = 0.0
    TLM_matrix[3, 5] = 0.0
    TLM_matrix[3, 6] = -DT * T1 * ALPHA_R1 / C1
    TLM_matrix[3, 7] = DT * (T2 - T1) * ALPHA_R1 * EPS / C1
    TLM_matrix[3, 8] = DT * (T2 - T1) * ALPHA_R1 * G / C1
    TLM_matrix[3, 9] = (DT * ALPHA_R1 / C1**2) * (
        -q_AT[tq] + T1 * (G * EPS + L) - T2 * G * EPS - F
    )
    TLM_matrix[3, 10] = 0.0
    TLM_matrix[3, 11] = DT * ALPHA_R1 * np.log(e.conc["CO2"][t] / 278.3) / C1
    TLM_matrix[3, 12] = (
        T1 + DT * (q_AT[tq] - T1 * (G * EPS + L) + T2 * G * EPS + F) / C1
    )
    TLM_matrix[3, 13] = 0.0
    TLM_matrix[3, 14] = 0.0
    TLM_matrix[3, 15] = e.emis["geo"][t + 1]
    TLM_matrix[3, 16] = 0.0
    TLM_matrix[3, 17] = 0.0
    TLM_matrix[3, 18 + tq] = DT * ALPHA_R1 / C1  # this bit is for model errors
    TLM_matrix[3, 18 + M + tq] = 1.0  # regional model errors for R1

    # fifth row
    # regional temperature 2
    TLM_matrix[4, 0] = ALPHA_R2 - DT * ALPHA_R2 * (G * EPS + L) / C1
    TLM_matrix[4, 1] = DT * ALPHA_R2 * G * EPS / C1
    TLM_matrix[4, 2] = 0.0
    TLM_matrix[4, 3] = 0.0
    TLM_matrix[4, 4] = 0.0
    TLM_matrix[4, 5] = 0.0
    TLM_matrix[4, 6] = -DT * T1 * ALPHA_R2 / C1
    TLM_matrix[4, 7] = DT * (T2 - T1) * ALPHA_R2 * EPS / C1
    TLM_matrix[4, 8] = DT * (T2 - T1) * ALPHA_R2 * G / C1
    TLM_matrix[4, 9] = (DT * ALPHA_R2 / C1**2) * (
        -q_AT[tq] + T1 * (G * EPS + L) - T2 * G * EPS - F
    )
    TLM_matrix[4, 10] = 0.0
    TLM_matrix[4, 11] = DT * ALPHA_R2 * np.log(e.conc["CO2"][t] / 278.3) / C1
    TLM_matrix[4, 12] = 0.0
    TLM_matrix[4, 13] = (
        T1 + DT * (q_AT[tq] - T1 * (G * EPS + L) + T2 * G * EPS + F) / C1
    )
    TLM_matrix[4, 14] = 0.0
    TLM_matrix[4, 15] = 0.0
    TLM_matrix[4, 16] = e.emis["geo"][t + 1]
    TLM_matrix[4, 17] = 0.0
    TLM_matrix[4, 18 + tq] = DT * ALPHA_R2 / C1  # this bit is for model errors
    TLM_matrix[4, 18 + 2 * M + tq] = 1.0  # regional model errors for R2

    # sixth row
    # regional temperature 3
    TLM_matrix[5, 0] = ALPHA_R3 - DT * ALPHA_R3 * (G * EPS + L) / C1
    TLM_matrix[5, 1] = DT * ALPHA_R3 * G * EPS / C1
    TLM_matrix[5, 2] = 0.0
    TLM_matrix[5, 3] = 0.0
    TLM_matrix[5, 4] = 0.0
    TLM_matrix[5, 5] = 0.0
    TLM_matrix[5, 6] = -DT * T1 * ALPHA_R3 / C1
    TLM_matrix[5, 7] = DT * (T2 - T1) * ALPHA_R3 * EPS / C1
    TLM_matrix[5, 8] = DT * (T2 - T1) * ALPHA_R3 * G / C1
    TLM_matrix[5, 9] = (DT * ALPHA_R3 / C1**2) * (
        -q_AT[tq] + T1 * (G * EPS + L) - T2 * G * EPS - F
    )
    TLM_matrix[5, 10] = 0.0
    TLM_matrix[5, 11] = DT * ALPHA_R3 * np.log(e.conc["CO2"][t] / 278.3) / C1
    TLM_matrix[5, 12] = 0.0
    TLM_matrix[5, 13] = 0.0
    TLM_matrix[5, 14] = (
        T1 + DT * (q_AT[tq] - T1 * (G * EPS + L) + T2 * G * EPS + F) / C1
    )
    TLM_matrix[5, 15] = 0.0
    TLM_matrix[5, 16] = 0.0
    TLM_matrix[5, 17] = e.emis["geo"][t + 1]
    TLM_matrix[5, 18 + tq] = DT * ALPHA_R3 / C1  # this bit is for model errors
    TLM_matrix[5, 18 + 3 * M + tq] = 1.0  # regional model errors for R3

    # all the parameters are just the identity
    unity_inds = np.arange(6, TLM_matrix.shape[0])
    TLM_matrix[unity_inds, unity_inds] = 1.0

    if CHECK_TLM:
        print(TLM_matrix[0])

    return TLM_matrix
