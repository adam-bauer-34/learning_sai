"""Dynamics functions.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.10.2024
"""

import numpy as np


def get_nonlin_path(e, theta, TMIN, TMAX, DT):
    """Get nonlinear path for our model.
    """

    # unpack parameters
    params = theta[:13]
    T10, T20, Q0, L, G, C1, C2, F1_CO2, F3_CO2, A_SO2, B_SO2, C_SO2, EPS = params
    qs = theta[13:]  # model errors

    # make time list and bare paths list
    times = np.arange(TMIN, TMAX + DT, DT)
    paths = np.zeros((len(theta), len(times)))

    # set ics
    paths[:, 0] = theta

    # initial condition of T1 is T1_0 + q0
    paths[0, 0] = T10 + qs[0]

    # paths[0] = T1, paths[1] = T2
    # iterate through time and integrate two layer model
    for t in range(1, len(times)):
        T1, T2 = paths[:2, t-1]
        F = get_forcing(e, F1_CO2, F3_CO2, A_SO2, B_SO2, C_SO2, t-1)
        paths[0, t] = (1 - DT * (L + G * EPS) * C1**(-1)) * T1\
            + DT * G * EPS * C1**(-1) * T2\
            + DT * C1**(-1) * F\
            + DT * C1**(-1) * qs[t]
        paths[1, t] = (1 - DT * G * C2**(-1)) * T2\
            + DT * G * C2**(-1) * T1

        # Q = T1 * C1 + T2 * C2
        paths[2, t] = paths[0, t] * C1 + paths[1, t] * C2

    # make error paths
    paths[3:] = np.array([
        [theta[3+i]] * len(times) for i in range(len(theta) - 3)])

    # return paths and times
    return paths, times


def get_forcing(e, F1_CO2, F3_CO2, A_SO2, B_SO2, C_SO2, t):
    """Compute forcing.
    """

    F_CO2 = F1_CO2 * np.log(e.conc['CO2'][t] / 278.3)\
        + F3_CO2 * (np.sqrt(e.conc['CO2'][t])
                    - np.sqrt(278.3))
    emis_SO2 = e.emis['Sulfur'][t] - e.ref_emis['Sulfur']
    F_SO2 = A_SO2 * emis_SO2\
        + B_SO2 * np.log(1 + (emis_SO2 / C_SO2))
    return F_CO2 + F_SO2


def get_tlm_path(e, theta, TMIN, TMAX, DT, nl_path):
    """Get tangent linear path for solo parameter in 2 layer model.
    """

    # make time list and bare paths list
    times = np.arange(TMIN, TMAX + DT, DT)
    paths = np.zeros_like(nl_path, dtype=float)

    # unpack parameters
    params = theta[:13]
    T10, T20, Q0, L, G, C1, C2, F1_CO2, F3_CO2, A_SO2, B_SO2, C_SO2, EPS = params
    qs = theta[13:]  # model errors

    # set ics
    paths[:, 0] = theta

    # initial conditions of T1 is T1_0 + q0
    paths[0, 0] = T10 + qs[0]

    # paths[0] = T1, paths[1] = T2
    # iterate through time and integrate two layer model
    for t in range(1, len(times)):
        # extract previous timesteps dPath
        dPath = paths[:, t-1]

        # compute nontrivial components of TLM
        TLM_matrix = get_TLM_matrix(e, t-1, nl_path, DT, False)

        # compute TLM paths
        paths[:, t] = TLM_matrix @ dPath

    # return paths and times
    return paths


def get_TLM_matrix(e, t, nl_path, DT, CHECK_TLM=False):
    """Make TLM matrix.
    """

    # unpack current nonlinear path
    T1, T2, Q, L, G, C1, C2, F1_CO2, F3_CO2, A_SO2, B_SO2, C_SO2, EPS = nl_path[:13, t]
    mod_error = nl_path[13:, t]

    # initialize empty TLM
    TLM_matrix = np.zeros((np.shape(nl_path)[0],
                           np.shape(nl_path)[0]))

    # get forcing
    F = get_forcing(e, F1_CO2, F3_CO2, A_SO2, B_SO2, C_SO2, t)

    # sulfur emissions, for clarity
    emis_so2 = e.emis['Sulfur'][t] - e.ref_emis['Sulfur']

    # first row
    # this is the most complicated, since it's the ODE with the forcing
    # equation in it
    TLM_matrix[0, 0] = 1 - (DT * (G * EPS + L) * C1**(-1))
    TLM_matrix[0, 1] = DT * G * EPS / C1
    TLM_matrix[0, 2] = 0.0
    TLM_matrix[0, 3] = - DT * T1 / C1
    TLM_matrix[0, 4] = DT * EPS * (T2 - T1) / C1
    TLM_matrix[0, 5] = (T1 * (G * EPS + L)
                        - T2 * G * EPS
                        - F - mod_error[t+1]) * DT * C1**(-2)
    TLM_matrix[0, 6] = 0.0
    TLM_matrix[0, 7] = DT * np.log(e.conc['CO2'][t] / 278.3) / C1
    TLM_matrix[0, 8] = DT * (np.sqrt(e.conc['CO2'][t]) - np.sqrt(278.3)) / C1
    TLM_matrix[0, 9] = DT * emis_so2 / C1
    TLM_matrix[0, 10] = DT * np.log(1 + (emis_so2 / C_SO2)) / C1
    TLM_matrix[0, 11] = -1 * DT * B_SO2 * emis_so2\
        * (C1 * C_SO2**2 * (1 + emis_so2/C_SO2))**(-1)
    TLM_matrix[0, 12] = DT * (T2 - T1) * G / C1
    TLM_matrix[0, 13 + t] = DT / C1  # this bit is for model errors

    # second row
    # this one is a bit simpler, as there is no forcing term in it
    TLM_matrix[1, 0] = DT * G / C2
    TLM_matrix[1, 1] = 1 - (DT * G * C2**(-1))
    TLM_matrix[1, 2] = 0.0
    TLM_matrix[1, 3] = 0.0
    TLM_matrix[1, 4] = DT * (T1 - T2) / C2
    TLM_matrix[1, 5] = 0.0
    TLM_matrix[1, 6] = G * DT * (T2 - T1) / C2**2

    # third row
    # for ocean heat content
    TLM_matrix[2, 0] = C1 - DT * (G * (EPS - 1) + L)
    TLM_matrix[2, 1] = C2 + DT * G * (EPS - 1)
    TLM_matrix[2, 2] = 0.0
    TLM_matrix[2, 3] = - DT * T1
    TLM_matrix[2, 4] = DT * (EPS - 1) * (T2 - T1)
    TLM_matrix[2, 5] = T1
    TLM_matrix[2, 6] = T2
    TLM_matrix[2, 7] = DT * np.log(e.conc['CO2'][t] / 278.3)
    TLM_matrix[2, 8] = DT * (np.sqrt(e.conc['CO2'][t]) - np.sqrt(278.3))
    TLM_matrix[2, 9] = DT * emis_so2
    TLM_matrix[2, 10] = DT * np.log(1 + (emis_so2/C_SO2))
    TLM_matrix[2, 11] = - DT * B_SO2 * emis_so2\
        * (C_SO2**2 * (1 + (emis_so2/C_SO2)))**(-1)
    TLM_matrix[2, 12] = DT * G * (T2 - T1)
    TLM_matrix[2, 13 + t] = DT

    # all the parameters are just the identity
    TLM_matrix[3:, 3:] = np.identity(len(nl_path[3:, t]))

    if CHECK_TLM:
        print(TLM_matrix[0])

    return TLM_matrix
