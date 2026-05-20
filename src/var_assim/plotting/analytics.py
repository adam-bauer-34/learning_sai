"""Analytics-based functions for analyzing 4DVAR solutions.

Adam Michael Bauer
University of Illinois Urbana-Champaign
9.24.2024
"""

import numpy as np


def tfast_flux(Ft, C1, phi_s, phi_f):
    """Fast piece flux.
    """
    
    flux = Ft * phi_s
    flux /= C1 * (phi_s - phi_f)
    return flux 


def tslow_flux(Ft, C1, phi_s, phi_f):
    """Slow piece flux.
    """
    
    flux = Ft * phi_f
    flux /= C1 * (phi_s - phi_f)
    return -flux


def get_path_pieces_analytic(tmin, tmax, params, F, assim, dt=1, equil_tol=1e-12):
    """Get analytic pieces of the temperature path.

    NOTE: I'm only computing the surface temperature here, because that's what
    I care about, but you could do the same thing for the deep ocean.
    """

    # initialize arrays
    times = np.arange(tmin, tmax, dt)

    # unpack parameters
    if assim == 'pco2sulwc':
        T10, T20, _, L, G, C1, C2 = params
        EPS = 1

    else:
        T10, T20, _, L, G, C1, C2, EPS = params

    T1_full = np.zeros_like(times, dtype=float)
    T2_full = np.zeros_like(times, dtype=float)

    T1_full[0] = T10
    T2_full[0] = T20

    for t in range(1, len(times)):
        T1_full[t] = T1_full[t-1] + (dt / C1)\
            * (F[t-1] - L * T1_full[t-1] + G * EPS * (T2_full[t-1] - T1_full[t-1]))
        T2_full[t] = T2_full[t-1] + (dt / C2)\
            * (G * (T1_full[t-1] - T2_full[t-1]))

    # compute a bunch of other quantities necessary to construct the analytic
    # solution. all of these are taken from Table 1 in Geoffroy et al 2014a
    ## eigenvalue parameters
    b = (L + G * EPS)/C1 + G/C2
    bstar = (L + G * EPS)/C1 - G/C2
    delta = b**2 - 4 * (L * G)/(C1 * C2)

    ## mode parameters
    ### fast mode
    phi_f = (C1 / (2 * G)) * (bstar - np.sqrt(delta))
    t_f = (C1 * C2 / (2 * L * G)) * (b - np.sqrt(delta))

    ### slow mode
    phi_s = (C1 / (2 * G)) * (bstar + np.sqrt(delta))
    t_s = (C1 * C2 / (2 * L * G)) * (b + np.sqrt(delta))

    if phi_s - phi_f <= 0:
        raise ValueError("Something went wrong: phi_f > phi_s. Maybe parameter values got mixed up?")

    ### initial condition simplifications
    T1_ic = phi_s * T10 - T20
    T2_ic = -1 * phi_f * T10 + T20

    # compute "pieces" of analytic temperature pathway
    ## initial condition piece
    T_ics = T1_ic * np.exp(-times / t_f) + T2_ic * np.exp(-times / t_s)
    T_ics /= phi_s - phi_f

    ## fast piece and slow pieces
    T_fast = np.zeros_like(times, dtype=float)
    T_slow = np.zeros_like(times, dtype=float)
    
    for t in range(1, len(times)):
        T_fast[t] = T_fast[t-1] + dt * ( -T_fast[t-1] / t_f + tfast_flux(F[t-1], C1, phi_s, phi_f))
    
        if t > 100:
            if abs(T_fast[t-1] - T_fast[t]) <= equil_tol:
                T_fast[t:] = T_fast[t]
                break
    
    for t in range(1, len(times)):
        T_slow[t] = T_slow[t-1] + dt * ( -T_slow[t-1] / t_s + tslow_flux(F[t-1], C1, phi_s, phi_f))
    
        if t > 100:
            if abs(T_slow[t-1] - T_slow[t]) <= equil_tol:
                T_slow[t:] = T_slow[t]
                break

    return T_ics, T_fast, T_slow, T1_full, T2_full
