"""Postprocessing functions of model simulations.

Adam Bauer
UChicago
"""

import numpy as np


def get_angle_r2(a1, a2, b1, b2, l, e, g, phi=0.09):
    """Compute the SAI inequality angle for the two region model.

    NOTE: a1, a2, b1, and b2 are indexed **by the angle** not by region!

    Parameters
    ----------
    a1: list or float
        alpha_1 (temperature pattern scale for region 1)
    a2: list or float
        alpha_2 (temperature pattern scale for region 2)
    b1: list or float
        beta_1 (SAI CO2 pattern breaking for region 1)
    b2: list or float
        beta_2 (SAI CO2 pattern breaking for region 2)
    l: float
        lambda (climate feedback parameter)
    e: float
        epsilon (pattern effect parameter in EBM)
    g: float
        gamma (ocean heat exchange parameter in EBM)
    phi: float
        phi (forcing efficacy of SAI)

    Returns
    -------
    angle: list or float
        the inequality angle parameter for SAI
    """

    two_normed = np.array([a1, a2])
    inside = (l + e * g) / phi
    r_normed = -np.array([[b1 * inside - a1], [b2 * inside - a2]])

    num = two_normed @ r_normed
    denom = np.sqrt(np.sum(two_normed**2) * np.sum(r_normed**2))
    return np.arccos(num / denom) * 180 / np.pi


def get_angles_for_dt_list(dts, winds, N_ENS, N_regs=2):
    """Get angles for a list of datatrees.

    Parameters
    ----------
    dts: list
        list of (cleaned) datatrees

    winds: list
        assimilation windows (used to index datatrees)

    N_ENS: int
        number of ensemble members

    Returns
    -------
    angles: (len(dts), len(winds), N_ENS) list
        list of angles
    """

    angles = np.ones((len(dts), len(winds), N_ENS)) * np.nan

    for d_, dt in enumerate(dts):
        for w_, w in enumerate(winds):
            tmp_a1 = dt[w].ds.controls.sel(vari="ALPHA_R1").values
            tmp_a2 = dt[w].ds.controls.sel(vari="ALPHA_R2").values
            tmp_b1 = dt[w].ds.controls.sel(vari="BETA_R1").values
            tmp_b2 = dt[w].ds.controls.sel(vari="BETA_R2").values
            tmp_l = dt[w].ds.controls.sel(vari="L").values
            tmp_g = dt[w].ds.controls.sel(vari="G").values
            tmp_eps = dt[w].ds.controls.sel(vari="EPS").values

            tmp_a3 = (
                dt[w].ds.controls.sel(vari="ALPHA_R3").values if N_regs == 3 else None
            )
            tmp_b3 = (
                dt[w].ds.controls.sel(vari="BETA_R3").values if N_regs == 3 else None
            )

            for i in range(N_ENS):
                if N_regs == 3:
                    angles[d_, w_, i] = get_angle_r3(
                        tmp_a1[i],
                        tmp_a2[i],
                        tmp_a3[i],
                        tmp_b1[i],
                        tmp_b2[i],
                        tmp_b3[i],
                        tmp_l[i],
                        tmp_eps[i],
                        tmp_g[i],
                    )
                else:
                    angles[d_, w_, i] = get_angle_r2(
                        tmp_a1[i],
                        tmp_a2[i],
                        tmp_b1[i],
                        tmp_b2[i],
                        tmp_l[i],
                        tmp_eps[i],
                        tmp_g[i],
                    )

    return angles


def get_angle_r3(a1, a2, a3, b1, b2, b3, l, e, g, phi=0.09):
    """Compute the SAI inequality angle for the three region model.

    NOTE: a1, a2, a3, b1, b2, and b3 are indexed **by the angle** not by region!

    Parameters
    ----------
    a1: list or float
        alpha_1 (temperature pattern scale for region 1)
    a2: list or float
        alpha_2 (temperature pattern scale for region 2)
    a3: list or float
        alpha_3 (temperature pattern scale for region 3)
    b1: list or float
        beta_1 (SAI CO2 pattern breaking for region 1)
    b2: list or float
        beta_2 (SAI CO2 pattern breaking for region 2)
    b3: list or float
        beta_3 (SAI CO2 pattern breaking for region 3)
    l: float
        lambda (climate feedback parameter)
    e: float
        epsilon (pattern effect parameter in EBM)
    g: float
        gamma (ocean heat exchange parameter in EBM)
    phi: float
        phi (forcing efficacy of SAI)

    Returns
    -------
    angle: list or float
        the inequality angle parameter for SAI
    """

    two_normed = np.array([a1, a2, a3])
    inside = (l + e * g) / phi
    r_normed = -np.array([[b1 * inside - a1], [b2 * inside - a2], [b3 * inside - a3]])

    num = two_normed @ r_normed
    denom = np.sqrt(np.sum(two_normed**2) * np.sum(r_normed**2))
    return np.arccos(num / denom) * 180 / np.pi
