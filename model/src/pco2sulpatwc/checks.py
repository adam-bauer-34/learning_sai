"""Checks on formulation.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.12.2024
"""

import os

import numpy as np
import pandas as pd

from .dynamics import get_nonlin_path, get_tlm_path
from .adjoint import get_adj_path
from .cost import cost, grad


def get_tlm_check(e, theta, TMIN, TMAX, DT, ALPHA_MIN, ALPHA_MAX,
                  SAVE_RESULTS=False):
    """Tangent linear model check.

    This verifies the tangent linear model following the prescription of
    p. 89 of Priciples of Data Assimilation by Park and Zupanski.

    Parameters
    ----------
    theta: (9,) list
        parameters

    e: `EmissionsBaseline` object
        the emissions baseline we're considering

    ics: (2,) list
        initial conditions for states

    TMIN: int
        initial time

    TMAX: int
        final time

    DT: float
        time discretization

    ALPHA_MIN: float
        minimum perturbation size

    ALPHA_MAX: float
        maximum perturbation size

    SAVE_RESULTS: bool (default False)
        save output to csv?

    Returns
    -------
    df: Pandas.DataFrame
        dataframe containing results of check
    """

    if ALPHA_MIN > ALPHA_MAX:
        raise ValueError("Perturbation size minimum is greater than maximum. Adjust alpha_min and alpha_max parameters.")

    oom_diff = int(np.log10(ALPHA_MAX) - np.log10(ALPHA_MIN))
    alphas = [ALPHA_MAX/10**i for i in range(abs(oom_diff))]

    # empty array for norms to be stored in
    norms = []

    # get TLM and nonlinear paths without perturbations
    data_base_p, _ = get_nonlin_path(e, theta, TMIN, TMAX, DT)
    tlm_p = get_tlm_path(e, theta, TMIN, TMAX, DT, data_base_p)

    tlm_norm_unscaled = np.sqrt(np.sum(tlm_p**2))

    for a in alphas:
        # get perturbed path, base path, and tangent linear path
        data_pert_p, _ = get_nonlin_path(e, theta * (1 + a), TMIN, TMAX,
                                         DT)

        # compute norms
        # NOTE: i'm using L2 norms because the book doesn't specify what norm
        # to take
        tlm_norm = a * tlm_norm_unscaled
        pert_norm = np.sqrt(np.sum((data_pert_p - data_base_p)**2))

        # append ratio of two norms
        norms.append(pert_norm/tlm_norm)

    # make dataframe of output and return
    data = {
        "perturbation_size": alphas,
        "norms": norms,
        "log(|norms - 1|)": np.log10(np.abs(np.array(norms) - 1))
    }
    df = pd.DataFrame(data)

    print("\nResults from tangent linear model test:")
    print("NOTE: norms should get closer to unity as you decrease perturbation size.\n")
    print(df)

    if SAVE_RESULTS:
        # save output to csv
        cwd = os.getcwd()
        datapath = cwd + "/data/checks/pco2sulpatwc/tlm_check.csv"

        df.to_csv(datapath, sep=',', index=False)

        print("\n------------------------------------------------------------------")
        print("Tangent linear model accuracy check results successfully saved to:\n{}".format(datapath))
        print("------------------------------------------------------------------\n")

    return df


def get_adj_id_check(e, theta, TMIN, TMAX, DT, SAVE_RESULTS=False):
    """Adjoint identity check.

    This verifies the adjoint identity equation following the prescription of
    p. 94 of Priciples of Data Assimilation by Park and Zupanski.

    Parameters
    ----------
    theta: (6,) list
        initial conditions plus three parameters (rho, sigma, beta)
    
    T_max: int
        maximum timesteps to take for verification
    
    dt: float
        time discretization

    save_results: bool (default False)
        save output to csv?

    Returns
    -------
    df: Pandas.DataFrame
        dataframe containing results of check
    """

    # make empty identity list
    id_t = []

    # loop through, taking progressively more timesteps
    # (presumably, the more timesteps you take, the worse the linear
    # approximation does)
    for t in range(1, TMAX - TMIN + 1):
        # get TLM trajectory and full nonlinear trajectory
        nonlin_p, times = get_nonlin_path(e, theta, TMIN, TMIN + t, DT)
        tlm_p = get_tlm_path(e, theta, TMIN, TMIN + t, DT, nonlin_p)

        # lhs of check
        check_lhs = 0
        for i in range(np.shape(tlm_p)[0]):
            check_lhs += tlm_p[i] @ tlm_p[i]  # tlm.T . tlm

        # get adjoint integration
        adj_p = get_adj_path(e, theta, TMIN, t + TMIN, DT, nonlin_p,
                             covars=None, obs=None, id_check=True)
        check_rhs = 0
        for i in range(np.shape(adj_p)[0]):
            check_rhs += nonlin_p[i] @ adj_p[i]

        id_t.append(check_lhs/check_rhs)

    # make dataframe of output and return
    data = {
        "timesteps taken": [t for t in range(1, TMAX - TMIN + 1)],
        "identity": id_t,
        "log(|identity - 1|)": np.log10(np.abs(np.array(id_t) - 1))
    }
    df = pd.DataFrame(data)

    print("\nResults from adjoint identity check:")
    print("NOTE: identity should get further away from unity the more timesteps you take.\n")
    print(df)

    if SAVE_RESULTS:
        # save output to csv
        cwd = os.getcwd()
        datapath = cwd + "/data/checks/pco2sulpatwc/adj_id_check.csv"

        df.to_csv(datapath, sep=',', index=False)

        print("\n------------------------------------------------------------------")
        print("Adjoint identity check results successfully saved to:\n{}".format(datapath))
        print("------------------------------------------------------------------\n")

    return df


def get_cost_grad_check(control, cost_args, ALPHA_MIN, ALPHA_MAX,
                        SAVE_RESULTS=False):
    """Cost gradient check function.
    """

    if ALPHA_MIN > ALPHA_MAX:
        raise ValueError("Perturbation size minimum is greater than maximum. Adjust alpha_min and alpha_max parameters.")

    oom_diff = int(np.log10(ALPHA_MAX) - np.log10(ALPHA_MIN))
    alphas = [ALPHA_MAX/(10**i) for i in range(oom_diff+1)]

    # empty array for phi to be stored in
    phi = []

    # compute base cost function with this control
    cost_base = cost(control, cost_args)

    # get base path and gradient from adjoint
    grad_base = grad(control, cost_args)

    # h vector, normalized to have unit length for perturbations
    h = grad_base / np.sqrt(np.sum(grad_base**2))

    # unscaled grad norm
    GRAD_NORM_UNSCALED = h.T @ grad_base

    for a in alphas:
        # get perturbed and base cost functions
        cost_pert = cost(control + a * h, cost_args)

        # compute diff between perturbed cost and base cost
        phi_num = cost_pert - cost_base

        # evaluate denominator of check
        phi_denom = a * GRAD_NORM_UNSCALED

        # append ratio
        phi.append(phi_num/phi_denom)

    # make dataframe of output and return
    data = {
        "perturbation_size": alphas,
        "phi": phi,
        "log(phi - 1)": np.log10(abs(np.array(phi) - 1))
    }
    df = pd.DataFrame(data)

    print("\nResults from cost function gradient test:")
    print("NOTE: phi should approach unity as the perturbation size shrinks.")
    print("NOTE: log(phi - 1) should be V-shaped.\n")
    print(df)

    if SAVE_RESULTS:
        # save output to csv
        cwd = os.getcwd()
        datapath = cwd + "/data/checks/pco2sulpatwc/cost_grad_check.csv"

        df.to_csv(datapath, sep=',', index=False)

        print("\n------------------------------------------------------------------")
        print("Cost function gradient check results successfully saved to:\n{}".format(datapath))
        print("------------------------------------------------------------------\n")

    return df
