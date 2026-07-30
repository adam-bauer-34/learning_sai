"""Checks on formulation.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.12.2024
"""

import importlib

import numpy as np
import pandas as pd

from var_assim.config import DATA_DIR
from pathlib import Path


def run_component_checks(args, e, controls, TMIN, TMAX, cost_args, DT=1.0):
    """Check components of variational data assimilation.

    Parameters
    ----------
    args: argparse.Namespace
        namespace for current simulation

    e: EmissionsBaseline
        emissions baseline object

    controls: list
        control vector

    TMIN: int
        minimum time for integration during checks

    TMAX: int
        maximum time for integration during checks

    cost_args: list
        arguments for model.cost and model.grad functions

    DT: float = 1.0
        time discretization (usually set to be globally 1.0)
    """

    # import necessary functions, this scales to whatever model i use without big registries
    # necessary functions for TLM check
    get_nonlin_path = importlib.import_module(
        f"var_assim.models.{args.model}.dynamics"
    ).get_nonlin_path
    get_tlm_path = importlib.import_module(
        f"var_assim.models.{args.model}.dynamics"
    ).get_tlm_path

    # necessary functions for ADJ check
    get_adj_path = importlib.import_module(
        f"var_assim.models.{args.model}.adjoint"
    ).get_adj_path

    # necessary check for cost gradient check
    cost = importlib.import_module(f"var_assim.models.{args.model}.cost").cost
    grad = importlib.import_module(f"var_assim.models.{args.model}.cost").grad

    # define upper and lower perturbation amounts
    ALPHA_MIN = 1e-16
    ALPHA_MAX = 1.0

    # check to see if parent directory for data saving exists; if it doesn't, make it
    check_dir = DATA_DIR / "checks" / args.model
    check_dir.mkdir(parents=True, exist_ok=True)

    # do each check: tlm, adj, and cost function gradient
    _do_tlm_check(
        args,
        get_nonlin_path,
        get_tlm_path,
        e,
        controls,
        TMIN,
        TMAX,
        DT,
        ALPHA_MIN,
        ALPHA_MAX,
    )

    _do_adj_id_check(
        args,
        get_nonlin_path,
        get_tlm_path,
        get_adj_path,
        e,
        controls,
        TMIN=TMIN,
        TMAX=TMAX,
        DT=DT,
    )

    _do_grad_cost_check(
        args, cost, grad, controls * 1.1, cost_args, ALPHA_MIN, ALPHA_MAX
    )


def _do_tlm_check(
    args,
    get_nonlin_path,
    get_tlm_path,
    e,
    controls,
    TMIN,
    TMAX,
    DT,
    ALPHA_MIN,
    ALPHA_MAX,
):
    """Tangent linear model check.

    This verifies the tangent linear model following the prescription of
    p. 89 of Priciples of Data Assimilation by Park and Zupanski.

    Parameters
    ----------
    args: argparse.Namespace
        CLI input for simulations

    get_nonlin_path: callable
        function from model.dynamics, integrates nonlinear model

    get_tlm_path: callable
        function from model.dynamics, integrates the tangent linear model

    theta: list
        parameters

    e: `EmissionsBaseline` object
        the emissions baseline we're considering

    controls: list
        control vector for integration

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
    """

    if ALPHA_MIN > ALPHA_MAX:
        raise ValueError(
            "Perturbation size minimum is greater than maximum. Adjust alpha_min and alpha_max parameters."
        )

    oom_diff = int(np.log10(ALPHA_MAX) - np.log10(ALPHA_MIN))
    alphas = [ALPHA_MAX / 10**i for i in range(abs(oom_diff))]

    # empty array for norms to be stored in
    norms = []

    # get TLM and nonlinear paths without perturbations
    data_base_p, _ = get_nonlin_path(e, controls, TMIN, TMAX, DT)
    tlm_p = get_tlm_path(e, controls, TMIN, TMAX, DT, data_base_p)

    tlm_norm_unscaled = np.sqrt(np.sum(tlm_p**2))

    for a in alphas:
        # get perturbed path, base path, and tangent linear path
        data_pert_p, _ = get_nonlin_path(e, controls * (1 + a), TMIN, TMAX, DT)

        # compute norms
        # NOTE: i'm using L2 norms because the book doesn't specify what norm
        # to take
        tlm_norm = a * tlm_norm_unscaled
        pert_norm = np.sqrt(np.sum((data_pert_p - data_base_p) ** 2))

        # append ratio of two norms
        norms.append(pert_norm / tlm_norm)

    # make dataframe of output and return
    data = {
        "perturbation_size": alphas,
        "norms": norms,
        "log(|norms - 1|)": np.log10(np.abs(np.array(norms) - 1)),
    }
    df = pd.DataFrame(data)

    # save output to csv
    datapath = DATA_DIR / "checks" / args.model / "tlm_check.csv"

    df.to_csv(datapath, sep=",", index=False)


def _do_adj_id_check(
    args, get_nonlin_path, get_tlm_path, get_adj_path, e, controls, TMIN, TMAX, DT
):
    """Adjoint identity check.

    This verifies the adjoint identity equation following the prescription of
    p. 94 of Priciples of Data Assimilation by Park and Zupanski.

    Parameters
    ----------
    args: argparse.Namespace
        CLI input for simulations

    get_nonlin_path: callable
        function from model.dynamics, integrates nonlinear model

    get_tlm_path: callable
        function from model.dynamics, integrates the tangent linear model

    get_adj_path: callable
        function from model.adjoint, integrates adjoint path backwards in time

    controls: list
        controls for simulation

    TMIN: int
        minimum time for verification

    TMIN: int
        maximum time for verification

    DT: float
        time discretization
    """

    # make empty identity list
    id_t = []

    # loop through, taking progressively more timesteps
    # (presumably, the more timesteps you take, the worse the linear
    # approximation does)
    for t in range(1, TMAX - TMIN + 1):
        # get TLM trajectory and full nonlinear trajectory
        nonlin_p, _ = get_nonlin_path(e, controls, TMIN, TMIN + t, DT)
        tlm_p = get_tlm_path(e, controls, TMIN, TMIN + t, DT, nonlin_p)

        # lhs of check
        check_lhs = 0
        for i in range(np.shape(tlm_p)[0]):
            check_lhs += tlm_p[i] @ tlm_p[i]  # tlm.T . tlm

        # get adjoint integration
        adj_p = get_adj_path(
            e,
            controls,
            TMIN,
            t + TMIN,
            DT,
            nonlin_p,
            covars=None,
            obs=None,
            id_check=True,
        )
        check_rhs = 0
        for i in range(np.shape(adj_p)[0]):
            check_rhs += nonlin_p[i] @ adj_p[i]

        id_t.append(check_lhs / check_rhs)

    # make dataframe of output and return
    data = {
        "timesteps taken": [t for t in range(1, TMAX - TMIN + 1)],
        "identity": id_t,
        "log(|identity - 1|)": np.log10(np.abs(np.array(id_t) - 1)),
    }
    df = pd.DataFrame(data)

    # save output to csv
    datapath = DATA_DIR / "checks" / args.model / "adj_id_check.csv"

    df.to_csv(datapath, sep=",", index=False)


def _do_grad_cost_check(args, cost, grad, control, cost_args, ALPHA_MIN, ALPHA_MAX):
    """Cost gradient check function.

    Parameters
    ----------
    args: argparse.Namespace
        CLI input for simulations

    cost: callable
        function from model.cost, evaluates cost function

    grad: callable
        function from model.cost, evaluates cost function gradient using adjoint

    control: list
        control vector

    cost_args: list
        arguments for cost and gradient functions

    ALPHA_MIN: float
        minimum perturbation size

    ALPHA_MAX: float
        maximum perturbation size
    """

    if ALPHA_MIN > ALPHA_MAX:
        raise ValueError(
            "Perturbation size minimum is greater than maximum. Adjust alpha_min and alpha_max parameters."
        )

    oom_diff = int(np.log10(ALPHA_MAX) - np.log10(ALPHA_MIN))
    alphas = [ALPHA_MAX / (10**i) for i in range(oom_diff + 1)]

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
        phi.append(phi_num / phi_denom)

    # make dataframe of output and return
    data = {
        "perturbation_size": alphas,
        "phi": phi,
        "log(phi - 1)": np.log10(abs(np.array(phi) - 1)),
    }
    df = pd.DataFrame(data)

    # save to csv file
    datapath = DATA_DIR / "checks" / args.model / "cost_grad_check.csv"

    df.to_csv(datapath, sep=",", index=False)
