"""Checks on formulation.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.12.2024
"""

import os
import importlib

from datetime import datetime

import numpy as np
import pandas as pd

from var_assim.config import DATA_DIR
from var_assim.logging_utils import get_git_hash, is_git_dirty
from pathlib import Path

# tolerance for the per-block gradient check. observed values: initial
# conditions ~1e-10, parameters ~4e-9, and a genuinely broken model-error block
# O(0.1-1). so this leaves ~25x headroom above the noisiest passing group while
# still sitting six orders of magnitude below anything actually wrong.
GRAD_BLOCK_TOL = 1e-7

# relative finite-difference step for the per-block gradient check. scaled per
# component, since C2 ~ 100 sits in the same control vector as model errors ~0.3
GRAD_BLOCK_FD_STEP = 1e-6


def _get_check_stamp():
    """Build a provenance stamp identifying the code that produced a check.

    A bare timestamp records *when* a check ran, not *what code* ran, which is
    what you need when comparing a check before and after a fix. The git hash
    is the actual identifier; the "-dirty" suffix flags uncommitted edits on
    top of it, and the timestamp only orders repeat runs of the same code.
    """

    dirty = "-dirty" if is_git_dirty() else ""
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    return f"{get_git_hash(short=True)}{dirty}_{timestamp}"


def _write_run_metadata(args, check_dir, stamp, TMIN, TMAX, DT, N_times, n_controls):
    """Record provenance for every check in this directory.

    Written once per check run rather than as columns on each check's .csv, so
    the existing checks' output format is untouched.
    """

    meta = {
        "stamp": stamp,
        "model": args.model,
        "git_sha": get_git_hash(),
        "git_dirty": is_git_dirty(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "scenario": args.scenario,
        "noise_model": args.noise_model,
        "reg_noise": args.reg_noise,
        "windowing": args.windowing,
        "theta": args.theta,
        "ecs": args.ecs,
        "TMIN": TMIN,
        "TMAX": TMAX,
        "DT": DT,
        "N_times": N_times,
        "n_controls": n_controls,
    }

    pd.DataFrame([meta]).to_csv(check_dir / "run_metadata.csv", sep=",", index=False)


def run_component_checks(logger, args, e, controls, TMIN, TMAX, cost_args, DT=1.0):
    """Check components of variational data assimilation.

    Parameters
    ----------
    logger: logging.Logger
        logger for the current experiment

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

    # stamp the output directory with the code that produced it, so a check run
    # before a fix and one after are both kept and can be told apart
    stamp = _get_check_stamp()
    check_dir = DATA_DIR / "checks" / args.model / stamp
    check_dir.mkdir(parents=True, exist_ok=True)

    # number of timesteps, derived exactly as get_nonlin_path derives it
    N_times = len(np.arange(TMIN, TMAX + DT, DT))

    _write_run_metadata(args, check_dir, stamp, TMIN, TMAX, DT, N_times, len(controls))

    logger.info(f"            >>> (FLAGGED) check output stamp: {stamp}")

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
        check_dir,
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
        check_dir=check_dir,
    )

    _do_grad_cost_check(
        args, cost, grad, controls * 1.1, cost_args, ALPHA_MIN, ALPHA_MAX, check_dir
    )

    # per-block gradient check. this is separate from _do_grad_cost_check
    # because that check contracts the gradient into a single scalar along the
    # full normalised gradient direction, where the parameter block dominates
    # and a badly wrong model-error block barely registers. reported per block,
    # the same defect is orders of magnitude more visible.
    _do_grad_block_check(
        logger, args, cost, grad, controls * 1.1, cost_args, check_dir, N_times
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
    check_dir,
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

    check_dir: Path
        stamped directory to write this check's .csv into
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
    datapath = check_dir / "tlm_check.csv"

    df.to_csv(datapath, sep=",", index=False)


def _do_adj_id_check(
    args,
    get_nonlin_path,
    get_tlm_path,
    get_adj_path,
    e,
    controls,
    TMIN,
    TMAX,
    DT,
    check_dir,
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

    check_dir: Path
        stamped directory to write this check's .csv into
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
    datapath = check_dir / "adj_id_check.csv"

    df.to_csv(datapath, sep=",", index=False)


def _do_grad_cost_check(
    args, cost, grad, control, cost_args, ALPHA_MIN, ALPHA_MAX, check_dir
):
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

    check_dir: Path
        stamped directory to write this check's .csv into
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
    datapath = check_dir / "cost_grad_check.csv"

    df.to_csv(datapath, sep=",", index=False)


def _do_grad_block_check(
    logger, args, cost, grad, control, cost_args, check_dir, N_times
):
    """Per-block check of the adjoint gradient against finite differences.

    The control vector splits into initial conditions, parameters, and one
    block of model errors per noise stream, and each group is checked
    independently. This matters because _do_grad_cost_check reduces the
    gradient to one scalar along the full normalised gradient direction: the
    parameter block dominates that norm, so a model-error block can be O(1)
    wrong while the scalar check still reads within ~1e-4 of unity, which is
    indistinguishable from second-order finite-difference error.

    Two errors are reported per block:
        - rel_err: the full gradient, which is the pass/fail criterion
        - rel_err_obs_only: the prior term (analytic, and independent of the
          adjoint) subtracted from both sides, isolating the piece the adjoint
          actually computes. Diagnostic only -- it makes a failure interpretable
          rather than just visible.

    Parameters
    ----------
    logger: logging.Logger
        logger for the current experiment

    args: argparse.Namespace
        CLI input for simulations

    cost: callable
        function from model.cost, evaluates cost function

    grad: callable
        function from model.cost, evaluates cost function gradient using adjoint

    control: list
        control vector to evaluate at

    cost_args: list
        arguments for cost and gradient functions

    check_dir: Path
        stamped directory to write this check's .csv into

    N_times: int
        number of timesteps, i.e. the length of one model-error block
    """

    control = np.asarray(control, dtype=float)
    n_controls = len(control)

    # Work out the control-vector layout, which for every model in this family
    # is three distinct groups:
    #
    #   initial conditions   3 + n_reg      T1, T2, Q, T_R1 ... T_Rn
    #   parameters           6 + 2 * n_reg  L, G, EPS, C1, C2, F1_CO2,
    #                                       ALPHA_R1 ... n, BETA_R1 ... n
    #   model errors         n_blocks * N_times
    #
    # These are reported separately rather than as one lump, so a failure is
    # localised to the initial conditions or the parameters rather than merely
    # to "the non-model-error part". n_reg comes from the observation vector,
    # which is [T1, Q, T_R1 ... T_Rn] for all of these models.
    #
    # NOTE: do NOT infer the split as n_controls // N_times. That is only
    # correct while the parameter count is below N_times, and it fails
    # silently -- producing a plausible but wrong split -- for short windows.
    obs = cost_args[-5]
    n_reg = np.shape(obs)[0] - 2

    n_ics = 3 + n_reg
    n_pars = 6 + 2 * n_reg

    n_q = n_controls - n_ics - n_pars

    if n_q < 0 or n_q % N_times != 0:
        raise ValueError(
            f"Control-vector layout for {args.model} does not match "
            f"expectations: {n_controls} controls with n_reg={n_reg} implies "
            f"{n_ics} initial conditions, {n_pars} parameters, and {n_q} model "
            f"errors, which is not a whole number of blocks of N_times={N_times}."
        )

    n_blocks = n_q // N_times

    # 0 blocks for a deterministic model, 1 for global noise only, and one per
    # region on top of that when regional noise is switched on
    if n_blocks not in (0, 1, 1 + n_reg):
        raise ValueError(
            f"Unexpected number of model-error blocks for {args.model}: "
            f"got {n_blocks}, expected 0, 1, or {1 + n_reg}."
        )

    # model-error block 0 is always the global (GMST) noise; any remaining
    # blocks are regional, matching the runners' var_names ordering
    block_names = ["ics", "params"]
    if n_blocks >= 1:
        block_names.append("qAT")
    block_names += [f"qR{i}" for i in range(1, n_blocks)]

    n_fixed = n_ics + n_pars
    block_bounds = [(0, n_ics), (n_ics, n_fixed)] + [
        (n_fixed + b * N_times, n_fixed + (b + 1) * N_times) for b in range(n_blocks)
    ]

    # prior term of the gradient. this is analytic and does not involve the
    # adjoint, so removing it isolates the observation piece
    x_f, inv_covar_prior = cost_args[0], cost_args[1]
    prior_grad = inv_covar_prior @ (control - np.asarray(x_f, dtype=float))

    # adjoint gradient
    grad_adj = np.asarray(grad(control, cost_args), dtype=float)

    # central finite-difference gradient, with a per-component step: the
    # control vector mixes O(100) heat capacities with O(0.1) model errors, so
    # a single absolute step would be badly scaled for one end or the other
    grad_fd = np.empty(n_controls)
    for j in range(n_controls):
        step = GRAD_BLOCK_FD_STEP * max(1.0, abs(control[j]))

        control_up, control_dn = control.copy(), control.copy()
        control_up[j] += step
        control_dn[j] -= step

        grad_fd[j] = (cost(control_up, cost_args) - cost(control_dn, cost_args)) / (
            2.0 * step
        )

    def _rel_err(approx, exact):
        """Relative L2 error, guarding against an all-zero reference block."""
        denom = np.linalg.norm(exact)
        if denom == 0.0:
            return 0.0 if np.linalg.norm(approx) == 0.0 else np.inf
        return np.linalg.norm(approx - exact) / denom

    rows = []
    for name, (lo, hi) in zip(block_names, block_bounds):
        rel_err = _rel_err(grad_adj[lo:hi], grad_fd[lo:hi])
        rel_err_obs = _rel_err(
            grad_adj[lo:hi] - prior_grad[lo:hi], grad_fd[lo:hi] - prior_grad[lo:hi]
        )
        passed = bool(rel_err < GRAD_BLOCK_TOL)

        rows.append(
            {
                "block": name,
                "i_start": lo,
                "i_end": hi,
                "n": hi - lo,
                "rel_err": rel_err,
                "rel_err_obs_only": rel_err_obs,
                "max_abs_err": float(np.max(np.abs(grad_adj[lo:hi] - grad_fd[lo:hi]))),
                "grad_norm_adj": float(np.linalg.norm(grad_adj[lo:hi])),
                "grad_norm_fd": float(np.linalg.norm(grad_fd[lo:hi])),
                "tol": GRAD_BLOCK_TOL,
                "pass": passed,
            }
        )

        logger.info(
            f"            >>> (FLAGGED) grad block {name:6s} "
            f"[{lo:4d}:{hi:4d}]  rel err = {rel_err:.3e}  "
            f"(obs only {rel_err_obs:.3e})  {'pass' if passed else 'FAIL'}"
        )

    df = pd.DataFrame(rows)

    n_failed = int((~df["pass"]).sum())
    if n_failed:
        logger.warning(
            f"            >>> (FLAGGED) per-block gradient check FAILED for "
            f"{n_failed}/{len(df)} blocks "
            f"({', '.join(df.loc[~df['pass'], 'block'])}) "
            f"at tol={GRAD_BLOCK_TOL:g}"
        )
    else:
        logger.info(
            f"            >>> (FLAGGED) per-block gradient check passed for all "
            f"{len(df)} blocks at tol={GRAD_BLOCK_TOL:g}"
        )

    # save to csv file
    datapath = check_dir / "grad_block_check.csv"

    df.to_csv(datapath, sep=",", index=False)
