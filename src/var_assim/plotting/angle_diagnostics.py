"""Diagnostics for the SAI inequality angle (vartheta) recovered by the
assimilation.

Motivation: in the `pco2geowc_reg` / `four` runs the ensemble-median posterior
angle at the *first* window (2040) sits at 16.5-16.9 degrees for every true
angle, whether the truth is 5 or 35 degrees -- roughly +1.2 degrees above the
prior angle median. The fan-out toward the truth after 2040 is expected; the
common offset at 2040 is not. These functions exist to attribute that offset to
individual control variables and to say whether it is the correct Bayesian
response to the single realization of internal variability that every run is
handed.

The angle itself is computed by `get_angle_r2` in `pproc.py`; nothing here
duplicates it.

Adam Michael Bauer
UChicago
"""

import argparse
import gc
import importlib

import numpy as np
import netCDF4 as nc

from var_assim.config import (
    DATA_DIR_ABS,
    NOISE_PATH,
    PRIOR_PATH,
    TRUTH_PATH,
    WINDOW_PATH,
)
from var_assim.calibration.noise import ClimateModelNoise
from var_assim.calibration.priors import ClimateModelPriors
from var_assim.calibration.truth import ClimateModelTruth
from var_assim.calibration.windowing import AssimilationWindowing
from var_assim.models import MODEL_REGISTRY
from var_assim.stats.covar import get_covar_ar1, get_covar_white
from var_assim.plotting.pproc import get_angle_r2

# the 15 non-model-error entries of the control vector, in the order
# `dynamics.get_nonlin_path` unpacks them
CONTROL_HEAD = [
    "T1",
    "T2",
    "Q",
    "T_R1",
    "T_R2",
    "L",
    "G",
    "EPS",
    "C1",
    "C2",
    "F1_CO2",
    "ALPHA_R1",
    "ALPHA_R2",
    "BETA_R1",
    "BETA_R2",
]
N_FIXED = len(CONTROL_HEAD)

# the seven controls `get_angle_r2` actually reads, in its argument order
ANGLE_PARAMS = ["ALPHA_R1", "ALPHA_R2", "BETA_R1", "BETA_R2", "L", "EPS", "G"]

# observation vector, and the state rows of `data_final` each entry comes from
# (`obs.get_obs_from_dynamics` selects paths[[0, 2, 3, 4]])
OBS_VARS = ["T1", "Q", "T_R1", "T_R2"]
OBS_STATE_ROWS = [0, 2, 3, 4]

# model-error blocks, in control-vector order after the 15 fixed entries
ERROR_BLOCKS = ["qAT", "qR1", "qR2"]

# forcing efficacy of SAI. hard-coded to match both `get_angle_r2`'s default and
# `F_EFF_GEO_TR` in truth.yaml; asserted against the calibration in
# `build_calibration` so the two cannot drift apart silently.
PHI = 0.09

# commit 85a2a91 ("observational noise weighting in cost function") added the
# obs_weighting block. Runs written before it used the legacy flat sigma = 1.0
# and would be re-weighted incorrectly by this module, so its timestamp is the
# provenance cutoff. Nothing in the netCDF records which weighting was used.
# verify with: git log -1 --format=%ct 85a2a91
OBS_WEIGHTING_COMMIT_EPOCH = 1790033894  # 2026-09-21 18:38:14 -0500


def model_q_offset(model):
    """Leading timesteps with no model error, per `MODEL_REGISTRY`.

    Each model-error block has `n_times - q_offset` entries: 0 for
    `pco2geowc_reg`, 1 for `pco2geowc_reg_noic`.
    """

    return MODEL_REGISTRY[model].get("q_offset", 0)


def model_dynamics(model):
    """`(get_nonlin_path, get_obs_from_dynamics)` for the named model."""

    dyn = importlib.import_module(f"var_assim.models.{model}.dynamics")
    obs = importlib.import_module(f"var_assim.models.{model}.obs")
    return dyn.get_nonlin_path, obs.get_obs_from_dynamics


def make_cli_namespace(theta, **overrides):
    """Build the `argparse.Namespace` the calibration dataclasses expect.

    The dataclasses are normally fed the real CLI namespace. Reconstructing one
    here is what lets the prior and observation-error covariances -- which are
    not archived in the output files -- be rebuilt at analysis time.

    Parameters
    ----------
    theta: int
        true angle, i.e. the `--theta` lookup key into truth.yaml

    **overrides
        any namespace field to override; defaults match the in-scope runs
        (ssp245, DEGpDEC 0.1, `four` windowing, Nens 1000)

    Returns
    -------
    args: argparse.Namespace
    """

    # both spellings are populated because the dataclasses disagree about
    # casing: ClimateModelNoise/Priors read `noise_model`/`reg_noise`, while
    # ClimateModelTruth reads `ecs`/`theta`. Supplying both avoids the
    # per-dataclass namespace shims the notebooks currently carry.
    fields = dict(
        model="pco2geowc_reg",
        scenario="ssp245",
        ssp="ssp245",
        noise_model="AR1",
        tmin=2025,
        AR=1,
        theta=int(theta),
        ecs=3.0,
        ECS=3.0,
        deg_p_dec=0.1,
        DEGpDEC="0.1",
        n_yrs_ramp=50,
        NYRSRAMP=50,
        n_ens=1000,
        Nens=1000,
        reg_noise=True,
        windowing="four",
        sai_ramp="linear",
        debug=False,
    )
    fields.update(overrides)
    return argparse.Namespace(**fields)


def build_calibration(theta, **overrides):
    """Rebuild the calibration objects for one run from `config/`.

    Returns
    -------
    args, Noise, Prior, Truth, Windowing
    """

    args = make_cli_namespace(theta, **overrides)

    Noise = ClimateModelNoise.from_cli_and_yaml(args, NOISE_PATH)
    Prior = ClimateModelPriors.from_cli_and_yaml_and_noise(args, PRIOR_PATH, Noise)
    Truth = ClimateModelTruth.from_cli_and_yaml(args, TRUTH_PATH)
    Windowing = AssimilationWindowing.from_cli_and_yaml(args, WINDOW_PATH)

    # `get_angle_r2` hard-codes phi rather than reading the calibration, so
    # check the two agree before any angle is computed with it
    if not np.isclose(Truth.F_EFF_GEO_TR, PHI):
        raise ValueError(
            f"F_EFF_GEO_TR={Truth.F_EFF_GEO_TR} disagrees with the phi={PHI} "
            "hard-coded in get_angle_r2; the angle would be wrong."
        )

    # the warm start is deliberately not run. it only overwrites the T1/T2/Q/
    # T_REG prior *centres*, and this module never needs them: the prior
    # standard deviations (hence B) are untouched by it, and where a prior
    # centre is needed it is taken from the saved ensemble instead (see
    # `prior_centre_from_ensemble`). That keeps the diagnostics independent of
    # the warm start reproducing bit-for-bit.
    return args, Noise, Prior, Truth, Windowing


def output_path(theta, **overrides):
    """Path of the saved assimilation output for one true angle.

    Mirrors the filename template in `postprocessing.make_master_datatree`.
    """

    a = make_cli_namespace(theta, **overrides)
    return (
        DATA_DIR_ABS
        / "output"
        / a.model
        / (
            f"var-assim-output_{a.scenario}_{a.model}_{a.windowing}_{a.noise_model}+reg"
            f"_TMIN{a.tmin}_THETA{a.theta}_ECS{a.ecs}"
            f"_ramprate{a.sai_ramp}_DEGpDEC{a.deg_p_dec}"
            f"_NYRSRAMP{a.n_yrs_ramp}_Nens{a.n_ens}.nc"
        )
    )


def check_provenance(paths, Noise):
    """Fail loudly if the inputs were not produced with the derived weighting.

    The observation-error standard deviations are rebuilt from `config/noise.yaml`
    at analysis time, but the output files record no weighting provenance. A run
    predating commit 85a2a91 used the legacy flat sigma = 1.0 and would be
    silently re-weighted here, so both the current config and every file's mtime
    are checked.
    """

    if Noise.OBS_WEIGHTING != "derived":
        raise ValueError(
            f"Noise.OBS_WEIGHTING is {Noise.OBS_WEIGHTING!r}, expected 'derived'. "
            "config/noise.yaml no longer has an obs_weighting entry for this "
            "model, so the rebuilt covariances would not match the runs."
        )

    stale = []
    for p in paths:
        mtime = p.stat().st_mtime
        if mtime < OBS_WEIGHTING_COMMIT_EPOCH:
            stale.append((p.name, mtime))

    if stale:
        listing = "\n".join(f"  {n} (mtime {m})" for n, m in stale)
        raise ValueError(
            "These files predate commit 85a2a91 and therefore used the legacy "
            f"flat obs weighting, not the derived weighting:\n{listing}"
        )


def obs_sigmas(Noise):
    """Observation-error standard deviations, in `OBS_VARS` order.

    `runner.py` builds one diagonal, time-constant inverse covariance per
    observable, so the whole observation term reduces to a per-observable sigma.
    """

    return np.array(
        [
            Noise.OBS_T1_STD,
            Noise.OBS_Q_STD,
            Noise.OBS_T_REG_STD[0],
            Noise.OBS_T_REG_STD[1],
        ]
    )


def build_inv_covar_prior(Prior, Noise, n_times, q_offset=0):
    """Rebuild B^-1 for a window of `n_times` steps.

    Reproduces `pco2geowc_reg/runner.py:204-222`: white everywhere except the
    global model-error block, which carries the inverse of the dense AR(1)
    covariance. The runner slices that block out of a longer matrix, but the
    AR(1) covariance is Toeplitz so rebuilding it at `n_times` is exact.

    `q_offset` (see `model_q_offset`) shortens each model-error block to
    `n_times - q_offset` entries; Toeplitz again makes the rebuild exact.
    """

    n_q = n_times - q_offset
    regional_stds = np.hstack([[s] * n_q for s in Noise.INT_T_REG_STD])
    # the global block's entries are placeholders -- overwritten below
    full_std_vector = np.hstack([np.ones(n_q), regional_stds])
    prior_stds = Prior.get_augmented_std_vector(full_std_vector)

    inv_covar_prior = get_covar_white(prior_stds, len(prior_stds), inv=True)

    mod_error_covar = get_covar_ar1(Noise.INT_VAR_STD, Noise.AUTO_CORR, n_q)
    inv_covar_prior[N_FIXED : N_FIXED + n_q, N_FIXED : N_FIXED + n_q] = (
        np.linalg.inv(mod_error_covar)
    )

    return inv_covar_prior, prior_stds


def window_names(path):
    """Window group names in ascending window order.

    Read from the file rather than from `config/windowing.yaml` so a run is
    described by its own contents, which also catches a windowing scheme that
    has been edited since the run.
    """

    ds = nc.Dataset(path)
    try:
        return sorted(ds.groups, key=int)
    finally:
        ds.close()


def load_window(path, window, want_data_final=False):
    """Read one window group, pulling only the slices the diagnostics need.

    `data_hist` is ~7.5 GB per group and is never touched. `controls_hist` is
    ~100 MB per group but only `iter=0` is wanted, and `data_final` is only
    wanted for the four observable rows -- so this reads through netCDF4 rather
    than opening the whole group, which is what `drop_variables` alone cannot do.

    Parameters
    ----------
    path: Path
        output file

    window: str
        window group name, e.g. "2040"

    want_data_final: bool
        also read the four observable rows of `data_final` (needed for the cost
        breakdown, skipped otherwise)

    Returns
    -------
    out: dict
        arrays keyed by variable name, plus `vari` (list of str) and `head`
        (indices of `CONTROL_HEAD` within `vari`)
    """

    ds = nc.Dataset(path)
    try:
        g = ds.groups[str(window)]
        vari = [str(v) for v in g.variables["vari"][:]]
        head = [vari.index(n) for n in CONTROL_HEAD]

        out = {
            "vari": vari,
            "head": head,
            "time": np.asarray(g.variables["time"][:]),
            "controls": np.asarray(g.variables["controls"][:, :]),
            "controls_truth": np.asarray(g.variables["controls_truth"][:]),
            "obs": np.asarray(g.variables["obs"][:, :]),
            "cost_hist": np.asarray(g.variables["cost_hist"][:, :]),
            "costs": np.asarray(g.variables["costs"][:]),
        }
        # iter=0 is the prior draw: `EnsembleMember` seeds
        # controls_hist[:, 0] = theta_p, and theta_p is also the pinned
        # background x_f passed to cost()
        out["controls_prior"] = np.asarray(g.variables["controls_hist"][:, :, 0])

        if want_data_final:
            out["obs_pred"] = np.asarray(
                g.variables["data_final"][:, OBS_STATE_ROWS, :]
            )
    finally:
        ds.close()
        gc.collect()

    return out


def screen_members(cost_hist, ratio=0.5):
    """Boolean keep-mask over ensemble members.

    Screens on the final/initial cost ratio, as `filtering.filter_by_cost_ratio_memeff`
    does, and additionally drops non-finite members. A handful of members blow
    up (costs up to 1e92, `data_final` up to 1e45), and those alone would wreck
    every ensemble statistic.

    Note that `cost_hist[:, 0]` is the *observation term only*: at iteration 0
    the control equals the background, so the prior term is identically zero.
    The ratio is therefore not a clean convergence measure -- it is used here
    only to remove blow-ups, which is all it reliably does.
    """

    first = cost_hist[:, 0]
    last = cost_hist[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = last / first
    return np.isfinite(first) & np.isfinite(last) & np.isfinite(r) & (r <= ratio)


def angle_from_controls(head_block):
    """Angle for each row of an (n, 15) block of fixed controls.

    `get_angle_r2` builds a column vector internally and so returns a length-1
    array per call; this loops and unwraps rather than reimplementing it.
    """

    cols = {n: head_block[:, CONTROL_HEAD.index(n)] for n in ANGLE_PARAMS}
    out = np.empty(head_block.shape[0])
    for i in range(head_block.shape[0]):
        out[i] = np.ravel(
            get_angle_r2(
                cols["ALPHA_R1"][i],
                cols["ALPHA_R2"][i],
                cols["BETA_R1"][i],
                cols["BETA_R2"][i],
                cols["L"][i],
                cols["EPS"][i],
                cols["G"][i],
            )
        )[0]
    return out


def angle_at(values):
    """Angle for a single dict of the seven `ANGLE_PARAMS`."""

    return float(
        np.ravel(
            get_angle_r2(
                values["ALPHA_R1"],
                values["ALPHA_R2"],
                values["BETA_R1"],
                values["BETA_R2"],
                values["L"],
                values["EPS"],
                values["G"],
            )
        )[0]
    )


def angle_sensitivities(base, stds, rel_step=1e-4):
    """d(angle)/dp * sigma_p for each angle parameter, by central difference.

    Returned in degrees per prior standard deviation, which is the unit that
    makes a shift in `BETA_R2` comparable to one in `L`.

    Parameters
    ----------
    base: dict
        the seven `ANGLE_PARAMS` to linearize about

    stds: dict
        prior standard deviation of each

    Returns
    -------
    sens: dict
        degrees of angle per prior sigma of each parameter
    """

    sens = {}
    for p in ANGLE_PARAMS:
        h = stds[p] * rel_step
        hi, lo = dict(base), dict(base)
        hi[p] = base[p] + h
        lo[p] = base[p] - h
        sens[p] = (angle_at(hi) - angle_at(lo)) / (2.0 * h) * stds[p]
    return sens


def attribute_angle_offset(prior_med, post_med, stds):
    """First-order attribution of an angle offset to each angle parameter.

    Parameters
    ----------
    prior_med, post_med: dict
        median of each angle parameter, prior and posterior

    stds: dict
        prior standard deviations

    Returns
    -------
    out: dict
        `shift_sigma`, `contrib_deg` and `sensitivity` per parameter, plus the
        exact offset `offset_exact`, the first-order sum `offset_linear`, and
        the closure residual. A small residual is what licenses reading the
        per-parameter contributions as independent causes.
    """

    sens = angle_sensitivities(prior_med, stds)

    shift_sigma = {p: (post_med[p] - prior_med[p]) / stds[p] for p in ANGLE_PARAMS}
    contrib = {p: shift_sigma[p] * sens[p] for p in ANGLE_PARAMS}

    exact = angle_at(post_med) - angle_at(prior_med)
    linear = float(sum(contrib.values()))

    return {
        "shift_sigma": shift_sigma,
        "contrib_deg": contrib,
        "sensitivity": sens,
        "offset_exact": exact,
        "offset_linear": linear,
        "residual": exact - linear,
    }


def substitute_angle(prior_med, post_med, group):
    """Angle with posterior values for `group` and prior values elsewhere.

    The exact (non-linearized) counterpart to `attribute_angle_offset`, so that
    nonlinear interaction between parameter groups can be seen rather than
    assumed away.
    """

    mix = dict(prior_med)
    for p in group:
        mix[p] = post_med[p]
    return angle_at(mix)


ANGLE_GROUPS = {
    "ALPHA": ["ALPHA_R1", "ALPHA_R2"],
    "BETA": ["BETA_R1", "BETA_R2"],
    "LGE": ["L", "G", "EPS"],
}


def cost_breakdown(win, keep, inv_covar_prior, sigmas):
    """Decompose J into its prior and per-observation terms.

    `cost()` is a background quadratic plus four diagonal-weighted observation
    quadratics with no cross terms, so the split is exact:

        J = 0.5 (x - x_f)' B^-1 (x - x_f)
          + 0.5 sum_k sum_t (H_k x - y_k)_t^2 / sigma_k^2

    Parameters
    ----------
    win: dict
        output of `load_window(..., want_data_final=True)`

    keep: ndarray of bool
        ensemble screen

    inv_covar_prior: ndarray
        B^-1 for this window

    sigmas: ndarray
        observation-error sigmas in `OBS_VARS` order

    Returns
    -------
    out: dict
        per-member `J_prior`, `J_obs_<var>`, `J_total`, the prior term split by
        control block, the reduced chi-square of each observation term,
        `J_truth`, and `max_rel_residual` against the archived `costs`
    """

    x = win["controls"][keep]
    x_f = win["controls_prior"][keep]
    d = x - x_f

    J_prior = 0.5 * np.einsum("ij,jk,ik->i", d, inv_covar_prior, d)

    n_times = len(win["time"])
    # model-error block length, inferred from the control vector so it holds
    # for models with no t = 0 model error too
    n_q = (x.shape[1] - N_FIXED) // len(ERROR_BLOCKS)
    out = {"J_prior": J_prior}

    # split the prior term by control block. the 15-parameter head and the two
    # regional error blocks are diagonal; the global qAT block is not, so it is
    # handled as a full quadratic form over its own sub-block.
    blocks = {"head": slice(0, N_FIXED)}
    for b, name in enumerate(ERROR_BLOCKS):
        s = N_FIXED + b * n_q
        blocks[name] = slice(s, s + n_q)

    for name, sl in blocks.items():
        sub = inv_covar_prior[sl, sl]
        out[f"J_prior_{name}"] = 0.5 * np.einsum("ij,jk,ik->i", d[:, sl], sub, d[:, sl])

    # observation terms
    pred = win["obs_pred"][keep]  # (n_keep, 4, n_times)
    J_obs_total = np.zeros(pred.shape[0])
    for k, name in enumerate(OBS_VARS):
        resid = pred[:, k, :] - win["obs"][k][None, :]
        Jk = 0.5 * np.sum(resid**2, axis=1) / sigmas[k] ** 2
        out[f"J_obs_{name}"] = Jk
        # reduced chi-square: ~1 means the assumed sigma matches the misfit,
        # >>1 means the observable cannot be fit at the sigma it was given
        out[f"chi2red_{name}"] = 2.0 * Jk / n_times
        J_obs_total += Jk

    out["J_obs"] = J_obs_total
    out["J_total"] = J_prior + J_obs_total

    # the pseudo-observations are the noiseless truth path, so the observation
    # term vanishes identically at the truth and J(truth) is the background term
    # alone -- no forward model run required
    d_tr = win["controls_truth"][None, :] - x_f
    out["J_truth"] = 0.5 * np.einsum("ij,jk,ik->i", d_tr, inv_covar_prior, d_tr)

    # validation gate: the recomputed total must reproduce the archived cost
    archived = win["costs"][keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(out["J_total"] - archived) / np.abs(archived)
    out["rel_residual"] = rel
    out["max_rel_residual"] = float(np.nanmax(rel))

    return out


def obs_jacobian(e, x0, t_min, t_max, dt=1.0, rel_step=1e-6, model="pco2geowc_reg"):
    """Jacobian of the observables with respect to the control vector.

    Finite-differenced through `get_nonlin_path` rather than assembled from the
    tangent linear model: the TLM exists (`dynamics.get_TLM_matrix`) but building
    the full observation Jacobian from it means chaining it by hand, whereas the
    model is cheap enough (a ~76-step Python loop) that 2*n_controls forward runs
    is both faster to write and independently checks the TLM.

    Returns
    -------
    H: ndarray, (4 * n_times, n_controls)
        rows ordered observable-major, i.e. all T1 times, then all Q times, ...
    """

    get_nonlin_path, get_obs_from_dynamics = model_dynamics(model)

    def observe(x):
        paths, _ = get_nonlin_path(e, x, t_min, t_max, dt)
        return get_obs_from_dynamics(paths).ravel()

    n = len(x0)
    base = observe(x0)
    H = np.empty((base.size, n))

    # scale each step to the control's own magnitude, with a floor so that
    # controls sitting at zero (the model errors, at the prior centre) still get
    # a finite perturbation
    steps = np.maximum(np.abs(x0) * rel_step, rel_step)

    for j in range(n):
        hi = x0.copy()
        lo = x0.copy()
        hi[j] += steps[j]
        lo[j] -= steps[j]
        H[:, j] = (observe(hi) - observe(lo)) / (2.0 * steps[j])

    return H


def gauss_newton_system(H, inv_covar_prior, sigmas, n_times):
    """Gauss-Newton Hessian of J and the observation precision.

    A = B^-1 + H' R^-1 H is the curvature of J at the linearization point. Its
    inverse is the linear-Gaussian posterior covariance, which is what makes
    "predicted spread" comparable to the measured ensemble spread, and its
    small-eigenvalue directions are the ones the observations cannot constrain.

    Returns
    -------
    A: ndarray
        the Hessian

    Rinv_diag: ndarray, (4 * n_times,)
        diagonal of R^-1, matching `H`'s row order
    """

    Rinv_diag = np.repeat(1.0 / sigmas**2, n_times)
    A = inv_covar_prior + H.T @ (H * Rinv_diag[:, None])
    return A, Rinv_diag


def kalman_shift(H, A, Rinv_diag, innovation):
    """Linear-Gaussian posterior-mean shift from the prior centre.

        dx = (B^-1 + H' R^-1 H)^-1 H' R^-1 (y - H x_cen)

    The assimilation is randomize-then-optimize: every member draws its own
    background from the prior and minimizes J against the *same* observations.
    In the linear-Gaussian limit the ensemble-mean posterior therefore sits at
    the prior centre plus this shift. Comparing it to the measured median shift
    is what distinguishes a correct Bayesian response to this particular noise
    realization from an optimizer failure.

    Parameters
    ----------
    innovation: ndarray, (4 * n_times,)
        y - H(x_cen), with rows in `H`'s order
    """

    rhs = H.T @ (Rinv_diag * innovation)
    return np.linalg.solve(A, rhs)


def prior_centre_from_ensemble(win, keep=None):
    """Prior centre, estimated as the mean of the saved prior draws.

    Used instead of `Prior.get_augmented_cen_vector` so the diagnostics do not
    depend on re-running the warm start, which is what sets the T1/T2/Q/T_REG
    centres. With 1000 draws the sampling error is sigma/sqrt(1000) ~ 0.03
    sigma, well below the shifts being attributed.

    The full ensemble is used by default, not the screened subset: the screen
    conditions on the optimization outcome, which would bias the estimate of the
    distribution the members were drawn *from*.
    """

    draws = win["controls_prior"] if keep is None else win["controls_prior"][keep]
    return draws.mean(axis=0)


def realization_stats(controls_truth, time, n_times=None):
    """Per-block statistics of the one internal-variability realization.

    Every run shares the same seeds, so the truth model errors stored in
    `controls_truth` are the same realization in all six files. Its mean and
    trend over a window are what the assimilation has to absorb, and at short
    windows it cannot average out.

    The block length is inferred from `controls_truth`, so models with no t = 0
    model error (blocks one shorter than `time`) are handled; `n_times` is
    accepted for backward compatibility and ignored.

    Returns
    -------
    out: dict
        per block: the series, the times it applies at, its mean, std and
        per-year trend
    """

    n_q = (len(controls_truth) - N_FIXED) // len(ERROR_BLOCKS)
    # a missing t = 0 entry drops the *leading* times
    q_time = np.asarray(time)[len(time) - n_q :]

    out = {}
    for b, name in enumerate(ERROR_BLOCKS):
        s = N_FIXED + b * n_q
        q = controls_truth[s : s + n_q]
        out[name] = {
            "series": q,
            "time": q_time,
            "mean": float(q.mean()),
            "std": float(q.std()),
            "trend": float(np.polyfit(np.arange(len(q)), q, 1)[0]),
            "first": float(q[0]),
        }
    return out


def internal_variability_sigmas(Noise):
    """Assumed observation error against the internal variability it represents.

    The observations are the noiseless truth path, which already contains one
    realization of internal variability. If sigma_obs is smaller than that
    variability, every member is required to fit the realization more closely
    than its own noise level warrants -- and since all members see the same
    realization, the resulting parameter shift is common-mode rather than
    spread-inflating.

    Returns
    -------
    rows: list of (name, sigma_obs, sigma_internal, overweight_factor)
    """

    return [
        (
            "T1",
            Noise.OBS_T1_STD,
            Noise.INT_VAR_STD,
            Noise.INT_VAR_STD / Noise.OBS_T1_STD,
        ),
        (
            "T_R1",
            Noise.OBS_T_REG_STD[0],
            Noise.INT_T_REG_STD[0],
            Noise.INT_T_REG_STD[0] / Noise.OBS_T_REG_STD[0],
        ),
        (
            "T_R2",
            Noise.OBS_T_REG_STD[1],
            Noise.INT_T_REG_STD[1],
            Noise.INT_T_REG_STD[1] / Noise.OBS_T_REG_STD[1],
        ),
    ]


def geo_information_split(e, time, ramp_end):
    """sum_t geo(t)^2 over a window, split into ramp and plateau years.

    `BETA_R2` is only identifiable through `BETA_R2 * geo(t)`. Once the SAI ramp
    levels off that product is a constant, degenerate with the `T_R2` initial
    condition and with a mean shift in the free `qR2` model errors -- so plateau
    years add far less information about `BETA_R2` than the raw sum suggests.
    Reporting the split is what shows why learning stalls after the ramp ends.
    """

    geo = np.asarray(e.emis["geo"])[: len(time)]
    ramp = np.asarray(time) <= ramp_end
    return {
        "total": float(np.sum(geo**2)),
        "ramp": float(np.sum(geo[ramp] ** 2)),
        "plateau": float(np.sum(geo[~ramp] ** 2)),
        "n_ramp": int(ramp.sum()),
        "n_plateau": int((~ramp).sum()),
        "geo": geo,
    }


def median_dict(head_block, names=ANGLE_PARAMS):
    """Column medians of a (n, 15) control head, keyed by control name."""

    return {n: float(np.median(head_block[:, CONTROL_HEAD.index(n)])) for n in names}


def std_dict(head_block, names=ANGLE_PARAMS):
    """Column standard deviations of a (n, 15) control head, keyed by name."""

    return {n: float(np.std(head_block[:, CONTROL_HEAD.index(n)])) for n in names}
