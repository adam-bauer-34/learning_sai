"""Cost function and gradient for the strong-constraint model.

For ensemble member i, with d = model - data and v = S^{-1} d:

    marginal  J = 1/2 |theta - theta_f|_B^2 + 1/2 d^T S(theta)^{-1} d + 1/2 log det S(theta)
              data = y + meas_i - L(theta_ref) xi_f       (fixed perturbation)
    profile   J = 1/2 |theta - theta_f|_B^2 + 1/2 d^T S(theta)^{-1} d
              data = y + meas_i - L(theta) xi_f           (moves with theta)
    fixed     J = 1/2 |theta - theta_f|_B^2 + 1/2 d^T S(theta_ref)^{-1} d
              data = y + meas_i - L(theta_ref) xi_f

`profile` is exactly the weak-constraint cost with the noise eliminated, so for
the flux design it reproduces pco2geowc_reg_noic's parameter estimates.

Gradient. All 15 controls are estimated, and S depends on seven of them, so
dS/dtheta must be carried through. It is, exactly, by the envelope identity:

    d/dtheta [1/2 d^T S^{-1} d] = v^T d/dtheta [ y(theta, xi*) ]  at fixed xi*

with xi* = xi_f - Q L^T v (profile), -Q L^T v (marginal), 0 (fixed). The
-1/2 v^T dS v term is absorbed by running the forced trajectory at xi*, so the
data-term gradient is one adjoint integration of that trajectory, forced by v.
Marginal adds 1/2 tr(S^{-1} dS) = tr(Q L^T S^{-1} dL).

`args` keeps x_f at [0], the prior inverse covariance at [1] and obs at [-5], so
the generic checks in tlm_adj_checks work unchanged:

    args = [x_f, inv_covar_prior, sc_ctx, obs, e, T_MIN, T_MAX, DT]

Adam Michael Bauer
UChicago
"""

import warnings

import numpy as np

from .adjoint import get_forced_adj
from .dynamics import get_forced_path, get_nonlin_path
from .noise_response import check_design
from .obs import get_obs_from_dynamics

warnings.filterwarnings("ignore", category=RuntimeWarning)

MODES = ("marginal", "profile", "fixed")

# returned when the dynamics leave the stable region (C1 far below its prior,
# where explicit Euler blows up and S is no longer positive definite). the weak
# model produces a silent inf there; raising here would kill the dask future
# and with it the whole ensemble, so the optimizer is handed a large finite
# value to back away from instead.
BIG_COST = 1e30


class SCContext:
    """Per-member inputs to the cost that aren't controls.

    Parameters
    ----------
    noise: NoiseModel
        theta-independent pieces of S for this window

    mode: str
        "marginal", "profile" or "fixed"

    xi_f: (3M,) array
        the member's background noise draw, in the design's units

    meas: (4N,) array
        measurement-noise perturbation of the observations (zeros for match_weak)

    ref_factor: Factor
        factorization of S at the prior center theta_ref. Used for S in "fixed"
        mode, and its L_g for the fixed perturbation in "marginal" and "fixed"
    """

    def __init__(self, noise, mode, xi_f, meas, ref_factor):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        check_design(noise.design)
        self.noise = noise
        self.design = noise.design
        self.mode = mode
        self.xi_f = np.asarray(xi_f, dtype=float)
        self.meas = np.asarray(meas, dtype=float)
        self.ref_factor = ref_factor

        # perturbation added to the observations when it doesn't move with theta
        self.pert_fixed = -noise.apply_L(ref_factor.L_g, self.xi_f)

        self._cache = None

    def __getstate__(self):
        # the memo is only valid within one optimization, so don't ship it
        state = self.__dict__.copy()
        state["_cache"] = None
        return state


def _unpack(args):
    x_f, inv_covar_prior, ctx, obs, e, T_MIN, T_MAX, DT = args
    return x_f, inv_covar_prior, ctx, obs, e, T_MIN, T_MAX, DT


def _evaluate(theta, args):
    """Everything cost and grad share at one theta, memoized on the context.

    SLSQP evaluates fun and jac at the same point, so this halves the cost of
    each iteration.
    """

    _, _, ctx, obs, e, T_MIN, T_MAX, DT = _unpack(args)
    key = theta.tobytes()
    if ctx._cache is not None and ctx._cache[0] == key:
        return ctx._cache[1]

    noise = ctx.noise

    path0, _ = get_nonlin_path(e, theta, T_MIN, T_MAX, DT)
    m = get_obs_from_dynamics(path0).ravel()
    y = np.asarray(obs, dtype=float).ravel() + ctx.meas

    if ctx.mode == "fixed":
        fac = ctx.ref_factor
        L_g = fac.L_g
    else:
        L_g = noise.L_g(theta)
        fac = noise.factor(L_g)

    if ctx.mode == "profile":
        pert = -noise.apply_L(L_g, ctx.xi_f)
    else:
        pert = ctx.pert_fixed

    d = m - (y + pert)
    v, quad = fac.solve_quad(d)

    out = {"L_g": L_g, "fac": fac, "d": d, "v": v, "quad": quad, "pert": pert}
    ctx._cache = (key, out)
    return out


def noise_star(theta, args, ev=None):
    """xi*: the profiled (posterior-mean) noise at theta, in the design's units.

    For "profile" this is exactly the weak model's optimal model error at theta.
    """

    _, _, ctx, *_ = _unpack(args)
    if ev is None:
        ev = _evaluate(np.asarray(theta, dtype=float), args)
    noise = ctx.noise
    if ctx.mode == "fixed":
        return np.zeros(3 * noise.M)
    QLtv = noise.apply_QLt(ev["L_g"], ev["v"])
    if ctx.mode == "profile":
        return ctx.xi_f - QLtv
    return -QLtv


def cost(control, args):
    """Strong-constraint cost for one ensemble member."""

    theta = np.asarray(control, dtype=float)
    x_f, inv_covar_prior, ctx, *_ = _unpack(args)

    try:
        ev = _evaluate(theta, args)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return _unstable_cost(theta, x_f, inv_covar_prior)

    J = 0.5 * (theta - x_f) @ inv_covar_prior @ (theta - x_f) + 0.5 * ev["quad"]
    if ctx.mode == "marginal":
        J += 0.5 * ev["fac"].logdet

    return float(J) if np.isfinite(J) else _unstable_cost(theta, x_f, inv_covar_prior)


def _unstable_cost(theta, x_f, inv_covar_prior):
    """Sentinel cost outside the stable region.

    It must not be constant. SLSQP stops when two successive costs agree, so a
    flat sentinel reads as convergence the moment an iterate lands in the
    unstable region. Growing with the prior distance keeps it consistent with
    the gradient grad() returns there, so the line search retreats. log1p keeps
    it finite however far out the iterate is.
    """

    dist = 0.5 * (theta - x_f) @ inv_covar_prior @ (theta - x_f)
    return float(BIG_COST * (1.0 + np.log1p(dist if np.isfinite(dist) else 1e300)))


def grad(control, args):
    """Gradient of the cost with respect to all 15 controls."""

    theta = np.asarray(control, dtype=float)
    x_f, inv_covar_prior, ctx, obs, e, T_MIN, T_MAX, DT = _unpack(args)
    noise = ctx.noise

    try:
        ev = _evaluate(theta, args)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        # outside the stable region the cost is the _unstable_cost sentinel. a
        # zero gradient there would look like a stationary point, and SLSQP
        # would report success at it. the prior term's gradient is always
        # defined and points back toward the prior draw, the same direction the
        # sentinel grows in, so the line search retreats instead
        return inv_covar_prior @ (theta - x_f)

    # adjoint of the trajectory driven by xi*, forced by v. this single
    # integration carries both dm/dtheta and the -1/2 v^T dS v term
    xi_s = noise_star(theta, args, ev)
    path_s, _ = get_forced_path(e, theta, xi_s, ctx.design, T_MIN, T_MAX, DT)
    lam = get_forced_adj(
        e, path_s, xi_s[: noise.M], ctx.design, ev["v"].reshape(4, noise.N), DT
    )

    g = inv_covar_prior @ (theta - x_f) + lam[:, 0]

    if ctx.mode == "marginal":
        g = g + logdet_grad(theta, ctx, ev["fac"])

    return g if np.all(np.isfinite(g)) else inv_covar_prior @ (theta - x_f)


def logdet_grad(theta, ctx, fac):
    """d/dtheta [1/2 log det S] = tr(Q_g L_g^T S^{-1} dL_g/dtheta_k).

    Only the global-noise columns of L depend on theta, so only they contribute.
    """

    noise = ctx.noise
    W = fac.solve(fac.L_g) @ noise.Q_g
    out = np.zeros_like(theta)
    for k, dL in noise.dL_g(theta).items():
        out[k] = np.sum(W * dL)
    return out
