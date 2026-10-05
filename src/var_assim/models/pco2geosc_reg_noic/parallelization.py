"""4DVAR runner for the strong-constraint model.

Mirrors pco2geowc_reg_noic/parallelization.py: x_f stays pinned to the member's
prior draw while x0 warm-starts from the previous outer iteration, with the same
outer loop and stopping rule. See the long comment in that file for why x_f
must not be re-centered on the iterate.

The bounds are the weak model's, so the two optimize over the same set (and
profile mode reproduces the weak solutions exactly). Explicit Euler goes
unstable for C1 below ~Dt (L + G EPS) / 2, where S(theta) can't be factored;
cost() returns a growing sentinel there, and a solve that still ends in that
region is recorded as diverged (flag 2) instead of crashing the ensemble.
Fixed floors on C1/C2 were tried and dropped: about 1% of prior draws sit below
C1 = 2, and the weak model finds stable, sensible solutions for them.

The control is the 15 fixed controls only; the profiled noise xi* and the
observation perturbation are stored alongside it at the end.

Adam Michael Bauer
UChicago
"""

import numpy as np
from scipy.optimize import minimize

from .cost import BIG_COST, _evaluate, cost, grad, noise_star
from .dynamics import N_FIXED, get_nonlin_path

def get_bounds():
    """Box bounds on the 15 controls: the weak model's, restricted to them."""

    bounds = np.array([(-np.inf, np.inf) for _ in range(N_FIXED)])
    bounds[5:13, 0] = 0  # L, G, EPS, C1, C2, F1, a1, a2 >= 0, as in the weak model
    return bounds


class EnsembleMember:
    """4DVAR ensemble member for the strong-constraint model."""

    def __init__(
        self,
        theta_p,
        flag,
        tol,
        max_iter,
        TMIN,
        TMAX,
        DT,
        theta_tr,
        inv_covar_prior,
        ctx,
        obs,
        times,
    ):
        self.theta_p = np.asarray(theta_p, dtype=float)
        self.flag = flag
        self.tol = tol
        self.max_iter = max_iter
        self.TMIN = TMIN
        self.TMAX = TMAX
        self.DT = DT
        self.theta_tr = np.asarray(theta_tr, dtype=float)
        self.inv_covar_prior = inv_covar_prior
        self.ctx = ctx
        self.obs = obs

        self.data_hist = np.zeros(
            (len(self.theta_p), max_iter + 1, len(times)), dtype=np.float32
        )
        self.controls_hist = np.zeros((len(self.theta_p), max_iter + 1), dtype=np.float32)
        self.cost_hist = np.zeros(max_iter + 1, dtype=np.float32)
        self.l2s_hist = np.zeros(max_iter + 1, dtype=np.float32)

        self.controls_hist[:, 0] = self.theta_p

    def cost_args(self, e):
        return [
            self.theta_p,
            self.inv_covar_prior,
            self.ctx,
            self.obs,
            e,
            self.TMIN,
            self.TMAX,
            self.DT,
        ]


def runner_4dvar(mem, e):
    """Optimize one ensemble member; designed to be submitted to a dask worker."""

    args = mem.cost_args(e)

    mem.l2 = np.sqrt(np.sum((mem.theta_p - mem.theta_tr) ** 2))
    mem.l2s_hist[0] = mem.l2

    iter_ = 1

    prior_p, _ = get_nonlin_path(e, mem.theta_p, mem.TMIN, mem.TMAX, mem.DT)
    mem.data_hist[:, 0] = prior_p

    J0 = cost(mem.theta_p, args)
    mem.cost_hist[0] = J0

    mem.control = mem.theta_p

    # defined up front so the member is well-formed even if the loop never runs
    # (the weak model leaves these undefined in that case)
    new_p, final_cost = prior_p, J0
    bounds = get_bounds()

    while mem.l2 > mem.tol:
        # x0 warm-starts from the previous outer iteration; x_f (args[0]) stays
        # pinned to the prior draw. see pco2geowc_reg_noic/parallelization.py
        sol = minimize(
            cost,
            x0=mem.control,
            args=(args,),
            bounds=bounds,
            method="SLSQP",
            jac=grad,
        )

        if sol.fun >= BIG_COST:
            # the solve ended outside the stable region, where the dynamics blow
            # up and the cost is only a sentinel. keep the last stable iterate
            # rather than recording a meaningless control, mark the member as
            # diverged (flag 2), and stop. cost_hist keeps the sentinel, so the
            # usual final/initial cost-ratio screen removes the member downstream
            mem.cost_hist[iter_:] = BIG_COST
            mem.controls_hist[:, iter_:] = mem.control[:, None]
            mem.l2s_hist[iter_:] = mem.l2
            mem.data_hist[:, iter_:] = new_p[:, None, :]
            final_cost = BIG_COST
            mem.flag = 2
            break

        new_theta = sol.x
        final_cost = sol.fun
        mem.cost_hist[iter_] = sol.fun

        mem.l2 = np.sqrt(np.sum((new_theta - mem.theta_tr) ** 2))
        mem.l2s_hist[iter_] = mem.l2

        new_p, _ = get_nonlin_path(e, new_theta, mem.TMIN, mem.TMAX, mem.DT)
        mem.data_hist[:, iter_] = new_p

        mem.control = new_theta
        mem.controls_hist[:, iter_] = mem.control

        iter_ += 1

        if iter_ > mem.max_iter:
            mem.flag = 1
            break

    mem.data = new_p
    mem.cost = final_cost

    # profiled noise and the observation perturbation actually used, both at the
    # solution. the perturbation only moves with theta in "profile" mode
    try:
        ev = _evaluate(np.asarray(mem.control, dtype=float), args)
        mem.xi_star = noise_star(mem.control, args, ev)
        mem.obs_pert = ev["pert"] + mem.ctx.meas
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        # only reachable if even the kept iterate is unstable; record NaNs so
        # the member is visibly unusable rather than crashing the ensemble
        mem.xi_star = np.full(3 * mem.ctx.noise.M, np.nan)
        mem.obs_pert = np.full(4 * mem.ctx.noise.N, np.nan)
        mem.flag = 2

    # the context carries the window's factorization, which is only needed while
    # optimizing; drop it before the member is shipped back to the client
    mem.ctx = None

    return mem
