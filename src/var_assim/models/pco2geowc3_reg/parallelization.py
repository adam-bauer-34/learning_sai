"""4DVAR Runner.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.6.2024
"""

import numpy as np

from time import perf_counter

from .cost import cost, grad
from .dynamics import get_nonlin_path
from scipy.optimize import minimize


class EnsembleMember:
    """4DVAR Ensemble Member Class."""

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
        inv_covar_T1_obs,
        inv_covar_Q_obs,
        inv_covar_T_R1_obs,
        inv_covar_T_R2_obs,
        inv_covar_T_R3_obs,
        obs,
        times,
    ):

        # set class attributes
        self.theta_p = theta_p
        self.flag = flag
        self.tol = tol
        self.max_iter = max_iter
        self.TMIN = TMIN
        self.TMAX = TMAX
        self.DT = DT
        self.theta_tr = theta_tr
        self.inv_covar_prior = inv_covar_prior
        self.inv_covar_T1_obs = inv_covar_T1_obs
        self.inv_covar_Q_obs = inv_covar_Q_obs
        self.inv_covar_T_R1_obs = inv_covar_T_R1_obs
        self.inv_covar_T_R2_obs = inv_covar_T_R2_obs
        self.inv_covar_T_R3_obs = inv_covar_T_R3_obs
        self.obs = obs
        self.timing = {}  # profiler for performance

        # make histories for cost, l2, time series, and controls
        self.data_hist = np.zeros(
            (len(theta_p), max_iter + 1, len(times)), dtype=np.float32
        )
        self.controls_hist = np.zeros((len(theta_p), max_iter + 1), dtype=np.float32)
        self.cost_hist = np.zeros(max_iter + 1, dtype=np.float32)
        self.l2s_hist = np.zeros(max_iter + 1, dtype=np.float32)

        # set prior in controls history
        self.controls_hist[:, 0] = theta_p


def runner_4dvar(mem, e):
    """4DVAR "runner" function.

    This function is designed to be used in tandem with Dask to parallelize the
    4DVAR procedure.

    Parameters
    ----------
    mem: `EnsembleMember` class
        ensemble member class

    e: `EmissionsBaseline` class
        contains emissions information that is shared among Dask workers

    Returns
    -------
    mem: `EnsembleMember` class
        ensemble member class with additional attributes related to 4DVAR
        simulation
    """

    # set the initial value of the l2 norm and store
    mem.l2 = np.sqrt(np.sum((np.array(mem.theta_p) - mem.theta_tr) ** 2))
    mem.l2s_hist[0] = mem.l2

    # set the iteration counter
    iter_ = 1

    # get prior paths
    t0 = perf_counter()
    prior_p, _ = get_nonlin_path(e, mem.theta_p, mem.TMIN, mem.TMAX, mem.DT)
    mem.timing["get_prior_path"] = perf_counter() - t0

    # set prior paths as first entry in data history
    mem.data_hist[:, 0] = prior_p

    # compute the cost function for prior and store in history
    t0 = perf_counter()
    J0 = cost(
        mem.theta_p,
        args=[
            mem.theta_p,
            mem.inv_covar_prior,
            mem.inv_covar_T1_obs,
            mem.inv_covar_Q_obs,
            mem.inv_covar_T_R1_obs,
            mem.inv_covar_T_R2_obs,
            mem.inv_covar_T_R3_obs,
            mem.obs,
            e,
            mem.TMIN,
            mem.TMAX,
            mem.DT,
        ],
    )

    mem.cost_hist[0] = J0
    mem.timing["get_first_cost"] = perf_counter() - t0

    # set the current member control as the prior
    mem.control = mem.theta_p

    # while the l2 norm is higher than the tolerance, do the following 4DVAR
    # process.
    while mem.l2 > mem.tol:
        # solve optimization problem
        bounds = np.array([(-np.inf, np.inf) for cont in mem.control])
        bounds[6:15, 0] = 0  # L, G, EPS, C1, C2, F1, a1, a2, a3 >= 0
        bounds[18:, :] = (
            -1.2,
            1.2,
        )  # implicit bound on global model errors of 4.5 sigma and regional ~3

        sol = minimize(
            cost,
            x0=mem.control,
            args=[
                mem.control,
                mem.inv_covar_prior,
                mem.inv_covar_T1_obs,
                mem.inv_covar_Q_obs,
                mem.inv_covar_T_R1_obs,
                mem.inv_covar_T_R2_obs,
                mem.inv_covar_T_R3_obs,
                mem.obs,
                e,
                mem.TMIN,
                mem.TMAX,
                mem.DT,
            ],
            bounds=bounds,
            method="SLSQP",
            jac=grad,
        )

        # set optimal solution as new estimate of control variables
        new_theta = sol.x

        # store the optimal cost function in the history
        mem.cost_hist[iter_] = sol.fun

        # compute L2 norm between truth and optimal value and store
        mem.l2 = np.sqrt(np.sum((new_theta - mem.theta_tr) ** 2))
        mem.l2s_hist[iter_] = mem.l2

        # store new trajectory in the data history
        t0 = perf_counter()
        new_p, _ = get_nonlin_path(e, new_theta, mem.TMIN, mem.TMAX, mem.DT)
        mem.data_hist[:, iter_] = new_p
        mem.timing["post_path"] = perf_counter() - t0

        # if diff > tol, set the first guess as the optimal solution and
        # try again
        mem.control = new_theta
        mem.controls_hist[:, iter_] = mem.control

        # index the iteration counter
        iter_ += 1

        if iter_ > mem.max_iter:
            # we've exceeded maximum iterations, and we end the 4DVAR
            # process
            mem.flag = 1
            break

    # set final values
    mem.data = new_p
    mem.cost = sol.fun

    return mem


if __name__ == "__main__":
    import pickle
    import numpy as np

    m = EnsembleMember(
        np.zeros(5 + 2 * 3 + 6 + 3 * 77),
        0,
        1e-2,
        100,
        2023,
        2100,
        1.0,
        np.zeros(5 + 2 * 3 + 6 + 3 * 77),
        np.zeros((5 + 2 * 3 + 6 + 3 * 77, 5 + 2 * 3 + 6 + 3 * 77)),
        np.zeros((77, 77)),
        np.zeros((77, 77)),
        np.zeros((77, 77)),
        np.zeros((77, 77)),
        np.zeros((77, 77)),
        np.zeros((5, 77)),
        np.zeros(77),
    )

    print(f"{len(pickle.dumps(m))/1024**2:.2f} MB")
