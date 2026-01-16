"""4DVAR Runner.

Adam Michael Bauer
University of Illinois Urbana-Champaign
8.6.2024
"""

import numpy as np

from .cost import cost, grad
from .dynamics import get_nonlin_path
from scipy.optimize import minimize


class EnsembleMember():
    """4DVAR Ensemble Member Class.
    """

    def __init__(self, theta_p, flag, tol, max_iter, TMIN, TMAX, DT,
                 theta_tr, inv_covar_prior, inv_covar_T1_obs, inv_covar_Q_obs,
                 inv_covar_T_R1_obs, inv_covar_T_R2_obs,
                 obs, times):

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

        # make histories for cost, l2, time series, and controls
        self.data_hist = np.zeros((len(theta_p),
                                   max_iter + 1, len(times)))
        self.controls_hist = np.zeros((len(theta_p),
                                       max_iter + 1))
        self.cost_hist = np.zeros(max_iter + 1)
        self.l2s_hist = np.zeros(max_iter + 1)

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
    mem.l2 = np.sqrt(np.sum((np.array(mem.theta_p)
                             - mem.theta_tr)**2))
    mem.l2s_hist[0] = mem.l2

    # set the iteration counter
    iter_ = 1

    # get prior paths
    prior_p, _ = get_nonlin_path(e,
                                 mem.theta_p,
                                 mem.TMIN,
                                 mem.TMAX,
                                 mem.DT)

    # set prior paths as first entry in data history
    mem.data_hist[:, 0] = prior_p

    # compute the cost function for prior and store in history
    J0 = cost(mem.theta_p,
              args=[mem.theta_p,
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
                    mem.DT])

    mem.cost_hist[0] = J0

    # set the current member control as the prior
    mem.control = mem.theta_p

    # while the l2 norm is higher than the tolerance, do the following 4DVAR
    # process.
    while mem.l2 > mem.tol:
        # solve optimization problem
        bounds = np.array([(-np.inf, np.inf) for cont in mem.control])
        bounds[5:13, 0] = 0  # L, G, EPS, C1, C2, F1, a1, a2 >= 0

        sol = minimize(cost, x0=mem.control,
                       args=[mem.control,
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
                             mem.DT],
                       bounds=bounds,
                       method="SLSQP",
                       jac=grad)

        # set optimal solution as new estimate of control variables
        new_theta = sol.x

        # store the optimal cost function in the history
        mem.cost_hist[iter_] = sol.fun

        # compute L2 norm between truth and optimal value and store
        mem.l2 = np.sqrt(np.sum((new_theta - mem.theta_tr)**2))
        mem.l2s_hist[iter_] = mem.l2

        # store new trajectory in the data history
        new_p, _ = get_nonlin_path(e, new_theta,
                                   mem.TMIN, mem.TMAX, mem.DT)
        mem.data_hist[:, iter_] = new_p

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
