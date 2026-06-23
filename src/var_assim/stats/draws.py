"""Draw from prior distribution.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.30.2024
"""

import numpy as np

# lambda, gamma, C1, C2, and F1CO2 are nonneg
nonneg_two_region_inds = [5, 6, 8, 9, 10]
nonneg_three_region_inds = [x + 1 for x in nonneg_two_region_inds]


def get_prior_draws(model, means, covar, N):
    prior_vec = np.random.multivariate_normal(means, covar, size=N)

    # manually correct parameters that have unphysical bounds. testing suggests
    # that most of the time the parameters aren't outside this range, and so
    # I'll just relocate to the mean for now.

    # if three regions, use the shifted indices
    if model == "pco2geowc3":
        for ind in nonneg_three_region_inds:
            prior_vec[:, ind] = np.where(
                prior_vec[:, ind] < 0, means[ind], prior_vec[:, ind]
            )

    else:
        for ind in nonneg_two_region_inds:
            prior_vec[:, ind] = np.where(
                prior_vec[:, ind] < 0, means[ind], prior_vec[:, ind]
            )

    return prior_vec
