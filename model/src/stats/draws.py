"""Draw from prior distribution.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.30.2024
"""

import numpy as np


def get_prior_draws(means, covar, N):
    prior_vec = np.random.multivariate_normal(means, covar, size=N)

    # manually correct parameters that have unphysical bounds. testing suggests
    # that most of the time the parameters aren't outside this range, and so
    # I'll just relocate to the mean for now.

    ## L >= 0
    prior_vec[:, 3] = np.where(prior_vec[:, 3] < 0, means[3], prior_vec[:, 3])

    ## G >= 0
    prior_vec[:, 4] = np.where(prior_vec[:, 4] < 0, means[4], prior_vec[:, 4])

    ## C1 >= 0
    prior_vec[:, 5] = np.where(prior_vec[:, 5] < 0, means[5], prior_vec[:, 5])

    ## C2 >= 0
    prior_vec[:, 6] = np.where(prior_vec[:, 6] < 0, means[6], prior_vec[:, 6])

    ## F1_CO2 >= 0
    prior_vec[:, 7] = np.where(prior_vec[:, 7] < 0, means[7], prior_vec[:, 7])

    ## F3_CO2 >= 0
    prior_vec[:, 8] = np.where(prior_vec[:, 8] < 0, means[8], prior_vec[:, 8])

    ## C_SO2 >= 0
    prior_vec[:, 11] = np.where(prior_vec[:, 11] < 0, means[11], prior_vec[:,
                                                                           11])

    return prior_vec
