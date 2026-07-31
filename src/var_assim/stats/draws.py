"""Draw from prior distribution.

Adam Michael Bauer
University of Illinois Urbana-Champaign
7.30.2024
"""

import numpy as np

# lambda, gamma, C1, C2, and F1CO2 are nonneg
nonneg_two_region_inds = [5, 6, 8, 9, 10]
nonneg_three_region_inds = [x + 1 for x in nonneg_two_region_inds]


def get_prior_draws(model, means, covar, N, rng=None):
    if rng is None:
        rng = np.random.default_rng()  # safe fallback for rng setting

    prior_vec = rng.multivariate_normal(means, covar, size=N)

    # manually correct parameters that have unphysical bounds. testing suggests
    # that most of the time the parameters aren't outside this range, and so
    # I'll just relocate to the mean for now.

    # if three regions, use the shifted indices.
    #
    # keyed off the registry rather than the model name, because an exact-string
    # test ("pco2geowc3") silently misses pco2geowc3_reg and pco2geowc3_nn: they
    # would fall through to the two-region indices, which for a three-region
    # control vector point at T_R3 and EPS instead of G and F1_CO2. that both
    # clips an initial condition that is allowed to be negative and leaves two
    # genuinely non-negative parameters unclipped.
    #
    # the import is function-level on purpose. var_assim.models imports every
    # model's runner, and each runner imports this module, so importing the
    # registry at module scope is circular.
    from var_assim.models import MODEL_REGISTRY

    if model not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model}': cannot tell how many regions it has, and "
            "so cannot pick the right non-negative parameter indices. Add it to "
            "MODEL_REGISTRY in var_assim.models."
        )

    if MODEL_REGISTRY[model]["N_regions"] == "three_region":
        nonneg_inds = nonneg_three_region_inds
    else:
        nonneg_inds = nonneg_two_region_inds

    for ind in nonneg_inds:
        prior_vec[:, ind] = np.where(
            prior_vec[:, ind] < 0, means[ind], prior_vec[:, ind]
        )

    return prior_vec
