"""Observation operator: the same [T1, Q, T_R1, T_R2] selector as the weak model.

Re-exported rather than copied so the two models cannot drift apart.
"""

from var_assim.models.pco2geowc_reg_noic.obs import get_obs_from_dynamics, get_obs_jac

__all__ = ["get_obs_from_dynamics", "get_obs_jac"]
