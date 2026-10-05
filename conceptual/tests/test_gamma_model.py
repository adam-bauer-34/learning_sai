"""Checks: covariance vs direct inverse and Monte Carlo; Delta_T scaling identity.

Run with `pytest test_gamma_model.py` or `python test_gamma_model.py`.
"""
import numpy as np

from gamma_model import design_matrix, gamma_standard_errors, time_axis


def test_matches_direct_inverse_and_monte_carlo():
    c = 1.5 + 0.01 * time_axis(75)
    X = design_matrix(c, 1.0, 0.5)
    r = gamma_standard_errors(c, 1.0, 0.5, 0.3)
    assert np.allclose(r.cov, 0.3**2 * np.linalg.inv(X.T @ X), rtol=1e-8)
    rng = np.random.default_rng(0)
    T = X @ [3.4, -1.6] + 0.3 * rng.standard_normal((20000, 75))
    est = np.linalg.lstsq(X, T.T, rcond=None)[0].T
    assert np.allclose(est.std(0), [r.se_co2, r.se_sai], rtol=0.03)


def test_delta_T_scaling():
    """Doubling Delta_T halves SE(Gamma_SAI) and leaves SE(Gamma_CO2) unchanged."""
    c = 1.5 + 0.01 * time_axis(60)
    a, b = gamma_standard_errors(c, 2.0, 0.5), gamma_standard_errors(c, 2.0, 1.0)
    assert np.isclose(a.se_co2, b.se_co2) and np.isclose(a.se_sai, 2 * b.se_sai)


if __name__ == "__main__":
    test_matches_direct_inverse_and_monte_carlo()
    test_delta_T_scaling()
    print("All tests passed.")
