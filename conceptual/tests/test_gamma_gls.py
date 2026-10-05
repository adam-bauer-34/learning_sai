"""Checks: white GLS = OLS; IV GLS vs direct formula and Monte Carlo; sig_IV = 0 reduces to
white; Delta_T scaling identity.

Run with `pytest test_gamma_gls.py` or `python test_gamma_gls.py`.
"""
from dataclasses import replace

import numpy as np

import gamma_gls as gg
from gamma_model import design_matrix, time_axis

c = 1.5 + 0.004 * time_axis(75)
P = gg.NoiseParams(rho=0.6)


def test_white_is_ols():
    X = design_matrix(c, 1.0, 0.5)
    cov = gg.gls_covariance(X, gg.noise_covariance(75, P, "white"))
    assert np.allclose(cov, 0.3**2 * np.linalg.inv(X.T @ X), rtol=1e-8)


def test_iv_gls_matches_direct_formula():
    X = design_matrix(c, 1 / 3, 0.5)
    S = gg.noise_covariance(75, P, "iv")
    assert np.allclose(gg.gls_covariance(X, S), np.linalg.inv(X.T @ np.linalg.solve(S, X)), rtol=1e-8)


def test_no_iv_reduces_to_white():
    p = replace(P, sig_IV=0.0)
    assert np.allclose(gg.gamma_standard_errors(c, 1.0, 0.5, p, "iv"), gg.gamma_standard_errors(c, 1.0, 0.5, p, "white"))


def test_gls_monte_carlo():
    X = design_matrix(c, 1.0, 0.5)
    S = gg.noise_covariance(75, P, "iv")
    L = np.linalg.cholesky(S); Si = np.linalg.inv(S)
    rng = np.random.default_rng(0)
    T = X @ [3.4, -1.6] + rng.standard_normal((20000, 75)) @ L.T
    est = np.linalg.solve(X.T @ Si @ X, X.T @ Si @ T.T).T
    assert np.allclose(est.std(0), np.sqrt(np.diag(gg.gls_covariance(X, S))), rtol=0.03)


def test_delta_T_scaling():
    a = gg.gamma_standard_errors(c, 2.0, 0.5, P, "iv")
    b = gg.gamma_standard_errors(c, 2.0, 1.0, P, "iv")
    assert np.isclose(a[0], b[0]) and np.isclose(a[1], 2 * b[1])


if __name__ == "__main__":
    test_white_is_ols()
    test_iv_gls_matches_direct_formula()
    test_no_iv_reduces_to_white()
    test_gls_monte_carlo()
    test_delta_T_scaling()
    print("All tests passed.")
