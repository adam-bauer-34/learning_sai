"""Gamma SEs without and with a Gamma_CO2-scaled AR(1) internal-variability term, by GLS.

Model (one region), t = 1 in 2026:

    T_t = G ln(c_t/c_pi) + S x_SAI(t) + noise_t,   x_SAI(t) = Delta_T (min(t, t_cap)/t_ref)^z,

with G = Gamma_CO2 and S = Gamma_SAI. Two noise models are compared:

  "white" (the gamma_model analysis): noise_t = eps_t,                     Sigma = sigma^2 I
  "iv":                               noise_t = eps_t + G (L_eff/f) q_t,   Sigma = sigma^2 I + (G k)^2 Var(q) R(rho)

where eps_t ~ N(0, sigma^2) white, k = L_eff/f, q_t = rho q_{t-1} + sig_IV w_t is AR(1)
(sig_IV = innovation std, Var(q) = sig_IV^2/(1 - rho^2)) and R_ij = rho^|i-j|.

Both are estimated by GLS with Sigma known: Cov(G_hat, S_hat) = (X^T Sigma^{-1} X)^{-1}
(exact at every N, best linear unbiased; Aitken). For "white" this is OLS. GLS does not use
the information that the fluctuation size carries about G (the IV "noise channel");
the Cramer-Rao bound including that channel is in the gammaiv package.

The design matrix, scenarios and QR-based covariance are shared with gamma_model.py.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg

from gamma_model import add_relative_se, concentration_ratio, covariance_from_design, design_matrix

MODEL_LABELS = {"white": "no IV (gammaramps)", "iv": "IV, GLS"}


# ---------------------------------------------------------------- noise

@dataclass(frozen=True)
class NoiseParams:
    """Noise parameters. G (true Gamma_CO2) scales the IV term, so it is needed for 'iv'."""
    sigma: float = 0.3                     # white-noise std
    G: float = 1.6 * 4.9 / 2.2896          # true Gamma_CO2 (alpha = 1.6)
    sig_IV: float = 0.09                   # AR(1) innovation std of q_t
    rho: float = 0.2                       # AR(1) autocorrelation of q_t
    L_eff: float = 2.2896
    f: float = 4.9

    @property
    def iv_std(self) -> float:
        """Marginal std of the IV term G (L_eff/f) q_t."""
        return abs(self.G) * self.L_eff / self.f * self.sig_IV / np.sqrt(1.0 - self.rho**2)


def noise_covariance(N: int, p: NoiseParams, model: str = "iv") -> np.ndarray:
    """Sigma for model 'white' (sigma^2 I) or 'iv' (sigma^2 I + iv_std^2 R(rho))."""
    S = p.sigma**2 * np.eye(N)
    if model == "white":
        return S
    if model != "iv":
        raise ValueError(model)
    k = np.arange(N)
    return S + p.iv_std**2 * float(p.rho) ** np.abs(k[:, None] - k[None, :])


# ---------------------------------------------------------------- GLS

def gls_covariance(X: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """(X^T Sigma^{-1} X)^{-1}: whiten with the Cholesky factor of Sigma, then use the
    QR-based OLS covariance of the whitened design with unit noise."""
    C = linalg.cholesky(Sigma, lower=True)                 # Sigma = C C^T
    Xw = linalg.solve_triangular(C, X, lower=True)         # whitened design C^{-1} X
    return covariance_from_design(Xw, 1.0)


def gamma_standard_errors(c_ratio, z: float, delta_T: float, p: NoiseParams = NoiseParams(),
                          model: str = "iv", **design_kwargs):
    """(SE(G), SE(S), corr) by GLS for one forcing series and N = len(c_ratio)."""
    X = design_matrix(c_ratio, z, delta_T, **design_kwargs)
    cov = gls_covariance(X, noise_covariance(len(X), p, model))
    se = np.sqrt(np.diag(cov))
    return float(se[0]), float(se[1]), float(cov[0, 1] / (se[0] * se[1]))


# ---------------------------------------------------------------- SE tables

def se_table(scenarios: pd.DataFrame, z_values, delta_T_values=(0.5, 1.0), p: NoiseParams = NoiseParams(),
             models=("white", "iv"), N_values=None, year0: int = 2025, **design_kwargs) -> pd.DataFrame:
    """Long table: one row per (model, scenario, z, delta_T, window).

    Columns: model, variant (readable label), scenario, z, delta_T, N, year, se_co2, se_sai, corr,
    se_co2_rel, se_sai_rel (relative to the first window). Window N ends in year0 + N.
    """
    N_values = np.arange(5, len(scenarios) + 1) if N_values is None else np.asarray(N_values)
    rows = []
    for m in models:
        for name, conc in scenarios.items():
            for z in z_values:
                for dT in delta_T_values:
                    for N in N_values:
                        se_g, se_s, rho = gamma_standard_errors(concentration_ratio(conc, int(N)), z, dT, p, m,
                                                                **design_kwargs)
                        rows.append((m, MODEL_LABELS.get(m, m), name, z, dT, int(N), year0 + int(N), se_g, se_s, rho))
    df = pd.DataFrame(rows, columns=["model", "variant", "scenario", "z", "delta_T", "N", "year",
                                     "se_co2", "se_sai", "corr"])
    return add_relative_se(df, ["model", "scenario", "z", "delta_T"])


def ratio_table(df: pd.DataFrame, num: str = "iv", den: str = "white") -> pd.DataFrame:
    """SE(num)/SE(den) per (scenario, z, delta_T, window): columns ratio_co2, ratio_sai."""
    keys = ["scenario", "z", "delta_T", "N", "year"]
    a = df[df.model == num].set_index(keys)[["se_co2", "se_sai"]]
    b = df[df.model == den].set_index(keys)[["se_co2", "se_sai"]]
    return (a / b).rename(columns={"se_co2": "ratio_co2", "se_sai": "ratio_sai"}).reset_index()
