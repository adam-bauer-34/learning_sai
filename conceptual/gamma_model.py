"""Regression model, scenario loading and SE tables for the Gamma-ramps figure.

Model (one region)
------------------
Let N be the number of annual observations, t_k = t0 + (k-1) for k = 1..N (t = 1 is 2026),
c_k the CO2 concentration in year k, and eps_k iid N(0, sigma^2). Then

    T_k = Gamma_CO2 * ln(c_k / c_pi) + Gamma_SAI * x_SAI(t_k) + eps_k,
    x_SAI(t) = Delta_T * (min(t, t_cap) / t_ref)^z   [+ 0.1 cos(t - t0) if modulated],

with Gamma_CO2 = alpha f / L_eff and Gamma_SAI = beta L_eff - alpha. The SEs of
(Gamma_CO2, Gamma_SAI) depend only on the regressors and sigma, not on alpha, beta,
f or L_eff (the model is linear with known white noise), so none of those appear here.

Mathematics
-----------
Least squares is unbiased with covariance sigma^2 (X^T X)^{-1}, which equals the
Cramer-Rao bound exactly at every N (Gaussian, linear mean, known covariance).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import linalg

C_PREINDUSTRIAL = 278.3  # ppm

SHORT_NAMES = {  # scenario column (possibly merged) -> legend label
    "high-extension / high-overshoot": "H",
    "high-extension": "H",
    "high-overshoot": "H",
    "medium-extension": "M",
    "medium-overshoot": "MO",
    "low": "L",
    "verylow": "VL",
    "verylow-overshoot": "VLO",
}


# ---------------------------------------------------------------- model


def time_axis(N: int, t0: float = 1.0, dt: float = 1.0) -> np.ndarray:
    """Observation times t_k = t0 + (k-1)*dt, k = 1..N (t0 > 0 so powers are defined)."""
    if t0 <= 0:
        raise ValueError("t0 must be > 0.")
    return t0 + dt * np.arange(N, dtype=float)


def sai_regressor(
    t: np.ndarray,
    z: float,
    delta_T: float,
    t_ref: float = 50.0,
    t_cap: float | None = 50.0,
    t0: float = 1.0,
    modulated: bool = False,
) -> np.ndarray:
    """x_SAI(t) = delta_T * (min(t, t_cap)/t_ref)^z; no plateau if t_cap is None.

    If modulated, 0.1 * cos(t - t0) is added (t0 = first observation time).
    """
    tt = t if t_cap is None else np.minimum(t, t_cap)  # freeze time at the cap
    x = delta_T * (tt / t_ref) ** float(z)
    if modulated:
        x = x + 0.05 * np.cos(t - t0)
    return x


def design_matrix(
    c_ratio,
    z: float,
    delta_T: float,
    t0: float = 1.0,
    dt: float = 1.0,
    t_ref: float = 50.0,
    t_cap: float | None = 50.0,
    modulated: bool = False,
) -> np.ndarray:
    """N x 2 design matrix [ln(c/c_pi), x_SAI(t)], with N = len(c_ratio).

    c_ratio must be c_t / c_pi (finite, > 0); element 0 is the first observation year.
    modulated adds 0.1 * cos(t - t0) to x_SAI (see `sai_regressor`).
    """
    c = np.asarray(c_ratio, dtype=float)
    if c.ndim != 1 or not np.all(np.isfinite(c)) or np.any(c <= 0):
        raise ValueError("c_ratio must be a 1-D array of finite positive values.")
    t = time_axis(len(c), t0, dt)
    return np.column_stack(
        [np.log(c), sai_regressor(t, z, delta_T, t_ref, t_cap, t0, modulated)]
    )


def covariance_from_design(X: np.ndarray, sigma: float) -> np.ndarray:
    """sigma^2 (X^T X)^{-1}, via QR of the column-scaled design (avoids squaring cond(X))."""
    s = np.linalg.norm(X, axis=0)  # column norms
    if np.any(s == 0):
        raise ValueError("A regressor is identically zero.")
    _, R = np.linalg.qr(X / s)  # X/s = Q R
    d = np.abs(np.diag(R))
    if d.min() < 1e-12 * d.max():
        raise ValueError(
            "Regressors are collinear: coefficients not jointly identified."
        )
    Rinv = linalg.solve_triangular(R, np.eye(X.shape[1]))
    return sigma**2 * (Rinv @ Rinv.T) / np.outer(s, s)  # undo the column scaling


@dataclass(frozen=True)
class GammaSE:
    """CRB summary for (Gamma_CO2, Gamma_SAI)."""

    se_co2: float
    se_sai: float
    cov: np.ndarray  # 2x2, ordered (Gamma_CO2, Gamma_SAI)

    @property
    def corr(self) -> float:
        return float(self.cov[0, 1] / (self.se_co2 * self.se_sai))


def gamma_standard_errors(
    c_ratio, z: float, delta_T: float, sigma: float = 0.3, **design_kwargs
) -> GammaSE:
    """SEs of (Gamma_CO2, Gamma_SAI) for one forcing series and record length N = len(c_ratio)."""
    cov = covariance_from_design(
        design_matrix(c_ratio, z, delta_T, **design_kwargs), sigma
    )
    se = np.sqrt(np.diag(cov))
    return GammaSE(float(se[0]), float(se[1]), cov)


# ---------------------------------------------------------------- scenarios


def load_scenarios(
    path, stat: str = "median", year_start: int = 2026, year_end: int = 2100
) -> pd.DataFrame:
    """Concentrations (ppm) with index = year, one column per scenario.

    `stat` selects the CSV column (mean, median, p05, ...). Scenarios identical over
    [year_start, year_end] (to 1e-6 ppm) are merged into one column named "a / b".
    Columns are ordered by concentration in year_end, high to low.
    """
    d = pd.read_csv(path)
    wide = d.pivot_table(index="year", columns="scenario", values=stat).loc[
        year_start:year_end
    ]
    merged, used = {}, set()
    for a in wide.columns:
        if a in used:
            continue
        twins = [
            b
            for b in wide.columns
            if b not in used and np.allclose(wide[a], wide[b], atol=1e-6)
        ]
        used.update(twins)
        merged[" / ".join(twins)] = wide[a]
    out = pd.DataFrame(merged)
    return out[out.iloc[-1].sort_values(ascending=False).index]


def concentration_ratio(
    conc: pd.Series, N: int, c_pi: float = C_PREINDUSTRIAL
) -> np.ndarray:
    """First N values of c_t / c_pi (the model takes the log)."""
    return conc.values[:N] / c_pi


# ---------------------------------------------------------------- SE table


def se_table(
    scenarios: pd.DataFrame,
    z_values,
    delta_T_values,
    sigma: float = 0.3,
    N_values=None,
    year0: int = 2025,
    **design_kwargs
) -> pd.DataFrame:
    """Tidy DataFrame of SEs for every (scenario, z, delta_T, window).

    Parameters
    ----------
    scenarios : DataFrame from `load_scenarios` (index = year, starting at year0 + 1)
    z_values, delta_T_values : iterables of ramp exponents and SAI amplitudes
    sigma : white-noise std
    N_values : window lengths (default 5..len(scenarios), i.e. windows ending 2030..2100)
    year0 : t = 1 is year0 + 1, so a window of N observations ends in year0 + N
    design_kwargs : passed to `design_matrix` (t_ref, t_cap, t0, dt, modulated)

    Columns: scenario, z, delta_T, N, year, se_co2, se_sai, corr, and the indices
    se_co2_rel, se_sai_rel (each SE divided by its value in the first window).
    """
    if N_values is None:
        N_values = np.arange(5, len(scenarios) + 1)
    rows = []
    for name, conc in scenarios.items():
        for z in z_values:
            for dT in delta_T_values:
                for N in N_values:
                    r = gamma_standard_errors(
                        concentration_ratio(conc, N), z, dT, sigma, **design_kwargs
                    )
                    rows.append((name, z, dT, N, year0 + N, r.se_co2, r.se_sai, r.corr))
    df = pd.DataFrame(
        rows,
        columns=["scenario", "z", "delta_T", "N", "year", "se_co2", "se_sai", "corr"],
    )
    return add_relative_se(df, ["scenario", "z", "delta_T"])


def add_relative_se(df: pd.DataFrame, keys) -> pd.DataFrame:
    """Add se_co2_rel, se_sai_rel: each SE divided by its first-window value within `keys` groups."""
    g = df.groupby(keys)
    df["se_co2_rel"] = df["se_co2"] / g["se_co2"].transform(
        "first"
    )  # relative to first window
    df["se_sai_rel"] = df["se_sai"] / g["se_sai"].transform("first")
    return df
