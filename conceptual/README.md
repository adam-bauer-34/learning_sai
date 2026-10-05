# Γ ramps

Standalone scripts for the Γ-ramps analysis: Cramér–Rao / GLS standard errors of Γ_CO₂ and
Γ_SAI vs observational window, for CO₂ scenarios and SAI ramp shapes.

    T_t = Γ_CO2 ln(c_t/278.3) + Γ_SAI ΔT (min(t, 50)/50)^z + noise_t,   t = 1 in 2026

Two noise models:
- **white** (`gamma_model.py`): noise_t = ε_t ~ N(0, σ²); SEs are the exact OLS covariance σ²(XᵀX)⁻¹.
- **iv** (`gamma_gls.py`): noise_t = ε_t + Γ_CO2 (L_eff/f) q_t with q_t AR(1); SEs are the GLS
  covariance (XᵀΣ⁻¹X)⁻¹. For the white model this reduces to the OLS result above.

## Use
No install needed. From this directory:

    python run_gamma_ramps.py      # white noise: gamma_ramps.png, gamma_ramps_se_table.csv
    python run_gamma_gls.py        # white vs IV: gamma_white.png, gamma_iv_gls.png, gamma_overlay.png, gamma_ratio.png
    python test_gamma_model.py     # or: pytest
    python test_gamma_gls.py
    jupyter lab gamma_ramps.ipynb  # or gamma_gls.ipynb

## Files
- `gamma_model.py`: time axis, SAI regressor, design matrix, QR-based covariance, `gamma_standard_errors` (OLS);
  `load_scenarios` (merges scenarios identical over the window), `concentration_ratio`, `SHORT_NAMES`;
  `se_table` over scenario × z × ΔT × window; `add_relative_se`
- `gamma_gls.py`: `NoiseParams`, `noise_covariance`, `gls_covariance`, `gamma_standard_errors` (GLS),
  `se_table` (adds a model dimension), `ratio_table`. Imports the design, scenarios and covariance from `gamma_model.py`
- `gamma_plot.py`: `plot_se_grid(df, hue=..., style=..., ...)`, drawing only; used by both analyses
- `run_gamma_ramps.py`, `run_gamma_gls.py`: configuration + compute + figure output
- `gamma_ramps.ipynb`, `gamma_gls.ipynb`: interactive versions with plotting and model examples
- `test_gamma_model.py`, `test_gamma_gls.py`: covariances vs direct formulas and Monte Carlo; ΔT scaling; IV→white limit
- Input: `../data/input/scenariomip7_conc_paths.csv` (scenario concentration statistics)
- Output: `../analysis/figs/conceptual/` (figures and SE table)

Assumptions: known noise covariance; no intercept; c_t and ΔT, z, cap known; annual spacing.
