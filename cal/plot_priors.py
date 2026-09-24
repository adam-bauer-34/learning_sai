"""Visualize parameter priors in config/priors.yaml and compare to published estimates.

Adam Michael Bauer
UChicago
Sep 2026

Published estimates (as recorded in config/priors.yaml):
    - F2x = 3.99 +/- 0.19 W m^-2 (Smith et al. 2020, RFMIP); compared against
      F1_CO2 * ln(2), since F_CO2 = F1_CO2 * ln(C / C0)
    - ECS likely (17-83%) range 2.5-4 K
    - TCR 1.8 K, likely range 1.4-2.2 K (IPCC AR6)

ECS and TCR are derived by sampling the (independent Gaussian) priors: ECS =
F1_CO2 ln2 / L and TCR = F1_CO2 ln2 / (L + EPS * G), the approximation used to
calibrate G (heat capacities are ignored).

To run (from the repo root): python cal/plot_priors.py [--regions two_region] [--save]
"""

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import stats

PRIOR_COLOR = "#0072B2"
PUB_COLOR = "#E69F00"
N_SAMPLES = 200_000
SEED = 0

# published estimates
SMITH_F2X_CEN, SMITH_F2X_STD = 3.99, 0.19
ECS_LIKELY = (2.5, 4.0)
TCR_CEN, TCR_LIKELY = 1.8, (1.4, 2.2)


def load_priors(path, regions):
    """Load priors.yaml, filling in L_CEN and factor-based stds as priors.py does."""
    with open(path, "r") as f:
        prior_data = yaml.safe_load(f)

    p = prior_data["global"] | prior_data[regions]
    p["L_CEN"] = p["F1_CO2_CEN"] * np.log(2) / p["ECS_CEN"]
    for cen, std in p["factor_vars"]:
        p[std] = p[cen] * p["PRIOR_STD_FACTOR"]

    return p


def sample_globals(p, rng):
    names = ["L", "G", "EPS", "C1", "C2", "F1_CO2"]
    return {n: rng.normal(p[f"{n}_CEN"], p[f"{n}_STD"], N_SAMPLES) for n in names}


def compute_tcr(s):
    """TCR = F2x / (lambda + eps * gamma), the approximation used to calibrate G."""
    return s["F1_CO2"] * np.log(2) / (s["L"] + s["EPS"] * s["G"])


def plot_gaussian(ax, cen, std, color, label, lw=2, ls="-"):
    x = np.linspace(cen - 4 * std, cen + 4 * std, 400)
    ax.plot(x, stats.norm.pdf(x, cen, std), color=color, lw=lw, ls=ls, label=label)


def plot_samples(ax, x, xlim, label):
    ax.hist(
        x,
        bins=np.linspace(*xlim, 120),
        density=True,
        histtype="stepfilled",
        color=PRIOR_COLOR,
        alpha=0.35,
        edgecolor=PRIOR_COLOR,
        lw=1.5,
        label=label,
    )
    ax.set_xlim(xlim)


def shade_likely(ax, lo, hi, x, label):
    """Shade published likely range and mark the prior's 17-83% range."""
    ax.axvspan(lo, hi, color=PUB_COLOR, alpha=0.2, label=label)
    q17, q83 = np.percentile(x, [17, 83])
    for q in (q17, q83):
        ax.axvline(q, color=PRIOR_COLOR, ls="--", lw=1.2)
    frac = np.mean((x >= lo) & (x <= hi))
    ax.text(
        0.97,
        0.95,
        f"prior 17-83%: {q17:.2f}-{q83:.2f}\n"
        f"prior mass in [{lo}, {hi}]: {frac:.0%}\n(likely = 66%)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors", default="config/priors.yaml")
    parser.add_argument(
        "--regions", default="two_region", choices=["two_region", "three_region"]
    )
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    p = load_priors(args.priors, args.regions)
    rng = np.random.default_rng(SEED)
    s = sample_globals(p, rng)

    # derived quantities need physical draws; report how many we drop
    phys = np.all([s[n] > 0 for n in ["L", "G", "F1_CO2"]], axis=0)
    for n in ["L", "G", "C1", "C2", "F1_CO2"]:
        print(f"P({n} <= 0) = {np.mean(s[n] <= 0):.2%}")
    print(f"Dropping {np.mean(~phys):.2%} of draws for ECS/TCR")
    s_phys = {n: v[phys] for n, v in s.items()}

    F2x = s_phys["F1_CO2"] * np.log(2)
    ECS = F2x / s_phys["L"]
    TCR = compute_tcr(s_phys)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    ax = axes.ravel()

    # F2x
    plot_gaussian(
        ax[0],
        p["F1_CO2_CEN"] * np.log(2),
        p["F1_CO2_STD"] * np.log(2),
        PRIOR_COLOR,
        r"prior: $F_{1,CO_2}\,\ln 2$",
    )
    plot_gaussian(
        ax[0], SMITH_F2X_CEN, SMITH_F2X_STD, PUB_COLOR, "Smith et al. (2020)", ls="--"
    )
    ax[0].set_title(r"$F_{2\times}$ [W m$^{-2}$]")
    ax[0].legend(fontsize=8, loc="upper left")

    # lambda
    plot_gaussian(ax[1], p["L_CEN"], p["L_STD"], PRIOR_COLOR, "prior")
    ax[1].set_title(r"$\lambda$ [W m$^{-2}$ K$^{-1}$]")

    # ECS
    plot_samples(ax[2], ECS, (0, 8), "prior (derived)")
    shade_likely(ax[2], *ECS_LIKELY, ECS, "published likely range")
    ax[2].axvline(p["ECS_CEN"], color="k", lw=1, label=f"ECS_CEN = {p['ECS_CEN']}")
    ax[2].set_title(r"ECS $= F_{2\times}/\lambda$ [K]")
    ax[2].legend(fontsize=8, loc="center right")

    # TCR
    plot_samples(ax[3], TCR, (0, 4), "prior (derived)")
    shade_likely(ax[3], *TCR_LIKELY, TCR, "AR6 likely range")
    ax[3].axvline(TCR_CEN, color=PUB_COLOR, lw=1.5, label=f"AR6 central ({TCR_CEN})")
    ax[3].set_title(r"TCR $= F_{2\times}/(\lambda + \varepsilon\gamma)$ [K]")
    ax[3].legend(fontsize=8, loc="center right")

    # remaining global parameters (no published comparison)
    for a, (n, title) in zip(
        ax[4:],
        [
            ("G", r"$\gamma$ [W m$^{-2}$ K$^{-1}$]"),
            ("EPS", r"$\varepsilon$ (efficacy)"),
            ("C1", r"$C_1$ [W yr m$^{-2}$ K$^{-1}$]"),
            ("C2", r"$C_2$ [W yr m$^{-2}$ K$^{-1}$]"),
        ],
    ):
        plot_gaussian(a, p[f"{n}_CEN"], p[f"{n}_STD"], PRIOR_COLOR, "prior")
        a.axvline(0, color="0.6", lw=0.8)
        a.set_title(title)

    for a in ax:
        a.set_yticks([])
        a.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Global parameter priors (config/priors.yaml)")

    # regional priors
    n_reg = len(p["ALPHA_CEN"])
    fig_r, axes_r = plt.subplots(
        2, n_reg, figsize=(4 * n_reg, 6), constrained_layout=True
    )
    for r in range(n_reg):
        for row, (n, sym) in enumerate([("ALPHA", r"\alpha"), ("BETA", r"\beta")]):
            a = axes_r[row, r]
            plot_gaussian(a, p[f"{n}_CEN"][r], p[f"{n}_STD"][r], PRIOR_COLOR, "prior")
            if n == "BETA":
                a.axvline(0, color="0.6", lw=0.8)
            a.set_title(rf"${sym}_{{{r + 1}}}$")
            a.set_yticks([])
            a.spines[["top", "right", "left"]].set_visible(False)
    fig_r.suptitle(f"Regional parameter priors ({args.regions})")

    if args.save:
        outdir = Path("analysis/figs/cal")
        outdir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        fig.savefig(outdir / f"{today}-global_priors.png", dpi=200)
        fig_r.savefig(outdir / f"{today}-{args.regions}_priors.png", dpi=200)
        print(f"Saved figures to {outdir}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
