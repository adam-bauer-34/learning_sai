"""Calibrate the beta priors in config/priors.yaml to the Harding et al. (2023) theta prior.

Adam Michael Bauer
UChicago
Sep 2026

The prior on the SAI inequality angle theta is implied by pushing draws from the
parameter priors through get_angle_r{2,3}. Everything except the betas is held at
its config/priors.yaml value; the beta prior central values and standard deviation
are chosen to minimize the squared misfit (in degrees) between the implied theta
prior's 5th/50th/95th percentiles and the Harding et al. (2023) reported values.

Constraints on the betas:
    - two region: beta_1 = -beta_2 (perfectly antisymmetric), one shared std
        free parameters: (b, s) -> BETA_CEN = [-b, b], BETA_STD = [s, s]
    - three region: same constraint on regions 1 and 2, beta_3 free, one shared std
        free parameters: (b, b3, s) -> BETA_CEN = [-b, b, b3], BETA_STD = [s, s, s]

With --joint_std the two and three region models are fit together with a single
std shared across both.

The theta samples use common random numbers (the standard normal draws are fixed
across optimizer iterations), so the objective is a deterministic, continuous
function of the betas and Nelder-Mead converges cleanly.

To run (from the repo root):
    python cal/calibrate_beta_priors.py [--regions two_region three_region]
                                        [--joint_std] [--write] [--save]

The run ends with the summary figure from theta_prior.ipynb (theta prior CDFs vs
Harding, beta_2 prior CDFs with the true beta_2s), drawn with the calibrated betas.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize, root

from var_assim.config import FIGS_DIR
from var_assim.plotting.pproc import get_angle_r2, get_angle_r3
from var_assim.plotting.presets import get_presets
from var_assim.plotting.utils import make_figure_filename

N_SAMPLES = 200_000
SEED = 42
PHI = 0.09  # forcing efficacy of SAI (get_angle_r{2,3} default)

# Harding et al. (2023) reported theta percentiles [deg]
PERCENTILES = [5, 50, 95]
HARDING = np.array([5.0, 15.0, 30.0])

N_REGS = {"two_region": 2, "three_region": 3}

# true theta values used in the experiments [deg]
THETA_TRUTH = [5, 11, 15, 21, 30, 35]

REGION_COLORS = {"two_region": "#0072B2", "three_region": "#D55E00"}
REGION_LABELS = {"two_region": "Two regions", "three_region": "Three regions"}
TRUTH_COLOR = "#009E73"


def load_priors(path, regions):
    """Load priors.yaml, filling in L_CEN as priors.py does."""
    with open(path, "r") as f:
        prior_data = yaml.safe_load(f)

    p = prior_data["global"] | prior_data[regions]
    p["L_CEN"] = p["F1_CO2_CEN"] * np.log(2) / p["ECS_CEN"]

    return p


def get_angle_vec(alphas, betas, l, e, g, phi=PHI):
    """Vectorized get_angle_r{2,3}; alphas and betas are (N_regs, N) arrays."""
    inside = (l + e * g) / phi
    r = -(betas * inside - alphas)

    num = np.sum(alphas * r, axis=0)
    denom = np.sqrt(np.sum(alphas**2, axis=0) * np.sum(r**2, axis=0))
    return np.arccos(num / denom) * 180 / np.pi


class ThetaPrior:
    """Implied theta prior for one model, with everything but the betas frozen."""

    def __init__(self, p, N_regs, rng):
        self.N_regs = N_regs

        # fixed draws of the non-beta parameters
        self.l = rng.normal(p["L_CEN"], p["L_STD"], N_SAMPLES)
        self.e = rng.normal(p["EPS_CEN"], p["EPS_STD"], N_SAMPLES)
        self.g = rng.normal(p["G_CEN"], p["G_STD"], N_SAMPLES)
        self.alphas = np.array(
            [
                rng.normal(p["ALPHA_CEN"][i], p["ALPHA_STD"][i], N_SAMPLES)
                for i in range(N_regs)
            ]
        )

        # fixed standard normal draws for the betas: beta = cen + std * Z
        self.Z = rng.standard_normal((N_regs, N_SAMPLES))

    def betas(self, beta_cen, beta_std):
        return np.asarray(beta_cen)[:, None] + beta_std * self.Z

    def sample(self, beta_cen, beta_std):
        return get_angle_vec(
            self.alphas, self.betas(beta_cen, beta_std), self.l, self.e, self.g
        )

    def percentiles(self, beta_cen, beta_std):
        return np.percentile(self.sample(beta_cen, beta_std), PERCENTILES)

    def check_against_library(self, beta_cen, beta_std, n=200):
        """Assert the vectorized angle reproduces get_angle_r{2,3}."""
        betas = self.betas(beta_cen, beta_std)
        f = get_angle_r2 if self.N_regs == 2 else get_angle_r3
        lib = np.array(
            [
                f(*self.alphas[:, k], *betas[:, k], self.l[k], self.e[k], self.g[k])
                for k in range(n)
            ]
        ).ravel()
        vec = get_angle_vec(
            self.alphas[:, :n], betas[:, :n], self.l[:n], self.e[:n], self.g[:n]
        )
        assert np.allclose(lib, vec, atol=1e-9), "vectorized angle mismatch"


def build_problem(priors, regions_list, joint_std):
    """Return (x0, unpack) for the requested models.

    The optimizer vector holds b (beta_2 = -beta_1) per model, b3 for the three
    region model, and log(std) either per model or shared (--joint_std). The std
    is optimized in log space so it stays positive.

    unpack(x) -> {regions: (BETA_CEN list, BETA_STD float)}
    """
    names, x0 = [], []
    for regions in regions_list:
        cen = priors[regions]["BETA_CEN"]
        names.append((regions, "b"))
        x0.append(cen[1])  # beta_2 = -beta_1
        if N_REGS[regions] == 3:
            names.append((regions, "b3"))
            x0.append(cen[2])

    if joint_std:
        names.append(("joint", "log_s"))
        x0.append(np.log(np.mean([priors[r]["BETA_STD"][0] for r in regions_list])))
    else:
        for regions in regions_list:
            names.append((regions, "log_s"))
            x0.append(np.log(priors[regions]["BETA_STD"][0]))

    def unpack(x):
        v = dict(zip(names, x))
        out = {}
        for regions in regions_list:
            b = v[(regions, "b")]
            log_s = v[("joint", "log_s")] if joint_std else v[(regions, "log_s")]
            beta_cen = [-b, b] + ([v[(regions, "b3")]] if N_REGS[regions] == 3 else [])
            out[regions] = (beta_cen, np.exp(log_s))
        return out

    return np.array(x0), unpack


def misfit(x, unpack, thetas):
    """Sum of squared percentile misfits [deg^2] across the fitted models."""
    J = 0.0
    for regions, (beta_cen, beta_std) in unpack(x).items():
        J += np.sum((thetas[regions].percentiles(beta_cen, beta_std) - HARDING) ** 2)
    return J


def write_yaml(path, regions, beta_cen, beta_std):
    """Overwrite the BETA_CEN / BETA_STD lines of one section, preserving comments."""
    with open(path, "r") as f:
        lines = f.readlines()

    new = {
        "BETA_CEN": "[" + ", ".join(f"{b:.8f}" for b in beta_cen) + "]",
        "BETA_STD": "[" + ", ".join(f"{beta_std:.8f}" for _ in beta_cen) + "]",
    }

    section = None
    for i, line in enumerate(lines):
        m = re.match(r"^(\w+):\s*$", line)
        if m:
            section = m.group(1)
            continue
        if section != regions:
            continue
        for key, val in new.items():
            m = re.match(rf"^(\s+{key}:\s*)\[[^\]]*\](.*)$", line.rstrip("\n"))
            if m:
                lines[i] = f"{m.group(1)}{val}{m.group(2)}\n"

    with open(path, "w") as f:
        f.writelines(lines)


def find_true_betas(p, N_regs, beta_cen, beta_std, theta_truths=THETA_TRUTH, n_sigma=2):
    """(beta_1, beta_2) giving each true theta, with everything else at its prior central value.

    beta_2 moves first, with beta_1 (and beta_3) at BETA_CEN, as in theta_prior.ipynb.
    If that beta_2 falls outside the n_sigma range of its prior, beta_2 is pinned at
    the nearest edge of that range and beta_1 moves instead.

    Returns
    -------
    true_betas: (len(theta_truths), 2) array
        [beta_1, beta_2] for each true theta
    """
    a, l, e, g = p["ALPHA_CEN"], p["L_CEN"], p["EPS_CEN"], p["G_CEN"]

    def angle(b1, b2):
        if N_regs == 2:
            return get_angle_r2(a[0], a[1], b1, b2, l, e, g)
        return get_angle_r3(a[0], a[1], a[2], b1, b2, beta_cen[2], l, e, g)

    b2_lo = beta_cen[1] - n_sigma * beta_std
    b2_hi = beta_cen[1] + n_sigma * beta_std

    true_betas = []
    for theta_true in theta_truths:
        sol = root(lambda x: theta_true - angle(beta_cen[0], x[0]).ravel(), x0=beta_cen[1])
        assert sol.success, f"no beta_2 for theta = {theta_true}: {sol.message}"
        b1, b2 = beta_cen[0], sol.x[0]

        if not b2_lo <= b2 <= b2_hi:
            b2 = np.clip(b2, b2_lo, b2_hi)
            sol = root(lambda x: theta_true - angle(x[0], b2).ravel(), x0=beta_cen[0])
            assert sol.success, f"no beta_1 for theta = {theta_true}: {sol.message}"
            b1 = sol.x[0]

        true_betas.append([b1, b2])

    return np.array(true_betas)


def report_true_betas(regions, p, beta_cen, beta_std, true_betas, theta_truths=THETA_TRUTH):
    """Print the true betas and how many prior sigmas each sits from its central value."""
    N_regs = N_REGS[regions]
    print(f"  {regions} true betas (2 sigma limit on beta_2 before beta_1 moves)")
    for theta_true, (b1, b2) in zip(theta_truths, true_betas):
        a = p["ALPHA_CEN"]
        args = (b1, b2, p["L_CEN"], p["EPS_CEN"], p["G_CEN"])
        check = (
            get_angle_r2(a[0], a[1], *args)
            if N_regs == 2
            else get_angle_r3(a[0], a[1], a[2], b1, b2, beta_cen[2], *args[2:])
        ).item()
        print(
            f"    theta {theta_true:>2}: beta_1 = {b1: .6f} ({(b1 - beta_cen[0]) / beta_std:+.2f} sig), "
            f"beta_2 = {b2: .6f} ({(b2 - beta_cen[1]) / beta_std:+.2f} sig), "
            f"check theta = {check:.4f}"
        )


def plot_summary(priors, thetas, results, save, outdir, theta_truths=THETA_TRUTH):
    """Theta prior CDFs against Harding, and beta_2 prior CDFs with the true beta_2s.

    Betas are divided by PHI so they are in forcing units, as reported in the paper.
    """
    presets, _ = get_presets()
    plt.rcParams.update(presets)

    regions_list = list(results)
    b1_keys = [f"{r}_b1" for r in regions_list]
    fig = plt.figure(figsize=(7 * len(regions_list), 15))
    axd = fig.subplot_mosaic(
        [["theta"] * len(regions_list), regions_list, b1_keys],
        gridspec_kw=dict(height_ratios=[1.2, 1, 1]),
    )

    # --- top panel: theta CDFs ---
    ax = axd["theta"]
    for regions, (beta_cen, beta_std) in results.items():
        x = np.sort(thetas[regions].sample(beta_cen, beta_std))
        cdf = np.arange(1, len(x) + 1) / len(x)
        color = REGION_COLORS[regions]
        ax.plot(x, cdf, color=color, linestyle="solid", linewidth=2.5,
                label=REGION_LABELS[regions])
        ax.axvline(np.median(x), color=color, linestyle="dashed", linewidth=2)
        for q in np.percentile(x, [5, 95]):
            ax.axvline(q, color=color, linestyle="dotted", linewidth=2)

    ax.axvline(HARDING[1], color="black", linestyle="dashed", linewidth=2)
    for q in HARDING[[0, 2]]:
        ax.axvline(q, color="black", linestyle="dotted", linewidth=2)

    for theta_true in theta_truths:
        ax.axvline(theta_true, color=TRUTH_COLOR, linestyle="dashdot", linewidth=1.5)

    # dummy legend entries
    ax.plot([], [], color="black", linestyle="solid", linewidth=2,
            label="Harding et al. (2023)")
    ax.plot([], [], color="grey", linestyle="dashed", linewidth=2, label="Median")
    ax.plot([], [], color="grey", linestyle="dotted", linewidth=2,
            label="5/95 Percentiles")
    ax.plot([], [], color=TRUTH_COLOR, linestyle="dashdot", linewidth=1.5,
            label="True values")

    ax.set_xlabel(r"$\vartheta$")
    ax.set_ylabel("CDF")
    ax.legend(loc="lower right", frameon=True, facecolor="white")
    ax.set_title(r"$\vartheta$ Prior", fontweight="bold")
    ax.set_xlim((0, 45))
    ax.set_ylim((0, 1))

    # --- lower panels: beta_2 (middle row) and beta_1 (bottom row) CDFs ---
    for regions, (beta_cen, beta_std) in results.items():
        betas = thetas[regions].betas(beta_cen, beta_std)
        true_betas = find_true_betas(
            priors[regions], N_REGS[regions], beta_cen, beta_std, theta_truths
        )

        for ax, i in [(axd[regions], 1), (axd[f"{regions}_b1"], 0)]:
            b = np.sort(betas[i]) / PHI
            cdf = np.arange(1, len(b) + 1) / len(b)
            ax.plot(b, cdf, color=REGION_COLORS[regions], linewidth=3, zorder=100)

            ax.axvspan((beta_cen[i] - 2 * beta_std) / PHI,
                       (beta_cen[i] + 2 * beta_std) / PHI,
                       color="grey", alpha=0.15, label=r"prior $\pm 2\sigma$")

            for k, bt in enumerate(true_betas[:, i]):
                ax.axvline(bt / PHI, color=TRUTH_COLOR, linestyle="dashdot", linewidth=2,
                           label=rf"$\beta^{{\dagger}}_{{{i + 1}}}$" if k == 0 else None)

            ax.set_xlabel(rf"$\beta_{i + 1}$")
            ax.set_ylabel("CDF")
            ax.set_title(f"{N_REGS[regions]}-Region Case", fontweight="bold")
            ax.legend(loc="center left" if i == 1 else "center right")

    # panel letters
    for letter, key in zip("ABCDE", ["theta", *regions_list, *b1_keys]):
        axd[key].text(0.02, 0.98, letter, transform=axd[key].transAxes,
                      fontweight="bold", verticalalignment="top",
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()

    if save:
        outdir.mkdir(parents=True, exist_ok=True)
        fname = make_figure_filename("theta_and_beta2_summary", outdir)
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        print(f"saved figure to {fname}")
    else:
        plt.show()


def report(label, theta_prior, beta_cen, beta_std):
    q = theta_prior.percentiles(beta_cen, beta_std)
    print(f"  {label}")
    print(f"    BETA_CEN: [{', '.join(f'{b:.8f}' for b in beta_cen)}]")
    print(f"    BETA_STD: [{', '.join(f'{beta_std:.8f}' for _ in beta_cen)}]")
    for pct, qi, hi in zip(PERCENTILES, q, HARDING):
        print(f"    {pct:>2}th pct: {qi:7.3f}  (Harding {hi:5.1f}, diff {qi - hi:+.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--priors", default="config/priors.yaml")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["two_region", "three_region"],
        choices=["two_region", "three_region"],
    )
    parser.add_argument(
        "--joint_std",
        action="store_true",
        help="share a single beta std across the two and three region models",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the calibrated betas into the priors file",
    )
    parser.add_argument(
        "--save", action="store_true", help="save the summary figure instead of showing it"
    )
    parser.add_argument("--outdir", type=Path, default=FIGS_DIR / "cal")
    parser.add_argument(
        "--theta_truth",
        type=float,
        nargs="+",
        default=THETA_TRUTH,
        help="true theta values [deg] to find true betas for",
    )
    args = parser.parse_args()

    priors = {r: load_priors(args.priors, r) for r in args.regions}
    rng = np.random.default_rng(SEED)
    thetas = {r: ThetaPrior(priors[r], N_REGS[r], rng) for r in args.regions}

    for r in args.regions:
        thetas[r].check_against_library(priors[r]["BETA_CEN"], priors[r]["BETA_STD"][0])

    # fit each model on its own unless the std is shared
    groups = [args.regions] if args.joint_std else [[r] for r in args.regions]

    results = {}
    for group in groups:
        x0, unpack = build_problem(priors, group, args.joint_std)

        print(f"=== {' + '.join(group)} ===")
        print(f"initial misfit: {misfit(x0, unpack, thetas):.4f} deg^2")
        for regions, (beta_cen, beta_std) in unpack(x0).items():
            report(f"{regions} (current priors.yaml)", thetas[regions], beta_cen, beta_std)

        sol = minimize(
            misfit,
            x0,
            args=(unpack, thetas),
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-8, "maxiter": 20_000, "maxfev": 20_000},
        )

        print(f"\n{sol.message} ({sol.nfev} evaluations)")
        print(f"final misfit: {sol.fun:.4f} deg^2")
        for regions, (beta_cen, beta_std) in unpack(sol.x).items():
            report(f"{regions} (calibrated)", thetas[regions], beta_cen, beta_std)
            results[regions] = (beta_cen, beta_std)
        print()

    if args.write:
        for regions, (beta_cen, beta_std) in results.items():
            write_yaml(args.priors, regions, beta_cen, beta_std)
        print(f"wrote calibrated betas to {args.priors}")

    for regions, (beta_cen, beta_std) in results.items():
        true_betas = find_true_betas(
            priors[regions], N_REGS[regions], beta_cen, beta_std, args.theta_truth
        )
        report_true_betas(
            regions, priors[regions], beta_cen, beta_std, true_betas, args.theta_truth
        )
    print()

    plot_summary(priors, thetas, results, args.save, args.outdir, args.theta_truth)


if __name__ == "__main__":
    main()
