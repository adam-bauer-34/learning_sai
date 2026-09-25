"""Corner plots of ALPHA_R2 against BETA_R2 at the first assimilation window.

One figure with a corner plot per true angle. Each has the posterior ensemble
as a scatter in the center, the ALPHA_R2 marginal above it and the BETA_R2
marginal to its right, with the true value (dashed black), the prior center
(dashed green) and the ensemble median (solid vermilion) marked in all three
panels. Axis limits are shared across the six panels so they can be compared.

To run:
    python analysis/scripts/alpha_beta_corner.py [--model MODEL] [--window 2040]
        [--screen 0.5] [--no-save]

Scope: ssp245 / DEGpDEC 0.1 / `four` windowing / Nens 1000, as in
`angle_drift_diagnostics.py`. Post-processing I/O only, safe to run
interactively.

Adam Michael Bauer
UChicago
"""

import argparse
import logging
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from var_assim.config import FIGS_DIR
from var_assim.plotting import angle_diagnostics as ad
from var_assim.plotting.presets import get_presets
from var_assim.plotting.utils import make_figure_filename

MODELS = ["pco2geowc_reg", "pco2geowc_reg_noic"]
THETAS = [5, 10, 15, 20, 25, 30]
X_PARAM, Y_PARAM = "ALPHA_R2", "BETA_R2"

# Okabe-Ito, as mandated by presets.py
C_ENS = "#0072B2"
C_MED = "#D55E00"
C_TRUTH = "#000000"
C_PRIOR = "#009E73"

LABELS = {"ALPHA_R2": r"$\alpha_{R2}$", "BETA_R2": r"$\beta_{R2}$"}


def theta_label(th):
    return r"$\vartheta^\dagger=${}$^\circ$".format(th)


def collect(model, window, screen):
    """Screened posterior (ALPHA_R2, BETA_R2) and truth for every true angle."""

    ix, iy = ad.CONTROL_HEAD.index(X_PARAM), ad.CONTROL_HEAD.index(Y_PARAM)
    paths = {th: ad.output_path(th, model=model) for th in THETAS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing input files:\n" + "\n".join(missing))

    res = {}
    for th in THETAS:
        win = ad.load_window(paths[th], window)
        keep = ad.screen_members(win["cost_hist"], screen)
        post = win["controls"][keep][:, win["head"]]
        truth = win["controls_truth"][win["head"]]
        res[th] = dict(
            x=post[:, ix],
            y=post[:, iy],
            x_true=truth[ix],
            y_true=truth[iy],
            n_keep=int(keep.sum()),
            n_ens=int(keep.size),
        )
        del win
    return res


def shared_limits(res, key, true_key, extra=(), pad=0.05):
    """Axis limits covering the 0.5-99.5 percentile of every ensemble, every
    truth and any `extra` reference values, so one stray member does not
    flatten all six panels."""

    lo = min(min(np.percentile(d[key], 0.5), d[true_key]) for d in res.values())
    hi = max(max(np.percentile(d[key], 99.5), d[true_key]) for d in res.values())
    lo, hi = min([lo, *extra]), max([hi, *extra])
    span = hi - lo
    return lo - pad * span, hi + pad * span


def draw_corner(fig, cell, d, th, xlim, ylim, prior, letter):
    """One corner plot inside gridspec cell `cell` of `fig`."""

    gs = cell.subgridspec(
        2, 2, width_ratios=(4, 1.3), height_ratios=(1.3, 4), wspace=0.05, hspace=0.05
    )
    ax = fig.add_subplot(gs[1, 0])
    axx = fig.add_subplot(gs[0, 0], sharex=ax)
    axy = fig.add_subplot(gs[1, 1], sharey=ax)

    x_med, y_med = np.median(d["x"]), np.median(d["y"])
    x_pr, y_pr = prior

    ax.scatter(d["x"], d["y"], s=8, color=C_ENS, alpha=0.35, linewidth=0,
               label="Ensemble")
    # the prior line is wider and drawn underneath, so it stays visible where
    # it coincides with the truth (the ALPHA_R2 prior center equals its truth)
    for a, orient in ((ax, None), (axx, "v"), (axy, "h")):
        if orient != "h":
            a.axvline(x_pr, color=C_PRIOR, linestyle="dashed", linewidth=3.2,
                      zorder=2)
            a.axvline(d["x_true"], color=C_TRUTH, linestyle="dashed",
                      linewidth=1.6, zorder=3)
            a.axvline(x_med, color=C_MED, linestyle="solid", linewidth=2.0,
                      zorder=4)
        if orient != "v":
            a.axhline(y_pr, color=C_PRIOR, linestyle="dashed", linewidth=3.2,
                      zorder=2)
            a.axhline(d["y_true"], color=C_TRUTH, linestyle="dashed",
                      linewidth=1.6, zorder=3)
            a.axhline(y_med, color=C_MED, linestyle="solid", linewidth=2.0,
                      zorder=4)
    ax.scatter(d["x_true"], d["y_true"], marker="X", s=200, color=C_TRUTH,
               edgecolor="white", linewidth=1.5, zorder=6, label="Truth")
    ax.scatter(x_med, y_med, marker="o", s=140, color=C_MED,
               edgecolor="white", linewidth=1.5, zorder=6, label="Ensemble Median")

    axx.hist(d["x"], bins=np.linspace(*xlim, 41), density=True, color=C_ENS,
             alpha=0.6, edgecolor="white", linewidth=0.8)
    axy.hist(d["y"], bins=np.linspace(*ylim, 41), density=True, color=C_ENS,
             alpha=0.6, edgecolor="white", linewidth=0.8,
             orientation="horizontal")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(LABELS[X_PARAM], fontsize=21)
    ax.set_ylabel(LABELS[Y_PARAM], fontsize=21)
    axx.tick_params(labelbottom=False)
    axy.tick_params(labelleft=False)
    axx.set_yticks([])
    axy.set_xticks([])
    axx.set_ylabel("normalized\ndensity", fontsize=13)
    axy.set_xlabel("normalized\ndensity", fontsize=13)
    axx.set_title(theta_label(th), fontsize=20)
    # panel label, top left of the corner plot, clear of the density label
    axx.text(-0.2, 1.08, letter, transform=axx.transAxes, fontsize=26,
             fontweight="bold", va="bottom", ha="left")
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pco2geowc_reg_noic", choices=MODELS,
                    help="model whose output is plotted")
    ap.add_argument("--window", default="2040", help="assimilation window group")
    ap.add_argument("--screen", type=float, default=0.5,
                    help="keep members with final/initial cost ratio <= this")
    ap.add_argument("--no-save", action="store_true", help="do not write outputs")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logger = logging.getLogger("alpha_beta_corner")

    presets, _ = get_presets()
    plt.rcParams.update(presets)

    logger.info(f"collecting {cli.model}, window {cli.window} ...")
    res = collect(cli.model, cli.window, cli.screen)

    args, _, Prior, _, _ = ad.build_calibration(THETAS[0], model=cli.model)
    # R2 is the second region in the two-region calibration lists
    prior = (Prior.ALPHA_CEN[1], Prior.BETA_CEN[1])

    xlim = shared_limits(res, "x", "x_true", extra=[prior[0]])
    ylim = shared_limits(res, "y", "y_true", extra=[prior[1]])

    fig = plt.figure(figsize=(27, 19))
    outer = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.3)
    for k, th in enumerate(THETAS):
        d = res[th]
        logger.info(
            f"theta {th:2d}: kept {d['n_keep']}/{d['n_ens']}  "
            f"{X_PARAM} med {np.median(d['x']):.4g} (truth {d['x_true']:.4g})  "
            f"{Y_PARAM} med {np.median(d['y']):.4g} (truth {d['y_true']:.4g})"
        )
        ax = draw_corner(
            fig, outer[k // 3, k % 3], d, th, xlim, ylim, prior, "abcdef"[k]
        )

    # one legend for the whole figure; the prior is a line only, so add a proxy
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=C_PRIOR, linestyle="dashed", linewidth=3.2))
    labels.append("Prior Center")
    fig.legend(handles, labels, loc="upper center", ncols=4, fontsize=18,
               bbox_to_anchor=(0.5, 0.95), markerscale=1.5)

    if not cli.no_save:
        out = make_figure_filename(
            # no figure title, so the run metadata lives in the filename
            f"alpha-beta-corner_{cli.model}_{args.scenario}_{args.windowing}"
            f"_window{cli.window}_DEGpDEC{args.deg_p_dec}_Nens{args.n_ens}"
            f"_screen{cli.screen}",
            outdir=FIGS_DIR / "results",
        )
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"figure -> {out}")

if __name__ == "__main__":
    main()
