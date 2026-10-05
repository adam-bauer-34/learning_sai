"""theta-time-bias-perc for the strong-constraint model `pco2geosc_reg_noic`.

The analysis of analysis/notebooks/theta-time-bias-perc.ipynb (cells 0-19), for
the strong-constraint runs: ensemble median of the recovered angle, its bias
against the truth, and the 5-95 percentile range, against assimilation window.
Three figures, as in the notebook:

  1. median / bias / range at each window
  2. the same with the prior pinned at TMIN
  3. as 2, using the ensemble mean instead of the median

Member screen. The weak model's final/initial cost-ratio <= 0.5 screen does not
transfer to this model. In profile mode the initial cost is already minimized over
the noise, so healthy members sit near a ratio of 0.55 (weak: ~0.12), and the
screen throws out a large, non-random share of good members. A member is kept
instead when:

  - it did not diverge (flag != 2: the solve ended in unstable dynamics), and
  - its final cost is finite and below the unstable-dynamics sentinel (1e30), and
  - its controls are finite.

The >90 degree mask the notebook applies to the angles is kept. The script also
reports how many members the old ratio screen would have kept, for comparison.

Differences from the notebook, on purpose:

  - true angles are computed from each file's controls_truth rather than from
    truth.yaml, so the figure describes what was actually run
  - the prior values pinned at TMIN in figures 2-3 are computed from the prior
    draws (median, mean, 5-95 range) instead of the hard-coded 15, 12 and 25
  - axis units in square brackets

To run (from the repo root; reads only the small variables of each file):
    python analysis/scripts/sc_theta_time_bias_perc.py \
        --sc_noise flux --sc_covar marginal --sc_obs_pert match_weak \
        --sai_ramp linear [--n_ens 1000] [--thetas 5 10 15 20 25 30] [--no-save]

Adam Michael Bauer
UChicago
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

from var_assim.config import DATA_DIR_ABS, FIGS_DIR
from var_assim.models.pco2geosc_reg_noic.cost import BIG_COST
from var_assim.plotting import angle_diagnostics as ad
from var_assim.plotting.presets import get_presets
from var_assim.plotting.utils import make_figure_filename

MODEL = "pco2geosc_reg_noic"
DIVERGED_FLAG = 2
PLO, PHI = 5, 95

# Okabe-Ito as in the notebook; the color follows the true angle's position in
# the full theta list, so it doesn't change when a file is missing
COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
          "#0072B2", "#CC79A7", "#D55E00"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sc_noise", default="flux", choices=["flux", "temp"])
    ap.add_argument("--sc_covar", default="marginal",
                    choices=["marginal", "profile", "fixed"])
    ap.add_argument("--sc_obs_pert", default="match_weak",
                    choices=["match_weak", "plus_meas"])
    ap.add_argument("--sai_ramp", default="linear", choices=["linear", "fast", "slow"])
    ap.add_argument("--scenario", default="ssp245", choices=["ssp245", "ssp585"])
    ap.add_argument("--windowing", default="gradual")
    ap.add_argument("--tmin", type=int, default=2025)
    ap.add_argument("--noise_model", default="AR1", choices=["AR1", "AR0"])
    ap.add_argument("--deg_p_dec", type=float, default=0.1)
    ap.add_argument("--n_yrs_ramp", type=int, default=50)
    ap.add_argument("--ecs", type=float, default=3.0)
    ap.add_argument("--n_ens", type=int, default=1000)
    ap.add_argument("--thetas", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30])
    ap.add_argument("--angle_max", type=float, default=90.0,
                    help="angles above this are masked, as in the notebook")
    ap.add_argument("--no-save", action="store_true", help="don't write figures/CSV")
    return ap.parse_args()


def output_path(cli, theta):
    """Mirrors postprocessing.make_master_datatree for this model."""

    return (
        DATA_DIR_ABS / "output" / MODEL / (
            f"var-assim-output_{cli.scenario}_{MODEL}_{cli.windowing}_{cli.noise_model}+reg"
            f"_TMIN{cli.tmin}_THETA{theta}_ECS{cli.ecs}"
            f"_ramprate{cli.sai_ramp}_DEGpDEC{cli.deg_p_dec}_NYRSRAMP{cli.n_yrs_ramp}"
            f"_Nens{cli.n_ens}_sc-{cli.sc_noise}-{cli.sc_covar}-{cli.sc_obs_pert}.nc"
        )
    )


def keep_mask(g):
    """The strong-constraint member screen (see module docstring)."""

    flag = np.asarray(g.variables["flag"][:])
    costs = np.asarray(g.variables["costs"][:], dtype=float)
    controls = np.asarray(g.variables["controls"][:], dtype=float)

    diverged = flag == DIVERGED_FLAG
    sentinel = ~np.isfinite(costs) | (costs >= BIG_COST)
    bad_controls = ~np.all(np.isfinite(controls), axis=1)
    keep = ~(diverged | sentinel | bad_controls)

    # what the weak model's screen would have done, for the report only
    ch = np.asarray(g.variables["cost_hist"][:], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ch[:, -1] / ch[:, 0]
    old = np.isfinite(ratio) & (ratio <= 0.5)

    counts = {
        "n_total": int(flag.size),
        "n_diverged": int(diverged.sum()),
        "n_sentinel": int((sentinel & ~diverged).sum()),
        "n_kept": int(keep.sum()),
        "n_ratio05_would_keep": int(old.sum()),
    }
    return keep, counts


def collect(cli):
    """Angles for every (theta, window), plus the prior and the true angles."""

    paths = {th: output_path(cli, th) for th in cli.thetas}
    present = [th for th in cli.thetas if paths[th].exists()]
    missing = [th for th in cli.thetas if th not in present]
    if missing:
        warnings.warn(
            "missing output for theta = " + ", ".join(map(str, missing))
            + "; they are left out of the figures:\n  "
            + "\n  ".join(str(paths[th]) for th in missing)
        )
    if not present:
        raise FileNotFoundError("no output files found for this configuration")

    windows = ad.window_names(paths[present[0]])
    rows = []
    angles = {}
    truth = {}
    prior_ref = None

    for th in present:
        ds = nc.Dataset(paths[th])
        try:
            if sorted(ds.groups, key=int) != windows:
                raise ValueError(f"THETA{th} has windows {sorted(ds.groups)}, expected {windows}")

            for w in windows:
                g = ds.groups[w]
                vari = [str(v) for v in g.variables["vari"][:]]
                head = [vari.index(n) for n in ad.CONTROL_HEAD]

                # the prior ensemble is the same draw in every file (same seed),
                # so it only has to be read once, but check that it is
                if w == windows[0]:
                    prior = np.asarray(g.variables["controls_hist"][:, :, 0],
                                       dtype=float)[:, head]
                    if prior_ref is None:
                        prior_ref = prior
                    elif not np.allclose(prior, prior_ref, rtol=1e-6, atol=0):
                        warnings.warn(f"THETA{th}: prior draws differ from THETA{present[0]}")

                    ct = np.asarray(g.variables["controls_truth"][:], dtype=float)[head]
                    truth[th] = ad.angle_at({p: ct[ad.CONTROL_HEAD.index(p)]
                                             for p in ad.ANGLE_PARAMS})

                keep, counts = keep_mask(g)
                post = np.asarray(g.variables["controls"][:], dtype=float)[:, head]

                ang = np.full(post.shape[0], np.nan)
                ang[keep] = ad.angle_from_controls(post[keep])
                over = keep & (ang > cli.angle_max)
                ang[over] = np.nan
                angles[(th, w)] = ang

                vals = ang[np.isfinite(ang)]
                rows.append({
                    "theta": th, "window": int(w), **counts,
                    "n_over_angle_max": int(over.sum()),
                    "median": float(np.median(vals)),
                    "mean": float(np.mean(vals)),
                    "bias_median": float(np.median(vals) - truth[th]),
                    "p5": float(np.percentile(vals, PLO)),
                    "p95": float(np.percentile(vals, PHI)),
                })
        finally:
            ds.close()

    ang_prior = ad.angle_from_controls(prior_ref)
    ang_prior = np.where(ang_prior > cli.angle_max, np.nan, ang_prior)

    return {
        "present": present,
        "windows": windows,
        "years": np.array([int(w) for w in windows]),
        "angles": angles,
        "truth": truth,
        "prior": ang_prior,
        "rows": rows,
    }


def make_figure(res, cli, stat, pinned, tag, save):
    """The notebook's 3-panel figure: center statistic, bias, and 5-95 range."""

    years = res["years"]
    windows = res["windows"]
    prior = res["prior"][np.isfinite(res["prior"])]
    prior_center = np.median(prior) if stat == "median" else np.mean(prior)
    prior_range = np.percentile(prior, PHI) - np.percentile(prior, PLO)
    center = np.median if stat == "median" else np.mean

    fig, ax = plt.subplots(1, 3, figsize=(22, 6))
    x = np.hstack([[cli.tmin], years]) if pinned else years

    range_max = prior_range
    for th in res["present"]:
        i = cli.thetas.index(th) + 1
        c = COLORS[i % len(COLORS)]
        tr = res["truth"][th]
        a = [res["angles"][(th, w)] for w in windows]
        cen = np.array([center(v[np.isfinite(v)]) for v in a])
        rng = np.array([np.percentile(v[np.isfinite(v)], PHI)
                        - np.percentile(v[np.isfinite(v)], PLO) for v in a])
        if pinned:
            cen = np.hstack([[prior_center], cen])
            rng = np.hstack([[prior_range], rng])
        range_max = max(range_max, float(np.max(rng)))

        ax[0].plot(x, cen, linestyle="solid", color=c, zorder=100,
                   label=r"$\vartheta^\dagger=${}$^\circ$".format(th))
        ax[0].axhline(tr, linestyle="dashed", linewidth=1.25, color=c)
        ax[1].plot(x, cen - tr, linestyle="solid", color=c, zorder=100)
        ax[2].plot(x, rng, linestyle="solid", color=c, zorder=100)

    for a in ax:
        a.set_xlim((cli.tmin if pinned else cli.tmin - 2, years[-1]))
        for w in years:
            a.axvline(w, linestyle="dotted", color="grey", linewidth=1.25)
        a.set_xlabel("Year")

    sym = "med" if stat == "median" else "mean"
    ax[0].set_ylabel(sym + r"$(\vartheta)$ [°]")
    ax[0].legend(bbox_to_anchor=(2.9, 1.15), ncols=len(res["present"]))
    ax[1].set_ylabel(sym + r"$(\vartheta) - \vartheta^\dagger$ [°]")
    ax[1].axhline(0, color="k", linestyle="solid")

    # upper limit: the prior range, as in the notebook, unless the data exceed it
    yub = max(prior_range, range_max) * 1.05
    ax[2].set_ylabel(f"{PLO}–{PHI} percentile range [°]")
    ax[2].set_ylim((0, yub))

    # a second scale for the same quantity, relative to the prior's range
    ax2_rel = ax[2].twinx()
    ax2_rel.set_ylabel(f"relative {PLO}–{PHI} range [vs prior]")
    ax2_rel.spines["right"].set_visible(True)
    ax2_rel.set_ylim((0, yub / prior_range))

    for a, lab in zip(ax, ["A", "B", "C"]):
        a.text(0.925, 0.98, lab, transform=a.transAxes,
               fontsize=20, fontweight="bold", va="top", ha="left")

    fig.suptitle(
        f"{MODEL}  |  {cli.scenario}  |  {cli.windowing}  |  TMIN {cli.tmin}  |  "
        f"{cli.noise_model}  |  DEGpDEC {cli.deg_p_dec}  |  ramp {cli.sai_ramp}  |  "
        f"sc {cli.sc_noise}/{cli.sc_covar}/{cli.sc_obs_pert}  |  Nens {cli.n_ens}  |  "
        f"screen: flag != 2, cost < 1e30",
        fontsize=14, y=1.10,
    )

    if not save:
        plt.close(fig)
        return None
    name = f"sc-theta-time-bias-perc-{'filled-' if pinned else ''}{stat}-{tag}"
    out = make_figure_filename(name, outdir=FIGS_DIR / "results")
    fig.savefig(out, dpi=350, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")
    return out


def report(res, cli, tag, save):
    prior = res["prior"][np.isfinite(res["prior"])]
    print(f"prior angle: median {np.median(prior):.2f}  mean {np.mean(prior):.2f}  "
          f"{PLO}-{PHI} range {np.percentile(prior, PHI) - np.percentile(prior, PLO):.2f}")
    print("true angles (from controls_truth): "
          + ", ".join(f"{th}: {res['truth'][th]:.3f}" for th in res["present"]))
    print()

    cols = ["theta", "window", "n_total", "n_diverged", "n_sentinel", "n_kept",
            "n_ratio05_would_keep", "n_over_angle_max", "median", "bias_median",
            "p5", "p95", "mean"]
    print("  ".join(f"{c:>9s}" if i > 1 else f"{c:>6s}" for i, c in enumerate(cols)))
    for r in res["rows"]:
        print("  ".join(
            (f"{r[c]:6d}" if i <= 1 else
             f"{r[c]:9d}" if isinstance(r[c], int) else f"{r[c]:9.3f}")
            for i, c in enumerate(cols)))

    if save:
        out = Path(make_figure_filename(f"sc-theta-time-bias-perc-{tag}",
                                        FIGS_DIR / "results", "csv"))
        with open(out, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in res["rows"]:
                f.write(",".join(str(r[c]) for c in cols) + "\n")
        print(f"\ntable -> {out}")


def main():
    cli = parse_args()
    save = not cli.no_save

    presets, _ = get_presets()
    plt.rcParams.update(presets)

    tag = (f"{cli.scenario}-{cli.windowing}-TMIN{cli.tmin}-{cli.noise_model}"
           f"-DEGpDEC{cli.deg_p_dec}-ramp{cli.sai_ramp}"
           f"-sc-{cli.sc_noise}-{cli.sc_covar}-{cli.sc_obs_pert}-Nens{cli.n_ens}")

    res = collect(cli)
    report(res, cli, tag, save)

    make_figure(res, cli, "median", pinned=False, tag=tag, save=save)
    make_figure(res, cli, "median", pinned=True, tag=tag, save=save)
    make_figure(res, cli, "mean", pinned=True, tag=tag, save=save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
