"""Per-parameter learning curves for the angle controls, compared across SAI ramps.

This is figure 1a of the angle diagnosis (`angle_drift_diagnostics.py`) re-cut so the
columns are the three geoengineering ramp shapes -- fast (t^1/3), linear, slow (t^3) --
instead of the three summary quantities. One figure per quantity:

  posterior median (truth dashed) | normalized bias vs truth | posterior/prior spread

Rows are the seven controls that enter `get_angle_r2`; one line per true angle.

Only the 15-control head is touched, so this works unchanged for `pco2geowc_reg_noic`
despite its `q_offset = 1` shortening each model-error block by one entry. Truth and
prior spread are both read out of the files, so nothing here depends on `config/`
still matching the runs.

To run (from the repo root, inside a compute allocation):
    python analysis/scripts/ramp_param_learning.py [--no-save]

Adam Michael Bauer
UChicago
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from var_assim.config import DATA_DIR_ABS, FIGS_DIR
from var_assim.plotting import angle_diagnostics as ad
from var_assim.plotting.presets import get_presets
from var_assim.plotting.utils import make_figure_filename

MODEL = "pco2geowc_reg_noic"
SCENARIO = "ssp245"
WINDOWING = "gradual"
TMIN = 2025
NOISE_MODEL = "AR1"
ECS = 3.0
DEG_P_DEC = 0.1
N_YRS_RAMP = 50
N_ENS = 1000

# AR0 was only run with the linear ramp, so AR1 is the only noise model that
# covers all three ramp shapes.
RAMPS = ["fast", "linear", "slow"]
RAMP_LABELS = {
    "fast": r"fast  ($t^{1/3}$)",
    "linear": r"linear  ($t$)",
    "slow": r"slow  ($t^{3}$)",
}
THETAS = [5, 10, 15, 20, 25, 30]
SCREEN = 0.5
PLO, PHI = 5, 95

# Okabe-Ito, index 0 reserved for reference lines; true angle i takes [i + 1]
COLORS = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#CC79A7",
]
REF = "#4d4d4d"

# units are the paper convention: square brackets, not parentheses
CONTROL_LABELS = {
    "L": r"$\lambda$" "\n" r"[W/(m$^2$ K)]",
    "G": r"$\gamma$" "\n" r"[W/(m$^2$ K)]",
    "EPS": r"$\varepsilon$" "\n" r"[--]",
    "ALPHA_R1": r"$\alpha_{R1}$" "\n" r"[--]",
    "ALPHA_R2": r"$\alpha_{R2}$" "\n" r"[--]",
    "BETA_R1": r"$\beta_{R1}$" "\n" r"[K/(W/m$^2$)]",
    "BETA_R2": r"$\beta_{R2}$" "\n" r"[K/(W/m$^2$)]",
}

# rows in the order get_angle_r2 reads them, grouped so the two alphas and the
# two betas sit together
ROWS = ["L", "G", "EPS", "ALPHA_R1", "ALPHA_R2", "BETA_R1", "BETA_R2"]


def output_path(ramp, theta):
    """Path of one run, mirroring `postprocessing.make_master_datatree`."""

    return (
        DATA_DIR_ABS
        / "output"
        / MODEL
        / (
            f"var-assim-output_{SCENARIO}_{MODEL}_{WINDOWING}_{NOISE_MODEL}+reg"
            f"_TMIN{TMIN}_THETA{theta}_ECS{ECS}"
            f"_ramprate{ramp}_DEGpDEC{DEG_P_DEC}"
            f"_NYRSRAMP{N_YRS_RAMP}_Nens{N_ENS}.nc"
        )
    )


def collect():
    """Read every window of every (ramp, theta) run.

    Returns
    -------
    res: dict
        `res[ramp][theta][window]` with the per-control median, percentiles,
        truth and prior spread; `res["windows"]` alongside.
    """

    paths = {(r, th): output_path(r, th) for r in RAMPS for th in THETAS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing input files:\n" + "\n".join(missing))

    res = {r: {th: {} for th in THETAS} for r in RAMPS}
    windows = None

    for ramp in RAMPS:
        for th in THETAS:
            p = paths[(ramp, th)]
            ws = ad.window_names(p)
            if windows is None:
                windows = ws
            elif ws != windows:
                raise ValueError(
                    f"{p.name} has windows {ws}, expected {windows}; the runs "
                    "cannot share an x axis"
                )

            for w in windows:
                # model errors are not needed, so data_final stays unread
                win = ad.load_window(p, w, want_data_final=False)
                keep = ad.screen_members(win["cost_hist"], SCREEN)
                head = win["head"]
                post = win["controls"][keep][:, head]
                prior = win["controls_prior"][keep][:, head]

                res[ramp][th][w] = dict(
                    post_med=np.median(post, axis=0),
                    post_lo=np.percentile(post, PLO, axis=0),
                    post_hi=np.percentile(post, PHI, axis=0),
                    prior_med=np.median(prior, axis=0),
                    prior_lo=np.percentile(prior, PLO, axis=0),
                    prior_hi=np.percentile(prior, PHI, axis=0),
                    prior_sd=prior.std(axis=0),
                    truth=win["controls_truth"][head],
                    keep=int(keep.sum()),
                    n_ens=int(keep.size),
                )
                del win

            print(f"  read {ramp:6s} theta {th:2d}", flush=True)

    res["windows"] = windows
    return res


def _series(res, ramp, th, j, kind, windows):
    """One control's curve for one run, in the requested units."""

    d = [res[ramp][th][w] for w in windows]
    truth = d[0]["truth"][j]

    if kind == "median":
        return np.array([x["post_med"][j] for x in d]), truth
    if kind == "bias":
        return (
            np.array([(x["post_med"][j] - truth) / x["prior_sd"][j] for x in d]),
            0.0,
        )
    if kind == "spread":
        return (
            np.array(
                [
                    (x["post_hi"][j] - x["post_lo"][j])
                    / (x["prior_hi"][j] - x["prior_lo"][j])
                    for x in d
                ]
            ),
            1.0,
        )
    raise ValueError(kind)


KIND_TITLES = {
    "median": "posterior median (dashed = truth)",
    "bias": r"normalized bias  (post $-$ truth) / $\sigma_{\rm prior}$",
    "spread": "posterior / prior spread  (1 = nothing learned)",
}
KIND_YLAB = {
    "median": "",
    "bias": r"bias / $\sigma_{\rm prior}$",
    "spread": "spread ratio",
}


def make_figure(res, kind, save):
    """Rows = angle controls, columns = ramp shape, one line per true angle."""

    windows = res["windows"]
    years = np.array([int(w) for w in windows])

    fig, axes = plt.subplots(
        len(ROWS), len(RAMPS), figsize=(6.2 * len(RAMPS), 2.7 * len(ROWS)),
        squeeze=False, sharex=True,
    )

    for r, name in enumerate(ROWS):
        j = ad.CONTROL_HEAD.index(name)

        # a shared y range per row makes the three ramps actually comparable,
        # which is the entire point of putting them side by side
        lo, hi = np.inf, -np.inf
        for ramp in RAMPS:
            for th in THETAS:
                v, ref = _series(res, ramp, th, j, kind, windows)
                lo = min(lo, np.nanmin(v), ref)
                hi = max(hi, np.nanmax(v), ref)
        if kind == "median":
            for ramp in RAMPS:
                for th in THETAS:
                    t = res[ramp][th][windows[0]]["truth"][j]
                    lo, hi = min(lo, t), max(hi, t)
        pad = 0.08 * (hi - lo) if hi > lo else 0.1
        ylim = (lo - pad, hi + pad)

        for c, ramp in enumerate(RAMPS):
            ax = axes[r][c]
            for y in years:
                ax.axvline(y, linestyle="dotted", color="grey", linewidth=1.25)

            for i, th in enumerate(THETAS):
                v, ref = _series(res, ramp, th, j, kind, windows)
                ax.plot(
                    years,
                    v,
                    color=COLORS[i + 1],
                    linestyle="solid",
                    marker="o",
                    markersize=6,
                    label=(
                        r"$\vartheta^\dagger=${}$^\circ$".format(th)
                        if (r == 0 and c == 0)
                        else None
                    ),
                )
                if kind == "median":
                    ax.axhline(
                        res[ramp][th][windows[0]]["truth"][j],
                        linestyle="dashed",
                        linewidth=1.25,
                        color=COLORS[i + 1],
                    )

            if kind == "bias":
                ax.axhline(0, color=COLORS[0], linewidth=1.5)
            elif kind == "spread":
                ax.axhline(1.0, color=REF, linestyle="dashdot", linewidth=1.5)

            ax.set_xlim(TMIN - 2, years[-1] + 3)
            ax.set_ylim(*ylim)
            if c == 0:
                ax.set_ylabel(
                    CONTROL_LABELS[name] if kind == "median"
                    else f"{CONTROL_LABELS[name]}\n{KIND_YLAB[kind]}",
                    fontsize=16,
                )
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(RAMP_LABELS[ramp], fontsize=19)
            if r == len(ROWS) - 1:
                ax.set_xlabel("Year")

    handles, labels = axes[0][0].get_legend_handles_labels()
    # the legend and the two-line suptitle have to be separated explicitly in
    # figure coordinates, otherwise they render on top of each other
    fig.legend(
        handles, labels, loc="upper center", ncols=len(THETAS),
        bbox_to_anchor=(0.5, 1.005), fontsize=16,
    )
    fig.suptitle(
        f"{KIND_TITLES[kind]}\n{MODEL}  |  {SCENARIO}  |  {WINDOWING}  |  "
        f"TMIN {TMIN}  |  {NOISE_MODEL}  |  DEGpDEC {DEG_P_DEC}  |  "
        f"cost-ratio screen $\\leq$ {SCREEN}",
        fontsize=17, y=1.055,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.982))

    if not save:
        return None
    out = make_figure_filename(
        f"ramp-param-learning-{kind}-{MODEL}-{WINDOWING}", outdir=FIGS_DIR / "results"
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out}")
    return out


def report(res):
    """Numbers behind the figure: bias and spread at the last window."""

    windows = res["windows"]
    wl = windows[-1]
    print()
    print(f"Screen kept (of {res[RAMPS[0]][THETAS[0]][windows[0]]['n_ens']}):")
    for ramp in RAMPS:
        k = [res[ramp][th][wl]["keep"] for th in THETAS]
        print(f"  {ramp:6s} window {wl}: {k}")

    # How far the prior already sits from the truth, before any assimilation.
    # A control whose prior is centered on the truth starts unbiased, so any
    # bias in it is leakage; a control whose prior is offset starts biased and
    # the plot mostly shows how much of that offset got closed. These differ
    # per true angle because truth.yaml sets ALPHA_TR/BETA_TR per angle.
    w0 = windows[0]
    print()
    print(f"Prior offset from truth at window {w0}, "
          f"(prior med $-$ truth) / sigma_prior  (columns = theta {THETAS}):")
    for name in ROWS:
        j = ad.CONTROL_HEAD.index(name)
        vals = []
        for th in THETAS:
            d = res[RAMPS[0]][th][w0]
            vals.append((d["prior_med"][j] - d["truth"][j]) / d["prior_sd"][j])
        flag = "   <-- prior not centered on truth" if max(
            abs(v) for v in vals
        ) > 0.5 else ""
        print(f"  {name:9s} " + "  ".join(f"{x:+7.3f}" for x in vals) + flag)

    for kind in ("bias", "spread"):
        print()
        print(f"{kind} at window {wl}, by control and ramp "
              f"(columns = theta {THETAS}):")
        for name in ROWS:
            j = ad.CONTROL_HEAD.index(name)
            print(f"  {name}")
            for ramp in RAMPS:
                vals = [
                    _series(res, ramp, th, j, kind, windows)[0][-1]
                    for th in THETAS
                ]
                print(f"    {ramp:6s} " + "  ".join(f"{x:+7.3f}" for x in vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-save", action="store_true", help="do not write figures")
    cli = ap.parse_args()
    save = not cli.no_save

    presets, _ = get_presets()
    plt.rcParams.update(presets)

    print("collecting ...", flush=True)
    res = collect()

    print("making figures ...", flush=True)
    for kind in ("median", "bias", "spread"):
        make_figure(res, kind, save)

    report(res)


if __name__ == "__main__":
    sys.exit(main())
