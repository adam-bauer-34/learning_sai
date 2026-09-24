"""Diagnose the common upward offset in the recovered SAI angle at window 2040.

The ensemble-median posterior angle at the first assimilation window sits at
16.5-16.9 degrees for every true angle, whether the truth is 5 or 35 degrees --
about +1.2 degrees above the prior angle median. The fan-out toward the truth
after 2040 is expected; the shared offset at 2040 is what this script is for.

Produces seven figures and a text/CSV report:

  1. learning curves for all 15 fixed controls
  2. the window-2040 shift fingerprint across the six true angles
  3. angle-offset attribution (first-order and exact substitution)
  4. linear-Gaussian prediction vs measurement (shift and spread)
  5. the internal-variability realization the runs share
  6. cost-function breakdown
  7. optimizer health

To run:
    python analysis/scripts/angle_drift_diagnostics.py [--model MODEL] [--no-save]

`--model` picks the model whose output is analyzed (default `pco2geowc_reg`;
also supports `pco2geowc_reg_noic`). Output filenames carry the model name.

Scope: ssp245 / DEGpDEC 0.1 / `four` windowing / Nens 1000, the six most recent
runs. This is post-processing I/O only -- no assimilation -- so it is safe to run
interactively.

Adam Michael Bauer
UChicago
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from var_assim.config import FIGS_DIR
from var_assim.emis import EmissionsBaseline
from var_assim.plotting import angle_diagnostics as ad
from var_assim.plotting.presets import get_presets
from var_assim.plotting.utils import make_figure_filename

# models the diagnostics support: same 15-entry control head and obs operator
MODELS = ["pco2geowc_reg", "pco2geowc_reg_noic"]
THETAS = [5, 11, 15, 21, 30, 35]
SCREEN = 0.5
SCREEN_ALTS = [1.0, 0.1]
PLO, PHI = 5, 95

# Okabe-Ito, as mandated by presets.py and used throughout the notebooks.
# Index 0 (black) is reserved for reference lines; true angle i takes [i + 1].
# Verified colorblind-safe for this six-series subset: adjacent-pair OKLab dE
# >= 11.3 under protanopia, deuteranopia and tritanopia (target 8), and
# >= 18.4 for normal vision (floor 15).
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

# LaTeX labels for the 15 fixed controls, matching the notebook conventions
CONTROL_LABELS = {
    "T1": r"$T_1(0)$",
    "T2": r"$T_2(0)$",
    "Q": r"$Q(0)$",
    "T_R1": r"$T_{R1}(0)$",
    "T_R2": r"$T_{R2}(0)$",
    "L": r"$\lambda$",
    "G": r"$\gamma$",
    "EPS": r"$\varepsilon$",
    "C1": r"$C_1$",
    "C2": r"$C_2$",
    "F1_CO2": r"$f^{(CO_2)}_1$",
    "ALPHA_R1": r"$\alpha_{R1}$",
    "ALPHA_R2": r"$\alpha_{R2}$",
    "BETA_R1": r"$\beta_{R1}$",
    "BETA_R2": r"$\beta_{R2}$",
}


def theta_label(th):
    return r"$\vartheta^\dagger=${}$^\circ$".format(th)


# ----------------------------------------------------------------------------
# collection
# ----------------------------------------------------------------------------


def collect(logger, model):
    """Read every window of every run and assemble the diagnostic quantities.

    Parameters
    ----------
    model: str
        model whose output is analyzed, one of `MODELS`

    Returns
    -------
    res: dict
        nested `res[theta][window]`, plus shared entries under `res["shared"]`
    """

    args, Noise, Prior, Truth0, Windowing = ad.build_calibration(THETAS[0], model=model)
    paths = {th: ad.output_path(th, model=model) for th in THETAS}
    get_nonlin_path, get_obs_from_dynamics = ad.model_dynamics(model)
    q_offset = ad.model_q_offset(model)

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing input files:\n" + "\n".join(missing))

    ad.check_provenance(list(paths.values()), Noise)
    logger.info("provenance OK: derived obs weighting, all files postdate 85a2a91")

    windows = [str(w[1]) for w in Windowing.windows]
    sigmas = ad.obs_sigmas(Noise)
    ramp_end = args.tmin + args.n_yrs_ramp

    res = {"shared": {}}
    res["shared"].update(
        model=model,
        args=args,
        Noise=Noise,
        Prior=Prior,
        windows=windows,
        window_years=np.array([int(w) for w in windows]),
        sigmas=sigmas,
        ramp_end=ramp_end,
    )

    # the emissions baseline and the observation Jacobian depend only on the
    # window and the prior centres, not on the true angle, so both are built
    # once per window and shared across the six runs
    for th in THETAS:
        res[th] = {}

    for w in windows:
        TMIN, TMAX = args.tmin, int(w)
        e = EmissionsBaseline(
            logger,
            args,
            TMIN,
            TMAX,
            geo=True,
            Prior=Prior,
            Truth=Truth0,
            T_START=TMIN,
            T_END=ramp_end,
            print_level=0,
        )
        n_times = len(e.conc["CO2"])
        inv_covar_prior, prior_stds = ad.build_inv_covar_prior(
            Prior, Noise, n_times, q_offset
        )

        H = None
        x_cen = None

        for th in THETAS:
            win = ad.load_window(paths[th], w, want_data_final=True)
            keep = ad.screen_members(win["cost_hist"], SCREEN)

            head = win["head"]
            post = win["controls"][keep][:, head]
            prior = win["controls_prior"][keep][:, head]
            truth = win["controls_truth"][head]

            # x_cen is estimated from the unscreened prior draws, which are
            # generated from the same seed and the same prior centre in every
            # run -- so it is angle-independent, and so are H and the Hessian
            this_x_cen = ad.prior_centre_from_ensemble(win)
            if x_cen is None:
                x_cen = this_x_cen
                H = ad.obs_jacobian(e, x_cen, TMIN, TMAX, model=model)
                A, Rinv = ad.gauss_newton_system(H, inv_covar_prior, sigmas, n_times)
                pred_sd = np.sqrt(np.diag(np.linalg.inv(A)))
                paths_cen, _ = get_nonlin_path(e, x_cen, TMIN, TMAX, 1.0)
                obs_cen = get_obs_from_dynamics(paths_cen)
            elif not np.allclose(this_x_cen, x_cen, atol=1e-8):
                raise AssertionError(
                    f"prior centre differs between runs at window {w}; the shared "
                    "Jacobian and Hessian would be invalid"
                )

            # linear-Gaussian posterior-mean shift for this run's observations
            innovation = (win["obs"] - obs_cen).ravel()
            dx_pred = ad.kalman_shift(H, A, Rinv, innovation)

            cb = ad.cost_breakdown(win, keep, inv_covar_prior, sigmas)

            prior_med = ad.median_dict(prior)
            post_med = ad.median_dict(post)
            stds = {p: prior_stds[ad.CONTROL_HEAD.index(p)] for p in ad.ANGLE_PARAMS}
            attrib = ad.attribute_angle_offset(prior_med, post_med, stds)

            ang_post = ad.angle_from_controls(post)
            ang_prior = ad.angle_from_controls(prior)
            # the >90 mask the notebooks apply before taking medians
            ang_post_f = np.where(ang_post > 90, np.nan, ang_post)
            ang_prior_f = np.where(ang_prior > 90, np.nan, ang_prior)

            # exact group substitution, as a check on the first-order split
            subs = {
                g: ad.substitute_angle(prior_med, post_med, grp)
                for g, grp in ad.ANGLE_GROUPS.items()
            }

            # predicted angle offset from the linear-Gaussian shift
            pred_med = {
                p: x_cen[ad.CONTROL_HEAD.index(p)] + dx_pred[ad.CONTROL_HEAD.index(p)]
                for p in ad.ANGLE_PARAMS
            }
            cen_med = {p: x_cen[ad.CONTROL_HEAD.index(p)] for p in ad.ANGLE_PARAMS}

            res[th][w] = dict(
                n_times=n_times,
                keep=int(keep.sum()),
                n_ens=int(keep.size),
                post_med=np.median(post, axis=0),
                post_p_lo=np.percentile(post, PLO, axis=0),
                post_p_hi=np.percentile(post, PHI, axis=0),
                post_sd=post.std(axis=0),
                prior_med=np.median(prior, axis=0),
                prior_p_lo=np.percentile(prior, PLO, axis=0),
                prior_p_hi=np.percentile(prior, PHI, axis=0),
                prior_sd=prior.std(axis=0),
                prior_stds_head=prior_stds[: ad.N_FIXED],
                truth=truth,
                angle_med=float(np.nanmedian(ang_post_f)),
                angle_lo=float(np.nanpercentile(ang_post_f, PLO)),
                angle_hi=float(np.nanpercentile(ang_post_f, PHI)),
                angle_prior_med=float(np.nanmedian(ang_prior_f)),
                angle_of_median=ad.angle_at(post_med),
                angle_of_prior_median=ad.angle_at(prior_med),
                n_over_90=int(np.sum(ang_post > 90)),
                attrib=attrib,
                subs=subs,
                dx_pred=dx_pred,
                dx_meas=np.median(post, axis=0) - np.median(prior, axis=0),
                pred_sd_head=pred_sd[: ad.N_FIXED],
                angle_pred=ad.angle_at(pred_med),
                angle_cen=ad.angle_at(cen_med),
                cost={
                    k: float(np.median(v))
                    for k, v in cb.items()
                    if isinstance(v, np.ndarray) and v.ndim == 1
                },
                cost_max_resid=cb["max_rel_residual"],
                frac_post_below_truth=float(np.mean(cb["J_total"] < cb["J_truth"])),
                cost_ratio=win["cost_hist"][:, -1] / win["cost_hist"][:, 0],
                realization=ad.realization_stats(
                    win["controls_truth"], win["time"], n_times
                ),
                geo=ad.geo_information_split(e, win["time"], ramp_end),
                time=win["time"],
            )

            # screen sensitivity, recorded once per run/window
            alts = {}
            for r in SCREEN_ALTS:
                k2 = ad.screen_members(win["cost_hist"], r)
                a2 = ad.angle_from_controls(win["controls"][k2][:, head])
                alts[r] = (
                    float(np.nanmedian(np.where(a2 > 90, np.nan, a2))),
                    int(k2.sum()),
                )
            res[th][w]["screen_alts"] = alts

            del win
        logger.info(f"window {w} done (n_times={n_times})")

    # true angles, from the saved truth control vectors
    res["shared"]["angle_truth"] = {
        th: ad.angle_at(
            {
                p: res[th][windows[0]]["truth"][ad.CONTROL_HEAD.index(p)]
                for p in ad.ANGLE_PARAMS
            }
        )
        for th in THETAS
    }
    return res


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------


def _mark_windows(ax, years):
    for y in years:
        ax.axvline(y, linestyle="dotted", color="grey", linewidth=1.25)


def fig_learning_curves(res, save, names, tag, title):
    """Posterior median, normalized bias and spread ratio, one row per control.

    No uncertainty band in column A: six overlapping bands occlude the medians,
    and the spread is column C's job anyway.
    """

    sh = res["shared"]
    years = sh["window_years"]
    tmin = sh["args"].tmin

    fig, axes = plt.subplots(
        len(names), 3, figsize=(18, 2.7 * len(names)), squeeze=False
    )

    for r, name in enumerate(names):
        j = ad.CONTROL_HEAD.index(name)
        a0, a1, a2 = axes[r]
        for ax in (a0, a1, a2):
            _mark_windows(ax, years)
            ax.set_xlim(tmin - 2, years[-1] + 2)

        for i, th in enumerate(THETAS):
            c = COLORS[i + 1]
            med = np.array([res[th][w]["post_med"][j] for w in sh["windows"]])
            sd_p = np.array([res[th][w]["prior_sd"][j] for w in sh["windows"]])
            tr = res[th][sh["windows"][0]]["truth"][j]
            pm = res[th][sh["windows"][0]]["prior_med"][j]

            # prior pinned at tmin, so the first window's movement is visible
            a0.plot(
                np.hstack([[tmin], years]),
                np.hstack([[pm], med]),
                color=c,
                linestyle="solid",
                marker="o",
                markersize=6,
                label=theta_label(th) if r == 0 else None,
            )
            a0.axhline(tr, linestyle="dashed", linewidth=1.25, color=c)

            a1.plot(
                np.hstack([[tmin], years]),
                np.hstack([[(pm - tr) / sd_p[0]], (med - tr) / sd_p]),
                color=c,
                linestyle="solid",
                marker="o",
                markersize=6,
            )

            ratio = np.array(
                [
                    (res[th][w]["post_p_hi"][j] - res[th][w]["post_p_lo"][j])
                    / (res[th][w]["prior_p_hi"][j] - res[th][w]["prior_p_lo"][j])
                    for w in sh["windows"]
                ]
            )
            a2.plot(
                years, ratio, color=c, linestyle="solid", marker="o", markersize=6
            )

        a0.set_ylabel(CONTROL_LABELS[name], fontsize=21)
        a1.axhline(0, color=COLORS[0], linewidth=1.5)
        a1.set_ylabel(r"bias / $\sigma_{\rm prior}$", fontsize=14)
        # 1.0 means "nothing learned"; 0 means "fully determined by the data"
        a2.axhline(1.0, color=REF, linestyle="dashdot", linewidth=1.5)
        a2.set_ylabel("spread ratio", fontsize=14)
        a2.set_ylim(0, 1.15)

        if r == 0:
            a0.set_title("posterior median (dashed = truth)", fontsize=16)
            a1.set_title("normalized bias vs truth", fontsize=16)
            a2.set_title("posterior / prior spread", fontsize=16)
        if r == len(names) - 1:
            for ax in (a0, a1, a2):
                ax.set_xlabel("Year")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncols=len(THETAS),
        bbox_to_anchor=(0.5, 1.0),
        fontsize=16,
    )
    fig.suptitle(title, fontsize=19, y=1.018)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _save(fig, res, tag, save)


def fig_shift_fingerprint(res, save):
    """Window-2040 prior->posterior shift for all 15 controls, six truths overlaid.

    The point of the figure: most controls shift by the same amount regardless of
    the true angle, which is what marks those shifts as driven by the shared
    internal-variability realization rather than by the angle signal.
    """

    sh = res["shared"]
    w0 = sh["windows"][0]
    names = ad.CONTROL_HEAD
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 1, figsize=(15, 11))
    ax, axb = axes

    for i, th in enumerate(THETAS):
        d = res[th][w0]
        shift = d["dx_meas"] / d["prior_stds_head"]
        ax.plot(
            x,
            shift,
            marker="o",
            markersize=9,
            linewidth=1.8,
            linestyle="solid",
            color=COLORS[i + 1],
            label=theta_label(th),
        )
        bias = (d["post_med"] - d["truth"]) / d["prior_stds_head"]
        axb.plot(
            x,
            bias,
            marker="s",
            markersize=9,
            linewidth=1.8,
            linestyle="solid",
            color=COLORS[i + 1],
        )

    for a in (ax, axb):
        a.axhline(0, color=COLORS[0], linewidth=1.5)
        a.set_xticks(x)
        a.set_xticklabels([CONTROL_LABELS[n] for n in names], fontsize=17)
        a.set_xlim(-0.5, len(names) - 0.5)

    ax.set_ylabel(
        r"(post $-$ prior) median / $\sigma_{\rm prior}$", fontsize=18
    )
    ax.set_title(
        f"Window {w0}: prior to posterior shift  "
        "(lines coinciding = shift independent of the true angle)",
        fontsize=17,
    )
    axb.set_ylabel(r"(post $-$ truth) median / $\sigma_{\rm prior}$", fontsize=18)
    axb.set_title(f"Window {w0}: posterior bias against truth", fontsize=17)

    # legend below the panels: at upper centre it lands on the T1(0) spike
    ax.legend(
        ncols=len(THETAS),
        fontsize=15,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.tight_layout()
    return _save(fig, res, "angle-diag-2-shift-fingerprint-2040", save)


def fig_attribution(res, save):
    """Which parameter carries how much of the angle offset, in degrees."""

    sh = res["shared"]
    w0 = sh["windows"][0]
    params = ad.ANGLE_PARAMS
    x = np.arange(len(params))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    ax, axc, axs = axes

    width = 0.13
    for i, th in enumerate(THETAS):
        at = res[th][w0]["attrib"]
        vals = [at["contrib_deg"][p] for p in params]
        ax.bar(
            x + (i - 2.5) * width,
            vals,
            width=width * 0.88,
            color=COLORS[i + 1],
            label=theta_label(th),
        )

    ax.axhline(0, color=COLORS[0], linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([CONTROL_LABELS[p] for p in params], fontsize=17)
    ax.set_ylabel("contribution to angle offset (degrees)", fontsize=17)
    ax.set_title(f"A. First-order attribution, window {w0}", fontsize=17)
    ax.legend(ncols=3, fontsize=13, loc="upper center", bbox_to_anchor=(0.5, -0.09))

    # closure: exact offset vs the first-order sum, all windows
    years = sh["window_years"]
    for i, th in enumerate(THETAS):
        ex = [res[th][w]["attrib"]["offset_exact"] for w in sh["windows"]]
        li = [res[th][w]["attrib"]["offset_linear"] for w in sh["windows"]]
        axc.plot(
            years, ex, color=COLORS[i + 1], linestyle="solid", label=theta_label(th)
        )
        axc.plot(years, li, color=COLORS[i + 1], linestyle="dotted", linewidth=2.5)
    _mark_windows(axc, years)
    axc.axhline(0, color=COLORS[0], linewidth=1.5)
    axc.set_xlabel("Year")
    axc.set_ylabel(r"med$(\vartheta)_{\rm post} - $med$(\vartheta)_{\rm prior}$ (deg)")
    axc.set_title("B. Offset: exact (solid) vs first-order (dotted)", fontsize=17)

    # exact group substitution. bar geometry: `gw` keeps the six bars inside
    # their group so neighbouring groups cannot run together
    groups = list(ad.ANGLE_GROUPS)
    xg = np.arange(len(groups))
    gw = 0.8 / len(THETAS)
    for i, th in enumerate(THETAS):
        d = res[th][w0]
        base = d["angle_of_prior_median"]
        vals = [d["subs"][g] - base for g in groups]
        axs.bar(
            xg + (i - (len(THETAS) - 1) / 2) * gw,
            vals,
            width=gw * 0.9,
            color=COLORS[i + 1],
            label=theta_label(th),
        )
    axs.axhline(0, color=COLORS[0], linewidth=1.5)
    axs.set_xticks(xg)
    axs.set_xticklabels(
        [r"$\alpha$ only", r"$\beta$ only", r"$\lambda,\gamma,\varepsilon$ only"],
        fontsize=16,
    )
    axs.set_ylabel("angle offset (degrees)")
    axs.set_title(f"C. Exact group substitution, window {w0}", fontsize=17)

    fig.tight_layout()
    return _save(fig, res, "angle-diag-3-attribution", save)


def fig_linear_gaussian(res, save):
    """Is the measured shift the linear-Gaussian posterior response?"""

    sh = res["shared"]
    w0 = sh["windows"][0]
    names = ad.CONTROL_HEAD
    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7.0))
    ax, axsd, axang = axes

    # predicted vs measured is two estimates of one quantity, so it reads as a
    # scatter against the 1:1 line: on the line means linear theory accounts for
    # the shift, off it means something else produced it. Shown for the middle
    # truth, since panel A of figure 2 establishes the shifts barely depend on it.
    d = res[15][w0]
    pr = d["dx_pred"][: ad.N_FIXED] / d["prior_stds_head"]
    me = d["dx_meas"] / d["prior_stds_head"]

    # a scatter against 1:1 crowds every control into a cluster next to the
    # T1(0) point, so the comparison is drawn as paired horizontal bars instead:
    # equal-length pairs mean linear theory accounts for the shift
    y = np.arange(len(names))[::-1]
    bh = 0.38
    ax.barh(
        y + bh / 2,
        pr,
        height=bh,
        color=COLORS[5],
        label="predicted (linear-Gaussian)",
    )
    ax.barh(y - bh / 2, me, height=bh, color=COLORS[1], label="measured")
    # flag the controls the prediction misses: wrong sign, or off by more than
    # half the measured shift. only shifts worth explaining in the first place
    # are eligible, so a control that barely moved is not flagged on round-off.
    unexplained = (np.abs(me) >= 0.05) & (
        (np.sign(pr) != np.sign(me)) | (np.abs(me - pr) > 0.5 * np.abs(me))
    )
    xr = float(np.abs(np.hstack([pr, me])).max())
    for j in range(len(names)):
        if unexplained[j]:
            ax.text(
                xr * 1.04,
                y[j],
                "not predicted",
                va="center",
                fontsize=13,
                color=COLORS[6],
            )
    ax.axvline(0, color=COLORS[0], linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([CONTROL_LABELS[n] for n in names], fontsize=17)
    ax.set_xlim(-xr * 1.15, xr * 1.5)
    ax.set_xlabel(r"shift / $\sigma_{\rm prior}$", fontsize=17)
    ax.set_title(rf"A. Shift at {w0}, $\vartheta^\dagger=15^\circ$", fontsize=17)
    ax.legend(fontsize=13, loc="lower right")

    # same treatment for the posterior spread
    prs = d["pred_sd_head"] / d["prior_stds_head"]
    mes = d["post_sd"] / d["prior_stds_head"]
    axsd.plot([0, 1.2], [0, 1.2], color=COLORS[0], linewidth=1.5, zorder=1)
    for j, n in enumerate(names):
        axsd.scatter(
            prs[j],
            mes[j],
            s=150,
            color=COLORS[5],
            zorder=5,
            edgecolor="white",
            linewidth=1.5,
        )
        axsd.annotate(
            CONTROL_LABELS[n],
            (prs[j], mes[j]),
            textcoords="offset points",
            xytext=(9, 6),
            fontsize=15,
        )
    axsd.set_xlim(0, 1.2)
    axsd.set_ylim(0, 1.2)
    axsd.set_xlabel(r"predicted sd / $\sigma_{\rm prior}$", fontsize=17)
    axsd.set_ylabel(r"measured sd / $\sigma_{\rm prior}$", fontsize=17)
    axsd.set_title(f"B. Posterior spread at {w0}", fontsize=17)

    # angle offset: measured vs linear-Gaussian prediction, per window
    years = sh["window_years"]
    for i, th in enumerate(THETAS):
        meas = [
            res[th][w]["angle_med"] - res[th][w]["angle_prior_med"]
            for w in sh["windows"]
        ]
        pred = [
            res[th][w]["angle_pred"] - res[th][w]["angle_cen"] for w in sh["windows"]
        ]
        axang.plot(
            years,
            meas,
            color=COLORS[i + 1],
            linestyle="solid",
            marker="o",
            markersize=7,
            label=theta_label(th),
        )
        axang.plot(years, pred, color=COLORS[i + 1], linestyle="dotted", linewidth=2.5)
    _mark_windows(axang, years)
    axang.axhline(0, color=COLORS[0], linewidth=1.5)
    axang.set_xlabel("Year")
    axang.set_ylabel("angle offset vs prior (degrees)")
    axang.set_title(
        "C. Angle offset: measured (solid) vs predicted (dotted)", fontsize=16
    )
    axang.legend(ncols=2, fontsize=13)

    fig.tight_layout()
    return _save(fig, res, "angle-diag-4-linear-gaussian", save)


def fig_realization(res, save):
    """The single internal-variability realization every run is fitted to."""

    sh = res["shared"]
    wlast = sh["windows"][-1]
    d = res[THETAS[0]][wlast]
    time = d["time"]
    years = sh["window_years"]
    Noise = sh["Noise"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.0))
    ax, axo, axg = axes

    block_colors = {"qAT": COLORS[5], "qR1": COLORS[1], "qR2": COLORS[3]}
    block_labels = {"qAT": r"$q_{AT}$ (global)", "qR1": r"$q_{R1}$", "qR2": r"$q_{R2}$"}
    # shade the first window: it is short enough that the realization's
    # low-frequency structure cannot average out, which is the whole point
    ax.axvspan(
        time[0], years[0], color=REF, alpha=0.12, linewidth=0, zorder=0
    )
    ax.text(
        (time[0] + years[0]) / 2,
        ax.get_ylim()[1],
        f"window {years[0]}",
        ha="center",
        va="bottom",
        fontsize=13,
    )
    for name in ad.ERROR_BLOCKS:
        ax.plot(
            d["realization"][name]["time"],
            d["realization"][name]["series"],
            color=block_colors[name],
            linestyle="solid",
            linewidth=1.6,
            label=block_labels[name],
        )
    _mark_windows(ax, years)
    ax.axhline(0, color=COLORS[0], linewidth=1.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("model error (K)")
    ax.set_title("A. The realization all six runs share", fontsize=16)
    ax.legend(fontsize=13, ncols=3, loc="lower center")

    # assumed observation error against the variability it represents
    rows = ad.internal_variability_sigmas(Noise)
    xo = np.arange(len(rows))
    axo.bar(
        xo - 0.2,
        [r[1] for r in rows],
        width=0.38,
        color=COLORS[5],
        label=r"$\sigma_{\rm obs}$ (assumed)",
    )
    axo.bar(
        xo + 0.2,
        [r[2] for r in rows],
        width=0.38,
        color=COLORS[1],
        label="internal variability (actual)",
    )
    axo.set_ylim(0, max(max(r[1], r[2]) for r in rows) * 1.42)
    for k, r in enumerate(rows):
        axo.text(
            k, max(r[1], r[2]) * 1.04, f"{r[3]:.1f}x too tight",
            ha="center", fontsize=14,
        )
    axo.set_xticks(xo)
    axo.set_xticklabels([r[0] for r in rows], fontsize=16)
    axo.set_ylabel("standard deviation (K)")
    axo.set_title("B. Observations weighted tighter than their own noise", fontsize=16)
    axo.legend(fontsize=13, loc="upper left")

    # geo information, ramp vs plateau
    xw = np.arange(len(sh["windows"]))
    ramp = [res[THETAS[0]][w]["geo"]["ramp"] for w in sh["windows"]]
    plat = [res[THETAS[0]][w]["geo"]["plateau"] for w in sh["windows"]]
    axg.bar(xw, ramp, width=0.6, color=COLORS[3], label=f"ramp (<= {sh['ramp_end']})")
    axg.bar(
        xw,
        plat,
        width=0.6,
        bottom=ramp,
        color=COLORS[4],
        label=f"plateau (> {sh['ramp_end']})",
    )
    axg.set_xticks(xw)
    axg.set_xticklabels(sh["windows"], fontsize=16)
    axg.set_xlabel("Window")
    axg.set_ylabel(r"$\sum_t {\rm geo}(t)^2$")
    axg.set_title(r"C. Information available on $\beta_{R2}$", fontsize=16)
    axg.legend(fontsize=13, loc="upper left")

    fig.tight_layout()
    return _save(fig, res, "angle-diag-5-realization", save)


def fig_cost(res, save):
    """Cost-function breakdown: shares, goodness of fit, and J against J(truth)."""

    sh = res["shared"]
    years = sh["window_years"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.0))
    axs, axc, axj = axes

    # stacked shares of J, for the middle truth
    th_ref = 15
    terms = ["J_prior_head", "J_prior_qAT", "J_prior_qR1", "J_prior_qR2"] + [
        f"J_obs_{v}" for v in ad.OBS_VARS
    ]
    labels = [
        r"prior: params",
        r"prior: $q_{AT}$",
        r"prior: $q_{R1}$",
        r"prior: $q_{R2}$",
        r"obs: $T_1$",
        r"obs: $Q$",
        r"obs: $T_{R1}$",
        r"obs: $T_{R2}$",
    ]
    tcolors = [COLORS[5], COLORS[2], COLORS[1], COLORS[6], COLORS[3], COLORS[4],
               "#8c8c8c", "#c9a0c0"]
    xw = np.arange(len(sh["windows"]))
    bottom = np.zeros(len(sh["windows"]))
    # shares are taken against the sum of the per-term medians, not against
    # median(J): the median of a sum is not the sum of the medians, so the
    # latter would leave the bars a few percent short of 1
    denom = np.array(
        [sum(res[th_ref][w]["cost"][t] for t in terms) for w in sh["windows"]]
    )
    for t, lab, c in zip(terms, labels, tcolors):
        vals = np.array(
            [res[th_ref][w]["cost"][t] for w in sh["windows"]]
        ) / denom
        axs.bar(
            xw,
            vals,
            width=0.6,
            bottom=bottom,
            color=c,
            label=lab,
            edgecolor="white",
            linewidth=1.2,
        )
        bottom += vals
    axs.set_xticks(xw)
    axs.set_xticklabels(sh["windows"], fontsize=16)
    axs.set_xlabel("Window")
    axs.set_ylabel("share of median $J$")
    axs.set_ylim(0, 1.0)
    axs.set_title(rf"A. Cost composition, $\vartheta^\dagger={th_ref}^\circ$", fontsize=16)
    axs.legend(fontsize=12, ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    # reduced chi-square per observation term
    ocolors = {"T1": COLORS[5], "Q": COLORS[4], "T_R1": COLORS[1], "T_R2": COLORS[3]}
    olabels = {"T1": r"$T_1$", "Q": r"$Q$", "T_R1": r"$T_{R1}$", "T_R2": r"$T_{R2}$"}
    for v in ad.OBS_VARS:
        vals = [res[th_ref][w]["cost"][f"chi2red_{v}"] for w in sh["windows"]]
        axc.plot(
            years,
            vals,
            marker="o",
            markersize=9,
            linestyle="solid",
            color=ocolors[v],
            label=olabels[v],
        )
    _mark_windows(axc, years)
    # 1.0 = misfit matches the assumed sigma; below it the observable is over-fit
    axc.axhline(1.0, color=COLORS[0], linewidth=1.5)
    axc.set_xlabel("Year")
    axc.set_ylabel(r"reduced $\chi^2$")
    axc.set_ylim(0, 1.1)
    axc.set_title(r"B. Fit quality ($<1$ = over-fit)", fontsize=16)
    axc.legend(fontsize=14, ncols=4, loc="upper center")

    # J(posterior) against J(truth)
    for i, th in enumerate(THETAS):
        ratio = [
            res[th][w]["cost"]["J_total"] / res[th][w]["cost"]["J_truth"]
            for w in sh["windows"]
        ]
        axj.plot(
            years,
            ratio,
            marker="o",
            markersize=8,
            linestyle="solid",
            color=COLORS[i + 1],
            label=theta_label(th),
        )
    _mark_windows(axj, years)
    axj.axhline(1.0, color=COLORS[0], linewidth=1.5)
    axj.set_ylim(0, 1.1)
    axj.set_xlabel("Year")
    axj.set_ylabel(r"median $J_{\rm post}$ / median $J_{\rm truth}$")
    axj.set_title(r"C. Below 1 = posterior beats the truth on cost", fontsize=16)
    axj.legend(ncols=2, fontsize=13, loc="lower right")

    fig.tight_layout()
    return _save(fig, res, "angle-diag-6-cost-breakdown", save)


def fig_optimizer(res, save):
    """Optimizer health: cost-ratio distribution and screen sensitivity."""

    sh = res["shared"]
    years = sh["window_years"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.0))
    axh, axn, axsc = axes

    w0 = sh["windows"][0]
    for i, th in enumerate(THETAS):
        r = res[th][w0]["cost_ratio"]
        r = r[np.isfinite(r)]
        axh.hist(
            r,
            bins=np.linspace(0, 1.0, 41),
            histtype="step",
            linewidth=2.2,
            color=COLORS[i + 1],
            label=theta_label(th),
        )
    axh.axvline(SCREEN, color=COLORS[0], linewidth=1.8)
    axh.set_xlabel(r"$J_{\rm final} / J_{\rm iter 0}$")
    axh.set_ylabel("members")
    axh.set_title(f"A. Cost ratio at window {w0} (line = screen {SCREEN})", fontsize=16)
    axh.legend(fontsize=13, loc="upper right", bbox_to_anchor=(1.0, 0.92))

    # members the screen removes. `costs` is float64 and never non-finite even
    # for the blown-up members, so counting non-finite costs would report zero
    # everywhere -- the screen's own reject count is the informative number.
    for i, th in enumerate(THETAS):
        n = [res[th][w]["n_ens"] - res[th][w]["keep"] for w in sh["windows"]]
        axn.plot(
            years,
            n,
            marker="o",
            markersize=9,
            linestyle="solid",
            color=COLORS[i + 1],
            label=theta_label(th),
        )
    _mark_windows(axn, years)
    axn.set_xlabel("Year")
    axn.set_ylabel("members removed by the screen")
    axn.set_title(f"B. Rejects at cost ratio > {SCREEN} (of 1000)", fontsize=16)
    axn.legend(ncols=2, fontsize=13)

    # screen sensitivity of the headline statistic
    markers = {1.0: "o", SCREEN: "s", 0.1: "^"}
    for i, th in enumerate(THETAS):
        for r, mk in markers.items():
            if r == SCREEN:
                vals = [res[th][w]["angle_med"] for w in sh["windows"]]
            else:
                vals = [res[th][w]["screen_alts"][r][0] for w in sh["windows"]]
            axsc.plot(
                years,
                vals,
                marker=mk,
                markersize=8,
                linewidth=1.4,
                color=COLORS[i + 1],
                linestyle="solid" if r == SCREEN else "dotted",
            )
    _mark_windows(axsc, years)
    axsc.set_xlabel("Year")
    axsc.set_ylabel(r"med$(\vartheta)$ (degrees)")
    axsc.set_title(
        r"C. Screen sensitivity: $\leq$1 (circle), 0.5 (square), 0.1 (triangle)",
        fontsize=15,
    )

    fig.tight_layout()
    return _save(fig, res, "angle-diag-7-optimizer", save)


def _save(fig, res, name, save):
    if not save:
        return None
    out = make_figure_filename(
        f"{name}-{res['shared']['model']}", outdir=FIGS_DIR / "results"
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {out}")
    return out


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------


def write_report(res, save):
    """Text and CSV report of the cost breakdown and the headline numbers."""

    sh = res["shared"]
    L = []

    def p(s=""):
        L.append(s)

    p("=" * 78)
    p("ANGLE DRIFT DIAGNOSTICS")
    p("=" * 78)
    p(f"model      : {sh['model']}")
    p(f"scope      : ssp245 / DEGpDEC {sh['args'].deg_p_dec} / "
      f"{sh['args'].windowing} / Nens {sh['args'].n_ens}")
    p(f"screen     : cost ratio <= {SCREEN}")
    p(f"obs weight : {sh['Noise'].OBS_WEIGHTING}")
    s = sh["sigmas"]
    p(f"obs sigmas : T1={s[0]:.4g}  Q={s[1]:.4g}  T_R1={s[2]:.4g}  T_R2={s[3]:.4g}")
    p(f"SAI ramp   : {sh['args'].tmin} -> {sh['ramp_end']}, then held constant")
    p()

    p("-" * 78)
    p("1. VALIDATION: recomputed J against archived `costs`")
    p("-" * 78)
    worst = 0.0
    for th in THETAS:
        row = "  ".join(
            f"{res[th][w]['cost_max_resid']:.2e}" for w in sh["windows"]
        )
        worst = max(worst, max(res[th][w]["cost_max_resid"] for w in sh["windows"]))
        p(f"  theta {th:2d}: max |dJ|/J per window = {row}")
    p(f"  worst overall: {worst:.2e}")
    p("  (x_f is stored as float32, so ~1e-7 is the floor; the decomposition is exact)")
    p()

    p("-" * 78)
    p("2. MEDIAN ANGLE BY WINDOW")
    p("-" * 78)
    p("  theta*  truth  " + "  ".join(f"{w:>7s}" for w in sh["windows"])
      + "   prior_med")
    for th in THETAS:
        row = "  ".join(f"{res[th][w]['angle_med']:7.2f}" for w in sh["windows"])
        p(f"  {th:5d} {sh['angle_truth'][th]:7.2f}  {row}   "
          f"{res[th][sh['windows'][0]]['angle_prior_med']:7.2f}")
    w0 = sh["windows"][0]
    a0 = [res[th][w0]["angle_med"] for th in THETAS]
    p(f"  window {w0} spread across truths: {max(a0) - min(a0):.2f} deg")
    p(f"  window {w0} mean offset vs prior median: "
      f"{np.mean([res[th][w0]['angle_med'] - res[th][w0]['angle_prior_med'] for th in THETAS]):+.2f} deg")
    p()

    p("-" * 78)
    p("2b. ESTIMATOR CHECKS (is the offset an artefact of the statistic?)")
    p("-" * 78)
    p("  med(angle) is a median of a nonlinear, non-negative function, so it need")
    p("  not equal angle(median parameters). a small gap rules out folding bias.")
    p("  theta  win   med(ang)  ang(med)      gap   5-95 range   n(ang>90)")
    for th in THETAS:
        for w in sh["windows"]:
            d = res[th][w]
            p(f"  {th:5d} {w}  {d['angle_med']:9.3f} {d['angle_of_median']:9.3f} "
              f"{d['angle_med'] - d['angle_of_median']:+8.3f}   "
              f"{d['angle_hi'] - d['angle_lo']:10.2f}   {d['n_over_90']:9d}")
    gaps = [
        abs(res[th][w]["angle_med"] - res[th][w]["angle_of_median"])
        for th in THETAS
        for w in sh["windows"]
    ]
    p(f"  largest |gap| over all runs and windows: {max(gaps):.3f} deg")
    p(f"  total members masked by the >90 deg cut: "
      f"{sum(res[th][w]['n_over_90'] for th in THETAS for w in sh['windows'])}")
    p()

    p("-" * 78)
    p(f"3. ANGLE OFFSET ATTRIBUTION AT WINDOW {w0} (degrees)")
    p("-" * 78)
    hdr = "  param     " + "  ".join(f"{('th'+str(th)):>8s}" for th in THETAS)
    p(hdr)
    for prm in ad.ANGLE_PARAMS:
        row = "  ".join(
            f"{res[th][w0]['attrib']['contrib_deg'][prm]:+8.3f}" for th in THETAS
        )
        p(f"  {prm:9s} {row}")
    p("  " + "-" * (len(hdr) - 2))
    for key, lab in (("offset_linear", "1st-order"), ("offset_exact", "exact"),
                     ("residual", "residual")):
        row = "  ".join(f"{res[th][w0]['attrib'][key]:+8.3f}" for th in THETAS)
        p(f"  {lab:9s} {row}")
    p()

    p("-" * 78)
    p(f"4. LINEAR-GAUSSIAN PREDICTION vs MEASUREMENT AT WINDOW {w0}")
    p("-" * 78)
    p("  shifts in units of prior sigma; pred = K(y - H x_cen)")
    p("  control    " + "  ".join(f"{('th'+str(th)+' p/m'):>16s}" for th in (5, 35)))
    for j, n in enumerate(ad.CONTROL_HEAD):
        cells = []
        for th in (5, 35):
            d = res[th][w0]
            cells.append(
                f"{d['dx_pred'][j] / d['prior_stds_head'][j]:+7.3f}/"
                f"{d['dx_meas'][j] / d['prior_stds_head'][j]:+7.3f}"
            )
        p(f"  {n:10s} " + "  ".join(f"{c:>16s}" for c in cells))
    p()
    p("  angle offset vs prior (degrees): measured / predicted")
    for th in THETAS:
        cells = "  ".join(
            f"{res[th][w]['angle_med'] - res[th][w]['angle_prior_med']:+6.2f}/"
            f"{res[th][w]['angle_pred'] - res[th][w]['angle_cen']:+6.2f}"
            for w in sh["windows"]
        )
        p(f"    theta {th:2d}: {cells}")
    p()

    p("-" * 78)
    p("5. COST BREAKDOWN (ensemble medians)")
    p("-" * 78)
    cols = [
        ("J_total", "J_tot"),
        ("J_prior", "J_pri"),
        ("J_obs", "J_obs"),
        ("J_truth", "J_true"),
        ("J_prior_head", "pri_par"),
        ("J_prior_qAT", "pri_qAT"),
        ("J_prior_qR1", "pri_qR1"),
        ("J_prior_qR2", "pri_qR2"),
        ("J_obs_T1", "obs_T1"),
        ("J_obs_Q", "obs_Q"),
        ("J_obs_T_R1", "obs_TR1"),
        ("J_obs_T_R2", "obs_TR2"),
    ]
    p("  th  win   " + "  ".join(f"{lab:>8s}" for _, lab in cols)
      + "   " + "  ".join(f"x2r_{v:<5s}" for v in ad.OBS_VARS))
    csv = ["theta,window," + ",".join(k for k, _ in cols) + ","
           + ",".join(f"chi2red_{v}" for v in ad.OBS_VARS)
           + ",frac_J_post_below_J_truth,max_rel_resid,n_keep"]
    for th in THETAS:
        for w in sh["windows"]:
            c = res[th][w]["cost"]
            p(f"  {th:2d} {w}  " + "  ".join(f"{c[k]:8.3g}" for k, _ in cols)
              + "   " + "  ".join(f"{c['chi2red_'+v]:9.4f}" for v in ad.OBS_VARS))
            csv.append(
                f"{th},{w}," + ",".join(f"{c[k]:.8g}" for k, _ in cols) + ","
                + ",".join(f"{c['chi2red_'+v]:.8g}" for v in ad.OBS_VARS)
                + f",{res[th][w]['frac_post_below_truth']:.4f}"
                + f",{res[th][w]['cost_max_resid']:.3e},{res[th][w]['keep']}"
            )
    p()
    p("  fraction of members with J(posterior) < J(truth):")
    for th in THETAS:
        row = "  ".join(
            f"{res[th][w]['frac_post_below_truth']:6.3f}" for w in sh["windows"]
        )
        p(f"    theta {th:2d}: {row}")
    p()

    p("-" * 78)
    p("6. THE SHARED INTERNAL-VARIABILITY REALIZATION")
    p("-" * 78)
    for w in sh["windows"]:
        r = res[THETAS[0]][w]["realization"]
        p(f"  window {w}: " + "   ".join(
            f"{b}: mean {r[b]['mean']:+.4f} std {r[b]['std']:.4f} "
            f"trend {r[b]['trend']:+.5f}/yr" for b in ad.ERROR_BLOCKS))
    p()
    p("  assumed obs error vs actual internal variability:")
    for name, so, si, f in ad.internal_variability_sigmas(sh["Noise"]):
        p(f"    {name:5s} sigma_obs {so:.4g}  internal {si:.4g}  "
          f"over-weighted {f:.2f}x")
    p()
    p("  geo information split (sum_t geo^2):")
    for w in sh["windows"]:
        g = res[THETAS[0]][w]["geo"]
        p(f"    window {w}: ramp {g['ramp']:10.1f} ({g['n_ramp']:2d} yr)   "
          f"plateau {g['plateau']:10.1f} ({g['n_plateau']:2d} yr)")
    p()

    p("-" * 78)
    p("7. SCREEN SENSITIVITY OF med(theta)")
    p("-" * 78)
    for r in [1.0, SCREEN, 0.1]:
        p(f"  cost ratio <= {r}:")
        for th in THETAS:
            if r == SCREEN:
                vals = [(res[th][w]["angle_med"], res[th][w]["keep"])
                        for w in sh["windows"]]
            else:
                vals = [res[th][w]["screen_alts"][r] for w in sh["windows"]]
            p(f"    theta {th:2d}: "
              + "  ".join(f"{v:6.2f} (n={n})" for v, n in vals))
    p()

    text = "\n".join(L)
    print(text)

    if save:
        out = Path(make_figure_filename(f"angle-diag-report-{sh['model']}", FIGS_DIR / "results", "txt"))
        out.write_text(text + "\n")
        print(f"\n  report -> {out}")
        outc = Path(
            make_figure_filename(f"angle-diag-cost-{sh['model']}", FIGS_DIR / "results", "csv")
        )
        outc.write_text("\n".join(csv) + "\n")
        print(f"  cost csv -> {outc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="pco2geowc_reg",
        choices=MODELS,
        help="model whose output is analyzed",
    )
    ap.add_argument("--no-save", action="store_true", help="do not write outputs")
    cli = ap.parse_args()
    save = not cli.no_save

    logging.basicConfig(
        level=logging.INFO, format="%(message)s", stream=sys.stdout
    )
    logger = logging.getLogger("angle_diag")

    presets, _ = get_presets()
    plt.rcParams.update(presets)

    logger.info("collecting ...")
    logger.info(f"model: {cli.model}")
    res = collect(logger, cli.model)

    logger.info("making figures ...")
    others = [n for n in ad.CONTROL_HEAD if n not in ad.ANGLE_PARAMS]
    fig_learning_curves(
        res,
        save,
        [n for n in ad.CONTROL_HEAD if n in ad.ANGLE_PARAMS],
        "angle-diag-1a-learning-curves-angle-params",
        "Controls that enter the angle",
    )
    fig_learning_curves(
        res,
        save,
        others,
        "angle-diag-1b-learning-curves-other",
        "Remaining controls (do not enter the angle directly)",
    )
    fig_shift_fingerprint(res, save)
    fig_attribution(res, save)
    fig_linear_gaussian(res, save)
    fig_realization(res, save)
    fig_cost(res, save)
    fig_optimizer(res, save)

    write_report(res, save)


if __name__ == "__main__":
    main()
