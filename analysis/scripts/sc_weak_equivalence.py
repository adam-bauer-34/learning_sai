"""Equivalence of the strong-constraint profile mode and the weak-constraint model.

For the flux design, `pco2geosc_reg_noic --sc_covar profile` minimizes exactly the
weak-constraint cost of `pco2geowc_reg_noic` with the model errors eliminated:

    J_prof(theta) = min_q J_w(theta, q),   argmin_q = q*(theta)

This script tests that against existing weak output, in three parts:

  8. draws    the regenerated background draws equal the weak file's prior
              (controls_hist at iter 0), member by member
  9. weak     no new runs: at each weak solution (theta_w, q_w),
              J_prof(theta_w) <= J_w(theta_w, q_w). The gap,
              1/2 |q_w - q*(theta_w)|_H^2, measures how far the weak inner
              problem is from converged. The projected grad J_prof(theta_w) ~ 0.
              Also counts weak members with active q bounds (+-1.2) and any that
              fall below the strong model's C1/C2 floors.
 10. runs     (if --sc_file is given) the strong profile run's theta and xi*
              against the weak run's theta and q, member by member

To run (inside a compute allocation; reads one 1000-member weak file):
    python analysis/scripts/sc_weak_equivalence.py --theta 15 --sai_ramp linear \
        [--sc_file data/output/pco2geosc_reg_noic/...sc-flux-profile-match_weak.nc]

Adam Michael Bauer
UChicago
"""

import argparse
import logging
import sys

import netCDF4 as nc
import numpy as np

from var_assim.calibration.noise import ClimateModelNoise
from var_assim.calibration.priors import ClimateModelPriors
from var_assim.calibration.truth import ClimateModelTruth
from var_assim.calibration.windowing import AssimilationWindowing
from var_assim.config import (
    DATA_DIR_ABS,
    NOISE_PATH,
    PRIOR_PATH,
    TRUTH_PATH,
    WINDOW_PATH,
)
from var_assim.emis import EmissionsBaseline
from var_assim.model_errors import get_window_prefix_inds
from var_assim.models.pco2geosc_reg_noic import cost as sc
from var_assim.models.pco2geosc_reg_noic.dynamics import N_FIXED
from var_assim.models.pco2geosc_reg_noic.noise_response import NoiseModel
# the floors the strong model once had; kept only to report how many weak
# members sit below them
C1_MIN, C2_MIN = 2.0, 10.0
from var_assim.models.pco2geosc_reg_noic.runner import make_shared_draws
from var_assim.models.pco2geowc_reg_noic.dynamics import get_nonlin_path as wc_path
from var_assim.stats.covar import get_covar_white
from var_assim.warm_start import warm_start_simulation

NAMES = ["T1", "T2", "Q", "T_R1", "T_R2", "L", "G", "EPS", "C1", "C2", "F1_CO2",
         "ALPHA_R1", "ALPHA_R2", "BETA_R1", "BETA_R2"]
Q_BOUND = 1.2
SCREEN = 0.5  # the cost-ratio screen used throughout the analysis


def weak_path(theta, ramp):
    return (
        DATA_DIR_ABS / "output" / "pco2geowc_reg_noic" /
        f"var-assim-output_ssp245_pco2geowc_reg_noic_gradual_AR1+reg_TMIN2025_THETA{theta}"
        f"_ECS3.0_ramprate{ramp}_DEGpDEC0.1_NYRSRAMP50_Nens1000.nc"
    )


def projected(g, theta, lower):
    """Zero the gradient components pushing out through an active lower bound."""

    g = g.copy()
    at = np.isfinite(lower) & (theta <= lower + 1e-10) & (g > 0)
    g[at] = 0.0
    return g


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theta", type=int, default=15)
    ap.add_argument("--sai_ramp", default="linear", choices=["linear", "fast", "slow"])
    ap.add_argument("--sc_file", default=None, help="strong-constraint profile-mode output")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger = logging.getLogger("sc_eq")

    wfile = weak_path(cli.theta, cli.sai_ramp)
    wds = nc.Dataset(wfile)
    n_ens = wds.groups[sorted(wds.groups, key=int)[0]].dimensions["ens_mem"].size

    # the namespace the weak run was launched with, so the warm start and the
    # draws are regenerated exactly
    args = argparse.Namespace(
        model="pco2geowc_reg_noic", scenario="ssp245", noise_model="AR1", tmin=2025,
        theta=cli.theta, ecs=3.0, deg_p_dec=0.1, n_yrs_ramp=50, windowing="gradual",
        n_ens=n_ens, reg_noise=True, sai_ramp=cli.sai_ramp, debug=False,
    )
    Noise = ClimateModelNoise.from_cli_and_yaml(args, NOISE_PATH)
    Prior = ClimateModelPriors.from_cli_and_yaml_and_noise(args, PRIOR_PATH, Noise)
    Truth = ClimateModelTruth.from_cli_and_yaml(args, TRUTH_PATH)
    Windowing = AssimilationWindowing.from_cli_and_yaml(args, WINDOW_PATH)
    DT = 1.0

    warm_start_simulation(logger, args, Truth, Prior, wc_path)
    draws = make_shared_draws(args, Prior, Noise, Windowing, DT)

    sig_obs = np.array([Noise.OBS_T1_STD, Noise.OBS_Q_STD, *Noise.OBS_T_REG_STD])
    sig_reg = np.array(Noise.INT_T_REG_STD, dtype=float)
    sig_prior = np.asarray(Prior.controls_std, dtype=float)
    invB = get_covar_white(sig_prior, N_FIXED, inv=True)
    theta_ref = np.asarray(Prior.controls_cen, dtype=float).copy()

    weak_lower = np.full(N_FIXED, -np.inf)
    weak_lower[5:13] = 0.0

    sds = nc.Dataset(cli.sc_file) if cli.sc_file else None

    print(f"weak file: {wfile.name}  (Nens {n_ens})")
    if sds is not None:
        print(f"sc file  : {cli.sc_file.split('/')[-1]}")
    print()

    all_ok = True
    for w in sorted(wds.groups, key=int):
        g = wds.groups[w]
        TMIN, TMAX = args.tmin, int(w)
        e = EmissionsBaseline(logger, args, TMIN, TMAX, geo=True, Prior=Prior,
                              Truth=Truth, T_START=TMIN, T_END=TMIN + args.n_yrs_ramp,
                              print_level=0)
        N = len(e.conc["CO2"])
        M = N - 1

        prefix = get_window_prefix_inds(N_FIXED, draws["N_BLOCKS"], draws["N_MAX"], N,
                                        q_offset=1)
        x_f = draws["theta_prior_full"][:, prefix]

        # ---- 8. draws ------------------------------------------------------------
        prior_w = np.asarray(g.variables["controls_hist"][:, :, 0], dtype=float)
        d8 = float(np.max(np.abs(prior_w - x_f) / np.maximum(np.abs(x_f), 1.0)))
        ok8 = d8 < 1e-6  # the weak file stores the draw as float32

        # ---- 9. weak solutions under the profile cost -----------------------------
        controls_w = np.asarray(g.variables["controls"][:], dtype=float)
        costs_w = np.asarray(g.variables["costs"][:], dtype=float)
        obs = np.asarray(g.variables["obs"][:], dtype=float)

        Q_g = draws["mod_error_covar_full"][1:N, 1:N]
        noise = NoiseModel("flux", sig_obs, sig_reg, Q_g, N, DT)
        ref = noise.factor(noise.L_g(theta_ref))

        gaps, gnorm, gnorm0 = [], [], []
        for i in range(n_ens):
            th_w = controls_w[i, :N_FIXED]
            ctx = sc.SCContext(noise, "profile", x_f[i, N_FIXED:], np.zeros(4 * N), ref)
            a = [x_f[i, :N_FIXED], invB, ctx, obs, e, TMIN, TMAX, DT]
            Jp = sc.cost(th_w, a)
            gaps.append(costs_w[i] - Jp)
            gp = projected(sc.grad(th_w, a), th_w, weak_lower) * sig_prior
            gnorm.append(np.linalg.norm(gp))
            g0 = sc.grad(x_f[i, :N_FIXED], a) * sig_prior
            gnorm0.append(np.linalg.norm(g0))
        gaps = np.array(gaps)
        rel_g = np.array(gnorm) / np.array(gnorm0)
        Jp_all = costs_w - gaps

        q_w = controls_w[:, N_FIXED:]
        n_qbound = int(np.sum(np.any(np.abs(q_w) >= Q_BOUND - 1e-8, axis=1)))
        below = (controls_w[:, 8] < C1_MIN) | (controls_w[:, 9] < C2_MIN)

        # the usual screen: final/initial cost ratio. members failing it are the
        # weak model's blow-ups, whose solutions are not meaningful minima
        ch_w = np.asarray(g.variables["cost_hist"][:], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_w = ch_w[:, -1] / ch_w[:, 0]
        good_w = np.isfinite(ratio_w) & (ratio_w <= SCREEN)

        # weak solutions where the strong model's covariance can't be formed
        # (explicit Euler unstable): J_prof is only the sentinel there
        unevaluable = Jp_all >= sc.BIG_COST

        use = good_w & ~unevaluable
        # negative gaps beyond round-off would mean the identity is violated
        ok9 = bool(np.all(gaps[use] > -1e-6 * np.maximum(np.abs(costs_w[use]), 1.0)))
        all_ok &= ok8 and ok9

        print(f"window {w} (N={N})")
        print(f"   8. draws: max rel |weak prior - regenerated| = {d8:.1e}  "
              f"{'PASS' if ok8 else 'FAIL'}")
        print(f"   9. weak members passing the cost-ratio <= {SCREEN} screen: "
              f"{int(good_w.sum())}/{n_ens}; of those, strong model can't evaluate "
              f"theta_w: {int((good_w & unevaluable).sum())}")
        print(f"      J_w - J_prof(theta_w) >= 0 over the {int(use.sum())} usable: "
              f"min {gaps[use].min():.3e}  median {np.median(gaps[use]):.3e}  "
              f"max {gaps[use].max():.3e}  {'PASS' if ok9 else 'FAIL'}")
        rg = rel_g[use & np.isfinite(rel_g)]
        print(f"      |proj grad J_prof(theta_w)| / |grad at prior|: "
              f"median {np.median(rg):.2e}  95% {np.percentile(rg, 95):.2e}  "
              f"max {rg.max():.2e}")
        print(f"      weak members with a q at the +-{Q_BOUND} bound: {n_qbound}/{n_ens}")
        print(f"      below the strong floors (C1<{C1_MIN} or C2<{C2_MIN}): "
              f"{int(below.sum())}/{n_ens} total, {int((below & good_w).sum())} of them "
              f"pass the screen" + (
                  "  -> " + ", ".join(
                      f"#{i} C1={controls_w[i, 8]:.2f} C2={controls_w[i, 9]:.1f}"
                      for i in np.where(below & good_w)[0][:6])
                  if (below & good_w).any() else ""))

        # ---- 10. strong profile run vs weak run -----------------------------------
        if sds is not None and w in sds.groups:
            s = sds.groups[w]
            n_sc = s.dimensions["ens_mem"].size
            prior_s = np.asarray(s.variables["controls_hist"][:, :, 0], dtype=float)
            d10a = float(np.max(np.abs(prior_s - x_f[:n_sc, :N_FIXED])
                                / np.maximum(np.abs(x_f[:n_sc, :N_FIXED]), 1.0)))
            th_s = np.asarray(s.variables["controls"][:], dtype=float)
            xi_s = np.asarray(s.variables["model_error_hat"][:], dtype=float)
            costs_s = np.asarray(s.variables["costs"][:], dtype=float)
            dth = np.abs(th_s - controls_w[:n_sc, :N_FIXED]) / sig_prior
            ch_s = np.asarray(s.variables["cost_hist"][:], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_s = ch_s[:, -1] / ch_s[:, 0]
            good_s = np.isfinite(ratio_s) & (ratio_s <= SCREEN)
            flag_s = np.asarray(s.variables["flag"][:])
            no_qb = ~np.any(np.abs(q_w[:n_sc]) >= Q_BOUND - 1e-8, axis=1)
            free = good_w[:n_sc] & good_s & no_qb
            dq = np.max(np.abs(xi_s - q_w[:n_sc]), axis=1)
            dJ = costs_s - costs_w[:n_sc]
            print(f"  10. sc run vs weak, members 0..{n_sc - 1}: compared {int(free.sum())} "
                  f"(both pass the screen, no active weak q bound)")
            print(f"      excluded: weak fails screen {int((~good_w[:n_sc]).sum())}, "
                  f"sc fails screen {int((~good_s).sum())} (sc diverged, flag 2: "
                  f"{int((flag_s == 2).sum())})")
            bf = free & below[:n_sc]
            if bf.any():
                print(f"      of those compared, {int(bf.sum())} have weak C1<{C1_MIN} or "
                      f"C2<{C2_MIN}; their max |d theta|/sigma: "
                      + ", ".join(f"#{i} {np.max(dth[i]):.1e}" for i in np.where(bf)[0]))
            # members only the strong run's screen rejects: if they still match the
            # weak solution, the rejection is an artifact of the screen, not a failure
            only_s = good_w[:n_sc] & ~good_s & no_qb
            if only_s.any():
                mx = np.max(dth[only_s], axis=1)
                print(f"      rejected only by the sc screen: {int(only_s.sum())}; their "
                      f"max |d theta|/sigma vs weak: median {np.median(mx):.2e}, "
                      f"max {mx.max():.2e}; sc cost ratio median "
                      f"{np.median(ratio_s[only_s]):.2f} (weak {np.median(ratio_w[:n_sc][only_s]):.2f})")
            both_bad = (~good_w[:n_sc]) & (~good_s)
            print(f"      of the weak blow-ups, sc converged for "
                  f"{int(((~good_w[:n_sc]) & good_s).sum())}; both failed for "
                  f"{int(both_bad.sum())}")
            print(f"      draws match: max rel diff {d10a:.1e}")
            print(f"      |d theta| / sigma_prior over free members: "
                  f"median {np.median(dth[free]):.2e}  95% {np.percentile(dth[free], 95):.2e}"
                  f"  max {dth[free].max():.2e}")
            worst = np.argsort(-np.max(dth[free], axis=0))[:3]
            print("      largest per-control max: " + ", ".join(
                f"{NAMES[k]} {np.max(dth[free][:, k]):.2e}" for k in worst))
            print(f"      max |xi* - q_w| over free members: median {np.median(dq[free]):.2e}"
                  f"  max {dq[free].max():.2e}")
            print(f"      J_sc - J_w over compared: median {np.median(dJ[free]):.3e}  "
                  f"min {dJ[free].min():.3e}  max {dJ[free].max():.3e}  "
                  f"(<= 0 means sc found an equal-or-lower minimum)")
            big = free & (np.max(dth, axis=1) > 0.1)
            if big.any():
                print(f"      compared members differing by > 0.1 sigma: {int(big.sum())}; "
                      f"J_sc - J_w for them: " + ", ".join(
                          f"#{i} {dJ[i]:+.3g}" for i in np.where(big)[0][:8]))
        print()

    wds.close()
    if sds is not None:
        sds.close()
    print("OVERALL (checks 8 and 9):", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
