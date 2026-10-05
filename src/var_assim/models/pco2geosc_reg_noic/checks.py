"""Component checks specific to the strong-constraint model.

These complement tlm_adj_checks.run_component_checks, which tests the noiseless
TLM/adjoint and the configured mode's gradient. Here every (design x mode) is
tested, together with the pieces the generic harness can't see:

  affine      y(theta, xi) = m(theta) + L(theta) xi exactly; the flux design's
              forced path equals pco2geowc_reg_noic's; T2 gets no direct noise
  L           analytic dL/dtheta_k against central differences, exactly zero for
              the eight controls L doesn't depend on; unit tests on the Q row
  TLM         flux forced TLM equals the weak model's TLM[:15, :15] on the same
              path; temp forced TLM against a one-step finite difference
  adjoint     dot-product identity of the forced TLM/adjoint pair
  negative    a gradient that runs the noiseless trajectory instead of the
              xi*-forced one must FAIL (it drops the dS/dtheta term)
  profile     J_w(theta, xi) - J_prof(theta) = 1/2 |xi - xi*|_H^2;
              R_o^{-1}(y(theta, xi*) - data) = v;
              flux: the weak model's own grad at [theta, xi*] reproduces the
              profile gradient (and marginal minus its log-det term) exactly,
              with q-part ~ 0
  gradient    central differences of the cost, every design x mode, with and
              without measurement perturbation, at two points (one with small C1)
  logdet      log-det gradient alone against differences of log det S
  truth       at theta_tr with xi_f = xi_tr: d = 0, v = 0, xi* = xi_tr; the
              flux design's truth obs equal the weak model's bit-for-bit

Run standalone (lightweight, no assimilation):
    python -m var_assim.models.pco2geosc_reg_noic.checks [--tmax 2100]

Adam Michael Bauer
UChicago
"""

import argparse
import csv
import logging

import numpy as np

from var_assim.models.pco2geowc_reg_noic import cost as wc_cost
from var_assim.models.pco2geowc_reg_noic import dynamics as wc_dyn
from var_assim.stats.covar import get_covar_white

from . import cost as sc
from .adjoint import OBS_ROWS, get_forced_adj
from .dynamics import N_FIXED, get_forced_path, get_forced_TLM_matrix, get_nonlin_path
from .noise_response import DESIGNS, I_C1, L_PARAMS, NoiseModel
from .obs import get_obs_from_dynamics

THRESHOLDS = {
    "affine": 1e-12,
    "flux_path_eq_weak": 0.0,
    "noiseless_eq_weak": 0.0,
    "T2_no_direct_noise": 0.0,
    "dL_vs_fd": 1e-5,
    "dL_zero": 0.0,
    "Q_row_unit": 1e-12,
    "tlm_eq_weak": 1e-13,
    "tlm_vs_fd": 1e-6,
    "adjoint_dot": 1e-12,
    "profile_gap": 1e-9,
    "profile_v": 1e-9,
    "weak_grad_theta": 1e-9,
    "weak_grad_q": 1e-9,
    "grad_vs_fd": 1e-6,
    "logdet_vs_fd": 1e-6,
    "truth": 1e-10,
    "truth_obs_eq_weak": 0.0,
    # inverted: the wrong gradient must be off by MORE than this
    "negative_control": 1e-4,
    # d^T S^{-1} d at parameters with violently unstable dynamics: must stay >= 0
    # (the naive Woodbury product goes large and negative there)
    "quad_nonneg": 0.0,
}


def _obs(path):
    return get_obs_from_dynamics(path).ravel()


def _full_L(noise, L_g):
    """Dense L = [L_g | regional shifted identities], (4N, 3M)."""

    N, M = noise.N, noise.M
    L = np.zeros((4 * N, 3 * M))
    L[:, :M] = L_g
    for r in range(2):
        b = 2 + r
        L[b * N + 1 : (b + 1) * N, (1 + r) * M : (2 + r) * M] = np.eye(M)
    return L


def _fd(f, theta, k, rel=1e-6):
    h = rel * max(abs(theta[k]), 1.0)
    tp, tm = theta.copy(), theta.copy()
    tp[k] += h
    tm[k] -= h
    return (f(tp) - f(tm)) / (2 * h)


def _rel(a, b):
    """max |a - b| scaled by max |b| (absolute when b is ~0)."""

    scale = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / scale


def run_sc_checks(
    logger,
    args,
    e,
    TMIN,
    TMAX,
    DT,
    Prior,
    Noise,
    Q_g_flux,
    q_tr,
    controls_tr,
    obs_wc=None,
    check_dir=None,
    seed=0,
):
    """Run every check for both designs and all three modes.

    Parameters
    ----------
    Q_g_flux: (M, M) array
        the window's AR(1) prior covariance of the heat-flux noise (weak units)

    q_tr: (3M,) array
        the truth's noise realization in weak units [q_AT, q_R1, q_R2]

    obs_wc: (4, N) array or None
        the weak model's truth observations, for the bit-for-bit comparison

    Returns
    -------
    passed: bool
    """

    from .runner import load_t1_noise_std

    rng = np.random.default_rng(seed)
    times = np.arange(TMIN, TMAX + DT, DT)
    N = len(times)
    M = N - 1

    sig_obs = np.array([Noise.OBS_T1_STD, Noise.OBS_Q_STD, *Noise.OBS_T_REG_STD])
    sig_reg = np.array(Noise.INT_T_REG_STD, dtype=float)
    invB = get_covar_white(np.asarray(Prior.controls_std), N_FIXED, inv=True)
    theta_ref = np.asarray(Prior.controls_cen, dtype=float).copy()
    controls_tr = np.asarray(controls_tr, dtype=float)

    # two test points: off the truth, and one with C1 near (but above) its floor
    theta_a = controls_tr * 1.1
    theta_b = controls_tr.copy()
    theta_b[I_C1] = 2.5
    points = {"x1.1": theta_a, "smallC1": theta_b}

    rows = []

    def record(check, design, mode, value, point=""):
        thr = THRESHOLDS[check]
        ok = bool(value > thr) if check == "negative_control" else bool(value <= thr)
        rows.append((check, design, mode, point, value, thr, ok))

    for design in DESIGNS:
        scale = 1.0 if design == "flux" else load_t1_noise_std() / Noise.INT_VAR_STD
        noise = NoiseModel(design, sig_obs, sig_reg, Q_g_flux * scale**2, N, DT)
        Qxi = noise.Q_blocks()
        Qxi_inv = np.linalg.inv(Qxi)
        ref_factor = noise.factor(noise.L_g(theta_ref))

        def draw_xi():
            return rng.multivariate_normal(np.zeros(3 * M), Qxi)

        # ---- affine exactness, and agreement with the weak model --------------
        for pname, th in points.items():
            xi = draw_xi()
            p, _ = get_forced_path(e, th, xi, design, TMIN, TMAX, DT)
            p0, _ = get_nonlin_path(e, th, TMIN, TMAX, DT)
            L_g = noise.L_g(th)
            y = _obs(p)
            record("affine", design, "", _rel(_obs(p0) + noise.apply_L(L_g, xi), y), pname)

            if design == "flux":
                pw, _ = wc_dyn.get_nonlin_path(e, np.hstack([th, xi]), TMIN, TMAX, DT)
                record("flux_path_eq_weak", design, "",
                       float(np.max(np.abs(pw[:N_FIXED] - p))), pname)
                pw0, _ = wc_dyn.get_nonlin_path(
                    e, np.hstack([th, np.zeros(3 * M)]), TMIN, TMAX, DT
                )
                record("noiseless_eq_weak", design, "",
                       float(np.max(np.abs(pw0[:N_FIXED] - p0))), pname)

            # a kick in step k -> k+1 must leave T2 at k+1 untouched: T2 has no
            # noise term of its own and only sees T1 one step later
            k = M // 2
            xi_k = np.zeros(3 * M)
            xi_k[k] = 1.0
            pk, _ = get_forced_path(e, th, xi_k, design, TMIN, TMAX, DT)
            record("T2_no_direct_noise", design, "",
                   float(abs(pk[1, k + 1] - p0[1, k + 1])), pname)

        # ---- L and its derivatives -------------------------------------------
        th = theta_a
        dL = noise.dL_g(th)
        worst, zero = 0.0, 0.0
        for k in range(N_FIXED):
            fd = _fd(noise.L_g, th, k)
            if k in L_PARAMS:
                worst = max(worst, _rel(dL[k], fd))
            else:
                zero = max(zero, float(np.max(np.abs(fd))))
        record("dL_vs_fd", design, "", worst)
        record("dL_zero", design, "", zero)

        # first response of Q to a global kick: Dt for flux (C1 * Dt/C1), C1 for
        # temp (C1 * 1); its C1-derivative is 0 and 1 respectively
        L_g = noise.L_g(th)
        q_first = L_g[N + 1, 0]
        want = DT if design == "flux" else th[I_C1]
        dq_first = dL[I_C1][N + 1, 0]
        want_d = 0.0 if design == "flux" else 1.0
        record("Q_row_unit", design, "",
               max(abs(q_first - want) / abs(want), abs(dq_first - want_d)))

        # ---- TLM ---------------------------------------------------------------
        xi = draw_xi()
        p, _ = get_forced_path(e, th, xi, design, TMIN, TMAX, DT)
        g = xi[:M]
        if design == "flux":
            pw, _ = wc_dyn.get_nonlin_path(e, np.hstack([th, xi]), TMIN, TMAX, DT)
            worst = 0.0
            for t in range(N - 1):
                Tw = wc_dyn.get_TLM_matrix(e, t, pw, DT)[:N_FIXED, :N_FIXED]
                Ts = get_forced_TLM_matrix(e, t, p, g, design, DT)
                worst = max(worst, _rel(Ts, Tw))
            record("tlm_eq_weak", design, "", worst)

        # one-step finite difference of the forced map, at a few steps
        worst = 0.0
        for t in (0, (N - 1) // 2, N - 2):
            T = get_forced_TLM_matrix(e, t, p, g, design, DT)
            J = np.empty((N_FIXED, N_FIXED))
            for k in range(N_FIXED):
                h = 1e-6 * max(abs(p[k, t]), 1.0)
                xp, xm = p[:, t].copy(), p[:, t].copy()
                xp[k] += h
                xm[k] -= h
                J[:, k] = (_one_step(e, xp, g[t], t, design, DT)
                           - _one_step(e, xm, g[t], t, design, DT)) / (2 * h)
            worst = max(worst, _rel(T, J))
        record("tlm_vs_fd", design, "", worst)

        # dot-product identity: <H M dtheta, w> = <dtheta, lam_0(w)>
        dth = rng.standard_normal(N_FIXED)
        w = rng.standard_normal((4, N))
        dx = dth.copy()
        lhs = float(w[:, 0] @ dx[list(OBS_ROWS)])
        for t in range(N - 1):
            dx = get_forced_TLM_matrix(e, t, p, g, design, DT) @ dx
            lhs += float(w[:, t + 1] @ dx[list(OBS_ROWS)])
        lam = get_forced_adj(e, p, g, design, w, DT)
        rhs = float(dth @ lam[:, 0])
        record("adjoint_dot", design, "", abs(lhs - rhs) / max(abs(lhs), 1e-300))

        # ---- per-mode checks ---------------------------------------------------
        y_obs = _obs(get_forced_path(e, controls_tr, draw_xi(), design, TMIN, TMAX, DT)[0])
        theta_f = theta_ref.copy()
        xi_f = draw_xi()

        for mode in sc.MODES:
            for meas_kind in ("none", "meas"):
                meas = (np.zeros(4 * N) if meas_kind == "none"
                        else np.repeat(sig_obs, N) * rng.standard_normal(4 * N))
                ctx = sc.SCContext(noise, mode, xi_f, meas, ref_factor)
                a = [theta_f, invB, ctx, y_obs.reshape(4, N), e, TMIN, TMAX, DT]
                tag = f"{mode}/{meas_kind}"

                for pname, thp in points.items():
                    gr = sc.grad(thp, a)
                    fd = np.array([_fd(lambda x: sc.cost(x, a), thp, k)
                                   for k in range(N_FIXED)])
                    record("grad_vs_fd", design, tag, _rel(gr, fd), pname)

                    # negative control: the noiseless trajectory in place of the
                    # xi*-forced one, which silently drops -1/2 v^T dS v
                    if mode != "fixed":
                        ev = sc._evaluate(thp, a)
                        p0, _ = get_nonlin_path(e, thp, TMIN, TMAX, DT)
                        lam0 = get_forced_adj(
                            e, p0, np.zeros(M), design, ev["v"].reshape(4, N), DT
                        )
                        bad = invB @ (thp - theta_f) + lam0[:, 0]
                        if mode == "marginal":
                            bad = bad + sc.logdet_grad(thp, ctx, ev["fac"])
                        record("negative_control", design, tag, _rel(bad, fd), pname)

                # far outside the stable region (the kind of point SLSQP's first,
                # identity-Hessian step can reach), the quadratic form must not
                # go negative. the dynamics can overflow outright out there, in
                # which case the factorization raises and cost() returns its
                # sentinel instead -- also fine
                wild = theta_a.copy()
                wild[5], wild[6], wild[7], wild[8] = 98.1, 198.0, 47.9, 13.3
                try:
                    q_wild = sc._evaluate(wild, a)["quad"]
                    record("quad_nonneg", design, tag, max(0.0, -q_wild))
                except (np.linalg.LinAlgError, ValueError):
                    record("quad_nonneg", design, tag, 0.0)

                if mode == "marginal":
                    ev = sc._evaluate(theta_a, a)
                    lg = sc.logdet_grad(theta_a, ctx, ev["fac"])
                    fd = np.array([
                        _fd(lambda x: 0.5 * noise.factor(noise.L_g(x)).logdet, theta_a, k)
                        for k in range(N_FIXED)
                    ])
                    record("logdet_vs_fd", design, tag, _rel(lg, fd))

                if mode == "fixed":
                    continue

                # profile identities, and the exact cross-check against the weak
                # model's own gradient
                thp = theta_a
                ev = sc._evaluate(thp, a)
                v = ev["v"]
                xs = sc.noise_star(thp, a, ev)
                ps, _ = get_forced_path(e, thp, xs, design, TMIN, TMAX, DT)
                data = y_obs + meas + (ev["pert"] if mode == "marginal" else 0.0)
                Ro_inv_resid = (_obs(ps) - data) / np.repeat(sig_obs**2, N)
                record("profile_v", design, tag, _rel(Ro_inv_resid, v))

                if mode == "profile":
                    L_full = _full_L(noise, ev["L_g"])
                    H = Qxi_inv + L_full.T @ (L_full / np.repeat(sig_obs**2, N)[:, None])
                    xi_try = draw_xi()
                    pt, _ = get_forced_path(e, thp, xi_try, design, TMIN, TMAX, DT)
                    r = _obs(pt) - (y_obs + meas)
                    Jw = (0.5 * (thp - theta_f) @ invB @ (thp - theta_f)
                          + 0.5 * (xi_try - xi_f) @ Qxi_inv @ (xi_try - xi_f)
                          + 0.5 * r @ (r / np.repeat(sig_obs**2, N)))
                    Jp = sc.cost(thp, a)
                    gap = 0.5 * (xi_try - xs) @ H @ (xi_try - xs)
                    record("profile_gap", design, tag, abs((Jw - Jp) - gap) / abs(Jw))

                if design == "flux":
                    x_f_w = np.hstack([theta_f, xi_f if mode == "profile" else np.zeros(3 * M)])
                    invB_w = np.zeros((N_FIXED + 3 * M, N_FIXED + 3 * M))
                    invB_w[:N_FIXED, :N_FIXED] = invB
                    invB_w[N_FIXED:, N_FIXED:] = Qxi_inv
                    invR = [get_covar_white(np.array([s] * N), N, inv=True) for s in sig_obs]
                    a_w = [x_f_w, invB_w, *invR, data.reshape(4, N), e, TMIN, TMAX, DT]
                    gw = wc_cost.grad(np.hstack([thp, xs]), a_w)
                    g_sc = sc.grad(thp, a)
                    if mode == "marginal":
                        g_sc = g_sc - sc.logdet_grad(thp, ctx, ev["fac"])
                    record("weak_grad_theta", design, tag, _rel(gw[:N_FIXED], g_sc))
                    record("weak_grad_q", design, tag,
                           float(np.max(np.abs(gw[N_FIXED:]))) / max(float(np.max(np.abs(gw[:N_FIXED]))), 1e-300))

        # ---- truth -------------------------------------------------------------
        xi_tr = np.asarray(q_tr, dtype=float).copy()
        xi_tr[:M] *= scale
        p_tr, _ = get_forced_path(e, controls_tr, xi_tr, design, TMIN, TMAX, DT)
        y_tr = _obs(p_tr)
        ctx = sc.SCContext(noise, "profile", xi_tr, np.zeros(4 * N), ref_factor)
        a = [theta_ref, invB, ctx, y_tr.reshape(4, N), e, TMIN, TMAX, DT]
        ev = sc._evaluate(controls_tr, a)
        xs = sc.noise_star(controls_tr, a, ev)
        record("truth", design, "profile",
               max(float(np.max(np.abs(ev["d"]))), float(np.max(np.abs(ev["v"]))),
                   float(np.max(np.abs(xs - xi_tr)))))
        if design == "flux" and obs_wc is not None:
            record("truth_obs_eq_weak", design, "",
                   float(np.max(np.abs(y_tr - np.asarray(obs_wc).ravel()))))

    return _report(logger, rows, check_dir, args)


def _one_step(e, x, g_t, t, design, DT):
    """One forced step from state x at time t (used for the TLM difference check)."""

    (T1, T2, Q, TR1, TR2, L, G, EPS, C1, C2, F1_CO2, A1, A2, B1, B2) = x
    F = wc_dyn.get_forcing(e, F1_CO2, t)
    kick = DT / C1 * g_t if design == "flux" else g_t
    T1n = ((1 - DT * (L + G * EPS) / C1) * T1 + DT * G * EPS / C1 * T2
           + DT / C1 * F + kick)
    T2n = (1 - DT * G / C2) * T2 + DT * G / C2 * T1
    out = x.copy()
    out[0] = T1n
    out[1] = T2n
    out[2] = C1 * T1n + C2 * T2n
    out[3] = A1 * T1n + B1 * e.emis["geo"][t + 1]
    out[4] = A2 * T1n + B2 * e.emis["geo"][t + 1]
    return out


def _report(logger, rows, check_dir, args):
    failed = [r for r in rows if not r[6]]

    # one line per (check, design): the worst value across modes and points.
    # for the negative control "worst" is the *smallest* detected error
    summary = {}
    for check, design, mode, point, value, thr, ok in rows:
        key = (check, design)
        pick = min if check == "negative_control" else max
        prev = summary.get(key)
        summary[key] = (
            (value, ok) if prev is None else (pick(prev[0], value), prev[1] and ok)
        )

    for (check, design), (value, ok) in sorted(summary.items()):
        verb = "smallest" if check == "negative_control" else "worst"
        logger.info(
            f"            >>> (SC CHECK) {'PASS' if ok else 'FAIL'}  {check:20s} "
            f"{design:5s} {verb} {value:.3e}  (threshold {THRESHOLDS[check]:.0e})"
        )

    if check_dir is None:
        try:
            from var_assim.config import DATA_DIR
            from var_assim.tlm_adj_checks import _get_check_stamp

            check_dir = DATA_DIR / "checks" / args.model / _get_check_stamp()
        except Exception:  # checks still report through the logger
            check_dir = None

    if check_dir is not None:
        check_dir.mkdir(parents=True, exist_ok=True)
        with open(check_dir / "sc_checks.csv", "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["check", "design", "mode", "point", "value", "threshold", "passed"])
            wtr.writerows(rows)
        logger.info(f"            >>> (SC CHECK) results written to {check_dir / 'sc_checks.csv'}")

    if failed:
        logger.warning(f"            >>> (SC CHECK) {len(failed)} of {len(rows)} checks FAILED")
    else:
        logger.info(f"            >>> (SC CHECK) all {len(rows)} checks passed")
    return not failed


def main():
    """Standalone run on one window, without the warm start or an assimilation."""

    from var_assim.calibration.noise import ClimateModelNoise
    from var_assim.calibration.priors import ClimateModelPriors
    from var_assim.calibration.truth import ClimateModelTruth
    from var_assim.config import (
        MOD_ERROR_SEED,
        NOISE_PATH,
        PRIOR_PATH,
        REG1_NOISE_SEED,
        REG2_NOISE_SEED,
        TRUTH_PATH,
    )
    from var_assim.emis import EmissionsBaseline
    from var_assim.model_errors import gen_noise_ts

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tmin", type=int, default=2025)
    ap.add_argument("--tmax", type=int, default=2100)
    ap.add_argument("--theta", type=int, default=15)
    ap.add_argument("--sai_ramp", default="linear", choices=["linear", "fast", "slow"])
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("sc_checks")

    args = argparse.Namespace(
        model="pco2geosc_reg_noic", scenario="ssp245", noise_model="AR1",
        tmin=cli.tmin, theta=cli.theta, ecs=3.0, deg_p_dec=0.1, n_yrs_ramp=50,
        windowing="gradual", n_ens=1, reg_noise=True, sai_ramp=cli.sai_ramp,
        debug=False,
    )
    Noise = ClimateModelNoise.from_cli_and_yaml(args, NOISE_PATH)
    Prior = ClimateModelPriors.from_cli_and_yaml_and_noise(args, PRIOR_PATH, Noise)
    Truth = ClimateModelTruth.from_cli_and_yaml(args, TRUTH_PATH)

    DT = 1.0
    e = EmissionsBaseline(logger, args, cli.tmin, cli.tmax, geo=True, Prior=Prior,
                          Truth=Truth, T_START=cli.tmin,
                          T_END=cli.tmin + args.n_yrs_ramp, print_level=0)
    N = len(e.conc["CO2"])
    q_full, cov_full = gen_noise_ts(Noise, N, rng=np.random.default_rng(MOD_ERROR_SEED))
    r1 = np.random.default_rng(REG1_NOISE_SEED).normal(0, Noise.INT_T_REG_STD[0], N)
    r2 = np.random.default_rng(REG2_NOISE_SEED).normal(0, Noise.INT_T_REG_STD[1], N)
    q_tr = np.hstack([q_full[1:], r1[1:], r2[1:]])
    controls_tr = np.asarray(Truth.controls_tr, dtype=float)

    # the weak model's truth obs for the same realization, for the bit-for-bit check
    pw, _ = wc_dyn.get_nonlin_path(
        e, Truth.get_augmented_truth_vector(q_tr), cli.tmin, cli.tmax, DT
    )
    obs_wc = get_obs_from_dynamics(pw)

    logger.info(f"Noise weighting: {Noise.OBS_WEIGHTING}; window {cli.tmin}-{cli.tmax} (N={N})")
    ok = run_sc_checks(logger, args, e, cli.tmin, cli.tmax, DT, Prior, Noise,
                       cov_full[1:, 1:], q_tr, controls_tr, obs_wc=obs_wc)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
