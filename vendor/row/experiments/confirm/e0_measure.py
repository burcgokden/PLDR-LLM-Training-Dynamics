#!/usr/bin/env python3
"""E0 estimator battery for the row-map collapse theory.

Implements the estimators of experiments/protocols/e0_construct_validity.md
on chain-merged confirmation runs, composing ONLY frozen theory/detector
code (analysis/theory.py, analysis/avalanche.py, analysis/gates.py) with
the dictionary mappings declared in the paper (subsec:observables) and
the launch-time operational choices recorded in LAUNCH_NOTES.md:

  stress x(t)      physical preconditioned Gauss--Newton Rayleigh
                   quotient q^T P^{-1/2} G P^{-1/2} q, with q tracked
                   in the same metric; full-Hessian and raw-metric
                   series are robustness observables only
  sigma path       sigma_t = (1 - beta1) * eta_applied(t) * x(t);
                   d_t = eta_applied(t) * lam_wd
  threshold/headroom theory.cert_band on the frozen box at the window's
                   measured directional drift budget; nearest-edge
                   margin is a certificate statistic, while positive
                   dose uses the oriented upper headroom
  jet J_i(t)       per-layer median rowmap sigma (eq. jetident, J_norm=1
                   so the phase criterion reads J < 0.1)
  coupling a_i     a_norm_layers;  carrier w_i: w_norm_layers
  source u_i       a_i * J_i (the visited-jet path product; the chain
                   factor is carried inside the sampled row map)
  oscillation      cumulative signed projection on the tracked GN mode
                   (per-step mode_proj_plga_gn), on exact trailing
                   windows; q is centered by its same-window mean
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RUNS = os.path.join(EXP, "runs")
sys.path.insert(0, os.path.join(EXP, "analysis"))
sys.path.insert(0, HERE)

import gates            # noqa: E402
import avalanche        # noqa: E402
import thresholds as T  # noqa: E402
import theory           # noqa: E402
import bridge           # noqa: E402
import oriented_headroom as OH  # noqa: E402
import stress_units as SU       # noqa: E402

CAL = json.load(open(os.path.join(EXP, "calibration",
                                  "calibration.json")))
GEN = json.load(open(os.path.join(EXP, "protocols",
                                  "generated_constants.json")))
BETA1 = CAL["optimizer"]["beta1"]
LAM_WD = CAL["optimizer"]["lam_wd"]
BOX = CAL["planning"]["cp_cert_box"]
BOUNDS = theory.cp_dir_bounds(BOX["sigma"][0], BOX["sigma"][1],
                              BOX["beta1"][0], BOX["beta1"][1],
                              BOX["d"][0], BOX["d"][1])
SC_CA = 2.0      # A_osc <= C_A sqrt(eta)      (declared regime, e0 md)
SC_SHIFT = 1.0   # adjacent half-window mean shift / A_osc bound
SC_DRIFT = 1.5   # K_s eps_s + K_q eps_q <= 1.5 eta per window
CENTERING_TOL = 64.0 * np.finfo(float).eps
CWIN_THR = 1e-4  # (C-win) clipped-fraction gate
P_LEDGER = 3.0   # decayed path factors in u = a g J (exact-law form)


# ---------------------------------------------------------------------
# series assembly


def build_series(name, runs=RUNS):
    cfg, steps, final = bridge.load_steps(name, runs=runs)
    Tt = max(d["step"] for d in steps)
    S = {"name": name, "config": cfg, "T": Tt, "final": final}

    def ps(key):
        v = np.full(Tt, np.nan)
        for d in steps:
            if key in d and d["step"] <= Tt:
                v[d["step"] - 1] = d[key]
        return v

    for k in ("loss", "gnorm", "gnorm_plga", "gnorm_phi", "gnorm_attn",
              "gnorm_ffn", "dtheta_plga", "dtheta_phi", "dtheta_attn",
              "dtheta_ffn", "clip_frac_plga", "clip_frac_phi",
              "clip_frac_attn", "clip_frac_ffn",
              "decay_ledger_residual_plga", "decay_ledger_residual_phi",
              "decay_ledger_residual_attn", "decay_ledger_residual_ffn",
              "mode_proj_plga_gn", "mode_proj_plga_live",
              "mode_proj_plga_raw"):
        S[k] = ps(k)
    S["lr_app"] = bridge.applied_lr(steps, Tt)

    # Primary state: physical preconditioned Gauss--Newton quotient.
    # No cross-family fallback is allowed. Missing primary probes make
    # the corresponding prerequisite incomplete. Full-Hessian and raw
    # coordinates remain separately named robustness observables.
    tx, x_phys = bridge.per_step_series(steps, "mode_lam_plga_gn")
    S["tx"], S["x_phys"] = tx, x_phys
    tg, gn_residual = bridge.per_step_series(
        steps, "mode_gn_residual_plga_gn")
    S["tg"], S["gn_residual"] = tg, gn_residual
    tq, q_count = bridge.per_step_series(
        steps, "mode_preconditioner_window_count_plga")
    S["tq"], S["q_window_count"] = tq, q_count
    tb, x_blk = bridge.per_step_series(steps, "gn_pre_live_plga")
    S["tb"], S["x_blk_gn"] = tb, x_blk
    th, x_hess = bridge.per_step_series(steps, "mode_lam_plga_live")
    S["th"], S["x_hess"] = th, x_hess
    tr, x_raw = bridge.per_step_series(steps, "mode_lam_plga_raw")
    S["tr"], S["x_raw"] = tr, x_raw
    # per-layer factor series at probe cadence
    tl, a_nl, w_nl, j_med = [], [], [], []
    for d in steps:
        if "rowmap_sigma_med" in d and "a_norm_layers" in d:
            tl.append(d["step"])
            j_med.append(d["rowmap_sigma_med"])
            a_nl.append(d["a_norm_layers"])
            w_nl.append(d["w_norm_layers"])
    S["tl"] = np.array(tl, dtype=int)
    S["J_layers"] = np.array(j_med, dtype=float)      # [probe, layer]
    S["a_layers"] = np.array(a_nl, dtype=float)
    S["w_layers"] = np.array(w_nl, dtype=float)
    S["xs"] = S["x_phys"]  # compatibility alias, still physical x
    S["stress_metadata"] = SU.PRIMARY_STRESS.as_dict()
    S["gate_open_pad"] = 50
    return S


def gate_closed_mask(S, name, t_axis, runs=RUNS, pad_pre=10,
                     pad_post=None):
    """True at probe times outside every detected event's window
    [start - pad_pre, end + pad_post]: the gate-closed domain the
    factor-constant fits (kappa, d0, rates, delta) run on, per the
    protocol's estimator clause (the saturation clip provably inert
    there)."""
    if pad_post is None:
        pad_post = S["gate_open_pad"]
    ev = gates.cell_events(name, runs=runs)["events"]
    m = np.ones(len(t_axis), dtype=bool)
    for e in ev:
        m &= ~((t_axis >= e["start"] - pad_pre)
               & (t_axis <= e["end"] + pad_post))
    return m


def sigma_path(S, x_key="x_phys"):
    """Derived normalized path sigma=(1-beta1)*eta*x.

    The primary x_phys is the physical preconditioned GN state.
    Other accepted keys are explicitly labelled robustness paths.
    """
    tp = {"x_phys": S["tx"], "x_hess": S["th"],
          "x_raw": S["tr"], "x_blk_gn": S["tb"]}[x_key]
    lam = S[x_key]
    eta = S["lr_app"][np.clip(tp - 1, 0, S["T"] - 1)]
    sig = (1.0 - BETA1) * eta * lam
    d = eta * LAM_WD
    ok = np.isfinite(sig) & np.isfinite(d)
    return tp[ok], sig[ok], d[ok], eta[ok]


def stress_at(S, t_axis):
    """Physical GN stress interpolated onto an arbitrary step axis."""
    ok = np.isfinite(S["xs"])
    if not ok.any():
        return np.full(np.shape(t_axis), np.nan, dtype=float)
    return np.interp(t_axis, S["tx"][ok], S["xs"][ok])


# ---------------------------------------------------------------------
# windows


def window_stats(S, n_windows=60, wlen=None):
    """Evaluate fixed trailing windows; return per-window (SC),
    clip, drift, band, and factor statistics."""
    Tt = S["T"]
    if wlen is None:
        wlen = Tt // n_windows
    tp, sig, dpar, eta_p = sigma_path(S)
    q_full = np.nancumsum(np.nan_to_num(S["mode_proj_plga_gn"],
                                        nan=0.0))
    wins = []
    for w in range(n_windows):
        lo, hi = w * wlen + 1, (w + 1) * wlen        # steps, incl.
        sl = slice(lo - 1, hi)
        eta_w = np.nanmean(S["lr_app"][sl])
        # oscillation stats on the tracked-mode coordinate
        qw = q_full[sl]
        nprj = np.sum(np.isfinite(S["mode_proj_plga_gn"][sl]))
        if nprj >= wlen // 2:
            qc = qw - qw.mean()
            A_osc = float(np.max(np.abs(qc)))
            centering_error = (
                abs(float(qc.mean())) / max(A_osc, 1e-300)
                if A_osc > 0 else 0.0)
            half = len(qw) // 2
            mean_shift = (
                abs(float(qw[half:].mean() - qw[:half].mean()))
                / max(A_osc, 1e-300) if A_osc > 0 else 0.0)
        else:
            A_osc = centering_error = mean_shift = np.nan
        # directional drift budget of the BAND's parameters: the
        # schedule-induced per-step motion of nu = (sigma(eta),
        # beta1, d(eta)) at the window's reference sharpness.  The
        # sigma path itself is the STATE read against the band (its
        # excursions are the crossings); at constant rate the band is
        # parameter-stationary and the budget is zero, while under an
        # anneal the budget is the schedule component of the I-edge
        # quotient.  A certificate also requires simultaneous intervals
        # for edge-state motion and the complete source increment.  The
        # raw probe-pair path drift is reported as a sensitivity statistic.
        m = (tp >= lo) & (tp <= hi)
        idx = np.where(m)[0]
        mq = (S["tq"] >= lo) & (S["tq"] <= hi)
        q_counts = S["q_window_count"][mq]
        q_expected = int(S["config"].get("sharp_block_every", 0))
        sc_preconditioner = bool(
            len(q_counts) > 0 and q_expected > 0
            and np.all(np.isfinite(q_counts))
            and np.all(q_counts == q_expected))
        eps_s = eps_q = eps_s_fast = eps_q_fast = 0.0
        eta_sl = S["lr_app"][sl]
        deta = np.abs(np.diff(eta_sl[np.isfinite(eta_sl)]))
        deta_max = float(deta.max()) if len(deta) else 0.0
        if len(idx) >= 2:
            lam_ref = float(np.nanmedian(sig[idx]
                                         / np.maximum(eta_p[idx],
                                                      1e-300)))
            # |ds| = ((1-b1) lam_ref + lam_wd) |deta|,
            # |dq| = b1 lam_wd |deta|   (s = 1+b1-sigma-d, q=b1(1-d))
            eps_s = (lam_ref + LAM_WD) * deta_max
            eps_q = BETA1 * LAM_WD * deta_max
            for i in idx[1:]:
                dsf, dqf = theory.drift_dir(
                    (sig[i - 1], BETA1, dpar[i - 1]),
                    (sig[i], BETA1, dpar[i]))
                eps_s_fast = max(eps_s_fast, dsf)
                eps_q_fast = max(eps_q_fast, dqf)
        budget = BOUNDS["K_s"] * eps_s + BOUNDS["K_q"] * eps_q
        # certified band at the window's measured drift budget
        band = theory.cert_band(BETA1, float(np.nanmean(dpar[m]))
                                if m.any() else 0.0,
                                eps_s, eps_q, BOUNDS)
        in_box = bool(np.all((sig[m] >= BOX["sigma"][0])
                             & (sig[m] <= BOX["sigma"][1]))) \
            if m.any() else False
        if band["empty"] or not m.any():
            in_band = False
            frac_in = 0.0
            nearest_margin = np.nan
            upper_headroom = np.nan
        else:
            blo, bhi = band["sigma_cert_lo"], band["sigma_cert_hi"]
            band_sig = sig[m]
            inside = (band_sig >= blo) & (band_sig <= bhi)
            in_band = bool(np.all(inside))
            frac_in = float(np.mean(inside))
            nearest_margin = float(np.min(
                OH.certificate_margin_nearest(band_sig, blo, bhi)))
            upper_headroom = float(np.min(
                OH.topple_headroom_upper(band_sig, bhi)))
        # clipping
        cf = np.nanmax(np.stack([S[f"clip_frac_{b}"][sl]
                                 for b in ("plga", "phi", "attn",
                                           "ffn")]), axis=0)
        cwin = bool(np.nanmax(cf) < CWIN_THR) if np.isfinite(
            np.nanmax(cf)) else False
        wins.append(dict(
            w=w, lo=lo, hi=hi, eta=float(eta_w),
            A_osc=A_osc, eps_av=centering_error,
            centering_error=centering_error, mean_shift=mean_shift,
            sc_A=bool(A_osc <= SC_CA * np.sqrt(eta_w)),
            sc_e=bool(centering_error <= CENTERING_TOL),
            sc_shift=bool(mean_shift <= SC_SHIFT * np.sqrt(eta_w)),
            sc_preconditioner=sc_preconditioner,
            eps_s=float(eps_s), eps_q=float(eps_q),
            eps_s_fast=float(eps_s_fast), eps_q_fast=float(eps_q_fast),
            drift_budget=float(budget),
            sc_drift=bool(budget <= SC_DRIFT * eta_w),
            band_lo=band.get("sigma_cert_lo"),
            band_hi=band.get("sigma_cert_hi"),
            band_empty=bool(band["empty"]),
            in_band=in_band, frac_in_band=frac_in,
            in_box=in_box, band_margin=nearest_margin,
            upper_headroom=upper_headroom,
            clip_max=float(np.nanmax(cf)),
            cwin=cwin,
            n_probes=int(m.sum()), n_proj=int(nprj),
        ))
    return wins


# ---------------------------------------------------------------------
# shared-step data events and the exclusion list


DATA_SPIKE_RATIO = 1.35   # launch-time constant (LAUNCH_NOTES);
#                           sensitivity at 1.25 / 1.50 reported


def data_event_windows(name_a, name_b, runs=RUNS,
                       ratio=DATA_SPIKE_RATIO):
    """The recurring shared-data-order LOSS spikes: steps at which
    BOTH cells' per-step loss exceeds `ratio` times its own centered
    running median (frozen window AV_MED_WIN), dilated by the frozen
    matching lag AV_MATCH_LAG and merged into intervals.  This is
    the E0 exclusion list (written before any fit)."""
    def loss_ratio(name):
        cfg, steps, _ = bridge.load_steps(name, runs=runs)
        Tt = max(d["step"] for d in steps)
        y = np.full(Tt, np.nan)
        for d in steps:
            y[d["step"] - 1] = d.get("loss", np.nan)
        y = np.nan_to_num(y, nan=np.nanmedian(y))
        med = avalanche.running_median(y, T.AV_MED_WIN)
        r = y / np.maximum(med, 1e-300)
        r[:cfg.get("warmup", 0)] = 0.0
        return r
    ra = loss_ratio(name_a)
    rb = loss_ratio(name_b)
    n = min(len(ra), len(rb))
    hot = (ra[:n] > ratio) & (rb[:n] > ratio)
    lag = T.AV_MATCH_LAG
    locked = []
    i = 0
    while i < n:
        if hot[i]:
            j = i
            while j + 1 < n and hot[j + 1]:
                j += 1
            locked.append((max(i + 1 - lag, 1), min(j + 1 + lag, n)))
            i = j + 1
        else:
            i += 1
    return locked


def apply_exclusion(wins, locked, pad=0):
    for w in wins:
        w["excluded"] = any(s - pad <= w["hi"] and w["lo"] <= e + pad
                            for s, e in locked)
    return wins


# ---------------------------------------------------------------------
# factor constants (kappa, d0, mu_b, rates, decay shares)


def factor_series(S, layer_agg="min"):
    """Aggregated factor series at probe cadence: (t, a, J, w, u)."""
    agg = {"min": np.nanmin, "median": np.nanmedian,
           "max": np.nanmax}[layer_agg]
    J = agg(S["J_layers"], axis=1)
    a = np.nanmean(S["a_layers"], axis=1)
    w = np.nanmean(S["w_layers"], axis=1)
    u = a * J
    return S["tl"], a, J, w, u


def kappa_fit(S, wins, gc_mask, layer_agg="min"):
    """kappa from the definitional relation x = b + kappa u^2 with a
    per-window intercept (b varies slowly): pooled least squares on
    within-window demeaned (x, u^2), GATE-CLOSED probes only, x in
    physical Gauss--Newton curvature units."""
    tl, a, J, w, u = factor_series(S, layer_agg)
    x = stress_at(S, tl)
    X, Y = [], []
    for wd in wins:
        if wd.get("excluded"):
            continue
        m = (tl >= wd["lo"]) & (tl <= wd["hi"]) & gc_mask
        if m.sum() < 6:
            continue
        uu = (u[m] ** 2) - np.nanmean(u[m] ** 2)
        xx = x[m] - np.nanmean(x[m])
        ok = np.isfinite(uu) & np.isfinite(xx)
        X.append(uu[ok])
        Y.append(xx[ok])
    X = np.concatenate(X) if X else np.array([])
    Y = np.concatenate(Y) if Y else np.array([])
    kappa = float(X @ Y / max(X @ X, 1e-300)) if len(X) else np.nan
    r = float(np.corrcoef(X, Y)[0, 1]) if len(X) > 2 else np.nan
    return dict(kappa=kappa, r=r, n=len(X))


def base_fit(S, wins, kappa, gc_mask, layer_agg="min"):
    """(d0, mu_b) from the base law db = eta d0 - eta mu_b b on
    gate-closed probe increments of b = x - kappa u^2 (physical
    Gauss--Newton stress)."""
    tl, a, J, w, u = factor_series(S, layer_agg)
    x = stress_at(S, tl)
    b = x - kappa * u ** 2
    eta = S["lr_app"][np.clip(tl - 1, 0, S["T"] - 1)]
    Z, Y = [], []
    for wd in wins:
        if wd.get("excluded"):
            continue
        m = (tl >= wd["lo"]) & (tl <= wd["hi"]) & gc_mask
        idx = np.where(m)[0]
        for j in range(1, len(idx)):
            i, ip = idx[j], idx[j - 1]
            dt = tl[i] - tl[ip]
            if dt <= 0 or dt > 3 * (tl[1] - tl[0]) \
                    or not np.isfinite(b[i] - b[ip]):
                continue
            Z.append([1.0, -b[ip]])
            Y.append((b[i] - b[ip]) / (eta[i] * dt))
    Z, Y = np.array(Z), np.array(Y)
    if len(Y) < 10:
        return dict(d0=np.nan, mu_b=np.nan, n=int(len(Y)))
    coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return dict(d0=float(coef[0]), mu_b=float(coef[1]), n=len(Y))


def factor_rates(S, wins, gc_mask, layer_agg="min"):
    """Per-factor per-step log-contraction rates in eta units and the
    decay-ledger split (H-dec): total rate th_f_tot with the decay
    share eta*lam_wd removed to give the loss rate."""
    tl, a, J, w, u = factor_series(S, layer_agg)
    eta = S["lr_app"][np.clip(tl - 1, 0, S["T"] - 1)]
    out = {}
    for fname, f in (("a", a), ("J", J), ("w", w)):
        num, den = [], []
        for wd in wins:
            if wd.get("excluded"):
                continue
            m = (tl >= wd["lo"]) & (tl <= wd["hi"]) & gc_mask
            idx = np.where(m)[0]
            for i in idx[1:]:
                dt = tl[i] - tl[i - 1]
                if dt <= 0 or f[i - 1] <= 0 or f[i] <= 0:
                    continue
                num.append(-np.log(f[i] / f[i - 1]) / dt)
                den.append(eta[i])
        num, den = np.array(num), np.array(den)
        ok = np.isfinite(num) & np.isfinite(den) & (den > 0)
        th_tot = float(np.mean(num[ok] / den[ok]))  # rate per eta
        out[fname] = dict(th_tot=th_tot,
                          decay_share=LAM_WD,
                          th_loss=th_tot - LAM_WD,
                          n=int(ok.sum()))
    return out


def delta_feed_fit(S, wins, rates, gc_mask, layer_agg="min"):
    """Rebuild feed delta of the a-law: da = -eta(th_a+lam_wd)a
    + eta*delta  =>  delta = da/(eta dt) + (th_a+lam_wd) a."""
    tl, a, J, w, u = factor_series(S, layer_agg)
    eta = S["lr_app"][np.clip(tl - 1, 0, S["T"] - 1)]
    th_a = rates["a"]["th_loss"] + LAM_WD
    vals = []
    for wd in wins:
        if wd.get("excluded"):
            continue
        m = (tl >= wd["lo"]) & (tl <= wd["hi"]) & gc_mask
        idx = np.where(m)[0]
        for i in idx[1:]:
            dt = tl[i] - tl[i - 1]
            if dt <= 0 or not np.isfinite(a[i] - a[i - 1]):
                continue
            vals.append((a[i] - a[i - 1]) / (eta[i] * dt)
                        + th_a * a[i - 1])
    return float(np.nanmean(vals))


def loading_slope(S, wins, kappa, d0, mu_b, rates, delta, name,
                  runs=RUNS, delta_J0=None, layer_agg="min"):
    """The curvature-law slope: regression of measured per-window
    physical Gauss--Newton stress increments on the exact moving-step
    law at the window factors and the separately measured
    constants.  Windows WITHOUT detected events are fitted against
    theory.curvature_increment_exact (gate closed, clip provably
    inert); windows WITH events are reported separately against the
    gated law (theory.curvature_increment_exact_gated, chi recorded)
    when delta_J0 is available.  Bootstrap over windows (each window
    is >= the declared 50-step block, so window resampling is block
    resampling at or above the declared length)."""
    tl, a, J, w, u = factor_series(S, layer_agg)
    x = stress_at(S, tl)
    eta = S["lr_app"][np.clip(tl - 1, 0, S["T"] - 1)]
    b = x - kappa * u ** 2
    ev = gates.cell_events(name, runs=runs)["events"]
    act_frac = np.zeros(S["T"])
    for e in ev:
        act_frac[e["start"] - 1:min(e["end"] + S["gate_open_pad"],
                                    S["T"])] = 1.0
    rows_closed, rows_open = [], []
    for wd in wins:
        if wd.get("excluded"):
            continue
        m = (tl >= wd["lo"]) & (tl <= wd["hi"])
        if m.sum() < 6:
            continue
        i0, i1 = np.where(m)[0][0], np.where(m)[0][-1]
        span = tl[i1] - tl[i0]
        if span <= 0:
            continue
        meas = (x[i1] - x[i0]) / span
        args = (float(np.nanmean(b[m])), kappa,
                float(np.nanmean(a[m])), 1.0, float(np.nanmean(J[m])),
                float(np.nanmean(w[m])), float(np.nanmean(eta[m])))
        A_w = float(np.mean(act_frac[wd["lo"] - 1:wd["hi"]]))
        if A_w == 0.0:
            first, exact = theory.curvature_increment_exact(
                *args, d0, delta, rates["a"]["th_loss"], 0.0,
                rates["J"]["th_loss"], LAM_WD, mu_b)
            rows_closed.append((wd["w"], meas, first, exact))
        elif delta_J0 is not None and delta_J0 > 0:
            first, exact, chi, defect = theory.curvature_increment_exact_gated(
                *args, d0, delta, delta_J0, A_w,
                rates["a"]["th_loss"], 0.0,
                rates["J"]["th_loss"], LAM_WD, mu_b)
            rows_open.append((wd["w"], meas, first, exact, chi, defect))

    def fit(rows):
        if len(rows) < 4:
            return dict(slope=np.nan, r=np.nan, n=len(rows),
                        ci=[np.nan, np.nan], windows=rows)
        arr = np.array([r[:3] for r in rows])
        meas, pred = arr[:, 1], arr[:, 2]
        ok = np.isfinite(meas) & np.isfinite(pred)
        meas, pred = meas[ok], pred[ok]
        slope = float(pred @ meas / max(pred @ pred, 1e-300))
        r = (float(np.corrcoef(pred, meas)[0, 1])
             if ok.sum() > 2 else np.nan)
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(2000):
            ii = rng.integers(0, len(meas), len(meas))
            p, mm = pred[ii], meas[ii]
            bs.append(p @ mm / max(p @ p, 1e-300))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(slope=slope, r=r, n=int(ok.sum()),
                    ci=[float(lo), float(hi)],
                    windows=[list(map(float, rr)) for rr in rows])

    return dict(closed=fit(rows_closed), open=fit(rows_open))


# ---------------------------------------------------------------------
# gate regression (H-gate): jet rebuild on reload windows


def gate_regression(S, name, runs=RUNS, cap=200, layer_agg="min"):
    ev = gates.cell_events(name, runs=runs)["events"]
    tl, a, J, w, u = factor_series(S, layer_agg)
    eta = S["lr_app"][np.clip(tl - 1, 0, S["T"] - 1)]
    X, Y = [], []
    dur = []
    for k, e in enumerate(ev):
        t0 = e["end"]
        t1 = min(ev[k + 1]["start"] - 1 if k + 1 < len(ev)
                 else S["T"], t0 + cap)
        m = (tl > t0) & (tl <= t1)
        idx = np.where(m)[0]
        if len(idx) < 2:
            continue
        base = J[idx[0]]
        rec = None
        for i in idx[1:]:
            # accumulated eta-steps since the reload window opened:
            # the regressor of J(t) - J(t0) = delta_J0 * sum eta
            X.append(float((tl[i] - tl[idx[0]])
                           * np.nanmean(eta[idx[0]:i + 1])))
            Y.append(float(J[i] - base))
            if rec is None and J[i] >= base:
                rec = tl[i] - t0
        dur.append(int(rec) if rec is not None else int(t1 - t0))
    X, Y = np.array(X), np.array(Y)
    ok = np.isfinite(X) & np.isfinite(Y) & (X > 0)
    slope = float(X[ok] @ Y[ok] / max(X[ok] @ X[ok], 1e-300))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        ii = rng.integers(0, ok.sum(), ok.sum())
        xx, yy = X[ok][ii], Y[ok][ii]
        bs.append(xx @ yy / max(xx @ xx, 1e-300))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta_J0=slope, ci=[float(lo), float(hi)],
                n=int(ok.sum()), n_events=len(ev),
                reload_durations=dur)


# ---------------------------------------------------------------------
# crossings and the loading-crossing-reset signature


def band_crossings(S, wins, min_gap=25):
    """Strict upper-edge crossings of the normalized GN path: probes
    where the path exits above the band after having been
    inside, separated by at least min_gap steps."""
    tp, sig, dpar, eta_p = sigma_path(S)
    events = []
    inside_prev = None
    last = -10 ** 9
    wmap = {}
    for wd in wins:
        for _t in range(wd["lo"], wd["hi"] + 1):
            wmap[_t] = wd
    for i, t in enumerate(tp):
        wd = wmap.get(int(t))
        if wd is None or wd["band_empty"] or wd["band_lo"] is None:
            inside_prev = None
            continue
        inside = bool(wd["band_lo"] <= sig[i] <= wd["band_hi"])
        if (inside_prev is True and sig[i] > wd["band_hi"]
                and t - last >= min_gap):
            events.append(dict(t=int(t), i=int(i), edge="hi",
                               sigma=float(sig[i]),
                               window=wd["w"],
                               excluded=bool(wd.get("excluded"))))
            last = t
        inside_prev = inside
    return events


def signature_stats(S, crossings, pre=100, post=100,
                    reset_lo=None):
    """Loading-crossing-reset signature per crossing: positive
    pre-crossing stress slope AND post-crossing reset (stress falls
    back with post/peak ratio inside the licensed interval)."""
    if reset_lo is None:
        reset_lo = GEN["reset_lo"]
    tp, xs = S["tx"], S["x_phys"]      # primary physical GN stress
    tr, xr = S["tx"], S["x_phys"]      # same state for reset ratio
    out = []
    for c in crossings:
        t = c["t"]
        mpre = (tp >= t - pre) & (tp < t)
        mpost = (tr > t) & (tr <= t + post)
        if mpre.sum() < 4 or mpost.sum() < 4:
            continue
        # loading: positive slope of the STRESS into the crossing
        A = np.vstack([tp[mpre], np.ones(mpre.sum())]).T
        sl = np.linalg.lstsq(A, xs[mpre], rcond=None)[0][0]
        loading = bool(sl > 0)
        # reset: post-crossing minimum of physical GN stress against
        # its crossing peak, inside the licensed interval
        peak = float(np.nanmax(xr[(tr >= t - 10) & (tr <= t + 10)]))
        post_min = float(np.nanmin(xr[mpost]))
        ratio = post_min / peak if peak > 0 else np.nan
        reset = bool(np.isfinite(ratio)
                     and reset_lo <= ratio < 1.0)
        out.append(dict(t=int(t), loading=loading, reset=reset,
                        signature=bool(loading and reset),
                        pre_slope=float(sl), ratio=float(ratio),
                        excluded=c["excluded"]))
    return out


def placement_signature(S_ctrl, steps_list, pre=100, post=100,
                        reset_lo=None):
    """The same signature statistic evaluated in the control cell at
    the SAME global steps (matched placements under the shared data
    order)."""
    fake = [dict(t=int(t), i=0, edge="", sigma=np.nan, window=-1,
                 excluded=False) for t in steps_list]
    return signature_stats(S_ctrl, fake, pre=pre, post=post,
                           reset_lo=reset_lo)


# ---------------------------------------------------------------------
# threshold-quotient schedule component (I-edge), fitted on capture windows


def iedge_schedule_component_fit(S, wins, edge="hi"):
    """Estimate the schedule component of the exact I-edge quotient.

    On anneal windows this regresses the direct increment of
    ``c=s/((1-beta1)*eta)`` at fixed window edge state ``s`` on
    ``eta_t-eta_{t+1}``.  This is a stationary-edge diagnostic only.
    It is not a capture certificate: the confirmatory bridge must also
    supply simultaneous intervals for ``s(t+1)-s(t)`` and the complete
    source increment, then classify those intervals with
    ``analysis.theory.classify_schedule_certificate``.
    """
    rows = []
    for wd in wins:
        if wd.get("excluded") or wd["band_empty"]:
            continue
        sig_edge = wd["band_hi"] if edge == "hi" else wd["band_lo"]
        if sig_edge is None:
            continue
        lo, hi = wd["lo"], wd["hi"]
        eta = S["lr_app"][lo - 1:hi]
        ok = np.isfinite(eta)
        if ok.sum() < 10:
            continue
        e = eta[ok]
        c = sig_edge / ((1.0 - BETA1) * e)
        dc = np.diff(c)
        de = -np.diff(e)          # eta_t - eta_{t+1}
        m = de > 0
        if m.sum() < 5:
            continue
        rows.append((dc[m], de[m]))
    if not rows:
        return dict(defined=False,
                    component_only=True, certificate_ready=False,
                    note="no capture (anneal) windows in this cell")
    dc = np.concatenate([r[0] for r in rows])
    de = np.concatenate([r[1] for r in rows])
    quotient_slope = float(de @ dc / max(de @ de, 1e-300))
    resid = dc - quotient_slope * de
    positive_residual_q95 = float(np.percentile(resid[resid > 0], 95)
                                  if (resid > 0).any() else 0.0)
    return dict(defined=True, component_only=True,
                certificate_ready=False,
                quotient_slope=quotient_slope,
                positive_residual_q95=positive_residual_q95,
                n=int(len(dc)))


# ---------------------------------------------------------------------
# event entry statistics (U-carr): seeded entry power and shed spread


def event_entry_stats(S, name, runs=RUNS):
    """Per detected event: the carrier power z = w^2 at entry (w from
    the mean coupling-carrier norm, U-carr identity), the stress drop
    over the event (the shed proxy), and their summary statistics
    (z_min_seed = smallest observed entry power; sigma_g2 = variance
    of the per-event drop)."""
    ev = gates.cell_events(name, runs=runs)["events"]
    tl = S["tl"]
    w_mean = np.nanmean(S["w_layers"], axis=1)
    xs = S["xs"]
    tr = S["tx"]
    recs = []
    for e in ev:
        i_pre = np.searchsorted(tl, e["start"] - 1, side="right") - 1
        j_pre = np.searchsorted(tr, e["start"] - 1, side="right") - 1
        j_post = np.searchsorted(tr, e["end"] + 5, side="right") - 1
        if i_pre < 0 or j_pre < 0 or j_post <= j_pre:
            continue
        z_entry = float(w_mean[i_pre] ** 2)
        drop = float(xs[j_pre] - xs[min(j_post, len(xs) - 1)])
        recs.append(dict(start=e["start"], end=e["end"],
                         z_entry=z_entry, drop=drop,
                         size=e["size"], dur=e["dur"]))
    z = np.array([r["z_entry"] for r in recs])
    dr = np.array([r["drop"] for r in recs])
    return dict(n=len(recs),
                z_min_seed=float(np.min(z)) if len(z) else None,
                z_median=float(np.median(z)) if len(z) else None,
                sigma_g2=float(np.var(dr)) if len(dr) else None,
                drop_median=float(np.median(dr)) if len(dr) else None,
                events=recs)
