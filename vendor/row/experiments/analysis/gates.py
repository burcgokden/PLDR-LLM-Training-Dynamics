"""Executable preregistration: every declared pass/kill clause of the
wave-5 and wave-6 preregistrations as pure functions over parsed run
records, with moving-block bootstrap intervals, the endpoint
sensitivity sweep, and dropped-window accounting.

Two consumers:
- the corrected wave-5 re-audit (`corrected_wave5_report`), which
  applies the criteria EXACTLY as declared in wave5_prereg.md,
  including the clauses the archival implementation did not encode
  (P3's second correlate and tertile monotonicity, P5's escape clause,
  P8's primary endpoints and deep-layer tilt criterion, P9 without the
  early-stop disjunct).  The archival script make_figures_w5.py and its
  w5_gate_report.json are frozen and never rewritten; the re-audit is
  emitted alongside them.
- the wave-6 gate report (`wave6_report`), implementing
  wave6_prereg.md W1-W8.

Every verdict function returns a dict with a "verdict" field in
{"pass", "kill", "fail", "insufficient", "confirmed", "falsified"}
plus its evidence, and window-based statistics carry dropped counts.
Synthetic-record tests in tests/test_gates_w6.py exercise every branch.
"""

import json
import os

import numpy as np

import thresholds as T

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")


# ---------------------------------------------------------------------------
# loading and primitives


def load_run(name, runs=None):
    recs = []
    with open(os.path.join(runs or RUNS, name, "log.jsonl")) as f:
        for line in f:
            recs.append(json.loads(line))
    config = next(r for r in recs if r.get("event") == "config")
    steps = [r for r in recs if "step" in r]
    final = next((r for r in recs if r.get("event") == "final"), None)
    return config, steps, final


def load_tilt(name, runs=None):
    p = os.path.join(runs or RUNS, name, "tilt_v2.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def series(steps, key):
    rs = [r for r in steps if key in r and r[key] is not None]
    return (np.array([r["step"] for r in rs]), [r[key] for r in rs])


def layer_agg(steps, key, agg="min"):
    xs, ys = series(steps, key)
    fn = {"min": min, "max": max,
          "median": lambda v: float(np.median(v))}[agg]
    return xs, np.array([fn(v) for v in ys])


def s_blk(steps):
    """Raw-metric block statistic eta_t * lam_plga (wave-5 declared)."""
    xs, ys = [], []
    for r in steps:
        if r.get("lam_plga") is not None and "lr" in r:
            xs.append(r["step"])
            ys.append(r["lam_plga"] * r["lr"])
    return np.array(xs), np.array(ys)


def s_pre_live(steps, block="plga"):
    """Metric-correct block statistic eta_t * lam_pre_live_<block>."""
    key = f"lam_pre_live_{block}"
    xs, ys = [], []
    for r in steps:
        if r.get(key) is not None and "lr" in r and np.isfinite(r[key]):
            xs.append(r["step"])
            ys.append(r[key] * r["lr"])
    return np.array(xs), np.array(ys)


def last_quarter(xs, ys):
    if len(xs) == 0:
        return np.array([])
    t = xs.max()
    return np.asarray(ys, float)[np.asarray(xs) > 0.75 * t]


def lq_median(xs, ys):
    v = last_quarter(xs, ys)
    return float(np.median(v)) if len(v) else None


def lq_mean(xs, ys):
    v = last_quarter(xs, ys)
    return float(np.mean(v)) if len(v) else None


def moving_block_bootstrap(vals, stat=np.median, blocklen=T.BOOT_BLOCKLEN,
                           n=T.BOOT_N, alpha=T.BOOT_ALPHA, seed=0):
    """Overlapping moving-block bootstrap interval for a time-series
    statistic; returns None when the series is shorter than one block
    (declared interval-free)."""
    vals = np.asarray(vals, float)
    m = len(vals)
    if m < blocklen:
        return None
    nblocks = int(np.ceil(m / blocklen))
    starts_max = m - blocklen + 1
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        picks = rng.integers(0, starts_max, nblocks)
        sample = np.concatenate([vals[s:s + blocklen] for s in picks])[:m]
        stats.append(stat(sample))
    return [float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2))]


def ci_disjoint(a, b):
    if a is None or b is None:
        return None
    return bool(a[1] < b[0] or b[1] < a[0])


def spearman_perm(x, y, n_perm=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)

    def pear(a, b):
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    rho = pear(rx, ry)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(pear(rx, rng.permutation(ry))) >= abs(rho):
            cnt += 1
    return rho, (cnt + 1) / (n_perm + 1)


def onset_step(steps, thresh=T.FLATTEN_THR, agg="min"):
    xs, ys = layer_agg(steps, "rowmap_sigma_med", agg)
    below = xs[ys < thresh]
    return int(below[0]) if len(below) else None


def onset_layer(steps, thresh=T.FLATTEN_THR):
    """The layer that first crosses the flattening threshold (the fixed
    per-run force-law coordinate; removes min-layer argmin switching)."""
    xs, ys = series(steps, "rowmap_sigma_med")
    for x, v in zip(xs, ys):
        for li, s in enumerate(v):
            if s < thresh:
                return li
    return None


def layer_series(steps, li):
    xs, ys = series(steps, "rowmap_sigma_med")
    return xs, np.array([v[li] for v in ys])


# ---------------------------------------------------------------------------
# endpoints and the sensitivity sweep


def endpoint_A(steps, thr=T.FLATTEN_THR, agg="max"):
    """All-layer flattening: last-quarter mean of the agg-over-layers
    median singular value below thr (declared primary uses agg=max)."""
    xs, ys = layer_agg(steps, "rowmap_sigma_med", agg)
    v = lq_mean(xs, ys)
    return {"value": v, "pass": bool(v is not None and v < thr),
            "thr": thr, "agg": agg}


def endpoint_B(final, thr=T.MGEN_THR):
    if final is None or "m_gen" not in final:
        return {"value": None, "pass": False, "thr": thr}
    v = final["m_gen"].get("GLM")
    out = {"value": v, "pass": bool(v is not None and v < thr), "thr": thr}
    if "m_gen_ci" in final and "GLM" in final["m_gen_ci"]:
        out["ci"] = final["m_gen_ci"]["GLM"]
    return out


def sensitivity_sweep(steps):
    """Every binary flattening endpoint re-evaluated over the declared
    threshold and aggregation grid."""
    out = {}
    for thr in T.FLATTEN_SWEEP:
        for agg in T.LAYER_AGGS:
            e = endpoint_A(steps, thr=thr, agg=agg)
            out[f"A_thr{thr}_{agg}"] = {"value": e["value"],
                                        "pass": e["pass"]}
    return out


# ---------------------------------------------------------------------------
# wave-5 window machinery (declared 250-step windows, raw statistic)


def w5_force_windows(steps, wlen=T.W5_WLEN):
    ux, uy = layer_agg(steps, "rowmap_sigma_med", "min")
    bx, by = s_blk(steps)
    dt = {r["step"]: r.get("dtheta_plga") for r in steps
          if r.get("dtheta_plga") is not None}
    gp = {r["step"]: r.get("gnorm_plga") for r in steps
          if r.get("gnorm_plga") is not None}
    if len(ux) == 0:
        return [], 0
    out, dropped = [], 0
    t0, tmax = int(ux.min()), int(ux.max())
    for w0 in range(t0, tmax - wlen + 1, wlen):
        w1 = w0 + wlen
        iu0 = np.searchsorted(ux, w0, side="right") - 1
        iu1 = np.searchsorted(ux, w1, side="right") - 1
        if iu0 < 0 or iu1 <= iu0:
            dropped += 1
            continue
        vs = [dt[s] ** 2 for s in range(w0, w1) if s in dt]
        ps = [gp[s] ** 2 for s in range(w0, w1) if s in gp]
        if not vs or not all(np.isfinite(vs)) or not all(np.isfinite(ps)):
            dropped += 1
            continue
        ib = np.argmin(np.abs(bx - w0)) if len(bx) else None
        sb = (float(by[ib]) if ib is not None
              and abs(bx[ib] - w0) <= wlen / 2 else None)
        out.append({"w0": w0, "u": float(uy[iu0]),
                    "du": float(uy[iu1] - uy[iu0]),
                    "V": float(np.mean(vs)), "P": float(np.mean(ps)),
                    "S_blk": sb})
    return out, dropped


def tertile_monotone(windows):
    """Declared tertile clause: within each u tertile, the mean
    decrement -du is monotone increasing across V tertiles.  Returns
    (monotone_bool_or_None, table); None when any cell is empty
    (insufficient for the clause)."""
    if len(windows) < 9:
        return None, {}
    us = np.array([w["u"] for w in windows])
    vs = np.array([w["V"] for w in windows])
    ds = np.array([-w["du"] for w in windows])
    uq = np.quantile(us, [1 / 3, 2 / 3])
    vq = np.quantile(vs, [1 / 3, 2 / 3])
    ub = np.digitize(us, uq)
    vb = np.digitize(vs, vq)
    table, mono = {}, True
    for i in range(3):
        row = []
        for j in range(3):
            sel = (ub == i) & (vb == j)
            if not sel.any():
                return None, table
            row.append(float(ds[sel].mean()))
        table[f"u_tertile_{i}"] = row
        if not (row[0] <= row[1] <= row[2]):
            mono = False
    return mono, table


def force_law_verdict(pooled, per_cell_windows, rho_pass=T.RHO_PASS,
                      p_pass=T.P_PASS, rho_kill=T.RHO_KILL,
                      min_windows=T.MIN_WINDOWS,
                      insuff=T.INSUFF_WINDOWS):
    """The declared force-law rule (shared by the corrected wave-5
    re-audit and W4): both correlates negative with p below the level,
    AND the pooled tertile pattern monotone.  Kill: |rho_uV| or
    |rho_uP| below rho_kill, or a non-monotone tertile pattern in every
    cell.  Insufficient below the window floor."""
    n = len(pooled)
    out = {"n_windows": n}
    if n < insuff:
        out["verdict"] = "insufficient"
        return out
    du = np.array([w["du"] for w in pooled])
    uv = np.array([w["u"] * w["V"] for w in pooled])
    up = np.array([w["u"] * w["P"] for w in pooled])
    rho_v, p_v = spearman_perm(du, uv)
    rho_p, p_p = spearman_perm(du, up)
    mono, table = tertile_monotone(pooled)
    cell_mono = {}
    for cell, ws in per_cell_windows.items():
        m, _ = tertile_monotone(ws)
        cell_mono[cell] = m
    out.update({"spearman_uV": rho_v, "p_uV": p_v,
                "spearman_uP": rho_p, "p_uP": p_p,
                "tertile_monotone": mono, "tertile_table": table,
                "per_cell_tertile": cell_mono})
    all_cells_nonmono = (len(cell_mono) > 0
                         and all(m is False for m in cell_mono.values()))
    if abs(rho_v) < rho_kill or abs(rho_p) < rho_kill or all_cells_nonmono:
        out["verdict"] = "kill"
    elif (n >= min_windows and rho_v < rho_pass and p_v < p_pass
          and rho_p < rho_pass and p_p < p_pass and mono is True):
        out["verdict"] = "pass"
    elif n < min_windows:
        out["verdict"] = "insufficient"
    else:
        out["verdict"] = "fail"
    return out


# ---------------------------------------------------------------------------
# corrected wave-5 re-audit (the declared letters of wave5_prereg.md)

W5_SUB = ["w5-base-lr3e-4", "w5-base-lr3e-4-s2", "w5-base-lr3e-4-s3"]
W5_COL = ["w5-base-lr1e-3", "w5-base-lr1e-3-s2", "w5-base-lr1e-3-s3"]


def corrected_wave5_report(runs=None):
    rep = {"note": "declared-letter re-audit; the archived "
                   "w5_gate_report.json is the as-implemented record"}
    data = {}
    for n in (W5_SUB + W5_COL + ["w5-sgdm-lr3e-2", "w5-sgdm-lr1e-1",
                                 "w5-b8-lr1e-3", "w5-b64-lr1e-3",
                                 "w5-last-b8-lr1e-3", "w5-frzplga-lr1e-3",
                                 "w5-frzplga-lr2e-3", "w5-rand1-lr1e-3",
                                 "w5-last-nowd-lr1e-3",
                                 "w5-frzvfix-lr1e-3"]):
        try:
            data[n] = load_run(n, runs)
        except FileNotFoundError:
            pass

    # P1': sub cells sustain lq median S_blk in [2, 40]; collapsing < 1;
    # SGDM collapsing < sub/2.  Bootstrap interval on every median.
    p1 = {}
    ok = True
    for n in W5_SUB + W5_COL:
        _, st, _ = data[n]
        xs, ys = s_blk(st)
        med = lq_median(xs, ys)
        ci = moving_block_bootstrap(last_quarter(xs, ys))
        cell_ok = (2 <= med <= 40) if n in W5_SUB else (med < 1)
        p1[n] = {"lq_med": med, "ci": ci, "pass": bool(cell_ok)}
        ok = ok and cell_ok
    _, st_sub, _ = data["w5-sgdm-lr3e-2"]
    _, st_col, _ = data["w5-sgdm-lr1e-1"]
    ssub = lq_median(*s_blk(st_sub))
    scol = lq_median(*s_blk(st_col))
    sg_ok = scol < ssub / 2
    p1["sgdm"] = {"sub": ssub, "col": scol, "pass": bool(sg_ok)}
    p1["verdict"] = "confirmed" if (ok and sg_ok) else "falsified"
    rep["P1_declared"] = p1

    # P2': per collapsing cell, excursion above 38 within +-250 of
    # onset OR quench transit within the surrounding 1000 (the archival
    # operationalization: a record above the boundary within +-500 of
    # onset AND one of the first four post-onset records below 1); the
    # declared falsifier (below boundary/4 throughout the +-250 window)
    # is evaluated too, which the archival code never did.
    p2 = {}
    bnd = T.BOUNDARY["adamw"]
    for n in W5_COL:
        _, st, _ = data[n]
        on = onset_step(st)
        bx, by = s_blk(st)
        e = {"onset": on}
        if on is None:
            e["verdict"] = "insufficient"
        else:
            near = by[np.abs(bx - on) <= 250]
            around = by[(bx >= on - 500) & (bx <= on + 500)]
            post = by[bx > on][:4]
            exc = bool(len(near) and (near > bnd).any())
            transit = bool(len(around) and (around > bnd).any()
                           and len(post) and (post < 1).any())
            falsifier = bool(len(near) and near.max() < bnd / 4)
            e.update({"excursion": exc, "quench_transit": transit,
                      "falsifier_below_quarter": falsifier,
                      "verdict": "confirmed" if (exc or transit)
                      else "falsified"})
        p2[n] = e
    p2["verdict"] = ("confirmed" if all(
        p2[n].get("verdict") == "confirmed" for n in W5_COL)
        else "falsified")
    rep["P2_declared"] = p2

    # P3': the declared gate with BOTH correlates and the tertile clause
    pooled, per_cell = [], {}
    total_dropped = 0
    for n in W5_COL:
        _, st, _ = data[n]
        on = onset_step(st)
        ws, dropped = w5_force_windows(st)
        total_dropped += dropped
        sel = [w for w in ws if on is not None and w["w0"] < on
               and w["S_blk"] is not None
               and w["S_blk"] > T.BOUNDARY["adamw"]]
        per_cell[n] = sel
        pooled += sel
    p3 = force_law_verdict(pooled, per_cell, min_windows=0,
                           insuff=1)
    p3["dropped_windows"] = total_dropped
    p3["selection"] = "pre-onset windows with raw S_blk > 38 "\
                      "(the declared, metric-mismatched selection)"
    rep["P3_declared"] = p3

    # P5': batch monotonicity AND the batch-8 last-position escape
    depths = {}
    for b, n in ((8, "w5-b8-lr1e-3"), (32, "w5-base-lr1e-3"),
                 (64, "w5-b64-lr1e-3")):
        _, st, _ = data[n]
        xs, ys = layer_agg(st, "rowmap_sigma_med", "min")
        depths[b] = {"lq": lq_mean(xs, ys),
                     "ci": moving_block_bootstrap(last_quarter(xs, ys),
                                                  stat=np.mean)}
    mono = (depths[8]["lq"] > depths[32]["lq"] > depths[64]["lq"])
    _, st_lb8, _ = data["w5-last-b8-lr1e-3"]
    on = onset_step(st_lb8)
    esc = None
    if on is not None:
        xs, ys = layer_agg(st_lb8, "rowmap_sigma_med", "min")
        after = ys[(xs > on) & (xs <= on + 2400)]
        esc = bool(len(after) and after.max() > T.FLATTEN_THR)
    p5 = {"depths": {k: v["lq"] for k, v in depths.items()},
          "cis": {k: v["ci"] for k, v in depths.items()},
          "monotone_deeper_at_larger_batch": bool(mono),
          "last_b8_onset": on,
          "last_b8_escape_within_2400": esc,
          "escape_clause": ("insufficient" if esc is None else bool(esc)),
          "verdict": "confirmed" if (mono and esc) else "falsified"}
    rep["P5_declared"] = p5

    # P6': frozen-coupling cells; kill on at-least-one-layer flattening
    p6 = {}
    killed = False
    for n in ("w5-frzplga-lr1e-3", "w5-frzplga-lr2e-3"):
        _, st, fin = data[n]
        xs, ys = series(st, "rowmap_sigma_med")
        nl = len(ys[-1])
        lastq = [[v[li] for x, v in zip(xs, ys) if x > 0.75 * xs.max()]
                 for li in range(nl)]
        med = [float(np.median(v)) for v in lastq]
        flat_any = any(m < T.FLATTEN_THR for m in med)
        eA = endpoint_A(st)
        eB = endpoint_B(fin)
        p6[n] = {"lastq_median_per_layer": med, "flat_any": flat_any,
                 "endpoint_A": eA["pass"], "endpoint_B": eB["pass"],
                 "pass": bool(not flat_any and not eA["pass"]
                              and not eB["pass"])}
        killed = killed or flat_any
    p6["verdict"] = "kill" if killed else "pass"
    rep["P6_declared"] = p6

    # P8': PRIMARY-endpoint match plus the deep-layer 10x tilt clause
    _, st_r, fin_r = data["w5-rand1-lr1e-3"]
    _, st_b, fin_b = data["w5-base-lr1e-3"]
    match_A = endpoint_A(st_r)["pass"] == endpoint_A(st_b)["pass"]
    match_B = endpoint_B(fin_r)["pass"] == endpoint_B(fin_b)["pass"]
    tilt = {"clause": "insufficient"}
    tr, tb = load_tilt("w5-rand1-lr1e-3", runs), load_tilt(
        "w5-base-lr1e-3", runs)
    if tr and tb and T.TILT_CKPT in tr["ckpts"] \
            and T.TILT_CKPT in tb["ckpts"]:
        tilt = {"ckpt": T.TILT_CKPT}
        within = True
        for mode in ("current", "collapsed"):
            br = tr["ckpts"][T.TILT_CKPT][mode]["batches"]
            bb = tb["ckpts"][T.TILT_CKPT][mode]["batches"]
            nl = len(br[0])
            deep = nl - 1
            mr = float(np.median([b[deep]["C_fro"] for b in br]))
            mb = float(np.median([b[deep]["C_fro"] for b in bb]))
            ratio = mb / mr if mr > 0 else float("inf")
            tilt[mode] = {"deep_layer": deep, "rand1": mr,
                          "base": mb, "suppression": ratio}
            within = within and (ratio <= T.TILT_FACTOR)
        tilt["clause"] = bool(within)
    p8 = {"primary_A_match": bool(match_A),
          "primary_B_match": bool(match_B), "tilt": tilt,
          "verdict": "confirmed" if (match_A and match_B
                                     and tilt.get("clause") is True)
          else "falsified"}
    rep["P8_declared"] = p8

    # P9': within 4000 steps of the freeze: loss above 2x its
    # pre-freeze median, or non-finite.  No early-stop disjunct.
    cfg, st, _ = data["w5-frzvfix-lr1e-3"]
    fz = cfg.get("freeze_v_at", 2000)
    loss = [(r["step"], r["loss"]) for r in st if "loss" in r]
    pre = [v for s, v in loss if s <= fz]
    win = [(s, v) for s, v in loss if fz < s <= fz + 4000]
    nonfin = any(v is None or not np.isfinite(v) for _, v in win)
    blow = bool(pre and any(v is not None and np.isfinite(v)
                            and v > 2 * np.median(pre) for _, v in win))
    rep["P9_declared"] = {
        "freeze_at": fz, "nonfinite_in_window": bool(nonfin),
        "loss_blowup_in_window": blow,
        "last_step": int(loss[-1][0]),
        "verdict": "confirmed" if (nonfin or blow) else "falsified"}

    rep["P4_P7_P10"] = ("archival implementation matches the declared "
                        "letter for these tests; see w5_gate_report.json")

    # sensitivity sweep for every cell entering a binary endpoint
    sweep = {}
    for n in W5_SUB + W5_COL + ["w5-b8-lr1e-3", "w5-b64-lr1e-3",
                                "w5-frzplga-lr1e-3", "w5-frzplga-lr2e-3",
                                "w5-rand1-lr1e-3"]:
        _, st, _ = data[n]
        sweep[n] = sensitivity_sweep(st)
    rep["endpoint_sensitivity"] = sweep
    return rep


# ---------------------------------------------------------------------------
# wave-6 window machinery (dyn-mode coordinates, declared 125 steps)


def dyn_windows(steps, block="plga", wlen=T.W6_WLEN, layer=None):
    """Per-window wave-6 force-law records: u at the window start and
    its decrement for the FIXED onset layer, oscillation power V (the
    variance of the drift-removed signed dyn projection series), flip
    rate, captured power fraction, and drive power P (mean
    gnorm_<block>^2).  Returns (windows, dropped_count)."""
    if layer is None:
        layer = onset_layer(steps)
    if layer is None:
        return [], 0
    ux, uy = layer_series(steps, layer)
    pk, dk_, gk = (f"mode_proj_{block}_dyn", f"dtheta_{block}",
                   f"gnorm_{block}")
    proj = {r["step"]: r[pk] for r in steps if pk in r}
    dth = {r["step"]: r.get(dk_) for r in steps
           if r.get(dk_) is not None}
    gp = {r["step"]: r.get(gk) for r in steps if r.get(gk) is not None}
    if len(ux) == 0 or not proj:
        return [], 0
    out, dropped = [], 0
    t0, tmax = int(ux.min()), int(ux.max())
    for w0 in range(t0, tmax - wlen + 1, wlen):
        w1 = w0 + wlen
        iu0 = np.searchsorted(ux, w0, side="right") - 1
        iu1 = np.searchsorted(ux, w1, side="right") - 1
        if iu0 < 0 or iu1 <= iu0:
            dropped += 1
            continue
        cs = [proj[s] for s in range(w0, w1) if s in proj]
        ds = [dth[s] for s in range(w0, w1) if s in dth]
        ps = [gp[s] ** 2 for s in range(w0, w1) if s in gp]
        if len(cs) < wlen // 2 or not all(np.isfinite(cs)):
            dropped += 1
            continue
        cs = np.array(cs, float)
        V = float(np.var(cs - cs.mean()))
        sgn = np.sign(cs)
        flips = float(np.mean(sgn[1:] != sgn[:-1])) if len(cs) > 1 else 0.0
        cap = [c * c / (d * d) for c, d in zip(cs, ds) if d and d > 0]
        out.append({"w0": w0, "u": float(uy[iu0]),
                    "du": float(uy[iu1] - uy[iu0]), "V": V,
                    "flip": flips,
                    "captured": float(np.median(cap)) if cap else None,
                    "P": float(np.mean(ps)) if ps else None})
    return out, dropped


# ---------------------------------------------------------------------------
# wave-6 verdicts (wave6_prereg.md W1-W8)

W6_REF = ["w6-ref-lr2e-3", "w6-ref-lr2e-3-s2", "w6-ref-lr2e-3-s3"]
W6_COL = ["w6-base-lr1e-3", "w6-base-lr1e-3-s2", "w6-base-lr1e-3-s3"]
W6_SUB = ["w6-base-lr3e-4", "w6-base-lr3e-4-s2", "w6-base-lr3e-4-s3"]


def w1_reference(runs=None):
    cells = {}
    n_b = n_a02 = n_a01 = 0
    for n in W6_REF:
        _, st, fin = load_run(n, runs)
        eB = endpoint_B(fin)
        eA01 = endpoint_A(st, thr=0.1)
        eA02 = endpoint_A(st, thr=0.2)
        cells[n] = {"B": eB, "A_0.1": eA01, "A_0.2": eA02,
                    "sweep": sensitivity_sweep(st)}
        n_b += eB["pass"]
        n_a02 += (eB["pass"] and eA02["pass"])
        n_a01 += eA01["pass"]
    try:
        _, st24, fin24 = load_run("w6-ref-lr2e-3-24k", runs)
        cells["w6-ref-lr2e-3-24k"] = {
            "B": endpoint_B(fin24), "A_0.2": endpoint_A(st24, thr=0.2)}
        anneal = cells["w6-ref-lr2e-3-24k"]["B"]["pass"]
    except FileNotFoundError:
        anneal = None
    ok = (n_a02 >= 2 and n_a01 >= 1)
    return {"cells": cells, "seeds_B": n_b, "seeds_A02_with_B": n_a02,
            "seeds_A01": n_a01, "anneal_survival_B": anneal,
            "verdict": "pass" if ok else "kill"}


def _contrast(sub_cells, col_cells, stat_fn, runs=None):
    pairs = []
    for ns, nc in zip(sub_cells, col_cells):
        _, ss, _ = load_run(ns, runs)
        _, sc, _ = load_run(nc, runs)
        xs, ys = stat_fn(ss)
        xc, yc = stat_fn(sc)
        ms, mc = lq_median(xs, ys), lq_median(xc, yc)
        cis = moving_block_bootstrap(last_quarter(xs, ys))
        cic = moving_block_bootstrap(last_quarter(xc, yc))
        ratio = (ms / mc) if (ms is not None and mc not in (None, 0)) \
            else None
        pairs.append({"sub": ns, "col": nc, "sub_med": ms, "col_med": mc,
                      "ratio": ratio, "sub_ci": cis, "col_ci": cic,
                      "disjoint": ci_disjoint(cis, cic),
                      "pass": bool(ratio is not None
                                   and ratio >= T.CONTRAST_RATIO
                                   and ci_disjoint(cis, cic))})
    npass = sum(p["pass"] for p in pairs)
    return {"pairs": pairs, "n_pass": npass,
            "verdict": "pass" if npass >= T.SEED_MAJORITY else "kill"}


def w2_contrast(runs=None):
    return {"raw": _contrast(W6_SUB, W6_COL, s_blk, runs),
            "pre_live": _contrast(W6_SUB, W6_COL, s_pre_live, runs)}


def w3_carrier_existence(runs=None):
    cells = {}
    any_osc = False
    all_dead = True
    for n in W6_COL:
        _, st, _ = load_run(n, runs)
        on = onset_step(st)
        ws, dropped = dyn_windows(st)
        pre = [w for w in ws
               if on is not None and on - 500 <= w["w0"] < on]
        oscn = [w for w in pre if w["flip"] > T.FLIP_OSC]
        capn = [w for w in oscn
                if w["captured"] is not None
                and w["captured"] > T.CAPTURED_MIN]
        cells[n] = {"onset": on, "n_pre_windows": len(pre),
                    "n_oscillatory": len(oscn),
                    "n_osc_with_power": len(capn),
                    "flips": [w["flip"] for w in pre],
                    "dropped": dropped}
        if pre:
            if len(oscn) >= max(1, len(pre) // 2) and len(capn) == len(oscn):
                any_osc = True
            if any(w["flip"] >= T.FLIP_DEAD for w in pre):
                all_dead = False
    if not any(c["n_pre_windows"] for c in cells.values()):
        verdict = "insufficient"
    elif any_osc:
        verdict = "confirmed"
    elif all_dead:
        verdict = "kill"
    else:
        verdict = "fail"
    return {"cells": cells, "verdict": verdict}


def w4_force_law(runs=None):
    pooled, per_cell = [], {}
    total_dropped = 0
    for n in W6_COL:
        _, st, _ = load_run(n, runs)
        on = onset_step(st)
        ws, dropped = dyn_windows(st)
        total_dropped += dropped
        sel = [w for w in ws if on is not None and w["w0"] < on
               and w["flip"] > T.FLIP_OSC and w["P"] is not None]
        per_cell[n] = sel
        pooled += sel
    out = force_law_verdict(pooled, per_cell)
    out["dropped_windows"] = total_dropped
    out["selection"] = ("pre-onset windows with dyn-mode flip rate "
                        f"> {T.FLIP_OSC} (the calibrated oscillatory "
                        "marker; no raw-vs-38 selection)")
    return out


def _depth(steps):
    xs, ys = layer_agg(steps, "rowmap_sigma_med", "min")
    return lq_mean(xs, ys), moving_block_bootstrap(
        last_quarter(xs, ys), stat=np.mean)


def w5_causal(runs=None):
    pairs = []
    for damp, base in (("w6-damp-plga-lr1e-3", "w6-base-lr1e-3"),
                       ("w6-damp-plga-lr1e-3-s2", "w6-base-lr1e-3-s2")):
        _, sd, _ = load_run(damp, runs)
        _, sb, _ = load_run(base, runs)
        dd, cid = _depth(sd)
        db, cib = _depth(sb)
        ratio = dd / db if (dd is not None and db) else None
        pairs.append({"damp": damp, "base": base, "damp_lq": dd,
                      "base_lq": db, "ratio_shallower": ratio,
                      "damp_ci": cid, "base_ci": cib,
                      "disjoint": ci_disjoint(cid, cib),
                      "pass": bool(ratio is not None
                                   and ratio >= T.DEPTH_RATIO
                                   and ci_disjoint(cid, cib))})
    both_pass = all(p["pass"] for p in pairs)
    both_null = all(not p["pass"] and p["disjoint"] is False
                    for p in pairs)
    out = {"damp_pairs": pairs}
    try:
        _, se, _ = load_run("w6-excite-plga-lr3e-4", runs)
        _, sb3, _ = load_run("w6-base-lr3e-4", runs)
        xe, ye = series(se, "rowmap_sigma_med")
        nl = len(ye[-1])
        med_e = [float(np.median([v[li] for x, v in zip(xe, ye)
                                  if x > 0.75 * xe.max()]))
                 for li in range(nl)]
        xb, yb = series(sb3, "rowmap_sigma_med")
        med_b = [float(np.median([v[li] for x, v in zip(xb, yb)
                                  if x > 0.75 * xb.max()]))
                 for li in range(nl)]
        out["excite"] = {
            "excite_lastq_median": med_e, "base_lastq_median": med_b,
            "excite_flattens": any(m < T.FLATTEN_THR for m in med_e),
            "base_flattens": any(m < T.FLATTEN_THR for m in med_b)}
    except FileNotFoundError:
        pass
    try:
        _, sa, _ = load_run("w6-damp-attn-lr1e-3", runs)
        da, _ci = _depth(sa)
        out["damp_attn_exploratory"] = {"lq": da}
    except FileNotFoundError:
        pass
    out["verdict"] = ("pass" if both_pass
                      else "kill" if both_null else "fail")
    return out


def w6_noise(runs=None):
    _, sb, fb = load_run("w6-base-lr1e-3", runs)
    _, sa, fa = load_run("w6-b8acc4-lr1e-3", runs)
    db, cib = _depth(sb)
    da, cia = _depth(sa)
    ident = {
        "base_lq": db, "acc_lq": da, "base_ci": cib, "acc_ci": cia,
        "endpoint_match": (endpoint_A(sa)["pass"] == endpoint_A(sb)["pass"]
                           and endpoint_B(fa)["pass"]
                           == endpoint_B(fb)["pass"]),
        "within_ci": bool(cib is not None and da is not None
                          and cib[0] <= da <= cib[1])}
    ident["pass"] = bool(ident["endpoint_match"] and ident["within_ci"])
    out = {"identity_control": ident}
    if not ident["pass"]:
        out["verdict"] = "halt"
        return out
    _, s8, _ = load_run("w6-b8tok-lr1e-3", runs)
    _, s64, _ = load_run("w6-b64tok-lr1e-3", runs)
    d8, ci8 = _depth(s8)
    d64, ci64 = _depth(s64)
    ordering = (d8 is not None and db is not None and d64 is not None
                and d8 < db < d64)
    disjoint = (ci_disjoint(ci8, cib) and ci_disjoint(cib, ci64))
    out["matched_tokens"] = {
        "b8tok_lq": d8, "b32_lq": db, "b64tok_lq": d64,
        "cis": [ci8, cib, ci64],
        "ordering_deeper_at_smaller_batch": bool(ordering),
        "cis_disjoint": bool(disjoint)}
    out["verdict"] = ("pass" if (ordering and disjoint)
                      else "kill")
    return out


def w7_carrier_freeze(runs=None):
    out = {}
    per_lr = {}
    for lr, base in (("lr1e-3", "w6-base-lr1e-3"),
                     ("lr2e-3", "w6-ref-lr2e-3")):
        _, sb, _ = load_run(base, runs)
        db, _cb = _depth(sb)
        onb = onset_step(sb)
        e = {"base_lq": db, "base_onset": onb}
        for blk in ("attn", "ffn"):
            _, sf, _ = load_run(f"w6-frz{blk}-{lr}", runs)
            df, _cf = _depth(sf)
            onf = onset_step(sf)
            ratio = df / db if (df is not None and db) else None
            delayed = (onf is None
                       or (onb is not None
                           and onf > T.ONSET_DELAY * onb))
            e[blk] = {"lq": df, "onset": onf, "depth_ratio": ratio,
                      "suppressed_or_delayed": bool(delayed),
                      "depth_matched": bool(ratio is not None
                                            and ratio <= T.DEPTH_RATIO)}
        per_lr[lr] = e
    out["per_lr"] = per_lr
    attn_suppressed = all(per_lr[lr]["attn"]["suppressed_or_delayed"]
                          for lr in per_lr)
    ffn_persists = all(per_lr[lr]["ffn"]["depth_matched"]
                       for lr in per_lr)
    every_matched = all(per_lr[lr][blk]["depth_matched"]
                        and not per_lr[lr][blk]["suppressed_or_delayed"]
                        for lr in per_lr for blk in ("attn", "ffn"))
    out["verdict"] = ("pass" if (attn_suppressed and ffn_persists)
                      else "kill" if every_matched else "fail")
    return out


def w8_parity(runs=None):
    _, st, _ = load_run("w6-dag005-lr1e-3", runs)
    on = onset_step(st)
    ratios = []
    for r in st:
        a, b = r.get("fdcurv_collapse"), r.get("fdcurv_collapse_fullobj")
        if a and b and (on is None or r["step"] <= on):
            for x, y in zip(a, b):
                if np.isfinite(x) and np.isfinite(y) and x != 0:
                    ratios.append(abs(y / x))
    if not ratios:
        return {"verdict": "insufficient", "n": 0}
    med = float(np.median(ratios))
    return {"median_ratio": med, "n": len(ratios),
            "verdict": "confirmed" if 0.5 <= med <= 2.0 else "falsified"}


def wave6_report(runs=None):
    rep = {}
    for name, fn in (("W1_reference", w1_reference),
                     ("W2_contrast", w2_contrast),
                     ("W3_carrier_existence", w3_carrier_existence),
                     ("W4_force_law", w4_force_law),
                     ("W5_causal", w5_causal),
                     ("W6_noise", w6_noise),
                     ("W7_carrier_freeze", w7_carrier_freeze),
                     ("W8_parity", w8_parity)):
        try:
            rep[name] = fn(runs)
        except FileNotFoundError as e:
            rep[name] = {"verdict": "insufficient",
                         "missing": str(e)}
    rep["GATES"] = {
        "causal_W5": rep["W5_causal"].get("verdict"),
        "carrier_W7": rep["W7_carrier_freeze"].get("verdict"),
        "integrity_W6a": rep["W6_noise"].get("verdict")}
    return rep


# ---------------------------------------------------------------------------
# wave 7: avalanche statistics and constant-eta stationarity
# (wave7_prereg.md; every constant from thresholds.py AV_*/W7_*)

import avalanche as av  # noqa: E402  (pure numpy, no back-import)

W7_CONST = ["w7-const-lr1e-3", "w7-const-lr1e-3-s2", "w7-const-lr1e-3-s3"]
W7_CONST_SUB = "w7-const-lr3e-4"
W7_DOFF = "w7-doff-lr1e-3"
W7_BLOCKS = ("plga", "phi", "attn", "ffn", "rest")
# archived flattening onsets, pinned in wave7_prereg.md disclosure 9
W7_ONSETS = {"w6-base-lr1e-3": 1350, "w6-base-lr1e-3-s2": 1900,
             "w6-base-lr1e-3-s3": 1600}

_GN_CACHE = {}


def gnorm_series(name, block, runs=None):
    """(T, warmup, y) for the per-step gnorm_<block> series; y[i] is
    step i+1.  Returns None when the block series is absent (reduced
    logs) or identically zero (frozen block)."""
    key = (name, block, runs)
    if key not in _GN_CACHE:
        config, steps, _ = load_run(name, runs)
        k = f"gnorm_{block}"
        rows = sorted((r["step"], r[k]) for r in steps if k in r)
        if not rows or len(rows) < 100:
            _GN_CACHE[key] = None
        else:
            y = np.array([v for _, v in rows], float)
            if not np.any(y > 0):
                _GN_CACHE[key] = None
            else:
                _GN_CACHE[key] = (int(rows[-1][0]),
                                  int(config.get("warmup", 0)), y)
    return _GN_CACHE[key]


def cell_events(name, block="plga", mult=T.AV_THR_MULT,
                win=T.AV_MED_WIN, runs=None, warmup=None):
    """Post-warmup events for one cell/block (warmup overridable for
    the constant-eta stationary window)."""
    g = gnorm_series(name, block, runs)
    if g is None:
        return None
    tt, wu, y = g
    ev, dropped = av.extract_events(y, mult, win,
                                    wu if warmup is None else warmup)
    return {"T": tt, "warmup": wu, "events": ev,
            "n_warmup_dropped": dropped}


def excluded_pair(name_a, name_b, block="plga", mult=T.AV_THR_MULT,
                  win=T.AV_WIN_SWEEP[1], runs=None):
    """Events of cell A with the cross-run data-event exclusion against
    cell B (shared data order), plus both dropped counts."""
    ca = cell_events(name_a, block, mult, win, runs)
    cb = cell_events(name_b, block, mult, win, runs)
    if ca is None or cb is None:
        return None
    kept, n_dil, n_start = av.exclude_matched(ca["events"], cb["events"],
                                              T.AV_MATCH_LAG)
    return {"T": ca["T"], "warmup": ca["warmup"], "events": kept,
            "n_raw": len(ca["events"]), "n_dropped_dilated": n_dil,
            "n_dropped_startrule": n_start}


def _tail_clauses(sizes, y, warmup, mult, win, seed):
    """The W7.1a clause battery for one cell: CSN fit, gof, Vuong
    comparisons, and the joint (alpha, S_max) surrogate check."""
    out = {"n_events": len(sizes)}
    if len(sizes) < T.AV_MIN_EVENTS:
        out["verdict"] = "insufficient"
        return out
    fit = av.fit_powerlaw(np.asarray(sizes), T.AV_XMIN_GRID)
    if fit is None or fit["n_tail"] < T.AV_MIN_EVENTS:
        out["fit"] = fit
        out["verdict"] = "insufficient"
        return out
    out["fit"] = fit
    out["alpha_ci"] = av.alpha_bootstrap_ci(sizes, fit["xmin"],
                                            T.BOOT_N, seed)
    out["gof_p"] = av.gof_pvalue(sizes, fit, T.AV_GOF_BOOT, seed + 1,
                                 T.AV_XMIN_GRID)
    r_e, p_e, w_e = av.vuong_vs_exponential(sizes, fit["alpha"],
                                            fit["xmin"])
    r_l, p_l, w_l = av.vuong_vs_lognormal(sizes, fit["alpha"],
                                          fit["xmin"])
    out["vs_exponential"] = {"R": r_e, "p": p_e, "winner": w_e}
    out["vs_lognormal"] = {"R": r_l, "p": p_l, "winner": w_l}
    smax = float(np.max(sizes))
    joint = {}
    for kind in ("ar1", "phase"):
        # determinism repair (wave 8): the archived code derived this
        # seed from the salted Python hash of the kind string, which is
        # process-dependent (PYTHONHASHSEED), so the archived report's
        # surrogate leaves were never byte-reproducible; verdicts are
        # unaffected (verified leaf by leaf against the archived
        # report).  Seeds are now derived deterministically.
        ss = av.surrogate_stats(y, kind, T.AV_N_SURR, mult, win,
                                warmup,
                                seed + {"ar1": 0, "phase": 1}[kind])
        dom = np.nanmean((ss["smax"] >= smax)
                         & (ss["alpha"] <= fit["alpha"]))
        joint[kind] = {"p_joint": float(dom),
                       "outside": bool(dom < (1 - T.AV_SURR_Q)),
                       "surr_smax_q95": float(
                           np.nanquantile(ss["smax"], T.AV_SURR_Q)),
                       "surr_cv_q95": float(
                           np.nanquantile(ss["cv"], T.AV_SURR_Q))}
    out["surrogates"] = joint
    out["clause_gof"] = bool(out["gof_p"] >= T.AV_GOF_P)
    out["clause_exp"] = bool(p_e < T.AV_LR_P and w_e == "powerlaw")
    out["clause_exp_favored"] = bool(p_e < T.AV_LR_P
                                     and w_e == "exponential")
    out["clause_ln_favored"] = bool(p_l < T.AV_LR_P
                                    and w_l == "lognormal")
    # surrogate size statistics are reported only (prereg W7.1a): for
    # 1-2-step events the size marginal is reproduced by rank-remapped
    # surrogates by construction; the surrogate kill lives in (d)
    out["pl_consistent"] = bool(out["clause_gof"]
                                and out["clause_exp"])
    if out["pl_consistent"]:
        out["verdict"] = "pass"
    elif out["clause_exp_favored"]:
        out["verdict"] = "exponential_favored"
    elif out["clause_ln_favored"]:
        out["verdict"] = "lognormal_favored"
    else:
        out["verdict"] = "indistinguishable"
    return out


def w7_1_tail(runs=None):
    """W7.1a-e: tail fits, common exponent, sweep survival, clustering,
    stationarity splits, over the archived primary six (plga)."""
    rep = {"cells": {}}
    alphas, cis = {}, {}
    for col, sub in zip(W6_COL, W6_SUB):
        ex = excluded_pair(col, sub, runs=runs)
        g = gnorm_series(col, "plga", runs)
        tt, wu, y = g
        sizes = np.array([e["size"] for e in ex["events"]])
        cell = {"exclusion": {k: ex[k] for k in
                              ("n_raw", "n_dropped_dilated",
                               "n_dropped_startrule")}}
        cell["a"] = _tail_clauses(sizes, y, wu, T.AV_THR_MULT,
                                  T.AV_MED_WIN, seed=0)
        if "fit" in cell["a"] and cell["a"].get("fit"):
            alphas[col] = cell["a"]["fit"]["alpha"]
            cis[col] = cell["a"].get("alpha_ci")
        # (c) sweep
        sweep = {}
        for m in T.AV_THR_SWEEP:
            for w in T.AV_WIN_SWEEP:
                exs = excluded_pair(col, sub, mult=m, win=w, runs=runs)
                sz = np.array([e["size"] for e in exs["events"]])
                f = (av.fit_powerlaw(sz, T.AV_XMIN_GRID)
                     if len(sz) >= T.AV_MIN_EVENTS else None)
                ent = {"n_events": len(sz),
                       "alpha": f["alpha"] if f else None,
                       "n_tail": f["n_tail"] if f else None}
                if m == 5.0 and w == T.AV_MED_WIN and len(sz) >= 10:
                    fx = av.fit_powerlaw(sz, T.AV_XMIN_GRID)
                    if fx:
                        _, p_e5, w_e5 = av.vuong_vs_exponential(
                            sz, fx["alpha"], fx["xmin"])
                        ent["exp_wins"] = bool(p_e5 < T.AV_LR_P
                                               and w_e5 == "exponential")
                sweep[f"{m}x_{w}"] = ent
        cell["c_sweep"] = sweep
        # (d) clustering: the surrogate null's primary site
        cv = av.interevent_cv(ex["events"])
        cvci = av.cv_bootstrap_ci(ex["events"], T.BOOT_N, seed=2)
        env = cell["a"].get("surrogates", {})
        envs = [env[k]["surr_cv_q95"] for k in ("ar1", "phase")
                if k in env and np.isfinite(env[k]["surr_cv_q95"])]
        cv_hi = max(envs) if envs else np.nan
        cv_lo = min(envs) if envs else np.nan
        cell["d"] = {"cv": cv, "ci": cvci, "surr_cv_q95_max": cv_hi,
                     "surr_cv_q95_min": cv_lo,
                     "inside_both_env": bool(
                         cv is not None and envs and cv <= cv_lo),
                     "pass": bool(cv is not None and cvci is not None
                                  and cv >= T.AV_CV_MIN
                                  and cvci[0] > 1.0
                                  and (np.isnan(cv_hi)
                                       or cv > cv_hi))}
        # (e) stationarity splits
        onset = W7_ONSETS.get(col) or onset_step(
            load_run(col, runs)[1])
        phases = av.phase_splits(onset, tt, wu, T.AV_DESCENT_LEN)
        e_rep = {}
        for pn, (lo, hi) in phases.items():
            pe = av.events_in(ex["events"], lo, hi)
            sz = np.array([e["size"] for e in pe])
            e_rep[pn] = {
                "n": len(pe), "span": (lo, hi),
                "rate_per_1k": len(pe) / max(hi - lo, 1) * 1000,
                "rate_ci": av.rate_ci(pe, hi, T.AV_RATE_BOOT_BLOCK,
                                      T.BOOT_N, 3, lo=lo),
                "q99": float(np.quantile(sz, 0.99)) if len(sz) else None,
                "alpha": (av.fit_powerlaw(sz)["alpha"]
                          if len(sz) >= T.AV_MIN_EVENTS else
                          "insufficient")}
        cell["e_phases"] = e_rep
        rep["cells"][col] = cell
    # sub-critical whole-run reference (excluded symmetrically)
    rep["sub_cells"] = {}
    for col, sub in zip(W6_COL, W6_SUB):
        exs = excluded_pair(sub, col, runs=runs)
        sz = np.array([e["size"] for e in exs["events"]])
        f = (av.fit_powerlaw(sz, T.AV_XMIN_GRID)
             if len(sz) >= T.AV_MIN_EVENTS else None)
        rep["sub_cells"][sub] = {
            "n": len(sz),
            "q99": float(np.quantile(sz, 0.99)) if len(sz) else None,
            "smax": float(sz.max()) if len(sz) else None,
            "alpha": f["alpha"] if f else "insufficient",
            "slope_c80": av.ccdf_slope(sz)}
    # verdicts
    a_states = [rep["cells"][c]["a"].get("verdict") for c in W6_COL]
    n_pass = sum(s == "pass" for s in a_states)
    n_exp = sum(s == "exponential_favored" for s in a_states)
    if n_pass >= T.SEED_MAJORITY:
        v_a = "pass"
    elif n_exp >= T.SEED_MAJORITY:
        v_a = "kill"
    elif "lognormal_favored" in a_states:
        v_a = "fail_lognormal_favored"
    elif all(s == "insufficient" for s in a_states):
        v_a = "insufficient"
    else:
        v_a = "fail"
    rep["a_verdict"] = v_a
    # b: common exponent
    if len(alphas) == 3:
        vals = list(alphas.values())
        gaps = [abs(vals[i] - vals[j])
                for i in range(3) for j in range(i + 1, 3)]
        pairs = list(cis.values())
        overl = sum(not ci_disjoint(pairs[i], pairs[j])
                    for i in range(3) for j in range(i + 1, 3))
        if max(gaps) > T.AV_ALPHA_KILL or overl == 0:
            v_b = "kill"
        elif max(gaps) <= T.AV_ALPHA_TOL and overl >= 2:
            v_b = "pass"
        else:
            v_b = "fail"
        rep["b_common"] = {"alphas": alphas, "max_gap": max(gaps),
                           "n_overlapping_pairs": overl, "verdict": v_b}
    else:
        rep["b_common"] = {"alphas": alphas, "verdict": "insufficient"}
    # c: sweep survival
    c_ok = c_kill = 0
    for col in W6_COL:
        prim = rep["cells"][col]["a"].get("fit")
        if not prim:
            continue
        sw = rep["cells"][col]["c_sweep"]
        devs = [abs(e["alpha"] - prim["alpha"]) for e in sw.values()
                if e["alpha"] is not None
                and e.get("n_tail", 0) >= T.AV_MIN_EVENTS]
        ok = all(d <= T.AV_SWEEP_TOL for d in devs) if devs else False
        five = sw.get(f"5.0x_{T.AV_MED_WIN}", {})
        died = (five.get("n_events", 0) < 25
                or five.get("exp_wins", False))
        c_ok += ok
        c_kill += died
    rep["c_verdict"] = ("kill" if c_kill >= T.SEED_MAJORITY else
                        "pass" if c_ok >= T.SEED_MAJORITY else "fail")
    # d: clustering
    d_pass = sum(rep["cells"][c]["d"]["pass"] for c in W6_COL)
    d_null = sum(1 for c in W6_COL
                 if (ci := rep["cells"][c]["d"]["ci"]) is not None
                 and ci[0] <= 1.0)
    d_surr = sum(rep["cells"][c]["d"]["inside_both_env"]
                 for c in W6_COL)
    rep["d_verdict"] = ("pass" if d_pass >= T.SEED_MAJORITY else
                        "kill" if (d_null == 3
                                   or d_surr >= T.SEED_MAJORITY)
                        else "fail")
    # e: splits
    trunc = 0
    indist = 0
    for c in W6_COL:
        ph = rep["cells"][c]["e_phases"]
        pre, post = ph["pre_onset"], ph["post_quench"]
        if pre["q99"] is not None and post["q99"] is not None:
            if post["q99"] < pre["q99"]:
                trunc += 1
            rate_over = not ci_disjoint(pre["rate_ci"],
                                        post["rate_ci"])
            q_close = (post["q99"] > 0
                       and 0.5 <= pre["q99"] / post["q99"] <= 2.0)
            if rate_over and q_close:
                indist += 1
    sub_gt = sum(
        1 for col, sub in zip(W6_COL, W6_SUB)
        if rep["sub_cells"][sub]["q99"] is not None
        and rep["cells"][col]["a"].get("fit")
        and rep["sub_cells"][sub]["q99"] > float(np.quantile(
            [e["size"] for e in
             excluded_pair(col, sub, runs=runs)["events"]], 0.99)))
    rep["e_splits"] = {
        "n_trunc_direction": trunc, "n_indistinguishable": indist,
        "n_sub_cutoff_gt": sub_gt,
        "verdict": ("kill" if indist == 3 else
                    "pass" if trunc >= T.SEED_MAJORITY
                    and sub_gt >= T.SEED_MAJORITY else "fail")}
    votes = [rep["a_verdict"], rep["b_common"]["verdict"],
             rep["c_verdict"], rep["d_verdict"],
             rep["e_splits"]["verdict"]]
    rep["verdict"] = ("kill" if "kill" in (rep["a_verdict"],
                                           rep["d_verdict"]) else
                      "pass" if all(v == "pass" for v in votes) else
                      "insufficient" if rep["a_verdict"] ==
                      "insufficient" else "fail")
    return rep


def _diverged(steps, start=2000):
    """Divergence clause: non-finite loss, or the 101-step running
    median of the loss exceeding twice its step-2000 reference, after
    the start of the stationary window."""
    xs, ys = series(steps, "loss")
    ys = np.asarray(ys, float)
    if not np.all(np.isfinite(ys[np.asarray(xs) > start])):
        return True
    ref_band = ys[(np.asarray(xs) >= start - 100)
                  & (np.asarray(xs) <= start + 100)]
    if not len(ref_band):
        return False
    ref = float(np.median(ref_band))
    med = av.running_median(ys, 101)
    return bool(np.any(med[np.asarray(xs) > start] > 2.0 * ref))


def w7_2_stationarity(runs=None, cells=None):
    """W7.2: constant-eta stationarity in thirds of [2000, 24000].
    cells defaults to the wave-7 constant trio; wave 8 reuses the
    identical clause battery on its 7.5e-4 trio."""
    rep = {"cells": {}}
    n_div = n_pass = n_kill = 0
    n_phase = 0
    for n in (cells or W7_CONST):
        config, st, fin = load_run(n, runs)
        cell = {"diverged": _diverged(st, T.W7_CONST_STAT_START)}
        if cell["diverged"]:
            n_div += 1
            rep["cells"][n] = cell
            continue
        g = gnorm_series(n, "plga", runs)
        tt, _, y = g
        ev, _ = av.extract_events(y, T.AV_THR_MULT, T.AV_MED_WIN,
                                  T.W7_CONST_STAT_START)
        lo, hi = T.W7_CONST_STAT_START, tt
        third = (hi - lo) // T.AV_STAT_THIRDS
        sizes_full = np.array([e["size"] for e in ev])
        f_full = (av.fit_powerlaw(sizes_full, T.AV_XMIN_GRID)
                  if len(sizes_full) >= T.AV_MIN_EVENTS else None)
        xs_m, ys_m = s_pre_live(st)
        thirds = []
        for k in range(T.AV_STAT_THIRDS):
            a, b = lo + k * third, lo + (k + 1) * third
            pe = av.events_in(ev, a + 1, b)
            sz = np.array([e["size"] for e in pe])
            mwin = ys_m[(xs_m > a) & (xs_m <= b)]
            thirds.append({
                "span": (a, b), "n": len(pe),
                "rate_per_1k": len(pe) / third * 1000,
                "rate_ci": av.rate_ci(pe, b, T.AV_RATE_BOOT_BLOCK,
                                      T.BOOT_N, 4, lo=a),
                "alpha": (av.fit_powerlaw(sz)["alpha"]
                          if len(sz) >= T.AV_MIN_EVENTS
                          else "insufficient"),
                "margin_med": (float(np.median(mwin))
                               if len(mwin) else None)})
        cell["thirds"] = thirds
        cell["alpha_full"] = f_full["alpha"] if f_full else None
        rates = [t["rate_per_1k"] for t in thirds]
        margins = [t["margin_med"] for t in thirds]
        adj_ok = all(
            rates[i] > 0 and rates[i + 1] > 0
            and max(rates[i] / rates[i + 1],
                    rates[i + 1] / rates[i]) < T.AV_STAT_DRIFT
            and not ci_disjoint(thirds[i]["rate_ci"],
                                thirds[i + 1]["rate_ci"])
            for i in range(len(rates) - 1))
        al_ok = all(
            abs(t["alpha"] - f_full["alpha"]) <= T.AV_ALPHA_TOL
            for t in thirds
            if f_full and isinstance(t["alpha"], float))
        mono = (lambda v: all(x < z for x, z in zip(v, v[1:]))
                or all(x > z for x, z in zip(v, v[1:])))
        m_ok = (None not in margins and min(margins) > 0
                and max(margins) / min(margins) < T.AV_STAT_DRIFT
                and not mono(margins))
        r_drift = (mono(rates) and rates[0] > 0 and rates[-1] > 0
                   and max(rates[0] / rates[-1],
                           rates[-1] / rates[0]) >= T.AV_STAT_DRIFT)
        m_drift = (None not in margins and mono(margins)
                   and min(margins) > 0
                   and max(margins) / min(margins) >= T.AV_STAT_DRIFT)
        cell["pass"] = bool(adj_ok and al_ok and m_ok)
        cell["kill"] = bool(r_drift and m_drift)
        n_pass += cell["pass"]
        n_kill += cell["kill"]
        eB = endpoint_B(fin)
        eA02 = endpoint_A(st, thr=0.2)
        cell["phase"] = {"B": eB, "A_0.2": eA02}
        n_phase += (eB["pass"] or eA02["pass"])
        rep["cells"][n] = cell
    # sub-critical control, reported
    try:
        _, st_s, fin_s = load_run(W7_CONST_SUB, runs)
        g = gnorm_series(W7_CONST_SUB, "plga", runs)
        ev_s, _ = av.extract_events(g[2], T.AV_THR_MULT, T.AV_MED_WIN,
                                    T.W7_CONST_STAT_START)
        rep["sub_control"] = {
            "n_events": len(ev_s),
            "diverged": _diverged(st_s, T.W7_CONST_STAT_START),
            "B": endpoint_B(fin_s)}
    except FileNotFoundError:
        rep["sub_control"] = None
    rep["n_diverged"] = n_div
    rep["phase_check"] = ("collapsed-phase" if n_phase >= T.SEED_MAJORITY
                          else "edge-riding")
    if n_div >= 2:
        rep["verdict"] = "halt"
    elif n_pass >= T.SEED_MAJORITY:
        rep["verdict"] = "pass"
    elif n_kill >= T.SEED_MAJORITY:
        rep["verdict"] = "kill"
    else:
        rep["verdict"] = "fail"
    return rep


def w7_3_margin(runs=None, cells=None, archived=True):
    """W7.3: upper-edge compression and fast relaxation of the margin
    distribution eta_t * lam_pre_live_plga (gate on const cells).
    cells defaults to the wave-7 constant trio; archived=False skips
    the archived-descriptive block (wave-8 reuse)."""
    rep = {"cells": {}, "blocks_reported": {}}
    n_a = n_a_kill = n_b = 0
    rets = []
    for n in (cells or W7_CONST):
        _, st, _ = load_run(n, runs)
        xs, ys = s_pre_live(st)
        w = ys[np.asarray(xs) > T.W7_CONST_STAT_START]
        skew = av.margin_skew(w)
        relax = av.margin_relaxation(w, T.AV_RELAX_PROBES)
        cell = {"skew_A": skew, "relax": relax,
                "n_probes": int(len(w))}
        if skew is not None:
            n_a += skew < T.AV_MARGIN_SKEW
            n_a_kill += skew > T.AV_MARGIN_SKEW_KILL
        if relax and relax["n_exc"] >= 5:
            n_b += relax["frac_relaxed"] >= T.AV_RELAX_FRAC
            rets.append(relax["med_return"])
        rep["cells"][n] = cell
        for blk in ("attn", "ffn", "phi"):
            xs2, ys2 = s_pre_live(st, blk)
            w2 = ys2[np.asarray(xs2) > T.W7_CONST_STAT_START]
            rep["blocks_reported"][f"{n}:{blk}"] = av.margin_skew(w2)
    # archived, descriptive per phase
    rep["archived_descriptive"] = {}
    for n in (W6_COL + W6_SUB if archived else []):
        _, st, _ = load_run(n, runs)
        xs, ys = s_pre_live(st)
        onset = W7_ONSETS.get(n)
        if onset:
            pre = ys[(np.asarray(xs) > 1000) & (np.asarray(xs) < onset)]
            post = ys[np.asarray(xs) > onset + T.AV_DESCENT_LEN]
            rep["archived_descriptive"][n] = {
                "pre_skew": av.margin_skew(pre),
                "post_skew": av.margin_skew(post)}
        else:
            rep["archived_descriptive"][n] = {
                "full_skew": av.margin_skew(
                    ys[np.asarray(xs) > 1000])}
    v_a = ("pass" if n_a >= T.SEED_MAJORITY else
           "kill" if n_a_kill >= T.SEED_MAJORITY else "fail")
    if not rets:
        v_b = "insufficient"
    elif n_b >= T.SEED_MAJORITY:
        v_b = "pass"
    elif np.median(rets) > 2 * T.AV_RELAX_PROBES:
        v_b = "kill"
    else:
        v_b = "fail"
    rep["a_verdict"], rep["b_verdict"] = v_a, v_b
    rep["verdict"] = ("pass" if v_a == v_b == "pass" else
                      "kill" if "kill" in (v_a, v_b) else "fail")
    return rep


def w7_4_scaling(runs=None):
    """W7.4: size-duration scaling and Omori aftershock decay at the
    lowered threshold, archived flattening seeds (gate) plus const
    cells (reported)."""
    rep = {"cells": {}}
    n_g = n_g_null = n_o = n_o_null = 0
    for col, sub in zip(W6_COL, W6_SUB):
        ex = excluded_pair(col, sub, mult=T.AV_DUR_MULT, runs=runs)
        g = gnorm_series(col, "plga", runs)
        tt, wu, y = g
        sd = av.size_duration(ex["events"], T.BOOT_N, seed=5)
        n_d2 = sd["n"] if sd else 0
        surr = av.surrogate_stats(y, "ar1", T.AV_N_SURR,
                                  T.AV_DUR_MULT, T.AV_MED_WIN, wu,
                                  seed=6, dur_mult=T.AV_DUR_MULT)
        g_env = float(np.nanquantile(surr["gamma"], T.AV_SURR_Q))
        om = av.omori(ex["events"], tt, T.AV_N_MAIN, T.AV_OMORI_EARLY,
                      T.AV_OMORI_LATE, T.AV_N_SHIFT, seed=7)
        cell = {"n_dur2": n_d2, "size_duration": sd,
                "gamma_surr_q95": g_env, "omori": om}
        if n_d2 < T.AV_MIN_SD_EVENTS or sd is None:
            cell["gamma_verdict"] = "insufficient"
        else:
            ok = (sd["ci"][0] > T.AV_GAMMA_FLOOR
                  and (np.isnan(g_env) or sd["gamma"] > g_env))
            null = sd["ci"][0] <= T.AV_GAMMA_FLOOR <= sd["ci"][1]
            cell["gamma_verdict"] = "pass" if ok else "fail"
            n_g += ok
            n_g_null += null
        if om is None:
            cell["omori_verdict"] = "insufficient"
        else:
            ok = (om["ratio"] > T.AV_OMORI_RATIO
                  and om["p"] < T.AV_LR_P)
            cell["omori_verdict"] = "pass" if ok else "fail"
            n_o += ok
            n_o_null += om["ratio"] <= 1.0
        rep["cells"][col] = cell
    for n in W7_CONST:
        try:
            ce = cell_events(n, mult=T.AV_DUR_MULT, runs=runs,
                             warmup=T.W7_CONST_STAT_START)
            sd = av.size_duration(ce["events"], T.BOOT_N, seed=8)
            om = av.omori(ce["events"], ce["T"], T.AV_N_MAIN,
                          T.AV_OMORI_EARLY, T.AV_OMORI_LATE,
                          T.AV_N_SHIFT, seed=9)
            rep["cells"][n] = {"size_duration": sd, "omori": om,
                               "reported": True}
        except FileNotFoundError:
            pass
    v_g = ("pass" if n_g >= T.SEED_MAJORITY else
           "kill" if n_g_null == 3 else "fail")
    v_o = ("pass" if n_o >= T.SEED_MAJORITY else
           "kill" if n_o_null == 3 else "fail")
    rep["gamma_verdict"], rep["omori_verdict"] = v_g, v_o
    rep["verdict"] = ("pass" if v_g == v_o == "pass" else
                      "kill" if "kill" in (v_g, v_o) else "fail")
    return rep


W7_FROZEN = {
    "w6-frzattn-lr1e-3": ("attn", "w6-base-lr1e-3"),
    "w6-frzattn-lr2e-3": ("attn", "w6-ref-lr2e-3"),
    "w6-frzffn-lr1e-3": ("ffn", "w6-base-lr1e-3"),
    "w6-frzffn-lr2e-3": ("ffn", "w6-ref-lr2e-3"),
    "w5-frzplga-lr1e-3": ("plga", "w6-base-lr1e-3"),
    "w5-frzplga-lr2e-3": ("plga", "w6-ref-lr2e-3"),
}


def _prop_matrix(name, runs=None):
    """Events per live block and the ordered-pair coincidence matrix
    with shift envelopes."""
    evb, tt = {}, None
    for blk in W7_BLOCKS:
        ce = cell_events(name, blk, runs=runs)
        if ce is not None and len(ce["events"]) >= 5:
            evb[blk] = ce["events"]
            tt = ce["T"]
    if tt is None:
        return None
    mat = av.coincidence_matrix(evb, tt, T.AV_COINC_LAG,
                                n_shift=T.AV_N_SHIFT, seed=10)
    total = sum(len(v) for v in evb.values())
    span = tt - max(cell_events(name, next(iter(evb)),
                                runs=runs)["warmup"], 0)
    return {"blocks": sorted(evb), "T": tt,
            "total_rate_per_1k": total / max(span, 1) * 1000,
            "matrix": {f"{a}->{b}": m
                       for (a, b), m in mat.items() if m}}


def w7_5_propagation(runs=None):
    """W7.5: rerouting of cross-block burst propagation under block
    freezes."""
    rep = {"base": {}, "frozen": {}}
    for n in W6_COL + ["w6-ref-lr2e-3"]:
        rep["base"][n] = _prop_matrix(n, runs)
    n_route = n_uniform = n_cells = 0
    for n, (frz, base) in W7_FROZEN.items():
        try:
            pm = _prop_matrix(n, runs)
        except FileNotFoundError:
            rep["frozen"][n] = {"verdict": "insufficient"}
            continue
        bm = rep["base"].get(base)
        if pm is None or bm is None:
            rep["frozen"][n] = {"verdict": "insufficient"}
            continue
        n_cells += 1
        live_ok = (pm["total_rate_per_1k"]
                   >= T.AV_LIVE_RATE_MIN * bm["total_rate_per_1k"])
        strong = [k for k, m in pm["matrix"].items()
                  if m["ratio"] >= T.AV_COINC_RATIO
                  and m["ratio"] > m.get("env_hi", np.inf)]
        changed = []
        all_close = True
        for k, m in pm["matrix"].items():
            b = bm["matrix"].get(k)
            if b is None or b["ratio"] <= 0:
                continue
            rr = m["ratio"] / b["ratio"]
            if rr >= T.AV_REROUTE_DELTA or rr <= 1 / T.AV_REROUTE_DELTA:
                changed.append((k, rr))
                all_close = False
        cell = {"frozen_block": frz, "base": base,
                "total_rate_per_1k": pm["total_rate_per_1k"],
                "base_rate_per_1k": bm["total_rate_per_1k"],
                "live_rate_ok": bool(live_ok),
                "n_strong_pairs": len(strong),
                "strong_pairs": strong[:6],
                "n_changed_pairs": len(changed),
                "changed_pairs": [(k, round(r, 2))
                                  for k, r in changed[:6]],
                "reduced_log": len(pm["blocks"]) < 4}
        route = live_ok and len(strong) >= 1 and len(changed) >= 1
        uniform = (not live_ok) and all_close
        cell["reroute"] = bool(route)
        cell["uniform_reduction"] = bool(uniform)
        n_route += route
        n_uniform += uniform
        rep["frozen"][n] = cell
    if n_cells == 0:
        rep["verdict"] = "insufficient"
    elif n_route == n_cells:
        rep["verdict"] = "pass"
    elif n_uniform == n_cells:
        rep["verdict"] = "kill"
    else:
        rep["verdict"] = "fail"
    return rep


def w7_6_finite_size(runs=None):
    """W7.6: descriptive 20.6M vs 110M cutoff comparison (declared:
    no gate, no scaling claim)."""
    pairs = [("w6-base-lr1e-3", "w5e7-base-lr1e-3"),
             ("w6-base-lr3e-4", "w5e7-base-lr3e-4"),
             ("w6-frzattn-lr1e-3", "w6e8-frzattn-lr1e-3")]
    rep = {"pairs": {}, "verdict": "descriptive"}
    for small, big in pairs:
        ent = {}
        for tag, name in (("20.6M", small), ("110M", big)):
            try:
                blocks = {}
                for blk in ("plga", "phi"):
                    ce = cell_events(name, blk, runs=runs)
                    if ce is None:
                        continue
                    sz = np.array([e["size"] for e in ce["events"]])
                    blocks[blk] = {
                        "n": len(sz),
                        "smax": float(sz.max()) if len(sz) else None,
                        "q99": (float(np.quantile(sz, 0.99))
                                if len(sz) else None)}
                ent[tag] = blocks
            except FileNotFoundError:
                ent[tag] = None
        if ent.get("20.6M") and ent.get("110M"):
            s, b = ent["20.6M"].get("plga"), ent["110M"].get("plga")
            if s and b and s["q99"] and b["q99"]:
                ent["declared_direction_plga"] = bool(
                    b["q99"] > s["q99"])
        rep["pairs"][f"{small}|{big}"] = ent
    return rep


def w7_7_dose(runs=None):
    """W7.7: noise-dose response at matched tokens, with the
    accumulation identity clause."""
    cells = {"b8tok": "w6-b8tok-lr1e-3", "base": "w6-base-lr1e-3",
             "b64tok": "w6-b64tok-lr1e-3", "b8acc4": "w6-b8acc4-lr1e-3"}
    rep = {"cells": {}}
    for tag, name in cells.items():
        config, st, _ = load_run(name, runs)
        ce = cell_events(name, runs=runs)
        ev, tt, wu = ce["events"], ce["T"], ce["warmup"]
        span = tt - wu
        sz = np.array([e["size"] for e in ev])
        toks = config["batch"] * config.get("accum", 1) * config["ctx"]
        rep["cells"][tag] = {
            "name": name, "n_events": len(ev),
            "rate_per_1k": len(ev) / span * 1000,
            "rate_ci": av.rate_ci(ev, tt, T.AV_RATE_BOOT_BLOCK,
                                  T.BOOT_N, 11, lo=wu),
            "rate_per_Mtok": len(ev) / (span * toks) * 1e6,
            "q99": float(np.quantile(sz, 0.99)) if len(sz) else None}
    c = rep["cells"]
    ident = (c["b8acc4"]["rate_per_1k"] >= c["base"]["rate_ci"][0]
             and c["b8acc4"]["rate_per_1k"] <= c["base"]["rate_ci"][1])
    rep["identity_clause"] = bool(ident)
    hi = c["b8tok"]["rate_per_1k"] > c["base"]["rate_per_1k"] \
        and ci_disjoint(c["b8tok"]["rate_ci"], c["base"]["rate_ci"])
    lo = c["base"]["rate_per_1k"] > c["b64tok"]["rate_per_1k"] \
        and ci_disjoint(c["base"]["rate_ci"], c["b64tok"]["rate_ci"])
    inv = c["b64tok"]["rate_per_1k"] > c["b8tok"]["rate_per_1k"] \
        and ci_disjoint(c["b64tok"]["rate_ci"], c["b8tok"]["rate_ci"])
    if not ident:
        rep["verdict"] = "descriptive"
    elif hi and lo:
        rep["verdict"] = "pass"
    elif inv:
        rep["verdict"] = "kill"
    else:
        rep["verdict"] = "fail"
    return rep


def w7_8_data_order(runs=None):
    """W7.8: the data-order control validates the matched-step
    exclusion rule."""
    base = cell_events("w6-base-lr1e-3", runs=runs)
    doff = cell_events(W7_DOFF, runs=runs)
    tt = doff["T"]
    co = av.coincidence_shift_p(
        doff["events"], [e["start"] for e in base["events"]],
        tt, T.AV_MATCH_LAG, T.AV_N_SHIFT, seed=12)
    sz_d = np.array([e["size"] for e in doff["events"]])
    sz_b = np.array([e["size"] for e in base["events"]])
    f_d = (av.fit_powerlaw(sz_d, T.AV_XMIN_GRID)
           if len(sz_d) >= T.AV_MIN_EVENTS else None)
    f_b = (av.fit_powerlaw(sz_b, T.AV_XMIN_GRID)
           if len(sz_b) >= T.AV_MIN_EVENTS else None)
    span_d = tt - doff["warmup"]
    span_b = base["T"] - base["warmup"]
    rate_d = len(doff["events"]) / span_d * 1000
    rate_b = len(base["events"]) / span_b * 1000
    _, st, fin = load_run(W7_DOFF, runs)
    rep = {
        "coincidence": co,
        "alpha_doff": f_d["alpha"] if f_d else None,
        "alpha_base": f_b["alpha"] if f_b else None,
        "rate_doff": rate_d, "rate_base": rate_b,
        "flattening": {"onset": onset_step(st),
                       "A_min": endpoint_A(st, agg="min"),
                       "B": endpoint_B(fin)}}
    ratio = co["ratio"] if co else np.inf
    a_close = (f_d and f_b
               and abs(f_d["alpha"] - f_b["alpha"]) <= T.AV_ALPHA_TOL)
    r_close = rate_b > 0 and 0.5 <= rate_d / rate_b <= 2.0
    if ratio > 3.0 or (rate_b > 0 and rate_d / rate_b < 0.2):
        rep["verdict"] = "kill"
    elif ratio <= T.AV_COINC_RATIO and a_close and r_close:
        rep["verdict"] = "pass"
    else:
        rep["verdict"] = "fail"
    return rep


def wave7_report(runs=None):
    rep = {}
    for name, fn in (("W7_1_tail", w7_1_tail),
                     ("W7_2_stationarity", w7_2_stationarity),
                     ("W7_3_margin", w7_3_margin),
                     ("W7_4_scaling", w7_4_scaling),
                     ("W7_5_propagation", w7_5_propagation),
                     ("W7_6_finite_size", w7_6_finite_size),
                     ("W7_7_dose", w7_7_dose),
                     ("W7_8_data_order", w7_8_data_order)):
        try:
            rep[name] = fn(runs)
        except FileNotFoundError as e:
            rep[name] = {"verdict": "insufficient", "missing": str(e)}
    rep["GATES"] = {
        "tail_W7_1": rep["W7_1_tail"].get("verdict"),
        "stationarity_W7_2": rep["W7_2_stationarity"].get("verdict"),
        "margin_W7_3": rep["W7_3_margin"].get("verdict"),
        "exclusion_integrity_W7_8": rep["W7_8_data_order"].get(
            "verdict")}
    return rep


# ---------------------------------------------------------------------------
# wave 8 (executable preregistration; frozen with wave8_prereg.md)

import sandpile as sp  # noqa: E402  (pure numpy; imports avalanche only)

W8_CONST75 = ["w8-const-lr7.5e-4", "w8-const-lr7.5e-4-s2",
              "w8-const-lr7.5e-4-s3"]
# exclusion partners: same seed suffix, shared data order, other rate
W8_EXCL_PARTNER = dict(zip(W8_CONST75, W7_CONST))
W8_PULSE = {"w8-pulse-const3e-4": 3e-4,
            "w8-pulse-const7.5e-4": 7.5e-4,
            "w8-pulse-const1e-3": 1e-3}
# un-pulsed same-rate exclusion partners for the pulse cells
W8_PULSE_PARTNER = {"w8-pulse-const3e-4": "w7-const-lr3e-4",
                    "w8-pulse-const7.5e-4": "w8-const-lr7.5e-4",
                    "w8-pulse-const1e-3": "w7-const-lr1e-3"}
W8_FLOOR_CELLS = [(0.1, "w6-base-lr1e-3"), (0.3, "w8-floor0.3-lr1e-3"),
                  (0.55, "w8-floor0.55-lr1e-3"),
                  (1.0, "w7-const-lr1e-3")]
W8_HOLD_CELL = "w8-hold6k-lr1e-3"
W8_SHORT_CELL = "w8-short6k-lr1e-3"
W8_LATE_CELL = "w8-lateann-lr1e-3"
# late-anneal source: lowest-numbered w7 const seed NOT row-map
# collapsed at step 8000 (min-layer median sigma 0.1367; pinned in
# wave8_prereg.md disclosure 6)
W8_LATE_SOURCE = "w7-const-lr1e-3"
W8_WIDTH = [(128, ["w8-h2-lr1e-3", "w8-h2-lr1e-3-s2"]),
            (256, W6_COL),
            (512, ["w8-h8-lr1e-3", "w8-h8-lr1e-3-s2"])]
W8_110M = "w5e7-base-lr1e-3"


def _dur_events_excluded(name, partner, runs=None):
    """D>=2-threshold events of a constant cell with the symmetric
    same-seed cross-rate data-event exclusion (W8.1 primary
    convention)."""
    ca = cell_events(name, mult=T.AV_DUR_MULT, runs=runs,
                     warmup=T.W7_CONST_STAT_START)
    if ca is None:
        return None
    try:
        cb = cell_events(partner, mult=T.AV_DUR_MULT, runs=runs,
                         warmup=T.W7_CONST_STAT_START)
    except FileNotFoundError:
        # partner not yet run: fall back to unexcluded events, flagged
        # by n_dropped_dilated = None (the declared convention applies
        # whenever the partner exists)
        cb = None
    if cb is None:
        return {"events": ca["events"], "n_raw": len(ca["events"]),
                "n_dropped_dilated": None, "T": ca["T"]}
    kept, n_dil, _ = av.exclude_matched(ca["events"], cb["events"],
                                        T.AV_MATCH_LAG)
    return {"events": kept, "n_raw": len(ca["events"]),
            "n_dropped_dilated": n_dil, "T": ca["T"]}


def w8_1_gamma(runs=None):
    """W8.1 (primary): superlinear size-duration scaling on the
    constant-drive cells.  Pass per trio: >= 2 evaluable cells with the
    gamma bootstrap CI excluding 1.0 from above AND gamma above both
    surrogate envelopes.  Kill: CI contains 1.0 in >= 2 evaluable cells
    of BOTH trios.  Band clause: every evaluable gamma inside
    W8_GAMMA_BAND (reported consistency, not universality)."""
    rep = {"cells": {}, "trios": {}}
    trio_names = {"const1e3_archived": list(zip(W7_CONST, W8_CONST75)),
                  "const75_new": list(zip(W8_CONST75, W7_CONST))}
    gammas = []
    for trio, pairs in trio_names.items():
        n_pass = n_null = n_eval = 0
        for name, partner in pairs:
            try:
                ex = _dur_events_excluded(name, partner, runs)
            except FileNotFoundError:
                rep["cells"][name] = {"verdict": "missing"}
                continue
            if ex is None:
                rep["cells"][name] = {"verdict": "insufficient",
                                      "n_dur2": 0}
                continue
            sd = av.size_duration(ex["events"], T.BOOT_N, seed=11)
            n_d2 = sd["n"] if sd else 0
            cell = {"n_raw": ex["n_raw"],
                    "n_dropped_dilated": ex["n_dropped_dilated"],
                    "n_dur2": n_d2, "size_duration": sd}
            if n_d2 < T.AV_MIN_SD_EVENTS or sd is None:
                cell["verdict"] = "insufficient"
            else:
                g = gnorm_series(name, "plga", runs)
                envs = {}
                for kind, sd_seed in (("ar1", 12), ("phase", 13)):
                    surr = av.surrogate_stats(
                        g[2], kind, T.AV_N_SURR, T.AV_DUR_MULT,
                        T.AV_MED_WIN, T.W7_CONST_STAT_START,
                        seed=sd_seed, dur_mult=T.AV_DUR_MULT)
                    envs[kind] = float(np.nanquantile(surr["gamma"],
                                                      T.AV_SURR_Q))
                cell["gamma_surr_q95"] = envs
                n_eval += 1
                gammas.append(sd["gamma"])
                ok = (sd["ci"][0] > T.AV_GAMMA_FLOOR
                      and all(np.isnan(e) or sd["gamma"] > e
                              for e in envs.values()))
                null = sd["ci"][0] <= T.AV_GAMMA_FLOOR <= sd["ci"][1]
                cell["verdict"] = "pass" if ok else "fail"
                n_pass += ok
                n_null += null
            rep["cells"][name] = cell
        rep["trios"][trio] = {
            "n_evaluable": n_eval, "n_pass": n_pass, "n_null": n_null,
            "verdict": ("insufficient" if n_eval < 2 else
                        "pass" if n_pass >= 2 else
                        "kill" if n_null >= 2 else "fail")}
    band_ok = all(T.W8_GAMMA_BAND[0] < gam < T.W8_GAMMA_BAND[1]
                  for gam in gammas)
    rep["band_clause"] = {"gammas": gammas, "band": T.W8_GAMMA_BAND,
                          "ok": bool(band_ok)}
    tv = [rep["trios"][t]["verdict"] for t in trio_names]
    if "insufficient" in tv:
        rep["verdict"] = "insufficient"
    elif all(v == "pass" for v in tv) and band_ok:
        rep["verdict"] = "pass"
    elif all(rep["trios"][t]["n_null"] >= 2 for t in trio_names
             if rep["trios"][t]["n_evaluable"] >= 2):
        rep["verdict"] = "kill"
    else:
        rep["verdict"] = "fail"
    return rep


def w8_2_anneal(runs=None):
    """W8.2: anneal-trajectory dependence.  (a,b) floor chain: the
    late-window ([8000, 12000]) event rate is monotone nondecreasing in
    the anneal floor across the ordered cells (pass: >= 2 of 3 adjacent
    pairs; kill: decreasing across >= 2 adjacent pairs with disjoint
    rate CIs).  (d) late-anneal: pure descent from a non-collapsed
    constant-drive checkpoint reaches row-map collapse and holds it
    (pass); the pre-written re-scope applies otherwise.  (c,e)
    hold/short cells reported."""
    rep = {"floor_chain": [], "cells": {}}
    rates = []
    for floor, name in W8_FLOOR_CELLS:
        wu = (T.W7_CONST_STAT_START if floor == 1.0 else None)
        ce = cell_events(name, runs=runs, warmup=wu)
        if ce is None:
            rep["floor_chain"].append({"floor": floor, "cell": name,
                                       "verdict": "missing"})
            rates.append(None)
            continue
        lw = sp.late_window_rate(ce["events"], T.W8_LATE_LO, 12000,
                                 T.AV_RATE_BOOT_BLOCK, T.BOOT_N,
                                 seed=14)
        rep["floor_chain"].append({"floor": floor, "cell": name, **lw})
        rates.append(lw)
    adj = []
    for a, b in zip(rates, rates[1:]):
        if a is None or b is None:
            adj.append(None)
        else:
            up = b["rate_per_1k"] >= a["rate_per_1k"]
            down_disjoint = (b["rate_per_1k"] < a["rate_per_1k"]
                             and ci_disjoint(a["rate_ci"],
                                             b["rate_ci"]))
            adj.append({"up": bool(up),
                        "down_disjoint": bool(down_disjoint)})
    rep["adjacent"] = adj
    n_up = sum(1 for x in adj if x and x["up"])
    n_down = sum(1 for x in adj if x and x["down_disjoint"])
    chain_v = ("insufficient" if any(x is None for x in adj) else
               "pass" if n_up >= 2 else
               "kill" if n_down >= 2 else "fail")
    rep["chain_verdict"] = chain_v
    # (d) late-anneal descent cell
    try:
        _, st, fin = load_run(W8_LATE_CELL, runs)
        onset = onset_step(st)
        eA = endpoint_A(st, agg="min")
        late = {"onset": onset, "endpoint_A_min": eA,
                "endpoint_B": endpoint_B(fin),
                "source": W8_LATE_SOURCE, "source_sigma_8000": 0.1367}
        late["verdict"] = ("pass" if onset is not None and eA["pass"]
                           else "fail")
        rep["cells"]["lateann"] = late
    except FileNotFoundError:
        rep["cells"]["lateann"] = {"verdict": "missing"}
    # (c) hold cell: onset segment reported
    try:
        _, st, fin = load_run(W8_HOLD_CELL, runs)
        onset = onset_step(st)
        seg = (None if onset is None else
               "plateau" if onset <= T.W8_HOLD else "descent")
        rep["cells"]["hold6k"] = {
            "onset": onset, "segment": seg,
            "endpoint_A_min": endpoint_A(st, agg="min"),
            "endpoint_B": endpoint_B(fin), "verdict": "descriptive"}
    except FileNotFoundError:
        rep["cells"]["hold6k"] = {"verdict": "missing"}
    # (e) short cell: shape-not-duration support, reported
    try:
        _, st, fin = load_run(W8_SHORT_CELL, runs)
        rep["cells"]["short6k"] = {
            "onset": onset_step(st),
            "endpoint_A_min": endpoint_A(st, agg="min"),
            "endpoint_B": endpoint_B(fin), "verdict": "descriptive"}
    except FileNotFoundError:
        rep["cells"]["short6k"] = {"verdict": "missing"}
    lv = rep["cells"]["lateann"].get("verdict")
    if chain_v == "kill":
        rep["verdict"] = "kill"
    elif chain_v == "pass" and lv == "pass":
        rep["verdict"] = "pass"
    elif "missing" in (chain_v, lv) or chain_v == "insufficient":
        rep["verdict"] = "insufficient"
    else:
        rep["verdict"] = "fail"
    return rep


def w8_3_susceptibility(runs=None):
    """W8.3 (decision pair with W8.1): perturbation-response at
    constant drive.  Pass: mean pulse response above the 95% placement
    envelope at 1e-3 AND 7.5e-4, and chi ordered chi(3e-4) <
    chi(7.5e-4) <= chi(1e-3) with the outer pair's CIs disjoint.
    Kill: response inside the envelope at both upper rates, or the
    outer ordering inverted with disjoint CIs."""
    rep = {"cells": {}}
    chis = {}
    above = {}
    for name, base_lr in W8_PULSE.items():
        try:
            config, st, _ = load_run(name, runs)
        except FileNotFoundError:
            rep["cells"][name] = {"verdict": "missing"}
            continue
        cell = {"base_lr": base_lr,
                "diverged": _diverged(st, T.W7_CONST_STAT_START)}
        if cell["diverged"]:
            cell["verdict"] = "halt"
            rep["cells"][name] = cell
            continue
        g = gnorm_series(name, "plga", runs)
        tt, _, y = g
        ev, _ = av.extract_events(y, T.AV_THR_MULT, T.AV_MED_WIN,
                                  T.W7_CONST_STAT_START)
        starts = list(T.W8_PULSE_STARTS)
        pr = sp.pulse_response(ev, starts, T.W8_PULSE_LEN,
                               T.W8_RESP_WIN, tt,
                               T.W7_CONST_STAT_START)
        R = pr["R"][pr["clean"]]
        cell["n_clean_pulses"] = int(pr["clean"].sum())
        direct, resp, _ = sp.classify_pulse_events(
            ev, starts, T.W8_PULSE_LEN, T.W8_RESP_WIN)
        cell["n_direct_pulse_events"] = len(direct)
        cell["n_response_events"] = len(resp)
        if cell["n_clean_pulses"] < T.W8_MIN_PULSES:
            cell["verdict"] = "insufficient"
            rep["cells"][name] = cell
            continue
        null = sp.placement_null(ev, starts, T.W8_PULSE_LEN,
                                 T.W8_RESP_WIN, tt,
                                 T.W7_CONST_STAT_START,
                                 T.W8_N_PLACE, seed=15)
        env = float(np.quantile(null["mean_R"], T.W8_NULL_Q))
        chance = float(null["mean_R"].mean())
        dose = (T.W8_PULSE_FACTOR - 1.0) * base_lr * T.W8_PULSE_LEN
        chi = sp.susceptibility(R, chance, dose, T.BOOT_N, seed=16)
        cell["envelope_q95"] = env
        cell["chance"] = chance
        cell["chi"] = chi
        cell["above_envelope"] = bool(chi["mean_R"] > env)
        cell["triggered_frac"] = float(
            (pr["n_resp"][pr["clean"]] > 0).mean())
        cell["triggered_env_q95"] = float(
            np.quantile(null["triggered"], T.W8_NULL_Q))
        # masked-median sensitivity variant (reported)
        mev, _ = sp.masked_median_events(y, T.AV_THR_MULT,
                                         T.AV_MED_WIN,
                                         T.W7_CONST_STAT_START,
                                         starts, T.W8_PULSE_LEN)
        prm = sp.pulse_response(mev, starts, T.W8_PULSE_LEN,
                                T.W8_RESP_WIN, tt,
                                T.W7_CONST_STAT_START)
        cell["mean_R_masked_median"] = float(
            prm["R"][prm["clean"]].mean())
        chis[base_lr] = chi
        above[base_lr] = cell["above_envelope"]
        cell["verdict"] = ("pass" if cell["above_envelope"]
                           else "fail")
        rep["cells"][name] = cell
    have = sorted(chis)
    if len(have) < 3:
        rep["verdict"] = "insufficient"
        return rep
    lo_lr, mid_lr, hi_lr = have
    order_ok = (chis[lo_lr]["chi"] < chis[mid_lr]["chi"]
                <= chis[hi_lr]["chi"])
    outer_disjoint = ci_disjoint(chis[lo_lr]["ci"], chis[hi_lr]["ci"])
    inverted = (chis[lo_lr]["chi"] > chis[hi_lr]["chi"]
                and outer_disjoint)
    rep["ordering"] = {"chis": {str(k): chis[k]["chi"] for k in have},
                       "order_ok": bool(order_ok),
                       "outer_disjoint": bool(outer_disjoint),
                       "inverted": bool(inverted)}
    if not above[mid_lr] and not above[hi_lr]:
        rep["verdict"] = "kill"
    elif inverted:
        rep["verdict"] = "kill"
    elif above[mid_lr] and above[hi_lr] and order_ok and outer_disjoint:
        rep["verdict"] = "pass"
    else:
        rep["verdict"] = "fail"
    return rep


def w8_4_branching(runs=None):
    """W8.4 (reported item, analysis-only): temporal branching proxy
    sigma_hat = 1 - n_clusters/n_events on the constant cells, against
    Poisson placements and both surrogate families; cross-block
    offspring excess reported.  Confirmed: sigma above the Poisson
    envelope in a majority of the 1e-3 and 7.5e-4 cells, the
    sub-critical control lowest, and every sigma < 1.  Falsified:
    excess inside the envelope everywhere."""
    rep = {"cells": {}}
    cells = W7_CONST + W8_CONST75 + [W7_CONST_SUB]
    n_above = n_eval = 0
    sub_sigma, trio_sigmas = None, []
    for name in cells:
        try:
            ce = cell_events(name, runs=runs,
                             warmup=T.W7_CONST_STAT_START)
        except FileNotFoundError:
            rep["cells"][name] = {"verdict": "missing"}
            continue
        if ce is None or len(ce["events"]) < T.AV_MIN_SD_EVENTS:
            rep["cells"][name] = {"verdict": "insufficient"}
            continue
        b = sp.branching_sigma(ce["events"], T.W8_BRANCH_LAG)
        pois = sp.branching_null_poisson(
            b["n_events"], ce["T"], T.W8_BRANCH_LAG, T.W8_N_PLACE,
            seed=17, lo=T.W7_CONST_STAT_START)
        g = gnorm_series(name, "plga", runs)
        surr = {kind: sp.branching_surrogates(
            g[2], kind, T.AV_N_SURR, T.AV_THR_MULT, T.AV_MED_WIN,
            T.W7_CONST_STAT_START, T.W8_BRANCH_LAG, seed=s)
            for kind, s in (("ar1", 18), ("phase", 19))}
        ci = sp.sigma_bootstrap_ci(ce["events"], T.W8_BRANCH_LAG,
                                   ce["T"], T.AV_RATE_BOOT_BLOCK,
                                   T.BOOT_N, seed=20,
                                   lo=T.W7_CONST_STAT_START)
        cell = {"sigma": b, "sigma_ci": ci,
                "poisson_mean": float(pois.mean()),
                "poisson_q95": float(np.quantile(pois, T.W8_NULL_Q)),
                "surr_q95": {k: float(np.nanquantile(v, T.AV_SURR_Q))
                             for k, v in surr.items()}}
        cell["above_poisson"] = bool(b["sigma"] > cell["poisson_q95"])
        ebb = {}
        for blk in ("plga", "phi", "attn", "ffn"):
            cb = cell_events(name, block=blk, runs=runs,
                             warmup=T.W7_CONST_STAT_START)
            ebb[blk] = cb["events"] if cb else []
        cell["cross_sigma"] = sp.cross_sigma(ebb, ce["T"],
                                             T.AV_COINC_LAG,
                                             lo=T.W7_CONST_STAT_START)
        n_eval += 1
        n_above += cell["above_poisson"]
        if name == W7_CONST_SUB:
            sub_sigma = b["sigma"]
        else:
            trio_sigmas.append(b["sigma"])
        cell["verdict"] = "descriptive"
        rep["cells"][name] = cell
    all_sub_1 = all(s < 1 for s in trio_sigmas + (
        [sub_sigma] if sub_sigma is not None else []))
    sub_lowest = (sub_sigma is not None and trio_sigmas
                  and sub_sigma < min(trio_sigmas))
    rep["summary"] = {"n_above_poisson": n_above, "n_eval": n_eval,
                      "all_sigma_below_1": bool(all_sub_1),
                      "sub_control_lowest": bool(sub_lowest)}
    if n_eval == 0:
        rep["verdict"] = "insufficient"
    elif n_above == 0:
        rep["verdict"] = "falsified"
    elif (n_above >= (n_eval - 1) // 2 + 1 and all_sub_1
          and sub_lowest):
        rep["verdict"] = "confirmed"
    else:
        rep["verdict"] = "descriptive"
    return rep


def w8_5_fss(runs=None):
    """W8.5 (reported trend; no scaling-law claim at three widths):
    pre-onset event-size cutoffs (q99, S_max) across d_model 128/256/
    512 at matched (eta_max, T_w).  Phase labels first; a phase change
    at a width is a finding, not a broken gate.  The 110M pair joins
    descriptively."""
    rep = {"widths": {}}
    med_q = {}
    for width, cells in W8_WIDTH:
        rows = []
        for name in cells:
            try:
                config, st, fin = load_run(name, runs)
            except FileNotFoundError:
                rows.append({"cell": name, "verdict": "missing"})
                continue
            eA = endpoint_A(st, agg="min")
            onset = onset_step(st)
            ce = cell_events(name, runs=runs)
            if ce is None:
                rows.append({"cell": name, "verdict": "insufficient"})
                continue
            hi = onset if onset is not None else ce["T"]
            pre = av.events_in(ce["events"], config.get("warmup", 0) + 1,
                               hi)
            cut = sp.fss_cutoffs(pre, T.W8_FSS_Q, T.BOOT_N, seed=21)
            rows.append({"cell": name, "collapsing": bool(eA["pass"]),
                         "sigma_end_min": eA["value"], "onset": onset,
                         "pre_onset_cutoffs": cut,
                         "endpoint_B": endpoint_B(fin)})
        rep["widths"][width] = rows
        qs = [r["pre_onset_cutoffs"]["q"] for r in rows
              if r.get("pre_onset_cutoffs")]
        med_q[width] = float(np.median(qs)) if qs else None
    rep["median_q"] = {str(k): v for k, v in med_q.items()}
    ws = [w for w, _ in W8_WIDTH]
    phase_change = any(
        all(not r.get("collapsing", False)
            for r in rep["widths"][w] if "collapsing" in r)
        for w in ws if rep["widths"][w])
    rep["phase_change"] = bool(phase_change)
    if any(med_q[w] is None for w in ws):
        rep["verdict"] = "insufficient"
    else:
        up1 = med_q[ws[0]] < med_q[ws[1]]
        up2 = med_q[ws[1]] < med_q[ws[2]]
        rep["monotone_up"] = {"d128_lt_d256": bool(up1),
                              "d256_lt_d512": bool(up2)}
        if up1 and up2:
            rep["verdict"] = "pass"
        elif not up1 and not up2:
            rep["verdict"] = "kill" if _fss_fully_inverted(rep, ws) \
                else "fail"
        else:
            rep["verdict"] = "fail"
    if phase_change:
        rep["verdict_note"] = ("phase change at a width: the collapse "
                               "boundary moves with system size; "
                               "comparison scoped to collapsing cells")
    try:
        ce = cell_events(W8_110M, runs=runs)
        _, st, fin = load_run(W8_110M, runs)
        rep["scale_110m_descriptive"] = {
            "cell": W8_110M,
            "cutoffs": sp.fss_cutoffs(ce["events"], T.W8_FSS_Q,
                                      T.BOOT_N, seed=22)
            if ce else None,
            "endpoint_B": endpoint_B(fin)}
    except FileNotFoundError:
        rep["scale_110m_descriptive"] = None
    return rep


def _fss_fully_inverted(rep, ws):
    """Kill only on full inversion: every wider cell's q-CI entirely
    below every narrower cell's q-CI, for both adjacent width pairs."""
    def cis(w):
        return [r["pre_onset_cutoffs"]["q_ci"]
                for r in rep["widths"][w] if r.get("pre_onset_cutoffs")]
    for a, b in zip(ws, ws[1:]):
        for ca in cis(a):
            for cb in cis(b):
                if not (cb[1] < ca[0]):
                    return False
    return True


def w8_6_const_fallback(runs=None):
    """W8.6: the wave-7 stationarity (W7.2) and margin (W7.3) clause
    batteries replayed verbatim on the constant 7.5e-4 trio, with both
    phase scopes pre-written and the declared 6e-4 relaunch contingency
    on majority divergence."""
    stat = w7_2_stationarity(runs, cells=W8_CONST75)
    marg = w7_3_margin(runs, cells=W8_CONST75, archived=False)
    rep = {"stationarity": stat, "margin": marg}
    # row-map phase per cell (the min-layer criterion), alongside the
    # operator endpoint already inside the stationarity cells
    rep["rowmap_phase"] = {}
    for n in W8_CONST75:
        try:
            _, st, _ = load_run(n, runs)
            eA = endpoint_A(st, agg="min")
            rep["rowmap_phase"][n] = {"sigma_end_min": eA["value"],
                                      "collapsing": bool(eA["pass"])}
        except FileNotFoundError:
            rep["rowmap_phase"][n] = None
    if stat.get("n_diverged", 0) >= 2:
        rep["verdict"] = "halt"
        rep["contingency"] = ("declared single relaunch at "
                              f"{T.W8_CONST_LR_FALLBACK2}")
    else:
        rep["verdict"] = ("pass" if stat["verdict"] == "pass"
                          and marg["verdict"] == "pass"
                          else "kill" if "kill" in (stat["verdict"],
                                                    marg["verdict"])
                          else "fail")
    rep["scope"] = stat.get("phase_check")
    return rep


def wave8_report(runs=None):
    rep = {}
    for name, fn in (("W8_1_gamma", w8_1_gamma),
                     ("W8_2_anneal", w8_2_anneal),
                     ("W8_3_susceptibility", w8_3_susceptibility),
                     ("W8_4_branching", w8_4_branching),
                     ("W8_5_fss", w8_5_fss),
                     ("W8_6_const_fallback", w8_6_const_fallback)):
        try:
            rep[name] = fn(runs)
        except FileNotFoundError as e:
            rep[name] = {"verdict": "insufficient", "missing": str(e)}
    rep["GATES"] = {
        "gamma_W8_1": rep["W8_1_gamma"].get("verdict"),
        "susceptibility_W8_3": rep["W8_3_susceptibility"].get(
            "verdict"),
        "anneal_W8_2": rep["W8_2_anneal"].get("verdict"),
        "const_fallback_W8_6": rep["W8_6_const_fallback"].get(
            "verdict")}
    return rep
