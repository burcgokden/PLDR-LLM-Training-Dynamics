"""Post-hoc reanalysis report: dependence-aware size-duration
uncertainty, the observed-value xmin scan, the causal-detector
sensitivity pass, the Lanczos residual quality audit, and the measured
separation-to-radius constants (collected from measure_separation.py
outputs).  Nothing here is preregistered; the frozen gate reports are
never rewritten, and every number is emitted into
docs/figures/reanalysis_report.json / residual_qc_report.json with the
manuscript-quoted subset in reanalysis_macros.tex.

Usage: python3 analysis/make_reanalysis.py            (from experiments/)
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avalanche as av  # noqa: E402
import gates  # noqa: E402
import reanalysis as ra  # noqa: E402
import thresholds as T  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "docs", "figures")
os.makedirs(OUT, exist_ok=True)

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "runs")

CONST_PAIRS = (list(zip(gates.W7_CONST, gates.W8_CONST75))
               + list(zip(gates.W8_CONST75, gates.W7_CONST)))

CONST_TAGS = {"w7-const-lr1e-3": "GamAOne",
              "w7-const-lr1e-3-s2": "GamATwo",
              "w7-const-lr1e-3-s3": "GamAThree",
              "w8-const-lr7.5e-4": "GamNOne",
              "w8-const-lr7.5e-4-s2": "GamNTwo",
              "w8-const-lr7.5e-4-s3": "GamNThree"}

TAIL_TAGS = dict(zip(gates.W6_COL, ("TailOne", "TailTwo", "TailThree")))

SEP_CELLS = (("w7-const-lr1e-3", "ckpt_8000.pt", "SepConstA"),
             ("w8-const-lr7.5e-4", "ckpt_1000.pt", "SepConstNWarm"),
             ("w8-const-lr7.5e-4", "ckpt_final.pt", "SepConstN"),
             ("w8-hold6k-lr1e-3", "ckpt_final.pt", "SepHold"))


# ---------------------------------------------------------------------------
# A1: dependence-aware size-duration uncertainty (block bootstrap)


def block_gamma():
    rep = {}
    for name, partner in CONST_PAIRS:
        try:
            ex = gates._dur_events_excluded(name, partner)
        except FileNotFoundError:
            rep[name] = {"verdict": "missing"}
            continue
        if ex is None:
            rep[name] = {"verdict": "insufficient"}
            continue
        sd = av.size_duration(ex["events"], T.BOOT_N, seed=11)
        sb = ra.size_duration_block(ex["events"], T.BOOT_BLOCKLEN,
                                    T.BOOT_N, seed=11)
        rep[name] = {"n_dur2": sd["n"] if sd else 0,
                     "gamma": sd["gamma"] if sd else None,
                     "ci_iid": sd.get("ci") if sd else None,
                     "ci_block": sb.get("ci_block") if sb else None,
                     "blocklen": T.BOOT_BLOCKLEN}
    return rep


# ---------------------------------------------------------------------------
# A2: observed-value xmin scan for the primary tail cells


def observed_tails():
    rep = {}
    for col, sub in zip(gates.W6_COL, gates.W6_SUB):
        ex = gates.excluded_pair(col, sub)
        if ex is None:
            rep[col] = {"verdict": "missing"}
            continue
        sizes = np.array([e["size"] for e in ex["events"]])
        fg = (av.fit_powerlaw(sizes, T.AV_XMIN_GRID)
              if len(sizes) >= T.AV_MIN_EVENTS else None)
        fo = (ra.fit_powerlaw_observed(sizes)
              if len(sizes) >= T.AV_MIN_EVENTS else None)
        rep[col] = {"n_events": int(len(sizes)),
                    "grid": fg, "observed": fo,
                    "alpha_shift": (fo["alpha"] - fg["alpha"]
                                    if fg and fo else None)}
    return rep


# ---------------------------------------------------------------------------
# A3: causal (trailing-median) detector sensitivity


def _causal_events(name, mult, win, warmup):
    g = gates.gnorm_series(name, "plga")
    if g is None:
        return None
    tt, wu, y = g
    ev, dropped = ra.extract_events_causal(
        y, mult, win, wu if warmup is None else warmup)
    return {"T": tt, "events": ev, "n_warmup_dropped": dropped}


def causal_pass():
    rep = {"const_cells": {}, "tail_cells": {}}
    # gamma cells: same warmup and same-seed cross-rate exclusion as
    # the frozen W8.1 pipeline, with both runs re-detected causally
    for name, partner in CONST_PAIRS:
        ca = _causal_events(name, T.AV_DUR_MULT, T.AV_MED_WIN,
                            T.W7_CONST_STAT_START)
        cb = _causal_events(partner, T.AV_DUR_MULT, T.AV_MED_WIN,
                            T.W7_CONST_STAT_START)
        if ca is None or cb is None:
            rep["const_cells"][name] = {"verdict": "missing"}
            continue
        kept, n_dil, _ = av.exclude_matched(ca["events"], cb["events"],
                                            T.AV_MATCH_LAG)
        cen = gates._dur_events_excluded(name, partner)
        sd = av.size_duration(kept)
        rep["const_cells"][name] = {
            "n_causal": len(kept), "n_dropped_dilated": n_dil,
            "n_centered": len(cen["events"]) if cen else None,
            "gamma_causal": sd["gamma"] if sd else None,
            "n_dur2_causal": sd["n"] if sd else 0}
    # tail cells: primary threshold, causal detection on both runs
    for col, sub in zip(gates.W6_COL, gates.W6_SUB):
        ca = _causal_events(col, T.AV_THR_MULT, T.AV_MED_WIN, None)
        cb = _causal_events(sub, T.AV_THR_MULT, T.AV_MED_WIN, None)
        if ca is None or cb is None:
            rep["tail_cells"][col] = {"verdict": "missing"}
            continue
        kept, n_dil, _ = av.exclude_matched(ca["events"], cb["events"],
                                            T.AV_MATCH_LAG)
        sizes = np.array([e["size"] for e in kept])
        f = (av.fit_powerlaw(sizes, T.AV_XMIN_GRID)
             if len(sizes) >= T.AV_MIN_EVENTS else None)
        cen = gates.excluded_pair(col, sub)
        rep["tail_cells"][col] = {
            "n_causal": len(kept),
            "n_centered": len(cen["events"]) if cen else None,
            "alpha_causal": f["alpha"] if f else None,
            "n_tail_causal": f["n_tail"] if f else None}
    return rep


# ---------------------------------------------------------------------------
# A4: Lanczos residual quality audit


def _resid_value_keys(resid_key):
    """Map a logged residual key to its extremal-value keys."""
    if resid_key == "lam_full_resid":
        return ["lam_full", "lam_full_min"]
    if resid_key.startswith("lam_pre_resid_"):
        g = resid_key[len("lam_pre_resid_"):]
        return [f"lam_pre_max_{g}", f"lam_pre_min_{g}"]
    for fam in ("lam_pre_blk", "lam_pre_live"):
        pre = f"{fam}_resid_"
        if resid_key.startswith(pre):
            b = resid_key[len(pre):]
            return [f"{fam}_{b}", f"{fam}_min_{b}"]
    return None


def _scalar(v):
    """Finite float or None (list-valued log fields are skipped)."""
    if isinstance(v, (int, float)) and np.isfinite(v):
        return float(v)
    return None


def residual_qc():
    """Per run, per residual-logging family: n records, 99th-percentile
    and max RELATIVE residual (|resid| / max |extremal eigenvalue|),
    and outlier counts.  Families without a logged residual (the
    raw-metric per-block lam_blk_* series) are named as uncovered."""
    rep = {"runs": {}, "uncovered_families": ["lam_blk_*"]}
    names = sorted(d for d in os.listdir(RUNS)
                   if os.path.isdir(os.path.join(RUNS, d))
                   and d not in ("toy",))
    for name in names:
        try:
            _, steps, _ = gates.load_run(name)
        except (FileNotFoundError, StopIteration):
            continue
        fams = {}
        for r in steps:
            for k, v in r.items():
                if "resid" not in k or not k.startswith("lam"):
                    continue
                vk = _resid_value_keys(k)
                v = _scalar(v)
                if vk is None or v is None:
                    continue
                vals = [_scalar(r.get(c)) for c in vk]
                scale = max((abs(c) for c in vals if c is not None),
                            default=None)
                if scale is None:
                    continue
                fams.setdefault(k, []).append(abs(v) / max(scale, 1e-30))
        out = {}
        for k, vals in fams.items():
            a = np.array(vals)
            out[k] = {"n": int(len(a)),
                      "q99_rel": float(np.quantile(a, 0.99)),
                      "max_rel": float(np.max(a)),
                      "n_gt_0.1": int(np.sum(a > 0.1)),
                      "n_gt_1": int(np.sum(a > 1.0))}
        if out:
            rep["runs"][name] = out
    # global summary
    q99s = [(n, k, d["q99_rel"]) for n, fams in rep["runs"].items()
            for k, d in fams.items()]
    rep["summary"] = {
        "n_run_families": len(q99s),
        "n_q99_gt_0.1": int(sum(q > 0.1 for _, _, q in q99s)),
        "n_q99_gt_1": int(sum(q > 1.0 for _, _, q in q99s)),
        "worst": sorted(q99s, key=lambda t: -t[2])[:12]}
    return rep


# ---------------------------------------------------------------------------
# A5: separation-to-radius constants (from measure_separation.py)


def separation():
    rep = {}
    for name, ckpt, tag in SEP_CELLS:
        p = os.path.join(RUNS, name, "separation.json")
        if not os.path.exists(p):
            rep[tag] = {"verdict": "missing", "run": name, "ckpt": ckpt}
            continue
        with open(p) as f:
            data = json.load(f)
        ck = data.get("ckpts", {}).get(ckpt)
        if ck is None:
            rep[tag] = {"verdict": "missing", "run": name, "ckpt": ckpt}
            continue
        rep[tag] = {"run": name, "ckpt": ckpt, "n_seq": data.get("n_seq"),
                    "layers": ck["layers"],
                    "ratio_min": ck["ratio_min"],
                    "ratio_min_q95": ck["ratio_min_q95"],
                    "sep_gt_2rho": ck["sep_gt_2rho"]}
    return rep


# ---------------------------------------------------------------------------


def write_macros(rep):
    path = os.path.join(OUT, "reanalysis_macros.tex")
    with open(path, "w") as fh:
        fh.write("% generated by analysis/make_reanalysis.py; "
                 "do not edit\n")
        for cell, tag in CONST_TAGS.items():
            c = rep["block_gamma"].get(cell, {})
            cb = c.get("ci_block")
            fh.write("")
            _m(fh, f"reBlk{tag}Lo", cb[0] if cb else None)
            _m(fh, f"reBlk{tag}Hi", cb[1] if cb else None)
            cc = rep["causal"]["const_cells"].get(cell, {})
            _m(fh, f"reCausalNev{tag}", cc.get("n_causal"), "{:d}")
            _m(fh, f"reCenteredNev{tag}", cc.get("n_centered"), "{:d}")
            _m(fh, f"reCausalGam{tag}", cc.get("gamma_causal"))
        for cell, tag in TAIL_TAGS.items():
            t = rep["observed_tails"].get(cell, {})
            fo, fg = t.get("observed"), t.get("grid")
            _m(fh, f"reObsAlpha{tag}", fo["alpha"] if fo else None)
            _m(fh, f"reGridAlpha{tag}", fg["alpha"] if fg else None)
            _m(fh, f"reObsNtail{tag}",
               fo["n_tail"] if fo else None, "{:d}")
            ct = rep["causal"]["tail_cells"].get(cell, {})
            _m(fh, f"reCausalAlpha{tag}", ct.get("alpha_causal"))
            _m(fh, f"reCausalNevTail{tag}", ct.get("n_causal"), "{:d}")
        s = rep["residual_summary"]
        _m(fh, "reResidFams", s["n_run_families"], "{:d}")
        _m(fh, "reResidGtTenth", s["n_q99_gt_0.1"], "{:d}")
        _m(fh, "reResidGtOne", s["n_q99_gt_1"], "{:d}")
        for run, tag in (("w6-base-lr1e-3", "Base"),
                         ("w7-const-lr1e-3", "ConstA"),
                         ("w8-const-lr7.5e-4", "ConstN")):
            fams = rep["residual_runs_quoted"].get(run, {})
            _m(fh, f"reResid{tag}Full",
               fams.get("lam_full_resid", {}).get("q99_rel"))
            blk = [d["q99_rel"] for k, d in fams.items()
                   if k.startswith("lam_pre_blk_resid_")]
            live = [d["q99_rel"] for k, d in fams.items()
                    if k.startswith("lam_pre_live_resid_")]
            _m(fh, f"reResid{tag}BlkMax", max(blk) if blk else None)
            _m(fh, f"reResid{tag}LiveMax", max(live) if live else None)
        for _, _, tag in SEP_CELLS:
            c = rep["separation"].get(tag, {})
            _m(fh, f"{tag}Ratio", c.get("ratio_min"))
            _m(fh, f"{tag}RatioQ", c.get("ratio_min_q95"))
            _m(fh, f"{tag}TwoRho",
               ("yes" if c.get("sep_gt_2rho") else "no")
               if "sep_gt_2rho" in c else None)
    print("  reanalysis_macros.tex")


def _m(fh, name, val, fmt="{:.3g}"):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        fh.write(f"\\gdef\\{name}{{--}}\n")
    elif isinstance(val, str):
        fh.write(f"\\gdef\\{name}{{{val}}}\n")
    else:
        fh.write(f"\\gdef\\{name}{{{fmt.format(val)}}}\n")


def main():
    rep = {"block_gamma": block_gamma(),
           "observed_tails": observed_tails(),
           "causal": causal_pass(),
           "separation": separation()}
    qc = residual_qc()
    rep["residual_summary"] = qc["summary"]
    rep["residual_runs_quoted"] = {
        n: qc["runs"][n] for n in
        ("w6-base-lr1e-3", "w6-base-lr3e-4", "w7-const-lr1e-3",
         "w8-const-lr7.5e-4") if n in qc["runs"]}
    with open(os.path.join(OUT, "reanalysis_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("  reanalysis_report.json")
    with open(os.path.join(OUT, "residual_qc_report.json"), "w") as f:
        json.dump(qc, f, indent=1, default=float)
    print("  residual_qc_report.json")
    write_macros(rep)


if __name__ == "__main__":
    main()
