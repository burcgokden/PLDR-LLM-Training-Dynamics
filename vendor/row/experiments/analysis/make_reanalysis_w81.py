"""Recalibrated W8.1 envelope comparison (reanalysis; NOT
preregistered).  The frozen W8.1 gate compares a partner-excluded
observed size-duration statistic against surrogate envelopes computed
on unexcluded single-series surrogates, so the observed statistic and
its null do not pass through the same pipeline.  This script computes,
per constant-rate cell and per surrogate family:

  observed   the frozen estimand (partner-excluded events,
             size-duration fit with the tie-aware Spearman);
  joint      the primary recalibrated null: paired surrogates through
             the identical extraction, exclusion, and fit
             (reanalysis.joint_surrogate_stats);
  fixedmask  sensitivity bracket (a): surrogate target against the
             observed partner mask;
  block      sensitivity bracket (b): aligned pair moving-block
             resample with shared block boundaries;
  single     the frozen-style single-series, no-exclusion envelope
             (avalanche.surrogate_stats, byte-identical code path),
             recomputed here only for the comparison table.

The frozen w8_gate_report.json and its macros are untouched; every
number this script emits is labeled a reanalysis in the manuscript and
is cited only at exploratory strength.  Outputs:
docs/figures/w81_reanalysis.json and w81_reanalysis_macros.tex.

Usage: python3 analysis/make_reanalysis_w81.py     (from experiments/)
"""

import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import avalanche as av  # noqa: E402
import gates  # noqa: E402
import reanalysis as ra  # noqa: E402
import thresholds as T  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "..", "docs", "figures")
os.makedirs(OUT, exist_ok=True)

CONST_PAIRS = (list(zip(gates.W7_CONST, gates.W8_CONST75))
               + list(zip(gates.W8_CONST75, gates.W7_CONST)))

CONST_TAGS = {"w7-const-lr1e-3": "GamAOne",
              "w7-const-lr1e-3-s2": "GamATwo",
              "w7-const-lr1e-3-s3": "GamAThree",
              "w8-const-lr7.5e-4": "GamNOne",
              "w8-const-lr7.5e-4-s2": "GamNTwo",
              "w8-const-lr7.5e-4-s3": "GamNThree"}

# series-level moving-block length for bracket (b): about five
# running-median windows; preserves cross-series structure at the
# scale of the detector while resampling the long-range order
SERIES_BLOCKLEN = 500
FAMILIES = ("ar1", "phase")


def _seed(name, tag):
    """Stable per-(cell, variant) seed (CRC32; process-independent)."""
    return 20000 + zlib.crc32(f"{name}|{tag}".encode()) % 10000


def _env(arr, q=None):
    a = np.asarray(arr, float)
    if not np.any(np.isfinite(a)):
        return None
    return float(np.nanquantile(a, T.AV_SURR_Q if q is None else q))


def _summ(stats):
    """Envelope summary of one surrogate family's per-draw arrays."""
    g = stats["gamma"]
    return {"n_surr": int(len(g)),
            "n_finite_gamma": int(np.sum(np.isfinite(g))),
            "gamma_q95": _env(g),
            "gamma_med": _env(g, 0.5),
            "spearman_q95": _env(stats["spearman"]),
            "kept_med": _env(stats["n_kept"], 0.5),
            "dropped_med": _env(stats["n_dropped"], 0.5)}


def cell_reanalysis(name, partner):
    ex = gates._dur_events_excluded(name, partner)
    if ex is None:
        return {"verdict": "missing"}
    sd = av.size_duration(ex["events"], T.BOOT_N, seed=11)
    gt = gates.gnorm_series(name, "plga")
    gp = gates.gnorm_series(partner, "plga")
    y_t, y_p = gt[2], gp[2]
    cb = gates.cell_events(partner, mult=T.AV_DUR_MULT,
                           warmup=T.W7_CONST_STAT_START)
    args = (T.AV_DUR_MULT, T.AV_MED_WIN, T.W7_CONST_STAT_START,
            T.AV_MATCH_LAG)
    rep = {"observed": {
        "n_raw": ex["n_raw"],
        "n_dropped_dilated": ex["n_dropped_dilated"],
        "n_kept": len(ex["events"]),
        "n_dur2": sd["n"] if sd else 0,
        "n_dur_distinct": sd["n_dur_distinct"] if sd else 0,
        "gamma": sd["gamma"] if sd else None,
        "ci": sd.get("ci") if sd else None,
        "spearman_tieaware": sd["spearman"] if sd else None},
        "joint": {}, "fixedmask": {}, "single": {}}
    for kind in FAMILIES:
        rep["joint"][kind] = _summ(ra.joint_surrogate_stats(
            y_t, y_p, kind, T.AV_N_SURR, *args,
            seed=_seed(name, "joint-" + kind)))
        rep["fixedmask"][kind] = _summ(ra.fixedmask_surrogate_stats(
            y_t, cb["events"], kind, T.AV_N_SURR, *args,
            seed=_seed(name, "fixed-" + kind)))
        # frozen-style single-series envelope (same seeds as the gate)
        surr = av.surrogate_stats(
            y_t, kind, T.AV_N_SURR, T.AV_DUR_MULT, T.AV_MED_WIN,
            T.W7_CONST_STAT_START,
            seed={"ar1": 12, "phase": 13}[kind],
            dur_mult=T.AV_DUR_MULT)
        rep["single"][kind] = {"gamma_q95": _env(surr["gamma"])}
    rep["block"] = _summ(ra.joint_block_surrogate_stats(
        y_t, y_p, SERIES_BLOCKLEN, T.AV_N_SURR, *args,
        seed=_seed(name, "block")))
    rep["block"]["blocklen"] = SERIES_BLOCKLEN
    # exploratory comparison flags (no gate semantics)
    if sd:
        env_joint = [rep["joint"][k]["gamma_q95"] for k in FAMILIES]
        rep["gamma_above_joint_env"] = bool(
            all(e is None or sd["gamma"] > e for e in env_joint))
        rep["gamma_above_block_env"] = bool(
            rep["block"]["gamma_q95"] is None
            or sd["gamma"] > rep["block"]["gamma_q95"])
    return rep


def write_macros(rep):
    path = os.path.join(OUT, "w81_reanalysis_macros.tex")
    with open(path, "w") as fh:
        fh.write("% generated by analysis/make_reanalysis_w81.py; "
                 "do not edit\n")
        for cell, tag in CONST_TAGS.items():
            c = rep.get(cell)
            if not c or "observed" not in c:
                continue
            o = c["observed"]
            _m(fh, f"rwObsGam{tag}", o["gamma"])
            _m(fh, f"rwObsRho{tag}", o["spearman_tieaware"])
            _m(fh, f"rwObsNd{tag}", o["n_dur2"], "{:d}")
            _m(fh, f"rwObsNdd{tag}", o["n_dur_distinct"], "{:d}")
            _m(fh, f"rwObsNraw{tag}", o["n_raw"], "{:d}")
            _m(fh, f"rwObsNdrop{tag}", o["n_dropped_dilated"], "{:d}")
            for kind, kk in (("ar1", "Ar"), ("phase", "Ph")):
                _m(fh, f"rwJointEnv{kk}{tag}",
                   c["joint"][kind]["gamma_q95"])
                _m(fh, f"rwFixEnv{kk}{tag}",
                   c["fixedmask"][kind]["gamma_q95"])
                _m(fh, f"rwSingleEnv{kk}{tag}",
                   c["single"][kind]["gamma_q95"])
            _m(fh, f"rwBlockEnv{tag}", c["block"]["gamma_q95"])
            _m(fh, f"rwAboveJoint{tag}",
               "yes" if c.get("gamma_above_joint_env") else "no")
    print("  w81_reanalysis_macros.tex")


def _m(fh, name, val, fmt="{:.3g}"):
    if val is None or (isinstance(val, float)
                       and not np.isfinite(val)):
        fh.write(f"\\gdef\\{name}{{--}}\n")
    elif isinstance(val, str):
        fh.write(f"\\gdef\\{name}{{{val}}}\n")
    else:
        fh.write(f"\\gdef\\{name}{{{fmt.format(val)}}}\n")


def main():
    rep = {}
    for name, partner in CONST_PAIRS:
        try:
            rep[name] = cell_reanalysis(name, partner)
        except FileNotFoundError as e:
            rep[name] = {"verdict": "missing", "missing": str(e)}
        print(f"  {name}: done")
    with open(os.path.join(OUT, "w81_reanalysis.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("  w81_reanalysis.json")
    write_macros(rep)


if __name__ == "__main__":
    main()
