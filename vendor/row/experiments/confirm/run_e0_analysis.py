#!/usr/bin/env python3
"""E0 analysis driver: ordering-enforced measurement only.

Order of operations (protocol discipline):
  1. merge both registered E0 cell chains;
  2. WRITE THE EXCLUSION LIST (shared-step data events) before any
     fit (confirm/out/e0/exclusion.json);
  3. window battery, (SC) statistics, band containment, both cells;
  4. crossings + loading-crossing-reset signature (boundary), matched
     placements at the same global steps in the control;
  5. gate-closed factor fits (kappa, d0, mu_b, rates, delta), gate
     regression (delta_J0), curvature-law slope (closed/open);
  6. per-layer (I-J) report with cross-layer phase synchrony;
  7. estimator output for exactly three inferential families.

Checkpoint batteries (fidelity, separation, h_max, fiber constants)
and the E6-pilot crossed-model fit run separately (GPU-bound /
pilot-data-bound). Only assemble_e0.py computes the total E0 outcome.

Usage: run_e0_analysis.py [--boundary c-e0-boundary]
                          [--control c-e0-subcrit] [--windows 60]
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "analysis"))
sys.path.insert(0, HERE)

import collect          # noqa: E402
import e0_measure as E  # noqa: E402

import source_edge_certificate as SEC  # noqa: E402
OUT = os.path.join(HERE, "out", "e0")


def jdump(obj, path):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=default)


def analyze_cell(name, partner, n_windows, results):
    S = E.build_series(name + "-merged")
    wins = E.window_stats(S, n_windows=n_windows)
    locked = results["exclusion"]["intervals"][name]
    E.apply_exclusion(wins, [tuple(x) for x in locked])
    keep = [w for w in wins if not w["excluded"]]
    gc = E.gate_closed_mask(S, name + "-merged", S["tl"], pad_post=10)

    kf = E.kappa_fit(S, wins, gc)
    bf = E.base_fit(S, wins, kf["kappa"], gc)
    rates = E.factor_rates(S, wins, gc)
    delta = E.delta_feed_fit(S, wins, rates, gc)
    gr = E.gate_regression(S, name + "-merged")
    dj0 = gr["delta_J0"] if gr["delta_J0"] > 0 else None
    ls = E.loading_slope(S, wins, kf["kappa"], bf["d0"], bf["mu_b"],
                         rates, delta, name + "-merged",
                         delta_J0=dj0)
    source_edge = SEC.finite_horizon_certificate(
        wins, ls, 1.0 - E.BETA1, E.theory.classify_schedule_certificate)
    edge_schedule = E.iedge_schedule_component_fit(S, wins)
    event_entry = E.event_entry_stats(S, name + "-merged")
    cr = E.band_crossings(S, wins)
    sig = E.signature_stats(S, [c for c in cr if not c["excluded"]])

    sc = dict(
        n_windows=len(wins), n_kept=len(keep),
        sc_A=int(sum(w["sc_A"] for w in keep)),
        sc_e=int(sum(w["sc_e"] for w in keep)),
        sc_shift=int(sum(w["sc_shift"] for w in keep)),
        sc_drift=int(sum(w["sc_drift"] for w in keep)),
        sc_preconditioner=int(sum(w["sc_preconditioner"] for w in keep)),
        cwin=int(sum(w["cwin"] for w in keep)),
        in_band=int(sum(w["in_band"] for w in keep)),
        in_box=int(sum(w["in_box"] for w in keep)),
        all_regular=int(sum(
            w["sc_A"] and w["sc_e"] and w["sc_shift"]
            and w["sc_preconditioner"] and w["sc_drift"] and w["cwin"]
            and w["in_band"] and w["in_box"] for w in keep)),
        mean_frac_in_band=float(np.mean([w["frac_in_band"]
                                         for w in keep])),
    )
    primary_ok = np.isfinite(S["x_phys"])
    if len(S["gn_residual"]) and primary_ok.any():
        x_on_residual_axis = np.interp(
            S["tg"], S["tx"][primary_ok], S["x_phys"][primary_ok])
        ratios = (np.abs(S["gn_residual"])
                  / np.maximum(np.abs(x_on_residual_axis), 1e-300))
        ok = np.isfinite(ratios)
        residual_rel = (
            float(np.max(ratios[ok])) if ok.any() else None)
    else:
        residual_rel = None
    ledger_values = np.concatenate([
        S[f"decay_ledger_residual_{name}"]
        for name in ("plga", "phi", "attn", "ffn")])
    ledger_values = ledger_values[np.isfinite(ledger_values)]
    ledger_max = (float(np.max(ledger_values))
                  if len(ledger_values) else None)
    return dict(series_T=S["T"], windows=wins, window_summary=sc,
                stress_metadata=S["stress_metadata"],
                averaging_preconditioner={
                    "operator": "inverse", "window": "trailing"}
                if sc["sc_preconditioner"] == sc["n_kept"] > 0
                else None,
                gn_full_residual_relative_max=residual_rel,
                kappa=kf, base=bf, rates=rates, delta_feed=delta,
                decay_ledger_residual_max=ledger_max,
                primary_series={
                    "t": S["tx"], "x": S["x_phys"],
                    "factor_t": S["tl"],
                    "a": np.nanmean(S["a_layers"], axis=1),
                    "J": np.nanmin(S["J_layers"], axis=1),
                },
                gate_regression=gr, loading=ls,
                source_edge_certificate=source_edge,
                edge_schedule_component=edge_schedule,
                event_entry=event_entry,
                crossings=cr, signatures=sig,
                gate_closed_frac=float(gc.mean())), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", default="c-e0-boundary")
    ap.add_argument("--control", default="c-e0-subcrit")
    ap.add_argument("--windows", type=int, default=60)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    for nm in (args.boundary, args.control):
        collect.merge_chain(nm)

    results = {}
    # exclusion list FIRST, from both cells jointly, written before
    # any fit
    locked_a = E.data_event_windows(args.boundary + "-merged",
                                    args.control + "-merged")
    locked_b = E.data_event_windows(args.control + "-merged",
                                    args.boundary + "-merged")
    sens = {str(r): len(E.data_event_windows(
        args.boundary + "-merged", args.control + "-merged",
        ratio=r)) for r in (1.25, 1.5)}
    results["exclusion"] = dict(
        rule="both cells' per-step loss > %.2fx own centered "
             "101-step running median, dilated by lag 2"
             % E.DATA_SPIKE_RATIO,
        intervals={args.boundary: locked_a, args.control: locked_b},
        sensitivity_counts=sens)
    jdump(results["exclusion"], os.path.join(OUT, "exclusion.json"))

    res_b, S_b = analyze_cell(args.boundary, args.control,
                              args.windows, results)
    res_c, S_c = analyze_cell(args.control, args.boundary,
                              args.windows, results)

    # matched placements: the control cell scored at the boundary
    # cell's crossing steps (shared data order)
    steps_cr = [c["t"] for c in res_b["crossings"]
                if not c["excluded"]]
    plc = E.placement_signature(S_c, steps_cr)
    res_b["control_placements"] = plc

    # (I-J): per-layer jet report + cross-layer phase synchrony
    def ij_report(S):
        J = S["J_layers"]
        n_lay = J.shape[1]
        corr = np.corrcoef(np.diff(J, axis=0).T) if len(J) > 3 \
            else np.full((n_lay, n_lay), np.nan)
        return dict(n_layers=int(n_lay),
                    layer_final=[float(v) for v in J[-1]],
                    layer_min=[float(v) for v in np.nanmin(J, axis=0)],
                    sync_corr=[[float(c) for c in row]
                               for row in corr],
                    agg_axes="median over sampled rows within layer "
                             "(eq. jetident, J_norm = 1); layer axis "
                             "consumed by min (layer-flat) / max "
                             "(all-flat) downstream")
    res_b["ij"] = ij_report(S_b)
    res_c["ij"] = ij_report(S_c)

    # ---- exactly three inferential-family estimators -----------------
    n_sig = sum(s["signature"] for s in res_b["signatures"])
    n_scored = len(res_b["signatures"])
    frac_sig = n_sig / n_scored if n_scored else np.nan
    n_plc = sum(s["signature"] for s in plc)
    n_plc_scored = len(plc)
    frac_plc = n_plc / n_plc_scored if n_plc_scored else np.nan

    slope = res_b["loading"]["closed"]["slope"]
    r_load = res_b["loading"]["closed"]["r"]
    ci = res_b["loading"]["closed"]["ci"]
    ws = res_b["window_summary"]
    kept = ws["n_kept"]
    regular = ws["all_regular"]
    regular_frac = regular / kept if kept else np.nan
    regular_hold = bool(kept and regular >= 0.8 * kept)
    n_crossings = len([c for c in res_b["crossings"]
                       if not c["excluded"]])

    # Family p-values: loading bootstrap, regular-window binomial,
    # and upper-crossing signature binomial against matched control
    # placements. The assembler alone performs Holm and decides E0.
    from scipy import stats as st
    # loading: one-sided bootstrap p for slope > 0 (re-run the window
    # bootstrap and count nonpositive resamples)
    p_load = None
    wrows = res_b["loading"]["closed"].get("windows") or []
    if len(wrows) >= 4:
        arr = np.array([r[:3] for r in wrows])
        meas, pred = arr[:, 1], arr[:, 2]
        ok = np.isfinite(meas) & np.isfinite(pred)
        meas, pred = meas[ok], pred[ok]
        rng = np.random.default_rng(1)
        neg = 0
        nres = 4000
        for _ in range(nres):
            ii = rng.integers(0, len(meas), len(meas))
            p_, m_ = pred[ii], meas[ii]
            if p_ @ m_ / max(p_ @ p_, 1e-300) <= 0:
                neg += 1
        p_load = float((neg + 1) / (nres + 1))
    p_regular = (float(st.binomtest(
        regular, kept, 0.5, alternative="greater").pvalue)
        if kept else None)
    if n_scored:
        p0 = frac_plc if np.isfinite(frac_plc) and frac_plc > 0 \
            else 0.15
        p_sig = float(st.binomtest(n_sig, n_scored, p0,
                                   alternative="greater").pvalue)
    else:
        p_sig = None

    direction_load = (
        np.isfinite(slope) and np.isfinite(r_load)
        and r_load >= 0.2 and 0.5 <= slope <= 2.0)
    direction_sig = (
        np.isfinite(frac_sig) and frac_sig >= 0.5
        and (not np.isfinite(frac_plc) or frac_sig > frac_plc))

    estimator_summary = dict(
        authorization="none; assemble_e0.py is the sole decision function",
        n_upper_crossings=n_crossings,
        crossings_minimum_met=bool(n_crossings >= 40),
        slope=slope, slope_ci=ci, r_loading=r_load,
        regular_fraction=regular_frac, window_summary=ws,
        signature_frac=frac_sig, placement_frac=frac_plc,
        n_signature=[n_sig, n_scored],
        n_placement=[n_plc, n_plc_scored],
        family_p_values={
            "loading_law": p_load,
            "averaged_stability_regularity": p_regular,
            "upper_crossing_signature": p_sig,
        },
        family_directional_pass={
            "loading_law": bool(direction_load),
            "averaged_stability_regularity": bool(regular_hold),
            "upper_crossing_signature": bool(direction_sig),
        },
        family_count=3,
        stress_metadata=res_b["stress_metadata"],
    )

    results.update(boundary=res_b, control=res_c,
                   estimator_summary=estimator_summary)
    jdump(results, os.path.join(OUT, "e0_results.json"))
    print(json.dumps(estimator_summary, indent=1, default=str))


if __name__ == "__main__":
    main()
