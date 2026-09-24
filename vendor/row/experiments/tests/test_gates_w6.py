"""Synthetic-record branch tests of the executable preregistration
(analysis/gates.py): every wave-6 verdict function is driven through
its pass, kill, fail/insufficient, and (where declared) halt branches
on crafted logs, and the corrected wave-5 re-audit is pinned to the
known manual-application outcomes on the archived runs."""

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "analysis"))

import gates  # noqa: E402
import thresholds as T  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic run construction


def write_run(runs, name, records, final_mgen=None, config=None):
    d = os.path.join(runs, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        f.write(json.dumps({"event": "config", **(config or {})}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
        if final_mgen is not None:
            f.write(json.dumps({"event": "final",
                                "m_gen": {"GLM": final_mgen,
                                          "A": 0.1, "ALM": 0.1,
                                          "AP": 0.1}}) + "\n")


def synth(steps=2000, sigma=None, lam=None, pre_live=None, proj=None,
          dtheta=1.0, gnorm=1.0, lr=1e-3, fd=None, fdf=None):
    """Build per-step records: sigma(step) -> per-layer list at cadence
    50; lam/pre_live at cadence 100; proj(step) -> signed dyn
    projection each step."""
    recs = []
    for s in range(1, steps + 1):
        r = {"step": s, "loss": 5.0, "lr": lr, "dtheta_plga": dtheta,
             "gnorm_plga": gnorm}
        if proj is not None:
            r["mode_proj_plga_dyn"] = proj(s)
        if s % 50 == 0 and sigma is not None:
            r["rowmap_sigma_med"] = sigma(s)
        if s % 100 == 0:
            if lam is not None:
                r["lam_plga"] = lam(s)
            if pre_live is not None:
                r["lam_pre_live_plga"] = pre_live(s)
        if s % 250 == 0 and fd is not None:
            r["fdcurv_collapse"] = [fd(s)]
            if fdf is not None:
                r["fdcurv_collapse_fullobj"] = [fdf(s)]
        recs.append(r)
    return recs


def flat_sigma(onset=1000, deep=0.01):
    def f(s):
        return [deep if s >= onset else 1.5, 1.2, 1.3]
    return f


def all_flat_sigma(onset=1000, deep=0.01):
    def f(s):
        return [deep, deep, deep] if s >= onset else [1.5, 1.2, 1.3]
    return f


def no_flat_sigma():
    return lambda s: [1.5, 1.2, 1.3]


# ---------------------------------------------------------------------------
# primitives


def test_bootstrap_short_series_is_interval_free():
    assert gates.moving_block_bootstrap([1.0] * 5) is None
    ci = gates.moving_block_bootstrap(list(np.linspace(0, 1, 40)))
    assert ci is not None and ci[0] <= 0.5 <= ci[1]


def test_endpoints_and_sweep(tmp_path):
    runs = str(tmp_path)
    write_run(runs, "r", synth(sigma=all_flat_sigma(500, 0.01)),
              final_mgen=0.001)
    _, st, fin = gates.load_run("r", runs)
    assert gates.endpoint_A(st)["pass"]
    assert gates.endpoint_B(fin)["pass"]
    sw = gates.sensitivity_sweep(st)
    assert sw["A_thr0.05_max"]["pass"] and sw["A_thr0.2_min"]["pass"]
    write_run(runs, "r2", synth(sigma=no_flat_sigma()), final_mgen=0.05)
    _, st2, fin2 = gates.load_run("r2", runs)
    assert not gates.endpoint_A(st2)["pass"]
    assert not gates.endpoint_B(fin2)["pass"]


def test_onset_layer_fixed_coordinate(tmp_path):
    runs = str(tmp_path)
    write_run(runs, "r", synth(sigma=flat_sigma(600)))
    _, st, _ = gates.load_run("r", runs)
    assert gates.onset_layer(st) == 0
    assert gates.onset_step(st) == 600


def test_dyn_windows_oscillation_vs_drift(tmp_path):
    runs = str(tmp_path)
    osc = lambda s: 0.5 * ((-1) ** s)  # noqa: E731
    write_run(runs, "o", synth(sigma=no_flat_sigma(), proj=osc))
    _, st, _ = gates.load_run("o", runs)
    ws, dropped = gates.dyn_windows(st, layer=0)
    assert ws and dropped == 0
    assert all(w["flip"] > 0.9 for w in ws)
    assert all(w["captured"] == pytest.approx(0.25) for w in ws)
    drift = lambda s: 0.3  # noqa: E731
    write_run(runs, "d", synth(sigma=no_flat_sigma(), proj=drift))
    _, st2, _ = gates.load_run("d", runs)
    ws2, _ = gates.dyn_windows(st2, layer=0)
    assert all(w["flip"] == 0.0 for w in ws2)
    assert all(w["V"] == pytest.approx(0.0) for w in ws2)


def test_force_law_verdict_branches():
    rng = np.random.default_rng(0)

    def mk(n, rho_sign, spread=1.0):
        ws = []
        for _ in range(n):
            u = rng.uniform(0.5, 1.5)
            v = rng.uniform(0.1, 2.0) * spread
            du = rho_sign * u * v + 0.01 * rng.normal()
            ws.append({"u": u, "du": du, "V": v, "P": v})
        return ws

    strong = mk(60, -1.0)
    out = gates.force_law_verdict(strong, {"c": strong})
    assert out["verdict"] == "pass", out
    null = mk(60, 0.0)
    # pure-noise du: correlations near zero -> kill
    for w in null:
        w["du"] = 0.001 * rng.normal()
    out2 = gates.force_law_verdict(null, {"c": null})
    assert out2["verdict"] == "kill"
    out3 = gates.force_law_verdict(strong[:10], {"c": strong[:10]})
    assert out3["verdict"] == "insufficient"
    # strong correlations but below the pooled window floor -> insufficient
    out4 = gates.force_law_verdict(strong[:20], {"c": strong[:20]})
    assert out4["verdict"] in ("insufficient", "kill")


def test_tertile_monotone_empty_cell_is_none():
    ws = [{"u": 1.0, "du": -0.1, "V": v, "P": v}
          for v in np.linspace(0.1, 1.0, 12)]
    mono, _ = gates.tertile_monotone(ws)  # all in one u tertile bin
    assert mono is None


# ---------------------------------------------------------------------------
# wave-6 verdicts, branch by branch


def _ref_cells(runs, passing=3, a01=1):
    for i, n in enumerate(gates.W6_REF):
        good = i < passing
        write_run(runs, n,
                  synth(sigma=all_flat_sigma(
                      500, 0.01 if i < a01 else 0.15) if good
                      else no_flat_sigma()),
                  final_mgen=0.001 if good else 0.05)
    write_run(runs, "w6-ref-lr2e-3-24k",
              synth(sigma=all_flat_sigma(500, 0.05)), final_mgen=0.002)


def test_w1_pass_and_kill(tmp_path):
    runs = str(tmp_path)
    _ref_cells(runs, passing=3, a01=1)
    out = gates.w1_reference(runs)
    assert out["verdict"] == "pass" and out["anneal_survival_B"]
    runs2 = str(tmp_path / "k")
    _ref_cells(runs2, passing=1)
    assert gates.w1_reference(runs2)["verdict"] == "kill"


def _contrast_cells(runs, ratio=10.0, pre_ratio=10.0):
    # 6000 steps so the last-quarter cadence-100 series holds 15
    # records, above the declared bootstrap block length
    for n in gates.W6_SUB:
        write_run(runs, n, synth(steps=6000, sigma=no_flat_sigma(),
                                 lam=lambda s: 10000.0 * ratio / 10,
                                 pre_live=lambda s: 1e7 * pre_ratio / 10))
    for n in gates.W6_COL:
        write_run(runs, n, synth(steps=6000, sigma=flat_sigma(800),
                                 lam=lambda s: 1000.0,
                                 pre_live=lambda s: 1e6))


def test_w2_contrast_pass_kill(tmp_path):
    runs = str(tmp_path)
    _contrast_cells(runs, ratio=10, pre_ratio=10)
    out = gates.w2_contrast(runs)
    assert out["raw"]["verdict"] == "pass"
    assert out["pre_live"]["verdict"] == "pass"
    runs2 = str(tmp_path / "k")
    _contrast_cells(runs2, ratio=10, pre_ratio=1.0)  # no pre contrast
    out2 = gates.w2_contrast(runs2)
    assert out2["pre_live"]["verdict"] == "kill"


def test_w3_confirmed_kill_insufficient(tmp_path):
    runs = str(tmp_path)
    osc = lambda s: 0.5 * ((-1) ** s)  # noqa: E731
    for n in gates.W6_COL:
        write_run(runs, n, synth(sigma=flat_sigma(1000), proj=osc))
    assert gates.w3_carrier_existence(runs)["verdict"] == "confirmed"
    runs2 = str(tmp_path / "k")
    drift = lambda s: 0.3  # noqa: E731
    for n in gates.W6_COL:
        write_run(runs2, n, synth(sigma=flat_sigma(1000), proj=drift))
    assert gates.w3_carrier_existence(runs2)["verdict"] == "kill"
    runs3 = str(tmp_path / "i")
    for n in gates.W6_COL:
        write_run(runs3, n, synth(sigma=no_flat_sigma(), proj=osc))
    assert gates.w3_carrier_existence(runs3)["verdict"] == "insufficient"


def test_w5_causal_pass_and_kill(tmp_path):
    runs = str(tmp_path)
    for damp, base in (("w6-damp-plga-lr1e-3", "w6-base-lr1e-3"),
                       ("w6-damp-plga-lr1e-3-s2", "w6-base-lr1e-3-s2")):
        write_run(runs, base, synth(sigma=flat_sigma(800, 0.01)))
        write_run(runs, damp, synth(sigma=flat_sigma(800, 0.2)))
    out = gates.w5_causal(runs)
    assert out["verdict"] == "pass"
    runs2 = str(tmp_path / "k")
    for damp, base in (("w6-damp-plga-lr1e-3", "w6-base-lr1e-3"),
                       ("w6-damp-plga-lr1e-3-s2", "w6-base-lr1e-3-s2")):
        write_run(runs2, base, synth(sigma=flat_sigma(800, 0.01)))
        write_run(runs2, damp, synth(sigma=flat_sigma(800, 0.01)))
    assert gates.w5_causal(runs2)["verdict"] == "kill"


def test_w6_noise_halt_pass_kill(tmp_path):
    runs = str(tmp_path)
    write_run(runs, "w6-base-lr1e-3", synth(sigma=flat_sigma(800, 0.04)),
              final_mgen=0.001)
    write_run(runs, "w6-b8acc4-lr1e-3", synth(sigma=no_flat_sigma()),
              final_mgen=0.05)  # identity broken
    assert gates.w6_noise(runs)["verdict"] == "halt"
    write_run(runs, "w6-b8acc4-lr1e-3", synth(sigma=flat_sigma(800, 0.04)),
              final_mgen=0.001)
    write_run(runs, "w6-b8tok-lr1e-3", synth(sigma=flat_sigma(800, 0.005)))
    write_run(runs, "w6-b64tok-lr1e-3", synth(sigma=flat_sigma(800, 0.5)))
    assert gates.w6_noise(runs)["verdict"] == "pass"
    write_run(runs, "w6-b8tok-lr1e-3", synth(sigma=flat_sigma(800, 0.5)))
    write_run(runs, "w6-b64tok-lr1e-3", synth(sigma=flat_sigma(800, 0.005)))
    assert gates.w6_noise(runs)["verdict"] == "kill"


def test_w7_carrier_pass_and_kill(tmp_path):
    runs = str(tmp_path)
    for base in ("w6-base-lr1e-3", "w6-ref-lr2e-3"):
        write_run(runs, base, synth(sigma=flat_sigma(600, 0.01)))
    for lr in ("lr1e-3", "lr2e-3"):
        write_run(runs, f"w6-frzattn-{lr}", synth(sigma=no_flat_sigma()))
        write_run(runs, f"w6-frzffn-{lr}",
                  synth(sigma=flat_sigma(700, 0.015)))
    assert gates.w7_carrier_freeze(runs)["verdict"] == "pass"
    runs2 = str(tmp_path / "k")
    for base in ("w6-base-lr1e-3", "w6-ref-lr2e-3"):
        write_run(runs2, base, synth(sigma=flat_sigma(600, 0.01)))
    for lr in ("lr1e-3", "lr2e-3"):
        for blk in ("attn", "ffn"):
            write_run(runs2, f"w6-frz{blk}-{lr}",
                      synth(sigma=flat_sigma(650, 0.012)))
    assert gates.w7_carrier_freeze(runs2)["verdict"] == "kill"


def test_w8_parity_branches(tmp_path):
    runs = str(tmp_path)
    write_run(runs, "w6-dag005-lr1e-3",
              synth(sigma=flat_sigma(1500), fd=lambda s: 1.0,
                    fdf=lambda s: 1.3))
    assert gates.w8_parity(runs)["verdict"] == "confirmed"
    write_run(runs, "w6-dag005-lr1e-3",
              synth(sigma=flat_sigma(1500), fd=lambda s: 1.0,
                    fdf=lambda s: 5.0))
    assert gates.w8_parity(runs)["verdict"] == "falsified"
    write_run(runs, "w6-dag005-lr1e-3", synth(sigma=flat_sigma(1500)))
    assert gates.w8_parity(runs)["verdict"] == "insufficient"


# ---------------------------------------------------------------------------
# the corrected wave-5 re-audit, pinned to the manual-application
# outcomes on the archived runs (integration; requires the repo's runs/)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(gates.RUNS, "w5-base-lr1e-3")),
    reason="archived wave-5 runs not present")
def test_corrected_wave5_report_matches_manual_application():
    rep = gates.corrected_wave5_report()
    assert rep["P3_declared"]["verdict"] == "kill"
    assert rep["P6_declared"]["verdict"] == "kill"
    assert rep["P5_declared"]["verdict"] == "falsified"
    assert rep["P5_declared"]["monotone_deeper_at_larger_batch"] is False
    assert rep["P8_declared"]["verdict"] == "falsified"
    assert rep["P8_declared"]["tilt"]["clause"] is False
    assert rep["P8_declared"]["tilt"]["current"]["suppression"] > 10
    assert rep["P9_declared"]["verdict"] == "confirmed"
    assert rep["P9_declared"]["nonfinite_in_window"] is True
    assert rep["P2_declared"]["verdict"] == "falsified"
    # every quoted median carries an interval or is declared
    # interval-free (None)
    for n, cell in rep["P1_declared"].items():
        if isinstance(cell, dict) and "lq_med" in cell:
            assert "ci" in cell
    # sensitivity sweep emitted for every listed cell
    assert all(len(v) == 9 for v in rep["endpoint_sensitivity"].values())
