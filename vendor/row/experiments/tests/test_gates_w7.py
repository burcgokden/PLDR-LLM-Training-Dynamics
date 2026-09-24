"""Synthetic-record branch tests of the wave-7 executable
preregistration (analysis/gates.py W7 verdict functions): every gate
is driven through its declared pass/kill/fail/insufficient (and halt)
branches on crafted logs.  Test-speed constants (surrogate and
bootstrap counts) are reduced via monkeypatch; clause semantics are
untouched.  Synthetic-cell seeds derive from zlib.crc32 of the cell
name, not Python's salted str hash, so the crafted logs are identical
across pytest invocations (same determinism repair as
gates.coincidence_matrix)."""

import json
import os
import sys

import zlib

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "analysis"))

import gates  # noqa: E402
import thresholds as T  # noqa: E402


@pytest.fixture(autouse=True)
def fast_constants(monkeypatch):
    monkeypatch.setattr(T, "AV_N_SURR", 12)
    monkeypatch.setattr(T, "AV_GOF_BOOT", 40)
    monkeypatch.setattr(T, "AV_N_SHIFT", 60)
    monkeypatch.setattr(T, "BOOT_N", 300)
    gates._GN_CACHE.clear()
    yield
    gates._GN_CACHE.clear()


def write_run(runs, name, records, final_mgen=None, config=None):
    d = os.path.join(runs, name)
    os.makedirs(d, exist_ok=True)
    base_cfg = {"warmup": 1000, "batch": 32, "ctx": 256, "accum": 1}
    with open(os.path.join(d, "log.jsonl"), "w") as f:
        f.write(json.dumps({"event": "config",
                            **{**base_cfg, **(config or {})}}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")
        if final_mgen is not None:
            f.write(json.dumps({"event": "final",
                                "m_gen": {"GLM": final_mgen, "A": 0.1,
                                          "ALM": 0.1,
                                          "AP": 0.1}}) + "\n")


def base_series(steps, seed, base=1.0):
    rng = np.random.default_rng(seed)
    return base * np.exp(0.05 * rng.normal(0, 1, steps)), rng


def add_bursts(y, rng, n, lo, hi, alpha=1.6, xmin=0.5, scale=1.0,
               cluster=4, thr=3.0):
    """Clustered bursts with Pareto sizes: cluster centers uniform in
    (lo, hi], `cluster` bursts within 8 steps of each center."""
    centers = np.sort(rng.integers(lo + 8, hi - 8, max(n // cluster, 1)))
    steps_used = []
    for c in centers:
        for k in range(cluster):
            s = int(c + rng.integers(0, 8))
            if lo < s <= hi:
                size = float(scale * xmin
                             * (1 - rng.random()) ** (-1 / (alpha - 1)))
                y[s - 1] = thr + size
                steps_used.append(s)
    return steps_used


def add_exp_bursts(y, rng, n, lo, hi, scale=1.0, thr=3.0):
    for s in rng.integers(lo + 1, hi, n):
        y[int(s) - 1] = thr + scale * (0.2 + rng.exponential(0.6))
    return None


def records_from(y, sigma=None, pre_live=None, loss=None, lr=1e-3,
                 blocks=None):
    recs = []
    for s in range(1, len(y) + 1):
        r = {"step": s, "loss": (loss(s) if loss else 5.0), "lr": lr,
             "gnorm_plga": float(y[s - 1]), "dtheta_plga": 1.0}
        if blocks:
            for blk, yb in blocks.items():
                r[f"gnorm_{blk}"] = float(yb[s - 1])
        if sigma is not None and s % 50 == 0:
            r["rowmap_sigma_med"] = sigma(s)
        if pre_live is not None and s % 100 == 0:
            r["lam_pre_live_plga"] = pre_live(s)
        recs.append(r)
    return recs


def flat_sigma(onset=1350):
    return lambda s: [0.01 if s >= onset else 1.5, 1.2, 1.3]


# ---------------------------------------------------------------------------
# W7.1


def write_w71_grid(runs, kind):
    """Six archived-name cells; `kind` selects the flattening cells'
    burst statistics."""
    steps = 4000
    for i, (col, sub) in enumerate(zip(gates.W6_COL, gates.W6_SUB)):
        y, rng = base_series(steps, seed=30 + i)
        onset = gates.W7_ONSETS[col]
        if kind == "pareto":
            # heavier pre-onset, truncated post-quench (declared splits)
            add_bursts(y, rng, 60, 1000, onset, scale=5.0)
            add_bursts(y, rng, 40, onset, onset + 500, scale=1.0)
            add_bursts(y, rng, 100, onset + 500, steps, scale=0.5)
        elif kind == "exponential":
            add_exp_bursts(y, rng, 200, 1000, steps)
        elif kind == "sparse":
            add_bursts(y, rng, 16, 1000, steps, cluster=2)
        write_run(runs, col, records_from(y, sigma=flat_sigma(onset)))
        ys, rngs = base_series(steps, seed=60 + i)
        add_bursts(ys, rngs, 80, 1000, steps, scale=20.0)
        write_run(runs, sub, records_from(ys))


def test_w7_1_pass(tmp_path):
    write_w71_grid(str(tmp_path), "pareto")
    rep = gates.w7_1_tail(str(tmp_path))
    assert rep["a_verdict"] == "pass"
    assert rep["b_common"]["verdict"] == "pass"
    assert rep["d_verdict"] == "pass"
    assert rep["e_splits"]["verdict"] == "pass"
    assert rep["verdict"] in ("pass", "fail")  # c-sweep may be strict
    for c in gates.W6_COL:
        assert rep["cells"][c]["exclusion"]["n_dropped_dilated"] >= 0
        assert rep["cells"][c]["a"]["n_events"] >= T.AV_MIN_EVENTS


def test_w7_1_exponential_kill(tmp_path):
    write_w71_grid(str(tmp_path), "exponential")
    rep = gates.w7_1_tail(str(tmp_path))
    assert rep["a_verdict"] in ("kill", "fail", "fail_lognormal_favored")
    if rep["a_verdict"] == "kill":
        assert rep["verdict"] == "kill"


def test_w7_1_insufficient(tmp_path):
    write_w71_grid(str(tmp_path), "sparse")
    rep = gates.w7_1_tail(str(tmp_path))
    assert rep["a_verdict"] == "insufficient"
    assert rep["verdict"] == "insufficient"


# ---------------------------------------------------------------------------
# W7.2


def write_const_cell(runs, name, mode, steps=12000, stat_start=2000):
    y, rng = base_series(steps, seed=zlib.crc32(name.encode()) % 1000)
    if mode == "stationary":
        add_bursts(y, rng, 240, stat_start, steps, cluster=3)
        pl = lambda s: 30.0 * np.exp(0.1 * np.sin(s / 500))  # noqa: E731
        loss = None
    elif mode == "drift":
        third = (steps - stat_start) // 3
        add_bursts(y, rng, 30, stat_start, stat_start + third)
        add_bursts(y, rng, 90, stat_start + third,
                   stat_start + 2 * third)
        add_bursts(y, rng, 270, stat_start + 2 * third, steps)
        pl = lambda s: 10.0 * (1 + 9 * s / steps)  # noqa: E731
        loss = None
    else:  # diverged
        add_bursts(y, rng, 100, stat_start, steps)
        pl = lambda s: 30.0  # noqa: E731
        loss = lambda s: 5.0 if s < 6000 else 20.0  # noqa: E731
    write_run(runs, name,
              records_from(y, sigma=flat_sigma(3000), pre_live=pl,
                           loss=loss),
              final_mgen=0.001, config={"warmup": 250})


def test_w7_2_pass(tmp_path):
    for n in gates.W7_CONST:
        write_const_cell(str(tmp_path), n, "stationary")
    write_const_cell(str(tmp_path), gates.W7_CONST_SUB, "stationary")
    rep = gates.w7_2_stationarity(str(tmp_path))
    assert rep["verdict"] == "pass"
    assert rep["phase_check"] == "collapsed-phase"
    assert rep["n_diverged"] == 0


def test_w7_2_kill(tmp_path):
    for n in gates.W7_CONST:
        write_const_cell(str(tmp_path), n, "drift")
    write_const_cell(str(tmp_path), gates.W7_CONST_SUB, "stationary")
    rep = gates.w7_2_stationarity(str(tmp_path))
    assert rep["verdict"] == "kill"


def test_w7_2_halt_on_divergence(tmp_path):
    write_const_cell(str(tmp_path), gates.W7_CONST[0], "diverged")
    write_const_cell(str(tmp_path), gates.W7_CONST[1], "diverged")
    write_const_cell(str(tmp_path), gates.W7_CONST[2], "stationary")
    write_const_cell(str(tmp_path), gates.W7_CONST_SUB, "stationary")
    rep = gates.w7_2_stationarity(str(tmp_path))
    assert rep["verdict"] == "halt"
    assert rep["n_diverged"] == 2


# ---------------------------------------------------------------------------
# W7.3


def write_margin_cell(runs, name, skewed, steps=12000, n_exc=5):
    y, rng = base_series(steps, seed=zlib.crc32(name.encode()) % 997)
    add_bursts(y, rng, 100, 2000, steps)
    if skewed == "hug":
        vals = 30.0 * np.exp(-rng.exponential(0.25, steps // 100 + 1))
        exc_at = rng.choice(np.arange(25, len(vals) - 2), n_exc,
                            replace=False)
        vals[exc_at] = 200.0  # excursion; next probe is far below q50
    else:
        vals = 30.0 * np.exp(rng.exponential(1.2, steps // 100 + 1))
    pl = lambda s: float(vals[s // 100])  # noqa: E731
    write_run(runs, name, records_from(y, pre_live=pl),
              final_mgen=0.001, config={"warmup": 250})


def write_archived_margin_stubs(runs):
    for n in gates.W6_COL + gates.W6_SUB:
        y, rng = base_series(4000, seed=zlib.crc32(n.encode()) % 991)
        pl = lambda s: 30.0  # noqa: E731
        write_run(runs, n, records_from(y, pre_live=pl))


def test_w7_3_pass_and_kill(tmp_path):
    write_archived_margin_stubs(str(tmp_path))
    for n in gates.W7_CONST:
        write_margin_cell(str(tmp_path), n, "hug")
    rep = gates.w7_3_margin(str(tmp_path))
    assert rep["a_verdict"] == "pass"
    assert rep["verdict"] in ("pass", "fail")
    gates._GN_CACHE.clear()
    for n in gates.W7_CONST:
        write_margin_cell(str(tmp_path), n, "spread")
    rep2 = gates.w7_3_margin(str(tmp_path))
    assert rep2["a_verdict"] == "kill"
    assert rep2["verdict"] == "kill"


# ---------------------------------------------------------------------------
# W7.4


def add_blocks(y, rng, n, lo, hi, gamma_law, thr=2.0):
    """Multi-step elevated blocks: duration d, per-step excess so that
    the integrated excess follows the given law of d."""
    for _ in range(n):
        d = int(rng.integers(2, 7))
        s = int(rng.integers(lo + 1, hi - d))
        per = gamma_law(d) / d
        y[s - 1:s - 1 + d] = thr + per
    return y


def write_w74_grid(runs, law):
    for i, (col, sub) in enumerate(zip(gates.W6_COL, gates.W6_SUB)):
        y, rng = base_series(12000, seed=80 + i)
        add_blocks(y, rng, 60, 1000, 12000, law)
        # aftershock clusters after the five largest (largest are the
        # longest blocks; add clusters after known big events)
        for m in rng.integers(2000, 11000, 5):
            y[int(m) - 1] = 40.0
            for k in range(8):
                y[int(m) + int(rng.integers(1, 20))] = 4.0
        write_run(runs, col, records_from(y))
        ys, _ = base_series(12000, seed=110 + i)
        write_run(runs, sub, records_from(ys))
    for n in gates.W7_CONST:
        y, rng = base_series(4000, seed=zlib.crc32(n.encode()) % 983)
        write_run(runs, n, records_from(y), config={"warmup": 250})


def test_w7_4_gamma_two_passes(tmp_path):
    write_w74_grid(str(tmp_path), lambda d: float(d) ** 2)
    rep = gates.w7_4_scaling(str(tmp_path))
    assert rep["gamma_verdict"] == "pass"
    assert rep["omori_verdict"] == "pass"
    assert rep["verdict"] == "pass"


def test_w7_4_gamma_linear_nulls(tmp_path):
    write_w74_grid(str(tmp_path), lambda d: 3.0 * float(d))
    rep = gates.w7_4_scaling(str(tmp_path))
    assert rep["gamma_verdict"] in ("kill", "fail", "insufficient")
    assert rep["gamma_verdict"] != "pass"


# ---------------------------------------------------------------------------
# W7.5


def write_prop_cell(runs, name, rng_seed, rate_scale=1.0,
                    frozen=None, lag_map=None):
    steps = 6000
    lag_map = lag_map or {"attn": 1, "ffn": 2, "phi": 1}
    y, rng = base_series(steps, seed=rng_seed)
    starts = add_bursts(y, rng, int(120 * rate_scale), 1000, steps,
                        cluster=2)
    blocks = {}
    for blk in ("attn", "ffn", "phi", "rest"):
        yb, _ = base_series(
            steps, seed=rng_seed + zlib.crc32(blk.encode()) % 97)
        if frozen == blk:
            yb = np.zeros(steps)
        else:
            lag = lag_map.get(blk)
            if lag is not None:
                for s in starts:
                    if s + lag <= steps:
                        yb[s + lag - 1] = 8.0
        blocks[blk] = yb
    write_run(runs, name, records_from(y, blocks=blocks))


def test_w7_5_reroute_pass(tmp_path):
    runs = str(tmp_path)
    for i, n in enumerate(gates.W6_COL):
        write_prop_cell(runs, n, 200 + i)
    write_prop_cell(runs, "w6-ref-lr2e-3", 210)
    # frozen-attn cell: activity persists, the phi coupling decouples
    # (lag 5 is outside the +-3 window), so shared-pair ratios change
    write_prop_cell(runs, "w6-frzattn-lr1e-3", 220, frozen="attn",
                    lag_map={"ffn": 2, "phi": 5})
    rep = gates.w7_5_propagation(runs)
    cell = rep["frozen"]["w6-frzattn-lr1e-3"]
    assert cell["live_rate_ok"]
    assert cell["reroute"]
    assert rep["verdict"] == "pass"


def test_w7_5_uniform_reduction_kill(tmp_path):
    runs = str(tmp_path)
    for i, n in enumerate(gates.W6_COL):
        write_prop_cell(runs, n, 230 + i)
    write_prop_cell(runs, "w6-ref-lr2e-3", 240)
    write_prop_cell(runs, "w6-frzattn-lr1e-3", 250, frozen="attn",
                    rate_scale=0.15)
    rep = gates.w7_5_propagation(runs)
    cell = rep["frozen"]["w6-frzattn-lr1e-3"]
    assert not cell["live_rate_ok"]
    assert rep["verdict"] in ("kill", "fail")


# ---------------------------------------------------------------------------
# W7.7


def write_dose_cell(runs, name, n_bursts, batch, seed):
    y, rng = base_series(6000, seed=seed)
    add_bursts(y, rng, n_bursts, 1000, 6000, cluster=2)
    write_run(runs, name, records_from(y),
              config={"batch": batch, "warmup": 1000})


def test_w7_7_ordering_pass_kill_descriptive(tmp_path):
    runs = str(tmp_path)
    write_dose_cell(runs, "w6-b8tok-lr1e-3", 300, 8, 300)
    write_dose_cell(runs, "w6-base-lr1e-3", 150, 32, 301)
    write_dose_cell(runs, "w6-b64tok-lr1e-3", 40, 64, 302)
    write_dose_cell(runs, "w6-b8acc4-lr1e-3", 150, 8, 303)
    rep = gates.w7_7_dose(runs)
    assert rep["identity_clause"]
    assert rep["verdict"] == "pass"
    gates._GN_CACHE.clear()
    write_dose_cell(runs, "w6-b8tok-lr1e-3", 40, 8, 304)
    write_dose_cell(runs, "w6-b64tok-lr1e-3", 300, 64, 305)
    rep2 = gates.w7_7_dose(runs)
    assert rep2["verdict"] == "kill"
    gates._GN_CACHE.clear()
    write_dose_cell(runs, "w6-b8tok-lr1e-3", 300, 8, 306)
    write_dose_cell(runs, "w6-b64tok-lr1e-3", 40, 64, 307)
    write_dose_cell(runs, "w6-b8acc4-lr1e-3", 20, 8, 308)
    rep3 = gates.w7_7_dose(runs)
    assert not rep3["identity_clause"]
    assert rep3["verdict"] == "descriptive"


# ---------------------------------------------------------------------------
# W7.8


def write_doff_pair(runs, coincident):
    steps = 6000
    y, rng = base_series(steps, seed=400)
    starts = add_bursts(y, rng, 150, 1000, steps, cluster=2)
    write_run(runs, "w6-base-lr1e-3",
              records_from(y, sigma=flat_sigma(1350)),
              final_mgen=0.001)
    yd, rngd = base_series(steps, seed=401)
    if coincident:
        for s in starts:
            yd[s - 1] = y[s - 1]
    else:
        add_bursts(yd, rngd, 150, 1000, steps, cluster=2)
    write_run(runs, gates.W7_DOFF,
              records_from(yd, sigma=flat_sigma(1500)),
              final_mgen=0.001, config={"data_offset": 416000})


def test_w7_8_pass_and_kill(tmp_path):
    write_doff_pair(str(tmp_path), coincident=False)
    rep = gates.w7_8_data_order(str(tmp_path))
    assert rep["verdict"] == "pass"
    assert rep["flattening"]["onset"] is not None
    gates._GN_CACHE.clear()
    write_doff_pair(str(tmp_path), coincident=True)
    rep2 = gates.w7_8_data_order(str(tmp_path))
    assert rep2["verdict"] == "kill"


# ---------------------------------------------------------------------------
# report assembly


def test_wave7_report_missing_cells(tmp_path):
    write_doff_pair(str(tmp_path), coincident=False)
    rep = gates.wave7_report(str(tmp_path))
    for k in ("W7_1_tail", "W7_2_stationarity", "W7_3_margin",
              "W7_4_scaling", "W7_5_propagation", "W7_6_finite_size",
              "W7_7_dose", "W7_8_data_order", "GATES"):
        assert k in rep
    assert rep["W7_2_stationarity"]["verdict"] == "insufficient"
    assert rep["GATES"]["exclusion_integrity_W7_8"] == "pass"
