"""Synthetic-record branch tests of the wave-8 executable
preregistration (analysis/gates.py W8 verdict functions): every item
is driven through its declared pass/kill/fail/insufficient branches on
crafted logs, in the house pattern of test_gates_w7.  Test-speed
constants (surrogate, bootstrap, and placement counts; the pulse grid)
are reduced via monkeypatch; clause semantics are untouched."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "analysis"))

import gates  # noqa: E402
import thresholds as T  # noqa: E402


@pytest.fixture(autouse=True)
def fast_constants(monkeypatch):
    monkeypatch.setattr(T, "AV_N_SURR", 10)
    monkeypatch.setattr(T, "AV_GOF_BOOT", 40)
    monkeypatch.setattr(T, "AV_N_SHIFT", 40)
    monkeypatch.setattr(T, "BOOT_N", 250)
    monkeypatch.setattr(T, "W8_N_PLACE", 40)
    monkeypatch.setattr(T, "W8_PULSE_STARTS",
                        tuple(range(3000, 7500, 900)))
    monkeypatch.setattr(T, "W8_MIN_PULSES", 4)
    gates._GN_CACHE.clear()
    yield
    gates._GN_CACHE.clear()


def write_run(runs, name, records, final_mgen=None, config=None):
    d = os.path.join(runs, name)
    os.makedirs(d, exist_ok=True)
    base_cfg = {"warmup": 250, "batch": 32, "ctx": 256, "accum": 1}
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


def add_blocks(y, rng, n, lo, hi, law, thr=2.0):
    for _ in range(n):
        d = int(rng.integers(2, 7))
        s = int(rng.integers(lo + 1, hi - d))
        y[s - 1:s - 1 + d] = thr + law(d) / d
    return y


def add_exp_bursts(y, rng, n, lo, hi, scale=1.0, thr=3.0):
    for s in rng.integers(lo + 1, hi, n):
        y[int(s) - 1] = thr + scale * (0.2 + rng.exponential(0.6))


def add_clustered(y, rng, n_clusters, lo, hi, per=3, thr=3.0):
    for c in rng.integers(lo + 50, hi - 50, n_clusters):
        for k in range(per):
            y[int(c) + 6 * k] = thr + 0.5 + rng.exponential(1.5)


def records_from(y, sigma=None, loss=None, lr=1e-3, pre_live=None):
    recs = []
    for s in range(1, len(y) + 1):
        r = {"step": s, "loss": (loss(s) if loss else 5.0), "lr": lr,
             "gnorm_plga": float(y[s - 1]), "dtheta_plga": 1.0}
        if sigma is not None and s % 50 == 0:
            r["rowmap_sigma_med"] = sigma(s)
        if pre_live is not None and s % 100 == 0:
            r["lam_pre_live_plga"] = pre_live(s)
        recs.append(r)
    return recs


def flat_sigma(onset):
    return lambda s: [0.01 if s >= onset else 1.5, 1.2, 1.3]


def sens_sigma():
    return lambda s: [1.5, 1.2, 1.3]


# ---------------------------------------------------------------------------
# W8.1 size-duration gamma (primary)


def write_gamma_grid(runs, law, n=70):
    for i, name in enumerate(gates.W7_CONST + gates.W8_CONST75):
        y, rng = base_series(9000, seed=200 + i)
        add_blocks(y, rng, n, T.W7_CONST_STAT_START, 9000, law)
        write_run(runs, name, records_from(y))


def test_w8_1_superlinear_passes(tmp_path):
    write_gamma_grid(str(tmp_path), lambda d: float(d) ** 2)
    rep = gates.w8_1_gamma(str(tmp_path))
    for trio in rep["trios"].values():
        assert trio["verdict"] == "pass"
    assert rep["band_clause"]["ok"]
    assert rep["verdict"] == "pass"


def test_w8_1_linear_kills(tmp_path):
    write_gamma_grid(str(tmp_path), lambda d: 3.0 * float(d))
    rep = gates.w8_1_gamma(str(tmp_path))
    assert rep["verdict"] == "kill"


def test_w8_1_insufficient(tmp_path):
    write_gamma_grid(str(tmp_path), lambda d: float(d) ** 2, n=6)
    rep = gates.w8_1_gamma(str(tmp_path))
    assert rep["verdict"] == "insufficient"


# ---------------------------------------------------------------------------
# W8.2 anneal trajectory


def write_anneal_cells(runs, late_counts, lateann=True):
    for (floor, name), n_late in zip(gates.W8_FLOOR_CELLS,
                                     late_counts):
        steps = 12000 if name != "w7-const-lr1e-3" else 12500
        y, rng = base_series(steps, seed=300 + int(floor * 100))
        add_exp_bursts(y, rng, 40, 2000, T.W8_LATE_LO)
        add_exp_bursts(y, rng, n_late, T.W8_LATE_LO + 1, 12000)
        write_run(runs, name, records_from(y, sigma=flat_sigma(1500)),
                  final_mgen=0.001,
                  config={"warmup": 1000 if floor < 1.0 else 250})
    if lateann:
        y, rng = base_series(4000, seed=310)
        write_run(runs, gates.W8_LATE_CELL,
                  records_from(y, sigma=flat_sigma(1200)),
                  final_mgen=0.001, config={"warmup": 1})
    for name, onset in ((gates.W8_HOLD_CELL, 7000),
                        (gates.W8_SHORT_CELL, 2500)):
        y, rng = base_series(12000 if name == gates.W8_HOLD_CELL
                             else 6000, seed=320)
        write_run(runs, name, records_from(y, sigma=flat_sigma(onset)),
                  final_mgen=0.001, config={"warmup": 1000})


def test_w8_2_lockin_chain_passes(tmp_path):
    write_anneal_cells(str(tmp_path), late_counts=(8, 30, 90, 260))
    rep = gates.w8_2_anneal(str(tmp_path))
    assert rep["chain_verdict"] == "pass"
    assert rep["cells"]["lateann"]["verdict"] == "pass"
    assert rep["cells"]["hold6k"]["segment"] == "descent"
    assert rep["verdict"] == "pass"


def test_w8_2_inverted_chain_kills(tmp_path):
    write_anneal_cells(str(tmp_path), late_counts=(500, 120, 25, 3))
    rep = gates.w8_2_anneal(str(tmp_path))
    assert rep["chain_verdict"] == "kill"
    assert rep["verdict"] == "kill"


def test_w8_2_missing_lateann_insufficient(tmp_path):
    write_anneal_cells(str(tmp_path), late_counts=(8, 30, 90, 260),
                       lateann=False)
    rep = gates.w8_2_anneal(str(tmp_path))
    assert rep["verdict"] == "insufficient"


# ---------------------------------------------------------------------------
# W8.3 susceptibility


def write_pulse_cells(runs, respond):
    sizes = {3e-4: 0.0, 7.5e-4: 9.0, 1e-3: 16.0}
    for name, lr in gates.W8_PULSE.items():
        y, rng = base_series(8000, seed=400 + int(lr * 1e5))
        add_exp_bursts(y, rng, 25, 2000, 8000, scale=0.4)
        if respond and sizes[lr] > 0:
            for t in T.W8_PULSE_STARTS:
                y[t + T.W8_PULSE_LEN + 2] = 3.0 + sizes[lr]
                y[t + T.W8_PULSE_LEN + 30] = 3.0 + sizes[lr] / 2
        write_run(runs, name, records_from(y, lr=lr),
                  final_mgen=0.5)


def test_w8_3_response_passes(tmp_path):
    write_pulse_cells(str(tmp_path), respond=True)
    rep = gates.w8_3_susceptibility(str(tmp_path))
    assert rep["ordering"]["order_ok"]
    assert rep["verdict"] == "pass"


def test_w8_3_no_response_kills(tmp_path):
    write_pulse_cells(str(tmp_path), respond=False)
    rep = gates.w8_3_susceptibility(str(tmp_path))
    assert rep["verdict"] == "kill"


# ---------------------------------------------------------------------------
# W8.4 branching (reported)


def test_w8_4_clustered_confirmed(tmp_path):
    for i, name in enumerate(gates.W7_CONST + gates.W8_CONST75):
        y, rng = base_series(9000, seed=500 + i)
        add_clustered(y, rng, 30, T.W7_CONST_STAT_START, 9000)
        write_run(str(tmp_path), name, records_from(y))
    ys, rng = base_series(9000, seed=510)
    add_exp_bursts(ys, rng, 60, T.W7_CONST_STAT_START, 9000)
    write_run(str(tmp_path), gates.W7_CONST_SUB, records_from(ys))
    rep = gates.w8_4_branching(str(tmp_path))
    assert rep["summary"]["all_sigma_below_1"]
    assert rep["verdict"] == "confirmed"


def test_w8_4_uniform_falsified(tmp_path):
    for i, name in enumerate(gates.W7_CONST + gates.W8_CONST75
                             + [gates.W7_CONST_SUB]):
        y, rng = base_series(9000, seed=520 + i)
        add_exp_bursts(y, rng, 45, T.W7_CONST_STAT_START, 9000)
        write_run(str(tmp_path), name, records_from(y))
    rep = gates.w8_4_branching(str(tmp_path))
    assert rep["verdict"] in ("falsified", "descriptive")
    assert rep["summary"]["n_above_poisson"] <= 2


# ---------------------------------------------------------------------------
# W8.5 finite size


def write_width_cells(runs, scales, collapsing=True):
    for (width, cells), scale in zip(gates.W8_WIDTH, scales):
        for j, name in enumerate(cells):
            y, rng = base_series(9000, seed=600 + width + j)
            for s in rng.integers(1100, 5000, 120):
                y[int(s) - 1] = 3.0 + scale * (
                    0.5 * (1 - rng.random()) ** (-1 / 1.5))
            sig = flat_sigma(5500) if collapsing else sens_sigma()
            write_run(runs, name, records_from(y, sigma=sig),
                      final_mgen=0.001 if collapsing else 0.5,
                      config={"warmup": 1000})


def test_w8_5_cutoff_growth_passes(tmp_path):
    write_width_cells(str(tmp_path), scales=(0.3, 1.0, 4.0))
    rep = gates.w8_5_fss(str(tmp_path))
    assert rep["monotone_up"]["d128_lt_d256"]
    assert rep["monotone_up"]["d256_lt_d512"]
    assert rep["verdict"] == "pass"
    assert not rep["phase_change"]


def test_w8_5_phase_change_flagged(tmp_path):
    write_width_cells(str(tmp_path), scales=(0.3, 1.0, 4.0),
                      collapsing=False)
    rep = gates.w8_5_fss(str(tmp_path))
    assert rep["phase_change"]
    assert "verdict_note" in rep


# ---------------------------------------------------------------------------
# W8.6 constant 7.5e-4 trio


def write_fallback_cell(runs, name, mode):
    steps = 12000
    y, rng = base_series(steps, seed=700 + sum(ord(c) for c in name))
    if mode == "stationary":
        # stratified cluster grid: stationary by construction
        for c in range(T.W7_CONST_STAT_START + 50, steps - 40, 45):
            c = c + int(rng.integers(0, 10))
            for k in range(3):
                y[int(c) + 7 * k] = 3.0 + 0.5 * (
                    1 - rng.random()) ** (-1 / 2.0)
        # hugging margins with a deliberate V-shape across thirds so
        # the medians are non-monotone by construction
        vals = 30.0 * np.exp(-rng.exponential(0.25, steps // 100 + 1))
        idx = np.arange(len(vals))
        third = len(vals) // 3
        vals[(idx >= third) & (idx < 2 * third)] *= 0.85
        exc = rng.choice(np.arange(25, len(vals) - 2), 5,
                         replace=False)
        vals[exc] = 200.0
        pl = lambda s: float(vals[s // 100])  # noqa: E731
        loss = None
    else:  # diverged
        add_exp_bursts(y, rng, 60, T.W7_CONST_STAT_START, steps)
        pl = lambda s: 30.0  # noqa: E731
        loss = lambda s: 5.0 if s < 6000 else 25.0  # noqa: E731
    write_run(runs, name,
              records_from(y, sigma=flat_sigma(3000), pre_live=pl,
                           loss=loss),
              final_mgen=0.001, config={"warmup": 250})


def test_w8_6_stationary_passes(tmp_path):
    for n in gates.W8_CONST75:
        write_fallback_cell(str(tmp_path), n, "stationary")
    write_fallback_cell(str(tmp_path), gates.W7_CONST_SUB,
                        "stationary")
    rep = gates.w8_6_const_fallback(str(tmp_path))
    assert rep["stationarity"]["verdict"] == "pass"
    assert rep["margin"]["a_verdict"] == "pass"
    assert rep["verdict"] in ("pass", "fail")
    assert all(v and v["collapsing"]
               for v in rep["rowmap_phase"].values())


def test_w8_6_divergence_halts_with_contingency(tmp_path):
    write_fallback_cell(str(tmp_path), gates.W8_CONST75[0], "diverged")
    write_fallback_cell(str(tmp_path), gates.W8_CONST75[1], "diverged")
    write_fallback_cell(str(tmp_path), gates.W8_CONST75[2],
                        "stationary")
    write_fallback_cell(str(tmp_path), gates.W7_CONST_SUB,
                        "stationary")
    rep = gates.w8_6_const_fallback(str(tmp_path))
    assert rep["verdict"] == "halt"
    assert "6e-04" in rep["contingency"] or "0.0006" in \
        rep["contingency"]
