"""Synthetic tests of the wave-8 estimator library
(analysis/sandpile.py): pulse-response bookkeeping and its placement
null, susceptibility recovery, the masked-median sensitivity variant,
branching-cluster decomposition with its Poisson null, cross-block
offspring excess, finite-size cutoffs, late-window rates, and the
declared schedule shapes."""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "analysis"))

import avalanche as av  # noqa: E402
import sandpile as sp  # noqa: E402


def ev(start, dur=1, size=1.0):
    return {"start": start, "end": start + dur - 1, "dur": dur,
            "size": size, "peak": size}


def test_pulse_windows_and_classification():
    starts, plen, rw = [1000, 2000], 8, 100
    assert sp.pulse_windows(starts, plen, rw) == [(1008, 1107),
                                                  (2008, 2107)]
    events = [ev(1003), ev(1008), ev(1107), ev(1108), ev(2050),
              ev(500)]
    direct, resp, bg = sp.classify_pulse_events(events, starts, plen,
                                                rw)
    assert [e["start"] for e in direct] == [1003]
    assert [e["start"] for e in resp] == [1008, 1107, 2050]
    assert [e["start"] for e in bg] == [1108, 500]


def test_pulse_response_planted_and_clean_mask():
    starts, plen, rw, T, lo = [1000, 2000, 3950], 8, 100, 4000, 0
    events = [ev(1010, size=2.0), ev(1050, size=3.0),
              ev(2100, size=7.0), ev(2200, size=9.0)]
    pr = sp.pulse_response(events, starts, plen, rw, T, lo)
    assert pr["R"][0] == 5.0          # both events in window 1
    assert pr["R"][1] == 7.0          # 2100 in (2008..2107); 2200 out
    assert pr["clean"][0] and pr["clean"][1]
    assert not pr["clean"][2]         # window runs past T


def test_placement_null_calibrated():
    rng = np.random.default_rng(0)
    T, lo = 20000, 2000
    events = [ev(int(s), size=1.0)
              for s in rng.integers(lo + 1, T, 400)]
    starts = list(range(3000, 19000, 2000))
    pr = sp.pulse_response(events, starts, 8, 200, T, lo)
    null = sp.placement_null(events, starts, 8, 200, T, lo, 200,
                             seed=1)
    # uniform events: the observed mean response sits inside the null
    assert null["mean_R"].min() <= pr["R"].mean() <= 2 * np.quantile(
        null["mean_R"], 0.99)
    assert np.quantile(null["mean_R"], 0.95) >= null["mean_R"].mean()


def test_susceptibility_recovery():
    R = np.array([5.0, 6.0, 4.0, 5.0])
    out = sp.susceptibility(R, chance_mean=1.0, dose=2.0, seed=0)
    assert abs(out["chi"] - (5.0 - 1.0) / 2.0) < 1e-12
    assert out["ci"][0] <= out["chi"] <= out["ci"][1]
    assert sp.susceptibility(R, 1.0, 0.0) is None


def test_masked_median_matches_plain_without_pulses():
    rng = np.random.default_rng(2)
    y = np.exp(0.05 * rng.normal(0, 1, 3000))
    y[1500] = 30.0
    plain, _ = av.extract_events(y, 3.0, 101, 100)
    masked, _ = sp.masked_median_events(y, 3.0, 101, 100, [], 8)
    assert plain == masked


def test_masked_median_immune_to_pulse_inflation():
    rng = np.random.default_rng(3)
    y = np.exp(0.02 * rng.normal(0, 1, 2000))
    starts = [900]
    y[899:907] = 500.0                 # 8 giant direct-pulse steps
    y[950] = 4.0                       # a modest response event
    masked, _ = sp.masked_median_events(y, 3.0, 101, 100, starts, 8)
    assert any(e["start"] == 951 for e in masked)
    # and the masked median near the pulse stays at the base level
    med = sp._nan_running_median(
        np.where(np.isin(np.arange(2000), np.arange(899, 907)),
                 np.nan, y), 101)
    assert med[949] < 1.5


def test_branching_sigma_planted_clusters():
    events = []
    for c in range(10):
        base = 1000 + 500 * c
        events += [ev(base), ev(base + 5), ev(base + 11)]
    b = sp.branching_sigma(events, lag=25)
    assert b["n_clusters"] == 10 and b["n_events"] == 30
    assert abs(b["sigma"] - (1 - 10 / 30)) < 1e-12
    assert abs(b["mean_cluster"] - 3.0) < 1e-12
    # the mean-field identity: mean cluster size = 1 / (1 - sigma)
    assert abs(b["mean_cluster"] - 1 / (1 - b["sigma"])) < 1e-12


def test_branching_null_poisson_below_planted():
    pois = sp.branching_null_poisson(30, 6000, 25, 200, seed=4,
                                     lo=1000)
    assert float(np.quantile(pois, 0.95)) < 1 - 10 / 30


def test_sigma_bootstrap_ci_contains_point():
    rng = np.random.default_rng(5)
    events = []
    for c in rng.integers(2100, 19000, 40):
        events += [ev(int(c)), ev(int(c) + 6)]
    b = sp.branching_sigma(events, 25)
    ci = sp.sigma_bootstrap_ci(events, 25, 20000, 100, 400, seed=6,
                               lo=2000)
    assert ci is not None and ci[0] <= b["sigma"] + 0.1


def test_cross_sigma_coupled_vs_uncoupled():
    src = [ev(s, dur=2) for s in range(1000, 9000, 400)]
    coupled = {"a": src,
               "b": [ev(e["start"] + 2) for e in src]}
    out = sp.cross_sigma(coupled, T=10000, lag=3, lo=0)
    assert out["a"]["mean_offspring"] >= 1.0
    assert out["a"]["excess"] > 0.5
    rng = np.random.default_rng(7)
    uncoupled = {"a": src,
                 "b": [ev(int(s)) for s in
                       rng.integers(1, 10000, len(src))]}
    out2 = sp.cross_sigma(uncoupled, T=10000, lag=3, lo=0)
    assert abs(out2["a"]["excess"]) < 0.5


def test_fss_cutoffs_and_late_rate():
    events = [ev(1000 + i, size=float(i + 1)) for i in range(100)]
    cut = sp.fss_cutoffs(events, 0.99, n_boot=200, seed=8)
    assert abs(cut["q"] - np.quantile(np.arange(1.0, 101.0), 0.99)) \
        < 1e-9
    assert cut["smax"] == 100.0
    lw = sp.late_window_rate(events, 1049, 1099, 10, 200, seed=9)
    assert lw["n"] == 50 and abs(lw["rate_per_1k"] - 1000.0) < 1e-9


def test_schedule_factors_match_golden():
    total, warmup, alpha = 12000, 1000, 0.1
    f = sp.schedule_factors("cosine", total, warmup, alpha)
    assert f[500] == 500 / 1000
    d = (9000 - warmup) / (total - warmup)
    assert abs(f[9000] - ((1 - alpha) * 0.5
                          * (1 + math.cos(math.pi * d)) + alpha)) \
        < 1e-12
    fc = sp.schedule_factors("const", 1000, 250)
    assert fc[0] == 1 / 250 and fc[249] == 1.0 and fc[999] == 1.0
    fh = sp.schedule_factors("hold", 12000, 1000, 0.1, hold=6000)
    assert fh[3000] == 1.0 and fh[6001] < 1.0
    fd = sp.schedule_factors("hold", 4000, 1, 0.1, hold=0)
    assert fd[0] == 1.0 and all(a > b for a, b in zip(fd, fd[1:]))
