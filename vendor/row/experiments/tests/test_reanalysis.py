"""Unit tests of the post-hoc reanalysis library
(analysis/reanalysis.py): causality of the trailing-median detector,
convention parity with the frozen extractor, observed-value xmin scan
recovery, and the block-bootstrap size-duration interval."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "analysis"))

import avalanche as av  # noqa: E402
import reanalysis as ra  # noqa: E402


def planted(steps, spikes, base=1.0):
    y = np.full(steps, base)
    for s, v in spikes.items():
        y[s - 1] = v
    return y


# ---------------------------------------------------------------------------
# trailing median


def test_trailing_median_is_causal():
    rng = np.random.default_rng(0)
    y = rng.lognormal(0.0, 0.5, 400)
    m1 = ra.trailing_median(y, 101)
    y2 = y.copy()
    y2[250:] = 1e6
    m2 = ra.trailing_median(y2, 101)
    assert np.allclose(m1[:250], m2[:250])


def test_trailing_median_matches_definition():
    rng = np.random.default_rng(1)
    y = rng.random(180)
    m = ra.trailing_median(y, 51)
    for i in (0, 7, 49, 50, 51, 120, 179):
        lo = max(0, i - 50)
        assert np.isclose(m[i], np.median(y[lo:i + 1]))


def test_causal_extraction_conventions_match_frozen():
    """On a series whose excursion is far from any median-window
    boundary effect, the causal and centered detectors agree on the
    event and its integrated-excess size convention."""
    y = planted(400, {200: 9.0, 201: 9.0})
    evc, _ = ra.extract_events_causal(y, 3.0, 101)
    ev, _ = av.extract_events(y, 3.0, 101)
    assert len(evc) == len(ev) == 1
    assert evc[0]["start"] == ev[0]["start"] == 200
    assert np.isclose(evc[0]["size"], ev[0]["size"])


def test_causal_extraction_no_future_leak():
    """A large future excursion cannot suppress an earlier event by
    raising the causal baseline (it can with the centered one)."""
    y = planted(400, {150: 9.0})
    y2 = y.copy()
    y2[160:220] = 50.0
    evc, _ = ra.extract_events_causal(y2, 3.0, 101)
    assert any(e["start"] == 150 for e in evc)


# ---------------------------------------------------------------------------
# observed-value xmin scan


def test_observed_grid_is_observed():
    rng = np.random.default_rng(2)
    sizes = av.sample_powerlaw(300, 2.2, 1.0, rng)
    grid = ra.xmin_grid_observed(sizes)
    assert np.all(np.isin(grid, sizes))
    f = ra.fit_powerlaw_observed(sizes)
    assert f is not None
    assert np.any(np.isclose(f["xmin"], sizes))


def test_observed_fit_recovers_alpha():
    rng = np.random.default_rng(3)
    sizes = av.sample_powerlaw(2000, 2.5, 1.0, rng)
    f = ra.fit_powerlaw_observed(sizes)
    assert abs(f["alpha"] - 2.5) < 0.25


def test_observed_fit_short_input():
    assert ra.fit_powerlaw_observed(np.ones(5)) is None


# ---------------------------------------------------------------------------
# block-bootstrap size-duration


def _events_gamma(n, gamma, seed):
    rng = np.random.default_rng(seed)
    ev = []
    t = 100
    for _ in range(n):
        d = int(rng.integers(2, 30))
        s = float(np.exp(gamma * np.log(d) + rng.normal(0, 0.05)))
        ev.append({"start": t, "end": t + d - 1, "dur": d, "size": s,
                   "peak": s})
        t += int(rng.integers(3, 40))
    return ev


def test_block_gamma_point_matches_frozen():
    ev = _events_gamma(120, 1.7, 4)
    sd = av.size_duration(ev)
    sb = ra.size_duration_block(ev, blocklen=10, n_boot=200, seed=5)
    assert np.isclose(sd["gamma"], sb["gamma"])


def test_block_gamma_ci_covers_truth():
    ev = _events_gamma(200, 1.5, 6)
    sb = ra.size_duration_block(ev, blocklen=10, n_boot=400, seed=7)
    lo, hi = sb["ci_block"]
    assert lo < 1.5 < hi


def test_block_gamma_degenerate():
    assert ra.size_duration_block([], 10, 100, 0) is None
    ev = _events_gamma(4, 1.5, 8)
    sb = ra.size_duration_block(ev, blocklen=10, n_boot=100, seed=9)
    assert sb is not None and "ci_block" not in sb


# ---------------------------------------------------------------------------
# recalibrated W8.1 joint null: exclusion-path identity and the paired
# pipeline


def _rand_events(rng, n, T):
    ev = []
    for _ in range(n):
        s = int(rng.integers(1, T - 4))
        e = s + int(rng.integers(0, 4))
        ev.append({"start": s, "end": e, "dur": e - s + 1,
                   "size": float(rng.exponential(1.0)) + 1e-3,
                   "peak": 1.0})
    return sorted(ev, key=lambda x: x["start"])


def test_exclude_fast_matches_frozen():
    rng = np.random.default_rng(40)
    for trial in range(20):
        ea = _rand_events(rng, int(rng.integers(0, 40)), 500)
        eb = _rand_events(rng, int(rng.integers(0, 40)), 500)
        for lag in (0, 2, 7):
            kept_f, nd_f, _ = av.exclude_matched(ea, eb, lag)
            kept_v, nd_v = ra.exclude_matched_fast(ea, eb, lag)
            assert nd_f == nd_v
            assert kept_f == kept_v


def test_pair_stats_match_observed_branch():
    """The surrogate branch's per-draw statistic
    (_pair_sd_stats) is the exact composition used on the observed
    side: exclude_matched followed by size_duration."""
    rng = np.random.default_rng(41)
    ea = _rand_events(rng, 60, 2000)
    eb = _rand_events(rng, 30, 2000)
    lag = 2
    got = ra._pair_sd_stats(ea, eb, lag)
    kept, nd, _ = av.exclude_matched(ea, eb, lag)
    sd = av.size_duration(kept)
    assert got["n_kept"] == len(kept)
    assert got["n_dropped"] == nd
    assert got["n_dur2"] == (sd["n"] if sd else 0)
    if sd:
        assert got["gamma"] == sd["gamma"]
        assert got["spearman"] == sd["spearman"]


def _bursty(rng, n=3000):
    y = np.exp(rng.normal(0.0, 0.2, n))
    for s in rng.integers(100, n - 10, 40):
        y[s:s + int(rng.integers(2, 6))] *= rng.uniform(3.0, 12.0)
    return y


def test_joint_surrogate_stats_structure():
    rng = np.random.default_rng(42)
    yt, yp = _bursty(rng), _bursty(rng)
    out = ra.joint_surrogate_stats(yt, yp, "ar1", 4, 2.0, 101, 100,
                                   2, seed=7)
    assert set(out) == {"n_kept", "n_dropped", "n_dur2", "gamma",
                        "spearman"}
    assert all(len(v) == 4 for v in out.values())
    assert np.all(out["n_kept"] >= 0)
    # determinism under the explicit seed
    out2 = ra.joint_surrogate_stats(yt, yp, "ar1", 4, 2.0, 101, 100,
                                    2, seed=7)
    assert np.array_equal(out["gamma"], out2["gamma"],
                          equal_nan=True)


def test_fixedmask_full_cover_drops_everything():
    rng = np.random.default_rng(43)
    yt = _bursty(rng)
    cover = [{"start": 1, "end": len(yt), "dur": len(yt),
              "size": 1.0, "peak": 1.0}]
    out = ra.fixedmask_surrogate_stats(yt, cover, "phase", 3, 2.0,
                                       101, 100, 2, seed=9)
    assert np.all(out["n_kept"] == 0)


def test_pair_block_resample_shared_boundaries():
    rng = np.random.default_rng(44)
    yt = np.arange(1000, dtype=float)
    yp = yt + 5000.0
    rt, rp = ra.pair_block_resample(yt, yp, 50, rng)
    assert len(rt) == len(rp) == 1000
    assert np.all(rp - rt == 5000.0)     # alignment preserved
