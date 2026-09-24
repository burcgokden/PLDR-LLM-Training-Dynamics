"""Unit tests of the wave-7 avalanche statistics library
(analysis/avalanche.py): extraction and exclusion conventions on
planted series, CSN fit recovery on sampled laws, Vuong comparisons,
clustering, surrogates, size-duration, Omori, coincidence arithmetic,
and the margin statistics."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "analysis"))

import avalanche as av  # noqa: E402


def planted(steps, spikes, base=1.0):
    """Constant baseline with spikes: {step: value}."""
    y = np.full(steps, base)
    for s, v in spikes.items():
        y[s - 1] = v
    return y


# ---------------------------------------------------------------------------
# extraction


def test_running_median_interior_and_edges():
    y = np.arange(1.0, 202.0)
    m = av.running_median(y, 101)
    assert m[100] == y[100]          # centered interior
    assert m[0] == y[0]              # radius 0 at the boundary
    assert m[5] == np.median(y[0:11])


def test_extract_events_sizes_and_warmup():
    y = planted(1000, {100: 10.0, 500: 4.0, 501: 5.0, 900: 2.0})
    ev, dropped = av.extract_events(y, 3.0, 101, warmup=0)
    # threshold = 3.0; step 900 value 2.0 is below it
    assert [e["start"] for e in ev] == [100, 500]
    assert ev[0]["size"] == pytest.approx(7.0)
    assert ev[1]["dur"] == 2
    assert ev[1]["size"] == pytest.approx((4.0 - 3.0) + (5.0 - 3.0))
    ev2, dropped2 = av.extract_events(y, 3.0, 101, warmup=200)
    assert [e["start"] for e in ev2] == [500]
    assert dropped2 == 1


def test_exclude_matched_rules_differ():
    # A at [10, 11]; B at [13, 14]: span gap 2 -> dilated match,
    # start gap 3 -> start-rule miss (the 47/48 analogue).
    a = [{"start": 10, "end": 11}]
    b = [{"start": 13, "end": 14}]
    kept, n_dil, n_start = av.exclude_matched(a, b, 2)
    assert n_dil == 1 and n_start == 0 and kept == []
    # gap 3 -> no match under either rule
    b2 = [{"start": 15, "end": 16}]
    kept2, n_dil2, n_start2 = av.exclude_matched(a, b2, 2)
    assert n_dil2 == 0 and n_start2 == 0 and len(kept2) == 1


# ---------------------------------------------------------------------------
# tail fits


def test_powerlaw_recovery():
    rng = np.random.default_rng(0)
    sizes = av.sample_powerlaw(2000, 2.5, 1.0, rng)
    f = av.fit_powerlaw(sizes)
    assert abs(f["alpha"] - 2.5) < 0.1
    assert f["xmin"] < 2.0


def test_gof_accepts_powerlaw():
    rng = np.random.default_rng(1)
    sizes = av.sample_powerlaw(300, 2.0, 1.0, rng)
    f = av.fit_powerlaw(sizes)
    p = av.gof_pvalue(sizes, f, 100, seed=2)
    assert p >= 0.1


def test_vuong_powerlaw_beats_exponential():
    rng = np.random.default_rng(3)
    sizes = av.sample_powerlaw(500, 2.0, 1.0, rng)
    f = av.fit_powerlaw(sizes)
    r, p, winner = av.vuong_vs_exponential(sizes, f["alpha"], f["xmin"])
    assert winner == "powerlaw" and p < 0.05


def test_vuong_exponential_sample():
    rng = np.random.default_rng(4)
    sizes = 1.0 + rng.exponential(0.5, 800)
    f = av.fit_powerlaw(sizes)
    r, p, winner = av.vuong_vs_exponential(sizes, f["alpha"], f["xmin"])
    assert not (winner == "powerlaw" and p < 0.05)


def test_vuong_lognormal_sample_favors_lognormal():
    rng = np.random.default_rng(5)
    sizes = rng.lognormal(0.0, 1.5, 1500)
    f = av.fit_powerlaw(sizes)
    r, p, winner = av.vuong_vs_lognormal(sizes, f["alpha"], f["xmin"])
    assert not (winner == "powerlaw" and p < 0.05)


def test_ccdf_slope_pareto():
    rng = np.random.default_rng(6)
    sizes = av.sample_powerlaw(5000, 2.6, 1.0, rng)
    s = av.ccdf_slope(sizes)
    assert abs(s - (-(2.6 - 1.0))) < 0.15


# ---------------------------------------------------------------------------
# clustering, surrogates, scaling


def test_cv_poisson_vs_bursty():
    rng = np.random.default_rng(7)
    starts = np.cumsum(rng.exponential(50, 200)).astype(int) + 1
    ev = [{"start": int(s)} for s in starts]
    cv = av.interevent_cv(ev)
    ci = av.cv_bootstrap_ci(ev, 500, seed=8)
    assert abs(cv - 1.0) < 0.25
    assert ci[0] < 1.0 < ci[1] or abs(cv - 1.0) < 0.15
    # bursty: clusters of 5 events, long gaps between clusters
    bursty = []
    t = 100
    for _ in range(40):
        for k in range(5):
            bursty.append({"start": t + 2 * k})
        t += 500
    cvb = av.interevent_cv(bursty)
    cib = av.cv_bootstrap_ci(bursty, 500, seed=9)
    assert cvb > 1.2 and cib[0] > 1.0


def test_surrogates_preserve_marginals():
    rng = np.random.default_rng(10)
    y = np.exp(rng.normal(0, 1, 512))
    s1 = av.surrogate_ar1(y, np.random.default_rng(11))
    s2 = av.surrogate_phase(y, np.random.default_rng(12))
    assert np.allclose(np.sort(s1), np.sort(y))
    assert np.allclose(np.sort(s2), np.sort(y))


def test_ar1_fit_recovery():
    rng = np.random.default_rng(13)
    n, phi_true = 5000, 0.7
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = phi_true * z[i - 1] + rng.normal(0, 0.5)
    phi, s = av.fit_ar1_log(np.exp(z))
    assert abs(phi - phi_true) < 0.05


def test_size_duration_gamma_two():
    ev = []
    rng = np.random.default_rng(14)
    for _ in range(80):
        d = int(rng.integers(2, 9))
        ev.append({"start": int(rng.integers(1, 10000)), "dur": d,
                   "size": float(d) ** 2})
    sd = av.size_duration(ev, n_boot=300, seed=15)
    assert abs(sd["gamma"] - 2.0) < 0.05
    assert sd["ci"][0] > 1.0


def test_coincidence_arithmetic():
    ev = [{"start": 10, "end": 10}]
    c = av.coincidence(ev, [12], 100, 3)
    assert c["observed"] == 1.0
    assert c["chance"] == pytest.approx(7 / 100)
    c2 = av.coincidence(ev, [20], 100, 3)
    assert c2["observed"] == 0.0


def test_omori_decay():
    ev = [{"start": 1000, "size": 100.0},
          {"start": 3000, "size": 90.0}]
    rng = np.random.default_rng(16)
    for m in (1000, 3000):
        for _ in range(12):
            ev.append({"start": m + int(rng.integers(1, 20)),
                       "size": 1.0})
    for _ in range(30):
        ev.append({"start": int(rng.integers(4000, 12000)),
                   "size": 1.0})
    om = av.omori(ev, 12000, 2, 25, 100, 200, seed=17)
    assert om["ratio"] > 1.5 and om["p"] < 0.05


def test_omori_uniform_null():
    rng = np.random.default_rng(18)
    ev = [{"start": int(s), "size": float(v)}
          for s, v in zip(rng.integers(1, 12000, 300),
                          rng.exponential(1.0, 300) + 1)]
    om = av.omori(ev, 12000, 5, 25, 100, 200, seed=19)
    assert om["p"] > 0.05 or om["ratio"] < 1.5


# ---------------------------------------------------------------------------
# margins


def test_margin_skew_directions():
    rng = np.random.default_rng(20)
    hug = np.exp(5.0 - rng.exponential(0.3, 500))   # mass at the top
    assert av.margin_skew(hug) < 1.0
    spread = np.exp(rng.exponential(1.0, 500))      # long upper tail
    assert av.margin_skew(spread) > 1.5


def test_margin_relaxation():
    v = np.ones(200)
    v[50] = 10.0    # excursion, next probe already back at the median
    v[120] = 10.0
    r = av.margin_relaxation(v, horizon=5)
    assert r["n_exc"] == 2
    assert r["frac_relaxed"] == 0.0  # constant series: never below q50
    rng = np.random.default_rng(21)
    w = rng.normal(1.0, 0.05, 200)
    w[100] = 10.0
    w[101] = 0.2
    r2 = av.margin_relaxation(w, horizon=5)
    assert r2["n_exc"] >= 1 and r2["frac_relaxed"] >= 0.5


# ---------------------------------------------------------------------------
# tie-aware Spearman (post-freeze correction; disclosed in
# avalanche.size_duration)


def test_rank_average_hand_example():
    # values 3, 1, 1, 2: ranks 4, 1.5, 1.5, 3
    r = av.rank_average([3.0, 1.0, 1.0, 2.0])
    assert np.allclose(r, [4.0, 1.5, 1.5, 3.0])
    # all tied: every rank is the mean rank
    assert np.allclose(av.rank_average([5.0] * 4), [2.5] * 4)


def test_rank_average_no_ties_matches_argsort():
    rng = np.random.default_rng(31)
    x = rng.standard_normal(200)          # ties a.s. absent
    assert np.allclose(av.rank_average(x),
                       np.argsort(np.argsort(x)) + 1.0)


def test_spearman_tie_aware_vs_scipy():
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(32)
    d = rng.integers(2, 6, 120).astype(float)     # heavily tied
    s = d ** 2.0 * np.exp(rng.standard_normal(120))
    events = [{"dur": int(di), "size": float(si)}
              for di, si in zip(d, s)]
    sd = av.size_duration(events)
    ref = stats.spearmanr(np.log(d), np.log(s)).statistic
    assert abs(sd["spearman"] - ref) < 1e-12


# (dur, size) pairs of the 70 D>=2 partner-excluded events of the
# first archived constant-rate cell (w7-const-lr1e-3 excluded against
# w8-const-lr7.5e-4; the W8.1 primary convention).  The tie-broken
# argsort-of-argsort implementation reported rho = 0.763 on these
# pairs; the tie-aware value is 0.520.  This fixture pins the
# correction against regression.
W81_CELL1_PAIRS = [
    (2, 0.021875), (3, 0.340987), (2, 0.222888), (2, 0.312897),
    (2, 0.299009), (2, 0.119518), (2, 0.101288), (2, 0.148509),
    (2, 0.231964), (2, 0.516281), (2, 0.254795), (2, 0.252703),
    (3, 0.344954), (2, 0.099143), (2, 0.162682), (3, 0.238232),
    (2, 0.729875), (2, 0.047022), (2, 0.107688), (2, 1.122168),
    (2, 0.820139), (2, 0.125697), (2, 2.088472), (2, 1.053733),
    (2, 0.070502), (2, 0.312849), (2, 0.976235), (2, 0.190541),
    (4, 23.903575), (3, 4.473268), (3, 5.391305), (2, 14.245495),
    (2, 0.974148), (2, 3.114352), (3, 21.296053), (5, 55.677972),
    (2, 38.273271), (2, 1.179534), (3, 35.161497), (2, 16.603659),
    (3, 484.945091), (4, 9.218194), (2, 12.065347), (2, 0.986025),
    (2, 4.752735), (4, 4.755375), (3, 21.710981), (4, 56.298009),
    (3, 34.171473), (3, 64.663889), (2, 28.980368), (3, 26.961396),
    (2, 19.11046), (2, 28.99374), (4, 75.190991), (2, 68.704952),
    (5, 235.935105), (4, 43.61982), (2, 5.878302), (2, 6.851329),
    (2, 18.529021), (2, 1.80252), (2, 17.264624), (2, 193.916391),
    (2, 71.236876), (2, 46.660364), (3, 52.284703), (4, 277.040021),
    (3, 203.395705), (2, 6.77141),
]


def test_spearman_archived_cell_fixture():
    events = [{"dur": d, "size": s} for d, s in W81_CELL1_PAIRS]
    sd = av.size_duration(events)
    assert sd["n"] == 70
    assert sd["n_dur_distinct"] == 4
    assert abs(sd["spearman"] - 0.520) < 5e-3     # tie-aware
    # the retired tie-broken implementation, for the contrast
    ld = np.log([d for d, _ in W81_CELL1_PAIRS])
    ls = np.log([s for _, s in W81_CELL1_PAIRS])
    rd = np.argsort(np.argsort(ld))
    rs = np.argsort(np.argsort(ls))
    old = float(np.corrcoef(rd, rs)[0, 1])
    assert abs(old - 0.763) < 5e-3


def test_size_duration_reports_distinct_durations():
    events = [{"dur": d, "size": float(d) ** 2.0}
              for d in (2, 2, 3, 3, 4, 5)]
    sd = av.size_duration(events)
    assert sd["n"] == 6
    assert sd["n_dur_distinct"] == 4
