"""Avalanche event statistics for the wave-7 preregistration
(wave7_prereg.md).  Pure numpy functions over per-step series and
event lists; no file I/O and no imports from gates.py (gates.py is the
consumer).  Every convention here is the one frozen in the
preregistration: centered shrinking running median, integrated-excess
event sizes, dilated-interval exclusion, target-start coincidence with
deduplicated-occupancy chance, CSN tail fits with KS xmin selection
and semiparametric goodness-of-fit bootstrap, Vuong-normalized
likelihood ratios against exponential and (edge-truncated) lognormal
alternatives, and rank-remapped AR(1) / phase-randomized surrogates.
Stochastic routines take an explicit seed.  Synthetic tests in
tests/test_avalanche.py.

One post-freeze correction is applied in place (disclosed in
size_duration): the Spearman rank correlation uses average ranks
(tie-aware) instead of the original argsort-of-argsort ranks, which
are arbitrary within ties.  No gate verdict conditions on the
Spearman value and the frozen gate reports are not rewritten.  The
paired-run joint surrogate null (the recalibration of the W8.1
envelope comparison) lives in reanalysis.py; the surrogate_stats
below is the frozen single-series pipeline.
"""

import math
import zlib

import numpy as np


# ---------------------------------------------------------------------------
# event extraction


def running_median(y, win):
    """Centered running median with symmetrically shrinking windows at
    the edges: at index i the radius is min(win//2, i, n-1-i)."""
    y = np.asarray(y, float)
    n = len(y)
    h = win // 2
    out = np.empty(n)
    if n >= win:
        sw = np.lib.stride_tricks.sliding_window_view(y, win)
        out[h:n - h] = np.median(sw, axis=1)
        edge = min(h, n - h)
    else:
        edge = (n + 1) // 2
    for i in range(edge):
        r = min(h, i, n - 1 - i)
        out[i] = np.median(y[i - r:i + r + 1])
        j = n - 1 - i
        r = min(h, j, n - 1 - j)
        out[j] = np.median(y[j - r:j + r + 1])
    return out


def extract_events(y, mult, win, warmup=0):
    """Maximal contiguous runs with y > mult * running median.  y[i] is
    the value at step i+1.  Events whose start step is <= warmup are
    discarded (the count of discarded events is returned).  Returns
    (events, n_warmup_dropped); each event is a dict with 1-based
    inclusive step indices start/end, dur, size (integrated excess
    above the threshold curve), and peak."""
    y = np.asarray(y, float)
    thr = mult * running_median(y, win)
    above = y > thr
    events, dropped = [], 0
    i, n = 0, len(y)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            ev = {"start": i + 1, "end": j + 1, "dur": j - i + 1,
                  "size": float(np.sum(y[i:j + 1] - thr[i:j + 1])),
                  "peak": float(np.max(y[i:j + 1]))}
            if ev["start"] <= warmup:
                dropped += 1
            else:
                events.append(ev)
            i = j + 1
        else:
            i += 1
    return events, dropped


def exclude_matched(ev_a, ev_b, lag):
    """Data-event exclusion between runs sharing the data order.
    Primary (dilated-interval) rule: A and B are matched when the
    dilated interval [start_A - lag, end_A + lag] intersects
    [start_B, end_B], equivalently when the gap between the two event
    spans is at most lag steps.  Sensitivity (start-step) rule:
    |start_A - start_B| <= lag.  Returns (kept_events,
    n_dropped_dilated, n_dropped_startrule); kept_events applies the
    primary rule."""
    kept, n_dil, n_start = [], 0, 0
    for a in ev_a:
        dil = any(a["start"] - lag <= b["end"]
                  and b["start"] - lag <= a["end"] for b in ev_b)
        srule = any(abs(a["start"] - b["start"]) <= lag for b in ev_b)
        if dil:
            n_dil += 1
        else:
            kept.append(a)
        if srule:
            n_start += 1
    return kept, n_dil, n_start


# ---------------------------------------------------------------------------
# tail fits (Clauset-Shalizi-Newman, continuous)


def _alpha_ks(sizes, xmin):
    tail = sizes[sizes >= xmin]
    n = len(tail)
    if n < 2:
        return None
    s = np.sum(np.log(tail / xmin))
    if s <= 0:
        return None
    alpha = 1.0 + n / s
    st = np.sort(tail)
    emp = np.arange(1, n + 1) / n
    fit = 1.0 - (st / xmin) ** (1.0 - alpha)
    ks = float(np.max(np.abs(emp - fit)))
    return {"alpha": float(alpha), "xmin": float(xmin), "n_tail": n,
            "ks": ks}


def _xmin_grid(sizes, max_candidates):
    lo = np.min(sizes)
    hi = np.quantile(sizes, 0.9)
    if hi <= lo:
        return np.array([lo])
    cand = np.geomspace(lo, hi, max_candidates)
    # log-spaced synthetic candidates, deduplicated; NOT snapped to
    # observed sizes (an earlier comment here said otherwise).  The
    # standard observed-value scan lives in
    # reanalysis.fit_powerlaw_observed and is reported alongside.
    return np.unique(cand)


def fit_powerlaw(sizes, max_candidates=100, min_tail=10):
    """CSN fit: alpha by MLE at each candidate xmin (log-spaced grid up
    to the 90th percentile), xmin by KS minimization; candidates whose
    tail is smaller than min_tail are skipped."""
    sizes = np.asarray(sizes, float)
    sizes = sizes[sizes > 0]
    if len(sizes) < min_tail:
        return None
    best = None
    for xmin in _xmin_grid(sizes, max_candidates):
        f = _alpha_ks(sizes, xmin)
        if f is None or f["n_tail"] < min_tail:
            continue
        if best is None or f["ks"] < best["ks"]:
            best = f
    return best


def sample_powerlaw(n, alpha, xmin, rng):
    u = rng.random(n)
    return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def gof_pvalue(sizes, fit, n_boot, seed, max_candidates=100):
    """CSN semiparametric bootstrap: each replicate draws body values
    from the empirical body and tail values from the fitted power law,
    is refit with the full xmin scan, and contributes its KS distance;
    p = fraction of replicates with KS >= the data's KS."""
    sizes = np.asarray(sizes, float)
    sizes = sizes[sizes > 0]
    body = sizes[sizes < fit["xmin"]]
    n = len(sizes)
    p_tail = fit["n_tail"] / n
    rng = np.random.default_rng(seed)
    worse = 0
    for _ in range(n_boot):
        m_tail = rng.binomial(n, p_tail)
        parts = []
        if m_tail:
            parts.append(sample_powerlaw(m_tail, fit["alpha"],
                                         fit["xmin"], rng))
        if n - m_tail:
            src = body if len(body) else sizes
            parts.append(rng.choice(src, n - m_tail, replace=True))
        synth = np.concatenate(parts)
        f = fit_powerlaw(synth, max_candidates)
        if f is not None and f["ks"] >= fit["ks"]:
            worse += 1
    return worse / n_boot


def alpha_bootstrap_ci(sizes, xmin, n_boot, seed, alpha_level=0.05):
    """Nonparametric bootstrap of the tail at fixed xmin (declared)."""
    tail = np.asarray(sizes, float)
    tail = tail[tail >= xmin]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        t = rng.choice(tail, len(tail), replace=True)
        s = np.sum(np.log(t / xmin))
        if s > 0:
            stats.append(1.0 + len(t) / s)
    lo, hi = np.quantile(stats, [alpha_level / 2, 1 - alpha_level / 2])
    return float(lo), float(hi)


def _loglik_powerlaw(tail, alpha, xmin):
    return (len(tail) * math.log((alpha - 1.0) / xmin)
            - alpha * float(np.sum(np.log(tail / xmin))))


def _pointwise_ll_powerlaw(tail, alpha, xmin):
    return np.log((alpha - 1.0) / xmin) - alpha * np.log(tail / xmin)


def _vuong(d):
    """Two-sided normal p for the Vuong-normalized log-likelihood
    ratio built from the pointwise differences d."""
    n = len(d)
    r = float(np.sum(d))
    s = float(np.std(d))
    if s == 0 or n < 2:
        return r, 1.0
    z = r / (s * math.sqrt(n))
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return r, p


def vuong_vs_exponential(sizes, alpha, xmin):
    """Power law vs exponential, both supported on [xmin, inf).
    Returns (R, p, winner): R > 0 favors the power law; the comparison
    is decided by the caller's significance level."""
    tail = np.asarray(sizes, float)
    tail = tail[tail >= xmin]
    lam = 1.0 / max(float(np.mean(tail - xmin)), 1e-300)
    d = (_pointwise_ll_powerlaw(tail, alpha, xmin)
         - (math.log(lam) - lam * (tail - xmin)))
    r, p = _vuong(d)
    return r, p, ("powerlaw" if r > 0 else "exponential")


def _lognormal_trunc_ll(logt, mu, sigma, logxmin):
    """Pointwise log-likelihood of the lognormal truncated to
    [xmin, inf), evaluated on log tail values."""
    z = (logxmin - mu) / (sigma * math.sqrt(2.0))
    tail_mass = 0.5 * math.erfc(z)
    if tail_mass <= 0:
        return None
    return (-np.log(np.exp(logt) * sigma * math.sqrt(2 * math.pi))
            - (logt - mu) ** 2 / (2 * sigma ** 2)
            - math.log(tail_mass))


def fit_lognormal_trunc(sizes, xmin):
    """MLE of the [xmin, inf)-truncated lognormal by coarse-to-fine
    grid search over (mu, sigma) around the untruncated estimates."""
    tail = np.asarray(sizes, float)
    tail = tail[tail >= xmin]
    logt = np.log(tail)
    lx = math.log(xmin)
    mu0, s0 = float(np.mean(logt)), max(float(np.std(logt)), 1e-3)
    best = (None, None, -np.inf)
    lo_mu, hi_mu = mu0 - 6 * s0, mu0 + 2 * s0
    lo_s, hi_s = 0.25 * s0, 6 * s0
    for _ in range(3):
        mus = np.linspace(lo_mu, hi_mu, 21)
        sgs = np.linspace(lo_s, hi_s, 21)
        for mu in mus:
            for sg in sgs:
                ll = _lognormal_trunc_ll(logt, mu, sg, lx)
                if ll is None:
                    continue
                tot = float(np.sum(ll))
                if tot > best[2]:
                    best = (mu, sg, tot)
        dm, ds = (hi_mu - lo_mu) / 10, (hi_s - lo_s) / 10
        lo_mu, hi_mu = best[0] - dm, best[0] + dm
        lo_s, hi_s = max(best[1] - ds, 1e-4), best[1] + ds
    return best[0], best[1]


def vuong_vs_lognormal(sizes, alpha, xmin):
    """Power law vs truncated lognormal on [xmin, inf).  Returns
    (R, p, winner): R > 0 favors the power law."""
    tail = np.asarray(sizes, float)
    tail = tail[tail >= xmin]
    mu, sg = fit_lognormal_trunc(tail, xmin)
    logt = np.log(tail)
    ll_ln = _lognormal_trunc_ll(logt, mu, sg, math.log(xmin))
    d = _pointwise_ll_powerlaw(tail, alpha, xmin) - ll_ln
    r, p = _vuong(d)
    return r, p, ("powerlaw" if r > 0 else "lognormal")


def ccdf_slope(sizes, central=0.8):
    """OLS slope of log10 CCDF against log10 size over the central
    fraction of the sorted sizes (the exploratory pass's statistic)."""
    s = np.sort(np.asarray(sizes, float))
    n = len(s)
    if n < 10:
        return None
    ccdf = 1.0 - np.arange(n) / n
    lo = int(n * (1 - central) / 2)
    hi = int(n * (1 + central) / 2)
    x = np.log10(s[lo:hi])
    y = np.log10(ccdf[lo:hi])
    good = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[good], y[good], 1)[0])


# ---------------------------------------------------------------------------
# surrogates (rank-remapped to the real marginals)


def _rank_remap(z, y):
    """Reorder the real values y according to the ranks of z."""
    out = np.empty_like(y)
    out[np.argsort(z)] = np.sort(y)
    return out


def fit_ar1_log(y):
    z = np.log(np.asarray(y, float))
    z = z - z.mean()
    denom = float(np.dot(z[:-1], z[:-1]))
    phi = float(np.dot(z[1:], z[:-1]) / denom) if denom > 0 else 0.0
    phi = min(max(phi, -0.999), 0.999)
    resid = z[1:] - phi * z[:-1]
    return phi, float(np.std(resid))


def surrogate_ar1(y, rng):
    y = np.asarray(y, float)
    phi, s = fit_ar1_log(y)
    n = len(y)
    z = np.empty(n)
    z[0] = rng.normal(0, s / math.sqrt(max(1 - phi * phi, 1e-6)))
    eps = rng.normal(0, s, n)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + eps[i]
    return _rank_remap(z, y)


def surrogate_phase(y, rng):
    y = np.asarray(y, float)
    z = np.log(y)
    f = np.fft.rfft(z - z.mean())
    phases = rng.uniform(0, 2 * math.pi, len(f))
    phases[0] = 0.0
    if len(z) % 2 == 0:
        phases[-1] = 0.0
    zs = np.fft.irfft(np.abs(f) * np.exp(1j * phases), n=len(z))
    return _rank_remap(zs, y)


def surrogate_stats(y, kind, n_surr, mult, win, warmup, seed,
                    dur_mult=None):
    """Run n_surr surrogates of y through the extraction (and, at
    dur_mult if given, the size-duration) pipeline.  Returns a dict of
    per-surrogate arrays: n_events, smax, alpha, cv, gamma."""
    rng = np.random.default_rng(seed)
    gen = {"ar1": surrogate_ar1, "phase": surrogate_phase}[kind]
    out = {"n_events": [], "smax": [], "alpha": [], "cv": [],
           "gamma": []}
    for _ in range(n_surr):
        ys = gen(y, rng)
        ev, _ = extract_events(ys, mult, win, warmup)
        sizes = np.array([e["size"] for e in ev])
        out["n_events"].append(len(ev))
        out["smax"].append(float(sizes.max()) if len(sizes) else 0.0)
        f = fit_powerlaw(sizes) if len(sizes) >= 10 else None
        out["alpha"].append(f["alpha"] if f else np.nan)
        out["cv"].append(interevent_cv(ev) if len(ev) >= 3 else np.nan)
        if dur_mult is not None:
            evd, _ = extract_events(ys, dur_mult, win, warmup)
            g = size_duration(evd)
            out["gamma"].append(g["gamma"] if g else np.nan)
        else:
            out["gamma"].append(np.nan)
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------------------
# temporal and cross-block statistics


def interevent_cv(events):
    starts = np.array([e["start"] for e in events], float)
    if len(starts) < 3:
        return None
    dt = np.diff(np.sort(starts))
    m = float(dt.mean())
    return float(dt.std() / m) if m > 0 else None


def cv_bootstrap_ci(events, n_boot, seed, alpha_level=0.05):
    starts = np.sort(np.array([e["start"] for e in events], float))
    dt = np.diff(starts)
    if len(dt) < 3:
        return None
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        d = rng.choice(dt, len(dt), replace=True)
        m = d.mean()
        if m > 0:
            stats.append(d.std() / m)
    lo, hi = np.quantile(stats, [alpha_level / 2, 1 - alpha_level / 2])
    return float(lo), float(hi)


def rank_average(x):
    """Average ranks (ties share the mean of the ranks they occupy),
    1-based.  Equal to argsort-of-argsort + 1 when all values are
    distinct; tie-correct otherwise."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def size_duration(events, n_boot=0, seed=0, alpha_level=0.05):
    """OLS of ln size on ln duration over events with dur >= 2; returns
    gamma, its bootstrap CI when n_boot > 0, the tie-aware Spearman
    rho, the event count, and the count of distinct duration values.

    Correction (post wave-8 freeze): the Spearman rho originally ranked
    by argsort(argsort(x)), which assigns arbitrary distinct ranks
    within ties; integer durations are heavily tied, and on the first
    archived constant-rate cell the tie-broken value 0.763 falls to
    0.520 under average ranks (n = 70).  The tie-aware value is the
    reported one; the frozen gate reports are not rewritten (no gate
    verdict conditions on the Spearman value).  Regression fixture in
    tests/test_avalanche.py."""
    ev = [e for e in events if e["dur"] >= 2 and e["size"] > 0]
    if len(ev) < 3:
        return None
    ld = np.log([e["dur"] for e in ev])
    ls = np.log([e["size"] for e in ev])
    gamma = float(np.polyfit(ld, ls, 1)[0])
    rd, rs = rank_average(ld), rank_average(ls)
    if len(np.unique(rd)) < 2 or len(np.unique(rs)) < 2:
        rho = float("nan")     # a constant rank vector has no rank
    else:                      # correlation (all durations tied)
        rho = float(np.corrcoef(rd, rs)[0, 1])
    out = {"gamma": gamma, "spearman": rho, "n": len(ev),
           "n_dur_distinct": int(len(np.unique(ld)))}
    if n_boot:
        rng = np.random.default_rng(seed)
        stats = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(ev), len(ev))
            if len(np.unique(ld[idx])) < 2:
                continue
            stats.append(float(np.polyfit(ld[idx], ls[idx], 1)[0]))
        lo, hi = np.quantile(stats,
                             [alpha_level / 2, 1 - alpha_level / 2])
        out["ci"] = (float(lo), float(hi))
    return out


def omori(events, T, n_main, early, late, n_shift, seed):
    """Aftershock decay: the n_main largest events (skipping any within
    100 steps after a larger one) define mainshocks; the statistic is
    the ratio of the mean event-start rate in (m, m+early] to the rate
    in (m+early, m+late]; the null is n_shift circular shifts of the
    start series."""
    if len(events) < n_main + 3:
        return None
    starts = np.array([e["start"] for e in events])
    order = np.argsort([-e["size"] for e in events])
    mains = []
    for i in order:
        s = events[i]["start"]
        if any(0 < s - m <= 100 for m in mains):
            continue
        mains.append(s)
        if len(mains) == n_main:
            break

    def ratio(st):
        e_cnt = l_cnt = 0
        for m in mains:
            e_cnt += int(np.sum((st > m) & (st <= m + early)))
            l_cnt += int(np.sum((st > m + early) & (st <= m + late)))
        er = e_cnt / (len(mains) * early)
        # one-event floor on the late rate keeps the statistic finite
        # and monotone when the late window is empty
        lr = max(l_cnt, 1) / (len(mains) * (late - early))
        return er / lr

    obs = ratio(starts)
    rng = np.random.default_rng(seed)
    null = [ratio((starts + rng.integers(1, T)) % T + 1)
            for _ in range(n_shift)]
    p = float(np.mean([x >= obs for x in null]))
    return {"ratio": float(obs), "p": p, "mains": mains,
            "null_med": float(np.median(null))}


def coincidence(ev_src, tgt_starts, T, lag):
    """Pinned convention: observed = fraction of source events [s, e]
    with a target-block event start in [s-lag, e+lag]; chance =
    fraction of all steps t with a target start in [t-lag, t+lag]
    (deduplicated occupancy)."""
    if not len(ev_src):
        return None
    tgt = np.asarray(sorted(tgt_starts), int)
    if not len(tgt):
        return None
    hit = sum(1 for e in ev_src
              if np.any((tgt >= e["start"] - lag)
                        & (tgt <= e["end"] + lag)))
    obs = hit / len(ev_src)
    occ = np.zeros(T + 1, bool)
    for s in tgt:
        occ[max(1, s - lag):min(T, s + lag) + 1] = True
    chance = float(occ[1:].mean())
    return {"observed": obs, "chance": chance,
            "ratio": obs / chance if chance > 0 else np.inf,
            "n_src": len(ev_src), "n_tgt": len(tgt)}


def coincidence_shift_p(ev_src, tgt_starts, T, lag, n_shift, seed):
    """Circular-shift null for the observed coincidence rate; returns
    (p, envelope_hi) where envelope_hi is the null's 95% quantile of
    the ratio."""
    base = coincidence(ev_src, tgt_starts, T, lag)
    if base is None:
        return None
    tgt = np.asarray(sorted(tgt_starts), int)
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_shift):
        sh = (tgt + rng.integers(1, T)) % T + 1
        c = coincidence(ev_src, sh, T, lag)
        ratios.append(c["ratio"] if c else np.nan)
    ratios = np.array(ratios)
    p = float(np.nanmean(ratios >= base["ratio"]))
    return {"p": p, "env_hi": float(np.nanquantile(ratios, 0.95)),
            **base}


def coincidence_matrix(events_by_block, T, lag, n_shift=0, seed=0):
    """Ordered-pair coincidence-ratio matrix over the given blocks."""
    out = {}
    for src, ev_s in events_by_block.items():
        for tgt, ev_t in events_by_block.items():
            if src == tgt:
                continue
            starts = [e["start"] for e in ev_t]
            if n_shift:
                # determinism repair (wave 8): the salted Python hash
                # used here made the shift-null seed process-dependent
                # (PYTHONHASHSEED); replaced by a stable CRC32 of the
                # pair name.  Verdicts unaffected.
                pair_seed = zlib.crc32(f"{src}|{tgt}".encode()) % 10000
                out[(src, tgt)] = coincidence_shift_p(
                    ev_s, starts, T, lag, n_shift, seed + pair_seed)
            else:
                out[(src, tgt)] = coincidence(ev_s, starts, T, lag)
    return out


# ---------------------------------------------------------------------------
# rates, stationarity, margins


def rate_series(events, T, win, lo=0):
    """Events per win steps over (lo, T], as (window_starts, rates)."""
    starts = np.array([e["start"] for e in events])
    edges = np.arange(lo, T + 1, win)
    rates = [int(np.sum((starts > a) & (starts <= a + win)))
             for a in edges[:-1]]
    return edges[:-1], np.array(rates, float)


def rate_ci(events, T, block, n_boot, seed, lo=0,
            alpha_level=0.05, per=1000):
    """Moving-block bootstrap CI for the mean event rate per `per`
    steps over (lo, T], resampling the binary start indicator in
    blocks."""
    ind = np.zeros(T - lo)
    for e in events:
        if lo < e["start"] <= T:
            ind[e["start"] - lo - 1] = 1
    m = len(ind)
    if m < block:
        return None
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(m / block))
    stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, m - block + 1, nblocks)
        r = np.concatenate([ind[s:s + block] for s in starts])[:m]
        stats.append(r.mean() * per)
    lo_q, hi_q = np.quantile(stats,
                             [alpha_level / 2, 1 - alpha_level / 2])
    return float(lo_q), float(hi_q)


def phase_splits(onset, T, warmup, descent_len):
    """Stationarity phases for an archived flattening cell."""
    return {"pre_onset": (warmup, onset - 1),
            "descent": (onset, min(onset + descent_len, T)),
            "post_quench": (min(onset + descent_len, T) + 1, T)}


def events_in(events, lo, hi):
    return [e for e in events if lo <= e["start"] <= hi]


def margin_skew(vals):
    """Upper-edge compression A = (q95-q50)/(q50-q05) of ln S."""
    v = np.log(np.asarray(vals, float))
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return None
    q05, q50, q95 = np.quantile(v, [0.05, 0.5, 0.95])
    denom = q50 - q05
    return float((q95 - q50) / denom) if denom > 0 else np.inf


def margin_relaxation(vals, horizon):
    """Excursions above the window q95 and whether each returns below
    the window q50 within `horizon` subsequent probes.  Returns
    (n_excursions, fraction_relaxed, median_return_probes)."""
    v = np.asarray(vals, float)
    good = np.isfinite(v)
    v = v[good]
    if len(v) < 10:
        return None
    q50, q95 = np.quantile(v, [0.5, 0.95])
    n_exc, relaxed, rets = 0, 0, []
    i = 0
    while i < len(v):
        if v[i] > q95:
            n_exc += 1
            ret = None
            for j in range(i + 1, min(i + horizon + 1, len(v))):
                if v[j] < q50:
                    ret = j - i
                    break
            if ret is not None:
                relaxed += 1
                rets.append(ret)
            else:
                # look past the horizon for the median-return report
                for j in range(i + 1, len(v)):
                    if v[j] < q50:
                        rets.append(j - i)
                        break
                else:
                    rets.append(horizon * 4)
            while i < len(v) and v[i] > q50:
                i += 1
        i += 1
    if n_exc == 0:
        return {"n_exc": 0, "frac_relaxed": None, "med_return": None}
    return {"n_exc": n_exc, "frac_relaxed": relaxed / n_exc,
            "med_return": float(np.median(rets))}
