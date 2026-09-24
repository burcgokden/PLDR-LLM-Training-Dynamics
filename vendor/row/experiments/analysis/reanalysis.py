"""Post-hoc robustness and sensitivity reanalyses over the archived
logs.  NOTHING in this module is preregistered: every number these
functions produce is labeled a reanalysis in the manuscript and is
reported next to, never in place of, the corresponding frozen
preregistered statistic (whose implementations in avalanche.py and
gates.py are untouched).

Contents:
- trailing_median / extract_events_causal: a causal (one-sided)
  variant of the frozen centered-median event detector.  The frozen
  detector is offline (index i uses up to win//2 future samples); this
  variant uses only samples at or before i, for the offline-detector
  sensitivity pass.
- xmin_grid_observed / fit_powerlaw_observed: the standard CSN
  discrete search with xmin candidates at the unique observed sizes
  (the frozen pipeline scans a log-spaced synthetic grid; both
  selections are reported).
- size_duration_block: the size-duration OLS of ln S on ln D with a
  moving-block bootstrap over the time-ordered event list (the frozen
  pipeline resamples events independently, which ignores the measured
  temporal clustering; both intervals are reported).
- exclude_matched_fast / joint_surrogate_stats /
  fixedmask_surrogate_stats / joint_block_surrogate_stats: the
  recalibrated W8.1 surrogate null.  The frozen envelope pipeline
  (avalanche.surrogate_stats, called by gates.w8_1_gamma) surrogates
  the single target series and never applies the paired-run exclusion,
  while the observed statistic is computed on partner-excluded events,
  so the observed statistic and its null do not pass through the same
  pipeline.  The primary joint null here draws a surrogate of BOTH
  series (each from its own fitted marginal model, independent draws:
  the null hypothesis is independence across runs with each series'
  marginal dependence preserved), passes both through the identical
  event extraction, applies the identical dilated exclusion, and fits
  the same size-duration statistic on the excluded set.  Two
  sensitivity brackets are reported alongside, never averaged in:
  (a) fixed-mask: surrogate target series excluded against the
  OBSERVED partner events (does not model cross-run correlation);
  (b) joint moving-block: the aligned pair resampled with shared
  block boundaries (preserves cross-series structure at the block
  scale).  Consumed by analysis/make_reanalysis_w81.py.

Pure numpy; no file I/O.  Consumed by analysis/make_reanalysis.py and
analysis/make_reanalysis_w81.py.  Tests in tests/test_reanalysis.py.
"""

import numpy as np

import avalanche as av


# ---------------------------------------------------------------------------
# causal detector variant


def trailing_median(y, win):
    """Causal running median: at index i the window is the trailing
    y[max(0, i - win + 1) : i + 1] (no future samples; shrinking at
    the start)."""
    y = np.asarray(y, float)
    n = len(y)
    out = np.empty(n)
    if n >= win:
        sw = np.lib.stride_tricks.sliding_window_view(y, win)
        out[win - 1:] = np.median(sw, axis=1)
        edge = win - 1
    else:
        edge = n
    for i in range(edge):
        out[i] = np.median(y[: i + 1])
    return out


def _events_from_threshold(y, thr, warmup):
    """Maximal contiguous runs with y > thr (the extraction loop of
    avalanche.extract_events, parameterized by the threshold curve so
    the frozen function stays byte-identical)."""
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


def extract_events_causal(y, mult, win, warmup=0):
    """extract_events with the trailing (causal) median in place of the
    centered one; conventions otherwise identical."""
    y = np.asarray(y, float)
    thr = mult * trailing_median(y, win)
    return _events_from_threshold(y, thr, warmup)


# ---------------------------------------------------------------------------
# observed-value xmin scan


def xmin_grid_observed(sizes):
    """Candidate xmin values at the unique observed sizes up to the
    90th percentile (the standard CSN discrete search)."""
    sizes = np.asarray(sizes, float)
    hi = np.quantile(sizes, 0.9)
    cand = np.unique(sizes[sizes <= hi])
    if not len(cand):
        cand = np.array([np.min(sizes)])
    return cand


def fit_powerlaw_observed(sizes, min_tail=10):
    """CSN fit with the xmin scan over unique observed sizes (sizes
    treated as continuous, exactly as in the frozen fit); alpha by MLE
    at each candidate, xmin by KS minimization."""
    sizes = np.asarray(sizes, float)
    sizes = sizes[sizes > 0]
    if len(sizes) < min_tail:
        return None
    best = None
    for xmin in xmin_grid_observed(sizes):
        f = av._alpha_ks(sizes, xmin)
        if f is None or f["n_tail"] < min_tail:
            continue
        if best is None or f["ks"] < best["ks"]:
            best = f
    return best


# ---------------------------------------------------------------------------
# dependence-aware size-duration uncertainty


def size_duration_block(events, blocklen=10, n_boot=1000, seed=0,
                        alpha_level=0.05):
    """Size-duration OLS (same estimand and event filter as the frozen
    avalanche.size_duration) with an overlapping moving-block
    bootstrap over the start-time-ordered event list: blocks of
    `blocklen` consecutive events are resampled, so temporally
    clustered events travel together.  Returns the point estimate,
    the block-bootstrap CI, the block length, and the event count."""
    ev = [e for e in events if e["dur"] >= 2 and e["size"] > 0]
    if len(ev) < 3:
        return None
    ev = sorted(ev, key=lambda e: e["start"])
    ld = np.log([e["dur"] for e in ev])
    ls = np.log([e["size"] for e in ev])
    gamma = float(np.polyfit(ld, ls, 1)[0])
    out = {"gamma": gamma, "n": len(ev), "blocklen": int(blocklen)}
    m = len(ev)
    if n_boot and m >= blocklen:
        rng = np.random.default_rng(seed)
        nblocks = int(np.ceil(m / blocklen))
        starts_max = m - blocklen + 1
        stats = []
        for _ in range(n_boot):
            picks = rng.integers(0, starts_max, nblocks)
            idx = np.concatenate(
                [np.arange(s, s + blocklen) for s in picks])[:m]
            if len(np.unique(ld[idx])) < 2:
                continue
            stats.append(float(np.polyfit(ld[idx], ls[idx], 1)[0]))
        if stats:
            lo, hi = np.quantile(
                stats, [alpha_level / 2, 1 - alpha_level / 2])
            out["ci_block"] = (float(lo), float(hi))
    return out


# ---------------------------------------------------------------------------
# recalibrated W8.1 surrogate null (paired series, identical exclusion)


def exclude_matched_fast(ev_a, ev_b, lag):
    """Vectorized implementation of the primary (dilated-interval)
    rule of avalanche.exclude_matched: A is dropped when some B has
    start_A - lag <= end_B and start_B - lag <= end_A.  Returns
    (kept_events, n_dropped).  Pinned equal to the frozen loop
    implementation by tests/test_reanalysis.py (the surrogate null
    calls this thousands of times per cell; the rule is identical)."""
    if not len(ev_b):
        return list(ev_a), 0
    if not len(ev_a):
        return [], 0
    sa = np.array([e["start"] for e in ev_a])
    ea = np.array([e["end"] for e in ev_a])
    sb = np.array([e["start"] for e in ev_b])
    eb = np.array([e["end"] for e in ev_b])
    matched = ((sa[:, None] - lag <= eb[None, :])
               & (sb[None, :] - lag <= ea[:, None])).any(axis=1)
    kept = [e for e, x in zip(ev_a, matched) if not x]
    return kept, int(matched.sum())


def _pair_sd_stats(ev_t, ev_p, lag):
    """Partner-excluded size-duration statistics of one (target,
    partner) event-list pair: the exact estimand of the observed W8.1
    pipeline (gates._dur_events_excluded followed by
    avalanche.size_duration)."""
    kept, n_dropped = exclude_matched_fast(ev_t, ev_p, lag)
    sd = av.size_duration(kept)
    return {"n_kept": len(kept), "n_dropped": n_dropped,
            "n_dur2": sd["n"] if sd else 0,
            "gamma": sd["gamma"] if sd else np.nan,
            "spearman": sd["spearman"] if sd else np.nan}


def _collect(rows):
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], float) for k in keys}


def joint_surrogate_stats(y_t, y_p, kind, n_surr, mult, win, warmup,
                          lag, seed):
    """Primary joint null: each draw generates a surrogate of both the
    target and the partner series (each from its own fitted marginal
    model; draws independent across series, which is the null
    hypothesis the exclusion is meant to control for), extracts events
    from both with the identical detector, applies the identical
    dilated exclusion of target events against surrogate-partner
    events, and fits the identical size-duration statistic on the
    excluded set.  Returns per-surrogate arrays of n_kept, n_dropped,
    n_dur2, gamma, spearman."""
    rng = np.random.default_rng(seed)
    gen = {"ar1": av.surrogate_ar1, "phase": av.surrogate_phase}[kind]
    rows = []
    for _ in range(n_surr):
        yt = gen(y_t, rng)
        yp = gen(y_p, rng)
        et, _ = av.extract_events(yt, mult, win, warmup)
        ep, _ = av.extract_events(yp, mult, win, warmup)
        rows.append(_pair_sd_stats(et, ep, lag))
    return _collect(rows)


def fixedmask_surrogate_stats(y_t, ev_p_obs, kind, n_surr, mult, win,
                              warmup, lag, seed):
    """Sensitivity bracket (a), fixed mask: surrogate target series
    excluded against the OBSERVED partner events.  Disclosed
    limitation: the surrogate target is independent of the real
    partner by construction, so this bracket does not model cross-run
    correlation; it isolates the pure selection effect of the observed
    mask."""
    rng = np.random.default_rng(seed)
    gen = {"ar1": av.surrogate_ar1, "phase": av.surrogate_phase}[kind]
    rows = []
    for _ in range(n_surr):
        yt = gen(y_t, rng)
        et, _ = av.extract_events(yt, mult, win, warmup)
        rows.append(_pair_sd_stats(et, ev_p_obs, lag))
    return _collect(rows)


def pair_block_resample(y_t, y_p, blocklen, rng):
    """Aligned moving-block resample of the (target, partner) pair:
    block start indices are shared between the two series, so
    cross-series structure survives at the block scale."""
    y_t = np.asarray(y_t, float)
    y_p = np.asarray(y_p, float)
    n = len(y_t)
    nblocks = int(np.ceil(n / blocklen))
    starts = rng.integers(0, n - blocklen + 1, nblocks)
    idx = np.concatenate(
        [np.arange(s, s + blocklen) for s in starts])[:n]
    return y_t[idx], y_p[idx]


def joint_block_surrogate_stats(y_t, y_p, blocklen, n_surr, mult, win,
                                warmup, lag, seed):
    """Sensitivity bracket (b), joint moving block: the aligned pair is
    resampled with shared block boundaries and passed through the
    identical extraction, exclusion, and fit.  Preserves cross-series
    dependence at the block scale (which the primary null deliberately
    destroys) at the cost of breaking long-range temporal order."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_surr):
        yt, yp = pair_block_resample(y_t, y_p, blocklen, rng)
        et, _ = av.extract_events(yt, mult, win, warmup)
        ep, _ = av.extract_events(yp, mult, win, warmup)
        rows.append(_pair_sd_stats(et, ep, lag))
    return _collect(rows)
