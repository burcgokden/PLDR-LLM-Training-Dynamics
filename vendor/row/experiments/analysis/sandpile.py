"""Wave-8 estimators: perturbation-response (susceptibility), temporal
branching, finite-size cutoffs, late-window rates, and declared schedule
shapes.

Pure numpy over already-extracted event lists and raw series; no file
I/O and no imports from gates (gates imports this module).  The wave-7
library (avalanche.py) is frozen; everything new lives here.  Every
numeric convention is declared in thresholds.py (W8_*) and frozen with
wave8_prereg.md.

Step conventions match avalanche.extract_events: series index i holds
step i+1; event "start"/"end" are 1-based inclusive steps.  A pulse at
start t with length L overrides steps [t, t+L-1]; its response window
is steps [t+L, t+L+resp_win-1] (the first post-pulse step opens the
window)."""

import math

import numpy as np

import avalanche as av


# ---------------------------------------------------------------------------
# perturbation-response (W8.3)


def pulse_windows(starts, plen, resp_win):
    """[(lo, hi)] inclusive step spans of the response windows."""
    return [(t + plen, t + plen + resp_win - 1) for t in starts]


def classify_pulse_events(events, starts, plen, resp_win):
    """Split events into direct (start inside a pulse), response (start
    inside a response window), and background."""
    direct, response, background = [], [], []
    wins = pulse_windows(starts, plen, resp_win)
    for e in events:
        s = e["start"]
        if any(t <= s < t + plen for t in starts):
            direct.append(e)
        elif any(lo <= s <= hi for lo, hi in wins):
            response.append(e)
        else:
            background.append(e)
    return direct, response, background


def _window_mass(events, lo_grid, resp_win, span, lo):
    """Summed event size and count per modular response window.  Window
    k opens at modular offset lo_grid[k] (0-based within (lo, lo+span])
    and spans resp_win steps, wrapping inside the span."""
    sizes = np.zeros(len(lo_grid))
    counts = np.zeros(len(lo_grid), int)
    for e in events:
        off = (e["start"] - lo - 1) % span
        for k, w0 in enumerate(lo_grid):
            if (off - w0) % span < resp_win:
                sizes[k] += e["size"]
                counts[k] += 1
    return sizes, counts


def pulse_response(events, starts, plen, resp_win, T, lo):
    """Per-pulse response: summed sizes and counts of events starting
    in each response window.  Only windows fully inside (lo, T] are
    clean; the clean mask is returned."""
    span = T - lo
    grid = [(t + plen - lo - 1) % span for t in starts]
    sizes, counts = _window_mass(events, grid, resp_win, span, lo)
    clean = np.array([t + plen > lo and t + plen + resp_win - 1 <= T
                      for t in starts])
    return {"R": sizes, "n_resp": counts, "clean": clean,
            "windows": pulse_windows(starts, plen, resp_win)}


def placement_null(events, starts, plen, resp_win, T, lo, n_place, seed):
    """Circular-shift null for the pulse grid: n_place seeded uniform
    shifts of the whole grid within (lo, T]; returns the per-shift mean
    window response (sizes) and mean triggered indicator."""
    span = T - lo
    rng = np.random.default_rng(seed)
    base = [(t + plen - lo - 1) % span for t in starts]
    mean_R, trig = [], []
    for _ in range(n_place):
        d = int(rng.integers(1, span))
        grid = [(g + d) % span for g in base]
        sizes, counts = _window_mass(events, grid, resp_win, span, lo)
        mean_R.append(float(sizes.mean()))
        trig.append(float((counts > 0).mean()))
    return {"mean_R": np.array(mean_R), "triggered": np.array(trig)}


def susceptibility(R, chance_mean, dose, n_boot=1000, seed=0,
                   alpha_level=0.05):
    """chi = (mean response - chance) / dose with a bootstrap-over-
    pulses interval.  dose = (pulse_lr - base_lr) * pulse_len."""
    R = np.asarray(R, float)
    if len(R) == 0 or dose <= 0:
        return None
    rng = np.random.default_rng(seed)
    stats = [(rng.choice(R, len(R), replace=True).mean() - chance_mean)
             / dose for _ in range(n_boot)]
    lo, hi = np.quantile(stats, [alpha_level / 2, 1 - alpha_level / 2])
    return {"chi": float((R.mean() - chance_mean) / dose),
            "ci": (float(lo), float(hi)), "mean_R": float(R.mean()),
            "chance": float(chance_mean), "dose": float(dose),
            "n_pulses": int(len(R))}


def _nan_running_median(y, win):
    """NaN-aware centered running median with shrinking edge windows
    (mirrors avalanche.running_median)."""
    y = np.asarray(y, float)
    n = len(y)
    h = win // 2
    out = np.empty(n)
    if n >= win:
        sw = np.lib.stride_tricks.sliding_window_view(y, win)
        out[h:n - h] = np.nanmedian(sw, axis=1)
        edge = min(h, n - h)
    else:
        edge = (n + 1) // 2
    for i in range(edge):
        r = min(h, i, n - 1 - i)
        out[i] = np.nanmedian(y[i - r:i + r + 1])
        j = n - 1 - i
        r = min(h, j, n - 1 - j)
        out[j] = np.nanmedian(y[j - r:j + r + 1])
    return out


def masked_median_events(y, mult, win, warmup, starts, plen):
    """Sensitivity variant of event extraction for pulse cells: the
    running median is computed with the direct-pulse steps masked out
    (NaN), so the threshold cannot be inflated by the pulses
    themselves.  Extraction and event fields mirror
    avalanche.extract_events."""
    y = np.asarray(y, float)
    masked = y.copy()
    for t in starts:
        masked[max(0, t - 1):t + plen - 1] = np.nan
    thr = mult * _nan_running_median(masked, win)
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


# ---------------------------------------------------------------------------
# temporal branching (W8.4)


def branching_sigma(events, lag):
    """Cluster decomposition of event starts: clusters are maximal
    chains with successive start gaps <= lag.  sigma_hat = 1 -
    n_clusters / n_events, the empirical mirror of the mean-field
    identity E[cluster size] = 1 / (1 - sigma).

    Naming note: sigma_hat equals 1 - 1/(mean cluster size), so it is
    below one for every finite nonempty event set BY CONSTRUCTION; it
    is a cluster-link density (edges over nodes of the clustering
    forest), not an estimated offspring mean, and the manuscript now
    names it that.  The alias cluster_link_density below is the
    documented name; this function body is frozen (the archived
    wave-8 gate report was computed from it and stays byte-stable)."""
    starts = np.sort(np.array([e["start"] for e in events], float))
    n = len(starts)
    if n == 0:
        return None
    breaks = int(np.sum(np.diff(starts) > lag))
    n_clusters = breaks + 1
    return {"sigma": float(1 - n_clusters / n),
            "n_events": n, "n_clusters": int(n_clusters),
            "mean_cluster": float(n / n_clusters)}


# documented name for the statistic branching_sigma computes (see its
# naming note); same function object, so behavior cannot drift
cluster_link_density = branching_sigma


def branching_null_poisson(n_events, T, lag, n_place, seed, lo=0):
    """sigma_hat over n_place same-count uniform (Poisson-like)
    placements on (lo, T]."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_place):
        starts = np.sort(rng.integers(lo + 1, T + 1, n_events))
        breaks = int(np.sum(np.diff(starts.astype(float)) > lag))
        out.append(1 - (breaks + 1) / n_events)
    return np.array(out)


def branching_surrogates(y, kind, n_surr, mult, win, warmup, lag, seed):
    """sigma_hat of each surrogate series run through the identical
    extraction + clustering pipeline (kind in {ar1, phase})."""
    rng = np.random.default_rng(seed)
    gen = {"ar1": av.surrogate_ar1, "phase": av.surrogate_phase}[kind]
    out = []
    for _ in range(n_surr):
        ev, _ = av.extract_events(gen(y, rng), mult, win, warmup)
        b = branching_sigma(ev, lag)
        out.append(b["sigma"] if b else np.nan)
    return np.array(out)


def sigma_bootstrap_ci(events, lag, T, block, n_boot, seed, lo=0,
                       alpha_level=0.05):
    """Moving-block bootstrap interval for sigma_hat over the start
    indicator series (block length in steps)."""
    starts = sorted(e["start"] for e in events)
    if len(starts) < 3:
        return None
    ind = np.zeros(T - lo, bool)
    for s in starts:
        if lo < s <= T:
            ind[s - lo - 1] = True
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(len(ind) / block)
    stats = []
    for _ in range(n_boot):
        pieces = [ind[b:b + block] for b in
                  rng.integers(0, max(1, len(ind) - block), n_blocks)]
        bi = np.concatenate(pieces)[:len(ind)]
        st = np.flatnonzero(bi).astype(float)
        if len(st) < 3:
            continue
        breaks = int(np.sum(np.diff(st) > lag))
        stats.append(1 - (breaks + 1) / len(st))
    if len(stats) < 10:
        return None
    lo_q, hi_q = np.quantile(stats,
                             [alpha_level / 2, 1 - alpha_level / 2])
    return (float(lo_q), float(hi_q))


def cross_sigma(events_by_block, T, lag, lo=0):
    """Cross-block offspring excess: for each source block, the mean
    number of other-block event starts in the causal window
    (start, end + lag] per source event, minus the uniform-chance
    expectation.  Returns {src: {mean_offspring, chance, excess}}."""
    span = T - lo
    out = {}
    for src, ev_s in events_by_block.items():
        if not ev_s:
            out[src] = None
            continue
        tgt_starts = []
        n_tgt = 0
        for b, ev in events_by_block.items():
            if b == src:
                continue
            tgt_starts.extend(e["start"] for e in ev)
            n_tgt += len(ev)
        tgt_starts = np.array(sorted(tgt_starts), float)
        count = 0
        wlen_sum = 0
        for e in ev_s:
            wlo, whi = e["start"], e["end"] + lag
            count += int(np.sum((tgt_starts > wlo)
                                & (tgt_starts <= whi)))
            wlen_sum += whi - wlo
        chance = wlen_sum / span * n_tgt / len(ev_s) if span > 0 else 0
        mean_off = count / len(ev_s)
        out[src] = {"mean_offspring": float(mean_off),
                    "chance": float(chance),
                    "excess": float(mean_off - chance),
                    "n_src": len(ev_s), "n_tgt": int(n_tgt)}
    return out


# ---------------------------------------------------------------------------
# finite-size cutoffs (W8.5) and late-window rates (W8.2)


def fss_cutoffs(events, q, n_boot=1000, seed=0, alpha_level=0.05):
    """The q-quantile and maximum of the event-size distribution with a
    bootstrap interval on the quantile."""
    sizes = np.array([e["size"] for e in events], float)
    if len(sizes) < 10:
        return None
    rng = np.random.default_rng(seed)
    qs = [float(np.quantile(rng.choice(sizes, len(sizes), replace=True),
                            q)) for _ in range(n_boot)]
    lo_q, hi_q = np.quantile(qs, [alpha_level / 2, 1 - alpha_level / 2])
    return {"q": float(np.quantile(sizes, q)),
            "q_ci": (float(lo_q), float(hi_q)),
            "smax": float(sizes.max()), "n": int(len(sizes))}


def late_window_rate(events, lo, hi, block, n_boot, seed):
    """Event rate per 1000 steps in (lo, hi] with a moving-block
    bootstrap interval."""
    pe = av.events_in(events, lo + 1, hi)
    return {"n": len(pe), "rate_per_1k": len(pe) / (hi - lo) * 1000,
            "rate_ci": av.rate_ci(pe, hi, block, n_boot, seed, lo=lo)}


# ---------------------------------------------------------------------------
# declared schedule shapes (figures only; mirrors train_run semantics)


def schedule_factors(kind, total, warmup, alpha=0.1, hold=None):
    """Factor sequence of the declared schedules: kind in
    {cosine, hold, const}; index s is the factor applied at optimizer
    step s (LambdaLR epoch)."""
    out = np.empty(total)
    w = max(1, warmup)
    for s in range(total):
        if kind == "const":
            out[s] = min(1.0, (s + 1) / w)
        elif kind == "cosine":
            sf = float(min(s, total))
            if sf <= warmup:
                out[s] = sf / warmup
            else:
                d = (sf - warmup) / (total - warmup)
                out[s] = (1 - alpha) * 0.5 * (1 + math.cos(math.pi * d)) \
                    + alpha
        elif kind == "hold":
            sf = float(min(s, total))
            if sf <= warmup and warmup > 1:
                out[s] = sf / warmup
            elif sf <= hold:
                out[s] = 1.0
            else:
                d = (sf - hold) / (total - hold)
                out[s] = (1 - alpha) * 0.5 * (1 + math.cos(math.pi * d)) \
                    + alpha
        else:
            raise ValueError(kind)
    return out
