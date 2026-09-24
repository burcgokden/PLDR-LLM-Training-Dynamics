#!/usr/bin/env python3
"""Bridge training logs to raw decision records.

The output contains no trusted collapse Boolean or index.  It keeps
global event/reload traces, exact probe times, the within-layer
visited-row singular-max samples, and the declared normalization.
``analysis.theory.derive_capture_endpoint`` is the only endpoint
constructor used by E5, E6, the pilot, and operating-characteristic
simulation.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RUNS = os.path.join(EXP, "runs")
sys.path.insert(0, os.path.join(EXP, "analysis"))

import gates  # noqa: E402
import thresholds as T  # noqa: E402
import theory  # noqa: E402


def load_steps(name, runs=RUNS):
    cfg, steps, final = gates.load_run(name, runs=runs)
    return cfg, sorted(steps, key=lambda d: d["step"]), final


def step_hold(t_probe, v_probe, total):
    """Causal probe-to-global-clock hold on one scalar series.

    Values before the first probe are unobserved and therefore remain
    ``nan``.  Confirmation records require a probe at step one, while
    this helper also preserves the honest missing prefix for diagnostic
    records that start later.
    """
    t_probe = np.asarray(t_probe, dtype=int)
    v_probe = np.asarray(v_probe, dtype=float)
    if t_probe.size == 0:
        raise ValueError("at least one row-map probe is required")
    idx = np.searchsorted(t_probe, np.arange(1, total + 1),
                          side="right") - 1
    observed = idx >= 0
    out = np.full(total, np.nan, dtype=float)
    out[observed] = v_probe[idx[observed]]
    return out


def raw_rowmap_probes(steps):
    """Return strict probe times and a rectangular raw sample cube."""
    times = []
    cubes = []
    shape = None
    for rec in steps:
        if "rowmap_sigma_samples" not in rec:
            continue
        cube = np.asarray(rec["rowmap_sigma_samples"], dtype=float)
        if cube.ndim != 2 or cube.shape[0] < 1 or cube.shape[1] < 1:
            raise ValueError("rowmap_sigma_samples must be layer by row")
        if shape is None:
            shape = cube.shape
        if cube.shape != shape:
            raise ValueError("raw row-map probe shape changed within a run")
        if not np.all(np.isfinite(cube)) or np.any(cube < 0):
            raise ValueError("raw row-map probes must be finite and nonnegative")
        times.append(int(rec["step"]))
        cubes.append(cube)
    if not cubes:
        raise ValueError(
            "run has no rowmap_sigma_samples; legacy summaries cannot "
            "be promoted to a confirmation endpoint")
    return np.asarray(times, dtype=int), np.stack(cubes, axis=0)


def applied_lr(steps, total):
    """Applied rate under the recorded previous-step convention."""
    lr_rec = np.full(total + 1, np.nan)
    pulse = np.zeros(total + 1, dtype=bool)
    for rec in steps:
        step = int(rec["step"])
        if step <= total and "lr" in rec:
            lr_rec[step] = rec["lr"]
            pulse[step] = bool(rec.get("pulse"))
    out = np.full(total, np.nan)
    for step in range(1, total + 1):
        if pulse[step]:
            out[step - 1] = lr_rec[step]
        elif step >= 2 and np.isfinite(lr_rec[step - 1]):
            out[step - 1] = lr_rec[step - 1]
        else:
            out[step - 1] = lr_rec[step]
    return out


def per_step_series(steps, key):
    """Return probe steps and values for one exact raw-record key.

    No fallback or aliasing occurs here.  A missing primary Gauss--Newton
    key therefore remains missing and makes the total E0 record incomplete.
    """
    pairs = [(int(record["step"]), float(record[key]))
             for record in steps if key in record]
    if not pairs:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    pairs.sort(key=lambda pair: pair[0])
    return (np.asarray([pair[0] for pair in pairs], dtype=int),
            np.asarray([pair[1] for pair in pairs], dtype=float))


def run_record(name, delta_J0, runs=RUNS, block="plga", J_crit=0.1,
               normalization=1.0, mult=None, win=None):
    """Build one raw record and a derived diagnostic sidecar."""
    cfg, steps, final = load_steps(name, runs=runs)
    total = max(int(d["step"]) for d in steps)
    events = gates.cell_events(
        name, block=block, mult=mult or T.AV_THR_MULT,
        win=win or T.AV_MED_WIN, runs=runs)["events"]
    word = np.zeros(total)
    for event in events:
        word[event["start"] - 1:event["end"]] = 1.0

    probe_steps, samples = raw_rowmap_probes(steps)
    norm = np.asarray(normalization, dtype=float)
    target = samples.shape[:2]
    if norm.ndim == 0:
        norm_probe_layer = np.full(target, float(norm))
    elif norm.ndim == 1 and norm.size == samples.shape[1]:
        norm_probe_layer = np.broadcast_to(
            norm[None, :], target).astype(float)
    elif norm.ndim == 2 and norm.shape == target:
        norm_probe_layer = norm.astype(float)
    else:
        raise ValueError(
            "normalization must be scalar, per layer, or probe by layer")
    if not np.all(np.isfinite(norm_probe_layer)) or np.any(norm_probe_layer <= 0):
        raise ValueError("normalization must be finite and positive")
    layer_probe = np.median(samples, axis=2) / norm_probe_layer
    all_layer_probe = np.max(layer_probe, axis=1)
    jet = step_hold(probe_steps, all_layer_probe, total)
    lr = applied_lr(steps, total)

    act = np.zeros(total)
    pending = 0
    event_end = {event["end"]: event for event in events}
    for step in range(1, total + 1):
        i = step - 1
        if pending > 0 and word[i] == 0:
            act[i] = pending
            pending -= 1
        if step in event_end:
            event = event_end[step]
            j_pre = jet[max(event["start"] - 2, 0)]
            j_post = jet[min(event["end"], total - 1)]
            eta = lr[i] if np.isfinite(lr[i]) else cfg["lr"]
            if eta > 0 and delta_J0 > 0:
                pending = theory.reload_steps(
                    j_pre, min(j_pre, j_post), eta, delta_J0)

    rec = dict(
        word=word,
        act=act,
        probe_steps=probe_steps,
        rowmap_samples=samples,
        normalization=(float(norm) if norm.ndim == 0
                       else norm.astype(float)),
        n_events=len(events),
        T=total,
    )
    endpoint = theory.derive_capture_endpoint(
        rec, J_crit=J_crit, min_persistence_probes=3,
        where=f"run {name}", require_step_one=True)
    return rec, dict(config=cfg, events=events, endpoint=endpoint,
                     lr=lr, final=final)


if __name__ == "__main__":
    dj = float(sys.argv[1])
    for run_name in sys.argv[2:]:
        record, aux = run_record(run_name, delta_J0=dj)
        print(run_name, dict(
            horizon=len(record["word"]),
            probes=len(record["probe_steps"]),
            endpoint=aux["endpoint"]["status"],
            entry=aux["endpoint"]["entry"],
        ))
