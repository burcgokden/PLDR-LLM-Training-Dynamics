#!/usr/bin/env python3
"""Measure one live matched intervention arm for confirmation experiment E7."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import campaign_design  # noqa: E402
from campaign_record import strict_dumps  # noqa: E402


def _load(value):
    if isinstance(value, dict):
        result = value
    else:
        with Path(value).open("r", encoding="utf-8") as stream:
            result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError("E7 input artifact must be a JSON object")
    return result


def _finite(value, name, *, nonnegative=False):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _measurement_values(value):
    rows = value.get("measurements")
    if not isinstance(rows, list):
        raise ValueError("measurement artifact has no measurement array")
    result = {row["name"]: _finite(row["value"], row["name"])
              for row in rows}
    if len(result) != len(rows):
        raise ValueError("measurement artifact contains duplicate names")
    return result


def _measurement(name, value, unit, method):
    return {
        "name": name,
        "value": _finite(value, name),
        "unit": unit,
        "method": method,
        "status": "OBSERVED",
        "reason_code": None,
    }


def _log_rows(run_directory):
    rows = []
    with (Path(run_directory) / "log.jsonl").open(
        "r", encoding="utf-8",
    ) as stream:
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("training log row must be an object")
            rows.append(value)
    return rows


def _threshold(values, rule):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("avalanche reference signal is missing or nonfinite")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(
        mad,
        float(rule["minimum_relative_scale"]) * abs(median),
        float(np.finfo(float).tiny),
    )
    return math.nextafter(
        median + float(rule["threshold_mad_multiplier"]) * scale,
        math.inf,
    )


def measure_avalanches(*, source_run_directory, branch_run_directory,
                       source_step, outcome_step):
    design = campaign_design.E7_INTERVENTIONS
    detector = design["avalanche_detector"]
    reference_length = int(detector["reference_window_updates"])
    branch_length = int(design["avalanche_window_updates"])
    if outcome_step - source_step != branch_length:
        raise ValueError("E7 outcome and source clocks have wrong separation")
    source_rows = {
        int(row["step"]): row for row in _log_rows(source_run_directory)
        if "loss" in row and int(row.get("step", -1)) <= source_step
    }
    branch_rows = {
        int(row["step"]): row for row in _log_rows(branch_run_directory)
        if "loss" in row
    }
    reference_steps = list(range(
        source_step - reference_length + 1, source_step + 1))
    outcome_steps = list(range(source_step + 1, outcome_step + 1))
    if any(step not in source_rows for step in reference_steps):
        raise ValueError("E7 source avalanche reference window is incomplete")
    if any(step not in branch_rows for step in outcome_steps):
        raise ValueError("E7 branch avalanche window is incomplete")

    signal_name = detector["signal"]
    signal_threshold = _threshold(
        [source_rows[step][signal_name] for step in reference_steps],
        detector,
    )
    support_names = list(detector["support_signals"])
    support_thresholds = {
        name: _threshold(
            [source_rows[step][name] for step in reference_steps],
            detector,
        )
        for name in support_names
    }
    signal = np.asarray([
        _finite(branch_rows[step][signal_name], signal_name, nonnegative=True)
        for step in outcome_steps
    ])
    active = signal > signal_threshold
    events = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            stop = index if not value else index + 1
            indices = range(start, stop)
            size = sum(signal[item] / signal_threshold for item in indices)
            support = sum(
                any(
                    _finite(
                        branch_rows[outcome_steps[item]][name],
                        name, nonnegative=True,
                    ) > support_thresholds[name]
                    for item in indices
                )
                for name in support_names
            )
            events.append({
                "start_step": outcome_steps[start],
                "end_step": outcome_steps[stop - 1],
                "duration": stop - start,
                "size": float(size),
                "support": int(support),
            })
            start = None
    return {
        "event_rate": len(events) / branch_length,
        "mean_size": (
            float(np.mean([row["size"] for row in events]))
            if events else 0.0
        ),
        "mean_duration": (
            float(np.mean([row["duration"] for row in events]))
            if events else 0.0
        ),
        "mean_support": (
            float(np.mean([row["support"] for row in events]))
            if events else 0.0
        ),
        "threshold": signal_threshold,
        "support_thresholds": support_thresholds,
        "events": events,
        "detector": detector,
    }


def _terminal_loss(run_directory, outcome_step):
    candidates = []
    for row in _log_rows(run_directory):
        if int(row.get("step", -1)) != outcome_step:
            continue
        if row.get("event") == "terminal_validation" and "val_loss" in row:
            candidates.append(float(row["val_loss"]))
        elif "loss" in row and "val_loss" in row:
            candidates.append(float(row["val_loss"]))
    if len(candidates) != 1:
        raise ValueError("E7 branch must have exactly one terminal validation")
    return _finite(candidates[0], "heldout_loss", nonnegative=True)


def measure_live_e7(*, e0_measurement, e2_measurements, e3_measurements,
                    e4_measurements, e6_measurement,
                    source_run_directory, branch_run_directory,
                    source_step, outcome_step, intervention_dose, arm):
    design = campaign_design.E7_INTERVENTIONS
    if arm not in design["arms"]:
        raise ValueError("unknown E7 intervention arm")
    expected = dict(zip(
        design["source_checkpoints"], design["outcome_checkpoints"]))
    if expected.get(int(source_step)) != int(outcome_step):
        raise ValueError("E7 source/outcome pair is not registered")
    e0 = _measurement_values(_load(e0_measurement))
    e2 = [_measurement_values(_load(value)) for value in e2_measurements]
    e3 = [_measurement_values(_load(value)) for value in e3_measurements]
    e4 = [_measurement_values(_load(value)) for value in e4_measurements]
    e6 = _measurement_values(_load(e6_measurement))
    if not e2 or not (len(e2) == len(e3) == len(e4)):
        raise ValueError("E7 criterion-layer artifacts must be aligned")
    avalanche = measure_avalanches(
        source_run_directory=source_run_directory,
        branch_run_directory=branch_run_directory,
        source_step=int(source_step),
        outcome_step=int(outcome_step),
    )
    values = {
        "robust_q": max(row["structured_family_q"] for row in e3),
        "complete_disturbance_envelope": max(
            row["disturbance_envelope"] for row in e4),
        "predicted_entry_step": e6["predicted_entry_step"],
        "direct_row_map_upper": e0["direct_row_map_upper"],
        "heldout_loss": _terminal_loss(
            branch_run_directory, int(outcome_step)),
        "intervention_dose": intervention_dose,
        "jury_lower_margin": min(
            row["jury_lower_margin"] for row in e2),
        "jury_upper_margin": min(
            row["jury_upper_margin"] for row in e2),
        "spectral_loading": max(row["spectral_loading"] for row in e2),
        "schedule_product_convolution_upper":
            e6["schedule_product_convolution_upper"],
        "avalanche_event_rate": avalanche["event_rate"],
        "avalanche_mean_size": avalanche["mean_size"],
        "avalanche_mean_support": avalanche["mean_support"],
    }
    units = {
        "robust_q": "H_gain",
        "complete_disturbance_envelope": "lifted_H_unit",
        "predicted_entry_step": "global_optimizer_step",
        "direct_row_map_upper": "output_row_per_input_row",
        "heldout_loss": "nats_per_token",
        "intervention_dose": "learning_rate_multiplier",
        "jury_lower_margin": "dimensionless",
        "jury_upper_margin": "dimensionless",
        "spectral_loading": "dimensionless",
        "schedule_product_convolution_upper": "lifted_H_unit",
        "avalanche_event_rate": "events_per_update",
        "avalanche_mean_size": "threshold_normalized_gradient_sum",
        "avalanche_mean_support": "parameter_blocks",
    }
    measurements = [
        _measurement(
            name, values[name], units[name],
            "live_matched_branch_certificate_and_robust_mad_events",
        )
        for name in values
    ]
    return {
        "schema_version": "pldr-live-e7-measurement-v1",
        "protocol_id": "E7",
        "arm": arm,
        "source_step": int(source_step),
        "outcome_step": int(outcome_step),
        "intervention_rule": design["arms"][arm],
        "avalanche": avalanche,
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    result = measure_live_e7(**value)
    Path(args.output).write_text(strict_dumps(result), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
