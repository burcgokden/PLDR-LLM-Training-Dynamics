#!/usr/bin/env python3
"""Analyze source-frozen causal row-map ledgers without trusting summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.causal_confirmation_specs import OPTIMIZER  # noqa: E402
from confirm.causal_row_map import stream_registry_prediction  # noqa: E402
from confirm.chronological_collapse import (  # noqa: E402
    adamw_chronological_direction,
)


LEDGER_SCHEMA = "pldr-causal-row-ledger-v1"
REPORT_SCHEMA = "pldr-causal-row-analysis-v1"
CERTIFICATE_SCHEMA = "pldr-causal-remainder-certificate-v1"


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"ledger omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def _array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"ledger omits {name}")
    value = np.asarray(record[name], dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return np.ascontiguousarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def analyze_gate_route(
    gate_source: Any,
    gate_predicted: Any,
    clipped_gradient_history: Any,
    first_moment_tail: Any,
    second_moment_before: Any,
    *,
    beta1: float,
    beta2: float,
    epsilon: float,
    learning_rate: float,
    weight_decay: float,
    decay_mask: Any,
    optimizer_step_after: int,
) -> dict[str, Any]:
    """Replay and enclose the one-step source-state AdamW gate route."""

    source = _array({"value": gate_source}, "value", 1)
    predicted = _array({"value": gate_predicted}, "value", 1)
    history = _array(
        {"value": clipped_gradient_history}, "value", 2)
    tail = _array({"value": first_moment_tail}, "value", 1)
    second = _array({"value": second_moment_before}, "value", 1)
    mask = _array({"value": decay_mask}, "value", 1)
    width = source.size
    if (
        predicted.shape != source.shape
        or history.shape[1] != width
        or tail.shape != source.shape
        or second.shape != source.shape
        or mask.shape != source.shape
    ):
        raise ValueError("gate-route dimensions disagree")
    beta1 = float(beta1)
    beta2 = float(beta2)
    epsilon = float(epsilon)
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    if (
        not np.isfinite([
            beta1, beta2, epsilon, learning_rate, weight_decay,
        ]).all()
        or learning_rate < 0.0
        or weight_decay < 0.0
        or np.any((mask < 0.0) | (mask > 1.0))
    ):
        raise ValueError("gate-route scalars or decay mask are invalid")

    chronology = adamw_chronological_direction(
        history,
        tail,
        second,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        optimizer_step_after=int(optimizer_step_after),
    )
    current_gradient = history[0]
    first_after = chronology["first_moment_after"]
    if beta1 == 0.0:
        first_before = np.zeros_like(first_after)
    else:
        first_before = (
            first_after - (1.0 - beta1) * current_gradient) / beta1
    moment_upper = (
        beta1 * np.abs(first_before)
        + (1.0 - beta1) * np.abs(current_gradient)
    )
    first_bias = 1.0 - beta1 ** int(optimizer_step_after)
    innovation_upper = (
        learning_rate * moment_upper / (epsilon * first_bias)
    )
    decay_multiplier = 1.0 - learning_rate * weight_decay * mask
    adaptive_innovation = -learning_rate * chronology["direction"]
    replayed = decay_multiplier * source + adaptive_innovation
    gate_upper = (
        np.abs(decay_multiplier) * np.abs(source) + innovation_upper
    )
    scale = max(
        1.0,
        float(np.max(np.abs(predicted))),
        float(np.max(np.abs(replayed))),
        float(np.max(np.abs(adaptive_innovation))),
        float(np.max(gate_upper)),
    )
    tolerance = 4096.0 * float(np.finfo(np.float64).eps) * scale
    checks = {
        "chronological_direction_reconstructs": bool(
            chronology["reconstruction_relative_residual"]
            <= 4096.0 * float(np.finfo(np.float64).eps)),
        "first_moment_is_enclosed": bool(np.all(
            np.abs(first_after) <= moment_upper + tolerance)),
        "adaptive_innovation_is_enclosed": bool(np.all(
            np.abs(adaptive_innovation) <= innovation_upper + tolerance)),
        "predicted_gate_replays": bool(np.all(
            np.abs(predicted - replayed) <= tolerance)),
        "predicted_gate_is_enclosed": bool(np.all(
            np.abs(predicted) <= gate_upper + tolerance)),
    }
    return {
        "history_length": int(history.shape[0]),
        "optimizer_step_after": int(optimizer_step_after),
        "decay_multiplier_minimum": float(np.min(decay_multiplier)),
        "decay_multiplier_maximum": float(np.max(decay_multiplier)),
        "maximum_first_moment_upper": float(np.max(moment_upper)),
        "maximum_adaptive_innovation": float(
            np.max(np.abs(adaptive_innovation))),
        "maximum_adaptive_innovation_upper": float(
            np.max(innovation_upper)),
        "maximum_predicted_gate_magnitude": float(
            np.max(np.abs(predicted))),
        "maximum_predicted_gate_upper": float(np.max(gate_upper)),
        "maximum_gate_replay_absolute_residual": float(
            np.max(np.abs(predicted - replayed))),
        "chronological_reconstruction_relative_residual": float(
            chronology["reconstruction_relative_residual"]),
        "float64_tolerance": tolerance,
        "checks": checks,
        "qualified": bool(all(checks.values())),
    }


def analyze(path: str | Path, *, pair_chunk_size: int) -> dict[str, Any]:
    ledger_path = Path(path).resolve()
    with np.load(ledger_path, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != LEDGER_SCHEMA:
            raise ValueError("unknown causal ledger schema")
        certificate_schema = _scalar(record, "certificate_schema", str)
        certificate_sha256 = _scalar(record, "certificate_sha256", str)
        frozen = _scalar(record, "certificate_frozen_before_successor", bool)
        source_rows = _array(record, "source_rows", 2)
        row_jvp = _array(record, "row_jvp", 2)
        row_radius = _array(record, "row_remainder_radius", 1)
        gate_source = _array(record, "gate_source", 1)
        gate_predicted = _array(record, "gate_predicted", 1)
        gate_history = _array(
            record, "gate_clipped_gradient_history", 2)
        gate_first_tail = _array(record, "gate_first_moment_tail", 1)
        gate_second_before = _array(
            record, "gate_second_moment_before", 1)
        gate_beta1 = _scalar(record, "gate_beta1", float)
        gate_beta2 = _scalar(record, "gate_beta2", float)
        gate_epsilon = _scalar(record, "gate_adam_epsilon", float)
        gate_learning_rate = _scalar(
            record, "gate_learning_rate", float)
        gate_weight_decay = _scalar(record, "gate_weight_decay", float)
        gate_decay_mask = _array(record, "gate_decay_mask", 1)
        gate_step_after = _scalar(
            record, "gate_optimizer_step_after", int)
        successor = (
            _array(record, "successor_rows", 2)
            if "successor_rows" in record else None
        )
        gate_successor = (
            _array(record, "gate_successor", 1)
            if "gate_successor" in record else None
        )
        gate_successor_recorded_residual = (
            _scalar(record, "gate_successor_max_abs_residual", float)
            if "gate_successor_max_abs_residual" in record else None
        )
        gate_native_update_residual = (
            _scalar(record, "gate_native_update_max_abs_residual", float)
            if "gate_native_update_max_abs_residual" in record else None
        )
        gate_successor_tolerance = (
            _scalar(record, "gate_successor_float32_tolerance", float)
            if "gate_successor_float32_tolerance" in record else None
        )
        metadata_json = _scalar(record, "metadata_json", str)
        attached_prediction_sha256 = (
            _scalar(record, "prediction_report_sha256", str)
            if "prediction_report_sha256" in record else None
        )
        successor_chronology_sha256 = (
            _scalar(record, "successor_chronology_sha256", str)
            if "successor_chronology_sha256" in record else None
        )
    if certificate_schema != CERTIFICATE_SCHEMA or not frozen:
        raise ValueError("a source-frozen remainder certificate is required")
    attached = (attached_prediction_sha256, successor_chronology_sha256)
    gate_attachment = (
        gate_successor,
        gate_successor_recorded_residual,
        gate_native_update_residual,
        gate_successor_tolerance,
    )
    if successor is None and any(value is not None for value in attached):
        raise ValueError("source-only ledger carries successor provenance")
    if successor is None and any(value is not None for value in gate_attachment):
        raise ValueError("source-only ledger carries successor gate values")
    if successor is not None and any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in attached
    ):
        raise ValueError("covered ledger has malformed successor provenance")
    if successor is not None and any(
        value is None for value in gate_attachment
    ):
        raise ValueError("covered ledger omits successor gate diagnostics")
    if len(certificate_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in certificate_sha256
    ):
        raise ValueError("certificate_sha256 is malformed")
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError as error:
        raise ValueError("metadata_json is malformed") from error
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must encode an object")
    gate_route = analyze_gate_route(
        gate_source,
        gate_predicted,
        gate_history,
        gate_first_tail,
        gate_second_before,
        beta1=gate_beta1,
        beta2=gate_beta2,
        epsilon=gate_epsilon,
        learning_rate=gate_learning_rate,
        weight_decay=gate_weight_decay,
        decay_mask=gate_decay_mask,
        optimizer_step_after=gate_step_after,
    )
    registered_tolerance = 64.0 * float(np.finfo(np.float64).eps)
    gate_route["checks"].update({
        "registered_beta1": bool(abs(
            gate_beta1 - float(OPTIMIZER["beta1"]))
            <= registered_tolerance),
        "registered_beta2": bool(abs(
            gate_beta2 - float(OPTIMIZER["beta2"]))
            <= registered_tolerance),
        "registered_epsilon": bool(abs(
            gate_epsilon - float(OPTIMIZER["epsilon"]))
            <= registered_tolerance),
        "registered_learning_rate": bool(abs(
            gate_learning_rate - float(OPTIMIZER["learning_rate"]))
            <= registered_tolerance),
        "registered_weight_decay": bool(abs(
            gate_weight_decay - float(OPTIMIZER["weight_decay"]))
            <= registered_tolerance),
        "registered_gate_decay_mask": bool(np.all(gate_decay_mask == 1.0)),
        "metadata_optimizer_step_agrees": bool(
            int(metadata.get("optimizer_step_after", -1))
            == gate_step_after),
    })
    gate_route["qualified"] = bool(all(gate_route["checks"].values()))
    gate_successor_coverage = None
    if gate_successor is not None:
        if gate_successor.shape != gate_predicted.shape:
            raise ValueError("predicted and successor gate widths disagree")
        replayed_residual = float(np.max(np.abs(
            gate_successor - gate_predicted)))
        diagnostic_tolerance = (
            256.0 * float(np.finfo(np.float64).eps)
            * max(1.0, replayed_residual))
        residual_replays = bool(abs(
            replayed_residual - float(gate_successor_recorded_residual)
        ) <= diagnostic_tolerance)
        gate_successor_coverage = {
            "maximum_predicted_to_successor_absolute_residual": (
                replayed_residual),
            "maximum_native_update_absolute_residual": float(
                gate_native_update_residual),
            "float32_tolerance": float(gate_successor_tolerance),
            "recorded_residual_replays": residual_replays,
            "qualified": bool(
                residual_replays
                and replayed_residual <= float(gate_successor_tolerance)
                and float(gate_native_update_residual)
                <= float(gate_successor_tolerance)),
        }
    prediction = stream_registry_prediction(
        source_rows,
        row_jvp,
        row_radius,
        pair_chunk_size=pair_chunk_size,
        successor_rows=successor,
    )
    coverage = prediction["coverage"]
    report = {
        "schema_version": REPORT_SCHEMA,
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "certificate": {
            "schema_version": certificate_schema,
            "sha256": certificate_sha256,
            "frozen_before_successor": frozen,
        },
        "metadata": metadata,
        "successor_attachment": (
            None if successor is None else {
                "prediction_report_sha256": attached_prediction_sha256,
                "successor_chronology_sha256": successor_chronology_sha256,
            }
        ),
        "gate_route": gate_route,
        "gate_successor_coverage": gate_successor_coverage,
        "prediction": prediction,
        "valid": bool(
            gate_route["qualified"]
            and (
                gate_successor_coverage is None
                or gate_successor_coverage["qualified"])
            and (coverage is None or coverage["all_pair_enclosures_hold"])),
    }
    return report


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ledger")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair-chunk-size", type=int, default=32768)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = analyze(
        arguments.ledger, pair_chunk_size=arguments.pair_chunk_size)
    write_json(arguments.output, report)
    print(json.dumps({
        "decision": report["prediction"]["decision"],
        "output": str(Path(arguments.output).resolve()),
        "valid": report["valid"],
    }, sort_keys=True))
    if arguments.require_pass and not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
