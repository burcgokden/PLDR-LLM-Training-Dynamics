#!/usr/bin/env python3
"""Independent analyzer for chronological all-pairs confirmation ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.chronological_collapse import (  # noqa: E402
    adamw_chronological_direction,
    chronological_pair_certificate,
    unordered_pairs,
    verify_shape_enclosure,
)
from confirm.chronological_confirmation_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    OPTIMIZER,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.confirmation_artifacts import digest_object  # noqa: E402


RECORD_SCHEMA = "pldr-chronological-pair-ledger-v1"
REPORT_SCHEMA = "pldr-chronological-analysis-v1"
LOCK_SCHEMA = "pldr-chronological-construction-lock-v1"


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"record omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def _array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"record omits {name}")
    value = np.asarray(record[name], dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return value


def _integer_array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"record omits {name}")
    value = np.asarray(record[name])
    if value.ndim != ndim or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array")
    return value.astype(np.int64, copy=False)


def _string_array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"record omits {name}")
    value = np.asarray(record[name])
    if value.ndim != ndim or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a string array")
    return value.astype(str, copy=False)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _load_lock(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("constant_enlargement_after_lock") is not False
    ):
        raise ValueError("unknown or mutable chronological construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("chronological construction lock digest does not replay")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_excess(actual: np.ndarray, bound: np.ndarray) -> float:
    scale = np.maximum.reduce((
        np.abs(actual),
        np.abs(bound),
        np.full_like(actual, np.finfo(np.float64).tiny),
    ))
    return float(np.max((actual - bound) / scale, initial=-math.inf))


def analyze_record(
    path: str | Path,
    *,
    pair_chunk_size: int = 32768,
    lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one block ledger without trusting producer summaries."""

    record_path = Path(path).resolve()
    with np.load(record_path, allow_pickle=False) as record:
        schema = _scalar(record, "schema_version", str)
        if schema != RECORD_SCHEMA:
            raise ValueError("unknown chronological ledger schema")
        if _scalar(record, "campaign_id", str) != CAMPAIGN_ID:
            raise ValueError("chronological ledger campaign identity changed")
        role = _scalar(record, "role", str)
        seed = _scalar(record, "seed", int)
        data_offset = _scalar(record, "data_offset_chunks", int)
        step_start = _scalar(record, "step_start", int)
        layer = _scalar(record, "layer", int)
        beta1 = _scalar(record, "beta1", float)
        beta2 = _scalar(record, "beta2", float)
        epsilon = _scalar(record, "epsilon", float)
        checkpoint_sha256 = _scalar(record, "checkpoint_sha256", str)
        registry_sha256 = _scalar(record, "registry_sha256", str)
        lock_sha256 = _scalar(record, "lock_sha256", str)
        shape_safety_factor = _scalar(record, "shape_safety_factor", float)
        numerical_relative_coefficient = _scalar(
            record, "numerical_relative_coefficient", float)
        numerical_absolute_charge = _scalar(
            record, "numerical_absolute_charge", float)

        normalized_rows = _array(record, "normalized_rows", 3)
        gate = _array(record, "gate", 2)
        gradient_history = _array(record, "clipped_gradient_history", 3)
        first_tail = _array(record, "first_moment_tail", 2)
        second_before = _array(record, "second_moment_before", 2)
        row_jvp = _array(record, "normalized_row_jvp", 3)
        row_radius = _array(record, "normalized_row_remainder_radius", 2)
        numerical_charge = _array(record, "numerical_charge", 2)
        learning_rate = _array(record, "learning_rate", 1)
        weight_decay = _array(record, "weight_decay", 1)
        decay_mask = _array(record, "decay_mask", 1)
        optimizer_step_after = _integer_array(
            record, "optimizer_step_after", 1)
        update_batch_sha256 = _string_array(
            record, "update_batch_sha256", 1)
        segment_grid = _array(record, "segment_grid", 1)
        producer_gate_residual = _array(
            record, "gate_update_max_abs_residual", 1)

        transitions = normalized_rows.shape[0] - 1
        vertex_count = normalized_rows.shape[1]
        width = normalized_rows.shape[2]
        if transitions < 1:
            raise ValueError("ledger must contain at least one transition")
        expected = {
            "gate": (transitions + 1, width),
            "gradient_history": (
                transitions, gradient_history.shape[1], width),
            "first_tail": (transitions, width),
            "second_before": (transitions, width),
            "row_jvp": (transitions, vertex_count, width),
            "row_radius": (transitions, vertex_count),
            "learning_rate": (transitions,),
            "weight_decay": (transitions,),
            "decay_mask": (width,),
        }
        actual = {
            "gate": gate.shape,
            "gradient_history": gradient_history.shape,
            "first_tail": first_tail.shape,
            "second_before": second_before.shape,
            "row_jvp": row_jvp.shape,
            "row_radius": row_radius.shape,
            "learning_rate": learning_rate.shape,
            "weight_decay": weight_decay.shape,
            "decay_mask": decay_mask.shape,
        }
        if actual != expected:
            raise ValueError(f"chronological ledger dimensions disagree: {actual}")
        expected_trajectory = [
            row for row in TRAJECTORIES
            if row["role"] == role and int(row["seed"]) == seed
        ]
        allowed_anchors = (
            REGISTRY["development_anchor_steps"]
            if role == "development" else REGISTRY["anchor_steps"]
        )
        if (
            len(expected_trajectory) != 1
            or data_offset
            != int(expected_trajectory[0]["data_offset_chunks"])
            or step_start not in allowed_anchors
            or not 0 <= layer < ARCHITECTURE["layers"]
            or beta1 != OPTIMIZER["beta1"]
            or beta2 != OPTIMIZER["beta2"]
            or epsilon != OPTIMIZER["epsilon"]
            or gradient_history.shape[1] != REGISTRY["history_length"]
            or optimizer_step_after.shape != (transitions,)
            or not np.array_equal(
                optimizer_step_after,
                np.arange(
                    step_start + 1, step_start + transitions + 1,
                    dtype=np.int64,
                ),
            )
            or update_batch_sha256.shape != (transitions,)
            or not all(_is_sha256(value) for value in update_batch_sha256)
            or not _is_sha256(checkpoint_sha256)
            or not _is_sha256(registry_sha256)
            or not np.array_equal(segment_grid, np.asarray([0.0, 0.5, 1.0]))
            or shape_safety_factor < 1.0
            or numerical_relative_coefficient < 0.0
            or numerical_absolute_charge < 0.0
            or producer_gate_residual.shape != (transitions,)
            or np.any(producer_gate_residual < 0.0)
        ):
            raise ValueError("chronological ledger metadata or provenance is invalid")
        if role == "heldout":
            if (
                lock is None
                or lock_sha256 != lock["lock_sha256"]
                or transitions != int(lock["primary_block_length"])
                or gradient_history.shape[1] != int(lock["history_length"])
                or not np.array_equal(
                    segment_grid, np.asarray(lock["segment_grid"], dtype=np.float64))
                or shape_safety_factor != float(lock["shape_safety_factor"])
                or numerical_relative_coefficient
                != float(lock["numerical_relative_coefficient"])
                or numerical_absolute_charge
                != float(lock["numerical_absolute_charge"])
            ):
                raise ValueError("held-out ledger is not bound to the construction lock")
        elif lock is not None or lock_sha256:
            raise ValueError(
                "development and construction ledgers must precede the lock")
        pairs = unordered_pairs(vertex_count)
        pair_count = len(pairs)
        if numerical_charge.shape not in {
            (transitions, pair_count), (transitions, vertex_count), (transitions, 1),
        }:
            raise ValueError(
                "numerical_charge must be per step, per row, or per step-pair"
            )
        if np.any(row_radius < 0.0) or np.any(numerical_charge < 0.0):
            raise ValueError("remainder and numerical charges must be nonnegative")
        if pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be positive")

        envelope = np.empty(pair_count, dtype=np.float64)
        cumulative_margin = np.zeros(pair_count, dtype=np.float64)
        source_pair_energy = np.empty(pair_count, dtype=np.float64)
        diameter_bound = []
        observed_diameter = []
        chronology_residual = 0.0
        gate_identity_residual = 0.0
        native_gate_successor_residual = 0.0
        shape_excess = -math.inf
        endpoint_excess = -math.inf
        shape_covered = True
        endpoint_covered = True
        float64_replay_tolerance = 128.0 * np.finfo(np.float64).eps

        for start in range(0, pair_count, pair_chunk_size):
            stop = min(start + pair_chunk_size, pair_count)
            chunk = pairs[start:stop]
            contrast = (
                normalized_rows[0, chunk[:, 0]]
                - normalized_rows[0, chunk[:, 1]]
            )
            energy = np.einsum(
                "pd,d,pd->p", contrast, gate[0] * gate[0], contrast)
            source_pair_energy[start:stop] = energy
            envelope[start:stop] = energy
        diameter_bound.append(float(np.sqrt(np.max(envelope))))
        observed_diameter.append(diameter_bound[0])

        step_summaries = []
        for step in range(transitions):
            chronology = adamw_chronological_direction(
                gradient_history[step],
                first_tail[step],
                second_before[step],
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                optimizer_step_after=step_start + step + 1,
            )
            chronology_residual = max(
                chronology_residual,
                chronology["reconstruction_relative_residual"],
            )
            step_min_margin = math.inf
            step_positive = 0
            step_observed_max = 0.0
            predicted_gate_next = None
            for start in range(0, pair_count, pair_chunk_size):
                stop = min(start + pair_chunk_size, pair_count)
                chunk = pairs[start:stop]
                left = chunk[:, 0]
                right = chunk[:, 1]
                contrast = (
                    normalized_rows[step, left]
                    - normalized_rows[step, right]
                )
                directional = row_jvp[step, left] - row_jvp[step, right]
                radius = row_radius[step, left] + row_radius[step, right]
                if numerical_charge.shape[1] == pair_count:
                    local_numerical_charge = numerical_charge[step, start:stop]
                elif numerical_charge.shape[1] == vertex_count:
                    local_numerical_charge = (
                        numerical_charge[step, left]
                        + numerical_charge[step, right]
                    )
                else:
                    local_numerical_charge = np.full(
                        stop - start, numerical_charge[step, 0])
                certificate = chronological_pair_certificate(
                    gate[step],
                    contrast,
                    chronology["direction"],
                    directional,
                    radius,
                    local_numerical_charge,
                    learning_rate=learning_rate[step],
                    weight_decay=weight_decay[step],
                    decay_mask=decay_mask,
                )
                if predicted_gate_next is None:
                    predicted_gate_next = certificate["gate_next"]
                gate_identity_residual = max(
                    gate_identity_residual,
                    float(np.max(
                        certificate["identity_relative_residual"],
                        initial=0.0,
                    )),
                )
                realized_increment = (
                    normalized_rows[step + 1, left]
                    - normalized_rows[step + 1, right]
                    - contrast
                )
                shape_check = verify_shape_enclosure(
                    contrast,
                    realized_increment,
                    certificate,
                    certificate["gate_next"],
                )
                chunk_shape_excess = _relative_excess(
                    shape_check["realized_shape_work"],
                    certificate["shape_work_upper"],
                )
                shape_excess = max(shape_excess, chunk_shape_excess)
                shape_covered = (
                    shape_covered
                    and bool(
                        chunk_shape_excess <= float64_replay_tolerance)

                )

                target_contrast = (
                    normalized_rows[step + 1, left]
                    - normalized_rows[step + 1, right]
                )
                realized_energy = np.einsum(
                    "pd,d,pd->p",
                    target_contrast,
                    gate[step + 1] * gate[step + 1],
                    target_contrast,
                )
                predicted = (
                    certificate["affine_factor"] * envelope[start:stop]
                    + certificate["affine_forcing"]
                )
                chunk_endpoint_excess = _relative_excess(
                    realized_energy, predicted)
                endpoint_excess = max(
                    endpoint_excess, chunk_endpoint_excess)
                endpoint_covered = (
                    endpoint_covered
                    and bool(
                        chunk_endpoint_excess <= float64_replay_tolerance)

                )
                envelope[start:stop] = predicted
                margin = certificate["chronological_margin_lower"]
                cumulative_margin[start:stop] += margin
                step_min_margin = min(
                    step_min_margin, float(np.min(margin)))
                step_positive += int(np.count_nonzero(margin > 0.0))
                step_observed_max = max(
                    step_observed_max,
                    float(np.max(realized_energy, initial=0.0)),
                )
            if predicted_gate_next is None:
                raise RuntimeError("pair replay produced no gate successor")
            gate_scale = max(
                float(np.linalg.norm(predicted_gate_next)),
                float(np.linalg.norm(gate[step + 1])),
                np.finfo(np.float64).tiny,
            )
            native_gate_successor_residual = max(
                native_gate_successor_residual,
                float(np.linalg.norm(
                    predicted_gate_next - gate[step + 1])) / gate_scale,
            )
            diameter_bound.append(float(np.sqrt(np.max(envelope))))
            observed_diameter.append(float(np.sqrt(step_observed_max)))
            step_summaries.append({
                "step": step_start + step,
                "minimum_pair_margin": step_min_margin,
                "positive_pair_fraction": step_positive / pair_count,
                "diameter_bound": diameter_bound[-1],
                "observed_diameter": observed_diameter[-1],
            })

    simultaneous_block_margin = float(np.min(cumulative_margin))
    bound_contracts = diameter_bound[-1] < diameter_bound[0]
    all_checks = {
        "chronological_moment_replay": chronology_residual <= 1e-12,
        "executed_gate_identity": gate_identity_residual <= 1e-12,
        "native_gate_successor_replay": bool(
            native_gate_successor_residual
            <= 128.0 * np.finfo(np.float32).eps),
        "directional_shape_enclosure": shape_covered,
        "all_pairs_successor_coverage": endpoint_covered,
        "simultaneous_block_margin_positive": simultaneous_block_margin > 0.0,
        "direct_diameter_envelope_contracts": bound_contracts,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "source": {
            "path": str(record_path),
            "sha256": _sha256(record_path),
        },
        "role": role,
        "seed": seed,
        "data_offset_chunks": data_offset,
        "step_start": step_start,
        "step_stop": step_start + transitions,
        "layer": layer,
        "vertex_count": vertex_count,
        "pair_count": pair_count,
        "history_length": int(gradient_history.shape[1]),
        "checkpoint_sha256": checkpoint_sha256,
        "registry_sha256": registry_sha256,
        "lock_sha256": lock_sha256 or None,
        "chronology_relative_residual_max": chronology_residual,
        "gate_identity_relative_residual_max": gate_identity_residual,
        "native_gate_successor_relative_residual_max": (
            native_gate_successor_residual),
        "producer_gate_successor_max_abs_residual_max": float(
            np.max(producer_gate_residual, initial=0.0)),
        "shape_relative_excess_max": shape_excess,
        "endpoint_relative_excess_max": endpoint_excess,
        "simultaneous_block_margin_lower": simultaneous_block_margin,
        "positive_block_margin_pair_fraction": float(
            np.mean(cumulative_margin > 0.0)),
        "source_diameter": diameter_bound[0],
        "endpoint_diameter_bound": diameter_bound[-1],
        "endpoint_diameter_observed": observed_diameter[-1],
        "diameter_contraction_fraction_bound": (
            diameter_bound[-1] / max(
                diameter_bound[0], np.finfo(np.float64).tiny)),
        "checks": all_checks,
        "step_summaries": step_summaries,
        "decision": "QUALIFIED" if all(all_checks.values()) else "NOT_QUALIFIED",
    }
    return report


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pair-chunk-size", type=int, default=32768)
    parser.add_argument("--lock")
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = analyze_record(
        arguments.record,
        pair_chunk_size=arguments.pair_chunk_size,
        lock=_load_lock(arguments.lock),
    )
    write_json(arguments.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "minimum_margin": report["simultaneous_block_margin_lower"],
        "diameter_fraction": report["diameter_contraction_fraction_bound"],
    }, sort_keys=True))
    if arguments.require_pass and report["decision"] != "QUALIFIED":
        raise SystemExit("chronological confirmation record did not qualify")


if __name__ == "__main__":
    main()
