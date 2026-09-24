#!/usr/bin/env python3
"""Independent analyzer for chronological occupied-segment PLGA ledgers."""

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
    rectangular_score_transfer,
)
from confirm.chronological_confirmation_specs import (  # noqa: E402
    CAMPAIGN_ID,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.confirmation_artifacts import digest_object  # noqa: E402
from confirm.gate_shape import plga_secant_chain  # noqa: E402


RECORD_SCHEMA = "pldr-chronological-plga-ledger-v1"
REPORT_SCHEMA = "pldr-chronological-plga-analysis-v1"
LOCK_SCHEMA = "pldr-chronological-construction-lock-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"PLGA ledger omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def _array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"PLGA ledger omits {name}")
    value = np.asarray(record[name], dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return value


def _integer_array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"PLGA ledger omits {name}")
    value = np.asarray(record[name])
    if value.ndim != ndim or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array")
    return value.astype(np.int64, copy=False)


def _string_array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"PLGA ledger omits {name}")
    value = np.asarray(record[name])
    if value.ndim != ndim or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a string array")
    return value.astype(str, copy=False)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _relative_norm(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.linalg.norm(left - right))
    denominator = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(np.float64).tiny,
    )
    return numerator / denominator


def _load_lock(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
    ):
        raise ValueError("unknown chronological construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("chronological construction lock digest does not replay")
    required = (
        "plga_power_secant_bound",
        "plga_transfer_coefficient_bound",
        "centered_logit_response_coefficient",
        "centered_logit_remainder_bound",
    )
    if any(
        not math.isfinite(float(value.get(name, math.nan)))
        or float(value[name]) < (0.0 if name.endswith("remainder_bound") else 1e-300)
        for name in required
    ):
        raise ValueError("chronological PLGA lock constants are invalid")
    return value


def analyze_record(path: str | Path, lock: dict[str, Any] | None) -> dict[str, Any]:
    record_path = Path(path).resolve()
    with np.load(record_path, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != RECORD_SCHEMA:
            raise ValueError("unknown chronological PLGA ledger schema")
        if _scalar(record, "campaign_id", str) != CAMPAIGN_ID:
            raise ValueError("PLGA ledger campaign identity changed")
        role = _scalar(record, "role", str)
        seed = _scalar(record, "seed", int)
        data_offset = _scalar(record, "data_offset_chunks", int)
        anchor = _scalar(record, "anchor", int)
        layer = _scalar(record, "layer", int)
        checkpoint_sha256 = _scalar(record, "checkpoint_sha256", str)
        registry_sha256 = _scalar(record, "registry_sha256", str)
        lock_sha256 = _scalar(record, "lock_sha256", str)
        positive_floor = _scalar(record, "positive_floor", float)
        if (
            role not in {"construction", "heldout"}
            or anchor not in REGISTRY["anchor_steps"]
            or not 0 <= layer < 3
            or positive_floor != 1e-9
            or len(checkpoint_sha256) != 64
            or len(registry_sha256) != 64
        ):
            raise ValueError("PLGA ledger metadata is invalid")
        if role == "heldout":
            if lock is None or lock_sha256 != lock["lock_sha256"]:
                raise ValueError("held-out PLGA ledger is not bound to the lock")
        elif lock is not None or lock_sha256:
            raise ValueError("construction PLGA ledger must precede the lock")

        generator = _array(record, "plga_generator", 3)
        collapsed = _array(
            record, "plga_collapsed_anchor_generator", 3)
        weight = _array(record, "plga_weight", 3)
        bias = _array(record, "plga_bias", 3)
        exponent = _array(record, "plga_exponent", 3)
        coupling = _array(record, "plga_coupling", 3)
        coupling_bias = _array(record, "plga_coupling_bias", 3)
        captured_curvature_reference = _array(
            record, "plga_curvature_left", 3)
        captured_curvature_candidate = _array(
            record, "plga_curvature_right", 3)
        query_reference = _array(record, "plga_reference_query", 3)
        query_candidate = _array(record, "plga_candidate_query", 3)
        key_reference = _array(record, "plga_reference_key", 3)
        key_candidate = _array(record, "plga_candidate_key", 3)
        score_reference = _array(
            record, "plga_reference_pre_mask_score", 3)
        score_candidate = _array(
            record, "plga_candidate_pre_mask_score", 3)
        logits_reference = _array(
            record, "plga_reference_centered_logits", 2)
        logits_candidate = _array(
            record, "plga_candidate_centered_logits", 2)
        cache_residual = _array(
            record, "plga_candidate_cache_replay_residual", 1)
        pair_ids = _string_array(record, "plga_pair_id", 1)
        context_indices = _integer_array(
            record, "plga_context_index", 1)
        heads = _integer_array(record, "plga_head", 1)
        anchor_rows = _integer_array(record, "plga_anchor_row", 1)

    cell_count, rows, width = generator.shape
    expected_contexts = 8 if role == "construction" else 16
    square_shape = (cell_count, width, width)
    expected_prefix = "ccc" if role == "construction" else "cch"
    expected_pair_ids = np.asarray([
        f"{expected_prefix}{index:03d}-l{layer}-h{index % 4}-anchor0"
        for index in range(expected_contexts)
    ])
    matching_trajectory = [
        row for row in TRAJECTORIES
        if row["role"] == role
        and int(row["seed"]) == seed
        and int(row["data_offset_chunks"]) == data_offset
    ]
    if (
        len(matching_trajectory) != 1
        or
        cell_count != expected_contexts
        or rows != width
        or width != 64
        or collapsed.shape != generator.shape
        or any(value.shape != square_shape for value in (
            weight, bias, exponent, coupling, coupling_bias,
            captured_curvature_reference, captured_curvature_candidate,
        ))
        or query_reference.shape != query_candidate.shape
        or key_reference.shape != key_candidate.shape
        or score_reference.shape != score_candidate.shape
        or query_reference.shape[0] != cell_count
        or key_reference.shape[0] != cell_count
        or score_reference.shape[0] != cell_count
        or logits_reference.shape != logits_candidate.shape
        or logits_reference.shape[0] != cell_count
        or cache_residual.shape != (cell_count,)
        or pair_ids.shape != (cell_count,)
        or not np.array_equal(pair_ids, expected_pair_ids)
        or not np.array_equal(
            context_indices, np.arange(cell_count, dtype=np.int64))
        or not np.array_equal(
            heads, np.arange(cell_count, dtype=np.int64) % 4)
        or not np.array_equal(
            anchor_rows, np.zeros(cell_count, dtype=np.int64))
        or not _is_sha256(checkpoint_sha256)
        or not _is_sha256(registry_sha256)
    ):
        raise ValueError("PLGA ledger tensor dimensions disagree")

    float64_tolerance = 1024.0 * np.finfo(np.float64).eps
    float32_tolerance = 32.0 * width * np.finfo(np.float32).eps
    cell_summaries = []
    maxima = {
        "power_secant": 0.0,
        "transfer_coefficient": 0.0,
        "centered_response_ratio": 0.0,
        "score_relative_residual": 0.0,
        "curvature_relative_residual": 0.0,
    }
    all_cells_qualified = True
    for index in range(cell_count):
        replay = plga_secant_chain(
            collapsed[index],
            generator[index],
            weight[index],
            bias[index],
            exponent[index],
            coupling[index],
            coupling_bias[index],
            positive_floor=positive_floor,
        )
        power_max = float(np.max(np.abs(replay["power_secant"])))
        transfer_coefficient = float(replay["operator_bound"])
        differences = (
            generator[index, :, None, :] - generator[index, None, :, :])
        diameter = float(np.max(np.linalg.norm(differences, axis=-1)))
        stack_norm = float(np.linalg.norm(
            generator[index] - collapsed[index]))
        row_bound = math.sqrt(rows) * diameter
        curvature_norm = float(np.linalg.norm(replay["realized_difference"]))
        curvature_bound = transfer_coefficient * row_bound
        curvature_capture_residual = max(
            _relative_norm(
                replay["curvature_left"],
                captured_curvature_reference[index],
            ),
            _relative_norm(
                replay["curvature_right"],
                captured_curvature_candidate[index],
            ),
        )
        score = rectangular_score_transfer(
            query_reference[index],
            query_candidate[index],
            replay["curvature_left"],
            replay["curvature_right"],
            key_reference[index],
            key_candidate[index],
        )
        score_capture_residual = max(
            _relative_norm(score["reference_score"], score_reference[index]),
            _relative_norm(score["candidate_score"], score_candidate[index]),
        )
        centered_reference = (
            logits_reference[index] - np.mean(logits_reference[index]))
        centered_candidate = (
            logits_candidate[index] - np.mean(logits_candidate[index]))
        delta_norm = float(np.linalg.norm(
            centered_candidate - centered_reference))
        response_ratio = delta_norm / max(
            curvature_norm, np.finfo(np.float64).tiny)
        if not all(math.isfinite(value) for value in (
            power_max, transfer_coefficient, diameter, stack_norm,
            curvature_norm, curvature_bound, response_ratio,
        )):
            raise ValueError("PLGA replay produced a nonfinite bound")
        winner = int(np.argmax(centered_reference))
        competitor = np.delete(centered_reference, winner)
        margin = float(centered_reference[winner] - np.max(competitor))
        candidate_winner = int(np.argmax(centered_candidate))

        identity_checks = {
            "positive_occupied_bases": bool(
                np.min(replay["metric_left"]) > 0.0
                and np.min(replay["metric_right"]) > 0.0),
            "signed_power_identity": bool(
                replay["identity_residual"]
                <= float64_tolerance * max(
                    1.0, float(np.linalg.norm(replay["realized_difference"])))),
            "row_diameter_bound": stack_norm <= (
                row_bound + float64_tolerance * max(1.0, row_bound)),
            "curvature_replay": curvature_capture_residual <= float64_tolerance,
            "rectangular_score_identity": (
                score["identity_relative_residual"] <= float64_tolerance),
            "rectangular_score_capture": (
                score_capture_residual <= float32_tolerance),
            "candidate_cache_replay": bool(
                cache_residual[index] <= float32_tolerance),
        }
        margin_invoked = False
        response_bound = math.inf
        if lock is not None:
            locked_curvature_bound = (
                float(lock["plga_transfer_coefficient_bound"]) * row_bound)
            response_bound = (
                float(lock["centered_logit_response_coefficient"])
                * locked_curvature_bound
                + float(lock["centered_logit_remainder_bound"])
            )
            margin_invoked = math.sqrt(2.0) * response_bound < margin
            identity_checks.update({
                "power_secant_lock": (
                    power_max <= float(lock["plga_power_secant_bound"])),
                "transfer_coefficient_lock": (
                    transfer_coefficient
                    <= float(lock["plga_transfer_coefficient_bound"])),
                "curvature_bound": curvature_norm <= (
                    locked_curvature_bound
                    + float64_tolerance * max(1.0, locked_curvature_bound)),
                "centered_logit_coverage": delta_norm <= (
                    response_bound
                    + float32_tolerance * max(1.0, response_bound)),
                "winner_if_margin_invoked": (
                    not margin_invoked or candidate_winner == winner),
            })
        cell_qualified = all(identity_checks.values())
        all_cells_qualified = all_cells_qualified and cell_qualified
        maxima["power_secant"] = max(maxima["power_secant"], power_max)
        maxima["transfer_coefficient"] = max(
            maxima["transfer_coefficient"], transfer_coefficient)
        maxima["centered_response_ratio"] = max(
            maxima["centered_response_ratio"], response_ratio)
        maxima["score_relative_residual"] = max(
            maxima["score_relative_residual"],
            score["identity_relative_residual"],
            score_capture_residual,
        )
        maxima["curvature_relative_residual"] = max(
            maxima["curvature_relative_residual"],
            curvature_capture_residual,
        )
        cell_summaries.append({
            "cell_index": index,
            "row_diameter": diameter,
            "row_stack_norm": stack_norm,
            "row_stack_bound": row_bound,
            "power_secant_max": power_max,
            "transfer_coefficient": transfer_coefficient,
            "curvature_difference_norm": curvature_norm,
            "curvature_bound": curvature_bound,
            "centered_logit_difference_norm": delta_norm,
            "centered_logit_response_ratio": response_ratio,
            "centered_logit_bound": response_bound,
            "reference_margin": margin,
            "margin_invoked": margin_invoked,
            "winner_preserved": candidate_winner == winner,
            "checks": identity_checks,
            "decision": "QUALIFIED" if cell_qualified else "NOT_QUALIFIED",
        })

    return {
        "schema_version": REPORT_SCHEMA,
        "source": {"path": str(record_path), "sha256": _sha256(record_path)},
        "role": role,
        "seed": seed,
        "data_offset_chunks": data_offset,
        "anchor": anchor,
        "layer": layer,
        "checkpoint_sha256": checkpoint_sha256,
        "registry_sha256": registry_sha256,
        "cell_count": cell_count,
        "maxima": maxima,
        "margin_invoked_cell_count": sum(
            int(row["margin_invoked"]) for row in cell_summaries),
        "cell_summaries": cell_summaries,
        "decision": (
            "QUALIFIED" if all_cells_qualified
            else "NOT_QUALIFIED"
        ) if role == "heldout" else (
            "CONSTRUCTION_SUMMARY" if all_cells_qualified
            else "CONSTRUCTION_INVALID"
        ),
    }


def analyze_campaign(
    paths: list[str], lock_path: str | Path | None,
) -> dict[str, Any]:
    lock = _load_lock(lock_path)
    reports = [analyze_record(path, lock) for path in paths]
    if not reports:
        raise ValueError("PLGA analysis needs at least one ledger")
    role = reports[0]["role"]
    if any(row["role"] != role for row in reports):
        raise ValueError("PLGA campaign analysis cannot mix roles")
    cells = {
        (row["seed"], row["data_offset_chunks"], row["anchor"], row["layer"])
        for row in reports
    }
    if len(cells) != len(reports):
        raise ValueError("PLGA campaign contains duplicate cells")
    if role == "construction":
        expected = {
            (8444, 8192, anchor, layer)
            for anchor in REGISTRY["anchor_steps"]
            for layer in range(3)
        }
        complete = cells == expected and len(reports) == 15
        checks = {
            "complete_construction_grid": complete,
            "all_identities_valid": all(
                row["decision"] == "CONSTRUCTION_SUMMARY" for row in reports),
        }
        decision = (
            "CONSTRUCTION_SUMMARY" if all(checks.values())
            else "CONSTRUCTION_INVALID")
    else:
        expected_trajectories = {
            (int(row["seed"]), int(row["data_offset_chunks"]))
            for row in TRAJECTORIES if row["role"] == "heldout"
        }
        expected = {
            (seed, offset, anchor, layer)
            for seed, offset in expected_trajectories
            for anchor in REGISTRY["anchor_steps"]
            for layer in range(3)
        }
        margin_by_seed = {
            seed: sum(
                row["margin_invoked_cell_count"]
                for row in reports if row["seed"] == seed)
            for seed, _offset in expected_trajectories
        }
        checks = {
            "complete_heldout_grid": cells == expected and len(reports) == 60,
            "all_cells_qualified": all(
                row["decision"] == "QUALIFIED" for row in reports),
            "nonempty_margin_set_per_seed": all(
                count > 0 for count in margin_by_seed.values()),
        }
        decision = "QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED"
    return {
        "schema_version": REPORT_SCHEMA,
        "role": role,
        "lock_sha256": lock["lock_sha256"] if lock is not None else None,
        "record_count": len(reports),
        "checks": checks,
        "maxima": {
            name: max(row["maxima"][name] for row in reports)
            for name in reports[0]["maxima"]
        },
        "reports": reports,
        "decision": decision,
    }


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
    parser.add_argument("records", nargs="+")
    parser.add_argument("--lock")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = analyze_campaign(arguments.records, arguments.lock)
    write_json(arguments.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "record_count": report["record_count"],
    }, sort_keys=True))
    if arguments.require_pass and report["decision"] not in {
        "QUALIFIED", "CONSTRUCTION_SUMMARY",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
