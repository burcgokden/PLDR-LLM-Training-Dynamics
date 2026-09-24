"""Outcome-neutral validation for staged predictive-state closure records.

The staged runner measures row maps and source energies.  This module does not
repeat those model forward passes.  It authenticates their sealed artifacts,
reconstructs checkpoint-dependent state, replays the registered embedding
intervention and token selection, derives every decision from detailed rows,
and checks all resource and registry aggregates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import stat
from typing import Any, Iterable

import numpy as np

from .analysis import (
    canonical_json_bytes,
    file_sha256,
    verify_record_seal,
)
from .provenance_v3 import git_blob_descriptor
from .staged_closure import (
    candidate_signatures,
    canonical_transition_state_sha256,
)


CONFIG_SCHEMA = "pldr-row-rg-staged-closure-config-v1"
MANIFEST_SCHEMA = "pldr-row-rg-staged-closure-run-v1"
RESULT_SCHEMA = "pldr-row-rg-staged-closure-analysis-v1"
RELOCATION_SCHEMA = "pldr-row-rg-staged-closure-relocation-v1"
CHECKPOINT_SCHEMA = "pldr-training-checkpoint-v3"
PRODUCER_SCHEMA = "pldr-row-rg-energy-producer-metadata-v3"
TOKEN_SELECTION_RULE = (
    "smallest token in immediate next-batch inputs and absent from "
    "fixed-probe inputs"
)
LEGACY_STAGED_SOURCE_PATHS = (
    "scripts/run_staged_predictive_closure.py",
    "scripts/analyze_staged_predictive_closure.py",
    "scripts/verify_staged_predictive_closure_relocation.py",
    "scripts/run_energy_closure_probe_v3.py",
    "scripts/pldr_energy_producer_v3.py",
    "src/row_rgmap/staged_closure.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/closure_contract.py",
    "src/row_rgmap/provenance_v3.py",
    "src/row_rgmap/recorder.py",
    "configs/staged-predictive-closure-v1.json",
)

STAGED_SOURCE_PATHS = (
    *LEGACY_STAGED_SOURCE_PATHS[:-2],
    "src/row_rgmap/staged_validation.py",
    *LEGACY_STAGED_SOURCE_PATHS[-2:],
)

DECISION_STATUSES = frozenset({"REJECTED", "NOT_REJECTED", "NOT_EVALUABLE"})
PROVENANCE_CHECKPOINT_FIELDS = frozenset({"code_commit", "protocol_sha256"})
STRUCTURAL_CHECK_KEYS = (
    "manifest_seal",
    "source_blobs_at_execution_commit",
    "frozen_config_matches_manifest",
    "source_and_predecessor_descriptors",
    "checkpoint_rewrites_exact",
    "checkpoint_inventory_exact",
    "baseline_replay",
    "gauge_quotient_control",
    "direct_vs_iterated_semigroup",
    "gpu_cap",
)
TRUST_BOUNDARY = {
    "remeasured_by_runner": ["row_maps", "source_energy_vectors"],
    "authenticated_not_remeasured_by_validator": [
        "row_map_sha256",
        "source_remeasurement.npz",
        "continuation_energy_arrays",
    ],
    "replayed_by_validator": [
        "configuration_and_protocol_bindings",
        "resource_arithmetic",
        "checkpoint_interventions",
        "source_null_token_selection",
        "candidate_state_fingerprints",
        "scientific_decisions",
    ],
}


def _fail(label: str, detail: str = "differs") -> None:
    raise ValueError(f"staged validation: {label} {detail}")


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(label, "is not an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(label, "is not a list")
    return value


def _require_exact_keys(value: dict[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    actual = set(value)
    if actual != expected:
        _fail(
            label,
            f"keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}",
        )


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if not _values_equal(actual, expected):
        _fail(label)


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(label, "is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        _fail(label, "is not finite and nonnegative")
    return result


def _same_float(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and math.isclose(
        a, b, rel_tol=1e-13, abs_tol=1e-12
    )


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        _fail(label, "is not a nonempty path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        _fail(label, "is not a safe relative path")
    return path


def resolve_descriptor(root: Path, descriptor: Any, label: str) -> Path:
    item = _require_object(descriptor, label)
    relative = _safe_relative(item.get("path"), f"{label}.path")
    path = root.joinpath(*relative.parts)
    root_resolved = root.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"staged validation: {label} escapes its root") from error
    if path.is_symlink() or not resolved.is_file():
        _fail(label, "does not name a regular non-link file")
    if isinstance(item.get("size_bytes"), bool) or item.get("size_bytes") != resolved.stat().st_size:
        _fail(f"{label}.size_bytes")
    digest = item.get("sha256")
    if not isinstance(digest, str) or file_sha256(resolved) != digest:
        _fail(f"{label}.sha256")
    return resolved


def resolve_locator(data_root: Path, locator: Any, label: str) -> Path:
    item = _require_object(locator, label)
    _require_exact_keys(item, {"anchor", "path"}, label)
    _require_equal(item.get("anchor"), "data_root", f"{label}.anchor")
    relative = _safe_relative(item.get("path"), f"{label}.path")
    resolved = data_root.joinpath(*relative.parts).resolve()
    try:
        resolved.relative_to(data_root.resolve())
    except ValueError as error:
        raise ValueError(f"staged validation: {label} escapes the data root") from error
    if not resolved.is_dir():
        _fail(label, "does not resolve to a directory")
    return resolved


def _anchored_descriptor(data_root: Path, descriptor: Any, label: str) -> Path:
    item = _require_object(descriptor, label)
    if item.get("anchor") != "data_root":
        _fail(f"{label}.anchor")
    reduced = {key: value for key, value in item.items() if key != "anchor"}
    return resolve_descriptor(data_root, reduced, label)


def _is_sign_gauge_parameter(name: str) -> bool:
    return bool(
        name.endswith(".mha1.layernorm1.weight")
        or name.endswith(".mha1.layernorm1.bias")
        or (
            ".mha1.reslayerAs." in name
            and ".denseAs." in name
            and (
                name.endswith("gluw1.weight")
                or name.endswith("gluw2.weight")
                or name.endswith("gluw3.weight")
                or name.endswith("gluw3.bias")
            )
        )
        or name.endswith(".mha1.reslayerAs.0.layernormA.bias")
        or name.endswith(".mha1.reslayerAs.1.layernormA.weight")
    )


def _sign_gauge_layer(name: str) -> int:
    marker = "decoder.dec_layers."
    if marker not in name:
        _fail("sign-gauge parameter", f"has no layer coordinate: {name}")
    try:
        return int(name.split(marker, 1)[1].split(".", 1)[0])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"staged validation: malformed sign-gauge layer in {name}"
        ) from error


def _values_equal(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(
            left, right
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


def differing_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    """Return one path per changed nested leaf, treating tensors as leaves."""

    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        paths: list[str] = []
        keys = sorted(
            set(left).union(right), key=lambda value: (type(value).__name__, repr(value))
        )
        for key in keys:
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(differing_paths(left[key], right[key], child))
        return paths
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return [prefix or "<root>"]
        paths = []
        for index, (a, b) in enumerate(zip(left, right)):
            child = f"{prefix}.{index}" if prefix else str(index)
            paths.extend(differing_paths(a, b, child))
        return paths
    return [] if _values_equal(left, right) else [prefix or "<root>"]


def validate_config_shape(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a staged configuration without assuming a particular outcome."""

    _require_object(config, "config")
    _require_exact_keys(
        config,
        {
            "schema_version",
            "stage_id",
            "source_run",
            "predecessor_closure",
            "source_step",
            "block_sizes",
            "semigroup_split",
            "gpu_hour_cap",
            "embedding_shift",
            "branch_ids",
            "candidate_states",
            "splits",
            "trajectories",
        },
        "config",
    )
    _require_equal(config.get("schema_version"), CONFIG_SCHEMA, "config.schema_version")
    if not isinstance(config.get("stage_id"), str) or not config["stage_id"]:
        _fail("config.stage_id", "is empty")
    for key in ("source_run", "predecessor_closure"):
        item = _require_object(config.get(key), f"config.{key}")
        _require_exact_keys(item, {"anchor", "path"}, f"config.{key}")
        _require_equal(item.get("anchor"), "data_root", f"config.{key}.anchor")
        _safe_relative(item.get("path"), f"config.{key}.path")

    source_step = config.get("source_step")
    if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step < 0:
        _fail("config.source_step", "is not a nonnegative integer")
    scales = _require_list(config.get("block_sizes"), "config.block_sizes")
    if (
        not scales
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in scales)
        or scales != sorted(scales)
        or len(set(scales)) != len(scales)
    ):
        _fail("config.block_sizes", "must be ordered unique positive integers")
    split_step = config.get("semigroup_split")
    if (
        isinstance(split_step, bool)
        or not isinstance(split_step, int)
        or split_step <= 0
        or split_step not in scales
    ):
        _fail("config.semigroup_split", "is outside the block registry")
    cap = _finite_nonnegative(config.get("gpu_hour_cap"), "config.gpu_hour_cap")
    if cap == 0.0:
        _fail("config.gpu_hour_cap", "must be positive")

    shift = _require_object(config.get("embedding_shift"), "config.embedding_shift")
    _require_exact_keys(
        shift,
        {"model_key", "coordinate", "offset", "token_selection_rule"},
        "config.embedding_shift",
    )
    if not isinstance(shift.get("model_key"), str) or not shift["model_key"]:
        _fail("config.embedding_shift.model_key", "is empty")
    if (
        isinstance(shift.get("coordinate"), bool)
        or not isinstance(shift.get("coordinate"), int)
        or shift["coordinate"] < 0
    ):
        _fail("config.embedding_shift.coordinate")
    offset = shift.get("offset")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, (int, float))
        or not math.isfinite(float(offset))
        or float(offset) == 0.0
    ):
        _fail("config.embedding_shift.offset", "must be finite and nonzero")
    _require_equal(
        shift.get("token_selection_rule"),
        TOKEN_SELECTION_RULE,
        "config.embedding_shift.token_selection_rule",
    )

    branches = _require_list(config.get("branch_ids"), "config.branch_ids")
    if (
        not branches
        or any(not isinstance(value, str) or not value for value in branches)
        or len(set(branches)) != len(branches)
    ):
        _fail("config.branch_ids", "is not a unique nonempty registry")
    required_branches = {
        "baseline-direct",
        "gauge-consistent",
        "gauge-params-only",
        "probe-null-embedding-shift",
    }
    if set(branches) != required_branches:
        _fail("config.branch_ids", "omits a required staged branch")

    candidates = _require_list(config.get("candidate_states"), "config.candidate_states")
    candidate_ids: list[str] = []
    for index, row in enumerate(candidates):
        item = _require_object(row, f"config.candidate_states[{index}]")
        _require_exact_keys(
            item,
            {"candidate_id", "challenge_branch", "description"},
            f"config.candidate_states[{index}]",
        )
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            _fail(f"config.candidate_states[{index}].candidate_id")
        if item.get("challenge_branch") not in branches:
            _fail(f"config.candidate_states[{index}].challenge_branch")
        if not isinstance(item.get("description"), str) or not item["description"]:
            _fail(f"config.candidate_states[{index}].description")
        candidate_ids.append(candidate_id)
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        _fail("config.candidate_states", "does not have unique candidates")

    trajectories = _require_list(config.get("trajectories"), "config.trajectories")
    trajectory_ids: list[str] = []
    for index, row in enumerate(trajectories):
        item = _require_object(row, f"config.trajectories[{index}]")
        _require_exact_keys(item, {"trajectory_id", "device"}, f"config.trajectories[{index}]")
        trajectory_id = item.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            _fail(f"config.trajectories[{index}].trajectory_id")
        if isinstance(item.get("device"), bool) or not isinstance(item.get("device"), int) or item["device"] < 0:
            _fail(f"config.trajectories[{index}].device")
        trajectory_ids.append(trajectory_id)
    if not trajectory_ids or len(set(trajectory_ids)) != len(trajectory_ids):
        _fail("config.trajectories", "does not have unique trajectories")

    splits = _require_object(config.get("splits"), "config.splits")
    if set(splits) != {"design", "holdout"}:
        _fail("config.splits", "must contain design and holdout")
    flattened: list[str] = []
    for name in ("design", "holdout"):
        values = _require_list(splits[name], f"config.splits.{name}")
        if not values or any(not isinstance(value, str) for value in values):
            _fail(f"config.splits.{name}")
        flattened.extend(values)
    if len(set(flattened)) != len(flattened) or set(flattened) != set(trajectory_ids):
        _fail("config.splits", "is not a partition of trajectories")
    return config


def _split_by_trajectory(config: dict[str, Any]) -> dict[str, str]:
    return {
        trajectory_id: split
        for split, trajectory_ids in config["splits"].items()
        for trajectory_id in trajectory_ids
    }


def _validate_defect_row(row: Any, scale: int, label: str, *, conditional: bool) -> None:
    item = _require_object(row, label)
    keys = {
        "block_size",
        "map_count",
        "bitwise_different_map_count",
        "maximum_absolute_energy_defect",
        "maximum_symmetric_relative_energy_defect",
    }
    if conditional:
        keys.add("conditional_equality_prediction_rejected")
    _require_exact_keys(item, keys, label)
    _require_equal(item.get("block_size"), scale, f"{label}.block_size")
    count = item.get("map_count")
    different = item.get("bitwise_different_map_count")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count <= 0
        or isinstance(different, bool)
        or not isinstance(different, int)
        or different < 0
        or different > count
    ):
        _fail(label, "has invalid map counts")
    absolute = _finite_nonnegative(
        item.get("maximum_absolute_energy_defect"),
        f"{label}.maximum_absolute_energy_defect",
    )
    relative = _finite_nonnegative(
        item.get("maximum_symmetric_relative_energy_defect"),
        f"{label}.maximum_symmetric_relative_energy_defect",
    )
    if different == 0 and (absolute != 0.0 or relative != 0.0):
        _fail(label, "has nonzero maxima with zero changed maps")
    if conditional and type(
        item["conditional_equality_prediction_rejected"]
    ) is not bool:
        _fail(f"{label}.conditional_equality_prediction_rejected")


def _count_word(value: int) -> str:
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return words.get(value, str(value))


def derive_scientific_payload(
    config: dict[str, Any],
    candidate_trajectories: dict[str, list[dict[str, Any]]],
    baseline_trajectories: list[dict[str, Any]],
    gauge_trajectories: list[dict[str, Any]],
    semigroup_trajectories: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive every staged scientific aggregate from detailed rows."""

    validate_config_shape(config)
    scales = list(config["block_sizes"])
    trajectory_ids = [row["trajectory_id"] for row in config["trajectories"]]
    split_lookup = _split_by_trajectory(config)
    candidates = {
        row["candidate_id"]: row for row in config["candidate_states"]
    }
    if set(candidate_trajectories) != set(candidates):
        _fail("candidate trajectory registry")

    def ordered_rows(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for row in _require_list(rows, label):
            item = _require_object(row, label)
            trajectory_id = item.get("trajectory_id")
            if trajectory_id in by_id:
                _fail(label, f"contains duplicate trajectory {trajectory_id}")
            by_id[trajectory_id] = item
        if set(by_id) != set(trajectory_ids):
            _fail(label, "trajectory set differs")
        return [by_id[value] for value in trajectory_ids]

    baseline_rows = ordered_rows(baseline_trajectories, "baseline control")
    for row in baseline_rows:
        trajectory_id = row["trajectory_id"]
        _require_exact_keys(
            row,
            {"trajectory_id", "split", "all_steps_bitwise_equal", "state_count", "map_count"},
            f"baseline control {trajectory_id}",
        )
        _require_equal(row["split"], split_lookup[trajectory_id], f"baseline {trajectory_id}.split")
        if type(row["all_steps_bitwise_equal"]) is not bool:
            _fail(f"baseline {trajectory_id}.all_steps_bitwise_equal")
        if any(
            isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] <= 0
            for key in ("state_count", "map_count")
        ):
            _fail(f"baseline {trajectory_id}", "has invalid counts")
    baseline_passed = all(row["all_steps_bitwise_equal"] for row in baseline_rows)

    gauge_rows = ordered_rows(gauge_trajectories, "gauge control")
    for row in gauge_rows:
        trajectory_id = row["trajectory_id"]
        _require_exact_keys(
            row,
            {
                "trajectory_id",
                "split",
                "source_canonical_transition_state_equal",
                "successor_scales",
                "final_canonical_transition_state_equal",
            },
            f"gauge control {trajectory_id}",
        )
        _require_equal(row["split"], split_lookup[trajectory_id], f"gauge {trajectory_id}.split")
        for key in (
            "source_canonical_transition_state_equal",
            "final_canonical_transition_state_equal",
        ):
            if type(row[key]) is not bool:
                _fail(f"gauge {trajectory_id}.{key}")
        scale_rows = _require_list(row["successor_scales"], f"gauge {trajectory_id}.successor_scales")
        if len(scale_rows) != len(scales):
            _fail(f"gauge {trajectory_id}.successor_scales")
        for scale, scale_row in zip(scales, scale_rows):
            _validate_defect_row(scale_row, scale, f"gauge {trajectory_id} scale {scale}", conditional=False)
    gauge_source_eligible = all(
        row["source_canonical_transition_state_equal"] for row in gauge_rows
    )
    gauge_passed = bool(
        gauge_source_eligible
        and all(
            row["final_canonical_transition_state_equal"]
            and all(scale["bitwise_different_map_count"] == 0 for scale in row["successor_scales"])
            for row in gauge_rows
        )
    )
    gauge_status = (
        "NOT_EVALUABLE"
        if not gauge_source_eligible
        else ("NOT_REJECTED" if gauge_passed else "REJECTED")
    )

    semigroup_rows = ordered_rows(semigroup_trajectories, "semigroup control")
    expected_restart = int(config["source_step"]) + int(config["semigroup_split"])
    for row in semigroup_rows:
        trajectory_id = row["trajectory_id"]
        _require_exact_keys(
            row,
            {
                "trajectory_id",
                "split",
                "restart_step",
                "tail_energy_arrays_bitwise_equal",
                "final_canonical_transition_state_equal",
            },
            f"semigroup control {trajectory_id}",
        )
        _require_equal(row["split"], split_lookup[trajectory_id], f"semigroup {trajectory_id}.split")
        _require_equal(row["restart_step"], expected_restart, f"semigroup {trajectory_id}.restart_step")
        for key in (
            "tail_energy_arrays_bitwise_equal",
            "final_canonical_transition_state_equal",
        ):
            if type(row[key]) is not bool:
                _fail(f"semigroup {trajectory_id}.{key}")
    semigroup_passed = all(
        row["tail_energy_arrays_bitwise_equal"]
        and row["final_canonical_transition_state_equal"]
        for row in semigroup_rows
    )

    controls_passed = baseline_passed and gauge_passed and semigroup_passed
    candidate_results = []
    promoted: list[str] = []
    for candidate_id, specification in candidates.items():
        rows = ordered_rows(candidate_trajectories[candidate_id], f"candidate {candidate_id}")
        for row in rows:
            trajectory_id = row["trajectory_id"]
            _require_exact_keys(
                row,
                {
                    "trajectory_id",
                    "split",
                    "challenge_branch",
                    "source_candidate_bitwise_equal",
                    "scales",
                },
                f"candidate {candidate_id} trajectory {trajectory_id}",
            )
            _require_equal(row["split"], split_lookup[trajectory_id], f"candidate {candidate_id} {trajectory_id}.split")
            _require_equal(
                row["challenge_branch"],
                specification["challenge_branch"],
                f"candidate {candidate_id} {trajectory_id}.challenge_branch",
            )
            if type(row["source_candidate_bitwise_equal"]) is not bool:
                _fail(f"candidate {candidate_id} {trajectory_id}.source_candidate_bitwise_equal")
            scale_rows = _require_list(row["scales"], f"candidate {candidate_id} {trajectory_id}.scales")
            if len(scale_rows) != len(scales):
                _fail(f"candidate {candidate_id} {trajectory_id}.scales")
            for scale, scale_row in zip(scales, scale_rows):
                label = f"candidate {candidate_id} {trajectory_id} scale {scale}"
                _validate_defect_row(scale_row, scale, label, conditional=True)
                expected_rejection = bool(
                    row["source_candidate_bitwise_equal"]
                    and scale_row["bitwise_different_map_count"] > 0
                )
                _require_equal(
                    scale_row["conditional_equality_prediction_rejected"],
                    expected_rejection,
                    f"{label}.conditional_equality_prediction_rejected",
                )

        split_results: dict[str, dict[str, Any]] = {}
        for split in ("design", "holdout"):
            selected = [row for row in rows if row["split"] == split]
            expected_ids = config["splits"][split]
            if [row["trajectory_id"] for row in selected] != expected_ids:
                _fail(f"candidate {candidate_id} {split} trajectory order")
            source_exact = all(row["source_candidate_bitwise_equal"] for row in selected)
            rejected_scales = [
                scale
                for scale in scales
                if any(
                    next(item for item in row["scales"] if item["block_size"] == scale)[
                        "conditional_equality_prediction_rejected"
                    ]
                    for row in selected
                )
            ]
            status = (
                "REJECTED"
                if source_exact and rejected_scales
                else ("NOT_REJECTED" if source_exact else "NOT_EVALUABLE")
            )
            split_results[split] = {
                "trajectory_count": len(selected),
                "source_candidate_bitwise_equal": source_exact,
                "conditional_prediction_executed": source_exact,
                "rejected_scales": rejected_scales,
                "status": status,
            }
        replicated = bool(
            split_results["design"]["status"] == "REJECTED"
            and split_results["holdout"]["status"] == "REJECTED"
        )
        statuses = {
            split_results["design"]["status"],
            split_results["holdout"]["status"],
        }
        status = (
            "REJECTED"
            if replicated
            else ("NOT_EVALUABLE" if "NOT_EVALUABLE" in statuses else "NOT_REJECTED")
        )
        promotion_passed = bool(
            all(
                split_results[name]["status"] == "NOT_REJECTED"
                for name in ("design", "holdout")
            )
            and controls_passed
        )
        if promotion_passed:
            promoted.append(candidate_id)
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "challenge_branch": specification["challenge_branch"],
                "status": status,
                "design_holdout_replication": replicated,
                "splits": split_results,
                "trajectories": rows,
                "promotion_gate_passed": promotion_passed,
            }
        )

    stage_status = "PROMOTED_TO_LOCAL_FLOW" if promoted else "COMPLETED_STOPPED_AT_CLOSURE"
    if promoted:
        stop_reason = None
    elif all(row["status"] == "REJECTED" for row in candidate_results):
        stop_reason = (
            "Every prespecified reduced candidate was rejected in both the "
            "design and held-out trajectories."
        )
    elif not controls_passed:
        stop_reason = "At least one required baseline, quotient, or semigroup control failed."
    elif any(row["status"] == "NOT_EVALUABLE" for row in candidate_results):
        stop_reason = "At least one prespecified reduced candidate was not evaluable on both frozen splits."
    else:
        stop_reason = "No prespecified reduced candidate passed the promotion gate on both frozen splits."

    return {
        "source_step": int(config["source_step"]),
        "block_sizes": scales,
        "splits": config["splits"],
        "candidate_results": candidate_results,
        "controls": {
            "baseline_replay": {"passed": baseline_passed, "trajectories": baseline_rows},
            "complete_state_gauge_quotient": {
                "status": gauge_status,
                "passed": gauge_passed,
                "trajectories": gauge_rows,
            },
            "direct_vs_iterated_semigroup": {
                "passed": semigroup_passed,
                "trajectories": semigroup_rows,
            },
        },
        "promotion": {
            "stage_status": stage_status,
            "promoted_candidate_ids": promoted,
            "local_reduced_flow_authorized": bool(promoted),
            "critical_surface_authorized": False,
            "critical_exponent_authorized": False,
            "empirical_universality_authorized": False,
            "stop_reason": stop_reason,
        },
        "conclusion": {
            "scale_16_executed": 16 in scales,
            "held_out_conditional_predictions_executed": all(
                row["splits"]["holdout"]["conditional_prediction_executed"]
                for row in candidate_results
            ),
            "direct_vs_iterated_agreement_tested": len(semigroup_rows) == len(trajectory_ids),
            "reduced_predictive_state_identified": bool(promoted),
            "complete_state_quotient_control_status": gauge_status,
            "scope": (
                f"{_count_word(len(trajectory_ids))} source checkpoints, exact represented "
                f"candidate fibers, {_count_word(len(scales))} finite block scales, and "
                "frozen design/holdout splits"
            ),
        },
    }


def validate_scientific_payload(
    payload: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Recompute all aggregates and require byte-level JSON agreement."""

    scientific = _require_object(payload, "scientific_payload")
    _require_exact_keys(
        scientific,
        {"source_step", "block_sizes", "splits", "candidate_results", "controls", "promotion", "conclusion"},
        "scientific_payload",
    )
    candidates = _require_list(scientific.get("candidate_results"), "scientific_payload.candidate_results")
    expected_ids = [row["candidate_id"] for row in config["candidate_states"]]
    if [row.get("candidate_id") for row in candidates if isinstance(row, dict)] != expected_ids:
        _fail("scientific_payload.candidate_results", "registry or order differs")
    candidate_rows = {
        row["candidate_id"]: row.get("trajectories") for row in candidates
    }
    controls = _require_object(scientific.get("controls"), "scientific_payload.controls")
    _require_exact_keys(
        controls,
        {"baseline_replay", "complete_state_gauge_quotient", "direct_vs_iterated_semigroup"},
        "scientific_payload.controls",
    )
    derived = derive_scientific_payload(
        config,
        candidate_rows,
        _require_object(controls["baseline_replay"], "baseline control").get("trajectories"),
        _require_object(controls["complete_state_gauge_quotient"], "gauge control").get("trajectories"),
        _require_object(controls["direct_vs_iterated_semigroup"], "semigroup control").get("trajectories"),
    )
    if not _values_equal(scientific, derived):
        for key in derived:
            if not _values_equal(scientific.get(key), derived[key]):
                _fail(f"scientific_payload.{key}", "does not replay from detailed rows")
        _fail("scientific_payload", "does not replay")
    return derived


def validate_resource_accounting(
    manifest: dict[str, Any], config: dict[str, Any], record: dict[str, Any] | None = None
) -> dict[str, float | bool]:
    """Derive GPU totals from route and producer leaves."""

    source = _require_object(manifest.get("source_remeasurement"), "manifest.source_remeasurement")
    source_seconds = 0.0
    for trajectory in _require_list(source.get("trajectories"), "manifest.source_remeasurement.trajectories"):
        row = _require_object(trajectory, "source remeasurement trajectory")
        routes = [_require_object(row.get("original_route"), "original route")]
        routes.extend(
            _require_object(item, "source branch").get("route")
            for item in _require_list(row.get("branches"), "source branches")
        )
        for route_value in routes:
            route = _require_object(route_value, "source route")
            wall = _finite_nonnegative(route.get("wall_seconds"), "source route wall_seconds")
            timing = _require_object(route.get("timing"), "source route timing")
            if not _same_float(timing.get("total_route_elapsed_seconds"), wall):
                _fail("source route timing total")
            source_seconds += wall
    if not _same_float(source.get("cuda_wall_seconds"), source_seconds):
        _fail("manifest.source_remeasurement.cuda_wall_seconds")

    continuation_seconds = 0.0
    rows = [
        row
        for trajectory_rows in _require_object(manifest.get("branches"), "manifest.branches").values()
        for row in _require_list(trajectory_rows, "manifest branch rows")
    ]
    rows.extend(
        row
        for trajectory_rows in _require_object(manifest.get("semigroup_branches"), "manifest.semigroup_branches").values()
        for row in _require_list(trajectory_rows, "manifest semigroup rows")
    )
    for row in rows:
        continuation_seconds += _finite_nonnegative(
            _require_object(row, "continuation row").get("wall_seconds"),
            "continuation wall_seconds",
        )

    accounting = _require_object(manifest.get("gpu_time_accounting"), "manifest.gpu_time_accounting")
    _require_exact_keys(
        accounting,
        {
            "source_remeasurement_cuda_wall_seconds",
            "continuation_producer_reported_wall_seconds",
            "cap_gpu_hours",
            "cap_passed",
        },
        "manifest.gpu_time_accounting",
    )
    if not _same_float(accounting.get("source_remeasurement_cuda_wall_seconds"), source_seconds):
        _fail("manifest.gpu_time_accounting.source_remeasurement_cuda_wall_seconds")
    if not _same_float(accounting.get("continuation_producer_reported_wall_seconds"), continuation_seconds):
        _fail("manifest.gpu_time_accounting.continuation_producer_reported_wall_seconds")
    cap = _finite_nonnegative(config.get("gpu_hour_cap"), "config.gpu_hour_cap")
    if not _same_float(accounting.get("cap_gpu_hours"), cap):
        _fail("manifest.gpu_time_accounting.cap_gpu_hours")
    seconds = source_seconds + continuation_seconds
    hours = seconds / 3600.0
    cap_passed = hours <= cap
    if not _same_float(manifest.get("gpu_seconds"), seconds):
        _fail("manifest.gpu_seconds")
    if not _same_float(manifest.get("gpu_hours"), hours):
        _fail("manifest.gpu_hours")
    if accounting.get("cap_passed") is not cap_passed:
        _fail("manifest.gpu_time_accounting.cap_passed")
    if record is not None:
        _require_equal(record.get("gpu_time_accounting"), accounting, "result.gpu_time_accounting")
        if not _same_float(record.get("gpu_seconds"), seconds):
            _fail("result.gpu_seconds")
        if not _same_float(record.get("gpu_hours"), hours):
            _fail("result.gpu_hours")
    return {
        "source_remeasurement_cuda_wall_seconds": source_seconds,
        "continuation_producer_reported_wall_seconds": continuation_seconds,
        "gpu_seconds": seconds,
        "gpu_hours": hours,
        "cap_gpu_hours": cap,
        "cap_passed": cap_passed,
    }


def _option(command: list[Any], name: str) -> str:
    positions = [index for index, value in enumerate(command) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        _fail("source producer command", f"lacks unique option {name}")
    value = command[positions[0] + 1]
    if not isinstance(value, str):
        _fail("source producer command option", name)
    return value


def _read_byte_tokens(dataset: Any, start: int, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    raw = bytearray()
    index = start
    while len(raw) < count:
        if index >= len(dataset):
            _fail("RefinedWeb stream", "ended before the requested prefix")
        content = dataset[index]["content"]
        if not isinstance(content, str):
            _fail(f"RefinedWeb row {index}", "has no string content")
        raw.extend(content.encode("utf-8", errors="strict"))
        raw.append(10)
        index += 1
    selected = bytes(raw[:count])
    values = np.frombuffer(selected, dtype=np.uint8).astype(np.int64) + 1
    return values, {
        "first_document": start,
        "last_document_exclusive": index,
        "document_count": index - start,
        "token_count": count,
        "byte_token_sha256": hashlib.sha256(selected).hexdigest(),
        "streaming": True,
        "maximum_buffer_policy": "one current UTF-8 document",
    }


def replay_source_null_selections(
    manifest: dict[str, Any], source_config: dict[str, Any], refinedweb_root: Path
) -> bool:
    """Rebuild each selected token from the bound probe and next batch."""

    from datasets import Dataset

    command = _require_list(source_config.get("producer_command"), "source producer_command")
    width = int(_option(command, "--context-length"))
    batch_size = int(_option(command, "--batch-size"))
    contexts = int(_option(command, "--contexts"))
    tokens_per_update = batch_size * (width + 1)
    source = _require_object(manifest.get("source_remeasurement"), "manifest.source_remeasurement")
    dataset_descriptor = _require_object(source.get("dataset"), "manifest.source_remeasurement.dataset")
    relative = _safe_relative(dataset_descriptor.get("relative_path"), "dataset.relative_path")
    dataset_path = refinedweb_root.joinpath(*relative.parts).resolve()
    try:
        dataset_path.relative_to(refinedweb_root.resolve())
    except ValueError as error:
        raise ValueError("staged validation: dataset path escapes refinedweb root") from error
    if (
        not dataset_path.is_file()
        or dataset_path.is_symlink()
        or dataset_path.stat().st_size != dataset_descriptor.get("size_bytes")
        or file_sha256(dataset_path) != dataset_descriptor.get("sha256")
    ):
        _fail("RefinedWeb dataset binding")
    command_dataset = Path(_option(command, "--dataset-shard")).resolve()
    if command_dataset != dataset_path:
        _fail("source producer dataset path")
    dataset = Dataset.from_file(str(dataset_path))

    probe_record = _require_object(source.get("probe"), "source probe")
    probe_count = contexts * (width + 1)
    _require_equal(probe_record.get("token_count"), probe_count, "source probe token_count")
    probe_values, probe_stream = _read_byte_tokens(
        dataset, int(probe_record["first_document"]), probe_count
    )
    _require_equal(probe_stream, probe_record, "source probe stream")
    probe_inputs = probe_values.reshape(contexts, width + 1)[:, :-1].reshape(-1)

    config_rows = {
        row["trajectory_id"]: row for row in source_config["trajectories"]
    }
    source_step = int(manifest["source_step"])
    for row in source["trajectories"]:
        trajectory_id = row["trajectory_id"]
        source_row = config_rows[trajectory_id]
        count = (source_step + 1) * tokens_per_update
        values, stream = _read_byte_tokens(
            dataset, int(source_row["train_document_offset"]), count
        )
        next_batch = values[
            source_step * tokens_per_update : (source_step + 1) * tokens_per_update
        ].reshape(batch_size, width + 1)
        next_inputs = next_batch[:, :-1].reshape(-1)
        choices = sorted(set(next_inputs.tolist()) - set(probe_inputs.tolist()))
        if not choices:
            _fail(f"embedding selection {trajectory_id}", "has no eligible token")
        selected = int(choices[0])
        expected = {
            "selected_token_id": selected,
            "next_input_occurrences": int(np.sum(next_inputs == selected)),
            "absent_from_probe_inputs": bool(np.all(probe_inputs != selected)),
            "next_batch_byte_token_sha256": hashlib.sha256(
                bytes(int(value) - 1 for value in next_batch.reshape(-1))
            ).hexdigest(),
            "stream_prefix": stream,
            "eligible_token_count": len(choices),
        }
        _require_equal(row.get("embedding_selection"), expected, f"embedding selection {trajectory_id}")
        phase = _require_object(row.get("phase"), f"source phase {trajectory_id}")
        expected_phase = {
            "source_step": source_step,
            "trajectory_id": trajectory_id,
            "seed": source_row["seed"],
            "train_document_offset": source_row["train_document_offset"],
            "tokens_per_update": tokens_per_update,
            "next_batch_byte_token_sha256": expected["next_batch_byte_token_sha256"],
            "dataset_sha256": dataset_descriptor["sha256"],
        }
        _require_equal(phase, expected_phase, f"source phase {trajectory_id}")
    return True


def validate_embedding_intervention(
    source_checkpoint: dict[str, Any],
    rewritten_checkpoint: dict[str, Any],
    specification: dict[str, Any],
    selection: dict[str, Any],
    serialized: dict[str, Any],
) -> bool:
    """Validate the exact one-coordinate checkpoint edit and serialized bits."""

    import torch

    key = specification["model_key"]
    token = selection["selected_token_id"]
    coordinate = specification["coordinate"]
    expected_serialized = {
        "operation": "source-null-embedding-shift",
        "role": "gauge-canonical-optimizer-candidate-closure-test",
        "model_key": key,
        "token_id": token,
        "coordinate": coordinate,
        "offset": float(specification["offset"]),
        "before_hex": serialized.get("before_hex"),
        "after_hex": serialized.get("after_hex"),
        "selection": selection,
    }
    _require_equal(serialized, expected_serialized, "serialized embedding intervention")
    source_tensor = _require_object(source_checkpoint.get("model"), "source model").get(key)
    rewritten_tensor = _require_object(rewritten_checkpoint.get("model"), "rewritten model").get(key)
    if not isinstance(source_tensor, torch.Tensor) or not isinstance(rewritten_tensor, torch.Tensor):
        _fail("embedding intervention tensor", "is absent")
    if source_tensor.ndim != 2 or source_tensor.shape != rewritten_tensor.shape or source_tensor.dtype != rewritten_tensor.dtype:
        _fail("embedding intervention tensor shape or dtype")
    if (
        isinstance(token, bool)
        or not isinstance(token, int)
        or isinstance(coordinate, bool)
        or not isinstance(coordinate, int)
        or token < 0
        or token >= source_tensor.shape[0]
        or coordinate < 0
        or coordinate >= source_tensor.shape[1]
    ):
        _fail("embedding intervention coordinate")
    changed = torch.nonzero(source_tensor != rewritten_tensor, as_tuple=False)
    if changed.shape != (1, 2) or changed.tolist()[0] != [token, coordinate]:
        _fail("embedding intervention", "does not change exactly the selected coordinate")
    before = source_tensor[token, coordinate].detach().cpu()
    after = rewritten_tensor[token, coordinate].detach().cpu()
    expected_after = before.clone()
    expected_after.add_(float(specification["offset"]))
    if not torch.equal(after, expected_after):
        _fail("embedding intervention represented dose")
    if before.numpy().tobytes().hex() != serialized["before_hex"]:
        _fail("embedding intervention before bits")
    if after.numpy().tobytes().hex() != serialized["after_hex"]:
        _fail("embedding intervention after bits")

    differences = set(differing_paths(source_checkpoint, rewritten_checkpoint))
    expected_paths = set(PROVENANCE_CHECKPOINT_FIELDS) | {f"model.{key}"}
    if differences != expected_paths:
        _fail("embedding intervention checkpoint leaves")
    return True


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(label, "is not a lowercase SHA-256 value")


def _validate_descriptor_shape(
    descriptor: Any,
    label: str,
    *,
    extra_keys: Iterable[str] = (),
) -> dict[str, Any]:
    item = _require_object(descriptor, label)
    _require_exact_keys(
        item,
        {"path", "sha256", "size_bytes"} | set(extra_keys),
        label,
    )
    _safe_relative(item.get("path"), f"{label}.path")
    _require_sha256(item.get("sha256"), f"{label}.sha256")
    size = item.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        _fail(f"{label}.size_bytes", "is not a nonnegative integer")
    return item


_ROUTE_KEYS = {
    "device",
    "device_name",
    "device_type",
    "energy_route",
    "hook_identity_check",
    "native_model_dtype",
    "row_map_element_count",
    "row_map_sha256",
    "row_map_tensor_count",
    "timing",
    "wall_seconds",
}
_ROUTE_TIMING_KEYS = {
    "component_intervals_include_their_terminal_synchronization",
    "cuda_synchronization_seconds",
    "forward_and_energy_seconds",
    "model_construction_seconds",
    "state_load_and_device_transfer_seconds",
    "total_route_elapsed_seconds",
}


def _validate_route_shape(route: Any, label: str) -> dict[str, Any]:
    item = _require_object(route, label)
    _require_exact_keys(item, _ROUTE_KEYS, label)
    for key in ("device", "device_name", "device_type", "energy_route", "native_model_dtype"):
        if not isinstance(item.get(key), str) or not item[key]:
            _fail(f"{label}.{key}", "is not a nonempty string")
    if item.get("hook_identity_check") is not True:
        _fail(f"{label}.hook_identity_check")
    for key in ("row_map_element_count", "row_map_tensor_count"):
        value = item.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            _fail(f"{label}.{key}", "is not a positive integer")
    _require_sha256(item.get("row_map_sha256"), f"{label}.row_map_sha256")
    wall = _finite_nonnegative(item.get("wall_seconds"), f"{label}.wall_seconds")
    timing = _require_object(item.get("timing"), f"{label}.timing")
    _require_exact_keys(timing, _ROUTE_TIMING_KEYS, f"{label}.timing")
    if timing.get("component_intervals_include_their_terminal_synchronization") is not True:
        _fail(f"{label}.timing synchronization flag")
    for key in _ROUTE_TIMING_KEYS - {
        "component_intervals_include_their_terminal_synchronization"
    }:
        _finite_nonnegative(timing.get(key), f"{label}.timing.{key}")
    if not _same_float(timing.get("total_route_elapsed_seconds"), wall):
        _fail(f"{label}.timing.total_route_elapsed_seconds")
    return item


def _validate_config_manifest(
    config: dict[str, Any],
    manifest: dict[str, Any],
    *,
    run_root: Path,
    data_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    validate_config_shape(config)
    _require_equal(manifest.get("schema_version"), MANIFEST_SCHEMA, "manifest.schema_version")
    _require_equal(manifest.get("stage_id"), config["stage_id"], "manifest.stage_id")
    _require_equal(manifest.get("source_step"), config["source_step"], "manifest.source_step")
    middle = config["source_step"] + config["semigroup_split"]
    final = config["source_step"] + max(config["block_sizes"])
    _require_equal(manifest.get("middle_step"), middle, "manifest.middle_step")
    _require_equal(manifest.get("final_step"), final, "manifest.final_step")
    for key in ("block_sizes", "splits", "candidate_states", "branch_ids"):
        _require_equal(manifest.get(key), config[key], f"manifest.{key}")
    source_run = _require_object(manifest.get("source_run"), "manifest.source_run")
    _require_exact_keys(
        source_run,
        {"locator", "run_id", "config_sha256", "protocol_sha256", "checkpoints"},
        "manifest.source_run",
    )
    predecessor = _require_object(manifest.get("predecessor"), "manifest.predecessor")
    _require_exact_keys(
        predecessor,
        {"locator", "run_manifest", "result"},
        "manifest.predecessor",
    )
    _require_equal(source_run.get("locator"), config["source_run"], "manifest.source_run.locator")
    _require_equal(predecessor.get("locator"), config["predecessor_closure"], "manifest.predecessor.locator")
    if not isinstance(source_run.get("run_id"), str) or not source_run["run_id"]:
        _fail("manifest.source_run.run_id", "is empty")
    _require_sha256(source_run.get("config_sha256"), "manifest.source_run.config_sha256")
    _require_sha256(source_run.get("protocol_sha256"), "manifest.source_run.protocol_sha256")

    input_rows = _require_object(manifest.get("inputs"), "manifest.inputs")
    _require_exact_keys(input_rows, {"config", "protocol", "source_remeasurement"}, "manifest.inputs")
    for key in ("config", "protocol", "source_remeasurement"):
        _validate_descriptor_shape(input_rows[key], f"manifest.inputs.{key}")
    frozen_config_path = resolve_descriptor(run_root, input_rows["config"], "manifest.inputs.config")
    frozen_config = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    _require_equal(frozen_config, config, "frozen config content")
    protocol_path = resolve_descriptor(run_root, input_rows["protocol"], "manifest.inputs.protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    resolve_descriptor(run_root, input_rows["source_remeasurement"], "manifest.inputs.source_remeasurement")

    source_root = resolve_locator(data_root, config["source_run"], "config.source_run")
    source_config_path = source_root / "config-resolved.json"
    source_protocol_path = source_root / "protocol-resolved.json"
    if not source_config_path.is_file() or not source_protocol_path.is_file():
        _fail("source run configuration or protocol", "is absent")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
    _require_equal(source_run.get("run_id"), source_config.get("run_id"), "manifest.source_run.run_id")
    _require_equal(source_run.get("config_sha256"), file_sha256(source_config_path), "manifest.source_run.config_sha256")
    _require_equal(source_run.get("protocol_sha256"), file_sha256(source_protocol_path), "manifest.source_run.protocol_sha256")

    source_subset = {
        "protocol_id": source_protocol["protocol_id"],
        "expected_updates": source_protocol["expected_updates"],
        "segments": source_protocol["segments"],
        "checkpoint_steps": source_protocol["plga_analysis"]["checkpoint_steps"],
    }
    staged_projection = {
        "source_step": config["source_step"],
        "block_sizes": config["block_sizes"],
        "semigroup_split": config["semigroup_split"],
        "candidate_states": config["candidate_states"],
        "splits": config["splits"],
        "branch_ids": config["branch_ids"],
    }
    successor_subset = {
        "protocol_id": config["stage_id"],
        "expected_updates": final,
        "segments": {
            "burn_in_updates": config["source_step"],
            "design_updates": 0,
            "holdout_updates": max(config["block_sizes"]),
        },
        "checkpoint_steps": [0, middle, final],
        "staged_predictive_closure": staged_projection,
    }
    rewrite = _require_object(manifest.get("protocol_rewrite"), "manifest.protocol_rewrite")
    _require_exact_keys(
        rewrite,
        {"source", "successor", "successor_canonical_sha256"},
        "manifest.protocol_rewrite",
    )
    _require_sha256(
        rewrite.get("successor_canonical_sha256"),
        "manifest.protocol_rewrite.successor_canonical_sha256",
    )
    _require_equal(rewrite.get("source"), source_subset, "manifest.protocol_rewrite.source")
    _require_equal(rewrite.get("successor"), successor_subset, "manifest.protocol_rewrite.successor")
    for key, expected in (
        ("protocol_id", config["stage_id"]),
        ("expected_updates", final),
        ("segments", successor_subset["segments"]),
    ):
        _require_equal(protocol.get(key), expected, f"frozen protocol.{key}")
    _require_equal(protocol.get("plga_analysis", {}).get("checkpoint_steps"), [0, middle, final], "frozen protocol checkpoint_steps")
    _require_equal(protocol.get("staged_predictive_closure"), staged_projection, "frozen protocol staged projection")
    protocol_digest = hashlib.sha256(canonical_json_bytes(protocol)).hexdigest()
    _require_equal(rewrite.get("successor_canonical_sha256"), protocol_digest, "manifest protocol canonical digest")

    predecessor_root = resolve_locator(data_root, config["predecessor_closure"], "config.predecessor_closure")
    for key in ("run_manifest", "result"):
        item = _require_object(predecessor.get(key), f"manifest.predecessor.{key}")
        _require_exact_keys(
            item,
            {"anchor", "path", "sha256", "size_bytes", "record_sha256"},
            f"manifest.predecessor.{key}",
        )
        _require_equal(item.get("anchor"), "data_root", f"manifest.predecessor.{key}.anchor")
        _require_sha256(item.get("record_sha256"), f"manifest.predecessor.{key}.record_sha256")
        path = _anchored_descriptor(data_root, predecessor.get(key), f"manifest.predecessor.{key}")
        try:
            path.relative_to(predecessor_root)
        except ValueError as error:
            raise ValueError(f"staged validation: predecessor {key} leaves its locator") from error
        predecessor_record = json.loads(path.read_text(encoding="utf-8"))
        verify_record_seal(predecessor_record)
        _require_equal(
            predecessor.get(key, {}).get("record_sha256"),
            predecessor_record.get("record_sha256"),
            f"manifest.predecessor.{key}.record_sha256",
        )

    trajectory_ids = [row["trajectory_id"] for row in config["trajectories"]]
    devices = {row["trajectory_id"]: row["device"] for row in config["trajectories"]}
    split_lookup = _split_by_trajectory(config)
    source_rows_list = _require_list(source_config.get("trajectories"), "source config trajectories")
    if [row.get("trajectory_id") for row in source_rows_list if isinstance(row, dict)] != trajectory_ids:
        _fail("source config trajectory registry or order")
    for row in source_rows_list:
        _require_equal(row.get("device"), devices[row["trajectory_id"]], f"source config {row['trajectory_id']}.device")
    source_checkpoint_rows = _require_list(source_run.get("checkpoints"), "manifest.source_run.checkpoints")
    expected_source_checkpoints = [
        (trajectory_id, step)
        for trajectory_id in trajectory_ids
        for step in (0, int(config["source_step"]))
    ]
    if [
        (row.get("trajectory_id"), row.get("step"))
        for row in source_checkpoint_rows
        if isinstance(row, dict)
    ] != expected_source_checkpoints:
        _fail("manifest.source_run.checkpoints", "registry or order differs")
    for index, row in enumerate(source_checkpoint_rows):
        label = f"manifest.source_run.checkpoints[{index}]"
        _validate_descriptor_shape(
            row,
            label,
            extra_keys={
                "trajectory_id",
                "step",
                "metadata_path",
                "metadata_sha256",
                "metadata_size_bytes",
            },
        )
        resolve_descriptor(source_root, row, label)
        metadata_descriptor = {
            "path": row["metadata_path"],
            "sha256": row["metadata_sha256"],
            "size_bytes": row["metadata_size_bytes"],
        }
        _validate_descriptor_shape(metadata_descriptor, f"{label}.metadata")
        resolve_descriptor(source_root, metadata_descriptor, f"{label}.metadata")

    branches = _require_object(manifest.get("branches"), "manifest.branches")
    semigroup = _require_object(manifest.get("semigroup_branches"), "manifest.semigroup_branches")
    if list(branches) != trajectory_ids or list(semigroup) != trajectory_ids:
        _fail("manifest trajectory registry or order")
    direct_keys = {
        "trajectory_id",
        "branch",
        "device",
        "split",
        "branch_specification",
        "source_candidate_signatures",
        "source_canonical_transition_state_sha256",
        "checkpoint_rewrite",
        "resume_checkpoint",
        "energies",
        "metrics",
        "metadata",
        "producer_log",
        "output_checkpoints",
        "wall_seconds",
    }
    semigroup_keys = direct_keys - {
        "source_candidate_signatures",
        "source_canonical_transition_state_sha256",
        "checkpoint_rewrite",
    }
    for trajectory_id in trajectory_ids:
        rows = _require_list(branches[trajectory_id], f"manifest.branches.{trajectory_id}")
        if [row.get("branch") for row in rows if isinstance(row, dict)] != config["branch_ids"]:
            _fail(f"manifest.branches.{trajectory_id}", "branch registry or order differs")
        for row in rows:
            branch = row.get("branch") if isinstance(row, dict) else None
            label = f"branch {trajectory_id}/{branch}"
            item = _require_object(row, label)
            _require_exact_keys(item, direct_keys, label)
            _require_equal(item.get("trajectory_id"), trajectory_id, f"{label}.trajectory_id")
            _require_equal(item.get("device"), devices[trajectory_id], f"{label}.device")
            _require_equal(item.get("split"), split_lookup[trajectory_id], f"{label}.split")
            _finite_nonnegative(item.get("wall_seconds"), f"{label}.wall_seconds")
            for key in ("resume_checkpoint", "energies", "metrics", "metadata", "producer_log"):
                _validate_descriptor_shape(item.get(key), f"{label}.{key}")
            checkpoint_rows = _require_list(item.get("output_checkpoints"), f"{label}.output_checkpoints")
            if [
                value.get("step") for value in checkpoint_rows if isinstance(value, dict)
            ] != [0, middle, final]:
                _fail(f"{label}.output_checkpoints", "step registry or order differs")
            for checkpoint_index, checkpoint in enumerate(checkpoint_rows):
                _validate_descriptor_shape(
                    checkpoint,
                    f"{label}.output_checkpoints[{checkpoint_index}]",
                    extra_keys={"trajectory_id", "branch", "step"},
                )
            signatures = _require_object(
                item.get("source_candidate_signatures"),
                f"{label}.source_candidate_signatures",
            )
            _require_exact_keys(
                signatures,
                [candidate["candidate_id"] for candidate in config["candidate_states"]],
                f"{label}.source_candidate_signatures",
            )
            _require_sha256(
                item.get("source_canonical_transition_state_sha256"),
                f"{label}.source_canonical_transition_state_sha256",
            )
            checkpoint_rewrite = _require_object(
                item.get("checkpoint_rewrite"), f"{label}.checkpoint_rewrite"
            )
            _require_exact_keys(
                checkpoint_rewrite,
                {"differing_leaf_paths", "expected_leaf_paths"},
                f"{label}.checkpoint_rewrite",
            )
            for key in ("differing_leaf_paths", "expected_leaf_paths"):
                paths = _require_list(
                    checkpoint_rewrite.get(key), f"{label}.checkpoint_rewrite.{key}"
                )
                if (
                    any(not isinstance(value, str) or not value for value in paths)
                    or paths != sorted(set(paths))
                ):
                    _fail(f"{label}.checkpoint_rewrite.{key}")
            specification = _require_object(
                item.get("branch_specification"), f"{label}.branch_specification"
            )
            if branch == "baseline-direct":
                _require_equal(
                    specification,
                    {"operation": "none", "role": "baseline-replay"},
                    f"{label}.branch_specification",
                )
            elif branch in {"gauge-consistent", "gauge-params-only"}:
                _require_exact_keys(
                    specification,
                    {
                        "operation",
                        "layers",
                        "transform_parameters",
                        "co_transform_first_moment",
                        "role",
                        "changed_tensor_count",
                        "changed_element_count",
                        "model_keys_sha256",
                        "model_keys",
                    },
                    f"{label}.branch_specification",
                )
                expected_role = (
                    "complete-state-quotient-null-control"
                    if branch == "gauge-consistent"
                    else "raw-optimizer-candidate-closure-test"
                )
                expected_core = {
                    "operation": "sign-gauge",
                    "layers": [0, 1, 2],
                    "transform_parameters": True,
                    "co_transform_first_moment": branch == "gauge-consistent",
                    "role": expected_role,
                }
                for key, expected in expected_core.items():
                    _require_equal(
                        specification.get(key), expected,
                        f"{label}.branch_specification.{key}",
                    )
                model_keys = _require_list(
                    specification.get("model_keys"),
                    f"{label}.branch_specification.model_keys",
                )
                if (
                    any(not isinstance(value, str) or not value for value in model_keys)
                    or model_keys != sorted(set(model_keys))
                ):
                    _fail(f"{label}.branch_specification.model_keys")
                count = specification.get("changed_tensor_count")
                elements = specification.get("changed_element_count")
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count != len(model_keys)
                    or isinstance(elements, bool)
                    or not isinstance(elements, int)
                    or elements <= 0
                ):
                    _fail(f"{label}.branch_specification measured support")
                _require_equal(
                    specification.get("model_keys_sha256"),
                    hashlib.sha256(canonical_json_bytes(model_keys)).hexdigest(),
                    f"{label}.branch_specification.model_keys_sha256",
                )
            elif branch == "probe-null-embedding-shift":
                _require_exact_keys(
                    specification,
                    {
                        "operation",
                        "role",
                        "model_key",
                        "token_id",
                        "coordinate",
                        "offset",
                        "before_hex",
                        "after_hex",
                        "selection",
                    },
                    f"{label}.branch_specification",
                )
        semigroup_rows = _require_list(semigroup[trajectory_id], f"manifest.semigroup_branches.{trajectory_id}")
        if len(semigroup_rows) != 1:
            _fail(f"manifest.semigroup_branches.{trajectory_id}", "is not unique")
        row = _require_object(semigroup_rows[0], f"semigroup {trajectory_id}")
        label = f"semigroup {trajectory_id}"
        _require_exact_keys(row, semigroup_keys, label)
        _require_equal(row.get("branch"), "baseline-iterated", f"{label}.branch")
        _require_equal(row.get("trajectory_id"), trajectory_id, f"{label}.trajectory_id")
        _require_equal(row.get("device"), devices[trajectory_id], f"{label}.device")
        _require_equal(row.get("split"), split_lookup[trajectory_id], f"{label}.split")
        _finite_nonnegative(row.get("wall_seconds"), f"{label}.wall_seconds")
        for key in ("resume_checkpoint", "energies", "metrics", "metadata", "producer_log"):
            _validate_descriptor_shape(row.get(key), f"{label}.{key}")
        checkpoint_rows = _require_list(row.get("output_checkpoints"), f"{label}.output_checkpoints")
        if [
            value.get("step") for value in checkpoint_rows if isinstance(value, dict)
        ] != [0, middle, final]:
            _fail(f"{label}.output_checkpoints", "step registry or order differs")
        for checkpoint_index, checkpoint in enumerate(checkpoint_rows):
            _validate_descriptor_shape(
                checkpoint,
                f"{label}.output_checkpoints[{checkpoint_index}]",
                extra_keys={"trajectory_id", "branch", "step"},
            )
        expected_spec = {
            "operation": "restart-at-intermediate-checkpoint",
            "source_branch": "baseline-direct",
            "restart_step": middle,
            "role": "direct-versus-iterated-semigroup-control",
        }
        _require_equal(row.get("branch_specification"), expected_spec, f"semigroup {trajectory_id}.branch_specification")

    measured = _require_object(manifest.get("source_remeasurement"), "manifest.source_remeasurement")
    _require_exact_keys(
        measured,
        {
            "observer",
            "reference_sources",
            "dataset",
            "probe",
            "trajectories",
            "cuda_wall_seconds",
        },
        "manifest.source_remeasurement",
    )
    if not isinstance(measured.get("observer"), str) or not measured["observer"]:
        _fail("manifest.source_remeasurement.observer", "is empty")
    _finite_nonnegative(
        measured.get("cuda_wall_seconds"),
        "manifest.source_remeasurement.cuda_wall_seconds",
    )
    references = _require_list(
        measured.get("reference_sources"),
        "manifest.source_remeasurement.reference_sources",
    )
    reference_paths = []
    source_command = _require_list(
        source_config.get("producer_command"), "source producer_command"
    )
    reference_root = Path(
        _option(source_command, "--reference-experiments")
    ).resolve()
    if not reference_root.is_dir():
        _fail("source reference root", "is absent")
    for index, reference in enumerate(references):
        item = _validate_descriptor_shape(
            reference,
            f"manifest.source_remeasurement.reference_sources[{index}]",
        )
        reference_paths.append(item["path"])
        resolve_descriptor(
            reference_root,
            item,
            f"manifest.source_remeasurement.reference_sources[{index}]",
        )
    if reference_paths != [
        "pldr_model_v510.py",
        "power_law_attention_layer_v510.py",
    ]:
        _fail(
            "manifest.source_remeasurement.reference_sources", "registry differs"
        )

    dataset = _require_object(measured.get("dataset"), "manifest.source_remeasurement.dataset")
    _require_exact_keys(
        dataset,
        {"relative_path", "root_role", "sha256", "size_bytes"},
        "manifest.source_remeasurement.dataset",
    )
    _safe_relative(dataset.get("relative_path"), "manifest.source_remeasurement.dataset.relative_path")
    _require_equal(
        dataset.get("root_role"),
        "refinedweb_read_only_root",
        "manifest.source_remeasurement.dataset.root_role",
    )
    _require_sha256(dataset.get("sha256"), "manifest.source_remeasurement.dataset.sha256")
    dataset_size = dataset.get("size_bytes")
    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int) or dataset_size < 0:
        _fail("manifest.source_remeasurement.dataset.size_bytes")

    probe = _require_object(measured.get("probe"), "manifest.source_remeasurement.probe")
    stream_keys = {
        "first_document",
        "last_document_exclusive",
        "document_count",
        "token_count",
        "byte_token_sha256",
        "streaming",
        "maximum_buffer_policy",
    }
    _require_exact_keys(probe, stream_keys, "manifest.source_remeasurement.probe")

    measured_rows = _require_list(measured.get("trajectories"), "manifest.source_remeasurement.trajectories")
    if [row.get("trajectory_id") for row in measured_rows if isinstance(row, dict)] != trajectory_ids:
        _fail("manifest.source_remeasurement.trajectories", "registry or order differs")
    source_rows = {row["trajectory_id"]: row for row in source_config["trajectories"]}
    candidate_ids = [row["candidate_id"] for row in config["candidate_states"]]
    source_row_keys = {
        "trajectory_id",
        "split",
        "phase",
        "embedding_selection",
        "original_route",
        "original_candidate_signatures",
        "original_canonical_transition_state_sha256",
        "branches",
    }
    phase_keys = {
        "source_step",
        "trajectory_id",
        "seed",
        "train_document_offset",
        "tokens_per_update",
        "next_batch_byte_token_sha256",
        "dataset_sha256",
    }
    selection_keys = {
        "selected_token_id",
        "next_input_occurrences",
        "absent_from_probe_inputs",
        "next_batch_byte_token_sha256",
        "stream_prefix",
        "eligible_token_count",
    }
    for row in measured_rows:
        trajectory_id = row["trajectory_id"]
        label = f"source measurement {trajectory_id}"
        _require_exact_keys(row, source_row_keys, label)
        _require_equal(row.get("split"), split_lookup[trajectory_id], f"{label}.split")
        phase = _require_object(row.get("phase"), f"{label}.phase")
        _require_exact_keys(phase, phase_keys, f"{label}.phase")
        _require_equal(phase.get("trajectory_id"), trajectory_id, f"{label}.phase.trajectory_id")
        _require_equal(phase.get("source_step"), config["source_step"], f"{label}.phase.source_step")
        _require_equal(phase.get("seed"), source_rows[trajectory_id]["seed"], f"{label}.phase.seed")
        _require_equal(
            phase.get("train_document_offset"),
            source_rows[trajectory_id]["train_document_offset"],
            f"{label}.phase.train_document_offset",
        )
        selection = _require_object(row.get("embedding_selection"), f"{label}.embedding_selection")
        _require_exact_keys(selection, selection_keys, f"{label}.embedding_selection")
        stream_prefix = _require_object(
            selection.get("stream_prefix"), f"{label}.embedding_selection.stream_prefix"
        )
        _require_exact_keys(
            stream_prefix,
            stream_keys,
            f"{label}.embedding_selection.stream_prefix",
        )
        original_route = _validate_route_shape(row.get("original_route"), f"{label}.original_route")
        _require_equal(
            original_route.get("device"),
            f"cuda:{devices[trajectory_id]}",
            f"{label}.original_route.device",
        )
        original_signatures = _require_object(
            row.get("original_candidate_signatures"),
            f"{label}.original_candidate_signatures",
        )
        _require_exact_keys(
            original_signatures,
            candidate_ids,
            f"{label}.original_candidate_signatures",
        )
        _require_sha256(
            row.get("original_canonical_transition_state_sha256"),
            f"{label}.original_canonical_transition_state_sha256",
        )
        source_branch_rows = _require_list(row.get("branches"), f"{label}.branches")
        if [item.get("branch") for item in source_branch_rows if isinstance(item, dict)] != config["branch_ids"]:
            _fail(f"{label}.branches", "registry or order differs")
        for branch_row in source_branch_rows:
            branch = branch_row.get("branch") if isinstance(branch_row, dict) else None
            branch_label = f"{label}.branches.{branch}"
            item = _require_object(branch_row, branch_label)
            _require_exact_keys(
                item,
                {
                    "branch",
                    "route",
                    "candidate_signatures",
                    "canonical_transition_state_sha256",
                },
                branch_label,
            )
            route = _validate_route_shape(item.get("route"), f"{branch_label}.route")
            _require_equal(
                route.get("device"),
                f"cuda:{devices[trajectory_id]}",
                f"{branch_label}.route.device",
            )
            signatures = _require_object(
                item.get("candidate_signatures"), f"{branch_label}.candidate_signatures"
            )
            _require_exact_keys(
                signatures, candidate_ids, f"{branch_label}.candidate_signatures"
            )
            _require_sha256(
                item.get("canonical_transition_state_sha256"),
                f"{branch_label}.canonical_transition_state_sha256",
            )
    inventory = _require_object(manifest.get("checkpoint_inventory"), "manifest.checkpoint_inventory")
    _require_exact_keys(
        inventory,
        {
            "descriptor_count",
            "total_size_bytes",
            "expected_steps_per_branch",
            "all_emitted_files_bound",
        },
        "manifest.checkpoint_inventory",
    )
    return source_config, protocol, source_root


def _source_checkpoint_path(
    source_root: Path, manifest: dict[str, Any], trajectory_id: str, step: int
) -> Path:
    rows = [
        row
        for row in manifest["source_run"]["checkpoints"]
        if row.get("trajectory_id") == trajectory_id and row.get("step") == step
    ]
    if len(rows) != 1:
        _fail(f"source checkpoint {trajectory_id}/{step}", "is not unique")
    return resolve_descriptor(source_root, rows[0], f"source checkpoint {trajectory_id}/{step}")


def _load_source_measurements(run_root: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    path = resolve_descriptor(
        run_root,
        manifest["inputs"]["source_remeasurement"],
        "manifest.inputs.source_remeasurement",
    )
    expected = {
        "trajectory_ids",
        "branch_ids",
        "map_ids",
        "restored_recorder_energy",
        "rewritten_branch_remeasured_energy",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            _fail("source measurement archive", "keys differ")
        values = {key: np.asarray(archive[key]) for key in expected}
    for key in ("trajectory_ids", "branch_ids", "map_ids"):
        values[key] = values[key].astype(str)
    for key in ("restored_recorder_energy", "rewritten_branch_remeasured_energy"):
        array = np.asarray(values[key], dtype=np.float64)
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            _fail(f"source measurement archive {key}")
        values[key] = array
    return values


def _validate_checkpoint_artifacts(
    manifest: dict[str, Any],
    config: dict[str, Any],
    run_root: Path,
    source_root: Path,
) -> tuple[bool, bool]:
    import torch

    measurements = _load_source_measurements(run_root, manifest)
    trajectory_ids = [row["trajectory_id"] for row in config["trajectories"]]
    if measurements["trajectory_ids"].tolist() != trajectory_ids:
        _fail("source measurement trajectory registry")
    if measurements["branch_ids"].tolist() != config["branch_ids"]:
        _fail("source measurement branch registry")
    map_count = len(measurements["map_ids"])
    if measurements["restored_recorder_energy"].shape != (len(trajectory_ids), map_count):
        _fail("restored source measurement shape")
    if measurements["rewritten_branch_remeasured_energy"].shape != (
        len(trajectory_ids), len(config["branch_ids"]), map_count
    ):
        _fail("rewritten source measurement shape")

    source_records = {
        row["trajectory_id"]: row for row in manifest["source_remeasurement"]["trajectories"]
    }
    expected_steps = [0, int(manifest["middle_step"]), int(manifest["final_step"])]
    descriptor_count = 0
    descriptor_bytes = 0
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        source_path = _source_checkpoint_path(
            source_root, manifest, trajectory_id, int(config["source_step"])
        )
        source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        source_record = source_records[trajectory_id]
        phase = source_record["phase"]
        original = candidate_signatures(
            source_checkpoint,
            measurements["restored_recorder_energy"][trajectory_index],
            source_record["original_route"]["row_map_sha256"],
            phase=phase,
            is_gauge_parameter=_is_sign_gauge_parameter,
            gauge_layer=_sign_gauge_layer,
        )
        _require_equal(original, source_record.get("original_candidate_signatures"), f"original signatures {trajectory_id}")
        original_transition = canonical_transition_state_sha256(
            source_checkpoint,
            phase=phase,
            is_gauge_parameter=_is_sign_gauge_parameter,
            gauge_layer=_sign_gauge_layer,
        )
        _require_equal(
            original_transition,
            source_record.get("original_canonical_transition_state_sha256"),
            f"original transition {trajectory_id}",
        )
        source_branch_records = {row["branch"]: row for row in source_record["branches"]}
        for branch_index, row in enumerate(manifest["branches"][trajectory_id]):
            branch = row["branch"]
            for key in ("energies", "metrics", "metadata", "producer_log", "resume_checkpoint"):
                path = resolve_descriptor(run_root, row[key], f"{trajectory_id}/{branch}/{key}")
                if key == "metadata":
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    _require_equal(metadata.get("schema_version"), PRODUCER_SCHEMA, f"{trajectory_id}/{branch} metadata schema")
                    _require_equal(metadata.get("code_commit"), manifest["code_commit"], f"{trajectory_id}/{branch} metadata commit")
                    _require_equal(metadata.get("trajectory_id"), trajectory_id, f"{trajectory_id}/{branch} metadata trajectory")
                    if not _same_float(metadata.get("wall_seconds"), row.get("wall_seconds")):
                        _fail(f"{trajectory_id}/{branch} metadata wall_seconds")
            checkpoint_rows = _require_list(row.get("output_checkpoints"), f"{trajectory_id}/{branch} output checkpoints")
            if [item.get("step") for item in checkpoint_rows if isinstance(item, dict)] != expected_steps:
                _fail(f"{trajectory_id}/{branch} output checkpoint steps")
            for item in checkpoint_rows:
                path = resolve_descriptor(run_root, item, f"{trajectory_id}/{branch} output checkpoint")
                descriptor_count += 1
                descriptor_bytes += path.stat().st_size
                _require_equal(item.get("trajectory_id"), trajectory_id, f"{trajectory_id}/{branch} checkpoint trajectory")
                _require_equal(item.get("branch"), branch, f"{trajectory_id}/{branch} checkpoint branch")

            rewritten_path = resolve_descriptor(run_root, row["resume_checkpoint"], f"{trajectory_id}/{branch} resume")
            rewritten = torch.load(rewritten_path, map_location="cpu", weights_only=False)
            _require_equal(rewritten.get("schema_version"), CHECKPOINT_SCHEMA, f"{trajectory_id}/{branch} checkpoint schema")
            _require_equal(rewritten.get("step"), config["source_step"], f"{trajectory_id}/{branch} checkpoint step")
            _require_equal(rewritten.get("code_commit"), manifest["code_commit"], f"{trajectory_id}/{branch} checkpoint commit")
            _require_equal(
                rewritten.get("protocol_sha256"),
                manifest["protocol_rewrite"]["successor_canonical_sha256"],
                f"{trajectory_id}/{branch} checkpoint protocol",
            )
            expected_paths = set(row["checkpoint_rewrite"]["expected_leaf_paths"])
            recorded_paths = set(row["checkpoint_rewrite"]["differing_leaf_paths"])
            actual_paths = set(differing_paths(source_checkpoint, rewritten))
            if expected_paths != recorded_paths or actual_paths != recorded_paths:
                _fail(f"{trajectory_id}/{branch} checkpoint rewrite")

            branch_record = source_branch_records[branch]
            route = branch_record["route"]
            signatures = candidate_signatures(
                rewritten,
                measurements["rewritten_branch_remeasured_energy"][trajectory_index, branch_index],
                route["row_map_sha256"],
                phase=phase,
                is_gauge_parameter=_is_sign_gauge_parameter,
                gauge_layer=_sign_gauge_layer,
            )
            transition = canonical_transition_state_sha256(
                rewritten,
                phase=phase,
                is_gauge_parameter=_is_sign_gauge_parameter,
                gauge_layer=_sign_gauge_layer,
            )
            _require_equal(signatures, row.get("source_candidate_signatures"), f"{trajectory_id}/{branch} manifest signatures")
            _require_equal(signatures, branch_record.get("candidate_signatures"), f"{trajectory_id}/{branch} source signatures")
            _require_equal(transition, row.get("source_canonical_transition_state_sha256"), f"{trajectory_id}/{branch} manifest transition")
            _require_equal(transition, branch_record.get("canonical_transition_state_sha256"), f"{trajectory_id}/{branch} source transition")

            specification = row["branch_specification"]
            if branch == "baseline-direct":
                _require_equal(specification, {"operation": "none", "role": "baseline-replay"}, f"{trajectory_id}/{branch} specification")
            elif branch == "probe-null-embedding-shift":
                selection = source_record["embedding_selection"]
                _require_equal(specification.get("selection"), selection, f"{trajectory_id}/{branch} selection")
                validate_embedding_intervention(
                    source_checkpoint,
                    rewritten,
                    config["embedding_shift"],
                    selection,
                    specification,
                )
            elif branch in {"gauge-consistent", "gauge-params-only"}:
                model_keys = sorted(
                    name
                    for name in source_checkpoint["model"]
                    if _is_sign_gauge_parameter(name)
                )
                expected_specification = {
                    "operation": "sign-gauge",
                    "layers": [0, 1, 2],
                    "transform_parameters": True,
                    "co_transform_first_moment": branch == "gauge-consistent",
                    "role": (
                        "complete-state-quotient-null-control"
                        if branch == "gauge-consistent"
                        else "raw-optimizer-candidate-closure-test"
                    ),
                    "changed_tensor_count": len(model_keys),
                    "changed_element_count": sum(
                        int(source_checkpoint["model"][name].numel())
                        for name in model_keys
                    ),
                    "model_keys_sha256": hashlib.sha256(
                        canonical_json_bytes(model_keys)
                    ).hexdigest(),
                    "model_keys": model_keys,
                }
                _require_equal(
                    specification,
                    expected_specification,
                    f"{trajectory_id}/{branch} specification",
                )

        semigroup_rows = manifest["semigroup_branches"][trajectory_id]
        for row in semigroup_rows:
            branch = row["branch"]
            for key in ("energies", "metrics", "metadata", "producer_log", "resume_checkpoint"):
                path = resolve_descriptor(run_root, row[key], f"{trajectory_id}/{branch}/{key}")
                if key == "metadata":
                    metadata = json.loads(path.read_text(encoding="utf-8"))
                    _require_equal(metadata.get("schema_version"), PRODUCER_SCHEMA, f"{trajectory_id}/{branch} metadata schema")
                    _require_equal(metadata.get("code_commit"), manifest["code_commit"], f"{trajectory_id}/{branch} metadata commit")
                    _require_equal(metadata.get("trajectory_id"), trajectory_id, f"{trajectory_id}/{branch} metadata trajectory")
                    if not _same_float(metadata.get("wall_seconds"), row.get("wall_seconds")):
                        _fail(f"{trajectory_id}/{branch} metadata wall_seconds")
            checkpoint_rows = row["output_checkpoints"]
            if [item.get("step") for item in checkpoint_rows] != expected_steps:
                _fail(f"{trajectory_id}/{branch} output checkpoint steps")
            for item in checkpoint_rows:
                path = resolve_descriptor(run_root, item, f"{trajectory_id}/{branch} output checkpoint")
                descriptor_count += 1
                descriptor_bytes += path.stat().st_size
                _require_equal(item.get("trajectory_id"), trajectory_id, f"{trajectory_id}/{branch} checkpoint trajectory")
                _require_equal(item.get("branch"), branch, f"{trajectory_id}/{branch} checkpoint branch")

    inventory = _require_object(manifest.get("checkpoint_inventory"), "manifest.checkpoint_inventory")
    expected_count = (len(config["branch_ids"]) + 1) * len(trajectory_ids) * len(expected_steps)
    inventory_valid = bool(
        descriptor_count == expected_count
        and inventory.get("descriptor_count") == descriptor_count
        and inventory.get("total_size_bytes") == descriptor_bytes
        and inventory.get("expected_steps_per_branch") == expected_steps
        and inventory.get("all_emitted_files_bound") is True
    )
    if not inventory_valid:
        _fail("manifest.checkpoint_inventory")
    return True, inventory_valid


def staged_source_paths_for_commit(
    repository_root: Path, commit: str
) -> tuple[str, ...]:
    """Return the exact registry required by the named producer commit."""

    try:
        git_blob_descriptor(
            repository_root, commit, "src/row_rgmap/staged_validation.py"
        )
    except subprocess.CalledProcessError:
        return LEGACY_STAGED_SOURCE_PATHS
    return STAGED_SOURCE_PATHS


def validate_staged_source_bindings(manifest: dict[str, Any], repository_root: Path) -> bool:
    commit = manifest.get("code_commit")
    sources = manifest.get("analysis_sources")
    if not isinstance(commit, str) or not isinstance(sources, list) or not sources:
        _fail("manifest.analysis_sources")
    expected_paths = staged_source_paths_for_commit(repository_root, commit)
    items: list[dict[str, Any]] = []
    paths: list[str] = []
    for row in sources:
        item = _require_object(row, "manifest analysis source")
        repository_path = item.get("repository_path")
        if not isinstance(repository_path, str):
            _fail("manifest.analysis_sources", "has an invalid path")
        items.append(item)
        paths.append(repository_path)
    _require_equal(
        paths,
        list(expected_paths),
        "manifest.analysis_sources registry and order",
    )
    for item, repository_path in zip(items, paths):
        digest, size = git_blob_descriptor(repository_root, commit, repository_path)
        expected_item = {
            "repository_path": repository_path,
            "code_commit": commit,
            "sha256": digest,
            "size_bytes": size,
        }
        _require_equal(item, expected_item, f"manifest analysis source {repository_path}")
    return True


def validate_staged_result(
    record: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, Any],
    *,
    run_root: Path,
    data_root: Path,
    repository_root: Path,
    refinedweb_root: Path,
) -> dict[str, Any]:
    """Perform the full release-grade staged-result validation."""

    run_root = run_root.resolve()
    data_root = data_root.resolve()
    repository_root = repository_root.resolve()
    refinedweb_root = refinedweb_root.resolve()
    try:
        run_root.relative_to(data_root)
    except ValueError as error:
        raise ValueError("staged validation: run root leaves data root") from error
    verify_record_seal(manifest)
    verify_record_seal(record)
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "completed_at_utc",
            "code_commit",
            "analysis_sources",
            "stage_id",
            "source_step",
            "middle_step",
            "final_step",
            "block_sizes",
            "splits",
            "candidate_states",
            "branch_ids",
            "source_run",
            "predecessor",
            "inputs",
            "source_remeasurement",
            "branches",
            "semigroup_branches",
            "protocol_rewrite",
            "checkpoint_inventory",
            "gpu_time_accounting",
            "gpu_seconds",
            "gpu_hours",
            "record_sha256",
        },
        "manifest",
    )
    _require_exact_keys(
        record,
        {
            "schema_version",
            "completed_at_utc",
            "code_commit",
            "analysis_sources",
            "input_run_manifest",
            "stage_id",
            "gpu_time_accounting",
            "gpu_seconds",
            "gpu_hours",
            "validity",
            "scientific_payload",
            "record_sha256",
        },
        "result",
    )
    _require_equal(record.get("schema_version"), RESULT_SCHEMA, "result.schema_version")
    _require_equal(record.get("code_commit"), manifest.get("code_commit"), "result.code_commit")
    _require_equal(record.get("stage_id"), manifest.get("stage_id"), "result.stage_id")
    _require_equal(record.get("analysis_sources"), manifest.get("analysis_sources"), "result.analysis_sources")
    manifest_path = run_root / "run-manifest.json"
    if not manifest_path.is_file():
        _fail("run-manifest.json", "is absent")
    stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_equal(stored_manifest, manifest, "run-manifest.json content")
    input_manifest = _require_object(record.get("input_run_manifest"), "result.input_run_manifest")
    _require_exact_keys(
        input_manifest,
        {"record_sha256", "sha256", "size_bytes"},
        "result.input_run_manifest",
    )
    _require_equal(input_manifest.get("record_sha256"), manifest.get("record_sha256"), "result input manifest record digest")
    _require_equal(input_manifest.get("sha256"), file_sha256(manifest_path), "result input manifest file digest")
    _require_equal(input_manifest.get("size_bytes"), manifest_path.stat().st_size, "result input manifest size")

    source_config, _protocol, source_root = _validate_config_manifest(
        config, manifest, run_root=run_root, data_root=data_root
    )
    validate_staged_source_bindings(manifest, repository_root)
    resources = validate_resource_accounting(manifest, config, record)
    rewrites_exact, inventory_exact = _validate_checkpoint_artifacts(
        manifest, config, run_root, source_root
    )
    replay_source_null_selections(manifest, source_config, refinedweb_root)
    scientific = validate_scientific_payload(record.get("scientific_payload"), config)

    controls = scientific["controls"]
    expected_checks = {
        "manifest_seal": True,
        "source_blobs_at_execution_commit": True,
        "frozen_config_matches_manifest": True,
        "source_and_predecessor_descriptors": True,
        "checkpoint_rewrites_exact": rewrites_exact,
        "checkpoint_inventory_exact": inventory_exact,
        "baseline_replay": controls["baseline_replay"]["passed"],
        "gauge_quotient_control": controls["complete_state_gauge_quotient"]["passed"],
        "direct_vs_iterated_semigroup": controls["direct_vs_iterated_semigroup"]["passed"],
        "gpu_cap": resources["cap_passed"],
    }
    validity = _require_object(record.get("validity"), "result.validity")
    _require_exact_keys(validity, {"structural_valid", "structural_checks"}, "result.validity")
    _require_equal(validity.get("structural_checks"), expected_checks, "result.validity.structural_checks")
    _require_equal(validity.get("structural_valid"), all(expected_checks.values()), "result.validity.structural_valid")
    return {
        "scientific_payload": scientific,
        "block_sizes": list(config["block_sizes"]),
        "splits": config["splits"],
        "candidate_states": config["candidate_states"],
        "gpu_seconds": resources["gpu_seconds"],
        "gpu_hours": resources["gpu_hours"],
        "gpu_cap_passed": resources["cap_passed"],
        "structural_valid": validity["structural_valid"],
        "trust_boundary": TRUST_BOUNDARY,
    }


def declared_relative_files(manifest: dict[str, Any]) -> set[PurePosixPath]:
    """Return the exact lexical staged tree declared by the run manifest."""

    paths = {PurePosixPath("run-manifest.json"), PurePosixPath("result.json")}

    def add(descriptor: Any, label: str) -> None:
        relative = _safe_relative(_require_object(descriptor, label).get("path"), f"{label}.path")
        if relative in paths:
            _fail(label, f"duplicates declared path {relative}")
        paths.add(relative)

    for key, item in _require_object(manifest.get("inputs"), "manifest.inputs").items():
        add(item, f"manifest.inputs.{key}")
    rows = [
        row
        for values in _require_object(manifest.get("branches"), "manifest.branches").values()
        for row in _require_list(values, "manifest branch rows")
    ]
    rows.extend(
        row
        for values in _require_object(manifest.get("semigroup_branches"), "manifest.semigroup_branches").values()
        for row in _require_list(values, "manifest semigroup rows")
    )
    for row_index, row in enumerate(rows):
        item = _require_object(row, f"manifest continuation row {row_index}")
        for key in ("resume_checkpoint", "energies", "metrics", "metadata", "producer_log"):
            add(item[key], f"manifest continuation row {row_index}.{key}")
        for checkpoint_index, checkpoint in enumerate(item["output_checkpoints"]):
            add(checkpoint, f"manifest continuation row {row_index}.output_checkpoints[{checkpoint_index}]")
    return paths


def lexical_file_census(
    run_root: Path,
    expected: set[PurePosixPath],
    *,
    ignored_root_files: frozenset[str] = frozenset({"relocation-validation.json"}),
) -> dict[str, Any]:
    """Census lexical entries with lstat and reject links and nonregular nodes."""

    root = run_root.resolve()
    if not root.is_dir():
        _fail("staged census root", "is not a directory")
    expected_strings = {path.as_posix() for path in expected}
    allowed_directories = {""}
    for path in expected:
        parent = path.parent
        while parent != PurePosixPath("."):
            allowed_directories.add(parent.as_posix())
            parent = parent.parent
    actual: set[str] = set()
    total_size = 0

    def visit(directory: Path, relative_parent: PurePosixPath) -> None:
        nonlocal total_size
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda value: value.name):
                relative = relative_parent / entry.name
                relative_string = relative.as_posix()
                metadata = entry.stat(follow_symlinks=False)
                mode = metadata.st_mode
                if stat.S_ISLNK(mode):
                    _fail("staged file census", f"refuses symbolic link {relative_string}")
                if stat.S_ISDIR(mode):
                    if relative_string not in allowed_directories:
                        _fail("staged file census", f"has unexpected directory {relative_string}")
                    visit(Path(entry.path), relative)
                elif stat.S_ISREG(mode):
                    if relative_parent == PurePosixPath(".") and entry.name in ignored_root_files:
                        continue
                    actual.add(relative_string)
                    total_size += metadata.st_size
                else:
                    _fail("staged file census", f"refuses nonregular entry {relative_string}")

    visit(root, PurePosixPath("."))
    if actual != expected_strings:
        _fail(
            "staged file census",
            f"is not exact; missing={sorted(expected_strings - actual)}, "
            f"extra={sorted(actual - expected_strings)}",
        )
    return {"file_count": len(actual), "total_size_bytes": total_size, "exact": True}


def validate_staged_relocation(
    relocation: dict[str, Any],
    record: dict[str, Any],
    manifest: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    """Validate the relocation record against a fresh lexical source census."""

    verify_record_seal(relocation)
    _require_exact_keys(
        relocation,
        {
            "schema_version",
            "completed_at_utc",
            "code_commit",
            "input_run_manifest_record_sha256",
            "input_analysis_record_sha256",
            "source_census",
            "relocated_census",
            "scientific_payload_equal",
            "temporary_copy_removed",
            "record_sha256",
        },
        "relocation",
    )
    _require_equal(relocation.get("schema_version"), RELOCATION_SCHEMA, "relocation.schema_version")
    _require_equal(relocation.get("code_commit"), manifest.get("code_commit"), "relocation.code_commit")
    _require_equal(relocation.get("input_run_manifest_record_sha256"), manifest.get("record_sha256"), "relocation manifest digest")
    _require_equal(relocation.get("input_analysis_record_sha256"), record.get("record_sha256"), "relocation result digest")
    source_census = lexical_file_census(run_root, declared_relative_files(manifest))
    _require_equal(relocation.get("source_census"), source_census, "relocation.source_census")
    _require_equal(relocation.get("relocated_census"), source_census, "relocation.relocated_census")
    _require_equal(relocation.get("scientific_payload_equal"), True, "relocation.scientific_payload_equal")
    _require_equal(relocation.get("temporary_copy_removed"), True, "relocation.temporary_copy_removed")
    return source_census
