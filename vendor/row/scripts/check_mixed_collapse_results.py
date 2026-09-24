#!/usr/bin/env python3
"""Verify technical closure of the completed mixed-collapse evidence bundle.

This gate checks provenance, registered coverage, numerical identities, and
resource limits.  It deliberately does not require a preferred scientific
outcome such as terminal contraction or PLGA attenuation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from analysis.summarize_mixed_manuscript_results import SHAPE_SUPPLEMENT_SCHEMA, _shape_supplement
from confirm.mixed_collapse_specs import (  # noqa: E402
    BLOCK_ANCHORS,
    CAMPAIGN_ID,
    DENSE_CENTERS,
    INTERVENTION_LENGTH,
    INTERVENTION_MODES,
    INTERVENTION_UPDATES,
    REGISTRY,
    RESOURCE_BUDGET,
    SOURCE_RADIUS_UPDATES,
    TERMINAL_UPDATE,
    TOLERANCES,
    TRAJECTORIES,
    dense_snapshot_updates,
    full_snapshot_updates,
)
from scripts.execute_mixed_collapse_plan import (  # noqa: E402
    _canonical as _canonical_command,
    _resolve_command,
)


DEFAULT_BUNDLE = (
    ROOT.parent / "experiment-data" / "manuscript-revisions" / "rev47"
    / "mixed-row-map-collapse-confirmation"
)
DEFAULT_TEX = ROOT / "docs" / "figures" / "mixed_collapse_confirmation_results.tex"
DEFAULT_MECHANISM_TEX = (
    ROOT / "docs" / "figures" / "mixed_collapse_mechanism_results.tex"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_tree(value: Any, location: str = "result") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"nonfinite value at {location}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{location}[{index}]")


def _verify_radius_outcomes(radius: dict[str, Any]) -> None:
    """Validate recorded radius outcomes without requiring a positive result."""

    records = radius["records"]
    for aggregate, field in (
        ("all_radius_contracts_closed", "radius_contract_closed"),
        ("all_energy_envelopes_closed", "energy_envelope_closed"),
    ):
        _require(
            isinstance(radius[aggregate], bool),
            f"{aggregate} is not a boolean",
        )
        for row in records:
            _require(
                isinstance(row[field], bool),
                f"recorded {field} is not a boolean",
            )
        _require(
            radius[aggregate] is all(row[field] for row in records),
            f"{aggregate} disagrees with its records",
        )


def _verify_factorization_outcomes(
    factorization: dict[str, Any],
    *,
    expected_timepoints: int,
    map_count: int,
) -> None:
    """Validate mapwise residual decisions without requiring a pass."""

    threshold = float(factorization["construction_frozen_tolerance"])
    maximum = float(factorization["maximum_per_map_relative_residual"])
    failing_timepoints = int(
        factorization["timepoints_with_any_map_above_tolerance"]
    )
    failing_map_timepoints = int(
        factorization["map_timepoints_above_tolerance"]
    )
    _require(
        threshold == TOLERANCES["construction_float32_factorization_relative"],
        "factorization tolerance drifted",
    )
    _require(
        expected_timepoints > 0 and map_count > 0,
        "factorization population is empty",
    )
    _require(maximum >= 0.0, "factorization maximum is negative")
    _require(
        0 <= failing_timepoints <= expected_timepoints,
        "factorization failing-timepoint count is impossible",
    )
    _require(
        failing_timepoints <= failing_map_timepoints
        <= failing_timepoints * map_count,
        "factorization failing-map count is impossible",
    )
    _require(
        (maximum <= threshold) == (failing_timepoints == 0),
        "factorization maximum disagrees with failing states",
    )
    _require(
        (failing_timepoints == 0) == (failing_map_timepoints == 0),
        "factorization state and map counts disagree",
    )


def _full_timepoint_payloads(
    path: Path, expected_steps: set[int]
) -> dict[int, dict[str, Any]]:
    """Read exactly one registered full payload at each requested step."""

    result: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            point = row.get("mixed_collapse_full_timepoint")
            if not isinstance(point, dict):
                continue
            step = int(point["step"])
            if step not in expected_steps:
                continue
            _require(step not in result, f"duplicate full timepoint {step}: {path}")
            result[step] = point
    missing = expected_steps - set(result)
    _require(not missing, f"missing full timepoints {sorted(missing)}: {path}")
    return result


def _verify_full_arm_reproduction(bundle: Path) -> None:
    """Require every untreated factorial control to replay its native path."""

    endpoints = {
        source_step + INTERVENTION_LENGTH
        for source_step in INTERVENTION_UPDATES
    }
    for trajectory in TRAJECTORIES:
        seed = int(trajectory["seed"])
        native_path = bundle / "runs" / str(trajectory["name"]) / "log.jsonl"
        native = _full_timepoint_payloads(native_path, endpoints)
        for source_step in INTERVENTION_UPDATES:
            endpoint = source_step + INTERVENTION_LENGTH
            control_path = (
                bundle / "runs" / f"gate-{seed}-{source_step}-full" / "log.jsonl"
            )
            control = _full_timepoint_payloads(control_path, {endpoint})[endpoint]
            _require(
                control == native[endpoint],
                "full factorial arm does not exactly reproduce native endpoint: "
                f"seed {seed}, source {source_step}",
            )


def _verify_prefix_reproduction(bundle: Path) -> None:
    """Require the registered long runs to reproduce both inherited prefixes."""

    expected_scopes = ["model", "opt", "scheduler", "rng_states"]
    for trajectory in TRAJECTORIES:
        name = str(trajectory["name"])
        for step in (16_384, 18_000):
            path = (
                bundle / "reports"
                / f"prefix-reproduction-{name}-{step}.json"
            )
            _require(path.is_file(), f"missing prefix comparison: {name} {step}")
            report = _load(path)
            _require(
                report.get("schema_version")
                == "pldr-checkpoint-exact-comparison-v1",
                f"prefix comparison schema drifted: {name} {step}",
            )
            _require(
                report.get("left_step") == step
                and report.get("right_step") == step,
                f"prefix comparison step drifted: {name} {step}",
            )
            _require(
                report.get("scopes") == expected_scopes,
                f"prefix comparison scope drifted: {name} {step}",
            )
            _require(
                report.get("exact_equal") is True
                and report.get("mismatch_count") == 0
                and report.get("mismatches") == [],
                f"prefix comparison is not exact: {name} {step}",
            )
            for side in ("left", "right"):
                record = report.get(side)
                digest = record.get("sha256") if isinstance(record, dict) else None
                _require(
                    isinstance(digest, str)
                    and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest),
                    f"prefix comparison digest is invalid: {name} {step} {side}",
                )


def _verify_nodes(bundle: Path, plan: dict[str, Any]) -> tuple[int, float]:
    nodes = plan.get("nodes")
    _require(isinstance(nodes, list), "launch plan nodes are absent")
    expected_node_count = (
        len(TRAJECTORIES)
        + len(TRAJECTORIES) * len(INTERVENTION_UPDATES) * len(INTERVENTION_MODES)
        + len(TRAJECTORIES) * len(SOURCE_RADIUS_UPDATES)
        + 1
    )
    _require(len(nodes) == expected_node_count, "launch node count drifted")
    identifiers = [str(node["id"]) for node in nodes]
    _require(len(set(identifiers)) == len(identifiers), "duplicate launch node")
    binding = _load(bundle / "protocol" / "input_binding.json")
    tokens = Path(str(binding["tokens_path"])).resolve()
    tokenizer = Path(str(binding["tokenizer_path"])).resolve()

    aggregate_gpu_seconds = 0.0
    for node in nodes:
        node_id = str(node["id"])
        completed_path = bundle / "runtime" / node_id / "completed.json"
        _require(completed_path.is_file(), f"incomplete node: {node_id}")
        completed = _load(completed_path)
        _require(completed.get("exit_code") == 0, f"failed node: {node_id}")
        resolved_command = _resolve_command(
            node["command"],
            bundle=bundle,
            tokens=tokens,
            tokenizer=tokenizer,
            device_override=None,
        )
        command_digest = hashlib.sha256(
            _canonical_command(resolved_command)
        ).hexdigest()
        _require(
            completed.get("schema_version")
            == "pldr-mixed-collapse-execution-v1"
            and completed.get("node_id") == node_id
            and completed.get("stage") == node["stage"],
            f"completion identity drift: {node_id}",
        )
        _require(
            completed.get("command") == resolved_command
            and completed.get("command_sha256") == command_digest,
            f"completion command drift: {node_id}",
        )
        outputs = {
            str(relative): _sha256(bundle / str(relative))
            for relative in node["expected_outputs"]
            if (bundle / str(relative)).is_file()
        }
        _require(
            len(outputs) == len(node["expected_outputs"]),
            f"missing registered output: {node_id}",
        )
        _require(
            completed.get("outputs") == outputs,
            f"registered output digest drift: {node_id}",
        )
        planned_device = str(node["device"])
        completed_elapsed = float(completed.get("elapsed_seconds", -1.0))
        _require(
            math.isfinite(completed_elapsed) and completed_elapsed >= 0.0,
            f"completion elapsed time is invalid: {node_id}",
        )
        if planned_device.startswith("cuda"):
            aggregate_gpu_seconds += completed_elapsed
        resource = completed.get("producer_resource_summary")
        if isinstance(resource, dict):
            device = str(resource.get("resolved_device", ""))
            elapsed = float(resource.get("elapsed_seconds", 0.0))
            _require(
                device == planned_device,
                f"producer device differs from launch plan: {node_id}",
            )
            _require(
                math.isfinite(elapsed) and elapsed >= 0.0,
                f"producer elapsed time is invalid: {node_id}",
            )
            reserved = int(resource.get("peak_gpu_reserved_bytes", 0))
            _require(
                reserved <= RESOURCE_BUDGET["process_reserved_memory_cap_bytes"],
                f"GPU reserve cap exceeded: {node_id}",
            )
        elif planned_device.startswith("cuda"):
            _require(
                node["stage"] == "source-radius",
                f"GPU producer resource summary is absent: {node_id}",
            )
    aggregate_gpu_hours = aggregate_gpu_seconds / 3600.0
    _require(
        aggregate_gpu_hours <= RESOURCE_BUDGET["hard_aggregate_gpu_hours"],
        "aggregate GPU-hour cap exceeded",
    )
    return len(nodes), aggregate_gpu_hours




def _verify_manifest(bundle: Path) -> int:
    manifest_path = bundle / "MANIFEST.sha256"
    _require(manifest_path.is_file(), "bundle manifest is absent")
    actual_files = sorted(
        path for path in bundle.rglob("*")
        if path.is_file() and path != manifest_path
    )
    expected_lines = [
        f"{_sha256(path)}  {path.relative_to(bundle).as_posix()}"
        for path in actual_files
    ]
    recorded = manifest_path.read_text(encoding="ascii")
    _require(
        recorded == "\n".join(expected_lines) + "\n",
        "bundle manifest membership or digest drifted",
    )
    total_bytes = sum(path.stat().st_size for path in actual_files)
    _require(
        total_bytes <= RESOURCE_BUDGET["persistent_output_cap_bytes"],
        "persistent evidence cap exceeded",
    )
    return total_bytes


def _verify_manuscript_binding(
    bundle: Path,
    *,
    aggregate_gpu_hours: float,
    total_bytes: int,
) -> None:
    source_path = ROOT / "docs" / "confirmation_program_rev46.tex"
    source = source_path.read_text(encoding="utf-8")
    required = {
        _sha256(bundle / "reports" / "final-analysis.json"):
            "final-analysis digest",
        _sha256(bundle / "MANIFEST.sha256"): "bundle-manifest digest",
        f"{aggregate_gpu_hours:.3f} aggregate GPU-hours":
            "aggregate GPU-hour value",
        f"{total_bytes / 1024**3:.3f} GiB": "persistent footprint",
    }
    for literal, description in required.items():
        _require(
            literal in source,
            f"manuscript does not bind the current {description}",
        )




