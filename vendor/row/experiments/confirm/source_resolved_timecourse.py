"""Assemble a strict finite time course from native trainer observations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from confirm.confirmation_artifacts import sha256_path
from confirm.source_resolved_energy import select_plga_attenuation_cap
from confirm.source_resolved_specs import TIMEPOINT_SCHEMA
from confirm.source_resolved_specs import (
    CAMPAIGN_ID,
    FROZEN_POLICY,
    REGISTRY,
    TIMECOURSE_SCHEMA,
    snapshot_updates,
)
from confirm.source_resolved_live import _schema
from confirm.strict_schema import load_json, validate


def _strict_json_line(value: str) -> dict[str, Any]:
    def duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON value {value}")

    result = json.loads(
        value,
        object_pairs_hook=duplicate_guard,
        parse_constant=reject_constant,
    )
    if not isinstance(result, dict):
        raise ValueError("trainer ledger rows must be objects")
    return result


def load_timepoints(
    log_path: str | Path,
    *,
    protocol_directory: str | Path,
) -> tuple[list[dict[str, Any]], str, str]:
    """Read and schema-check only native source-resolved trainer events."""

    log_file = Path(log_path).resolve()
    points: dict[int, dict[str, Any]] = {}
    run_ids: set[str] = set()
    lineage_ids: set[str] = set()
    schema = _schema(protocol_directory, "timepoint.schema.json")
    with log_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = _strict_json_line(line)
            point = record.get("source_resolved_timepoint")
            if point is None:
                continue
            validate(point, schema)
            step = int(point["step"])
            if step in points:
                raise ValueError(f"duplicate source-resolved timepoint {step}")
            if record.get("step") != step:
                raise ValueError("trainer event and timepoint clocks disagree")
            points[step] = point
            run_ids.add(str(record.get("run_id", "")))
            lineage_ids.add(str(record.get("lineage_root", "")))
    if len(run_ids) != 1 or "" in run_ids or len(lineage_ids) != 1 or "" in lineage_ids:
        raise ValueError("timepoints do not have one resolved run lineage")
    expected = snapshot_updates()
    if sorted(points) != expected:
        missing = sorted(set(expected) - set(points))
        extra = sorted(set(points) - set(expected))
        raise ValueError(f"timepoint cadence differs; missing={missing}, extra={extra}")
    return [points[step] for step in expected], run_ids.pop(), lineage_ids.pop()


def _map_identities(registry: dict[str, Any]) -> list[dict[str, Any]]:
    contexts = list(registry["construction"]) + list(registry["validation"])
    result = []
    for context in contexts:
        for layer in REGISTRY["layers"]:
            for head in REGISTRY["heads"]:
                result.append({
                    "map_id": f"{context['id']}-l{layer}-h{head}",
                    "context": str(context["id"]),
                    "layer": int(layer),
                    "head": int(head),
                })
    if len(result) != REGISTRY["map_count"]:
        raise ValueError("timecourse registry map population changed")
    return result


def produce_timecourse(
    log_path: str | Path,
    registry_path: str | Path,
    binding_paths: list[str | Path],
    policy_path: str | Path,
    role: str,
    *,
    protocol_directory: str | Path,
    selected_plga_cap: float | None = None,
) -> dict[str, Any]:
    """Derive every gain and map record from the native online observations."""

    if role not in {"construction", "heldout"}:
        raise ValueError("timecourse role must be construction or heldout")
    points, run_id, lineage_id = load_timepoints(
        log_path, protocol_directory=protocol_directory
    )
    registry = load_json(registry_path)
    expected_digest = registry.get("registry_sha256")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("timecourse registry is not content addressed")
    if any(point["registry_sha256"] != expected_digest for point in points):
        raise ValueError("a timepoint changed the frozen registry")
    policy = load_json(policy_path)
    if policy.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("timecourse policy belongs to another campaign")

    binding_schema = _schema(
        protocol_directory, "checkpoint_binding.schema.json"
    )
    bindings = []
    binding_digests = []
    for raw_path in binding_paths:
        path = Path(raw_path).resolve()
        binding = load_json(path)
        validate(binding, binding_schema)
        if not binding["technical_valid"]:
            raise ValueError("timecourse received an invalid checkpoint binding")
        bindings.append(binding)
        binding_digests.append(sha256_path(path))
    if not bindings:
        raise ValueError("timecourse needs at least one complete checkpoint binding")
    trajectories = {binding["trajectory"] for binding in bindings}
    seeds = {binding["seed"] for binding in bindings}
    if len(trajectories) != 1 or len(seeds) != 1:
        raise ValueError("timecourse checkpoint lineage is inconsistent")
    trajectory = trajectories.pop()
    seed = seeds.pop()
    if run_id != trajectory:
        raise ValueError("trainer run and checkpoint trajectory disagree")

    identities = _map_identities(registry)
    updates = snapshot_updates()
    energy = np.asarray([point["energy"] for point in points], dtype=np.float64)
    diameter = np.sqrt(np.asarray(
        [point["diameter_squared"] for point in points], dtype=np.float64
    ))
    coordinates = np.asarray(
        [point["gate_shape_coordinate_energy"] for point in points],
        dtype=np.float64,
    )
    normalized_coordinates = np.asarray(
        [point["normalized_shape_coordinate_energy"] for point in points],
        dtype=np.float64,
    )
    layer_gates = np.asarray(
        [point["layer_final_gate"] for point in points], dtype=np.float64
    )
    factorization_residuals = np.asarray(
        [point["coordinate_factorization_relative_residual"] for point in points],
        dtype=np.float64,
    )
    ratios = np.asarray(
        [point["plga_quotient_ratio"] for point in points], dtype=np.float64
    )
    forces = np.asarray([point["plga_force"] for point in points], dtype=np.float64)
    expected_shape = (len(updates), REGISTRY["map_count"])
    if (
        energy.shape != expected_shape
        or diameter.shape != expected_shape
        or coordinates.shape != expected_shape + (REGISTRY["features_per_row"],)
        or normalized_coordinates.shape != expected_shape + (REGISTRY["features_per_row"],)
        or layer_gates.shape != (len(updates), 3, REGISTRY["features_per_row"])
        or factorization_residuals.shape != (len(updates),)
        or ratios.shape != expected_shape
        or forces.shape != expected_shape
        or not all(np.isfinite(value).all() for value in (
            energy, diameter, coordinates, normalized_coordinates,
            layer_gates, factorization_residuals, ratios, forces
        ))
    ):
        raise ValueError("timecourse point arrays have invalid shape or values")

    gains = np.zeros_like(energy)
    gain_defined = np.zeros_like(energy, dtype=bool)
    for index in range(1, len(updates)):
        positive = energy[index - 1] > 0.0
        gains[index, positive] = energy[index, positive] / energy[index - 1, positive]
        gain_defined[index, positive] = True
    replay = np.where(gain_defined[1:], energy[:-1] * gains[1:], energy[1:])
    no_omission = bool(np.allclose(
        replay, energy[1:], rtol=1.0e-12, atol=1.0e-12
    ))
    initial_positive = energy[0] > 0.0
    cumulative_gain = np.divide(
        energy, energy[0][None], out=np.zeros_like(energy),
        where=initial_positive[None],
    )
    cumulative_replays = bool(np.allclose(
        energy[:, initial_positive],
        energy[0, initial_positive][None]
        * cumulative_gain[:, initial_positive],
        rtol=1.0e-12, atol=1.0e-12,
    ))

    if role == "construction":
        selection = select_plga_attenuation_cap(ratios)
        cap = selection["selected_cap"]
        if selected_plga_cap is not None and cap != selected_plga_cap:
            raise ValueError("caller-selected PLGA cap differs from construction")
    else:
        cap = (
            policy.get("plga_selected_cap")
            if selected_plga_cap is None else selected_plga_cap
        )
        if cap not in {0.05, 0.10, 0.20, 0.50, None}:
            raise ValueError("heldout PLGA cap is outside the frozen menu")

    records = []
    for time_index, update in enumerate(updates):
        for map_index, identity in enumerate(identities):
            records.append({
                **identity,
                "update": int(update),
                "energy": float(energy[time_index, map_index]),
                "diameter": float(diameter[time_index, map_index]),
                "gate_shape_coordinate_energy": coordinates[
                    time_index, map_index
                ].tolist(),
                "normalized_shape_coordinate_energy": normalized_coordinates[
                    time_index, map_index
                ].tolist(),
                "source_gain_from_previous": float(gains[time_index, map_index]),
                "gain_defined": bool(gain_defined[time_index, map_index]),
                "cumulative_gain": (
                    float(cumulative_gain[time_index, map_index])
                    if initial_positive[map_index] else None
                ),
                "cumulative_log_gain": (
                    float(np.log(cumulative_gain[time_index, map_index]))
                    if initial_positive[map_index]
                    and cumulative_gain[time_index, map_index] > 0.0
                    else None
                ),
                "plga_quotient_ratio": float(ratios[time_index, map_index]),
                "plga_force": float(forces[time_index, map_index]),
            })
    gate_records = [
        {
            "update": int(update),
            "layer_final_gate": layer_gates[index].tolist(),
            "coordinate_factorization_relative_residual": float(
                factorization_residuals[index]
            ),
        }
        for index, update in enumerate(updates)
    ]
    policy_precedes = True
    if role == "heldout":
        created = policy.get("created_at_ns")
        policy_precedes = isinstance(created, int) and created < min(
            binding["created_at_ns"] for binding in bindings
        )
    checks = {
        "cadence_exact": [point["step"] for point in points] == updates,
        "terminal_present": points[-1]["step"] == 9000,
        "map_population_complete": len(records) == len(updates) * REGISTRY["map_count"],
        "no_omitted_gain_factor": no_omission,
        "cumulative_gain_replays": cumulative_replays,
        "coordinate_factorization_closed": bool(np.max(
            factorization_residuals
        ) <= FROZEN_POLICY["float32_factorization_relative_tolerance"]),
        "policy_precedes_heldout": policy_precedes,
        "all_values_finite": True,
    }
    result = {
        "schema_version": TIMECOURSE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "trajectory": trajectory,
        "seed": int(seed),
        "role": role,
        "policy_sha256": sha256_path(policy_path),
        "updates": updates,
        "checkpoint_binding_sha256": binding_digests,
        "map_records": records,
        "gate_records": gate_records,
        "plga_selected_cap": cap,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(result, _schema(protocol_directory, "timecourse.schema.json"))
    if not result["technical_valid"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"timecourse checks failed: {failed}")
    return result
