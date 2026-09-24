#!/usr/bin/env python3
"""Lock and analyze the fresh-context radial-predictor holdout."""

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

from confirm.resource_executor import resource_records, record_is_admissible

from analysis.analyze_direct_work_precision import (  # noqa: E402
    array,
    canonical_bytes,
    scalar,
    sha256_path,
    write_bytes,
)
from analysis.analyze_direct_work_replication import (  # noqa: E402
    MAP_SHAPE,
    arm_metrics,
    scalar_invariants,
)
from analysis.radial_tangential import paired_decomposition, radial_decomposition  # noqa: E402
from confirm.radial_context_holdout_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CAMPAIGN_ID,
    DECISION_POLICY,
    LAYERS,
    LOCK_SCHEMA,
    RECORD_SCHEMA,
    RELEASE_ID,
    SOURCE_STEPS,
)


ARMS = ("natural", "control")
TERM_NAMES = (
    "radial_linear",
    "radial_charge",
    "tangent_interaction",
    "tangent_charge",
)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_bytes(path, canonical_bytes(value))


def digest_payload(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def qualification(root: Path) -> dict[str, Any]:
    rows = []
    for index in (0, 1):
        path = root / "records" / f"qualification-cuda{index}.npz"
        with np.load(path, allow_pickle=False) as record:
            if (
                scalar(record, "schema_version", str) != RECORD_SCHEMA
                or scalar(record, "campaign_id", str) != CAMPAIGN_ID
                or scalar(record, "release_id", str) != RELEASE_ID
            ):
                raise ValueError("radial holdout qualification identity changed")
            rows.append({
                "path": path,
                "sha256": sha256_path(path),
                "device": scalar(record, "device", str),
                "registry_sha256": scalar(record, "registry_sha256", str),
                "checkpoint_sha256": scalar(record, "checkpoint_sha256", str),
                "source_z": array(record, "source_z", MAP_SHAPE),
                "source_energy": array(record, "source_energy", (24, 4)),
                "natural_endpoint_energy": array(
                    record, "natural_endpoint_energy", (24, 4)
                ),
                "control_endpoint_energy": array(
                    record, "control_endpoint_energy", (24, 4)
                ),
                "natural_endpoint_z": array(
                    record, "natural_endpoint_z", MAP_SHAPE
                ),
                "control_endpoint_z": array(
                    record, "control_endpoint_z", MAP_SHAPE
                ),
                "duplicate": scalar(
                    record, "qualification_duplicate_max_abs", float
                ),
            })
    if {row["device"] for row in rows} != {"cuda:0", "cuda:1"}:
        raise ValueError("radial holdout qualification omitted a GPU")
    for name in ("registry_sha256", "checkpoint_sha256"):
        if len({row[name] for row in rows}) != 1:
            raise ValueError(f"radial holdout qualification differs at {name}")
    cross_energy = max(
        float(np.max(np.abs(rows[0][name] - rows[1][name])))
        for name in (
            "source_energy", "natural_endpoint_energy", "control_endpoint_energy"
        )
    )
    cross_observer = max(
        float(np.max(np.abs(rows[0][name] - rows[1][name])))
        for name in ("source_z", "natural_endpoint_z", "control_endpoint_z")
    )
    within = max(float(row["duplicate"]) for row in rows)
    effect_floor = max(1.0e-12, 20.0 * within, 20.0 * cross_energy)
    return {
        "effect_floor": effect_floor,
        "within_device_duplicate_energy_max_abs": within,
        "cross_device_energy_max_abs": cross_energy,
        "cross_device_observer_max_abs": cross_observer,
        "cross_device_observer_bitwise": all(
            np.array_equal(rows[0][name], rows[1][name])
            for name in ("source_z", "natural_endpoint_z", "control_endpoint_z")
        ),
        "records": [
            {
                "path": str(row["path"].resolve()),
                "sha256": row["sha256"],
                "device": row["device"],
            }
            for row in rows
        ],
    }


def unit_path(root: Path, trajectory: str, step: int, layer: int) -> Path:
    return root / "records" / f"{trajectory}-S{step}-L{layer}.npz"


def load_unit(path: Path, effect_floor: float) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as record:
        if (
            scalar(record, "schema_version", str) != RECORD_SCHEMA
            or scalar(record, "campaign_id", str) != CAMPAIGN_ID
            or scalar(record, "release_id", str) != RELEASE_ID
        ):
            raise ValueError(f"unknown radial holdout record: {path}")
        source = array(record, "source_z", MAP_SHAPE)
        source_flat = source.reshape(96, -1)
        source_energy = array(record, "source_energy", (24, 4)).reshape(-1)
        stored_contrast = array(record, "intervention_contrast", (24, 4)).reshape(-1)
        scalar_pass = scalar_invariants(record)
        arm_rows = {}
        for arm in ARMS:
            technical = arm_metrics(record, arm, source)
            radial = radial_decomposition(
                source_flat,
                technical["increment"].reshape(96, -1),
                energy_floor=effect_floor,
            )
            arm_rows[arm] = {"technical": technical, "radial": radial}
        paired = paired_decomposition(
            source_flat,
            arm_rows["natural"]["technical"]["increment"].reshape(96, -1),
            arm_rows["control"]["technical"]["increment"].reshape(96, -1),
            energy_floor=effect_floor,
        )
        source_eligible = source_energy > effect_floor
        technical = (
            arm_rows["natural"]["technical"]["technical_mask"]
            & arm_rows["control"]["technical"]["technical_mask"]
        )
        if not scalar_pass:
            technical[:] = False
        radial_residual = np.maximum(
            np.abs(arm_rows["natural"]["radial"]["gain_residual"]),
            np.abs(arm_rows["control"]["radial"]["gain_residual"]),
        )
        paired_residual = (
            stored_contrast / source_energy
            - np.sum(paired["normalized_terms"], axis=1)
        )
        radial_closure = np.zeros(96, dtype=bool)
        radial_closure[source_eligible] = (
            radial_residual[source_eligible] <= 1.0e-12
        ) & (np.abs(paired_residual[source_eligible]) <= 1.0e-12)
        eligible = source_eligible & technical & radial_closure
        radial_energy = paired["radial_normalized"] * source_energy
        full_raw_sign = np.sign(stored_contrast)
        radial_raw_sign = np.sign(radial_energy)
        raw_agree = (
            eligible
            & (full_raw_sign != 0)
            & (radial_raw_sign != 0)
            & (full_raw_sign == radial_raw_sign)
        )
        full_floor_sign = np.zeros(96, dtype=np.int8)
        full_floor_sign[stored_contrast > effect_floor] = 1
        full_floor_sign[stored_contrast < -effect_floor] = -1
        radial_floor_sign = np.zeros(96, dtype=np.int8)
        radial_floor_sign[radial_energy > effect_floor] = 1
        radial_floor_sign[radial_energy < -effect_floor] = -1
        floor_comparable = (
            eligible & (full_floor_sign != 0) & (radial_floor_sign != 0)
        )
        floor_agree = floor_comparable & (full_floor_sign == radial_floor_sign)
        terms = paired["normalized_terms"]
        dominant = np.argmax(np.abs(terms), axis=1)
        count = int(np.count_nonzero(eligible))
        raw_agreements = int(np.count_nonzero(raw_agree))
        floor_comparisons = int(np.count_nonzero(floor_comparable))
        floor_agreements = int(np.count_nonzero(floor_agree))
        return {
            "trajectory": scalar(record, "trajectory", str),
            "step": scalar(record, "source_step", int),
            "layer": scalar(record, "layer", int),
            "role": scalar(record, "role", str),
            "record_sha256": sha256_path(path),
            "registry_sha256": scalar(record, "registry_sha256", str),
            "checkpoint_sha256": scalar(record, "checkpoint_sha256", str),
            "planned_maps": 96,
            "source_eligible_maps": int(np.count_nonzero(source_eligible)),
            "radial_eligible_maps": count,
            "raw_sign_agreements": raw_agreements,
            "raw_sign_disagreements": count - raw_agreements,
            "raw_sign_agreement_fraction": (
                float(raw_agreements / count) if count else None
            ),
            "raw_zero_sign_maps": int(np.count_nonzero(
                eligible & ((full_raw_sign == 0) | (radial_raw_sign == 0))
            )),
            "floor_sign_comparisons": floor_comparisons,
            "floor_sign_agreements": floor_agreements,
            "floor_sign_agreement_fraction": (
                float(floor_agreements / floor_comparisons)
                if floor_comparisons else None
            ),
            "dominant_terms": {
                name: int(np.count_nonzero(eligible & (dominant == index)))
                for index, name in enumerate(TERM_NAMES)
            },
            "maximum_gain_identity_residual": (
                float(np.max(radial_residual[source_eligible]))
                if np.any(source_eligible) else 0.0
            ),
            "maximum_paired_identity_residual": (
                float(np.max(np.abs(paired_residual[source_eligible])))
                if np.any(source_eligible) else 0.0
            ),
            "scalar_invariants_pass": bool(scalar_pass),
            "native_technical_maps": int(np.count_nonzero(technical)),
            "all_native_technical_maps_pass": bool(np.all(technical)),
            "all_radial_identities_pass_where_defined": bool(
                np.all(radial_closure[source_eligible])
            ),
            "producer_peak_gpu_allocated_bytes": scalar(
                record, "peak_gpu_allocated_bytes", int
            ),
            "producer_peak_gpu_reserved_bytes": scalar(
                record, "peak_gpu_reserved_bytes", int
            ),
        }


def construction_lock(root: Path) -> dict[str, Any]:
    q0 = qualification(root)
    effect_floor = float(q0["effect_floor"])
    units = [
        load_unit(unit_path(root, "A", step, layer), effect_floor)
        for step in SOURCE_STEPS for layer in LAYERS
    ]
    if any(unit["trajectory"] != "A" or unit["role"] != "construction" for unit in units):
        raise ValueError("construction lock opened a nonconstruction record")
    technical = bool(all(
        unit["all_native_technical_maps_pass"]
        and unit["all_radial_identities_pass_where_defined"]
        for unit in units
    ))
    payload = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "release_id": RELEASE_ID,
        "decision_policy": DECISION_POLICY,
        "qualification": q0,
        "effect_floor": effect_floor,
        "construction_sets_no_outcome_threshold": True,
        "construction_technical_pass": technical,
        "construction_units": units,
    }
    payload["content_sha256"] = digest_payload(payload)
    write_json(root / "records/construction-lock.json", payload)
    print(json.dumps({
        "effect_floor": effect_floor,
        "construction_technical_pass": technical,
        "construction_eligible_maps": sum(
            unit["radial_eligible_maps"] for unit in units
        ),
    }, sort_keys=True))
    return payload


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    digest = value.pop("content_sha256", None)
    expected = digest_payload(value)
    value["content_sha256"] = digest
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("release_id") != RELEASE_ID
        or value.get("decision_policy") != DECISION_POLICY
        or value.get("construction_sets_no_outcome_threshold") is not True
        or value.get("construction_technical_pass") is not True
        or digest != expected
    ):
        raise ValueError("radial holdout construction lock changed")
    return value


def resource_summary(root: Path, units: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for path in resource_records(root):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "executor_contract" in value and not record_is_admissible(value):
            raise ValueError(f"resource admission failed: {path}")
        if str(value.get("device", "")).startswith("cuda:"):
            records.append(value)
    return {
        "gpu_resource_records_available_at_analysis": len(records),
        "aggregate_reserved_device_hours": float(sum(
            float(row["accelerator_reservation_wall_seconds"]) for row in records
        ) / 3600.0),
        "peak_producer_allocated_bytes": max(
            unit["producer_peak_gpu_allocated_bytes"] for unit in units
        ),
        "peak_producer_reserved_bytes": max(
            unit["producer_peak_gpu_reserved_bytes"] for unit in units
        ),
        "process_query_status_counts": {
            status: sum(
                (row.get("gpu_probe_observation") or {}).get(
                    "process_query_status", "unavailable"
                ) == status
                for row in records
            )
            for status in sorted({
                (row.get("gpu_probe_observation") or {}).get(
                    "process_query_status", "unavailable"
                )
                for row in records
            })
        },
        "producer_authoritative_monitor_incidents": sum(
            row.get("gpu_monitor_status") == "unavailable" for row in records
        ),
    }


def decide_outcome(
    *,
    technical: bool,
    eligible_counts: list[int],
    cell_agreement_fractions: list[float | None],
    overall_agreement_fraction: float | None,
) -> tuple[str, list[int], list[int]]:
    """Apply the frozen E1 rule without any pooled fallback."""

    if len(eligible_counts) != len(cell_agreement_fractions) or not eligible_counts:
        raise ValueError("radial holdout decision cells are malformed")
    minimum_maps = int(
        DECISION_POLICY["minimum_radial_eligible_maps_per_heldout_cell"]
    )
    inadequate = [
        index for index, count in enumerate(eligible_counts)
        if int(count) < minimum_maps
    ]
    low_powered = [
        index
        for index, (count, fraction) in enumerate(
            zip(eligible_counts, cell_agreement_fractions, strict=True)
        )
        if int(count) >= minimum_maps
        and (
            fraction is None
            or float(fraction)
            < DECISION_POLICY[
                "minimum_powered_cell_raw_sign_agreement_fraction"
            ]
        )
    ]
    if not technical or inadequate or overall_agreement_fraction is None:
        return "insufficient", inadequate, low_powered
    supported = bool(
        overall_agreement_fraction
        >= DECISION_POLICY["minimum_overall_raw_sign_agreement_fraction"]
        and not low_powered
    )
    return ("supported" if supported else "refuted"), inadequate, low_powered


def final_report(root: Path) -> dict[str, Any]:
    lock = load_lock(root / "records/construction-lock.json")
    floor = float(lock["effect_floor"])
    construction = list(lock["construction_units"])
    heldout = [
        load_unit(unit_path(root, trajectory, step, layer), floor)
        for trajectory in ("B", "C")
        for step in SOURCE_STEPS for layer in LAYERS
    ]
    if any(unit["role"] != "heldout" for unit in heldout):
        raise ValueError("radial holdout analysis opened a nonheldout record")
    minimum_maps = int(
        DECISION_POLICY["minimum_radial_eligible_maps_per_heldout_cell"]
    )
    for unit in heldout:
        unit["adequately_powered"] = unit["radial_eligible_maps"] >= minimum_maps
        unit["below_powered_cell_agreement_floor"] = bool(
            unit["adequately_powered"]
            and unit["raw_sign_agreement_fraction"]
            < DECISION_POLICY[
                "minimum_powered_cell_raw_sign_agreement_fraction"
            ]
        )
    all_units = construction + heldout
    technical = bool(all(
        unit["all_native_technical_maps_pass"]
        and unit["all_radial_identities_pass_where_defined"]
        for unit in all_units
    ))
    eligible = sum(unit["radial_eligible_maps"] for unit in heldout)
    agreements = sum(unit["raw_sign_agreements"] for unit in heldout)
    overall = float(agreements / eligible) if eligible else None
    outcome, inadequate_indices, below_indices = decide_outcome(
        technical=technical,
        eligible_counts=[unit["radial_eligible_maps"] for unit in heldout],
        cell_agreement_fractions=[
            unit["raw_sign_agreement_fraction"] for unit in heldout
        ],
        overall_agreement_fraction=overall,
    )
    inadequate = [
        f"{heldout[index]['trajectory']}-S{heldout[index]['step']}-L{heldout[index]['layer']}"
        for index in inadequate_indices
    ]
    below = [
        f"{heldout[index]['trajectory']}-S{heldout[index]['step']}-L{heldout[index]['layer']}"
        for index in below_indices
    ]
    eligibility_adequate = not inadequate
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "release_id": RELEASE_ID,
        "construction_lock_sha256": sha256_path(
            root / "records/construction-lock.json"
        ),
        "decision_policy": DECISION_POLICY,
        "effect_floor": floor,
        "outcome": outcome,
        "technical_identity_pass": technical,
        "heldout_eligibility_adequate": eligibility_adequate,
        "heldout_inadequate_cells": inadequate,
        "heldout_powered_cells_below_agreement_floor": below,
        "heldout_planned_cells": len(heldout),
        "heldout_powered_cells": sum(unit["adequately_powered"] for unit in heldout),
        "heldout_planned_maps": len(heldout) * 96,
        "heldout_radial_eligible_maps": eligible,
        "heldout_raw_sign_agreements": agreements,
        "heldout_raw_sign_disagreements": eligible - agreements,
        "heldout_overall_raw_sign_agreement_fraction": overall,
        "heldout_floor_sign_comparisons": sum(
            unit["floor_sign_comparisons"] for unit in heldout
        ),
        "heldout_floor_sign_agreements": sum(
            unit["floor_sign_agreements"] for unit in heldout
        ),
        "heldout_dominant_terms": {
            name: sum(unit["dominant_terms"][name] for unit in heldout)
            for name in TERM_NAMES
        },
        "maximum_gain_identity_residual": max(
            unit["maximum_gain_identity_residual"] for unit in all_units
        ),
        "maximum_paired_identity_residual": max(
            unit["maximum_paired_identity_residual"] for unit in all_units
        ),
        "construction_units": construction,
        "heldout_units": heldout,
        "record_sha256": {
            f"{unit['trajectory']}-S{unit['step']}-L{unit['layer']}.npz": unit[
                "record_sha256"
            ]
            for unit in all_units
        },
        "resources": resource_summary(root, all_units),
        "inferential_scope": {
            "state_conditional_predictor_only": True,
            "persistent_intervention_sign_not_tested": True,
            "eventual_contraction_not_inferred": True,
            "zero_face_nonevaluability_is_an_outcome": True,
        },
    }
    report["content_sha256"] = digest_payload(report)
    return report


def tex_integer(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def tex_scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    return rf"{value / 10.0**exponent:.3g}\times 10^{{{exponent}}}"








