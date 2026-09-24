#!/usr/bin/env python3
"""Analyze the complete layer-resolved row-map cocycle confirmation."""

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
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT))

from confirm.layer_cocycle_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CAMPAIGN_ID,
    GATE_OPERATOR_POLICY,
    GATE_OPERATOR_SCHEMA,
    INTERVENTION_POLICY,
    INTERVENTION_SCHEMA,
    PERTURBATION_POLICY,
    PERTURBATION_SCHEMA,
    PLGA_POLICY,
    PLGA_SCHEMA,
    PRECISION_SCHEMA,
    RESOLUTION_POLICY,
    TRAJECTORIES,
)


SCHEMAS = {
    "precision": PRECISION_SCHEMA,
    "gate_operator": GATE_OPERATOR_SCHEMA,
    "intervention": INTERVENTION_SCHEMA,
    "perturbation": PERTURBATION_SCHEMA,
    "plga": PLGA_SCHEMA,
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def compact_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _scalar(record: dict[str, np.ndarray], name: str) -> Any:
    value = record[name]
    if value.ndim != 0:
        raise ValueError(f"{name} is not scalar")
    return value.item()


def load_record(path: Path, schema: str) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"declared record is absent: {path}")
    with np.load(path, allow_pickle=False) as archive:
        record = {name: np.array(archive[name]) for name in archive.files}
    if _scalar(record, "schema_version") != schema:
        raise ValueError(f"record schema drifted: {path}")
    if _scalar(record, "campaign_id") != CAMPAIGN_ID:
        raise ValueError(f"record campaign drifted: {path}")
    return record


def _finite(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _median_log10_gain(target: np.ndarray, source: np.ndarray) -> float:
    tiny = np.finfo(np.float64).tiny
    return float(np.median(np.log10(np.maximum(target, tiny) / np.maximum(source, tiny))))


def _aggregate_norm(per_map_norm: np.ndarray) -> float:
    value = _finite(per_map_norm, "per-map response")
    return float(np.sqrt(np.mean(value * value)))


def _label_maps() -> tuple[dict[str, str], dict[str, str]]:
    label_to_name = {str(row["label"]): str(row["name"]) for row in TRAJECTORIES}
    name_to_label = {value: key for key, value in label_to_name.items()}
    return label_to_name, name_to_label


def _validate_common(
    record: dict[str, np.ndarray],
    *,
    label: str,
    step: int,
    binding: dict[str, Any],
) -> None:
    label_to_name, _name_to_label = _label_maps()
    if _scalar(record, "trajectory") != label_to_name[label]:
        raise ValueError(f"trajectory label drifted: {label}/{step}")
    if int(_scalar(record, "step")) != step:
        raise ValueError(f"record step drifted: {label}/{step}")
    if _scalar(record, "checkpoint_sha256") != binding["checkpoints"][label][str(step)]["sha256"]:
        raise ValueError(f"checkpoint digest drifted in record: {label}/{step}")


def precision_analysis(
    bundle: Path,
    inventory: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    records: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    factorization_max = 0.0
    for item in inventory["precision"]:
        label = str(item["trajectory"])
        step = int(item["step"])
        record = load_record(bundle / item["path"], PRECISION_SCHEMA)
        _validate_common(record, label=label, step=step, binding=binding)
        expected_shape = (24, 3, 4)
        for name in (
            "float64_energy",
            "float64_shape_energy",
            "float64_effective_squared_gate",
            "resolution_relative_resolved",
        ):
            if record[name].shape != expected_shape:
                raise ValueError(f"precision shape drifted: {item['path']}:{name}")
        residual = float(_scalar(record, "float64_factorization_relative_residual"))
        factorization_max = max(factorization_max, residual)
        if residual > float(RESOLUTION_POLICY["float64_factorization_relative_tolerance"]):
            raise ValueError("float64 gate-shape factorization regression failed")
        records[label, step] = record
    rows = []
    threshold = float(RESOLUTION_POLICY["finite_registry_energy_threshold"])
    small_units = 0
    unresolved_coordinates = 0
    for trajectory in TRAJECTORIES:
        label = str(trajectory["label"])
        source = records[label, 16_384]
        target = records[label, 65_536]
        for layer in range(3):
            source_energy = _finite(source["float64_energy"][:, layer, :], "source energy")
            target_energy = _finite(target["float64_energy"][:, layer, :], "target energy")
            source_shape = _finite(source["float64_shape_energy"][:, layer, :], "source shape")
            target_shape = _finite(target["float64_shape_energy"][:, layer, :], "target shape")
            source_gate = _finite(
                source["float64_effective_squared_gate"][:, layer, :], "source gate"
            )
            target_gate = _finite(
                target["float64_effective_squared_gate"][:, layer, :], "target gate"
            )
            endpoint = float(np.median(target_energy))
            is_small = endpoint < threshold
            small_units += int(is_small)
            resolved = np.asarray(
                target["resolution_relative_resolved"][:, layer, :], dtype=bool
            )
            unresolved_coordinates += int(np.size(resolved) - np.count_nonzero(resolved))
            rows.append(
                {
                    "trajectory": label,
                    "layer": layer,
                    "source_median_energy": float(np.median(source_energy)),
                    "endpoint_median_energy": endpoint,
                    "endpoint_minimum_energy": float(np.min(target_energy)),
                    "endpoint_maximum_energy": float(np.max(target_energy)),
                    "late_log10_energy_gain": _median_log10_gain(target_energy, source_energy),
                    "late_log10_shape_gain": _median_log10_gain(target_shape, source_shape),
                    "late_log10_effective_gate_gain": _median_log10_gain(target_gate, source_gate),
                    "small_at_endpoint": is_small,
                    "float32_unresolved_map_count": int(np.size(resolved) - np.count_nonzero(resolved)),
                }
            )
    return {
        "rows": rows,
        "small_path_layer_units": small_units,
        "total_path_layer_units": len(rows),
        "endpoint_small_threshold": threshold,
        "float32_unresolved_map_count": unresolved_coordinates,
        "maximum_float64_factorization_relative_residual": factorization_max,
    }


def gate_operator_analysis(
    bundle: Path,
    inventory: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    autodiff_tolerance = float(GATE_OPERATOR_POLICY["autodiff_relative_tolerance"])
    probe_tolerance = float(
        GATE_OPERATOR_POLICY["maximum_smaller_probe_relative_residual"]
    )
    reduction_minimum = float(GATE_OPERATOR_POLICY["minimum_probe_residual_reduction"])
    defect_tolerance = float(
        GATE_OPERATOR_POLICY["scaled_invariance_defect_zero_tolerance"]
    )
    for item in inventory["gate_operator"]:
        label = str(item["trajectory"])
        layer = int(item["layer"])
        step = int(item["step"])
        record = load_record(bundle / item["path"], GATE_OPERATOR_SCHEMA)
        _validate_common(record, label=label, step=step, binding=binding)
        if int(_scalar(record, "layer")) != layer:
            raise ValueError("gate-operator layer drifted")
        analytic = _finite(record["analytic_operator"], "analytic operator")
        autodiff = _finite(record["autodiff_operator"], "autodiff operator")
        if analytic.shape != (192, 192) or autodiff.shape != analytic.shape:
            raise ValueError("lifted operator dimension drifted")
        autodiff_residual = float(_scalar(record, "operator_relative_residual"))
        gradient_residual = float(
            _scalar(record, "gradient_aggregation_relative_residual")
        )
        if autodiff_residual > autodiff_tolerance or gradient_residual > autodiff_tolerance:
            raise ValueError("implemented AdamW derivative cross-check failed")
        probes = _finite(record["probe_relative_residuals"], "probe residuals")
        if probes.shape != (4, 2, 2):
            raise ValueError("signed probe grid drifted")
        large = float(np.max(probes[:, 0, :]))
        small = float(np.max(probes[:, 1, :]))
        probe_pass = small <= probe_tolerance and large >= reduction_minimum * small
        operator_norm = float(_scalar(record, "scaled_operator_norm"))
        spectral_radius = float(_scalar(record, "scaled_spectral_radius"))
        defect = float(_scalar(record, "scaled_invariance_defect"))
        rows.append(
            {
                "trajectory": label,
                "layer": layer,
                "scaled_operator_norm": operator_norm,
                "scaled_spectral_radius": spectral_radius,
                "scaled_nonnormality_ratio": float(
                    _scalar(record, "scaled_nonnormality_ratio")
                ),
                "scaled_invariance_defect": defect,
                "endpoint_scaling_condition": float(
                    _scalar(record, "endpoint_scaling_condition")
                ),
                "operator_autodiff_relative_residual": autodiff_residual,
                "gradient_aggregation_relative_residual": gradient_residual,
                "larger_probe_maximum_relative_residual": large,
                "smaller_probe_maximum_relative_residual": small,
                "nonlinear_probe_pass": probe_pass,
                "instantaneous_norm_contraction": operator_norm < 1.0,
                "instantaneous_spectral_contraction": spectral_radius < 1.0,
                "invariance_defect_resolved_as_zero": defect <= defect_tolerance,
            }
        )
    return {
        "rows": rows,
        "nonlinear_probe_pass_count": sum(row["nonlinear_probe_pass"] for row in rows),
        "instantaneous_norm_contraction_count": sum(
            row["instantaneous_norm_contraction"] for row in rows
        ),
        "instantaneous_spectral_contraction_count": sum(
            row["instantaneous_spectral_contraction"] for row in rows
        ),
        "zero_invariance_defect_count": sum(
            row["invariance_defect_resolved_as_zero"] for row in rows
        ),
        "path_layer_count": len(rows),
    }


def intervention_analysis(
    bundle: Path,
    inventory: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    treatment_names = (
        "gate_update_removed",
        "upstream_update_removed",
        "joint_row_program_update_removed",
    )
    rows = []
    for item in inventory["intervention"]:
        label = str(item["trajectory"])
        layer = int(item["layer"])
        step = int(item["step"])
        record = load_record(bundle / item["path"], INTERVENTION_SCHEMA)
        _validate_common(record, label=label, step=step, binding=binding)
        if int(_scalar(record, "layer")) != layer:
            raise ValueError("intervention layer drifted")
        names = [str(value) for value in record["arm_names"].tolist()]
        responses = _finite(
            record["arm_physical_response_from_native"], "intervention responses"
        )
        if responses.shape != (8, 24, 3, 4):
            raise ValueError("intervention response grid drifted")
        selected = {
            name: _aggregate_norm(responses[names.index(name), :, layer, :])
            for name in names
        }
        placebo = selected["one_ulp_placebo"]
        treatments = [selected[name] for name in treatment_names]
        smallest = min(treatments)
        identified = bool(
            smallest > float(INTERVENTION_POLICY["energy_effect_floor"])
            and placebo
            <= float(INTERVENTION_POLICY["placebo_fraction_of_treatment"])
            * smallest
        )
        native_energy = _finite(record["arm_energy"][names.index("native"), :, layer, :], "native energy")
        rows.append(
            {
                "trajectory": label,
                "layer": layer,
                "native_endpoint_median_energy": float(np.median(native_energy)),
                "placebo_response": placebo,
                "gate_update_removed_response": selected["gate_update_removed"],
                "upstream_update_removed_response": selected["upstream_update_removed"],
                "joint_update_removed_response": selected[
                    "joint_row_program_update_removed"
                ],
                "gate_decay_only_response": selected["gate_decay_only"],
                "gate_adaptive_only_response": selected["gate_adaptive_only"],
                "placebo_fraction_of_smallest_treatment": placebo / smallest,
                "one_step_effect_identified": identified,
            }
        )
    return {
        "rows": rows,
        "identified_path_layer_count": sum(
            row["one_step_effect_identified"] for row in rows
        ),
        "path_layer_count": len(rows),
        "analysis_unit": "trajectory-layer",
        "registered_maps_are_within_unit_subsamples": True,
    }


def perturbation_analysis(
    bundle: Path,
    inventory: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    tolerance = float(PERTURBATION_POLICY["maximum_validation_relative_residual"])
    for item in inventory["perturbation"]:
        label = str(item["trajectory"])
        layer = int(item["layer"])
        step = int(item["step"])
        prediction_path = bundle / item["prediction_path"]
        validation_path = bundle / item["validation_path"]
        prediction = load_record(prediction_path, PERTURBATION_SCHEMA)
        validation = load_record(validation_path, PERTURBATION_SCHEMA)
        _validate_common(validation, label=label, step=step, binding=binding)
        if _scalar(prediction, "stage") != "prediction" or _scalar(validation, "stage") != "validation":
            raise ValueError("perturbation stage order drifted")
        if int(_scalar(prediction, "layer")) != layer or int(_scalar(validation, "layer")) != layer:
            raise ValueError("perturbation layer drifted")
        created = int(_scalar(validation, "prediction_created_at_ns"))
        opened = int(_scalar(validation, "validation_opened_at_ns"))
        if created >= opened or created != int(_scalar(prediction, "prediction_created_at_ns")):
            raise ValueError("validation was opened before the prediction was sealed")
        if _scalar(validation, "prediction_sha256") != sha256_path(prediction_path):
            raise ValueError("prediction digest drifted")
        if _scalar(prediction, "direction_sha256") != _scalar(validation, "direction_sha256"):
            raise ValueError("perturbation direction drifted")
        source = _finite(validation["source_distance_by_map"], "source distance")
        endpoint = _finite(validation["endpoint_distance_by_map"], "endpoint distance")
        if source.shape != (24, 4) or endpoint.shape != source.shape:
            raise ValueError("perturbation map grid drifted")
        relative = float(_scalar(validation, "relative_residual"))
        rows.append(
            {
                "trajectory": label,
                "layer": layer,
                "prediction_preceded_validation": True,
                "relative_prediction_residual": relative,
                "absolute_prediction_residual": float(
                    _scalar(validation, "absolute_residual")
                ),
                "observed_response": float(_scalar(validation, "observed_response")),
                "predicted_response": float(_scalar(validation, "predicted_response")),
                "median_source_distance": float(np.median(source)),
                "median_endpoint_distance": float(np.median(endpoint)),
                "median_one_step_gain": float(
                    np.median(endpoint / np.maximum(source, np.finfo(float).tiny))
                ),
                "prospective_prediction_pass": relative <= tolerance,
            }
        )
    return {
        "rows": rows,
        "prospective_prediction_pass_count": sum(
            row["prospective_prediction_pass"] for row in rows
        ),
        "path_layer_count": len(rows),
        "maximum_relative_residual": tolerance,
    }


def plga_analysis(
    bundle: Path,
    inventory: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    condition_total = 0
    condition_satisfied = 0
    tolerance = float(PLGA_POLICY["row_preservation_tolerance"])
    zero = float(PLGA_POLICY["defect_zero_tolerance"])
    for item in inventory["plga"]:
        label = str(item["trajectory"])
        step = int(item["step"])
        record = load_record(bundle / item["path"], PLGA_SCHEMA)
        _validate_common(record, label=label, step=step, binding=binding)
        conditions = _finite(record["row_preservation_residuals"], "PLGA residuals")
        if conditions.shape != (3, 4, 5):
            raise ValueError("PLGA structural grid drifted")
        condition_total += int(conditions.size)
        condition_satisfied += int(np.count_nonzero(conditions <= tolerance))
        slack = _finite(record["triangle_signed_slack"], "PLGA triangle slack")
        if float(np.min(slack)) < -1.0e-10:
            raise ValueError("PLGA quotient-defect triangle check failed")
        for layer in range(3):
            defect = _finite(record["row_constant_defect_norm"][layer], "PLGA defect")
            rows.append(
                {
                    "trajectory": label,
                    "layer": layer,
                    "median_input_quotient_norm": float(
                        np.median(record["input_quotient_norm"][layer])
                    ),
                    "median_output_quotient_norm": float(
                        np.median(record["output_quotient_norm"][layer])
                    ),
                    "median_segment_response_norm": float(
                        np.median(record["segment_response_norm"][layer])
                    ),
                    "median_row_constant_defect_norm": float(np.median(defect)),
                    "median_realized_secant_gain": float(
                        np.median(record["realized_secant_gain"][layer])
                    ),
                    "minimum_triangle_slack": float(np.min(slack[layer])),
                    "row_constant_face_preserved": bool(np.max(defect) <= zero),
                }
            )
    return {
        "rows": rows,
        "satisfied_sufficient_condition_count": condition_satisfied,
        "tested_sufficient_condition_count": condition_total,
        "row_constant_face_preserved_count": sum(
            row["row_constant_face_preserved"] for row in rows
        ),
        "path_layer_count": len(rows),
    }


def resource_analysis(bundle: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    gpu_seconds = 0.0
    peak_reserved = 0
    record_count = 0
    record_paths = []
    for category in ("precision", "gate_operator", "intervention", "perturbation", "plga"):
        for item in inventory[category]:
            key = "validation_path" if category == "perturbation" else "path"
            path = bundle / item[key]
            schema = SCHEMAS[category]
            record = load_record(path, schema)
            gpu_seconds += float(_scalar(record, "resource_gpu_device_seconds"))
            peak_reserved = max(
                peak_reserved, int(_scalar(record, "resource_peak_gpu_reserved_bytes"))
            )
            record_count += 1
            record_paths.append(path)
    attempts = []
    for path in sorted((bundle / "runtime").glob("*/attempt-*/attempt.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        attempts.append(
            {
                "node": value["node"],
                "exit_code": int(value["exit_code"]),
                "elapsed_seconds": float(value["elapsed_seconds"]),
            }
        )
    return {
        "atomic_record_count": record_count,
        "aggregate_gpu_device_seconds": gpu_seconds,
        "aggregate_gpu_device_hours": gpu_seconds / 3600.0,
        "maximum_process_reserved_bytes": peak_reserved,
        "attempt_count": len(attempts),
        "failed_attempt_count": sum(item["exit_code"] != 0 for item in attempts),
        "attempts": attempts,
        "record_paths": [path.relative_to(bundle).as_posix() for path in record_paths],
    }


def _scientific_summary(
    precision: dict[str, Any],
    gate: dict[str, Any],
    intervention: dict[str, Any],
    perturbation: dict[str, Any],
    plga: dict[str, Any],
) -> list[str]:
    return [
        (
            f"{precision['small_path_layer_units']} of "
            f"{precision['total_path_layer_units']} trajectory-layer endpoints "
            "are below the fixed row-quotient energy threshold."
        ),
        (
            f"{gate['instantaneous_norm_contraction_count']} of "
            f"{gate['path_layer_count']} lifted final-gate operators contract in "
            "the declared instantaneous induced norm."
        ),
        (
            f"{gate['zero_invariance_defect_count']} of {gate['path_layer_count']} "
            "lifted final-gate successors have an invariance defect resolved as zero."
        ),
        (
            f"{intervention['identified_path_layer_count']} of "
            f"{intervention['path_layer_count']} one-step component comparisons pass "
            "the same-device placebo gate."
        ),
        (
            f"{perturbation['prospective_prediction_pass_count']} of "
            f"{perturbation['path_layer_count']} unopened half-amplitude responses "
            "meet the prospective local prediction tolerance."
        ),
        (
            f"{plga['row_constant_face_preserved_count']} of "
            f"{plga['path_layer_count']} PLGA trajectory-layer maps preserve the "
            "row-constant face at the registered endpoint inputs."
        ),
    ]


def _format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 1.0e4 or magnitude < 1.0e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"




def analyze(bundle: Path) -> dict[str, Any]:
    inventory = json.loads(
        (bundle / "protocol" / "record-inventory.json").read_text(encoding="utf-8")
    )
    binding = json.loads(
        (bundle / "protocol" / "input_binding.json").read_text(encoding="utf-8")
    )
    precision = precision_analysis(bundle, inventory, binding)
    gate = gate_operator_analysis(bundle, inventory, binding)
    intervention = intervention_analysis(bundle, inventory, binding)
    perturbation = perturbation_analysis(bundle, inventory, binding)
    plga = plga_analysis(bundle, inventory, binding)
    resource = resource_analysis(bundle, inventory)
    record_hashes = {
        path: sha256_path(bundle / path) for path in resource.pop("record_paths")
    }
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "input_record_manifest_sha256": compact_digest(record_hashes),
        "input_record_hashes": record_hashes,
        "population": {
            "trajectories": 3,
            "path_layer_units": 9,
            "registered_maps_per_path_layer": 96,
            "precision_records": len(inventory["precision"]),
            "gate_operator_records": len(inventory["gate_operator"]),
            "intervention_records": len(inventory["intervention"]),
            "perturbation_records": len(inventory["perturbation"]),
            "plga_records": len(inventory["plga"]),
        },
        "precision": precision,
        "gate_operator": gate,
        "intervention": intervention,
        "perturbation": perturbation,
        "plga": plga,
        "resource": resource,
        "scientific_summary": _scientific_summary(
            precision, gate, intervention, perturbation, plga
        ),
        "scope": {
            "finite_horizon_only": True,
            "gate_operator_state_dimension": 192,
            "gate_operator_excludes_upstream_optimizer_coordinates": True,
            "upstream_dynamics_sampled_by_one_step_perturbations": True,
            "registered_maps_are_not_treated_as_independent_replicates": True,
            "asymptotic_theorem_requires_additional_product_and_forcing_hypotheses": True,
        },
    }
    return result




