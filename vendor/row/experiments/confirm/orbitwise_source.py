"""Prospective source-only reopening prediction and native linkage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from confirm.capture_chronological_confirmation import (
    optimizer_displacements,
)
from confirm.confirmation_artifacts import (
    load_complete_checkpoint,
    sha256_path,
    tensor_mapping_digest,
)
from confirm.finite_increment_live import (
    capture_maps,
    optimizer_from_checkpoint,
    ordered_training_batch,
    registered_tokens,
)
from confirm.finite_transport import centered_energy
from confirm.orbitwise_specs import (
    CAMPAIGN_ID,
    GATE_PREDICTION_SCHEMA,
    GATE_RESULT_SCHEMA,
    REGISTRY,
    SOURCE_LINK_SCHEMA,
    SOURCE_PREDICTION_SCHEMA,
    TOLERANCES,
)
from confirm.row_map_live import _model_from_raw
from confirm.source_resolved_energy import (
    center_rows,
    permutation_symmetric_attribution,
    source_work_gram,
)
from confirm.source_resolved_live import (
    _canonical,
    _flatten_maps,
    _macro_source,
    _map_identity,
    _schema,
    produce_edge,
)
from confirm.source_restoring_live import load_registry
from confirm.strict_schema import load_json, validate
from train_run import create_masks, masked_loss_function


def _new_schema(protocol_directory: str | Path, name: str) -> dict[str, Any]:
    return load_json(Path(protocol_directory) / name)


def produce_source_prediction(
    source_binding_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    order_path: str | Path,
    device_name: str,
    *,
    binding_protocol_directory: str | Path,
    orbitwise_protocol_directory: str | Path,
    producer_command: list[str],
    created_at_ns: int | None = None,
) -> dict[str, Any]:
    """Predict one optimizer edge without accepting a successor input."""

    source_binding_file = Path(source_binding_path).resolve()
    source_binding = load_json(source_binding_file)
    validate(
        source_binding,
        _schema(binding_protocol_directory, "checkpoint_binding.schema.json"),
    )
    if not source_binding["technical_valid"]:
        raise ValueError("source prediction received an invalid binding")
    source_path = Path(source_binding["checkpoint_path"]).resolve()
    if sha256_path(source_path) != source_binding["checkpoint_sha256"]:
        raise ValueError("source checkpoint changed after binding")

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    checkpoint = load_complete_checkpoint(source_path, map_location="cpu")
    registry = load_registry(registry_path)
    if checkpoint["measurement_registry"] != registry:
        raise ValueError("source checkpoint and registry disagree")
    registered, contexts = registered_tokens(
        registry, tokens_path, device
    )
    model = _model_from_raw(checkpoint, device)
    source_parameters = dict(model.named_parameters())
    source_shape, source_rows = capture_maps(model, registered)

    model.train()
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    update_rows, _chunks = ordered_training_batch(
        checkpoint, tokens_path, order_path, device
    )
    optimizer.zero_grad(set_to_none=True)
    inputs = update_rows[:, :-1]
    targets = update_rows[:, 1:]
    logits = model([inputs, create_masks(inputs, inputs.device)])[0]
    loss = masked_loss_function(targets, logits)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    names, _values, _directions, successors = optimizer_displacements(
        model, optimizer
    )
    predicted_parameters = dict(zip(names, successors, strict=True))
    _predicted_shape, predicted_rows = capture_maps(
        model, registered, predicted_parameters
    )
    predicted_parameter_sha256 = tensor_mapping_digest(predicted_parameters)

    macro_names = ("upstream", "row_program", "normalization")

    def evaluate(subset: frozenset[str]) -> np.ndarray:
        parameters = {
            name: (
                predicted_parameters[name]
                if _macro_source(name) in subset
                else source_parameters[name]
            )
            for name in source_parameters
        }
        _shape, rows = capture_maps(model, registered, parameters)
        return _flatten_maps(rows)

    shapley = permutation_symmetric_attribution(macro_names, evaluate)
    source_flat = _flatten_maps(source_rows)
    predicted_flat = _flatten_maps(predicted_rows)
    if not np.array_equal(shapley["baseline"], source_flat):
        raise RuntimeError("source-only empty coalition changed the source")
    if not np.array_equal(shapley["endpoint"], predicted_flat):
        raise RuntimeError("source-only full coalition changed the prediction")
    work = source_work_gram(source_flat, shapley["contributions"])
    source_geometry = centered_energy(source_flat)
    predicted_geometry = centered_energy(predicted_flat)
    records = []
    for index in range(len(source_flat)):
        source_energy = float(source_geometry["energy"][index])
        predicted_energy = float(predicted_geometry["energy"][index])
        records.append({
            "map_id": _map_identity(index, contexts)["map_id"],
            "source_energy": source_energy,
            "predicted_energy": predicted_energy,
            "predicted_reopening": max(predicted_energy - source_energy, 0.0),
            "gain_defined": bool(work["gain_defined"][index]),
            "source_gain": float(work["source_gain"][index]),
            "radial_dissipation": [
                float(value) for value in work["radial_dissipation"][index]
            ],
            "charge_gram": [
                [float(value) for value in row]
                for row in work["charge_gram"][index]
            ],
            "predicted_native_parameter_sha256": predicted_parameter_sha256,
        })
    checks = {
        "source_checkpoint_opened": True,
        "map_population_complete": len(records) == REGISTRY["map_count"],
        "functional_step_finite": all(
            np.isfinite(value)
            for record in records
            for value in (
                record["source_energy"],
                record["predicted_energy"],
                record["predicted_reopening"],
                record["source_gain"],
            )
        ),
        "source_ledger_closed": (
            work["maximum_abs_gain_residual"]
            <= TOLERANCES["source_transport_relative"]
        ),
        "prediction_has_no_native_successor_input": True,
    }
    result = {
        "schema_version": SOURCE_PREDICTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "trajectory": source_binding["trajectory"],
        "seed": source_binding["seed"],
        "source_step": source_binding["step"],
        "endpoint_step": source_binding["step"] + 1,
        "source_binding_sha256": sha256_path(source_binding_file),
        "producer_command_sha256": hashlib.sha256(
            _canonical(producer_command)
        ).hexdigest(),
        "created_at_ns": int(
            time.time_ns() if created_at_ns is None else created_at_ns
        ),
        "map_records": records,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(
        result,
        _new_schema(
            orbitwise_protocol_directory, "source_prediction.schema.json"
        ),
    )
    if not result["technical_valid"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"source prediction checks failed: {failed}")
    return result


def link_source_prediction(
    prediction_path: str | Path,
    source_binding_path: str | Path,
    endpoint_binding_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    order_path: str | Path,
    device_name: str,
    *,
    binding_protocol_directory: str | Path,
    orbitwise_protocol_directory: str | Path,
    producer_command: list[str],
    created_at_ns: int | None = None,
) -> dict[str, Any]:
    """Open the successor only after a source prediction has been sealed."""

    prediction_file = Path(prediction_path).resolve()
    prediction = load_json(prediction_file)
    validate(
        prediction,
        _new_schema(
            orbitwise_protocol_directory, "source_prediction.schema.json"
        ),
    )
    if not prediction["technical_valid"]:
        raise ValueError("native link received an invalid prediction")
    endpoint_binding_file = Path(endpoint_binding_path).resolve()
    endpoint_binding = load_json(endpoint_binding_file)
    validate(
        endpoint_binding,
        _schema(binding_protocol_directory, "checkpoint_binding.schema.json"),
    )
    edge = produce_edge(
        source_binding_path,
        endpoint_binding_path,
        registry_path,
        tokens_path,
        order_path,
        device_name,
        protocol_directory=binding_protocol_directory,
        producer_command=producer_command,
    )
    predicted_by_id = {
        record["map_id"]: record for record in prediction["map_records"]
    }
    records = []
    maximum_tolerance = TOLERANCES["source_reopening_absolute"]
    ledger_agrees = True
    for native in edge["map_records"]:
        predicted = predicted_by_id[native["map_id"]]
        source = predicted["source_energy"]
        predicted_energy = predicted["predicted_energy"]
        endpoint = native["endpoint_energy"]
        native_charge = max(native["native_energy_upper"] - predicted_energy, 0.0)
        if source > 0.0:
            upper = max(
                source * (predicted["source_gain"] - 1.0) + native_charge,
                0.0,
            )
        else:
            upper = native["native_energy_upper"]
        reopening = max(endpoint - source, 0.0)
        scale = max(source, predicted_energy, endpoint, 1.0)
        ledger_agrees = ledger_agrees and (
            abs(native["source_energy"] - source) <= maximum_tolerance * scale
            and abs(native["endpoint_energy"] - predicted_energy)
            <= TOLERANCES["source_transport_relative"] * scale
        )
        records.append({
            "map_id": native["map_id"],
            "source_energy": source,
            "endpoint_energy": endpoint,
            "realized_reopening": reopening,
            "source_reopening_upper": upper,
            "native_radius": native["native_radius"],
            "native_charge": native_charge,
            "bound_slack": upper - reopening,
            "zero_source_energy": source == 0.0,
        })
    checks = {
        "prediction_precedes_successor_binding": (
            prediction["created_at_ns"] <= endpoint_binding["created_at_ns"]
        ),
        "checkpoint_chronology": (
            prediction["source_step"] + 1 == endpoint_binding["step"]
            == prediction["endpoint_step"]
        ),
        "native_parameter_link_closed": bool(
            edge["checks"]["native_link_closed"] and ledger_agrees
        ),
        "map_population_complete": len(records) == REGISTRY["map_count"],
        "all_values_finite": all(
            np.isfinite(value)
            for record in records
            for key, value in record.items()
            if key not in {"map_id", "zero_source_energy"}
        ),
    }
    scientific_checks = {
        "source_reopening_bound_closed": all(
            record["bound_slack"]
            >= -maximum_tolerance
            * max(
                record["source_energy"],
                record["endpoint_energy"],
                record["source_reopening_upper"],
                1.0,
            )
            for record in records
        ),
    }
    result = {
        "schema_version": SOURCE_LINK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "trajectory": prediction["trajectory"],
        "seed": prediction["seed"],
        "source_step": prediction["source_step"],
        "endpoint_step": prediction["endpoint_step"],
        "prediction_sha256": sha256_path(prediction_file),
        "endpoint_binding_sha256": sha256_path(endpoint_binding_file),
        "created_at_ns": int(
            time.time_ns() if created_at_ns is None else created_at_ns
        ),
        "map_records": records,
        "checks": checks,
        "scientific_checks": scientific_checks,
        "technical_valid": all(checks.values()),
    }
    validate(
        result,
        _new_schema(orbitwise_protocol_directory, "source_link.schema.json"),
    )
    if not result["technical_valid"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"source native-link checks failed: {failed}")
    return result


def seal_gate_prediction(
    source_binding_path: str | Path,
    *,
    binding_protocol_directory: str | Path,
    orbitwise_protocol_directory: str | Path,
    created_at_ns: int | None = None,
) -> dict[str, Any]:
    """Seal the registered directional gate-freeze prediction."""

    binding_file = Path(source_binding_path).resolve()
    binding = load_json(binding_file)
    validate(
        binding,
        _schema(binding_protocol_directory, "checkpoint_binding.schema.json"),
    )
    result = {
        "schema_version": GATE_PREDICTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "trajectory": binding["trajectory"],
        "seed": binding["seed"],
        "source_step": binding["step"],
        "horizon": 64,
        "source_checkpoint_sha256": binding["checkpoint_sha256"],
        "prediction": "freeze-retains-more-physical-energy",
        "created_at_ns": int(
            time.time_ns() if created_at_ns is None else created_at_ns
        ),
        "technical_valid": True,
    }
    validate(
        result,
        _new_schema(
            orbitwise_protocol_directory, "gate_prediction.schema.json"
        ),
    )
    return result


def _load_endpoint_timepoint(
    log_path: Path,
    step: int,
    *,
    orbitwise_protocol_directory: str | Path,
) -> dict[str, Any]:
    point = None
    with log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON constant {value}")
                ),
            )
            candidate = row.get("orbitwise_full_timepoint")
            if candidate is not None and int(candidate["step"]) == step:
                if point is not None:
                    raise ValueError(f"duplicate branch timepoint {step}")
                point = candidate
    if point is None:
        raise ValueError(f"branch log is missing endpoint update {step}")
    validate(
        point,
        _new_schema(
            orbitwise_protocol_directory, "full_timepoint.schema.json"
        ),
    )
    return point


def produce_gate_result(
    prediction_path: str | Path,
    control_log_path: str | Path,
    frozen_log_path: str | Path,
    registry_path: str | Path,
    *,
    orbitwise_protocol_directory: str | Path,
) -> dict[str, Any]:
    """Compare one content-identical control and final-gate-freeze branch."""

    prediction_file = Path(prediction_path).resolve()
    control_file = Path(control_log_path).resolve()
    frozen_file = Path(frozen_log_path).resolve()
    prediction = load_json(prediction_file)
    validate(
        prediction,
        _new_schema(
            orbitwise_protocol_directory, "gate_prediction.schema.json"
        ),
    )
    endpoint_step = int(prediction["source_step"] + prediction["horizon"])
    control = _load_endpoint_timepoint(
        control_file,
        endpoint_step,
        orbitwise_protocol_directory=orbitwise_protocol_directory,
    )
    frozen = _load_endpoint_timepoint(
        frozen_file,
        endpoint_step,
        orbitwise_protocol_directory=orbitwise_protocol_directory,
    )
    registry = load_registry(registry_path)
    contexts = list(registry["construction"]) + list(registry["validation"])
    control_energy = np.asarray(control["energy"], dtype=np.float64)
    frozen_energy = np.asarray(frozen["energy"], dtype=np.float64)
    difference = frozen_energy - control_energy
    records = [
        {
            "map_id": _map_identity(index, contexts)["map_id"],
            "control_energy": float(control_energy[index]),
            "frozen_gate_energy": float(frozen_energy[index]),
            "frozen_minus_control_energy": float(difference[index]),
            "supports_registered_direction": bool(difference[index] > 0.0),
        }
        for index in range(len(control_energy))
    ]
    checks = {
        "prediction_precedes_branches": (
            prediction["created_at_ns"]
            <= min(control_file.stat().st_mtime_ns, frozen_file.stat().st_mtime_ns)
        ),
        "endpoint_present": (
            control["step"] == frozen["step"] == endpoint_step
        ),
        "map_population_complete": len(records) == REGISTRY["map_count"],
        "all_values_finite": bool(
            np.isfinite(control_energy).all()
            and np.isfinite(frozen_energy).all()
            and np.isfinite(difference).all()
        ),
    }
    result = {
        "schema_version": GATE_RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_step": prediction["source_step"],
        "endpoint_step": endpoint_step,
        "horizon": prediction["horizon"],
        "prediction_sha256": sha256_path(prediction_file),
        "control_log_sha256": sha256_path(control_file),
        "frozen_log_sha256": sha256_path(frozen_file),
        "map_records": records,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(
        result,
        _new_schema(orbitwise_protocol_directory, "gate_result.schema.json"),
    )
    if not result["technical_valid"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"gate-result checks failed: {failed}")
    return result
