"""Prediction-before-native producers for source-resolved controlled interventions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from confirm.capture_chronological_confirmation import optimizer_displacements
from confirm.confirmation_artifacts import load_complete_checkpoint, sha256_path
from confirm.finite_increment_live import (
    capture_maps,
    optimizer_from_checkpoint,
    ordered_training_batch,
    registered_tokens,
)
from confirm.row_map_live import _model_from_raw
from confirm.source_restoring_live import load_registry
from confirm.source_resolved_energy import center_rows
from confirm.source_resolved_live import _map_identity, _schema
from confirm.source_resolved_specs import (
    CAMPAIGN_ID,
    FROZEN_POLICY,
    INTERVENTION_PREDICTION_SCHEMA,
    INTERVENTION_SCHEMA,
    INTERVENTIONS,
    REGISTRY,
)
from confirm.strict_schema import load_json, validate
from train_run import create_masks, masked_loss_function


WINDOWS = ("pre-onset", "onset", "late")


def _lineage_digest(binding: dict[str, Any]) -> str:
    """Return the canonical state digest used by checkpoint run identities."""

    return str(binding["content_sha256"])


def _validated_binding(
    path: str | Path, protocol_directory: str | Path
) -> tuple[Path, dict[str, Any]]:
    binding_path = Path(path).resolve()
    value = load_json(binding_path)
    validate(
        value,
        _schema(protocol_directory, "checkpoint_binding.schema.json"),
    )
    checkpoint = Path(value["checkpoint_path"]).resolve()
    if not value["technical_valid"] or sha256_path(checkpoint) != value["checkpoint_sha256"]:
        raise ValueError("intervention checkpoint binding is invalid or stale")
    return checkpoint, value


def _energy(rows: np.ndarray) -> np.ndarray:
    centered = center_rows(rows.reshape(-1, 64, 64))
    return np.sum(centered * centered, axis=(-2, -1))


def _reset_moment_successors(
    names: tuple[str, ...],
    values: tuple[torch.Tensor, ...],
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
) -> dict[str, torch.Tensor]:
    group = optimizer.param_groups[0]
    beta1, beta2 = (float(value) for value in group["betas"])
    epsilon = float(group["eps"])
    rate = float(group["lr"])
    decay = float(group["weight_decay"])
    result = {}
    by_name = dict(model.named_parameters())
    for name, value in zip(names, values, strict=True):
        gradient = by_name[name].grad
        if gradient is None:
            result[name] = value.detach().clone()
            continue
        clipped = gradient.detach()
        first = (1.0 - beta1) * clipped
        second = (1.0 - beta2) * clipped * clipped
        adaptive = (first / (1.0 - beta1)) / (
            torch.sqrt(second / (1.0 - beta2)) + epsilon
        )
        result[name] = (
            (1.0 - rate * decay) * value.detach() - rate * adaptive
        )
    return result


def _intervention_successors(
    intervention: str,
    names: tuple[str, ...],
    values: tuple[torch.Tensor, ...],
    control: tuple[torch.Tensor, ...],
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
) -> dict[str, torch.Tensor]:
    source = dict(zip(names, values, strict=True))
    endpoint = dict(zip(names, control, strict=True))
    if intervention == "moment-reset":
        return _reset_moment_successors(names, values, model, optimizer)
    if intervention == "learning-rate-half":
        factor = 0.5
    elif intervention == "learning-rate-double":
        factor = 2.0
    else:
        factor = None
    if factor is not None:
        return {
            name: source[name] + factor * (endpoint[name] - source[name])
            for name in names
        }

    final_gate = ".mha1.reslayerAs.7.layernormA.weight"
    for name in names:
        in_row_program = ".mha1.reslayerAs." in name
        frozen = (
            intervention == "final-gate-freeze" and final_gate in name
        ) or (
            intervention == "row-program-weight-freeze" and in_row_program
        ) or (
            intervention == "upstream-shape-freeze"
            and in_row_program
            and final_gate not in name
        )
        if frozen:
            endpoint[name] = source[name]
    return endpoint


def produce_prediction(
    source_binding_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    order_path: str | Path,
    intervention: str,
    window: str,
    device_name: str,
    *,
    protocol_directory: str | Path,
) -> dict[str, Any]:
    """Freeze a one-update intervention prediction before native launch."""

    if intervention not in INTERVENTIONS or window not in WINDOWS:
        raise ValueError("unknown source-resolved intervention or window")
    checkpoint_path, binding = _validated_binding(
        source_binding_path, protocol_directory
    )
    expected_step = int(
        FROZEN_POLICY["intervention_window_updates"][window]
    )
    if binding["step"] != expected_step:
        raise ValueError("intervention source differs from its frozen window")
    checkpoint = load_complete_checkpoint(checkpoint_path, map_location="cpu")
    registry = load_registry(registry_path)
    if checkpoint["measurement_registry"] != registry:
        raise ValueError("intervention source and registry disagree")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    model = _model_from_raw(checkpoint, device)
    registered, contexts = registered_tokens(registry, tokens_path, device)
    update_rows, _chunks = ordered_training_batch(
        checkpoint, tokens_path, order_path, device
    )
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = update_rows[:, :-1]
    targets = update_rows[:, 1:]
    logits = model([inputs, create_masks(inputs, device)])[0]
    masked_loss_function(targets, logits).backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    names, values, _directions, control_values = optimizer_displacements(
        model, optimizer
    )
    control = dict(zip(names, control_values, strict=True))
    arm = _intervention_successors(
        intervention, names, values, control_values, model, optimizer
    )
    _source_shape, _source_rows = capture_maps(model, registered)
    _control_shape, control_rows = capture_maps(model, registered, control)
    _arm_shape, arm_rows = capture_maps(model, registered, arm)
    predicted = _energy(arm_rows) - _energy(control_rows)
    records = [
        {
            **_map_identity(index, contexts),
            "source_predicted_energy_change": float(predicted[index]),
        }
        for index in range(REGISTRY["map_count"])
    ]
    checks = {
        "source_checkpoint_opened": True,
        "source_step_matches_frozen_window": binding["step"] == expected_step,
        "control_and_arm_functionally_derived": set(control) == set(arm),
        "map_population_complete": len(records) == REGISTRY["map_count"],
        "all_values_finite": bool(np.isfinite(predicted).all()),
    }
    result = {
        "schema_version": INTERVENTION_PREDICTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_checkpoint_binding_sha256": sha256_path(source_binding_path),
        "source_checkpoint_sha256": _lineage_digest(binding),
        "source_step": int(binding["step"]),
        "window": window,
        "intervention": intervention,
        "created_at_ns": time.time_ns(),
        "records": records,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(
        result,
        _schema(protocol_directory, "intervention_prediction.schema.json"),
    )
    if not result["technical_valid"]:
        raise ValueError("source-only intervention prediction failed")
    return result


def produce_intervention_result(
    prediction_path: str | Path,
    control_binding_path: str | Path,
    arm_binding_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    policy_path: str | Path,
    device_name: str,
    *,
    protocol_directory: str | Path,
) -> dict[str, Any]:
    """Open native endpoints only after a source-only prediction exists."""

    prediction = load_json(prediction_path)
    validate(
        prediction,
        _schema(protocol_directory, "intervention_prediction.schema.json"),
    )
    if not prediction["technical_valid"]:
        raise ValueError("native intervention received an invalid prediction")
    control_path, control_binding = _validated_binding(
        control_binding_path, protocol_directory
    )
    arm_path, arm_binding = _validated_binding(
        arm_binding_path, protocol_directory
    )
    if prediction["created_at_ns"] >= min(
        control_binding["created_at_ns"], arm_binding["created_at_ns"]
    ):
        raise ValueError("native endpoints were bound before the prediction")
    control = load_complete_checkpoint(control_path, map_location="cpu")
    arm = load_complete_checkpoint(arm_path, map_location="cpu")
    opened_at = time.time_ns()
    registry = load_registry(registry_path)
    if control["measurement_registry"] != registry or arm["measurement_registry"] != registry:
        raise ValueError("native intervention endpoints changed the registry")
    if control_binding["step"] != arm_binding["step"]:
        raise ValueError("paired intervention endpoint clocks disagree")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rows, contexts = registered_tokens(registry, tokens_path, device)
    control_model = _model_from_raw(control, device)
    arm_model = _model_from_raw(arm, device)
    _control_shape, control_rows = capture_maps(control_model, rows)
    _arm_shape, arm_rows = capture_maps(arm_model, rows)
    native = _energy(arm_rows) - _energy(control_rows)
    predicted_by_map = {
        record["map_id"]: record for record in prediction["records"]
    }
    records = []
    for index in range(REGISTRY["map_count"]):
        identity = _map_identity(index, contexts)
        predicted = float(
            predicted_by_map[identity["map_id"]]["source_predicted_energy_change"]
        )
        observed = float(native[index])
        records.append({
            **identity,
            "window": prediction["window"],
            "intervention": prediction["intervention"],
            "source_predicted_energy_change": predicted,
            "native_energy_change": observed,
            "prediction_residual": observed - predicted,
            "relative_prediction_residual": abs(observed - predicted) / max(
                abs(predicted), abs(observed), 1.0
            ),
            "control_checkpoint_sha256": control_binding["checkpoint_sha256"],
            "intervention_checkpoint_sha256": arm_binding["checkpoint_sha256"],
        })
    policy = load_json(policy_path)
    validate(
        policy,
        _schema(protocol_directory, "construction_policy.schema.json"),
    )
    if not policy["technical_valid"]:
        raise ValueError("intervention received an invalid construction policy")
    windows = policy.get("window_updates")
    if not isinstance(windows, dict) or set(windows) != set(WINDOWS):
        raise ValueError("construction policy omits frozen intervention windows")
    control_parent = control.get("run_identity", {}).get(
        "source_checkpoint_sha256"
    )
    arm_parent = arm.get("run_identity", {}).get(
        "source_checkpoint_sha256"
    )
    same_parent = (
        control_parent == arm_parent == prediction["source_checkpoint_sha256"]
    )
    same_cursor = (
        control["data_state"]["cursor_end"]
        == arm["data_state"]["cursor_end"]
    )
    same_batch = (
        control_binding["batch_sha256"] == arm_binding["batch_sha256"]
    )
    endpoint_step_exact = (
        control_binding["step"] == arm_binding["step"]
        == prediction["source_step"] + 1
    )
    policy_window_exact = (
        int(windows[prediction["window"]]) == prediction["source_step"]
    )
    checks = {
        "windows_frozen_before_interventions": policy.get("created_at_ns", opened_at) < prediction["created_at_ns"],
        "source_step_matches_policy_window": policy_window_exact,
        "paired_batches_equal": same_cursor and same_batch,
        "paired_rng_equal": same_parent,
        "source_prediction_precedes_native": prediction["created_at_ns"] < min(
            control_binding["created_at_ns"], arm_binding["created_at_ns"]
        ),
        "checkpoint_links_exact": same_parent and endpoint_step_exact,
        "map_population_complete": len(records) == REGISTRY["map_count"],
        "all_values_finite": bool(np.isfinite(native).all()),
    }
    result = {
        "schema_version": INTERVENTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "construction_policy_sha256": sha256_path(policy_path),
        "prediction_sha256": sha256_path(prediction_path),
        "prediction_created_at_ns": prediction["created_at_ns"],
        "native_opened_at_ns": opened_at,
        "source_step": int(prediction["source_step"]),
        "endpoint_step": int(control_binding["step"]),
        "prediction_relative_tolerance": float(
            FROZEN_POLICY["intervention_prediction_relative_tolerance"]
        ),
        "window_updates": {name: int(windows[name]) for name in WINDOWS},
        "records": records,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(result, _schema(protocol_directory, "intervention.schema.json"))
    if not result["technical_valid"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"native intervention checks failed: {failed}")
    return result
