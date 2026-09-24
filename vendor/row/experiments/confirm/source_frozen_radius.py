"""Cross-arithmetic one-step test with a source-frozen row radius."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from confirm.confirmation_artifacts import (
    load_complete_checkpoint,
    sha256_path,
)
from confirm.finite_increment_live import (
    capture_maps,
    optimizer_from_checkpoint,
    ordered_training_batch,
    registered_tokens,
)
from confirm.finite_transport import centered_energy
from confirm.mixed_collapse_specs import (
    CAMPAIGN_ID,
    REGISTRY,
    SOURCE_FROZEN_RADIUS,
    SOURCE_RADIUS_SCHEMA,
)
from confirm.row_map_live import _model_from_raw
from confirm.source_resolved_energy import center_rows
from confirm.source_resolved_live import _flatten_maps, _map_identity
from confirm.source_restoring_live import load_registry
from confirm.strict_schema import load_json, validate
from train_run import create_masks, masked_loss_function


def _next_batch(
    checkpoint: dict[str, Any],
    tokens_path: str | Path,
    order_path: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """Open the next batch under the globally unique fixed order."""

    rows, _indices = ordered_training_batch(
        checkpoint, tokens_path, order_path, device
    )
    return rows


def _one_successor(
    checkpoint: dict[str, Any],
    registry: dict[str, Any],
    tokens_path: str | Path,
    order_path: str | Path,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return source and one-step successor maps on one arithmetic path."""

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    model = _model_from_raw(checkpoint, device)
    registered, contexts = registered_tokens(registry, tokens_path, device)
    _source_shape, source_rows = capture_maps(model, registered)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    batch = _next_batch(checkpoint, tokens_path, order_path, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    logits = model([inputs, create_masks(inputs, device)])[0]
    loss = masked_loss_function(targets, logits)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optimizer.step()
    _successor_shape, successor_rows = capture_maps(model, registered)
    del optimizer, model, registered, batch, inputs, targets, logits, loss
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()
    return (
        _flatten_maps(source_rows),
        _flatten_maps(successor_rows),
        contexts,
    )


def produce(
    checkpoint_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    order_path: str | Path,
    prediction_device: str,
    comparison_device: str,
    *,
    protocol_directory: str | Path,
) -> dict[str, Any]:
    """Evaluate a predeclared radius on a comparison arithmetic path."""

    checkpoint_file = Path(checkpoint_path).resolve()
    checkpoint = load_complete_checkpoint(checkpoint_file, map_location="cpu")
    binding = checkpoint["data_state"].get("campaign_binding")
    if not isinstance(binding, dict) or binding.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("checkpoint is outside the mixed-collapse campaign")
    registry = load_registry(registry_path)
    if checkpoint["measurement_registry"] != registry:
        raise ValueError("checkpoint and measurement registry disagree")

    source, predicted, contexts = _one_successor(
        checkpoint, registry, tokens_path, order_path, prediction_device
    )
    comparison_source, comparison, comparison_contexts = _one_successor(
        checkpoint, registry, tokens_path, order_path, comparison_device
    )
    if contexts != comparison_contexts:
        raise ValueError("arithmetic paths opened different registered contexts")
    if len(predicted) != REGISTRY["map_count"]:
        raise ValueError("source-radius map population changed")

    source_geometry = centered_energy(source)
    predicted_geometry = centered_energy(predicted)
    comparison_geometry = centered_energy(comparison)
    delta = np.linalg.norm(
        center_rows(comparison - predicted), axis=(-2, -1)
    )
    source_delta = np.linalg.norm(
        center_rows(comparison_source - source), axis=(-2, -1)
    )
    declared = float(SOURCE_FROZEN_RADIUS)
    predicted_energy = np.asarray(predicted_geometry["energy"], dtype=np.float64)
    energy_upper = (
        np.sqrt(np.maximum(predicted_energy, 0.0)) + declared
    ) ** 2
    comparison_energy = np.asarray(
        comparison_geometry["energy"], dtype=np.float64
    )
    records = []
    for index in range(len(predicted)):
        records.append({
            "map_id": _map_identity(index, contexts)["map_id"],
            "source_energy": float(source_geometry["energy"][index]),
            "predicted_energy": float(predicted_energy[index]),
            "comparison_energy": float(comparison_energy[index]),
            "realized_radius": float(delta[index]),
            "predeclared_radius": declared,
            "energy_upper": float(energy_upper[index]),
            "energy_slack": float(energy_upper[index] - comparison_energy[index]),
            "radius_slack": float(declared - delta[index]),
        })
    finite = all(
        np.isfinite(value)
        for record in records
        for key, value in record.items()
        if key != "map_id"
    ) and np.isfinite(source_delta).all()
    radius_closed = bool(np.all(delta <= declared))
    energy_closed = bool(np.all(comparison_energy <= energy_upper + 1.0e-12))
    result = {
        "schema_version": SOURCE_RADIUS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "trajectory": str(checkpoint["config"]["name"]),
        "seed": int(checkpoint["config"]["seed"]),
        "source_step": int(checkpoint["step"]),
        "endpoint_step": int(checkpoint["step"]) + 1,
        "checkpoint_sha256": sha256_path(checkpoint_file),
        "data_order_sha256": sha256_path(order_path),
        "prediction_device": str(prediction_device),
        "comparison_device": str(comparison_device),
        "predeclared_radius": declared,
        "map_records": records,
        "checkpoint_opened": True,
        "population_complete": len(records) == REGISTRY["map_count"],
        "radius_contract_closed": radius_closed,
        "energy_envelope_closed": energy_closed,
        "technical_valid": bool(finite and len(records) == REGISTRY["map_count"]),
    }
    schema = load_json(Path(protocol_directory) / "source_radius.schema.json")
    validate(result, schema)
    if not result["technical_valid"]:
        raise ValueError("source-radius producer returned invalid arithmetic")
    return result
