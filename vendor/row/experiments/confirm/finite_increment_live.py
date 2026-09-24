"""Native checkpoint replay and all-layer row-map capture.

This module owns the live boundary used by the finite-increment campaign.
Source capture constructs the next AdamW endpoint on isolated parameter and
optimizer clones.  Successor capture starts again from the same immutable
complete checkpoint and executes the native optimizer step only after a
source-bound prediction exists.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.func import functional_call


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.capture_chronological_confirmation import (  # noqa: E402
    optimizer_displacements,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION,
    sha256_path,
    tensor_mapping_digest,
    validate_complete_checkpoint,
    validate_measurement_registry,
    validate_model_only_checkpoint,
)
from confirm.finite_increment import (  # noqa: E402
    affine_increment,
    per_map_diameters,
    two_variable_power_increment,
)
from confirm.finite_increment_specs import (  # noqa: E402
    ARCHITECTURE,
    OPTIMIZER,
    REGISTRY_SCHEMA,
)
from confirm.row_map_live import (  # noqa: E402
    _causal_mask,
    _loss,
    _model_from_raw,
)
from train_run import validate_data_order  # noqa: E402


SOURCE_SCHEMA = "pldr-finite-increment-source-v1"
SUCCESSOR_SCHEMA = "pldr-finite-increment-successor-v1"


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"artifact omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def _array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"artifact omits {name}")
    value = np.asarray(record[name])
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return np.ascontiguousarray(value)


def load_registry(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_measurement_registry(value, require_nonempty=True)
    if value.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("unknown finite-increment registry schema")
    return value


def load_checkpoint(
    path: str | Path,
    *,
    require_optimizer: bool,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    value = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    schema = value.get("schema_version") if isinstance(value, dict) else None
    if schema == CHECKPOINT_SCHEMA_VERSION:
        validate_complete_checkpoint(value)
    elif schema == MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION:
        validate_model_only_checkpoint(value)
    else:
        raise ValueError("finite-increment capture needs a versioned checkpoint")
    if require_optimizer and "opt" not in value:
        raise ValueError("finite-increment source capture needs optimizer state")
    config = value["config"]
    expected = {
        "layers": ARCHITECTURE["layers"],
        "heads": ARCHITECTURE["heads"],
        "dk": ARCHITECTURE["head_width"],
        "adff": 170,
        "ctx": OPTIMIZER["context_length"],
        "batch": OPTIMIZER["batch_size"],
        "optimizer": "adamw",
        "lr": OPTIMIZER["learning_rate"],
        "warmup": OPTIMIZER["warmup_updates"],
        "const_lr": OPTIMIZER["constant_rate_after_warmup"],
        "wd": OPTIMIZER["weight_decay"],
        "clip": OPTIMIZER["gradient_clip_value"],
        "accum": 1,
    }
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise ValueError(f"checkpoint configuration disagrees at {key}")
    value["content_sha256"] = sha256_path(checkpoint_path)
    value["source_path"] = str(checkpoint_path)
    return value


def optimizer_from_checkpoint(
    model: torch.nn.Module, checkpoint: dict[str, Any]
) -> torch.optim.AdamW:
    config = checkpoint["config"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        betas=(float(OPTIMIZER["beta1"]), float(OPTIMIZER["beta2"])),
        eps=float(OPTIMIZER["epsilon"]),
        weight_decay=float(config["wd"]),
    )
    optimizer.load_state_dict(checkpoint["opt"])
    return optimizer


def ordered_training_batch(
    checkpoint: dict[str, Any],
    tokens_path: str | Path,
    order_path: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor, list[int]]:
    config = checkpoint["config"]
    context = int(config["ctx"])
    batch = int(config["batch"])
    cursor = int(checkpoint["data_offset_end"])
    tokens_file = Path(tokens_path).resolve()
    order_file = Path(order_path).resolve()
    data = np.memmap(tokens_file, dtype=np.uint16, mode="r")
    total_chunks = len(data) // context
    chunks = data[: total_chunks * context].reshape(total_chunks, context)
    order = validate_data_order(
        np.load(order_file, allow_pickle=False), total_chunks
    )
    expected_digest = checkpoint["data_state"].get("data_order_sha256")
    if expected_digest != sha256_path(order_file):
        raise ValueError("checkpoint and frozen training order disagree")
    if checkpoint["data_state"].get("dataset_sha256") != sha256_path(tokens_file):
        raise ValueError("checkpoint and token archive disagree")
    selected = order[cursor: cursor + batch]
    if len(selected) != batch:
        raise ValueError("the next training batch leaves the frozen order")
    rows = torch.from_numpy(
        chunks[selected].astype(np.int64, copy=True)
    ).to(device)
    return rows, [int(value) for value in selected]


def registered_tokens(
    registry: dict[str, Any],
    tokens_path: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    contexts = list(registry["construction"]) + list(registry["validation"])
    indices = [int(row["chunk_index"]) for row in contexts]
    context = int(registry["context_length"])
    data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    total_chunks = len(data) // context
    chunks = data[: total_chunks * context].reshape(total_chunks, context)
    if any(index < 0 or index >= total_chunks for index in indices):
        raise ValueError("a registered context leaves the token archive")
    rows = torch.from_numpy(
        chunks[np.asarray(indices)].astype(np.int64, copy=True)
    ).to(device)
    return rows, contexts


def capture_maps(
    model: torch.nn.Module,
    token_rows: torch.Tensor,
    parameters: dict[str, torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture normalized shapes and physical rows for every layer and head."""

    layer_count = int(ARCHITECTURE["layers"])
    shapes: list[list[torch.Tensor]] = [[] for _ in range(layer_count)]
    outputs: list[list[torch.Tensor]] = [[] for _ in range(layer_count)]
    handles = []
    for layer in range(layer_count):
        layernorm = model.decoder.dec_layers[layer].mha1.reslayerAs[-1].layernormA

        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: torch.Tensor,
            *,
            layer_index: int = layer,
        ) -> None:
            value = inputs[0]
            centered = value - value.mean(dim=-1, keepdim=True)
            shape = centered / torch.sqrt(
                module.eps
                + torch.mean(centered * centered, dim=-1, keepdim=True)
            )
            shapes[layer_index].append(shape.detach().cpu())
            outputs[layer_index].append(output.detach().cpu())

        handles.append(layernorm.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        inputs = token_rows[:, :-1]
        dtype = next(model.parameters()).dtype
        mask = _causal_mask(inputs.shape[1], inputs.device, dtype)
        with torch.no_grad():
            if parameters is None:
                model([inputs, mask])
            else:
                functional_call(model, parameters, ([inputs, mask],), strict=False)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if any(len(layer) != 1 for layer in shapes + outputs):
        raise RuntimeError("a row-map hook did not fire exactly once")
    shape = torch.stack([shapes[layer][0] for layer in range(layer_count)], dim=1)
    physical = torch.stack(
        [outputs[layer][0] for layer in range(layer_count)], dim=1
    )
    expected = (
        len(token_rows),
        layer_count,
        int(ARCHITECTURE["heads"]),
        int(ARCHITECTURE["rows_per_map"]),
        int(ARCHITECTURE["head_width"]),
    )
    if tuple(shape.shape) != expected or tuple(physical.shape) != expected:
        raise ValueError(
            f"captured maps have {tuple(shape.shape)} and "
            f"{tuple(physical.shape)}, expected {expected}"
        )
    return shape.double().numpy(), physical.double().numpy()


def timecourse_record(
    model: torch.nn.Module,
    token_rows: torch.Tensor,
    *,
    step: int,
    registry_sha256: str,
) -> dict[str, Any]:
    _shape, physical = capture_maps(model, token_rows)
    flattened = physical.reshape(-1, physical.shape[-2], physical.shape[-1])
    diameter = per_map_diameters(flattened)
    return {
        "schema_version": "pldr-finite-increment-timecourse-point-v1",
        "step": int(step),
        "registry_sha256": registry_sha256,
        "map_order": "context-major-layer-major-head-major-v1",
        "diameter_squared": [
            float(value) for value in diameter["diameter_squared"]
        ],
        "maximizing_pairs": diameter["maximizing_pairs"],
        "aggregate_diameter_squared": diameter["aggregate_diameter_squared"],
        "cross_map_pairs_formed": diameter["cross_map_pairs_formed"],
    }


def _parameter(
    mapping: dict[str, torch.Tensor], layer: int, name: str
) -> np.ndarray:
    key = f"decoder.dec_layers.{layer}.mha1.plgatt_layer.{name}"
    if key not in mapping:
        raise KeyError(f"missing PLGA parameter {key}")
    return mapping[key].detach().double().cpu().numpy()


def _gate(
    mapping: dict[str, torch.Tensor], layer: int, name: str
) -> np.ndarray:
    key = f"decoder.dec_layers.{layer}.mha1.reslayerAs.7.layernormA.{name}"
    if key not in mapping:
        raise KeyError(f"missing final LayerNorm parameter {key}")
    return mapping[key].detach().double().cpu().numpy()


def plga_report(
    layer: int,
    source_rows: np.ndarray,
    endpoint_rows: np.ndarray,
    source_parameters: dict[str, torch.Tensor],
    endpoint_parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    w0 = _parameter(source_parameters, layer, "Wlst")
    w1 = _parameter(endpoint_parameters, layer, "Wlst")
    b0 = _parameter(source_parameters, layer, "blst")
    b1 = _parameter(endpoint_parameters, layer, "blst")
    p0 = _parameter(source_parameters, layer, "pwlst")
    p1 = _parameter(endpoint_parameters, layer, "pwlst")
    a0 = _parameter(source_parameters, layer, "alst")
    a1 = _parameter(endpoint_parameters, layer, "alst")
    ba0 = _parameter(source_parameters, layer, "balst")
    ba1 = _parameter(endpoint_parameters, layer, "balst")
    context_count = source_rows.shape[0]
    z0 = np.matmul(w0[None], source_rows) + b0[None]
    z1 = np.matmul(w1[None], endpoint_rows) + b1[None]
    powers = two_variable_power_increment(
        z0,
        z1,
        np.broadcast_to(p0, (context_count,) + p0.shape),
        np.broadcast_to(p1, (context_count,) + p1.shape),
    )
    h0 = np.asarray(powers["source"])
    h1 = np.asarray(powers["endpoint"])
    g0 = np.matmul(a0[None], h0) + ba0[None]
    g1 = np.matmul(a1[None], h1) + ba1[None]
    reconstructed = (
        np.matmul(a1[None], h1 - h0)
        + np.matmul((a1 - a0)[None], h0)
        + (ba1 - ba0)[None]
    )
    affine = affine_increment(source_rows, endpoint_rows, w0, w1, b0, b1)
    direct = g1 - g0
    return {
        "layer": int(layer),
        "strict_crossing_count": int(powers["strict_crossing_count"]),
        "touching_zero_count": int(powers["touching_zero_count"]),
        "minimum_power_base": float(powers["minimum_base"]),
        "maximum_abs_z_secant": float(powers["maximum_abs_z_secant"]),
        "maximum_abs_p_secant": float(powers["maximum_abs_p_secant"]),
        "maximum_abs_multiplied_contribution": float(
            powers["maximum_abs_multiplied_contribution"]
        ),
        "power_base_first_max_abs_residual": float(
            powers["base_first_max_abs_residual"]
        ),
        "power_exponent_first_max_abs_residual": float(
            powers["exponent_first_max_abs_residual"]
        ),
        "preactivation_telescope_max_abs_residual": float(
            affine["maximum_abs_residual"]
        ),
        "plga_output_telescope_max_abs_residual": float(
            np.max(np.abs(reconstructed - direct))
        ),
        "plga_output_scale": max(float(np.max(np.abs(direct))), 1.0),
    }


def prepare_transition(
    checkpoint_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    order_path: str | Path,
    device_name: str,
    *,
    execute_successor: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu"
    )
    registry = load_registry(registry_path)
    if checkpoint["measurement_registry"] != registry:
        raise ValueError("checkpoint and finite-increment registry disagree")
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    update_rows, update_chunks = ordered_training_batch(
        checkpoint, tokens_path, order_path, device
    )
    registry_rows, contexts = registered_tokens(registry, tokens_path, device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(model, update_rows)
    loss.backward()
    torch.nn.utils.clip_grad_value_(
        model.parameters(), float(OPTIMIZER["gradient_clip_value"])
    )
    names, values, _directions, successors = optimizer_displacements(
        model, optimizer
    )
    source_parameters = dict(zip(names, values, strict=True))
    endpoint_parameters = dict(zip(names, successors, strict=True))
    source_parameter_sha256 = tensor_mapping_digest(source_parameters)
    predicted_parameter_sha256 = tensor_mapping_digest(endpoint_parameters)
    shape0, rows0 = capture_maps(model, registry_rows)
    shape1, rows1 = capture_maps(model, registry_rows, endpoint_parameters)
    gate0 = np.stack([
        _gate(source_parameters, layer, "weight")
        for layer in range(int(ARCHITECTURE["layers"]))
    ])
    gate1 = np.stack([
        _gate(endpoint_parameters, layer, "weight")
        for layer in range(int(ARCHITECTURE["layers"]))
    ])
    plga = [
        plga_report(
            layer,
            rows0[:, layer],
            rows1[:, layer],
            source_parameters,
            endpoint_parameters,
        )
        for layer in range(int(ARCHITECTURE["layers"]))
    ]
    actual_shape = None
    actual_rows = None
    actual_parameter_sha256 = None
    parameter_max_abs_residual = None
    if execute_successor:
        optimizer.step()
        actual_parameters = dict(model.named_parameters())
        parameter_max_abs_residual = max(
            float(torch.max(torch.abs(
                actual_parameters[name].detach() - endpoint_parameters[name]
            )).cpu())
            for name in names
        )
        actual_parameter_sha256 = tensor_mapping_digest(actual_parameters)
        actual_shape, actual_rows = capture_maps(model, registry_rows)
    elapsed = time.monotonic() - started
    cuda = device.type == "cuda"
    return {
        "checkpoint": checkpoint,
        "registry": registry,
        "contexts": contexts,
        "update_chunks": update_chunks,
        "source_loss": float(loss.detach().cpu()),
        "source_shape": shape0,
        "predicted_shape": shape1,
        "source_rows": rows0,
        "predicted_rows": rows1,
        "source_gate": gate0,
        "predicted_gate": gate1,
        "source_parameter_sha256": source_parameter_sha256,
        "predicted_parameter_sha256": predicted_parameter_sha256,
        "plga": plga,
        "actual_shape": actual_shape,
        "actual_rows": actual_rows,
        "actual_parameter_sha256": actual_parameter_sha256,
        "native_parameter_max_abs_residual": parameter_max_abs_residual,
        "resources": {
            "wall_seconds": elapsed,
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if cuda else 0
            ),
            "peak_gpu_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if cuda else 0
            ),
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ) * (1024 if sys.platform != "darwin" else 1),
        },
    }


def write_npz_atomic(path: str | Path, **arrays: Any) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def source_metadata(
    transition: dict[str, Any], *, role: str, seed: int, step: int
) -> dict[str, Any]:
    checkpoint = transition["checkpoint"]
    return {
        "role": role,
        "seed": int(seed),
        "step": int(step),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": checkpoint["content_sha256"],
        "registry_sha256": transition["registry"]["registry_sha256"],
        "data_order_sha256": checkpoint["data_state"]["data_order_sha256"],
        "data_cursor_before": int(checkpoint["data_offset_end"]),
        "update_chunks": transition["update_chunks"],
        "source_loss": transition["source_loss"],
        "source_parameter_sha256": transition["source_parameter_sha256"],
        "predicted_parameter_sha256": transition["predicted_parameter_sha256"],
        "context_ids": [row["id"] for row in transition["contexts"]],
        "context_chunks": [
            int(row["chunk_index"]) for row in transition["contexts"]
        ],
        "map_order": "context-major-layer-major-head-major-v1",
        "successor_evaluated": False,
        "plga": transition["plga"],
        "resources": transition["resources"],
    }
