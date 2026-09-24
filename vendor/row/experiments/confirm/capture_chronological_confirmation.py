#!/usr/bin/env python3
"""Capture a pre-update chronological all-pairs confirmation ledger.

The producer opens one complete checkpoint and its digest-bound gradient
history sidecar, follows the actual continuation minibatches, and constructs
every deciding quantity before each native optimizer step. Successor gate and
row tensors are recorded only afterward for independent coverage checks.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.func import functional_call, jvp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from confirm.chronological_confirmation_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    NUMERICAL_POLICIES,
    OPTIMIZER,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.chronological_history import (  # noqa: E402
    final_gate_parameters,
    restore_history,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    nested_state_digest,
    sha256_path,
    tensor_mapping_digest,
    validate_complete_checkpoint,
    validate_measurement_registry,
)
from confirm.gate_shape_evidence import write_npz_atomic  # noqa: E402
from confirm.row_map_live import (  # noqa: E402
    _causal_mask,
    _gate_parameter_name,
    _loss,
    _model_from_raw,
)


RECORD_SCHEMA = "pldr-chronological-pair-ledger-v1"
LOCK_SCHEMA = "pldr-chronological-construction-lock-v1"


def _load_prediction_dependency(
    path: str | Path | None,
    expected_source_sha256: str | None,
) -> dict[str, Any] | None:
    """Validate a source-only prediction before any successor is opened."""

    if path is None:
        if expected_source_sha256 is not None:
            raise ValueError("a source digest requires a prediction report")
        return None
    if expected_source_sha256 is None:
        raise ValueError("a prediction report requires its source digest")
    with Path(path).open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    prediction = (
        report.get("prediction", {}) if isinstance(report, dict) else {})
    schema = report.get("schema_version") if isinstance(report, dict) else None
    if schema == "pldr-causal-row-analysis-v1":
        bound_source = report.get("ledger_sha256")
    elif schema == "pldr-contrast-energy-analysis-v1":
        bound_source = report.get("source_sha256")
    else:
        bound_source = None
    if (
        not isinstance(report, dict)
        or bound_source != expected_source_sha256
        or prediction.get("decision_uses_successor") is not False
        or prediction.get("coverage") is not None
        or report.get("valid") is not True
    ):
        raise ValueError(
            "successor capture needs a valid source-only prediction")
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_path(path),
        "source_ledger_sha256": expected_source_sha256,
        "decision": prediction.get("decision"),
        "metadata": report.get("metadata"),
        "report_schema_version": schema,
    }


def _load_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    validate_measurement_registry(value, require_nonempty=True)
    if value.get("schema_version") != "pldr-chronological-probe-registry-v1":
        raise ValueError("chronological capture needs its registered probe schema")
    contexts = value["construction"] + value["validation"]
    if len(contexts) != REGISTRY["context_count"]:
        raise ValueError("the chronological registry must contain 24 contexts")
    if len({int(row["chunk_index"]) for row in contexts}) != len(contexts):
        raise ValueError("registered context chunks must be unique")
    return value


def _load_lock(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("constant_enlargement_after_lock") is not False
    ):
        raise ValueError("unknown or mutable chronological construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("chronological construction lock digest does not replay")
    return value


def _token_matrix(path: str | Path, context_length: int) -> np.ndarray:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    count = len(data) // context_length
    return data[:count * context_length].reshape(count, context_length)


def _registered_batch(
    matrix: np.ndarray,
    registry: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    contexts = registry["construction"] + registry["validation"]
    indices = [int(row["chunk_index"]) for row in contexts]
    rows = torch.from_numpy(
        matrix[indices].astype(np.int64, copy=True)
    ).to(device)
    heads = torch.arange(len(indices), device=device) % ARCHITECTURE["heads"]
    return rows, heads


def _shape_from_hook(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    heads: torch.Tensor,
    layer: int,
    parameters: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return the ordered union of complete normalized row-map blocks."""

    captured: list[torch.Tensor] = []
    layernorm = (
        model.decoder.dec_layers[layer].mha1.reslayerAs[-1].layernormA
    )

    def hook(module: torch.nn.Module, inputs: tuple[Any, ...], _output: Any) -> None:
        value = inputs[0]
        centered = value - value.mean(dim=-1, keepdim=True)
        captured.append(centered / torch.sqrt(
            module.eps + torch.mean(centered * centered, dim=-1, keepdim=True)
        ))

    handle = layernorm.register_forward_hook(hook)
    inputs = tokens[:, :-1]
    mask = _causal_mask(
        inputs.shape[1], inputs.device, next(model.parameters()).dtype)
    try:
        if parameters is None:
            model([inputs, mask])
        else:
            functional_call(
                model, parameters, ([inputs, mask],), strict=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("the final row-map shape hook did not fire once")
    shape = captured[0]
    if (
        shape.ndim != 4
        or shape.shape[0] != len(tokens)
        or shape.shape[1] != ARCHITECTURE["heads"]
        or shape.shape[2:] != (
            REGISTRY["complete_rows_per_context"], ARCHITECTURE["head_width"])
    ):
        raise ValueError("registered row-map shape has an unexpected size")
    blocks = shape[
        torch.arange(len(tokens), device=tokens.device), heads]
    return blocks.reshape(-1, blocks.shape[-1])


def registered_shape(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    heads: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return _shape_from_hook(model, tokens, heads, layer).detach()
    finally:
        model.train(was_training)


def optimizer_displacements(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[
    tuple[str, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    """Execute AdamW on isolated clones and return its exact native endpoint."""

    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("chronological capture requires AdamW")
    for group in optimizer.param_groups:
        if group.get("amsgrad", False) or group.get("maximize", False):
            raise ValueError("chronological capture requires standard AdamW")

    named = dict(model.named_parameters())
    names = tuple(named)
    values = tuple(named.values())
    clones = tuple(
        torch.nn.Parameter(
            parameter.detach().clone(),
            requires_grad=parameter.requires_grad,
        )
        for parameter in values
    )
    by_original = {
        id(original): clone
        for original, clone in zip(values, clones, strict=True)
    }
    clone_groups: list[dict[str, Any]] = []
    grouped_original_ids: set[int] = set()
    for group in optimizer.param_groups:
        cloned_group = {
            name: value for name, value in group.items() if name != "params"
        }
        cloned_parameters = []
        for parameter in group["params"]:
            if id(parameter) not in by_original:
                raise ValueError("optimizer contains an unregistered parameter")
            grouped_original_ids.add(id(parameter))
            cloned_parameters.append(by_original[id(parameter)])
        cloned_group["params"] = cloned_parameters
        clone_groups.append(cloned_group)
    if grouped_original_ids != set(by_original):
        raise ValueError("model and optimizer parameter registries disagree")

    clone_optimizer = torch.optim.AdamW(clone_groups)
    clone_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for original, clone in zip(values, clones, strict=True):
        clone.grad = (
            None
            if original.grad is None
            else original.grad.detach().clone()
        )
    clone_optimizer.step()
    successors = tuple(clone.detach().clone() for clone in clones)
    directions = tuple(
        successor - original.detach()
        for successor, original in zip(successors, values, strict=True)
    )
    return names, values, directions, successors


def directional_shape_enclosure(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    heads: torch.Tensor,
    layer: int,
    names: tuple[str, ...],
    values: tuple[torch.Tensor, ...],
    directions: tuple[torch.Tensor, ...],
    *,
    segment_grid: tuple[float, ...],
    safety_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stream a full-model shape JVP and sampled segment second derivative."""

    if (
        not segment_grid
        or segment_grid[0] != 0.0
        or segment_grid[-1] != 1.0
        or tuple(sorted(set(segment_grid))) != segment_grid
    ):
        raise ValueError("the segment grid must increase uniquely from zero to one")
    if safety_factor < 1.0 or not math.isfinite(safety_factor):
        raise ValueError("the shape safety factor must be finite and at least one")
    was_training = model.training
    model.eval()

    def shape_function(*local: torch.Tensor) -> torch.Tensor:
        return _shape_from_hook(
            model, tokens, heads, layer, dict(zip(names, local, strict=True)))

    try:
        shape, directional = jvp(shape_function, values, directions)
        maximum_second = torch.zeros(
            shape.shape[0], dtype=shape.dtype, device=shape.device)
        for fraction in segment_grid:
            point = tuple(
                value + fraction * direction
                for value, direction in zip(values, directions, strict=True)
            )

            def first_direction(*local: torch.Tensor) -> torch.Tensor:
                return jvp(shape_function, local, directions)[1]

            _first, second = jvp(first_direction, point, directions)
            maximum_second = torch.maximum(
                maximum_second, torch.linalg.vector_norm(second, dim=1))
        radius = 0.5 * safety_factor * maximum_second
        return shape.detach(), directional.detach(), radius.detach()
    finally:
        model.train(was_training)


def _resource_record(device: torch.device, started: float) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "elapsed_seconds": time.monotonic() - started,
        "device": str(device),
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if cuda else 0),
        "peak_gpu_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if cuda else 0),
        "peak_host_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1024 if sys.platform != "darwin" else 1),
    }


def produce(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    prediction_dependency = _load_prediction_dependency(
        arguments.prediction_report, arguments.expected_source_sha256)
    if arguments.defer_successor and prediction_dependency is not None:
        raise ValueError("source-only capture cannot depend on a prediction")
    if arguments.defer_successor and arguments.updates != 1:
        raise ValueError("source-only capture is exactly one update")
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    validate_complete_checkpoint(checkpoint)
    registry = _load_registry(arguments.registry)
    lock = _load_lock(arguments.lock)
    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint and fixed measurement registry disagree")
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("token archive disagrees with the registry")
    if int(checkpoint["config"]["seed"]) != int(arguments.seed):
        raise ValueError("checkpoint seed disagrees with the requested job")
    if int(checkpoint["config"].get("accum", 1)) != 1:
        raise ValueError("chronological capture currently requires accum=1")
    if not checkpoint["config"].get("const_lr", False):
        raise ValueError("the registered campaign requires constant post-warmup rate")
    if int(checkpoint["step"]) < int(checkpoint["config"]["warmup"]):
        raise ValueError("capture anchors must follow the optimizer warmup")
    matching_trajectory = [
        row for row in TRAJECTORIES
        if row["role"] == arguments.role
        and int(row["seed"]) == int(arguments.seed)
        and int(row["data_offset_chunks"])
        == int(checkpoint["config"]["data_offset"])
    ]
    allowed_anchors = (
        REGISTRY["development_anchor_steps"]
        if arguments.role == "development" else REGISTRY["anchor_steps"]
    )
    if (
        len(matching_trajectory) != 1
        or int(checkpoint["step"]) not in allowed_anchors
    ):
        raise ValueError("capture checkpoint is outside the registered trajectory grid")

    layer = int(arguments.layer)
    if not 0 <= layer < ARCHITECTURE["layers"]:
        raise ValueError("layer is outside the registered architecture")
    if prediction_dependency is not None:
        metadata = prediction_dependency.get("metadata")
        expected_metadata = {
            "checkpoint_sha256": sha256_path(checkpoint_path),
            "role": arguments.role,
            "seed": int(arguments.seed),
            "layer": layer,
            "global_step_before": int(checkpoint["step"]),
        }
        if (
            not isinstance(metadata, dict)
            or any(metadata.get(name) != value
                   for name, value in expected_metadata.items())
        ):
            raise ValueError(
                "prediction report and successor checkpoint disagree")
    if arguments.updates < 1:
        raise ValueError("updates must be positive")
    segment_grid = tuple(
        float(value) for value in arguments.segment_grid.split(",") if value)
    if arguments.role == "heldout":
        if (
            lock is None
            or arguments.updates != int(lock["primary_block_length"])
            or REGISTRY["history_length"] != int(lock["history_length"])
            or segment_grid != tuple(map(float, lock["segment_grid"]))
            or float(arguments.shape_safety_factor)
            != float(lock["shape_safety_factor"])
            or float(arguments.numerical_relative_coefficient)
            != float(lock["numerical_relative_coefficient"])
            or float(arguments.numerical_absolute_charge)
            != float(lock["numerical_absolute_charge"])
        ):
            raise ValueError(
                "held-out capture settings are not bound to the construction lock")
    elif lock is not None:
        raise ValueError(
            "development and construction capture must precede the lock")
    elif arguments.updates not in REGISTRY["construction_block_lengths"]:
        raise ValueError("capture block length is outside the registered grid")

    model = _model_from_raw(checkpoint, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(checkpoint["config"]["lr"]),
        betas=(float(OPTIMIZER["beta1"]), float(OPTIMIZER["beta2"])),
        eps=float(OPTIMIZER["epsilon"]),
        weight_decay=float(checkpoint["config"]["wd"]),
    )
    optimizer.load_state_dict(checkpoint["opt"])
    if (
        nested_state_digest(optimizer.state_dict())
        != checkpoint["state_manifest"]["optimizer_sha256"]
    ):
        raise ValueError("loaded optimizer state differs from its manifest")
    all_gates = final_gate_parameters(model)
    histories = restore_history(
        checkpoint_path,
        all_gates,
        checkpoint_step=int(checkpoint["step"]),
        history_length=REGISTRY["history_length"],
    )
    gate_name = _gate_parameter_name(layer)
    named_parameters = dict(model.named_parameters())
    gate = named_parameters[gate_name]
    bias_name = gate_name.removesuffix(".weight") + ".bias"
    bias = named_parameters[bias_name]
    local_history = histories[gate_name]
    if len(local_history) < REGISTRY["history_length"]:
        raise ValueError("checkpoint sidecar lacks a complete K-step history")

    matrix = _token_matrix(arguments.tokens, registry["context_length"])
    registered_tokens, registered_heads = _registered_batch(
        matrix, registry, device)
    registered_chunks = {
        int(row["chunk_index"])
        for row in registry["construction"] + registry["validation"]
    }
    cursor = int(checkpoint["data_offset_end"])
    if int(checkpoint["data_state"]["cursor_end"]) != cursor:
        raise ValueError("checkpoint cursor fields disagree")
    batch_size = int(checkpoint["config"]["batch"])
    recorded_order_digest = checkpoint["data_state"].get(
        "data_order_sha256")
    data_order = None
    if recorded_order_digest is not None:
        if arguments.data_order is None:
            raise ValueError("an ordered checkpoint requires --data-order")
        if sha256_path(arguments.data_order) != recorded_order_digest:
            raise ValueError("sealed data order disagrees with the checkpoint")
        loaded_order = np.load(arguments.data_order, allow_pickle=False)
        if loaded_order.ndim != 1 or loaded_order.dtype.kind not in "iu":
            raise ValueError("sealed data order must be one-dimensional integers")
        data_order = np.asarray(loaded_order, dtype=np.int64)
        if (
            len(data_order) == 0 or int(data_order.min()) < 0
            or int(data_order.max()) >= len(matrix)
            or len(np.unique(data_order)) != len(data_order)
        ):
            raise ValueError("sealed data order contains invalid chunk indices")
    elif arguments.data_order is not None:
        raise ValueError("checkpoint does not bind the supplied data order")
    available = len(data_order) if data_order is not None else len(matrix)
    if cursor + arguments.updates * batch_size > available:
        raise ValueError("the requested continuation leaves the token archive")

    model.train()
    initial_shape = registered_shape(
        model, registered_tokens, registered_heads, layer
    ).double().cpu().numpy()
    shapes = [initial_shape]
    gates = [gate.detach().double().cpu().numpy()]
    biases = [bias.detach().double().cpu().numpy()]
    physical_rows = [shapes[0] * gates[0] + biases[0]]
    histories_out: list[np.ndarray] = []
    tails_out: list[np.ndarray] = []
    second_before_out: list[np.ndarray] = []
    jvp_out: list[np.ndarray] = []
    physical_jvp_out: list[np.ndarray] = []
    gate_jvp_out: list[np.ndarray] = []
    shape_jvp_out: list[np.ndarray] = []
    bias_jvp_out: list[np.ndarray] = []
    radius_out: list[np.ndarray] = []
    numerical_out: list[np.ndarray] = []
    learning_rates: list[float] = []
    weight_decays: list[float] = []
    optimizer_steps: list[int] = []
    source_losses: list[float] = []
    raw_gradient_sha256: list[str] = []
    clipped_gradient_sha256: list[str] = []
    optimizer_state_sha256: list[str] = []
    source_parameter_sha256: list[str] = []
    parameter_displacement_sha256: list[str] = []
    predicted_successor_parameter_sha256: list[str] = []
    successor_parameter_sha256: list[str] = []
    full_parameter_update_bitwise_match: list[bool] = []
    update_batch_sha256: list[str] = []
    update_batch_chunk_indices: list[np.ndarray] = []
    gate_update_residuals: list[float] = []
    decay_mask: np.ndarray | None = None

    for local_step in range(arguments.updates):
        begin = cursor + local_step * batch_size
        indices = (
            np.arange(begin, begin + batch_size, dtype=np.int64)
            if data_order is None else data_order[begin:begin + batch_size]
        )
        if any(index in registered_chunks for index in indices):
            raise ValueError("a continuation minibatch overlaps fixed probes")
        batch_array = matrix[indices].astype(
            np.int64, copy=True)
        update_batch_chunk_indices.append(np.asarray(indices, dtype=np.int64))
        update_batch_sha256.append(sha256_path_bytes(batch_array.tobytes()))
        rows = torch.from_numpy(batch_array).to(device)

        source_parameter_sha256.append(
            tensor_mapping_digest(dict(model.named_parameters())))
        optimizer_state_sha256.append(
            nested_state_digest(optimizer.state_dict()))
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, rows)
        source_losses.append(float(loss.detach().cpu()))
        loss.backward()
        raw_gradient_sha256.append(tensor_mapping_digest({
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }))
        torch.nn.utils.clip_grad_value_(
            model.parameters(),
            clip_value=float(checkpoint["config"]["clip"]),
        )
        clipped_gradient_sha256.append(tensor_mapping_digest({
            name: parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }))
        gate_state = optimizer.state[gate]
        state_step = gate_state["step"]
        state_step = int(state_step.item()) if torch.is_tensor(state_step) else int(state_step)
        expected_step = int(checkpoint["step"]) + local_step
        if state_step != expected_step:
            raise ValueError("gate optimizer step and checkpoint chronology disagree")
        current = {
            "global_step": expected_step + 1,
            "clipped_gradient": gate.grad.detach().cpu().clone(),
            "first_moment_before": gate_state["exp_avg"].detach().cpu().clone(),
        }
        local_history.append(current)
        del local_history[:-REGISTRY["history_length"]]
        newest = list(reversed(local_history))
        histories_out.append(np.stack([
            row["clipped_gradient"].double().numpy() for row in newest
        ]))
        tails_out.append(
            local_history[0]["first_moment_before"].double().numpy())
        second_before_out.append(
            gate_state["exp_avg_sq"].detach().double().cpu().numpy())

        names, values, directions, native_successors = (
            optimizer_displacements(model, optimizer)
        )
        parameter_displacement_sha256.append(tensor_mapping_digest(
            dict(zip(names, directions, strict=True))))
        predicted_successor_digest = tensor_mapping_digest(
            dict(zip(names, native_successors, strict=True)))
        predicted_successor_parameter_sha256.append(
            predicted_successor_digest)
        shape, shape_jvp, radius = directional_shape_enclosure(
            model,
            registered_tokens,
            registered_heads,
            layer,
            names,
            values,
            directions,
            segment_grid=segment_grid,
            safety_factor=float(arguments.shape_safety_factor),
        )
        source_residual = float(torch.max(torch.abs(
            shape.double().cpu()
            - torch.from_numpy(shapes[-1])
        )).item())
        if source_residual > 64.0 * torch.finfo(shape.dtype).eps:
            raise ValueError("functional and native source-shape probes disagree")
        jvp_out.append(shape_jvp.double().cpu().numpy())
        gate_index = names.index(gate_name)
        bias_index = names.index(bias_name)
        gate_component = shape * directions[gate_index]
        shape_component = shape_jvp * gate
        bias_component = directions[bias_index].expand_as(shape)
        physical_jvp = gate_component + shape_component + bias_component
        gate_jvp_out.append(gate_component.double().cpu().numpy())
        shape_jvp_out.append(shape_component.double().cpu().numpy())
        bias_jvp_out.append(bias_component.double().cpu().numpy())
        physical_jvp_out.append(physical_jvp.double().cpu().numpy())
        radius_out.append(radius.double().cpu().numpy())

        group = next(
            candidate for candidate in optimizer.param_groups
            if any(parameter is gate for parameter in candidate["params"])
        )
        learning_rate = float(group["lr"])
        weight_decay = float(group.get("weight_decay", 0.0))
        learning_rates.append(learning_rate)
        weight_decays.append(weight_decay)
        optimizer_steps.append(state_step + 1)
        predicted_gate = native_successors[gate_index]
        predicted_shape = shape + shape_jvp
        row_scale = torch.sum(
            (predicted_shape * predicted_gate) ** 2, dim=1
        ).detach().double().cpu().numpy()
        relative = float(arguments.numerical_relative_coefficient)
        absolute = float(arguments.numerical_absolute_charge)
        numerical_out.append(
            absolute + relative * np.maximum(row_scale, 1.0))
        local_decay_mask = np.full(
            gate.numel(), weight_decay != 0.0, dtype=np.float64)
        if decay_mask is None:
            decay_mask = local_decay_mask
        elif not np.array_equal(decay_mask, local_decay_mask):
            raise ValueError("the gate decay mask changed within a block")

        successor_digest = ""
        if not arguments.defer_successor:
            optimizer.step()
            actual_gate = gate.detach()
            gate_update_residuals.append(float(torch.max(torch.abs(
                actual_gate - predicted_gate
            )).item()))
            gates.append(actual_gate.double().cpu().numpy())
            biases.append(bias.detach().double().cpu().numpy())
            shapes.append(registered_shape(
                model, registered_tokens, registered_heads, layer
            ).double().cpu().numpy())
            physical_rows.append(shapes[-1] * gates[-1] + biases[-1])
            successor_digest = tensor_mapping_digest(
                dict(model.named_parameters()))
            bitwise_match = successor_digest == predicted_successor_digest
            full_parameter_update_bitwise_match.append(bitwise_match)
            if not bitwise_match:
                raise ValueError(
                    "native AdamW successor differs from its cloned replay")
        successor_parameter_sha256.append(successor_digest)

    if decay_mask is None:
        raise RuntimeError("capture produced no transitions")
    resources = _resource_record(device, started)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(RECORD_SCHEMA),
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "role": np.asarray(arguments.role),
        "seed": np.asarray(arguments.seed, dtype=np.int64),
        "data_offset_chunks": np.asarray(
            int(checkpoint["config"]["data_offset"]), dtype=np.int64),
        "step_start": np.asarray(checkpoint["step"], dtype=np.int64),
        "layer": np.asarray(layer, dtype=np.int64),
        "beta1": np.asarray(OPTIMIZER["beta1"]),
        "beta2": np.asarray(OPTIMIZER["beta2"]),
        "epsilon": np.asarray(OPTIMIZER["epsilon"]),
        "normalized_rows": np.asarray(shapes),
        "gate": np.asarray(gates),
        "final_bias": np.asarray(biases),
        "physical_rows": np.asarray(physical_rows),
        "clipped_gradient_history": np.asarray(histories_out),
        "first_moment_tail": np.asarray(tails_out),
        "second_moment_before": np.asarray(second_before_out),
        "normalized_row_jvp": np.asarray(jvp_out),
        "gate_row_jvp": np.asarray(gate_jvp_out),
        "shape_row_jvp": np.asarray(shape_jvp_out),
        "bias_row_jvp": np.asarray(bias_jvp_out),
        "physical_row_jvp": np.asarray(physical_jvp_out),
        "normalized_row_remainder_radius": np.asarray(radius_out),
        "numerical_charge": np.asarray(numerical_out),
        "learning_rate": np.asarray(learning_rates),
        "weight_decay": np.asarray(weight_decays),
        "decay_mask": decay_mask,
        "optimizer_step_after": np.asarray(optimizer_steps, dtype=np.int64),
        "source_loss": np.asarray(source_losses),
        "raw_gradient_sha256": np.asarray(raw_gradient_sha256),
        "clipped_gradient_sha256": np.asarray(clipped_gradient_sha256),
        "optimizer_state_sha256": np.asarray(optimizer_state_sha256),
        "source_parameter_sha256": np.asarray(source_parameter_sha256),
        "parameter_displacement_sha256": np.asarray(
            parameter_displacement_sha256),
        "predicted_successor_parameter_sha256": np.asarray(
            predicted_successor_parameter_sha256),
        "successor_parameter_sha256": np.asarray(
            successor_parameter_sha256),
        "full_parameter_update_bitwise_match": np.asarray(
            full_parameter_update_bitwise_match, dtype=np.bool_),
        "update_batch_sha256": np.asarray(update_batch_sha256),
        "update_batch_chunk_indices": np.asarray(
            update_batch_chunk_indices, dtype=np.int64),
        "data_cursor_before": np.asarray(cursor, dtype=np.int64),
        "data_cursor_after": np.asarray(
            cursor + arguments.updates * batch_size, dtype=np.int64),
        "data_order_sha256": np.asarray(recorded_order_digest or ""),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "lock_sha256": np.asarray(
            lock["lock_sha256"] if lock is not None else ""),
        "segment_grid": np.asarray(segment_grid),
        "shape_safety_factor": np.asarray(arguments.shape_safety_factor),
        "numerical_relative_coefficient": np.asarray(
            arguments.numerical_relative_coefficient),
        "numerical_absolute_charge": np.asarray(
            arguments.numerical_absolute_charge),
        "gate_update_max_abs_residual": np.asarray(gate_update_residuals),
        "successor_evaluated": np.asarray(not arguments.defer_successor),
        "prediction_dependency_json": np.asarray(json.dumps(
            prediction_dependency, sort_keys=True, separators=(",", ":")
        ) if prediction_dependency is not None else ""),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in resources.items()
        },
    }
    write_npz_atomic(arguments.output, **payload)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "step_start": int(checkpoint["step"]),
        "step_stop": int(checkpoint["step"]) + arguments.updates,
        "layer": layer,
        "vertices": int(np.asarray(shapes).shape[1]),
        "peak_gpu_reserved_bytes": resources["peak_gpu_reserved_bytes"],
    }, sort_keys=True))


def sha256_path_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    float32_policy = NUMERICAL_POLICIES.get(
        "float32_model_tensor", {"safety_factor": 8.0})
    default_relative = (
        float(float32_policy["safety_factor"])
        * 64.0 * np.finfo(np.float32).eps
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--lock")
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--data-order")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--role", choices=[
        "development", "construction", "heldout"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--segment-grid", default="0,0.5,1")
    parser.add_argument("--shape-safety-factor", type=float, default=2.0)
    parser.add_argument(
        "--numerical-relative-coefficient", type=float,
        default=default_relative)
    parser.add_argument("--numerical-absolute-charge", type=float, default=0.0)
    parser.add_argument("--defer-successor", action="store_true")
    parser.add_argument("--prediction-report")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if (
        arguments.numerical_relative_coefficient < 0.0
        or arguments.numerical_absolute_charge < 0.0
    ):
        raise ValueError("numerical charges must be nonnegative")
    produce(arguments)


if __name__ == "__main__":
    main()
