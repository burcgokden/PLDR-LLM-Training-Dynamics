#!/usr/bin/env python3
"""Live producers for the layer-resolved row-map cocycle study."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.func import jacrev


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.capture_chronological_confirmation import (  # noqa: E402
    optimizer_displacements,
)
from confirm.finite_increment_live import (  # noqa: E402
    capture_maps,
    load_checkpoint,
    optimizer_from_checkpoint,
    ordered_training_batch,
    registered_tokens,
)
from confirm.gate_shape import adamw_gate_block  # noqa: E402
from confirm.gate_shape_evidence import write_npz_atomic  # noqa: E402
from confirm.layer_cocycle import (  # noqa: E402
    arithmetic_resolution,
    center_rows,
    layer_observables,
    operator_diagnostics,
    prediction_residual,
    row_preservation_residuals,
)
from confirm.layer_cocycle_specs import (  # noqa: E402
    CAMPAIGN_ID,
    GATE_OPERATOR_POLICY,
    GATE_OPERATOR_SCHEMA,
    INTERVENTION_ARMS,
    INTERVENTION_POLICY,
    INTERVENTION_SCHEMA,
    PERTURBATION_POLICY,
    PERTURBATION_SCHEMA,
    PLGA_POLICY,
    PLGA_SCHEMA,
    PRECISION_SCHEMA,
    RESOLUTION_POLICY,
)
from confirm.native_determinism import sha256_path  # noqa: E402
from confirm.row_map_dynamics import (  # noqa: E402
    dimensionless_state_scale,
    relative_matrix_residual,
)
from confirm.row_map_live import (  # noqa: E402
    _gate_loss_components,
    _gate_microbatches,
    _gate_parameter_name,
    _model_from_raw,
    _weighted_gate_gradient_function,
)
from confirm.source_resolved_observer import _iswiglu  # noqa: E402
from train_run import (  # noqa: E402
    create_masks,
    final_gate_step_target,
    masked_loss_function,
)


def _resource(device: torch.device, started: float) -> dict[str, Any]:
    elapsed = time.monotonic() - started
    cuda = device.type == "cuda"
    return {
        "elapsed_seconds": elapsed,
        "gpu_device_seconds": elapsed if cuda else 0.0,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if cuda else 0
        ),
        "peak_gpu_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if cuda else 0
        ),
        "peak_host_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        * (1024 if sys.platform != "darwin" else 1),
    }


def _prepare_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    return device


def _gates_from_model(model: torch.nn.Module) -> np.ndarray:
    return np.stack(
        [
            model.decoder.dec_layers[layer]
            .mha1.reslayerAs[-1]
            .layernormA.weight.detach()
            .double()
            .cpu()
            .numpy()
            for layer in range(3)
        ]
    )


def _gates_from_parameters(parameters: dict[str, torch.Tensor]) -> np.ndarray:
    return np.stack(
        [
            parameters[_gate_parameter_name(layer)]
            .detach()
            .double()
            .cpu()
            .numpy()
            for layer in range(3)
        ]
    )


def _double_parameters(
    parameters: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device=device, dtype=torch.float64)
        for name, value in parameters.items()
    }


def _capture_parameter_state(
    checkpoint: dict[str, Any],
    token_rows: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float]]:
    model = _model_from_raw(checkpoint, device, dtype=torch.float64)
    local = _double_parameters(parameters, device)
    shape, physical = capture_maps(model, token_rows, local)
    observables = layer_observables(
        shape, physical, _gates_from_parameters(local)
    )
    del model, local
    return shape, physical, observables


def _record_common(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    started: float,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "trajectory": np.asarray(checkpoint["config"]["name"]),
        "seed": np.asarray(checkpoint["config"]["seed"], dtype=np.int64),
        "step": np.asarray(checkpoint["step"], dtype=np.int64),
        "device": np.asarray(str(device)),
    }
    for name, value in _resource(device, started).items():
        values[f"resource_{name}"] = np.asarray(value)
    return values


def precision_record(arguments: argparse.Namespace) -> None:
    """Re-evaluate one checkpoint on float32 and float64 arithmetic paths."""

    device = _prepare_device(arguments.device)
    started = time.monotonic()
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=False, device="cpu"
    )
    registry = __import__(
        "json"
    ).loads(Path(arguments.registry).read_text(encoding="utf-8"))
    token_rows, _contexts = registered_tokens(registry, arguments.tokens, device)
    records: dict[str, dict[str, Any]] = {}
    for label, dtype in (("float32", None), ("float64", torch.float64)):
        model = _model_from_raw(checkpoint, device, dtype=dtype)
        shape, physical = capture_maps(model, token_rows)
        records[label] = {
            "shape": shape,
            "physical": physical,
            "gate": _gates_from_model(model),
            "observables": layer_observables(
                shape, physical, _gates_from_model(model)
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    resolution = arithmetic_resolution(
        records["float64"]["observables"]["energy"],
        records["float32"]["observables"]["energy"],
        safety_factor=float(RESOLUTION_POLICY["arithmetic_safety_factor"]),
        denominator_floor=float(
            RESOLUTION_POLICY["relative_denominator_floor"]
        ),
    )
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PRECISION_SCHEMA),
        **_record_common(checkpoint_path, checkpoint, device, started),
    }
    for arithmetic, record in records.items():
        payload[f"{arithmetic}_energy"] = np.asarray(
            record["observables"]["energy"]
        )
        payload[f"{arithmetic}_shape_energy"] = np.asarray(
            record["observables"]["shape_energy"]
        )
        payload[f"{arithmetic}_effective_squared_gate"] = np.asarray(
            record["observables"]["effective_squared_gate"]
        )
        payload[f"{arithmetic}_coordinate_energy"] = np.asarray(
            record["observables"]["coordinate_energy"]
        )
        payload[f"{arithmetic}_coordinate_shape_energy"] = np.asarray(
            record["observables"]["coordinate_shape_energy"]
        )
        payload[f"{arithmetic}_gate"] = np.asarray(record["gate"])
        payload[f"{arithmetic}_factorization_relative_residual"] = np.asarray(
            record["observables"]["factorization_relative_residual"]
        )
    for name, value in resolution.items():
        payload[f"resolution_{name}"] = np.asarray(value)
    write_npz_atomic(arguments.output, **payload)


def _gate_optimizer(
    checkpoint: dict[str, Any], model: torch.nn.Module, layer: int
) -> tuple[torch.optim.AdamW, torch.nn.Parameter, dict[str, torch.Tensor]]:
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    gate = dict(model.named_parameters())[_gate_parameter_name(layer)]
    state = optimizer.state.get(gate)
    if not state or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
        raise ValueError("checkpoint omits the selected gate optimizer state")
    return optimizer, gate, state


def _sequential_gate_derivatives(
    model: torch.nn.Module,
    token_batches: list[torch.Tensor],
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate the exact loss gradient and Hessian one microbatch at a time."""

    gate_name = _gate_parameter_name(layer)
    gate = dict(model.named_parameters())[gate_name]
    counts = [
        _gate_loss_components(model, rows, gate_name)[2]
        for rows in token_batches
    ]
    total = float(sum(counts))
    gradient = torch.zeros_like(gate)
    hessian_value = torch.zeros(
        (gate.numel(), gate.numel()),
        device=gate.device,
        dtype=gate.dtype,
    )
    for rows, count in zip(token_batches, counts, strict=True):
        _logits, loss_term, _source_count = _gate_loss_components(
            model, rows, gate_name
        )
        gradient_term = torch.func.grad(loss_term)
        local_gradient = gradient_term(gate).detach()
        weight = float(count) / total
        gradient = gradient + weight * local_gradient
        for column in range(gate.numel()):
            tangent = torch.zeros_like(gate)
            tangent[column] = 1.0
            _primal, hessian_column = torch.func.jvp(
                gradient_term, (gate,), (tangent,)
            )
            hessian_value[:, column] = (
                hessian_value[:, column] + weight * hessian_column.detach()
            )
            del _primal, hessian_column, tangent
        del gradient_term, local_gradient, loss_term
        if gate.device.type == "cuda":
            torch.cuda.empty_cache()
    hessian_value = 0.5 * (hessian_value + hessian_value.T)
    return gradient, hessian_value


def _probe_directions(dimension: int, source_scale: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(int(GATE_OPERATOR_POLICY["direction_seed"]))
    directions = []
    for name in GATE_OPERATOR_POLICY["direction_classes"]:
        value = np.zeros(3 * dimension, dtype=np.float64)
        if name == "gate":
            target = slice(0, dimension)
        elif name == "first_moment":
            target = slice(dimension, 2 * dimension)
        elif name == "second_moment":
            target = slice(2 * dimension, 3 * dimension)
        elif name == "mixed":
            target = slice(0, 3 * dimension)
        else:
            raise ValueError("unknown gate probe direction class")
        value[target] = rng.normal(size=len(value[target]))
        value /= np.linalg.norm(value)
        directions.append(value / source_scale)
    return np.stack(directions)


def gate_operator_record(arguments: argparse.Namespace) -> None:
    """Construct and test the exact lifted gate successor derivative."""

    device = _prepare_device(arguments.device)
    started = time.monotonic()
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu"
    )
    model = _model_from_raw(checkpoint, device, dtype=torch.float64)
    _optimizer, gate, state = _gate_optimizer(
        checkpoint, model, int(arguments.layer)
    )
    rows, batch_indices = ordered_training_batch(
        checkpoint, arguments.tokens, arguments.order, device
    )
    microbatches = _gate_microbatches(rows)
    gradient_function = _weighted_gate_gradient_function(
        model, microbatches, int(arguments.layer)
    )
    base_gate = gate.detach()
    gradient_at_gate, true_hessian = _sequential_gate_derivatives(
        model, microbatches, int(arguments.layer)
    )
    direct_gradient = gradient_function(base_gate).detach()
    gradient_denominator = torch.maximum(
        torch.linalg.vector_norm(gradient_at_gate),
        torch.linalg.vector_norm(direct_gradient),
    ).clamp_min(torch.finfo(base_gate.dtype).tiny)
    gradient_relative_residual = torch.linalg.vector_norm(
        gradient_at_gate - direct_gradient
    ) / gradient_denominator
    first = state["exp_avg"].detach()
    second = state["exp_avg_sq"].detach()
    group = _optimizer.param_groups[0]
    beta1, beta2 = map(float, group["betas"])
    epsilon = float(group["eps"])
    rate = float(group["lr"])
    decay = float(group["weight_decay"])
    clip_value = float(checkpoint["config"]["clip"])
    step_before = int(state["step"].item())
    step_after = step_before + 1
    analytic = adamw_gate_block(
        true_hessian.cpu().numpy(),
        base_gate.detach().cpu().numpy(),
        first.detach().cpu().numpy(),
        second.detach().cpu().numpy(),
        gradient_at_gate.cpu().numpy(),
        learning_rate=rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        optimizer_step=step_after,
        weight_decay=decay,
        clip_value=clip_value,
        boundary_tolerance=float(
            GATE_OPERATOR_POLICY["clipping_boundary_tolerance"]
        ),
    )
    dimension = base_gate.numel()
    source_state = torch.cat([base_gate, first, second])

    def complete_successor(vector: torch.Tensor) -> torch.Tensor:
        local_gate = vector[:dimension]
        local_first = vector[dimension : 2 * dimension]
        local_second = vector[2 * dimension :]
        raw = gradient_function(local_gate)
        clipped = torch.clamp(raw, -clip_value, clip_value)
        first_next = beta1 * local_first + (1.0 - beta1) * clipped
        second_next = beta2 * local_second + (1.0 - beta2) * clipped.square()
        first_hat = first_next / (1.0 - beta1**step_after)
        second_hat = second_next / (1.0 - beta2**step_after)
        gate_next = (1.0 - rate * decay) * local_gate - rate * first_hat / (
            torch.sqrt(second_hat) + epsilon
        )
        return torch.cat([gate_next, first_next, second_next])

    def linearized_successor(vector: torch.Tensor) -> torch.Tensor:
        local_gate = vector[:dimension]
        local_first = vector[dimension : 2 * dimension]
        local_second = vector[2 * dimension :]
        raw = gradient_at_gate + true_hessian @ (local_gate - base_gate)
        clipped = torch.clamp(raw, -clip_value, clip_value)
        first_next = beta1 * local_first + (1.0 - beta1) * clipped
        second_next = beta2 * local_second + (1.0 - beta2) * clipped.square()
        first_hat = first_next / (1.0 - beta1**step_after)
        second_hat = second_next / (1.0 - beta2**step_after)
        gate_next = (1.0 - rate * decay) * local_gate - rate * first_hat / (
            torch.sqrt(second_hat) + epsilon
        )
        return torch.cat([gate_next, first_next, second_next])

    operator = np.asarray(analytic["operator"], dtype=np.float64)
    autodiff = (
        jacrev(linearized_successor, chunk_size=1)(source_state)
        .detach()
        .cpu()
        .numpy()
    )
    successor = complete_successor(source_state).detach()
    forcing = complete_successor(torch.zeros_like(source_state)).detach()
    source_second_hat = (
        second / (1.0 - beta2**step_before)
    ).detach().cpu().numpy()
    target_second_hat = (
        successor[2 * dimension :] / (1.0 - beta2**step_after)
    ).detach().cpu().numpy()
    source_scale = dimensionless_state_scale(source_second_hat, epsilon=epsilon)
    target_scale = dimensionless_state_scale(target_second_hat, epsilon=epsilon)
    diagnostics = operator_diagnostics(
        operator, source_scale, target_scale, forcing.cpu().numpy()
    )
    directions = _probe_directions(dimension, source_scale)
    amplitudes = np.asarray(
        GATE_OPERATOR_POLICY["dimensionless_probe_amplitudes"],
        dtype=np.float64,
    )
    absolute_residuals = np.empty((len(directions), len(amplitudes), 2))
    relative_residuals = np.empty_like(absolute_residuals)
    source_tensor = source_state.detach()
    base_successor = successor.detach()
    for direction_index, direction in enumerate(directions):
        tangent = torch.as_tensor(direction, device=device, dtype=source_tensor.dtype)
        linear = torch.as_tensor(
            operator @ direction, device=device, dtype=source_tensor.dtype
        )
        for amplitude_index, amplitude in enumerate(amplitudes):
            for sign_index, sign in enumerate((-1.0, 1.0)):
                delta = sign * float(amplitude) * tangent
                observed = complete_successor(source_tensor + delta).detach()
                predicted = base_successor + sign * float(amplitude) * linear
                residual = torch.linalg.vector_norm(observed - predicted)
                response = torch.linalg.vector_norm(observed - base_successor)
                predicted_response = torch.linalg.vector_norm(
                    predicted - base_successor
                )
                denominator = torch.maximum(response, predicted_response).clamp_min(
                    torch.finfo(response.dtype).tiny
                )
                absolute_residuals[
                    direction_index, amplitude_index, sign_index
                ] = float(residual.cpu())
                relative_residuals[
                    direction_index, amplitude_index, sign_index
                ] = float((residual / denominator).cpu())
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(GATE_OPERATOR_SCHEMA),
        **_record_common(checkpoint_path, checkpoint, device, started),
        "layer": np.asarray(arguments.layer, dtype=np.int64),
        "batch_indices": np.asarray(batch_indices, dtype=np.int64),
        "source_state": source_state.detach().cpu().numpy(),
        "successor_state": successor.cpu().numpy(),
        "invariance_defect": forcing.cpu().numpy(),
        "raw_gradient": gradient_at_gate.cpu().numpy(),
        "true_hessian": true_hessian.cpu().numpy(),
        "gradient_aggregation_relative_residual": np.asarray(
            float(gradient_relative_residual.cpu())
        ),
        "hessian_construction": np.asarray(
            GATE_OPERATOR_POLICY["hessian_construction"]
        ),
        "autodiff_crosscheck_scope": np.asarray(
            GATE_OPERATOR_POLICY["autodiff_crosscheck_scope"]
        ),
        "analytic_operator": operator,
        "autodiff_operator": autodiff,
        "operator_relative_residual": np.asarray(
            relative_matrix_residual(operator, autodiff)
        ),
        "source_state_scale": source_scale,
        "target_state_scale": target_scale,
        "probe_direction_classes": np.asarray(
            GATE_OPERATOR_POLICY["direction_classes"]
        ),
        "probe_directions": directions,
        "probe_amplitudes": amplitudes,
        "probe_absolute_residuals": absolute_residuals,
        "probe_relative_residuals": relative_residuals,
        "learning_rate": np.asarray(rate),
        "weight_decay": np.asarray(decay),
        "beta1": np.asarray(beta1),
        "beta2": np.asarray(beta2),
        "adam_epsilon": np.asarray(epsilon),
        "optimizer_step_before": np.asarray(step_before, dtype=np.int64),
        "optimizer_step_after": np.asarray(step_after, dtype=np.int64),
    }
    for name, value in diagnostics.items():
        payload[name] = np.asarray(value)
    write_npz_atomic(arguments.output, **payload)


def _run_update(
    checkpoint: dict[str, Any],
    tokens: str,
    order: str,
    device: torch.device,
    *,
    parameter_transform: Callable[[dict[str, torch.nn.Parameter]], None]
    | None = None,
) -> dict[str, Any]:
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    named = dict(model.named_parameters())
    if parameter_transform is not None:
        parameter_transform(named)
    source = {name: value.detach().clone() for name, value in named.items()}
    rows, batch_indices = ordered_training_batch(
        checkpoint, tokens, order, device
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    inputs = rows[:, :-1]
    targets = rows[:, 1:]
    logits = model([inputs, create_masks(inputs, device)])[0]
    loss = masked_loss_function(targets, logits)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), float(checkpoint["config"]["clip"]))
    names, _values, _directions, successors = optimizer_displacements(
        model, optimizer
    )
    endpoint = {
        name: value for name, value in zip(names, successors, strict=True)
    }
    del model, optimizer, logits, loss
    return {
        "source": source,
        "endpoint": endpoint,
        "batch_indices": batch_indices,
    }


def _copy_parameters(
    parameters: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in parameters.items()}


def _intervention_parameters(
    source: dict[str, torch.Tensor],
    native: dict[str, torch.Tensor],
    arm: str,
    *,
    layer: int,
    learning_rate: float,
    weight_decay: float,
) -> dict[str, torch.Tensor]:
    result = _copy_parameters(native)
    row_prefix = f"decoder.dec_layers.{layer}.mha1.reslayerAs."
    gate_name = _gate_parameter_name(layer)
    for name in result:
        row_program = name.startswith(row_prefix)
        final_gate = name == gate_name
        if arm == "gate_update_removed" and final_gate:
            result[name] = source[name]
        elif arm == "upstream_update_removed" and row_program and not final_gate:
            result[name] = source[name]
        elif arm == "joint_row_program_update_removed" and row_program:
            result[name] = source[name]
        elif arm in {"gate_decay_only", "gate_adaptive_only"} and final_gate:
            mode = "decay_only" if arm == "gate_decay_only" else "adaptive_only"
            result[name] = final_gate_step_target(
                source[name], native[name], learning_rate, weight_decay, mode
            )
    return result


def intervention_record(arguments: argparse.Namespace) -> None:
    """Run one-step component arms and a same-device one-ulp placebo."""

    device = _prepare_device(arguments.device)
    started = time.monotonic()
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu"
    )
    registry = __import__("json").loads(
        Path(arguments.registry).read_text(encoding="utf-8")
    )
    token_rows, _contexts = registered_tokens(registry, arguments.tokens, device)
    native_update = _run_update(
        checkpoint, arguments.tokens, arguments.order, device
    )
    placebo_name = _gate_parameter_name(int(arguments.layer))
    coordinate = int(INTERVENTION_POLICY["placebo_coordinate"])
    placebo_values: dict[str, float] = {}

    def apply_placebo(named: dict[str, torch.nn.Parameter]) -> None:
        parameter = named[placebo_name]
        flat = parameter.view(-1)
        before = flat[coordinate].detach().clone()
        after = torch.nextafter(before, torch.full_like(before, math.inf))
        with torch.no_grad():
            flat[coordinate].copy_(after)
        placebo_values["before"] = float(before.cpu())
        placebo_values["after"] = float(after.cpu())

    placebo_update = _run_update(
        checkpoint,
        arguments.tokens,
        arguments.order,
        device,
        parameter_transform=apply_placebo,
    )
    config = checkpoint["config"]
    arm_parameters: dict[str, dict[str, torch.Tensor]] = {
        "native": native_update["endpoint"],
        "noop_restamp": _copy_parameters(native_update["endpoint"]),
        "one_ulp_placebo": placebo_update["endpoint"],
    }
    for arm in INTERVENTION_ARMS:
        if arm not in arm_parameters:
            arm_parameters[arm] = _intervention_parameters(
                native_update["source"],
                native_update["endpoint"],
                arm,
                layer=int(arguments.layer),
                learning_rate=float(config["lr"]),
                weight_decay=float(config["wd"]),
            )
    if tuple(arm_parameters) != INTERVENTION_ARMS:
        raise AssertionError("intervention arm order drifted")
    source_shape, source_physical, source_observables = _capture_parameter_state(
        checkpoint,
        token_rows,
        native_update["source"],
        device,
    )
    arm_energy = []
    arm_shape_energy = []
    arm_effective_gate = []
    arm_gates = []
    arm_physical_response = []
    arm_shape_response = []
    native_physical: np.ndarray | None = None
    native_shape: np.ndarray | None = None
    for arm in INTERVENTION_ARMS:
        shape, physical, observables = _capture_parameter_state(
            checkpoint, token_rows, arm_parameters[arm], device
        )
        if arm == "native":
            native_physical = physical
            native_shape = shape
        assert native_physical is not None and native_shape is not None
        arm_energy.append(observables["energy"])
        arm_shape_energy.append(observables["shape_energy"])
        arm_effective_gate.append(observables["effective_squared_gate"])
        arm_gates.append(_gates_from_parameters(arm_parameters[arm]))
        arm_physical_response.append(
            np.linalg.norm(center_rows(physical - native_physical), axis=(-2, -1))
        )
        arm_shape_response.append(
            np.linalg.norm(center_rows(shape - native_shape), axis=(-2, -1))
        )
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(INTERVENTION_SCHEMA),
        "layer": np.asarray(arguments.layer, dtype=np.int64),
        **_record_common(checkpoint_path, checkpoint, device, started),
        "horizon_updates": np.asarray(
            INTERVENTION_POLICY["horizon_updates"], dtype=np.int64
        ),
        "batch_indices": np.asarray(
            native_update["batch_indices"], dtype=np.int64
        ),
        "placebo_parameter": np.asarray(placebo_name),
        "placebo_coordinate": np.asarray(coordinate, dtype=np.int64),
        "placebo_before": np.asarray(placebo_values["before"]),
        "placebo_after": np.asarray(placebo_values["after"]),
        "arm_names": np.asarray(INTERVENTION_ARMS),
        "source_energy": np.asarray(source_observables["energy"]),
        "source_shape_energy": np.asarray(source_observables["shape_energy"]),
        "source_effective_squared_gate": np.asarray(
            source_observables["effective_squared_gate"]
        ),
        "source_gate": _gates_from_parameters(native_update["source"]),
        "arm_energy": np.asarray(arm_energy),
        "arm_shape_energy": np.asarray(arm_shape_energy),
        "arm_effective_squared_gate": np.asarray(arm_effective_gate),
        "arm_gate": np.asarray(arm_gates),
        "arm_physical_response_from_native": np.asarray(arm_physical_response),
        "arm_shape_response_from_native": np.asarray(arm_shape_response),
        "source_shape_capture_norm": np.asarray(np.linalg.norm(source_shape)),
        "source_physical_capture_norm": np.asarray(
            np.linalg.norm(source_physical)
        ),
    }
    write_npz_atomic(arguments.output, **payload)


def _upstream_direction(
    parameters: dict[str, torch.Tensor], layer: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], float, str]:
    prefix = f"decoder.dec_layers.{layer}.mha1.reslayerAs."
    final_gate = prefix + "7.layernormA.weight"
    names = sorted(
        name
        for name in parameters
        if name.startswith(prefix) and name != final_gate
    )
    if not names:
        raise ValueError("upstream row-program parameter selector is empty")
    rng = np.random.default_rng(
        int(PERTURBATION_POLICY["direction_seed"]) + layer
    )
    direction: dict[str, torch.Tensor] = {}
    direction_squared = 0.0
    parameter_squared = 0.0
    digest = hashlib.sha256()
    for name in names:
        value = parameters[name]
        array = rng.normal(size=tuple(value.shape)).astype(np.float64)
        direction_squared += float(np.sum(array * array))
        parameter_squared += float(
            torch.sum(value.detach().double().cpu() ** 2).item()
        )
        direction[name] = torch.from_numpy(array).to(
            device=device, dtype=value.dtype
        )
    direction_norm = math.sqrt(direction_squared)
    parameter_norm = math.sqrt(parameter_squared)
    if direction_norm == 0.0 or parameter_norm == 0.0:
        raise ValueError("row-program perturbation has zero scale")
    for name in names:
        direction[name] = direction[name] / direction_norm
        digest.update(name.encode("utf-8"))
        digest.update(direction[name].detach().cpu().numpy().tobytes())
    return direction, parameter_norm, digest.hexdigest()


def perturbation_record(arguments: argparse.Namespace) -> None:
    """Construct a sign-paired secant, write it, then open half amplitude."""

    device = _prepare_device(arguments.device)
    started = time.monotonic()
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu"
    )
    registry = __import__("json").loads(
        Path(arguments.registry).read_text(encoding="utf-8")
    )
    token_rows, _contexts = registered_tokens(registry, arguments.tokens, device)
    base_model = _model_from_raw(checkpoint, device)
    base_parameters = dict(base_model.named_parameters())
    direction, parameter_norm, direction_sha256 = _upstream_direction(
        base_parameters, int(arguments.layer), device
    )
    del base_model, base_parameters
    amplitude = float(PERTURBATION_POLICY["construction_amplitude"])

    def transform(sign: float, fraction: float = 1.0):
        def apply(named: dict[str, torch.nn.Parameter]) -> None:
            with torch.no_grad():
                for name, value in direction.items():
                    named[name].add_(
                        sign * fraction * amplitude * parameter_norm * value
                    )

        return apply

    native = _run_update(checkpoint, arguments.tokens, arguments.order, device)
    plus = _run_update(
        checkpoint,
        arguments.tokens,
        arguments.order,
        device,
        parameter_transform=transform(1.0),
    )
    minus = _run_update(
        checkpoint,
        arguments.tokens,
        arguments.order,
        device,
        parameter_transform=transform(-1.0),
    )
    native_source = _capture_parameter_state(
        checkpoint, token_rows, native["source"], device
    )[1]
    native_endpoint = _capture_parameter_state(
        checkpoint, token_rows, native["endpoint"], device
    )[1]
    plus_source = _capture_parameter_state(
        checkpoint, token_rows, plus["source"], device
    )[1]
    plus_endpoint = _capture_parameter_state(
        checkpoint, token_rows, plus["endpoint"], device
    )[1]
    minus_source = _capture_parameter_state(
        checkpoint, token_rows, minus["source"], device
    )[1]
    minus_endpoint = _capture_parameter_state(
        checkpoint, token_rows, minus["endpoint"], device
    )[1]
    fraction = float(PERTURBATION_POLICY["validation_amplitude_fraction"])
    predicted = native_endpoint + 0.5 * fraction * (
        plus_endpoint - minus_endpoint
    )
    prediction_created_at_ns = time.time_ns()
    prediction_payload = {
        "schema_version": np.asarray(PERTURBATION_SCHEMA),
        "stage": np.asarray("prediction"),
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "trajectory": np.asarray(checkpoint["config"]["name"]),
        "step": np.asarray(checkpoint["step"], dtype=np.int64),
        "layer": np.asarray(arguments.layer, dtype=np.int64),
        "device": np.asarray(str(device)),
        "direction_sha256": np.asarray(direction_sha256),
        "construction_amplitude": np.asarray(amplitude),
        "validation_fraction": np.asarray(fraction),
        "prediction_created_at_ns": np.asarray(
            prediction_created_at_ns, dtype=np.int64
        ),
        "native_source": native_source,
        "plus_source": plus_source,
        "minus_source": minus_source,
        "native_endpoint": native_endpoint,
        "plus_endpoint": plus_endpoint,
        "minus_endpoint": minus_endpoint,
        "predicted_validation_endpoint": predicted,
    }
    write_npz_atomic(arguments.prediction_output, **prediction_payload)
    prediction_sha256 = sha256_path(arguments.prediction_output)
    validation_opened_at_ns = time.time_ns()
    validation = _run_update(
        checkpoint,
        arguments.tokens,
        arguments.order,
        device,
        parameter_transform=transform(1.0, fraction),
    )
    validation_source = _capture_parameter_state(
        checkpoint, token_rows, validation["source"], device
    )[1]
    validation_endpoint = _capture_parameter_state(
        checkpoint, token_rows, validation["endpoint"], device
    )[1]
    layer = int(arguments.layer)
    residual = prediction_residual(
        predicted[:, layer],
        validation_endpoint[:, layer],
        native_endpoint[:, layer],
    )
    source_distance = np.linalg.norm(
        center_rows(validation_source[:, layer] - native_source[:, layer]),
        axis=(-2, -1),
    )
    endpoint_distance = np.linalg.norm(
        center_rows(validation_endpoint[:, layer] - native_endpoint[:, layer]),
        axis=(-2, -1),
    )
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PERTURBATION_SCHEMA),
        "stage": np.asarray("validation"),
        **_record_common(checkpoint_path, checkpoint, device, started),
        "layer": np.asarray(layer, dtype=np.int64),
        "direction_sha256": np.asarray(direction_sha256),
        "prediction_sha256": np.asarray(prediction_sha256),
        "prediction_created_at_ns": np.asarray(
            prediction_created_at_ns, dtype=np.int64
        ),
        "validation_opened_at_ns": np.asarray(
            validation_opened_at_ns, dtype=np.int64
        ),
        "construction_amplitude": np.asarray(amplitude),
        "validation_fraction": np.asarray(fraction),
        "parameter_norm": np.asarray(parameter_norm),
        "source_distance_by_map": source_distance,
        "endpoint_distance_by_map": endpoint_distance,
        "predicted_validation_endpoint": predicted[:, layer],
        "observed_validation_endpoint": validation_endpoint[:, layer],
    }
    for name, value in residual.items():
        payload[name] = np.asarray(value)
    write_npz_atomic(arguments.output, **payload)


def _plga_apply(
    value: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    exponent: np.ndarray,
    coupling: np.ndarray,
    coupling_bias: np.ndarray,
) -> np.ndarray:
    base = _iswiglu(weight @ value + bias) + float(
        PLGA_POLICY["regularization_constant"]
    )
    return coupling @ np.exp(exponent * np.log(base)) + coupling_bias


def plga_record(arguments: argparse.Namespace) -> None:
    """Measure the exact row-constant PLGA defect and structural residuals."""

    device = _prepare_device(arguments.device)
    started = time.monotonic()
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=False, device="cpu"
    )
    registry = __import__("json").loads(
        Path(arguments.registry).read_text(encoding="utf-8")
    )
    token_rows, _contexts = registered_tokens(registry, arguments.tokens, device)
    model = _model_from_raw(checkpoint, device, dtype=torch.float64)
    _shape, physical = capture_maps(model, token_rows)
    input_norms = []
    output_norms = []
    response_norms = []
    defect_norms = []
    secant_gains = []
    condition_names = (
        "weight_ones_quotient",
        "bias_row_quotient",
        "exponent_row_quotient",
        "coupling_ones_quotient",
        "coupling_bias_row_quotient",
    )
    conditions = np.empty((3, 4, len(condition_names)))
    for layer in range(3):
        module = model.decoder.dec_layers[layer].mha1.plgatt_layer
        weights = module.Wlst.detach().cpu().numpy()
        biases = module.blst.detach().cpu().numpy()
        exponents = module.pwlst.detach().cpu().numpy()
        couplings = module.alst.detach().cpu().numpy()
        coupling_biases = module.balst.detach().cpu().numpy()
        for head in range(4):
            row = row_preservation_residuals(
                weights[head],
                biases[head],
                exponents[head],
                couplings[head],
                coupling_biases[head],
            )
            conditions[layer, head] = [row[name] for name in condition_names]
        layer_input = []
        layer_output = []
        layer_response = []
        layer_defect = []
        layer_gain = []
        for context in range(physical.shape[0]):
            for head in range(physical.shape[2]):
                value = physical[context, layer, head]
                constant = np.broadcast_to(
                    value.mean(axis=-2, keepdims=True), value.shape
                )
                direct = _plga_apply(
                    value,
                    weights[head],
                    biases[head],
                    exponents[head],
                    couplings[head],
                    coupling_biases[head],
                )
                baseline = _plga_apply(
                    constant,
                    weights[head],
                    biases[head],
                    exponents[head],
                    couplings[head],
                    coupling_biases[head],
                )
                source = float(np.linalg.norm(center_rows(value)))
                response = float(np.linalg.norm(center_rows(direct - baseline)))
                defect = float(np.linalg.norm(center_rows(baseline)))
                output = float(np.linalg.norm(center_rows(direct)))
                layer_input.append(source)
                layer_output.append(output)
                layer_response.append(response)
                layer_defect.append(defect)
                layer_gain.append(0.0 if source == 0.0 else response / source)
        input_norms.append(layer_input)
        output_norms.append(layer_output)
        response_norms.append(layer_response)
        defect_norms.append(layer_defect)
        secant_gains.append(layer_gain)
    input_norms = np.asarray(input_norms)
    output_norms = np.asarray(output_norms)
    response_norms = np.asarray(response_norms)
    defect_norms = np.asarray(defect_norms)
    secant_gains = np.asarray(secant_gains)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PLGA_SCHEMA),
        **_record_common(checkpoint_path, checkpoint, device, started),
        "condition_names": np.asarray(condition_names),
        "row_preservation_residuals": conditions,
        "input_quotient_norm": input_norms,
        "output_quotient_norm": output_norms,
        "segment_response_norm": response_norms,
        "row_constant_defect_norm": defect_norms,
        "realized_secant_gain": secant_gains,
        "triangle_upper": response_norms + defect_norms,
        "triangle_signed_slack": response_norms + defect_norms - output_norms,
    }
    write_npz_atomic(arguments.output, **payload)
    del model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser, *, order: bool = False) -> None:
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--tokens", required=True)
        command.add_argument("--registry", required=True)
        if order:
            command.add_argument("--order", required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output", required=True)

    command = subparsers.add_parser("precision")
    common(command)
    command.set_defaults(function=precision_record)

    command = subparsers.add_parser("gate-operator")
    common(command, order=True)
    command.add_argument("--layer", required=True, type=int, choices=(0, 1, 2))
    command.set_defaults(function=gate_operator_record)

    command = subparsers.add_parser("intervention")
    common(command, order=True)
    command.add_argument("--layer", required=True, type=int, choices=(0, 1, 2))
    command.set_defaults(function=intervention_record)

    command = subparsers.add_parser("perturbation")
    common(command, order=True)
    command.add_argument("--layer", required=True, type=int, choices=(0, 1, 2))
    command.add_argument("--prediction-output", required=True)
    command.set_defaults(function=perturbation_record)

    command = subparsers.add_parser("plga")
    common(command)
    command.set_defaults(function=plga_record)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
