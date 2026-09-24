#!/usr/bin/env python3
"""Matrix-free full-state AdamW cocycle and row-face live producers.

The homogeneous producer propagates tangent vectors in the complete
``(parameter, first moment, second moment)`` state.  It forms Hessian-vector
products of the actual next minibatch losses and never materializes a dense
Jacobian.  Signed nonlinear branches are executed from the same immutable
checkpoint for construction calibration and held-out validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.func import functional_call, jvp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT))

from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.finite_increment_live import (  # noqa: E402
    load_checkpoint,
    optimizer_from_checkpoint,
    registered_tokens,
)
from confirm.gate_shape_evidence import write_npz_atomic  # noqa: E402
from confirm.observable_cocycle import (  # noqa: E402
    aggregate_norm,
    digest_object,
)
from confirm.observable_cocycle_specs import (  # noqa: E402
    ARCHITECTURE,
    CALIBRATION_SCHEMA,
    CAMPAIGN_ID,
    DECISION_POLICY,
    DIRECTION_CLASSES,
    DIRECTION_POLICY,
    HORIZON_GRID,
    LOCK_SCHEMA,
    NUMERICAL_POLICY,
    PREDICTION_SCHEMA,
    QUALIFICATION_SCHEMA,
    RESOURCE_BUDGET,
    STRATUM_SCHEMA,
    VALIDATION_SCHEMA,
    amplitudes,
)
from confirm.row_map_live import (  # noqa: E402
    _causal_mask,
    _gate_parameter_name,
    _loss,
    _model_from_raw,
)
from train_run import validate_data_order  # noqa: E402


TensorTree = tuple[torch.Tensor, ...]
Direction = tuple[TensorTree, TensorTree, TensorTree]


def _prepare_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    if bool(NUMERICAL_POLICY["deterministic_algorithms"]):
        torch.use_deterministic_algorithms(True)
    return device


def _resource(device: torch.device, started: float) -> dict[str, np.ndarray]:
    elapsed = time.monotonic() - started
    cuda = device.type == "cuda"
    return {
        "resource_elapsed_seconds": np.asarray(elapsed),
        "resource_reserved_device_seconds": np.asarray(elapsed if cuda else 0.0),
        "resource_peak_gpu_allocated_bytes": np.asarray(
            int(torch.cuda.max_memory_allocated(device)) if cuda else 0,
            dtype=np.int64,
        ),
        "resource_peak_gpu_reserved_bytes": np.asarray(
            int(torch.cuda.max_memory_reserved(device)) if cuda else 0,
            dtype=np.int64,
        ),
        "resource_peak_host_bytes": np.asarray(
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1024 if sys.platform != "darwin" else 1),
            dtype=np.int64,
        ),
    }


def _load_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("observable-cocycle registry must be an object")
    return value


def _common_record(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    layer: int,
) -> dict[str, np.ndarray]:
    return {
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "trajectory": np.asarray(str(checkpoint["config"]["name"])),
        "seed": np.asarray(int(checkpoint["config"]["seed"]), dtype=np.int64),
        "source_step": np.asarray(int(checkpoint["step"]), dtype=np.int64),
        "layer": np.asarray(int(layer), dtype=np.int64),
        "device": np.asarray(str(device)),
    }


def _ordered_batches(
    checkpoint: dict[str, Any],
    tokens_path: str | Path,
    order_path: str | Path,
    device: torch.device,
    count: int,
) -> tuple[list[torch.Tensor], np.ndarray]:
    """Return the exact consecutive minibatches after a complete checkpoint."""

    config = checkpoint["config"]
    context = int(config["ctx"])
    batch = int(config["batch"])
    cursor = int(checkpoint["data_offset_end"])
    tokens_file = Path(tokens_path).resolve()
    order_file = Path(order_path).resolve()
    if checkpoint["data_state"].get("dataset_sha256") != sha256_path(tokens_file):
        raise ValueError("checkpoint and token archive disagree")
    if checkpoint["data_state"].get("data_order_sha256") != sha256_path(order_file):
        raise ValueError("checkpoint and frozen data order disagree")
    data = np.memmap(tokens_file, dtype=np.uint16, mode="r")
    chunk_count = len(data) // context
    chunks = data[: chunk_count * context].reshape(chunk_count, context)
    order = validate_data_order(
        np.load(order_file, allow_pickle=False), chunk_count
    )
    selected = order[cursor : cursor + count * batch]
    if len(selected) != count * batch:
        raise ValueError("requested horizon leaves the frozen data order")
    result = []
    for index in range(count):
        local = selected[index * batch : (index + 1) * batch]
        result.append(
            torch.from_numpy(chunks[local].astype(np.int64, copy=True)).to(device)
        )
    return result, np.asarray(selected, dtype=np.int64).reshape(count, batch)


def _named_state(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
) -> tuple[tuple[str, ...], TensorTree, TensorTree, TensorTree]:
    named = tuple(model.named_parameters())
    names = tuple(name for name, _parameter in named)
    parameters = tuple(parameter for _name, parameter in named)
    first: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    steps = set()
    for parameter in parameters:
        state = optimizer.state.get(parameter)
        if not state or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
            raise ValueError("complete checkpoint omits an AdamW state coordinate")
        first.append(state["exp_avg"])
        second.append(state["exp_avg_sq"])
        step = state["step"]
        steps.add(int(step.item()) if torch.is_tensor(step) else int(step))
    if len(steps) != 1:
        raise ValueError("optimizer clocks are not synchronized")
    if sum(parameter.numel() for parameter in parameters) != int(
        ARCHITECTURE["parameter_count"]
    ):
        raise ValueError("full-state parameter registry has the wrong dimension")
    return names, parameters, tuple(first), tuple(second)


def _tree_squared_norm(values: TensorTree) -> torch.Tensor:
    if not values:
        raise ValueError("a state block cannot be empty")
    result = torch.zeros((), dtype=torch.float64, device=values[0].device)
    for value in values:
        result = result + torch.sum(value.detach().double().square())
    return result


def _tree_norm(values: TensorTree) -> float:
    return float(torch.sqrt(_tree_squared_norm(values)).item())


def _zeros_like(values: TensorTree) -> TensorTree:
    return tuple(torch.zeros_like(value) for value in values)


def _direction_summary(direction: Direction) -> dict[str, float | int]:
    theta, first, second = direction
    return {
        "parameter_norm": _tree_norm(theta),
        "first_moment_norm": _tree_norm(first),
        "second_moment_norm": _tree_norm(second),
        "parameter_nonzero": sum(int(torch.count_nonzero(x).item()) for x in theta),
        "first_moment_nonzero": sum(
            int(torch.count_nonzero(x).item()) for x in first
        ),
        "second_moment_nonzero": sum(
            int(torch.count_nonzero(x).item()) for x in second
        ),
    }


def _rademacher_like(
    values: TensorTree,
    generator: torch.Generator,
) -> TensorTree:
    result = []
    for value in values:
        bits = torch.randint(
            0,
            2,
            value.shape,
            device=value.device,
            generator=generator,
            dtype=torch.int8,
        )
        result.append(bits.to(value.dtype).mul_(2.0).sub_(1.0))
    return tuple(result)


def _physical_observation(
    model: torch.nn.Module,
    rows: torch.Tensor,
    layer: int,
    parameters: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return all registered centered physical row maps for one layer."""

    captured: list[torch.Tensor] = []
    layernorm = (
        model.decoder.dec_layers[layer].mha1.reslayerAs[-1].layernormA
    )

    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append(output)

    handle = layernorm.register_forward_hook(hook)
    inputs = rows[:, :-1]
    mask = _causal_mask(
        inputs.shape[1], inputs.device, next(model.parameters()).dtype
    )
    try:
        if parameters is None:
            model([inputs, mask])
        else:
            functional_call(model, parameters, ([inputs, mask],), strict=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("selected row-map observation did not fire once")
    value = captured[0]
    expected = (
        int(ARCHITECTURE["registered_contexts"]),
        int(ARCHITECTURE["heads_per_layer"]),
        int(ARCHITECTURE["head_width"]),
        int(ARCHITECTURE["head_width"]),
    )
    if tuple(value.shape) != expected:
        raise ValueError(
            f"registered physical observation has {tuple(value.shape)}, "
            f"expected {expected}"
        )
    return value - value.mean(dim=-2, keepdim=True)


def _observation_and_actions(
    model: torch.nn.Module,
    rows: torch.Tensor,
    layer: int,
    names: tuple[str, ...],
    parameters: TensorTree,
    directions: dict[str, Direction],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    def observe(*local: torch.Tensor) -> torch.Tensor:
        return _physical_observation(
            model, rows, layer, dict(zip(names, local, strict=True))
        )

    base: torch.Tensor | None = None
    actions: dict[str, np.ndarray] = {}
    for direction_name in DIRECTION_CLASSES:
        local_base, action = jvp(
            observe, parameters, directions[direction_name][0]
        )
        if base is None:
            base = local_base.detach()
        elif not torch.equal(base, local_base.detach()):
            raise ArithmeticError("observation primal changed across JVP directions")
        actions[direction_name] = (
            action.detach().to(dtype=torch.float32).cpu().numpy()
        )
        del local_base, action
    assert base is not None
    return base.to(dtype=torch.float32).cpu().numpy(), actions


def _raw_gradient(
    model: torch.nn.Module,
    parameters: TensorTree,
    rows: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, TensorTree]:
    loss = _loss(model, rows)
    gradients = torch.autograd.grad(
        loss, parameters, create_graph=create_graph, allow_unused=False
    )
    return loss.detach(), tuple(gradients)


def _optimizer_displacement(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    parameters: TensorTree,
    first: TensorTree,
    second: TensorTree,
    rows: torch.Tensor,
) -> tuple[Direction, dict[str, Any]]:
    """Construct the realized next-step direction at the source state."""

    _loss_value, raw = _raw_gradient(
        model, parameters, rows, create_graph=False
    )
    group = optimizer.param_groups[0]
    beta1, beta2 = map(float, group["betas"])
    rate = float(group["lr"])
    epsilon = float(group["eps"])
    decay = float(group["weight_decay"])
    clip = float(getattr(model, "_observable_clip", 1.0))
    step_before = int(optimizer.state[parameters[0]]["step"].item())
    step_after = step_before + 1
    a1 = 1.0 - beta1**step_after
    a2 = 1.0 - beta2**step_after
    dtheta: list[torch.Tensor] = []
    dfirst: list[torch.Tensor] = []
    dsecond: list[torch.Tensor] = []
    adjusted = 0
    largest = max(amplitudes("optimizer_displacement"))
    for parameter, moment, variance, gradient in zip(
        parameters, first, second, raw, strict=True
    ):
        clipped = torch.clamp(gradient.detach(), -clip, clip)
        moment_next = beta1 * moment + (1.0 - beta1) * clipped
        variance_next = beta2 * variance + (1.0 - beta2) * clipped.square()
        moment_hat = moment_next / a1
        variance_hat = variance_next / a2
        parameter_next = (1.0 - rate * decay) * parameter.detach() - rate * (
            moment_hat / (torch.sqrt(variance_hat) + epsilon)
        )
        local_second = variance_next - variance
        positive = variance > 0.0
        local_second = torch.where(positive, local_second, torch.zeros_like(local_second))
        limit = torch.where(
            positive,
            0.9 * variance / largest,
            torch.zeros_like(variance),
        )
        clipped_second = torch.maximum(
            torch.minimum(local_second, limit), -limit
        )
        adjusted += int(torch.count_nonzero(clipped_second != local_second).item())
        dtheta.append((parameter_next - parameter.detach()).detach())
        dfirst.append((moment_next - moment).detach())
        dsecond.append(clipped_second.detach())
    direction = (tuple(dtheta), tuple(dfirst), tuple(dsecond))
    return direction, {
        "second_moment_adjusted_coordinates": adjusted,
        "step_after": step_after,
    }


def _row_adjoint_direction(
    model: torch.nn.Module,
    rows: torch.Tensor,
    layer: int,
    parameters: TensorTree,
    first: TensorTree,
    second: TensorTree,
) -> Direction:
    output = _physical_observation(model, rows, layer)
    generator = torch.Generator(device=output.device)
    generator.manual_seed(int(DIRECTION_POLICY["row_adjoint_seed"]) + layer)
    cotangent = _rademacher_like((output,), generator)[0]
    cotangent = cotangent - cotangent.mean(dim=-2, keepdim=True)
    cotangent = cotangent / torch.linalg.vector_norm(cotangent).clamp_min(
        torch.finfo(cotangent.dtype).tiny
    )
    scalar = torch.sum(output * cotangent)
    raw_gradient = torch.autograd.grad(
        scalar, parameters, allow_unused=True
    )
    gradient = tuple(
        torch.zeros_like(parameter) if value is None else value
        for parameter, value in zip(parameters, raw_gradient, strict=True)
    )
    gradient_norm = _tree_norm(gradient)
    parameter_norm = _tree_norm(parameters)
    if gradient_norm == 0.0 or parameter_norm == 0.0:
        raise ValueError("row-normal adjoint direction has zero norm")
    theta = tuple(
        value.detach() * (parameter_norm / gradient_norm) for value in gradient
    )
    return theta, _zeros_like(first), _zeros_like(second)


def _balanced_direction(
    parameters: TensorTree,
    first: TensorTree,
    second: TensorTree,
    layer: int,
) -> Direction:
    generator = torch.Generator(device=parameters[0].device)
    generator.manual_seed(int(DIRECTION_POLICY["balanced_seed"]) + layer)
    parameter_sign = _rademacher_like(parameters, generator)
    first_sign = _rademacher_like(first, generator)
    second_sign = _rademacher_like(second, generator)
    count = sum(value.numel() for value in parameters)
    parameter_norm = _tree_norm(parameters)
    first_norm = _tree_norm(first)
    scale = math.sqrt(3.0 * count)
    theta = tuple(value * (parameter_norm / scale) for value in parameter_sign)
    moment = tuple(value * (first_norm / scale) for value in first_sign)
    variance = tuple(
        sign * value.detach() / math.sqrt(3.0)
        for sign, value in zip(second_sign, second, strict=True)
    )
    return theta, moment, variance


def _make_directions(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    registered_rows: torch.Tensor,
    first_batch: torch.Tensor,
    layer: int,
) -> tuple[
    tuple[str, ...],
    TensorTree,
    TensorTree,
    TensorTree,
    dict[str, Direction],
    dict[str, Any],
]:
    names, parameters, first, second = _named_state(model, optimizer)
    setattr(model, "_observable_clip", float(checkpoint["config"]["clip"]))
    displacement, displacement_metadata = _optimizer_displacement(
        model, optimizer, parameters, first, second, first_batch
    )
    directions = {
        "row_adjoint": _row_adjoint_direction(
            model, registered_rows, layer, parameters, first, second
        ),
        "optimizer_displacement": displacement,
        "balanced_full_state": _balanced_direction(
            parameters, first, second, layer
        ),
    }
    metadata = {
        "direction_classes": list(DIRECTION_CLASSES),
        "directions": {
            name: _direction_summary(direction)
            for name, direction in directions.items()
        },
        "optimizer_displacement": displacement_metadata,
        "algorithm": {
            "row_adjoint_seed": int(DIRECTION_POLICY["row_adjoint_seed"]) + layer,
            "balanced_seed": int(DIRECTION_POLICY["balanced_seed"]) + layer,
            "second_moment_boundary_policy": DIRECTION_POLICY[
                "second_moment_boundary_policy"
            ],
        },
    }
    metadata["algorithm_sha256"] = digest_object(
        {
            "checkpoint_sha256": checkpoint["content_sha256"],
            "layer": layer,
            "algorithm": metadata["algorithm"],
            "direction_classes": metadata["direction_classes"],
        }
    )
    return names, parameters, first, second, directions, metadata


def _propagate_one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    parameters: TensorTree,
    directions: dict[str, Direction],
    rows: torch.Tensor,
    clip_value: float,
) -> tuple[dict[str, Direction], dict[str, float | int]]:
    """Apply the exact smooth-cell lifted AdamW derivative once."""

    loss, gradients = _raw_gradient(
        model, parameters, rows, create_graph=True
    )
    hessians: dict[str, TensorTree] = {}
    for index, direction_name in enumerate(DIRECTION_CLASSES):
        tangent = directions[direction_name][0]
        scalar = torch.zeros((), device=parameters[0].device)
        for gradient, local in zip(gradients, tangent, strict=True):
            scalar = scalar + torch.sum(gradient * local)
        hessians[direction_name] = tuple(
            value.detach()
            for value in torch.autograd.grad(
                scalar,
                parameters,
                retain_graph=index + 1 < len(DIRECTION_CLASSES),
                allow_unused=False,
            )
        )
        del scalar
    raw = tuple(value.detach() for value in gradients)
    del gradients
    group = optimizer.param_groups[0]
    if group.get("amsgrad", False) or group.get("maximize", False):
        raise ValueError("observable cocycle requires standard AdamW")
    beta1, beta2 = map(float, group["betas"])
    rate = float(group["lr"])
    epsilon = float(group["eps"])
    decay = float(group["weight_decay"])
    steps = {
        int(optimizer.state[parameter]["step"].item())
        for parameter in parameters
    }
    if len(steps) != 1:
        raise ValueError("optimizer clocks diverged during propagation")
    step_after = next(iter(steps)) + 1
    a1 = 1.0 - beta1**step_after
    a2 = 1.0 - beta2**step_after
    clipped = tuple(
        torch.clamp(value, -clip_value, clip_value) for value in raw
    )
    clip_masks = tuple(value.abs() < clip_value for value in raw)
    boundary_distance = min(
        float(torch.min(torch.abs(value.abs() - clip_value)).item())
        for value in raw
    )
    moment_next: list[torch.Tensor] = []
    variance_next: list[torch.Tensor] = []
    parameter_manual: list[torch.Tensor] = []
    next_directions: dict[str, list[list[torch.Tensor]]] = {
        name: [[], [], []] for name in DIRECTION_CLASSES
    }
    zero_variance_coordinates = 0
    zero_variance_active_tangents = 0
    for tensor_index, parameter in enumerate(parameters):
        state = optimizer.state[parameter]
        moment = state["exp_avg"]
        variance = state["exp_avg_sq"]
        gradient = clipped[tensor_index]
        local_moment = beta1 * moment + (1.0 - beta1) * gradient
        local_variance = beta2 * variance + (1.0 - beta2) * gradient.square()
        moment_hat = local_moment / a1
        variance_hat = local_variance / a2
        root = torch.sqrt(variance_hat)
        denominator = root + epsilon
        local_parameter = (1.0 - rate * decay) * parameter.detach() - rate * (
            moment_hat / denominator
        )
        moment_next.append(local_moment)
        variance_next.append(local_variance)
        parameter_manual.append(local_parameter)
        zero = root == 0.0
        zero_variance_coordinates += int(torch.count_nonzero(zero).item())
        for direction_name in DIRECTION_CLASSES:
            dtheta, dmoment, dvariance = directions[direction_name]
            hessian = hessians[direction_name][tensor_index]
            clipped_hessian = torch.where(
                clip_masks[tensor_index], hessian, torch.zeros_like(hessian)
            )
            next_moment = beta1 * dmoment[tensor_index] + (
                1.0 - beta1
            ) * clipped_hessian
            next_variance = beta2 * dvariance[tensor_index] + 2.0 * (
                1.0 - beta2
            ) * gradient * clipped_hessian
            active_zero = zero & (
                next_variance.abs()
                > float(NUMERICAL_POLICY["zero_second_moment_tangent_tolerance"])
            )
            zero_variance_active_tangents += int(
                torch.count_nonzero(active_zero).item()
            )
            denominator_action = torch.zeros_like(next_variance)
            positive = ~zero
            denominator_action[positive] = (
                moment_hat[positive]
                * (next_variance[positive] / a2)
                / (
                    2.0
                    * root[positive]
                    * denominator[positive].square()
                )
            )
            adaptive_action = (next_moment / a1) / denominator
            adaptive_action = adaptive_action - denominator_action
            next_parameter = (
                (1.0 - rate * decay) * dtheta[tensor_index]
                - rate * adaptive_action
            )
            next_directions[direction_name][0].append(next_parameter.detach())
            next_directions[direction_name][1].append(next_moment.detach())
            next_directions[direction_name][2].append(next_variance.detach())
    if zero_variance_active_tangents:
        raise ArithmeticError(
            "a propagated direction leaves a nonsmooth zero-variance face"
        )
    optimizer.zero_grad(set_to_none=True)
    for parameter, gradient in zip(parameters, clipped, strict=True):
        parameter.grad = gradient.clone()
    optimizer.step()
    theta_error = torch.zeros((), dtype=torch.float64, device=parameters[0].device)
    theta_scale = torch.zeros_like(theta_error)
    moment_error = torch.zeros_like(theta_error)
    moment_scale = torch.zeros_like(theta_error)
    variance_error = torch.zeros_like(theta_error)
    variance_scale = torch.zeros_like(theta_error)
    for parameter, expected_theta, expected_m, expected_v in zip(
        parameters,
        parameter_manual,
        moment_next,
        variance_next,
        strict=True,
    ):
        state = optimizer.state[parameter]
        theta_error += torch.sum((parameter.detach() - expected_theta).double().square())
        theta_scale += torch.sum(expected_theta.double().square())
        moment_error += torch.sum((state["exp_avg"] - expected_m).double().square())
        moment_scale += torch.sum(expected_m.double().square())
        variance_error += torch.sum(
            (state["exp_avg_sq"] - expected_v).double().square()
        )
        variance_scale += torch.sum(expected_v.double().square())

    def relative(error: torch.Tensor, scale: torch.Tensor) -> float:
        numerator = float(torch.sqrt(error).item())
        denominator_value = float(torch.sqrt(scale).item())
        return numerator / max(denominator_value, np.finfo(np.float64).tiny)

    output = {
        name: (
            tuple(blocks[0]),
            tuple(blocks[1]),
            tuple(blocks[2]),
        )
        for name, blocks in next_directions.items()
    }
    return output, {
        "loss": float(loss.item()),
        "optimizer_step_after": step_after,
        "minimum_clipping_boundary_distance": boundary_distance,
        "zero_variance_coordinates": zero_variance_coordinates,
        "parameter_replay_relative_residual": relative(theta_error, theta_scale),
        "first_moment_replay_relative_residual": relative(moment_error, moment_scale),
        "second_moment_replay_relative_residual": relative(
            variance_error, variance_scale
        ),
    }


def _propagate(
    checkpoint: dict[str, Any],
    device: torch.device,
    batches: list[torch.Tensor],
    registered_rows: torch.Tensor,
    layer: int,
    horizons: tuple[int, ...],
) -> tuple[
    dict[int, np.ndarray],
    dict[str, np.ndarray],
    dict[int, dict[str, np.ndarray]],
    dict[str, Direction],
    dict[str, Any],
    list[dict[str, float | int]],
]:
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    names, parameters, _first, _second, directions, metadata = _make_directions(
        checkpoint,
        model,
        optimizer,
        registered_rows,
        batches[0],
        layer,
    )
    initial_directions = {
        name: tuple(tuple(value.detach() for value in block) for block in direction)
        for name, direction in directions.items()
    }
    source_observation, source_actions = _observation_and_actions(
        model, registered_rows, layer, names, parameters, directions
    )
    observations: dict[int, np.ndarray] = {}
    actions: dict[int, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, float | int]] = []
    clip_value = float(checkpoint["config"]["clip"])
    for local_step, rows in enumerate(batches, start=1):
        directions, row = _propagate_one_step(
            model, optimizer, parameters, directions, rows, clip_value
        )
        diagnostics.append(row)
        if local_step in horizons:
            observation, local_actions = _observation_and_actions(
                model, registered_rows, layer, names, parameters, directions
            )
            observations[local_step] = observation
            actions[local_step] = local_actions
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        observations,
        source_actions,
        actions,
        initial_directions,
        metadata,
        diagnostics,
    )


def _apply_direction(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    direction: Direction,
    signed_amplitude: float,
) -> None:
    _names, parameters, first, second = _named_state(model, optimizer)
    for target, tangent in zip(parameters, direction[0], strict=True):
        target.data.add_(tangent, alpha=signed_amplitude)
    for target, tangent in zip(first, direction[1], strict=True):
        target.add_(tangent, alpha=signed_amplitude)
    for target, tangent in zip(second, direction[2], strict=True):
        target.add_(tangent, alpha=signed_amplitude)
        if torch.any(target < 0.0):
            raise ArithmeticError("signed branch left the AdamW variance cone")


def _native_step(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    rows: torch.Tensor,
    clip_value: float,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(model, rows)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
    optimizer.step()
    return float(loss.detach().item())


def _run_branch(
    checkpoint: dict[str, Any],
    device: torch.device,
    batches: list[torch.Tensor],
    registered_rows: torch.Tensor,
    layer: int,
    horizons: tuple[int, ...],
    direction: Direction | None = None,
    signed_amplitude: float = 0.0,
    transform: Callable[[torch.nn.Module, torch.optim.AdamW], None] | None = None,
) -> tuple[dict[int, np.ndarray], list[float]]:
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    if direction is not None:
        _apply_direction(model, optimizer, direction, signed_amplitude)
    if transform is not None:
        transform(model, optimizer)
    observations: dict[int, np.ndarray] = {}
    losses = []
    clip_value = float(checkpoint["config"]["clip"])
    with torch.enable_grad():
        for local_step, rows in enumerate(batches, start=1):
            losses.append(_native_step(model, optimizer, rows, clip_value))
            if local_step in horizons:
                with torch.no_grad():
                    observations[local_step] = (
                        _physical_observation(model, registered_rows, layer)
                        .to(dtype=torch.float32)
                        .cpu()
                        .numpy()
                    )
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return observations, losses


def _placebo_transform(layer: int) -> Callable[[torch.nn.Module, torch.optim.AdamW], None]:
    def apply(model: torch.nn.Module, _optimizer: torch.optim.AdamW) -> None:
        gate = dict(model.named_parameters())[_gate_parameter_name(layer)]
        with torch.no_grad():
            flat = gate.reshape(-1)
            flat[0] = torch.nextafter(
                flat[0], torch.full_like(flat[0], math.inf)
            )

    return apply


def _load_inputs(arguments: argparse.Namespace, horizon: int) -> tuple[
    torch.device,
    Path,
    dict[str, Any],
    dict[str, Any],
    torch.Tensor,
    list[torch.Tensor],
    np.ndarray,
]:
    device = _prepare_device(arguments.device)
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu"
    )
    registry = _load_registry(arguments.registry)
    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint and registered observation set disagree")
    registered_rows, _contexts = registered_tokens(
        registry, arguments.tokens, device
    )
    batches, indices = _ordered_batches(
        checkpoint, arguments.tokens, arguments.order, device, horizon
    )
    return (
        device,
        checkpoint_path,
        checkpoint,
        registry,
        registered_rows,
        batches,
        indices,
    )


def qualification_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    (
        device,
        checkpoint_path,
        checkpoint,
        _registry,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, 1)
    layer = int(arguments.layer)
    observations, source_actions, actions, directions, metadata, diagnostics = (
        _propagate(
            checkpoint,
            device,
            batches,
            registered_rows,
            layer,
            (1,),
        )
    )
    direction = directions["balanced_full_state"]
    amplitude = amplitudes("balanced_full_state")[1]
    plus, _plus_loss = _run_branch(
        checkpoint,
        device,
        batches,
        registered_rows,
        layer,
        (1,),
        direction,
        amplitude,
    )
    minus, _minus_loss = _run_branch(
        checkpoint,
        device,
        batches,
        registered_rows,
        layer,
        (1,),
        direction,
        -amplitude,
    )
    observed = 0.5 * (plus[1] - minus[1])
    predicted = amplitude * actions[1]["balanced_full_state"]
    relative = aggregate_norm(observed - predicted) / max(
        aggregate_norm(observed),
        aggregate_norm(predicted),
        1.0e-8,
    )
    replay_maximum = max(
        float(diagnostics[0][name])
        for name in (
            "parameter_replay_relative_residual",
            "first_moment_replay_relative_residual",
            "second_moment_replay_relative_residual",
        )
    )
    resources = _resource(device, started)
    peak = int(resources["resource_peak_gpu_reserved_bytes"].item())
    qualified = bool(
        np.isfinite(relative)
        and relative
        <= float(DECISION_POLICY["maximum_centered_response_relative_residual"])
        and replay_maximum
        <= float(NUMERICAL_POLICY["adamw_replay_relative_tolerance"])
        and peak
        <= int(RESOURCE_BUDGET["process_reserved_memory_cap_bytes"])
    )
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(QUALIFICATION_SCHEMA),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        "qualified": np.asarray(qualified),
        "training_chunk_indices": indices,
        "direction_algorithm_sha256": np.asarray(metadata["algorithm_sha256"]),
        "source_action": source_actions["balanced_full_state"],
        "endpoint_action": actions[1]["balanced_full_state"],
        "base_endpoint": observations[1],
        "predicted_response": predicted,
        "observed_response": observed,
        "relative_response_residual": np.asarray(relative),
        "maximum_adamw_replay_relative_residual": np.asarray(replay_maximum),
        **resources,
    }
    write_npz_atomic(arguments.output, **payload)
    if arguments.require_pass and not qualified:
        raise SystemExit("full-state development qualification did not pass")


def calibration_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    horizon_grid = tuple(int(value) for value in HORIZON_GRID)
    maximum = max(horizon_grid)
    (
        device,
        checkpoint_path,
        checkpoint,
        _registry,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, maximum)
    layer = int(arguments.layer)
    observations, source_actions, actions, directions, metadata, diagnostics = (
        _propagate(
            checkpoint,
            device,
            batches,
            registered_rows,
            layer,
            horizon_grid,
        )
    )
    placebo, _placebo_losses = _run_branch(
        checkpoint,
        device,
        batches,
        registered_rows,
        layer,
        horizon_grid,
        transform=_placebo_transform(layer),
    )
    direction_names = list(DIRECTION_CLASSES)
    amplitude_grid = np.asarray(
        [amplitudes(name) for name in direction_names], dtype=np.float64
    )
    predicted = np.empty(
        (len(direction_names), len(horizon_grid), 2) + observations[1].shape,
        dtype=np.float32,
    )
    observed = np.empty_like(predicted)
    for direction_index, direction_name in enumerate(direction_names):
        direction = directions[direction_name]
        for amplitude_index, amplitude in enumerate(
            amplitude_grid[direction_index]
        ):
            plus, _plus_losses = _run_branch(
                checkpoint,
                device,
                batches,
                registered_rows,
                layer,
                horizon_grid,
                direction,
                float(amplitude),
            )
            minus, _minus_losses = _run_branch(
                checkpoint,
                device,
                batches,
                registered_rows,
                layer,
                horizon_grid,
                direction,
                -float(amplitude),
            )
            for horizon_index, horizon in enumerate(horizon_grid):
                predicted[direction_index, horizon_index, amplitude_index] = (
                    float(amplitude) * actions[horizon][direction_name]
                )
                observed[direction_index, horizon_index, amplitude_index] = (
                    0.5 * (plus[horizon] - minus[horizon])
                )
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(CALIBRATION_SCHEMA),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        "training_chunk_indices": indices,
        "horizons": np.asarray(horizon_grid, dtype=np.int64),
        "direction_names": np.asarray(direction_names),
        "amplitudes": amplitude_grid,
        "direction_algorithm_sha256": np.asarray(metadata["algorithm_sha256"]),
        "source_actions": np.stack(
            [source_actions[name] for name in direction_names]
        ),
        "base_endpoints": np.stack(
            [observations[horizon] for horizon in horizon_grid]
        ),
        "endpoint_actions": np.stack(
            [
                np.stack([actions[horizon][name] for horizon in horizon_grid])
                for name in direction_names
            ]
        ),
        "predicted_responses": predicted,
        "observed_centered_responses": observed,
        "placebo_responses": np.stack(
            [placebo[horizon] - observations[horizon] for horizon in horizon_grid]
        ),
        "step_losses": np.asarray([row["loss"] for row in diagnostics]),
        "minimum_clipping_boundary_distance": np.asarray(
            min(row["minimum_clipping_boundary_distance"] for row in diagnostics)
        ),
        "zero_variance_coordinates": np.asarray(
            max(int(row["zero_variance_coordinates"]) for row in diagnostics),
            dtype=np.int64,
        ),
        "maximum_parameter_replay_relative_residual": np.asarray(
            max(row["parameter_replay_relative_residual"] for row in diagnostics)
        ),
        "maximum_first_moment_replay_relative_residual": np.asarray(
            max(row["first_moment_replay_relative_residual"] for row in diagnostics)
        ),
        "maximum_second_moment_replay_relative_residual": np.asarray(
            max(row["second_moment_replay_relative_residual"] for row in diagnostics)
        ),
        **_resource(device, started),
    }
    write_npz_atomic(arguments.output, **payload)


def _load_lock(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("unknown observable-cocycle construction lock")
    unsigned = dict(value)
    digest = unsigned.pop("lock_sha256", None)
    if digest != digest_object(unsigned):
        raise ValueError("construction lock digest does not replay")
    return value


def prediction_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    lock = _load_lock(arguments.lock)
    layer = int(arguments.layer)
    horizon = int(lock["layers"][str(layer)]["selected_horizon"])
    (
        device,
        checkpoint_path,
        checkpoint,
        _registry,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, horizon)
    observations, source_actions, actions, _directions, metadata, diagnostics = (
        _propagate(
            checkpoint,
            device,
            batches,
            registered_rows,
            layer,
            (horizon,),
        )
    )
    names = list(DIRECTION_CLASSES)
    amplitude_values = np.asarray(
        [amplitudes(name)[1] for name in names], dtype=np.float64
    )
    endpoint_actions = np.stack([actions[horizon][name] for name in names])
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(PREDICTION_SCHEMA),
        "stage": np.asarray("prediction"),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        "lock_sha256": np.asarray(str(lock["lock_sha256"])),
        "horizon": np.asarray(horizon, dtype=np.int64),
        "training_chunk_indices": indices,
        "direction_names": np.asarray(names),
        "amplitudes": amplitude_values,
        "direction_algorithm_sha256": np.asarray(metadata["algorithm_sha256"]),
        "source_actions": np.stack([source_actions[name] for name in names]),
        "base_endpoint": observations[horizon],
        "endpoint_actions": endpoint_actions,
        "predicted_centered_responses": (
            amplitude_values[:, None, None, None, None] * endpoint_actions
        ).astype(np.float32),
        "maximum_parameter_replay_relative_residual": np.asarray(
            max(row["parameter_replay_relative_residual"] for row in diagnostics)
        ),
        "maximum_first_moment_replay_relative_residual": np.asarray(
            max(row["first_moment_replay_relative_residual"] for row in diagnostics)
        ),
        "maximum_second_moment_replay_relative_residual": np.asarray(
            max(row["second_moment_replay_relative_residual"] for row in diagnostics)
        ),
        "minimum_clipping_boundary_distance": np.asarray(
            min(row["minimum_clipping_boundary_distance"] for row in diagnostics)
        ),
        "zero_variance_coordinates": np.asarray(
            max(int(row["zero_variance_coordinates"]) for row in diagnostics),
            dtype=np.int64,
        ),
        **_resource(device, started),
    }
    write_npz_atomic(arguments.output, **payload)


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name]) for name in archive.files}


def validation_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    prediction_path = Path(arguments.prediction).resolve()
    prediction = _load_npz(prediction_path)
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError("validation dependency is not a prediction record")
    if str(prediction["stage"].item()) != "prediction":
        raise ValueError("validation dependency has the wrong node role")
    horizon = int(prediction["horizon"].item())
    layer = int(arguments.layer)
    (
        device,
        checkpoint_path,
        checkpoint,
        _registry,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, horizon)
    if (
        str(prediction["checkpoint_sha256"].item())
        != sha256_path(checkpoint_path)
        or int(prediction["layer"].item()) != layer
        or not np.array_equal(prediction["training_chunk_indices"], indices)
    ):
        raise ValueError("prediction and held-out validation source disagree")
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    _names, _parameters, _first, _second, directions, metadata = _make_directions(
        checkpoint,
        model,
        optimizer,
        registered_rows,
        batches[0],
        layer,
    )
    del optimizer, model
    if metadata["algorithm_sha256"] != str(
        prediction["direction_algorithm_sha256"].item()
    ):
        raise ValueError("held-out direction construction drifted")
    names = [str(value) for value in prediction["direction_names"].tolist()]
    amplitude_values = np.asarray(prediction["amplitudes"], dtype=np.float64)
    observed = []
    for direction_name, amplitude in zip(names, amplitude_values, strict=True):
        plus, _plus_losses = _run_branch(
            checkpoint,
            device,
            batches,
            registered_rows,
            layer,
            (horizon,),
            directions[direction_name],
            float(amplitude),
        )
        minus, _minus_losses = _run_branch(
            checkpoint,
            device,
            batches,
            registered_rows,
            layer,
            (horizon,),
            directions[direction_name],
            -float(amplitude),
        )
        observed.append(0.5 * (plus[horizon] - minus[horizon]))
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(VALIDATION_SCHEMA),
        "stage": np.asarray("validation"),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        "prediction_sha256": np.asarray(sha256_path(prediction_path)),
        "lock_sha256": prediction["lock_sha256"],
        "horizon": np.asarray(horizon, dtype=np.int64),
        "training_chunk_indices": indices,
        "direction_names": np.asarray(names),
        "amplitudes": amplitude_values,
        "direction_algorithm_sha256": np.asarray(metadata["algorithm_sha256"]),
        "observed_centered_responses": np.stack(observed).astype(np.float32),
        **_resource(device, started),
    }
    write_npz_atomic(arguments.output, **payload)


def stratum_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    (
        device,
        checkpoint_path,
        checkpoint,
        _registry,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, 1)
    layer = int(arguments.layer)
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    gate = dict(model.named_parameters())[_gate_parameter_name(layer)]
    gate_state = optimizer.state[gate]
    with torch.no_grad():
        gate.zero_()
        gate_state["exp_avg"].zero_()
        source = (
            _physical_observation(model, registered_rows, layer)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
    loss = _native_step(
        model, optimizer, batches[0], float(checkpoint["config"]["clip"])
    )
    with torch.no_grad():
        endpoint = (
            _physical_observation(model, registered_rows, layer)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
        gate_after = gate.detach().double().cpu().numpy()
        gate_first_after = gate_state["exp_avg"].detach().double().cpu().numpy()
        gate_second_after = gate_state["exp_avg_sq"].detach().double().cpu().numpy()
    source_norm = aggregate_norm(source)
    endpoint_norm = aggregate_norm(endpoint)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(STRATUM_SCHEMA),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        "training_chunk_indices": indices,
        "source_row_quotient": source,
        "endpoint_row_quotient": endpoint,
        "source_row_quotient_norm": np.asarray(source_norm),
        "observed_force_norm": np.asarray(endpoint_norm),
        "gate_after": gate_after,
        "gate_first_moment_after": gate_first_after,
        "gate_second_moment_after": gate_second_after,
        "loss": np.asarray(loss),
        **_resource(device, started),
    }
    write_npz_atomic(arguments.output, **payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--tokens", required=True)
        command.add_argument("--order", required=True)
        command.add_argument("--registry", required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--layer", required=True, type=int, choices=(0, 1, 2))
        command.add_argument("--output", required=True)

    qualification = subparsers.add_parser("qualify")
    common(qualification)
    qualification.add_argument("--require-pass", action="store_true")
    qualification.set_defaults(function=qualification_record)

    calibration = subparsers.add_parser("calibrate")
    common(calibration)
    calibration.set_defaults(function=calibration_record)

    prediction = subparsers.add_parser("predict")
    common(prediction)
    prediction.add_argument("--lock", required=True)
    prediction.set_defaults(function=prediction_record)

    validation = subparsers.add_parser("validate")
    common(validation)
    validation.add_argument("--prediction", required=True)
    validation.set_defaults(function=validation_record)

    stratum = subparsers.add_parser("stratum")
    common(stratum)
    stratum.set_defaults(function=stratum_record)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
