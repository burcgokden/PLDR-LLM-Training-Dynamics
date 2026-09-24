"""Matrix-free reduced response of a native 16-update lifted AdamW block."""

from __future__ import annotations

import gc
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.func import functional_call, grad, jvp

from confirm.confirmation_artifacts import load_complete_checkpoint, sha256_path
from confirm.finite_increment_live import (
    optimizer_from_checkpoint,
    ordered_training_batch,
    registered_tokens,
)
from confirm.row_map_live import _model_from_raw
from confirm.source_restoring_live import load_registry
from confirm.source_resolved_live import _schema
from confirm.source_resolved_response import _rows_function
from confirm.source_resolved_specs import (
    BLOCK_RESPONSE_SCHEMA,
    CAMPAIGN_ID,
    FROZEN_POLICY,
    REGISTRY,
)
from confirm.strict_schema import load_json, validate
from train_run import create_masks, masked_loss_function


State = tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]


def _state_tensors(state: State):
    for block in state:
        for name in sorted(block):
            yield block[name]


def _state_dot(left: State, right: State) -> float:
    return float(sum(
        float(torch.sum(a.double() * b.double()))
        for a, b in zip(
            _state_tensors(left), _state_tensors(right), strict=True
        )
    ))


def _state_norm(state: State) -> float:
    return math.sqrt(max(_state_dot(state, state), 0.0))


def _stable_vector_norm(value: torch.Tensor) -> float:
    """Accumulate diagnostic Euclidean norms without float32 overflow."""

    return float(torch.linalg.vector_norm(value.double()).detach().cpu())


def _actions_finite(
    records: list[dict[str, Any]], matrix: np.ndarray
) -> bool:
    """Return a schema-native boolean for action and matrix finiteness."""

    return bool(
        all(
            math.isfinite(value)
            for record in records
            for value in record.values()
            if isinstance(value, float)
        )
        and np.isfinite(matrix).all()
    )


def _state_clone(state: State) -> State:
    return tuple(
        {name: value.detach().clone() for name, value in block.items()}
        for block in state
    )  # type: ignore[return-value]


def _state_to(state: State, device: torch.device) -> State:
    return tuple(
        {
            name: value.detach().to(device=device, non_blocking=True)
            for name, value in block.items()
        }
        for block in state
    )  # type: ignore[return-value]


def _state_scale_in_place(state: State, factor: float) -> None:
    for value in _state_tensors(state):
        value.mul_(factor)


def _state_add_in_place(state: State, other: State, factor: float) -> None:
    for value, direction in zip(
        _state_tensors(state), _state_tensors(other), strict=True
    ):
        value.add_(direction, alpha=factor)


def _state_finite(state: State) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in _state_tensors(state))


def _normalize(state: State) -> float:
    norm = _state_norm(state)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("lifted AdamW direction is zero or nonfinite")
    _state_scale_in_place(state, 1.0 / norm)
    return norm


def _orthogonalize(
    state: State, basis: list[State]
) -> tuple[list[float], float]:
    coefficients = [0.0] * len(basis)
    for _pass in range(2):
        for index, direction in enumerate(basis):
            coefficient = _state_dot(direction, state)
            coefficients[index] += coefficient
            _state_add_in_place(state, direction, -coefficient)
    return coefficients, _state_norm(state)


def _optimizer_blocks(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    float,
    dict[str, float],
]:
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    first = {}
    second = {}
    clocks = []
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict) or not {
            "step", "exp_avg", "exp_avg_sq"
        }.issubset(state):
            raise ValueError(f"complete AdamW state is missing for {name}")
        first[name] = state["exp_avg"].detach()
        second[name] = state["exp_avg_sq"].detach()
        clocks.append(float(state["step"]))
    if not clocks or len(set(clocks)) != 1:
        raise ValueError("lifted AdamW parameter clocks disagree")
    group = optimizer.param_groups[0]
    constants = {
        "learning_rate": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
        "beta1": float(group["betas"][0]),
        "beta2": float(group["betas"][1]),
        "epsilon": float(group["eps"]),
        "clip": float(checkpoint["config"]["clip"]),
    }
    return first, second, clocks[0], constants


def _random_source_state(
    checkpoint: dict[str, Any], seed: int
) -> State:
    model = _model_from_raw(checkpoint, torch.device("cpu"))
    first, second, _clock, _constants = _optimizer_blocks(model, checkpoint)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    parameter_direction = {}
    first_direction = {}
    second_direction = {}
    for name, parameter in model.named_parameters():
        parameter_direction[name] = torch.randn(
            parameter.shape, dtype=parameter.dtype, generator=generator
        )
        active = second[name] > 0.0
        first_direction[name] = torch.where(
            active,
            torch.randn(
                first[name].shape,
                dtype=first[name].dtype,
                generator=generator,
            ),
            torch.zeros_like(first[name]),
        )
        second_direction[name] = torch.where(
            active,
            torch.randn(
                second[name].shape,
                dtype=second[name].dtype,
                generator=generator,
            ),
            torch.zeros_like(second[name]),
        )
    result: State = (
        parameter_direction,
        first_direction,
        second_direction,
    )
    _normalize(result)
    del model
    return result


def _transition_action(
    checkpoint: dict[str, Any],
    tangent: State,
    tokens_path: str | Path,
    order_path: str | Path,
    device: torch.device,
) -> State:
    model = _model_from_raw(checkpoint, device)
    model.train()
    parameters = {
        name: value.detach() for name, value in model.named_parameters()
    }
    buffers = {
        name: value.detach() for name, value in model.named_buffers()
    }
    first, second, clock, constants = _optimizer_blocks(model, checkpoint)
    update_rows, _chunks = ordered_training_batch(
        checkpoint, tokens_path, order_path, device
    )
    inputs = update_rows[:, :-1]
    targets = update_rows[:, 1:]
    mask = create_masks(inputs, device)

    def loss_function(local: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = functional_call(
            model, (local, buffers), ([inputs, mask],), strict=False
        )[0]
        return masked_loss_function(targets, logits)

    beta1 = constants["beta1"]
    beta2 = constants["beta2"]
    rate = constants["learning_rate"]
    decay = constants["weight_decay"]
    epsilon = constants["epsilon"]
    clipping = constants["clip"]
    correction1 = 1.0 - beta1 ** (clock + 1.0)
    correction2 = 1.0 - beta2 ** (clock + 1.0)

    def transition(
        local_parameters: dict[str, torch.Tensor],
        local_first: dict[str, torch.Tensor],
        local_second: dict[str, torch.Tensor],
    ) -> State:
        gradients = grad(loss_function)(local_parameters)
        clipped = {
            name: torch.clamp(value, -clipping, clipping)
            for name, value in gradients.items()
        }
        first_next = {
            name: beta1 * local_first[name] + (1.0 - beta1) * clipped[name]
            for name in local_parameters
        }
        second_next = {
            name: beta2 * local_second[name]
            + (1.0 - beta2) * clipped[name] * clipped[name]
            for name in local_parameters
        }
        parameter_next = {}
        for name in local_parameters:
            active = second_next[name] > 0.0
            safe_second = torch.where(
                active,
                second_next[name] / correction2,
                torch.ones_like(second_next[name]),
            )
            adaptive = torch.where(
                active,
                (first_next[name] / correction1)
                / (torch.sqrt(safe_second) + epsilon),
                torch.zeros_like(first_next[name]),
            )
            parameter_next[name] = (
                (1.0 - rate * decay) * local_parameters[name]
                - rate * adaptive
            )
        return parameter_next, first_next, second_next

    local_tangent = _state_to(tangent, device)
    _primal, output = jvp(
        transition,
        (parameters, first, second),
        local_tangent,
    )
    detached: State = tuple(
        {name: value.detach() for name, value in block.items()}
        for block in output
    )  # type: ignore[assignment]
    if not _state_finite(detached):
        raise FloatingPointError("lifted AdamW block action is nonfinite")
    del local_tangent, output, _primal, model, update_rows, inputs, targets
    return detached


def _apply_block(
    checkpoint_paths: list[Path],
    tangent: State,
    tokens_path: str | Path,
    order_path: str | Path,
    device: torch.device,
) -> tuple[State, dict[str, float]]:
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device.index)
    current = _state_to(tangent, device)
    for path in checkpoint_paths[:-1]:
        checkpoint = load_complete_checkpoint(path, map_location="cpu")
        successor = _transition_action(
            checkpoint, current, tokens_path, order_path, device
        )
        del current, checkpoint
        current = successor
        gc.collect()
    result = _state_to(current, torch.device("cpu"))
    del current
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = float(torch.cuda.max_memory_reserved(device.index))
        torch.cuda.empty_cache()
    else:
        peak = 0.0
    return result, {
        "wall_seconds": time.monotonic() - started,
        "peak_gpu_reserved_bytes": peak,
    }


def _observable_norm(
    checkpoint: dict[str, Any],
    parameter_tangent: dict[str, torch.Tensor],
    registered_rows: torch.Tensor,
    device: torch.device,
) -> float:
    model = _model_from_raw(checkpoint, device)
    parameters = {
        name: value.detach() for name, value in model.named_parameters()
    }
    buffers = {
        name: value.detach() for name, value in model.named_buffers()
    }
    row_function = _rows_function(model, buffers, registered_rows)
    direction = {
        name: parameter_tangent[name].to(device=device)
        for name in parameters
    }
    model.eval()
    _rows, action = jvp(row_function, (parameters,), (direction,))
    centered = action - action.mean(dim=-2, keepdim=True)
    norm = _stable_vector_norm(centered)
    del model, parameters, buffers, direction, _rows, action, centered
    return norm


def _binding_paths(
    binding_paths: list[str | Path],
    protocol_directory: str | Path,
) -> tuple[list[Path], list[dict[str, Any]], list[str]]:
    if len(binding_paths) != FROZEN_POLICY["block_length"] + 1:
        raise ValueError("block response needs all 17 endpoint bindings")
    schema = _schema(protocol_directory, "checkpoint_binding.schema.json")
    values = []
    paths = []
    digests = []
    for raw_path in binding_paths:
        binding_path = Path(raw_path).resolve()
        value = load_json(binding_path)
        validate(value, schema)
        checkpoint = Path(value["checkpoint_path"]).resolve()
        if (
            not value["technical_valid"]
            or sha256_path(checkpoint) != value["checkpoint_sha256"]
        ):
            raise ValueError("block response received a stale binding")
        values.append(value)
        paths.append(checkpoint)
        digests.append(sha256_path(binding_path))
    if any(
        right["step"] != left["step"] + 1
        for left, right in zip(values, values[1:])
    ):
        raise ValueError("block response bindings are not consecutive")
    if len({value["seed"] for value in values}) != 1:
        raise ValueError("block response bindings changed training seed")
    if len({value["order_sha256"] for value in values}) != 1:
        raise ValueError("block response bindings changed data order")
    return paths, values, digests


def _edge_closure(
    edge_paths: list[str | Path],
    binding_values: list[dict[str, Any]],
    protocol_directory: str | Path,
) -> list[str]:
    if len(edge_paths) != FROZEN_POLICY["block_length"]:
        raise ValueError("block response needs every native edge")
    schema = _schema(protocol_directory, "edge.schema.json")
    edges = []
    digests = []
    for path in edge_paths:
        edge = load_json(path)
        validate(edge, schema)
        if not edge["technical_valid"] or not edge["checks"]["native_link_closed"]:
            raise ValueError("block response received an open native edge")
        edges.append(edge)
        digests.append(sha256_path(path))
    edges.sort(key=lambda edge: edge["source_step"])
    expected = [value["step"] for value in binding_values]
    observed = [edges[0]["source_step"]] + [
        edge["endpoint_step"] for edge in edges
    ]
    if observed != expected:
        raise ValueError("block response edges and bindings disagree")
    return digests


def produce_block_response(
    binding_paths: list[str | Path],
    edge_paths: list[str | Path],
    mask_response_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    order_path: str | Path,
    device_name: str,
    *,
    protocol_directory: str | Path,
    random_seed: int = 45301,
) -> dict[str, Any]:
    """Run reduced Arnoldi actions and independent orthogonal audits."""

    checkpoint_paths, bindings, binding_digests = _binding_paths(
        binding_paths, protocol_directory
    )
    edge_digests = _edge_closure(
        edge_paths, bindings, protocol_directory
    )
    mask_response = load_json(mask_response_path)
    validate(
        mask_response,
        _schema(protocol_directory, "mask_response.schema.json"),
    )
    if not mask_response["technical_valid"]:
        raise ValueError("block response received an invalid coordinate mask")

    registry = load_registry(registry_path)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    source_checkpoint = load_complete_checkpoint(
        checkpoint_paths[0], map_location="cpu"
    )
    endpoint_checkpoint = load_complete_checkpoint(
        checkpoint_paths[-1], map_location="cpu"
    )
    if (
        source_checkpoint["measurement_registry"] != registry
        or endpoint_checkpoint["measurement_registry"] != registry
    ):
        raise ValueError("block response registry changed across endpoints")
    registered, _contexts = registered_tokens(
        registry, tokens_path, device
    )

    requested = int(FROZEN_POLICY["reduced_krylov_rank"])
    audit_count = int(FROZEN_POLICY["orthogonal_audit_directions"])
    hessenberg = np.zeros((requested + 1, requested), dtype=np.float64)
    basis = [_random_source_state(source_checkpoint, random_seed)]
    actions = []
    peak_gpu = 0.0
    total_action_seconds = 0.0
    breakdown = False
    for column in range(requested):
        if column >= len(basis):
            breakdown = True
            break
        direction = basis[column]
        source_row_norm = _observable_norm(
            source_checkpoint, direction[0], registered, device
        )
        image, resources = _apply_block(
            checkpoint_paths, direction, tokens_path, order_path, device
        )
        endpoint_row_norm = _observable_norm(
            endpoint_checkpoint, image[0], registered, device
        )
        state_gain = _state_norm(image)
        coefficients, residual = _orthogonalize(image, basis)
        for row, coefficient in enumerate(coefficients):
            hessenberg[row, column] = coefficient
        if column + 1 < hessenberg.shape[0]:
            hessenberg[column + 1, column] = residual
        actions.append({
            "direction_class": "krylov",
            "direction_index": column,
            "input_state_norm": 1.0,
            "output_state_norm": state_gain,
            "source_row_response_norm": source_row_norm,
            "endpoint_row_response_norm": endpoint_row_norm,
            "row_response_gain": (
                endpoint_row_norm / source_row_norm
                if source_row_norm > 0.0 else 0.0
            ),
            "arnoldi_residual_norm": residual,
        })
        peak_gpu = max(peak_gpu, resources["peak_gpu_reserved_bytes"])
        total_action_seconds += resources["wall_seconds"]
        if column + 1 < requested:
            if residual <= 1.0e-12:
                breakdown = True
                break
            _state_scale_in_place(image, 1.0 / residual)
            basis.append(image)

    achieved = len(actions)
    reduced = hessenberg[:achieved, :achieved]
    ritz = (
        np.abs(np.linalg.eigvals(reduced)).tolist() if achieved else []
    )
    singular = (
        np.linalg.svd(reduced, compute_uv=False).tolist() if achieved else []
    )

    audits = []
    for index in range(audit_count):
        direction = _random_source_state(
            source_checkpoint, random_seed + 1000 + index
        )
        _coefficients, input_residual = _orthogonalize(direction, basis)
        if input_residual <= 1.0e-12:
            raise RuntimeError("audit direction lies in the Krylov span")
        _state_scale_in_place(direction, 1.0 / input_residual)
        source_row_norm = _observable_norm(
            source_checkpoint, direction[0], registered, device
        )
        image, resources = _apply_block(
            checkpoint_paths, direction, tokens_path, order_path, device
        )
        endpoint_row_norm = _observable_norm(
            endpoint_checkpoint, image[0], registered, device
        )
        output_norm = _state_norm(image)
        _output_coefficients, omitted = _orthogonalize(image, basis)
        audits.append({
            "direction_class": "orthogonal-audit",
            "direction_index": index,
            "input_state_norm": 1.0,
            "output_state_norm": output_norm,
            "source_row_response_norm": source_row_norm,
            "endpoint_row_response_norm": endpoint_row_norm,
            "row_response_gain": (
                endpoint_row_norm / source_row_norm
                if source_row_norm > 0.0 else 0.0
            ),
            "omitted_state_response_norm": omitted,
        })
        peak_gpu = max(peak_gpu, resources["peak_gpu_reserved_bytes"])
        total_action_seconds += resources["wall_seconds"]

    orthogonality = 0.0
    for left in range(len(basis)):
        for right in range(left):
            orthogonality = max(
                orthogonality, abs(_state_dot(basis[left], basis[right]))
            )
    finite = _actions_finite(actions + audits, hessenberg)
    checks = {
        "binding_sequence_complete": len(bindings) == 17,
        "native_edge_clone_closure": len(edge_digests) == 16,
        "coordinate_mask_opened": True,
        "krylov_actions_finite": finite,
        "audit_population_complete": len(audits) == audit_count,
        "basis_orthogonality_within_tolerance": orthogonality <= 1.0e-5,
        "reduced_operator_labeled": True,
        "full_operator_not_claimed": True,
    }
    result = {
        "schema_version": BLOCK_RESPONSE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "checkpoint_binding_sha256": binding_digests,
        "edge_sha256": edge_digests,
        "mask_response_sha256": sha256_path(mask_response_path),
        "block_start": int(bindings[0]["step"]),
        "block_length": int(FROZEN_POLICY["block_length"]),
        "state_scope": "parameters-first-moments-second-moments",
        "reduced_operator_kind": "euclidean-arnoldi-diagnostic",
        "reduced_rank_requested": requested,
        "reduced_rank_achieved": achieved,
        "krylov_breakdown": breakdown,
        "hessenberg": hessenberg[: achieved + 1, :achieved].tolist(),
        "ritz_modulus": [float(value) for value in ritz],
        "reduced_singular_values": [float(value) for value in singular],
        "krylov_actions": actions,
        "orthogonal_audits": audits,
        "maximum_audit_omitted_response": float(max(
            record["omitted_state_response_norm"] for record in audits
        )),
        "maximum_audit_row_gain": float(max(
            record["row_response_gain"] for record in audits
        )),
        "smooth_full_state_eligible": bool(
            mask_response["smooth_full_state_eligible"]
        ),
        "unresolved_coordinate_count": int(
            mask_response["unresolved_coordinate_count"]
        ),
        "basis_maximum_abs_inner_product": float(orthogonality),
        "resources": {
            "action_wall_seconds": float(total_action_seconds),
            "peak_gpu_reserved_bytes": int(peak_gpu),
        },
        "full_operator_theorem_claimed": False,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(
        result,
        _schema(protocol_directory, "block_response.schema.json"),
    )
    return result
