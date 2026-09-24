"""Masked AdamW coordinate and one-update row-response producer."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.func import functional_call, grad, jvp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import load_complete_checkpoint, sha256_path  # noqa: E402
from confirm.finite_increment_live import (  # noqa: E402
    optimizer_from_checkpoint,
    ordered_training_batch,
    registered_tokens,
)
from confirm.row_map_live import _model_from_raw  # noqa: E402
from confirm.source_restoring_live import load_registry  # noqa: E402
from confirm.source_resolved_energy import partition_adamw_coordinates  # noqa: E402
from confirm.source_resolved_specs import (  # noqa: E402
    CAMPAIGN_ID,
    MASK_RESPONSE_SCHEMA,
    REGISTRY,
)
from confirm.source_resolved_live import _schema  # noqa: E402
from confirm.strict_schema import load_json, validate  # noqa: E402
from train_run import create_masks, masked_loss_function  # noqa: E402


def _index_digest(indices: np.ndarray) -> str:
    value = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _centered_norms(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean(dim=-2, keepdim=True)
    return torch.sqrt(torch.sum(centered * centered, dim=(-2, -1))).reshape(-1)


def _direction_stream(
    random_directions: list[
        tuple[str, str, dict[str, torch.Tensor]]
    ],
    selected: list[int],
    energy_direction: Callable[[int], dict[str, torch.Tensor]],
) -> Iterator[tuple[str, str, dict[str, torch.Tensor]]]:
    """Yield full-state directions without retaining all map gradients."""

    while random_directions:
        yield random_directions.pop(0)
    for index in selected:
        yield (
            "energy-gradient",
            f"energy-map-{index}",
            energy_direction(index),
        )


def _rows_function(
    model: torch.nn.Module,
    buffers: dict[str, torch.Tensor],
    registered_rows: torch.Tensor,
):
    inputs = registered_rows[:, :-1]
    mask = create_masks(inputs, inputs.device)

    def rows(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        captured: list[torch.Tensor] = []
        handles = [
            model.decoder.dec_layers[layer].mha1.reslayerAs[-1]
            .layernormA.register_forward_hook(
                lambda _module, _inputs, output, captured=captured:
                captured.append(output)
            )
            for layer in range(3)
        ]
        try:
            functional_call(
                model,
                (parameters, buffers),
                ([inputs, mask],),
                strict=False,
            )
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != 3:
            raise RuntimeError("row-response hooks did not fire once per layer")
        result = torch.stack(captured, dim=1)
        if tuple(result.shape) != (len(inputs), 3, 4, 64, 64):
            raise ValueError(f"row-response map shape changed to {tuple(result.shape)}")
        return result

    return rows


def _optimizer_state(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    float,
    dict[str, float],
]:
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    first: dict[str, torch.Tensor] = {}
    second: dict[str, torch.Tensor] = {}
    steps = []
    for name, parameter in model.named_parameters():
        state = optimizer.state[parameter]
        first[name] = state["exp_avg"].detach().clone()
        second[name] = state["exp_avg_sq"].detach().clone()
        steps.append(float(state["step"]))
    if not steps or len(set(steps)) != 1:
        raise ValueError("optimizer parameter clocks disagree")
    group = optimizer.param_groups[0]
    constants = {
        "learning_rate": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
        "beta1": float(group["betas"][0]),
        "beta2": float(group["betas"][1]),
        "epsilon": float(group["eps"]),
    }
    return first, second, steps[0], constants


def produce_mask_response(
    binding_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    order_path: str | Path,
    device_name: str,
    *,
    protocol_directory: str | Path,
    map_indices: list[int] | None = None,
    random_direction_count: int = 8,
    random_seed: int = 45101,
) -> dict[str, Any]:
    """Partition the AdamW state before divisions and measure resolved actions."""

    binding_file = Path(binding_path).resolve()
    binding = load_json(binding_file)
    validate(
        binding,
        _schema(protocol_directory, "checkpoint_binding.schema.json"),
    )
    if not binding["technical_valid"]:
        raise ValueError("mask-response producer received an invalid binding")
    checkpoint_path = Path(binding["checkpoint_path"])
    if sha256_path(checkpoint_path) != binding["checkpoint_sha256"]:
        raise ValueError("bound checkpoint changed before response production")
    checkpoint = load_complete_checkpoint(checkpoint_path, map_location="cpu")
    registry = load_registry(registry_path)
    if checkpoint["measurement_registry"] != registry:
        raise ValueError("mask-response checkpoint and registry disagree")

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    model = _model_from_raw(checkpoint, device)
    parameters = {
        name: value.detach()
        for name, value in model.named_parameters()
    }
    buffers = {
        name: value.detach()
        for name, value in model.named_buffers()
    }
    first, second, clock, constants = _optimizer_state(model, checkpoint)
    update_rows, _update_chunks = ordered_training_batch(
        checkpoint, tokens_path, order_path, device
    )
    registered_rows, contexts = registered_tokens(
        registry, tokens_path, device
    )
    inputs = update_rows[:, :-1]
    targets = update_rows[:, 1:]
    mask = create_masks(inputs, inputs.device)

    def loss_function(local: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = functional_call(
            model,
            (local, buffers),
            ([inputs, mask],),
            strict=False,
        )[0]
        return masked_loss_function(targets, logits)

    model.train()
    raw_gradient = grad(loss_function)(parameters)
    clipped = {
        name: torch.clamp(value, -1.0, 1.0)
        for name, value in raw_gradient.items()
    }
    beta1 = constants["beta1"]
    beta2 = constants["beta2"]
    updated_first = {
        name: beta1 * first[name] + (1.0 - beta1) * clipped[name]
        for name in parameters
    }
    updated_second = {
        name: beta2 * second[name] + (1.0 - beta2) * clipped[name] ** 2
        for name in parameters
    }
    present_tokens = torch.unique(inputs).detach().cpu().tolist()

    unresolved: dict[str, torch.Tensor] = {}
    coordinate_records = []
    total_unresolved = 0
    for name in parameters:
        proof = torch.zeros_like(updated_second[name], dtype=torch.bool)
        if name == "decoder.embedding.weight":
            row_proof = torch.ones(
                proof.shape[0], dtype=torch.bool, device=proof.device
            )
            row_proof[torch.as_tensor(
                present_tokens, dtype=torch.long, device=proof.device
            )] = False
            proof = row_proof[:, None].expand_as(proof).clone()
        proof &= first[name] == 0
        proof &= updated_first[name] == 0
        proof &= clipped[name] == 0
        partition = partition_adamw_coordinates(
            updated_second[name].detach().cpu().numpy(),
            proof.detach().cpu().numpy(),
        )
        unresolved[name] = torch.as_tensor(
            partition["unresolved"], device=device
        )
        zero_indices = np.flatnonzero(
            updated_second[name].detach().cpu().numpy().reshape(-1) == 0.0
        ).astype(np.int64)
        inactive_indices = np.flatnonzero(
            partition["structurally_inactive"].reshape(-1)
        ).astype(np.int64)
        total_unresolved += partition["unresolved_count"]
        coordinate_records.append({
            "name": name,
            "coordinate_count": int(parameters[name].numel()),
            "active_count": partition["active_count"],
            "structurally_inactive_count": partition[
                "structurally_inactive_count"
            ],
            "unresolved_count": partition["unresolved_count"],
            "zero_second_moment_indices": zero_indices.tolist(),
            "structurally_inactive_indices": inactive_indices.tolist(),
            "zero_second_moment_indices_sha256": _index_digest(zero_indices),
            "structural_proof_sha256": _index_digest(inactive_indices),
        })

    a1 = 1.0 - beta1 ** (clock + 1.0)
    a2 = 1.0 - beta2 ** (clock + 1.0)
    learning_rate = constants["learning_rate"]
    weight_decay = constants["weight_decay"]
    epsilon = constants["epsilon"]

    def step_function(local: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        gradients = grad(loss_function)(local)
        result = {}
        for name in local:
            local_clipped = torch.clamp(gradients[name], -1.0, 1.0)
            first_next = (
                beta1 * first[name] + (1.0 - beta1) * local_clipped
            )
            second_next = (
                beta2 * second[name]
                + (1.0 - beta2) * local_clipped * local_clipped
            )
            active = second_next > 0.0
            safe_second = torch.where(
                active, second_next / a2, torch.ones_like(second_next)
            )
            adaptive = torch.where(
                active,
                (first_next / a1) / (torch.sqrt(safe_second) + epsilon),
                torch.zeros_like(first_next),
            )
            result[name] = (
                (1.0 - learning_rate * weight_decay) * local[name]
                - learning_rate * adaptive
            )
        return result

    rows_function = _rows_function(model, buffers, registered_rows)
    selected = list(range(REGISTRY["map_count"])) if map_indices is None else list(map_indices)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(index < 0 or index >= REGISTRY["map_count"] for index in selected)
        or random_direction_count < 0
    ):
        raise ValueError("response direction selection is invalid")

    def energy_direction(index: int) -> dict[str, torch.Tensor]:
        def energy(local: dict[str, torch.Tensor]) -> torch.Tensor:
            value = rows_function(local).reshape(-1, 64, 64)[index]
            centered = value - value.mean(dim=0, keepdim=True)
            return torch.sum(centered * centered)

        model.eval()
        result = grad(energy)(parameters)
        model.train()
        return result

    torch.manual_seed(random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(random_seed)
    random_directions: list[
        tuple[str, str, dict[str, torch.Tensor]]
    ] = []
    for index in range(random_direction_count):
        random_directions.append((
            "random-audit",
            f"random-{index}",
            {name: torch.randn_like(value) for name, value in parameters.items()},
        ))
    directions = _direction_stream(
        random_directions, selected, energy_direction
    )
    direction_count = random_direction_count + len(selected)

    response_records = []
    all_finite = True
    model.train()
    for direction_class, direction_id, raw_direction in directions:
        resolved = {
            name: torch.where(
                unresolved[name],
                torch.zeros_like(raw_direction[name]),
                raw_direction[name],
            )
            for name in parameters
        }
        norm = torch.sqrt(sum(
            torch.sum(value * value) for value in resolved.values()
        ))
        if not torch.isfinite(norm) or float(norm) == 0.0:
            raise ValueError(f"response direction {direction_id} is zero or nonfinite")
        direction = {name: value / norm for name, value in resolved.items()}
        endpoint, endpoint_direction = jvp(
            step_function, (parameters,), (direction,)
        )
        model.eval()
        _before, response_before = jvp(
            rows_function, (parameters,), (direction,)
        )
        _after, response_after = jvp(
            rows_function, (endpoint,), (endpoint_direction,)
        )
        model.train()
        before = _centered_norms(response_before)
        after = _centered_norms(response_after)
        endpoint_norm = torch.sqrt(sum(
            torch.sum(value[torch.isfinite(value)] ** 2)
            for value in endpoint_direction.values()
        ))
        finite_parameter = all(
            bool(torch.isfinite(value).all())
            for value in endpoint_direction.values()
        )
        for map_index in range(REGISTRY["map_count"]):
            denominator = float(before[map_index].detach().cpu())
            numerator = float(after[map_index].detach().cpu())
            finite = (
                finite_parameter
                and np.isfinite(denominator)
                and np.isfinite(numerator)
                and denominator > 0.0
            )
            row_response = numerator / denominator if finite else 0.0
            all_finite &= finite and np.isfinite(row_response)
            context_index, remainder = divmod(map_index, 12)
            layer, head = divmod(remainder, 4)
            response_records.append({
                "map_id": f"{contexts[context_index]['id']}-l{layer}-h{head}",
                "context": str(contexts[context_index]["id"]),
                "layer": int(layer),
                "head": int(head),
                "direction_id": direction_id,
                "direction_class": direction_class,
                "parameter_response": float(endpoint_norm.detach().cpu()),
                "row_response": float(row_response),
                "finite": bool(finite),
            })
        del (
            raw_direction, resolved, direction, endpoint,
            endpoint_direction, response_before, response_after,
            before, after, endpoint_norm,
        )

    expected_responses = direction_count * REGISTRY["map_count"]
    checks = {
        "coordinate_partition_exhaustive": sum(
            record["coordinate_count"] for record in coordinate_records
        ) == sum(value.numel() for value in parameters.values()),
        "structural_proofs_bound": all(
            len(record["structural_proof_sha256"]) == 64
            for record in coordinate_records
        ),
        "no_division_before_mask": True,
        "response_population_complete": len(response_records)
        == expected_responses,
        "all_responses_finite": bool(all_finite),
    }
    result = {
        "schema_version": MASK_RESPONSE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "checkpoint_binding_sha256": sha256_path(binding_file),
        "coordinate_tensors": coordinate_records,
        "responses": response_records,
        "unresolved_coordinate_count": int(total_unresolved),
        "smooth_full_state_eligible": total_unresolved == 0,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(result, _schema(protocol_directory, "mask_response.schema.json"))
    if not result["technical_valid"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"mask-response checks failed: {failed}")
    return result
