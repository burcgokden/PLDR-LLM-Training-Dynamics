"""Bounded, checkpoint-bound gradient history for chronological AdamW replay.

The complete training checkpoint schema is intentionally closed. This module
therefore stores the signed, post-clipping gate-gradient history in a separate
sidecar whose digest binds it to exactly one checkpoint. The sidecar contains
only final row-map LayerNorm scales and is sufficient to prepend the next
gradient and reconstruct a length-K first-moment chronology without assuming
that gradients keep a fixed sign.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .confirmation_artifacts import sha256_path


SCHEMA_VERSION = "pldr-chronological-gradient-history-v1"
FINAL_GATE_SUFFIX = ".mha1.reslayerAs.7.layernormA.weight"


def sidecar_path(checkpoint_path: str | Path) -> Path:
    """Return the deterministic sidecar path for a checkpoint."""

    path = Path(checkpoint_path)
    return path.with_suffix(".chronological.pt")


def final_gate_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """Select every, and only, final row-map LayerNorm scale."""

    selected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.endswith(FINAL_GATE_SUFFIX)
    }
    if not selected:
        raise ValueError("the model exposes no final row-map gate parameters")
    expected = len(model.decoder.dec_layers)
    if len(selected) != expected:
        raise ValueError(
            "the final row-map gate selection is incomplete: "
            f"expected {expected}, found {len(selected)}"
        )
    return dict(sorted(selected.items()))


def empty_history(
    gate_parameters: dict[str, torch.nn.Parameter],
) -> dict[str, list[dict[str, Any]]]:
    return {name: [] for name in gate_parameters}


def record_clipped_gate_gradients(
    history: dict[str, list[dict[str, Any]]],
    gate_parameters: dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    history_length: int,
) -> None:
    """Append post-clipping gradients immediately before optimizer.step."""

    if history_length < 1:
        raise ValueError("history_length must be positive")
    if set(history) != set(gate_parameters):
        raise ValueError("history and gate parameter registries disagree")
    for name, parameter in gate_parameters.items():
        if parameter.grad is None:
            raise ValueError(f"final gate {name} has no gradient at step {global_step}")
        state = optimizer.state.get(parameter, {})
        first_before = state.get("exp_avg")
        if first_before is None:
            first_before = torch.zeros_like(parameter)
        record = {
            "global_step": int(global_step),
            "clipped_gradient": parameter.grad.detach().cpu().clone(),
            "first_moment_before": first_before.detach().cpu().clone(),
        }
        rows = history[name]
        rows.append(record)
        del rows[:-history_length]


def _optimizer_group(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
) -> dict[str, Any]:
    for group in optimizer.param_groups:
        if any(candidate is parameter for candidate in group["params"]):
            return group
    raise ValueError("a final gate is absent from the optimizer")


def _state_step(state: dict[str, Any]) -> int:
    value = state.get("step", 0)
    if torch.is_tensor(value):
        value = value.item()
    return int(value)


def build_sidecar(
    checkpoint_path: str | Path,
    checkpoint_step: int,
    history: dict[str, list[dict[str, Any]]],
    gate_parameters: dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    history_length: int,
) -> dict[str, Any]:
    """Build a CPU-only sidecar after the checkpointed optimizer update."""

    if history_length < 1:
        raise ValueError("history_length must be positive")
    checkpoint = Path(checkpoint_path)
    gates: dict[str, Any] = {}
    for name, parameter in gate_parameters.items():
        rows = history[name]
        state = optimizer.state.get(parameter, {})
        first_after = state.get("exp_avg")
        second_after = state.get("exp_avg_sq")
        if first_after is None or second_after is None:
            raise ValueError(f"AdamW state is incomplete for final gate {name}")
        group = _optimizer_group(optimizer, parameter)
        betas = group.get("betas")
        if betas is None:
            raise ValueError("chronological sidecars require AdamW betas")
        newest = list(reversed(rows))
        gradients = torch.stack(
            [row["clipped_gradient"] for row in newest], dim=0
        )
        moments_before = torch.stack(
            [row["first_moment_before"] for row in newest], dim=0
        )
        steps = torch.tensor(
            [row["global_step"] for row in newest], dtype=torch.int64
        )
        ready = len(rows) >= history_length
        previous_count = max(history_length - 1, 0)
        previous = gradients[:previous_count]
        if history_length == 1:
            next_tail = first_after.detach().cpu().clone()
        elif ready:
            oldest = rows[0]
            beta1 = float(betas[0])
            next_tail = (
                beta1 * oldest["first_moment_before"]
                + (1.0 - beta1) * oldest["clipped_gradient"]
            )
        else:
            next_tail = torch.empty(0, dtype=first_after.dtype)
        gates[name] = {
            "optimizer_step": _state_step(state),
            "beta1": float(betas[0]),
            "beta2": float(betas[1]),
            "epsilon": float(group["eps"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "steps_newest_first": steps,
            "clipped_gradients_newest_first": gradients,
            "first_moments_before_newest_first": moments_before,
            "first_moment_after_checkpoint": first_after.detach().cpu().clone(),
            "second_moment_after_checkpoint": second_after.detach().cpu().clone(),
            "previous_gradients_for_next_newest_first": previous,
            "first_moment_tail_for_next": next_tail,
            "ready_for_next_update": ready,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": sha256_path(checkpoint),
        "checkpoint_step": int(checkpoint_step),
        "history_length": int(history_length),
        "gate_parameter_names": list(gates),
        "gates": gates,
    }


def save_sidecar(
    checkpoint_path: str | Path,
    checkpoint_step: int,
    history: dict[str, list[dict[str, Any]]],
    gate_parameters: dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    *,
    history_length: int,
) -> Path:
    output = sidecar_path(checkpoint_path)
    payload = build_sidecar(
        checkpoint_path,
        checkpoint_step,
        history,
        gate_parameters,
        optimizer,
        history_length=history_length,
    )
    torch.save(payload, output)
    return output


def restore_history(
    checkpoint_path: str | Path,
    gate_parameters: dict[str, torch.nn.Parameter],
    *,
    checkpoint_step: int,
    history_length: int,
) -> dict[str, list[dict[str, Any]]]:
    """Restore a bound sidecar when a captured run is continued."""

    history = empty_history(gate_parameters)
    path = sidecar_path(checkpoint_path)
    if not path.is_file():
        return history
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("chronological sidecar schema mismatch")
    if payload.get("checkpoint_sha256") != sha256_path(checkpoint_path):
        raise ValueError("chronological sidecar checkpoint digest mismatch")
    if int(payload.get("checkpoint_step", -1)) != int(checkpoint_step):
        raise ValueError("chronological sidecar step mismatch")
    if int(payload.get("history_length", -1)) != int(history_length):
        raise ValueError("chronological sidecar history length mismatch")
    if payload.get("gate_parameter_names") != list(gate_parameters):
        raise ValueError("chronological sidecar gate registry mismatch")
    for name in gate_parameters:
        gate = payload["gates"][name]
        steps = gate["steps_newest_first"]
        gradients = gate["clipped_gradients_newest_first"]
        moments = gate["first_moments_before_newest_first"]
        if not (len(steps) == len(gradients) == len(moments)):
            raise ValueError("chronological sidecar tensor lengths disagree")
        for index in reversed(range(len(steps))):
            history[name].append({
                "global_step": int(steps[index]),
                "clipped_gradient": gradients[index].clone(),
                "first_moment_before": moments[index].clone(),
            })
    return history
