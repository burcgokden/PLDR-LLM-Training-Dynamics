#!/usr/bin/env python3
"""Run isolated theorem-directed chronological intervention arms."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from confirm.capture_chronological_confirmation import (  # noqa: E402
    registered_shape,
)
from confirm.chronological_collapse import source_removal_response  # noqa: E402
from confirm.chronological_confirmation_specs import (  # noqa: E402
    CAMPAIGN_ID,
    OPTIMIZER,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    capture_rng_states,
    digest_object,
    nested_state_digest,
    restore_rng_states,
    sha256_path,
    validate_measurement_registry,
)
from confirm.gate_shape_evidence import write_npz_atomic  # noqa: E402
from confirm.row_map_live import (  # noqa: E402
    _gate_parameter_name,
    _load_raw_checkpoint,
    _loss,
    _model_from_raw,
)


RECORD_SCHEMA = "pldr-chronological-intervention-ledger-v1"
LOCK_SCHEMA = "pldr-chronological-construction-lock-v1"
ARMS = (
    "baseline",
    "adaptive-source-removal",
    "decay-removal",
    "shape-direction-removal",
)


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _load_registry(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    validate_measurement_registry(value, require_nonempty=True)
    if value.get("schema_version") != "pldr-chronological-probe-registry-v1":
        raise ValueError("interventions need the chronological registry")
    return value


def _load_lock(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
    ):
        raise ValueError("unknown chronological construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("chronological construction lock digest does not replay")
    if (
        value.get("constant_enlargement_after_lock") is not False
        or float(value.get("intervention_minimum_effect", 0.0)) <= 0.0
    ):
        raise ValueError("chronological intervention lock is invalid")
    return value


def _token_matrix(path: str | Path, context_length: int) -> np.ndarray:
    data = np.memmap(path, dtype=np.uint16, mode="r")
    count = len(data) // context_length
    return data[:count * context_length].reshape(count, context_length)


def _batch_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _optimizer(
    model: torch.nn.Module, checkpoint: dict[str, Any],
) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(checkpoint["config"]["lr"]),
        betas=(float(OPTIMIZER["beta1"]), float(OPTIMIZER["beta2"])),
        eps=float(OPTIMIZER["epsilon"]),
        weight_decay=float(checkpoint["config"]["wd"]),
    )
    optimizer.load_state_dict(checkpoint["opt"])
    return optimizer


def _step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: torch.Tensor,
    clip: float,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(model, rows)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip)


def _current_gate_source(
    gate: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, float, float, int]:
    state = optimizer.state[gate]
    group = next(
        candidate for candidate in optimizer.param_groups
        if any(parameter is gate for parameter in candidate["params"])
    )
    gradient = gate.grad
    if gradient is None:
        raise ValueError("selected gate gradient is absent")
    beta1, beta2 = map(float, group["betas"])
    step_before = state["step"]
    step_before = (
        int(step_before.item()) if torch.is_tensor(step_before)
        else int(step_before)
    )
    step_after = step_before + 1
    first_after = (
        beta1 * state["exp_avg"] + (1.0 - beta1) * gradient)
    second_after = (
        beta2 * state["exp_avg_sq"] + (1.0 - beta2) * gradient * gradient)
    first_source = (
        (1.0 - beta1) * gradient / (1.0 - beta1 ** step_after))
    second_hat = second_after / (1.0 - beta2 ** step_after)
    direction = first_source / (
        torch.sqrt(second_hat) + float(group["eps"]))
    return (
        direction.detach().clone(),
        float(group["lr"]),
        float(group.get("weight_decay", 0.0)),
        step_after,
    )


def _pair_state(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    head: int,
    layer: int,
    left: int,
    right: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    heads = torch.tensor([head], device=tokens.device, dtype=torch.long)
    shape = registered_shape(model, tokens, heads, layer)
    contrast = shape[left] - shape[right]
    gate_name = _gate_parameter_name(layer)
    gate = dict(model.named_parameters())[gate_name]
    energy = torch.sum((gate * contrast) ** 2)
    return (
        gate.detach().double().cpu().numpy(),
        contrast.detach().double().cpu().numpy(),
        float(energy.detach().double().cpu().item()),
    )


def _resources(device: torch.device, started: float) -> dict[str, Any]:
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
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = _load_raw_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu")
    registry = _load_registry(arguments.registry)
    lock = _load_lock(arguments.lock)
    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint and chronological registry disagree")
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("token archive disagrees with the registry")
    if int(checkpoint["config"]["seed"]) != int(arguments.seed):
        raise ValueError("checkpoint seed disagrees with the intervention job")
    if not checkpoint["config"].get("const_lr", False):
        raise ValueError("interventions require the constant-rate branch")
    if int(checkpoint["step"]) != REGISTRY["intervention_anchor_step"]:
        raise ValueError("intervention source is not the registered anchor")
    matching_trajectory = [
        row for row in TRAJECTORIES
        if row["role"] == "heldout"
        and int(row["seed"]) == int(arguments.seed)
        and int(row["data_offset_chunks"])
        == int(checkpoint["config"]["data_offset"])
    ]
    if len(matching_trajectory) != 1:
        raise ValueError("intervention checkpoint is outside the held-out grid")

    units = [
        row for row in registry["intervention_units"]
        if row["id"] == arguments.unit_id and row["seed"] == arguments.seed
    ]
    if len(units) != 1:
        raise ValueError("intervention unit is absent or duplicated")
    unit = units[0]
    ordinal = int(unit["ordinal"])
    layer = ordinal % 3
    context_index = ordinal % len(registry["validation"])
    head = context_index % 4
    left, right = REGISTRY["intervention_pair"]
    updates = int(unit["update_count"])
    future_start = int(unit["future_update_start"])
    if CAMPAIGN_ID == "pldr-causal-row-map-confirmation-v1":
        if updates != REGISTRY["intervention_steps"] or future_start != 0:
            raise ValueError("causal intervention mapping changed")
    else:
        if updates != REGISTRY["intervention_steps"]:
            raise ValueError("intervention update count changed")
        if future_start != ordinal * updates:
            raise ValueError("intervention future-update mapping changed")

    matrix = _token_matrix(arguments.tokens, registry["context_length"])
    cursor = int(checkpoint["data_offset_end"])
    batch_size = int(checkpoint["config"]["batch"])
    stop = cursor + (future_start + updates) * batch_size
    if stop > len(matrix):
        raise ValueError("intervention continuation leaves the token archive")
    context_chunk = int(registry["validation"][context_index]["chunk_index"])
    probe_tokens = torch.from_numpy(
        matrix[context_chunk:context_chunk + 1].astype(np.int64, copy=True)
    ).to(device)

    source_model = _model_from_raw(checkpoint, device)
    source_optimizer = _optimizer(source_model, checkpoint)
    restore_rng_states(checkpoint["rng_states"])
    clip = float(checkpoint["config"]["clip"])
    for local_step in range(future_start):
        begin = cursor + local_step * batch_size
        rows = torch.from_numpy(
            matrix[begin:begin + batch_size].astype(np.int64, copy=True)
        ).to(device)
        _step(source_model, source_optimizer, rows, clip)
        source_optimizer.step()

    source_rng = capture_rng_states()
    source_model_digest = nested_state_digest(source_model.state_dict())
    source_optimizer_digest = nested_state_digest(source_optimizer.state_dict())
    source_gate, source_contrast, source_energy = _pair_state(
        source_model, probe_tokens, head, layer, left, right)

    energy_histories = []
    batch_digests = []
    step_histories = []
    initial_model_digests = []
    initial_optimizer_digests = []
    final_model_digests = []
    final_optimizer_digests = []
    baseline_gate_successor = None
    adaptive_source_direction = None
    learning_rate = None
    predicted_response = None

    for arm in ARMS:
        model, optimizer = copy.deepcopy((source_model, source_optimizer))
        restore_rng_states(copy.deepcopy(source_rng))
        initial_model_digests.append(nested_state_digest(model.state_dict()))
        initial_optimizer_digests.append(
            nested_state_digest(optimizer.state_dict()))
        history = [source_energy]
        local_batch_digests = []
        local_steps = []
        for arm_step in range(updates):
            begin = cursor + (future_start + arm_step) * batch_size
            batch_array = matrix[begin:begin + batch_size].astype(
                np.int64, copy=True)
            local_batch_digests.append(_batch_digest(batch_array))
            rows = torch.from_numpy(batch_array).to(device)
            _step(model, optimizer, rows, clip)
            gate_name = _gate_parameter_name(layer)
            gate = dict(model.named_parameters())[gate_name]
            gate_before = gate.detach().clone()
            parameter_before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }
            source_direction, eta, decay, step_after = _current_gate_source(
                gate, optimizer)
            optimizer.step()
            if arm == "adaptive-source-removal":
                with torch.no_grad():
                    gate.add_(eta * source_direction)
            elif arm == "decay-removal":
                with torch.no_grad():
                    gate.add_(eta * decay * gate_before)
            elif arm == "shape-direction-removal":
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name != gate_name:
                            parameter.copy_(parameter_before[name])
            gate_after, _contrast_after, energy_after = _pair_state(
                model, probe_tokens, head, layer, left, right)
            history.append(energy_after)
            local_steps.append(step_after)
            if arm_step == 0 and arm == "baseline":
                baseline_gate_successor = gate_after
                adaptive_source_direction = (
                    source_direction.detach().double().cpu().numpy())
                learning_rate = eta
                response = source_removal_response(
                    baseline_gate_successor,
                    source_contrast[None, :],
                    adaptive_source_direction,
                    learning_rate=eta,
                )
                predicted_response = float(
                    response["predicted_energy_response"][0])
        energy_histories.append(history)
        batch_digests.append(local_batch_digests)
        step_histories.append(local_steps)
        final_model_digests.append(nested_state_digest(model.state_dict()))
        final_optimizer_digests.append(
            nested_state_digest(optimizer.state_dict()))

    if (
        baseline_gate_successor is None
        or adaptive_source_direction is None
        or learning_rate is None
        or predicted_response is None
    ):
        raise RuntimeError("baseline intervention prediction was not constructed")
    resources = _resources(device, started)
    payload = {
        "schema_version": np.asarray(RECORD_SCHEMA),
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "unit_id": np.asarray(arguments.unit_id),
        "ordinal": np.asarray(ordinal, dtype=np.int64),
        "rng_substream": np.asarray(
            int(unit["rng_substream"]), dtype=np.int64),
        "seed": np.asarray(arguments.seed, dtype=np.int64),
        "data_offset_chunks": np.asarray(
            int(checkpoint["config"]["data_offset"]), dtype=np.int64),
        "anchor": np.asarray(int(checkpoint["step"]), dtype=np.int64),
        "future_update_start": np.asarray(future_start, dtype=np.int64),
        "update_count": np.asarray(updates, dtype=np.int64),
        "layer": np.asarray(layer, dtype=np.int64),
        "context_index": np.asarray(context_index, dtype=np.int64),
        "head": np.asarray(head, dtype=np.int64),
        "row_left": np.asarray(left, dtype=np.int64),
        "row_right": np.asarray(right, dtype=np.int64),
        "arm_names": np.asarray(ARMS),
        "source_gate": source_gate,
        "source_contrast": source_contrast,
        "baseline_gate_successor": baseline_gate_successor,
        "adaptive_source_direction": adaptive_source_direction,
        "learning_rate": np.asarray(learning_rate),
        "predicted_source_removal_response": np.asarray(predicted_response),
        "pair_energy_history": np.asarray(energy_histories),
        "update_batch_sha256": np.asarray(batch_digests),
        "optimizer_step_after": np.asarray(step_histories, dtype=np.int64),
        "initial_model_sha256": np.asarray(initial_model_digests),
        "initial_optimizer_sha256": np.asarray(initial_optimizer_digests),
        "source_model_sha256": np.asarray(source_model_digest),
        "source_optimizer_sha256": np.asarray(source_optimizer_digest),
        "final_model_sha256": np.asarray(final_model_digests),
        "final_optimizer_sha256": np.asarray(final_optimizer_digests),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "lock_sha256": np.asarray(lock["lock_sha256"]),
        "intervention_minimum_effect": np.asarray(
            float(lock["intervention_minimum_effect"])),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in resources.items()
        },
    }
    write_npz_atomic(arguments.output, **payload)
    actual_response = (
        payload["pair_energy_history"][1, 1]
        - payload["pair_energy_history"][0, 1])
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "unit_id": arguments.unit_id,
        "predicted_response": predicted_response,
        "actual_response": float(actual_response),
        "peak_gpu_reserved_bytes": resources["peak_gpu_reserved_bytes"],
    }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    produce(parser.parse_args())


if __name__ == "__main__":
    main()
