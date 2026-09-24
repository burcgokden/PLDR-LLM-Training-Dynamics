#!/usr/bin/env python3
"""Capture compact exact-endpoint ledgers for gate-shape confirmation.

The deciding path records native source and successor endpoints, the exact
standard-AdamW gate multiplier and innovation, and frozen registered shapes.
No Taylor enclosure or source-state JVP remainder enters this ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from confirm.chronological_confirmation_specs import (  # noqa: E402
    ARCHITECTURE,
    OPTIMIZER,
    REGISTRY,
)
from confirm.capture_chronological_confirmation import (  # noqa: E402
    _load_registry,
    _registered_batch,
    _resource_record,
    _token_matrix,
    registered_shape,
    sha256_path_bytes,
)
from confirm.comprehensive_campaign import (  # noqa: E402
    campaign_spec,
    validate_checkpoint_campaign_binding,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    restore_rng_states,
    sha256_path,
)
from confirm.gate_shape_evidence import write_npz_atomic  # noqa: E402
from confirm.row_map_live import (  # noqa: E402
    _gate_parameter_name,
    _load_raw_checkpoint,
    _loss,
    _model_from_raw,
)


RECORD_SCHEMA = "pldr-comprehensive-endpoint-ledger-v1"


def _registered_shape_chunked(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    heads: torch.Tensor,
    layer: int,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size < 1:
        raise ValueError("context chunk size must be positive")
    blocks = []
    for start in range(0, len(tokens), chunk_size):
        stop = min(start + chunk_size, len(tokens))
        blocks.append(registered_shape(
            model, tokens[start:stop], heads[start:stop], layer))
    return torch.cat(blocks, dim=0)


def _load_data_order(
    path: str | Path,
    checkpoint: dict[str, Any],
    *,
    matrix_rows: int,
) -> np.ndarray:
    source = Path(path).resolve()
    if sha256_path(source) != checkpoint["data_state"].get("data_order_sha256"):
        raise ValueError("data-order file disagrees with the checkpoint digest")
    order = np.load(source, allow_pickle=False)
    if order.ndim != 1 or order.dtype.kind not in "iu":
        raise ValueError("data order must be a one-dimensional integer array")
    value = np.asarray(order, dtype=np.int64)
    if (
        value.size == 0
        or int(value.min()) < 0
        or int(value.max()) >= matrix_rows
        or np.unique(value).size != value.size
    ):
        raise ValueError("data order is out of range or contains duplicates")
    return value


def _gate_step_terms(
    gate: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, torch.Tensor, int, float, float]:
    """Return multiplier and innovation for the next standard-AdamW step."""

    group = next(
        candidate for candidate in optimizer.param_groups
        if any(parameter is gate for parameter in candidate["params"])
    )
    if group.get("amsgrad", False) or group.get("maximize", False):
        raise ValueError("capture requires standard AdamW")
    state = optimizer.state[gate]
    before = state["step"]
    before = int(before.item()) if torch.is_tensor(before) else int(before)
    after = before + 1
    beta1, beta2 = map(float, group["betas"])
    gradient = gate.grad
    if gradient is None:
        raise ValueError("the selected gate has no clipped gradient")
    first = beta1 * state["exp_avg"] + (1.0 - beta1) * gradient
    second = beta2 * state["exp_avg_sq"] + (1.0 - beta2) * gradient * gradient
    first_hat = first / (1.0 - beta1 ** after)
    second_hat = second / (1.0 - beta2 ** after)
    learning_rate = float(group["lr"])
    weight_decay = float(group.get("weight_decay", 0.0))
    multiplier = torch.full_like(gate, 1.0 - learning_rate * weight_decay)
    innovation = (
        learning_rate * first_hat
        / (torch.sqrt(second_hat) + float(group["eps"]))
    )
    return multiplier, innovation, after, learning_rate, weight_decay


def produce(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    spec = campaign_spec()
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = _load_raw_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu")
    validate_checkpoint_campaign_binding(checkpoint)
    registry = _load_registry(arguments.registry)
    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint and fixed measurement registry disagree")
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("token archive disagrees with the registry")
    role_seeds = spec["trajectories"][arguments.role]
    if int(arguments.seed) not in role_seeds:
        raise ValueError("seed is outside the prospective role")
    if int(checkpoint["config"]["seed"]) != int(arguments.seed):
        raise ValueError("checkpoint seed disagrees with the requested job")
    if int(checkpoint["config"].get("accum", 1)) != 1:
        raise ValueError("capture requires gradient accumulation one")
    if not checkpoint["config"].get("const_lr", False):
        raise ValueError("capture requires constant rate after warmup")
    if int(checkpoint["step"]) not in spec["trajectories"]["anchors"]:
        raise ValueError("checkpoint is outside the registered anchor grid")
    expected_updates = (
        spec["trajectories"]["development_block_updates"]
        if arguments.role == "development"
        else spec["trajectories"]["confirmation_block_updates"]
    )
    if arguments.updates != expected_updates:
        raise ValueError("block length disagrees with the prospective role")
    layer = int(arguments.layer)
    if layer not in spec["trajectories"]["layer_indices"]:
        raise ValueError("layer is outside the registered architecture")

    matrix = _token_matrix(arguments.tokens, registry["context_length"])
    order = _load_data_order(
        arguments.data_order, checkpoint, matrix_rows=len(matrix))
    cursor = int(checkpoint["data_offset_end"])
    batch_size = int(checkpoint["config"]["batch"])
    stop_cursor = cursor + arguments.updates * batch_size
    if stop_cursor > len(order):
        raise ValueError("requested continuation leaves the frozen data order")
    registered_chunks = {
        int(row["chunk_index"])
        for row in registry["construction"] + registry["validation"]
    }

    model = _model_from_raw(checkpoint, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(checkpoint["config"]["lr"]),
        betas=(float(OPTIMIZER["beta1"]), float(OPTIMIZER["beta2"])),
        eps=float(OPTIMIZER["epsilon"]),
        weight_decay=float(checkpoint["config"]["wd"]),
    )
    optimizer.load_state_dict(checkpoint["opt"])
    restore_rng_states(checkpoint["rng_states"])
    gate_name = _gate_parameter_name(layer)
    gate = dict(model.named_parameters())[gate_name]
    registered_tokens, registered_heads = _registered_batch(
        matrix, registry, device)

    model.train()
    shapes = [_registered_shape_chunked(
        model,
        registered_tokens,
        registered_heads,
        layer,
        arguments.context_chunk_size,
    ).detach().cpu().numpy().astype(np.float32, copy=False)]
    gates = [gate.detach().cpu().numpy().astype(np.float32, copy=True)]
    multipliers = []
    innovations = []
    learning_rates = []
    weight_decays = []
    optimizer_steps = []
    update_batch_sha256 = []
    update_chunk_indices = []
    replay_residuals = []

    for local_step in range(arguments.updates):
        start = cursor + local_step * batch_size
        indices = np.asarray(order[start:start + batch_size], dtype=np.int64)
        if registered_chunks.intersection(map(int, indices)):
            raise ValueError("continuation minibatch overlaps fixed probes")
        batch_array = matrix[indices].astype(np.int64, copy=True)
        update_batch_sha256.append(sha256_path_bytes(batch_array.tobytes()))
        update_chunk_indices.append(indices)
        rows = torch.from_numpy(batch_array).to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, rows)
        loss.backward()
        torch.nn.utils.clip_grad_value_(
            model.parameters(),
            clip_value=float(checkpoint["config"]["clip"]),
        )
        multiplier, innovation, opt_step, lr, wd = _gate_step_terms(
            gate, optimizer)
        expected = multiplier * gate.detach() - innovation
        multipliers.append(multiplier.detach().cpu().numpy().astype(
            np.float32, copy=False))
        innovations.append(innovation.detach().cpu().numpy().astype(
            np.float32, copy=False))
        learning_rates.append(lr)
        weight_decays.append(wd)
        optimizer_steps.append(opt_step)

        optimizer.step()
        scale = torch.maximum(
            torch.maximum(torch.abs(expected), torch.abs(gate.detach())),
            torch.ones_like(gate),
        )
        replay_residuals.append(float(torch.max(
            torch.abs(gate.detach() - expected) / scale).item()))
        gates.append(gate.detach().cpu().numpy().astype(np.float32, copy=True))
        shapes.append(_registered_shape_chunked(
            model,
            registered_tokens,
            registered_heads,
            layer,
            arguments.context_chunk_size,
        ).detach().cpu().numpy().astype(np.float32, copy=False))

    resources = _resource_record(device, started)
    payload: dict[str, np.ndarray] = {
        "schema_version": np.asarray(RECORD_SCHEMA),
        "campaign_id": np.asarray(spec["campaign_id"]),
        "campaign_spec_sha256": np.asarray(spec["spec_sha256"]),
        "role": np.asarray(arguments.role),
        "seed": np.asarray(arguments.seed, dtype=np.int64),
        "step_start": np.asarray(checkpoint["step"], dtype=np.int64),
        "layer": np.asarray(layer, dtype=np.int64),
        "normalized_rows": np.asarray(shapes, dtype=np.float32),
        "gate": np.asarray(gates, dtype=np.float32),
        "gate_multiplier": np.asarray(multipliers, dtype=np.float32),
        "gate_innovation": np.asarray(innovations, dtype=np.float32),
        "learning_rate": np.asarray(learning_rates, dtype=np.float64),
        "weight_decay": np.asarray(weight_decays, dtype=np.float64),
        "optimizer_step_after": np.asarray(optimizer_steps, dtype=np.int64),
        "update_chunk_indices": np.asarray(update_chunk_indices, dtype=np.int64),
        "update_batch_sha256": np.asarray(update_batch_sha256),
        "native_gate_replay_relative_residual": np.asarray(
            replay_residuals, dtype=np.float64),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "data_order_sha256": np.asarray(sha256_path(arguments.data_order)),
        "context_chunk_size": np.asarray(
            arguments.context_chunk_size, dtype=np.int64),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--data-order", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument(
        "--role",
        choices=["development", "construction", "heldout"],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-chunk-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    produce(parser.parse_args())


if __name__ == "__main__":
    main()
