#!/usr/bin/env python3
"""Stream a registered PLDR trajectory and preserve segment checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SCHEMA = "pldr-row-rg-confirmation-protocol-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-device", default="cuda:0")
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--reference-experiments", required=True)
    parser.add_argument("--dataset-shard", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dk", type=int, default=64)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-document-offset", type=int, required=True)
    parser.add_argument("--probe-document-offset", type=int, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    return parser.parse_args()


class ByteTokenStream:
    """Bounded-memory UTF-8 byte-plus-one stream with an incremental digest."""

    def __init__(self, dataset: Any, start: int, total_count: int) -> None:
        self.dataset = dataset
        self.start = start
        self.index = start
        self.total_count = total_count
        self.consumed = 0
        self.documents = 0
        self.buffer = bytearray()
        self.cursor = 0
        self.digest = hashlib.sha256()

    def _fill(self) -> None:
        if self.index >= len(self.dataset):
            raise ValueError("dataset shard ended before the registered stream")
        content = self.dataset[self.index]["content"]
        if not isinstance(content, str):
            raise ValueError(f"dataset row {self.index} has no string content")
        if self.cursor:
            del self.buffer[: self.cursor]
            self.cursor = 0
        self.buffer.extend(content.encode("utf-8", errors="strict"))
        self.buffer.append(10)
        self.index += 1
        self.documents += 1

    def take(self, count: int) -> list[int]:
        if count < 1 or self.consumed + count > self.total_count:
            raise ValueError("byte stream request exceeds the registered length")
        raw = bytearray()
        while len(raw) < count:
            if self.cursor == len(self.buffer):
                self._fill()
            available = min(count - len(raw), len(self.buffer) - self.cursor)
            raw.extend(self.buffer[self.cursor : self.cursor + available])
            self.cursor += available
        self.digest.update(raw)
        self.consumed += count
        return [value + 1 for value in raw]

    def skip(self, count: int, chunk_size: int = 1024 * 1024) -> None:
        remaining = count
        while remaining:
            take = min(remaining, chunk_size)
            self.take(take)
            remaining -= take

    def metadata(self) -> dict[str, Any]:
        if self.consumed != self.total_count:
            raise ValueError("training byte stream is incomplete")
        return {
            "first_document": self.start,
            "last_document_exclusive": self.index,
            "document_count": self.documents,
            "token_count": self.total_count,
            "byte_token_sha256": self.digest.hexdigest(),
            "streaming": True,
            "maximum_buffer_policy": "one current UTF-8 document",
        }


def _short_byte_tokens(dataset: Any, start: int, count: int) -> tuple[list[int], dict]:
    stream = ByteTokenStream(dataset, start, count)
    values = stream.take(count)
    return values, stream.metadata()


def _center(value: Any) -> Any:
    return value - value.mean(dim=0, keepdim=True)


def _gate_shape_residuals(
    torch: Any,
    before: Any,
    after: Any,
    gamma: Any,
    bias: Any,
    epsilon: float,
) -> dict[str, float]:
    """Return stable and deliberately ill-conditioned LayerNorm checks.

    Quotient-energy relative error is undefined at the collapsed face and is
    badly conditioned near it. The registered integrity gate therefore uses
    a full-map backward residual. The symmetric quotient-energy residual is
    retained as a diagnostic, not as a pass/fail quantity.
    """

    centered = before - before.mean(dim=-1, keepdim=True)
    shape = centered / torch.sqrt(
        epsilon + (centered * centered).mean(dim=-1, keepdim=True)
    )
    expected = shape.double() * gamma.double() + bias.double()
    observed = after.double()
    expected_quotient = _center(expected)
    observed_quotient = _center(observed)
    predicted_energy = float((expected_quotient * expected_quotient).sum())
    direct_energy = float((observed_quotient * observed_quotient).sum())
    energy_residual = abs(predicted_energy - direct_energy)
    tiny = float(torch.finfo(torch.float64).tiny)
    return {
        "map_backward_relative": float(
            torch.linalg.vector_norm(observed - expected)
        ) / max(float(torch.linalg.vector_norm(observed)), tiny),
        "quotient_energy_absolute": energy_residual,
        "quotient_energy_symmetric_relative": energy_residual
        / max(predicted_energy, direct_energy, tiny),
    }


def _summary(np: Any, values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _plga_parameters(layer: Any, head: int) -> tuple[Any, ...]:
    plga = layer.mha1.plgatt_layer
    return tuple(
        value[head].detach().double()
        for value in (plga.Wlst, plga.blst, plga.pwlst, plga.alst, plga.balst)
    )


def _plga_map(functional: Any, parameters: tuple[Any, ...], value: Any) -> Any:
    weight, bias, power, coupling, coupling_bias = parameters
    z = weight @ value + bias
    base = z * functional.silu(z) + 1e-9
    return coupling @ (base ** power) + coupling_bias


def _derivative_pair(
    torch: Any,
    functional: Any,
    parameters: tuple[Any, ...],
    point: Any,
):
    weight, bias, power, coupling, _ = parameters
    z = weight @ point + bias
    sigmoid = torch.sigmoid(z)
    silu = functional.silu(z)
    phi_prime = silu + z * (sigmoid + z * sigmoid * (1.0 - sigmoid))
    base = z * silu + 1e-9
    multiplier = power * (base ** (power - 1.0)) * phi_prime

    def action(direction: Any) -> Any:
        return _center(
            coupling @ (multiplier * (weight @ _center(direction)))
        )

    def adjoint(direction: Any) -> Any:
        return _center(
            weight.T @ (multiplier * (coupling.T @ _center(direction)))
        )

    return action, adjoint


def _operator_lower_bound(
    torch: Any,
    functional: Any,
    parameters: tuple[Any, ...],
    point: Any,
    iterations: int,
    starts: int,
) -> float:
    action, adjoint = _derivative_pair(torch, functional, parameters, point)
    flat = torch.arange(point.numel(), dtype=torch.float64, device=point.device)
    estimates = []
    for start in range(starts):
        direction = torch.sin((start + 1.61803398875) * (flat + 1.0)).reshape_as(point)
        direction = _center(direction)
        direction = direction / torch.linalg.vector_norm(direction)
        for _ in range(iterations):
            next_direction = adjoint(action(direction))
            norm = torch.linalg.vector_norm(next_direction)
            if float(norm) == 0.0:
                break
            direction = next_direction / norm
        estimates.append(float(torch.linalg.vector_norm(action(direction))))
    return max(estimates)


def _residual(torch: Any, value: Any) -> dict[str, float]:
    centered = _center(value)
    absolute = float(torch.linalg.vector_norm(centered))
    total = float(torch.linalg.vector_norm(value))
    return {
        "absolute": absolute,
        "relative": absolute / max(total, 1e-300),
    }


def _structural_residuals(torch: Any, parameters: tuple[Any, ...]) -> dict[str, Any]:
    weight, bias, power, coupling, coupling_bias = parameters
    ones = torch.ones((weight.shape[1], 1), dtype=weight.dtype, device=weight.device)
    return {
        "W_one": _residual(torch, weight @ ones),
        "bias": _residual(torch, bias),
        "power": _residual(torch, power),
        "coupling_one": _residual(torch, coupling @ ones),
        "coupling_bias": _residual(torch, coupling_bias),
    }


def _constrain_row_sum(torch: Any, matrix: Any) -> Any:
    ones = torch.ones((matrix.shape[1], 1), dtype=matrix.dtype, device=matrix.device)
    row_sum = matrix @ ones
    deviation = row_sum - row_sum.mean(dim=0, keepdim=True)
    return matrix - deviation @ ones.T / matrix.shape[1]


def _constrained_parameters(torch: Any, parameters: tuple[Any, ...]) -> tuple[Any, ...]:
    weight, bias, power, coupling, coupling_bias = parameters
    return (
        _constrain_row_sum(torch, weight),
        bias.mean(dim=0, keepdim=True).expand_as(bias).clone(),
        power.mean(dim=0, keepdim=True).expand_as(power).clone(),
        _constrain_row_sum(torch, coupling),
        coupling_bias.mean(dim=0, keepdim=True).expand_as(coupling_bias).clone(),
    )


def _plga_interventions(
    torch: Any,
    np: Any,
    functional: Any,
    model: Any,
    row_maps: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    rules = protocol["plga_analysis"]
    segment_grid = tuple(float(value) for value in rules["segment_grid"])
    iterations = int(rules["operator_power_iterations"])
    starts = int(rules["operator_start_count"])
    layers = model.decoder.dec_layers
    rows = []
    controls = []
    controlled_heads: set[tuple[int, int]] = set()
    for map_id, matrix in row_maps.items():
        fields = map_id.replace("c", "").replace("L", "").replace("H", "").split(".")
        context, layer_index, head = (int(value) for value in fields)
        value = matrix.detach().double()
        quotient = _center(value)
        constant = value.mean(dim=0, keepdim=True).expand_as(value)
        parameters = _plga_parameters(layers[layer_index], head)
        output = _plga_map(functional, parameters, value)
        constant_output = _plga_map(functional, parameters, constant)
        output_quotient = _center(output)
        defect = _center(constant_output)
        response = _center(output - constant_output)
        input_norm = float(torch.linalg.vector_norm(quotient))
        output_norm = float(torch.linalg.vector_norm(output_quotient))
        defect_norm = float(torch.linalg.vector_norm(defect))
        response_norm = float(torch.linalg.vector_norm(response))
        operator_estimates = [
            _operator_lower_bound(
                torch,
                functional,
                parameters,
                constant + fraction * quotient,
                iterations,
                starts,
            )
            for fraction in segment_grid
        ]
        values = [
            input_norm, output_norm, defect_norm, response_norm, *operator_estimates
        ]
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"non-finite PLGA diagnostic at {map_id}")
        residual = float(
            torch.linalg.vector_norm(output_quotient - response - defect)
        ) / max(output_norm, 1e-300)
        structural = _structural_residuals(torch, parameters)
        rows.append(
            {
                "map_id": map_id,
                "context": context,
                "layer": layer_index,
                "head": head,
                "input_quotient_norm": input_norm,
                "output_quotient_norm": output_norm,
                "constant_input_defect_norm": defect_norm,
                "realized_response_norm": response_norm,
                "defect_to_output_norm_ratio": defect_norm / max(output_norm, 1e-300),
                "realized_secant_gain": response_norm / max(input_norm, 1e-300),
                "sampled_operator_norm_lower_bound": max(operator_estimates),
                "segment_grid": list(segment_grid),
                "segment_operator_lower_bounds": operator_estimates,
                "structural_residuals": structural,
                "decomposition_relative_residual": residual,
            }
        )
        key = (layer_index, head)
        if key not in controlled_heads:
            controlled_heads.add(key)
            constrained = _constrained_parameters(torch, parameters)
            constrained_output = _plga_map(functional, constrained, constant)
            control_structural = _structural_residuals(torch, constrained)
            controls.append(
                {
                    "layer": layer_index,
                    "head": head,
                    "constant_input_defect_norm": float(
                        torch.linalg.vector_norm(_center(constrained_output))
                    ),
                    "maximum_structural_residual": max(
                        value["absolute"] for value in control_structural.values()
                    ),
                }
            )
    structural_names = ("W_one", "bias", "power", "coupling_one", "coupling_bias")
    return {
        "schema_version": "pldr-plga-checkpoint-diagnostics-v3",
        "map_count": len(rows),
        "rows": rows,
        "constant_input_defect_norm": _summary(
            np, [row["constant_input_defect_norm"] for row in rows]
        ),
        "defect_to_output_norm_ratio": _summary(
            np, [row["defect_to_output_norm_ratio"] for row in rows]
        ),
        "realized_secant_gain": _summary(
            np, [row["realized_secant_gain"] for row in rows]
        ),
        "sampled_operator_norm_lower_bound": _summary(
            np, [row["sampled_operator_norm_lower_bound"] for row in rows]
        ),
        "structural_residuals": {
            name: {
                "absolute": _summary(
                    np, [row["structural_residuals"][name]["absolute"] for row in rows]
                ),
                "relative": _summary(
                    np, [row["structural_residuals"][name]["relative"] for row in rows]
                ),
            }
            for name in structural_names
        },
        "maximum_decomposition_relative_residual": max(
            row["decomposition_relative_residual"] for row in rows
        ),
        "constrained_head_controls": {
            "count": len(controls),
            "maximum_constant_input_defect_norm": max(
                row["constant_input_defect_norm"] for row in controls
            ),
            "maximum_structural_residual": max(
                row["maximum_structural_residual"] for row in controls
            ),
            "rows": controls,
        },
    }


def _write_checkpoint(
    torch: Any,
    path: Path,
    *,
    step: int,
    model: Any,
    optimizer: Any,
    recorder: Any,
    losses: list[float],
    gate_shape_map_backward_maximum: float,
    gate_shape_energy_absolute_maximum: float,
    gate_shape_energy_symmetric_maximum: float,
    hook_identity_checks: int,
    plga_by_step: dict[int, Any],
    arguments: argparse.Namespace,
    protocol_digest: str,
) -> None:
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "schema_version": "pldr-training-checkpoint-v3",
        "step": step,
        "trajectory_id": arguments.trajectory_id,
        "seed": arguments.seed,
        "protocol_sha256": protocol_digest,
        "code_commit": arguments.code_commit,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "recorder": recorder.state_dict(),
        "losses": losses,
        "gate_shape_map_backward_maximum": gate_shape_map_backward_maximum,
        "gate_shape_energy_absolute_maximum": gate_shape_energy_absolute_maximum,
        "gate_shape_energy_symmetric_maximum": gate_shape_energy_symmetric_maximum,
        "hook_identity_checks": hook_identity_checks,
        "plga_by_step": plga_by_step,
        "python_random_state": random.getstate(),
        "numpy_random_state": __import__("numpy").random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    arguments = _parse()
    started_at = _utc_now()
    wall_start = time.perf_counter()
    output = Path(arguments.output)
    metrics_output = Path(arguments.metrics_output)
    metadata_output = Path(arguments.metadata_output)
    checkpoint_dir = Path(arguments.checkpoint_dir)
    protocol_path = Path(arguments.protocol).resolve()
    reference = Path(arguments.reference_experiments).resolve()
    shard = Path(arguments.dataset_shard).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_digest = _canonical_digest(protocol)
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unknown v3 protocol schema")
    if protocol.get("expected_updates") != arguments.steps:
        raise ValueError("producer horizon differs from protocol")
    if any(path.exists() for path in (output, metrics_output, metadata_output)):
        raise FileExistsError("producer refuses to overwrite final artifacts")
    if arguments.steps < 2 or arguments.dk % 2:
        raise ValueError("invalid step count or odd rotary head width")
    if arguments.context_length + 1 > arguments.max_sequence_length:
        raise ValueError("context length exceeds rotary registry")
    if min(arguments.train_document_offset, arguments.probe_document_offset) < 0:
        raise ValueError("document offsets must be nonnegative")
    model_source = reference / "pldr_model_v510.py"
    attention_source = reference / "power_law_attention_layer_v510.py"
    if not model_source.is_file() or not attention_source.is_file() or not shard.is_file():
        raise FileNotFoundError("reference source or dataset shard is missing")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from datasets import Dataset

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("producer requires exactly one visible CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(arguments.physical_device):
        raise RuntimeError("physical CUDA binding differs from the command")

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(reference))
    from row_rgmap.recorder import RowEnergyRecorder, row_centered_energy
    from pldr_model_v510 import PLDR_Model

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = Dataset.from_file(str(shard))
    sequence_width = arguments.context_length + 1
    tokens_per_update = arguments.batch_size * sequence_width
    train_count = arguments.steps * tokens_per_update
    train_stream = ByteTokenStream(
        dataset, arguments.train_document_offset, train_count
    )
    probe_count = arguments.contexts * sequence_width
    probe_values, probe_data = _short_byte_tokens(
        dataset, arguments.probe_document_offset, probe_count
    )

    device = torch.device(arguments.logical_device)
    d_model = arguments.heads * arguments.dk
    model = PLDR_Model(
        num_layers=arguments.layers,
        d_model=d_model,
        num_heads=arguments.heads,
        dff=int(2.7 * d_model),
        input_vocab_size=257,
        A_dff=2 * arguments.dk,
        num_reslayerA=2,
        num_denseA=2,
        max_seq_len=arguments.max_sequence_length,
        device=device,
    ).to(device=device, dtype=torch.float32)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-5,
        weight_decay=0.1,
    )
    mask = torch.triu(
        torch.ones(
            arguments.context_length,
            arguments.context_length,
            device=device,
            dtype=torch.float32,
        ),
        diagonal=1,
    )[None, None]
    probe = torch.tensor(probe_values, dtype=torch.long).reshape(
        arguments.contexts, sequence_width
    ).to(device)
    layernorms = [
        model.decoder.dec_layers[index].mha1.reslayerAs[-1].layernormA
        for index in range(arguments.layers)
    ]
    captured: dict[int, Any] = {}

    def make_hook(layer_index: int):
        def hook(module: Any, inputs: Any, result: Any) -> None:
            captured[layer_index] = (
                inputs[0].detach(),
                result.detach(),
                module.weight.detach(),
                module.bias.detach(),
                float(module.eps),
            )
        return hook

    map_ids = [
        f"c{context}.L{layer}.H{head}"
        for context in range(arguments.contexts)
        for layer in range(arguments.layers)
        for head in range(arguments.heads)
    ]
    recorder = RowEnergyRecorder(map_ids)
    losses: list[float] = []
    gate_shape_map_backward_maximum = 0.0
    gate_shape_energy_absolute_maximum = 0.0
    gate_shape_energy_symmetric_maximum = 0.0
    hook_identity_checks = 0
    plga_by_step: dict[int, Any] = {}
    checkpoint_steps = set(int(value) for value in protocol["plga_analysis"]["checkpoint_steps"])
    if 0 not in checkpoint_steps or arguments.steps not in checkpoint_steps:
        raise ValueError("protocol checkpoint registry must include initial and final states")

    def observe(step: int) -> tuple[Any, dict[str, Any]]:
        nonlocal gate_shape_map_backward_maximum
        nonlocal gate_shape_energy_absolute_maximum
        nonlocal gate_shape_energy_symmetric_maximum, hook_identity_checks
        model.eval()
        handles = [
            layernorm.register_forward_hook(make_hook(index))
            for index, layernorm in enumerate(layernorms)
        ]
        try:
            with torch.no_grad():
                _, _, _, cache = model([probe[:, :-1], mask])
        finally:
            for handle in handles:
                handle.remove()
        rows = {
            f"c{context}.L{layer_index}.H{head}": cache[layer_index][2][context, head]
            for context in range(arguments.contexts)
            for layer_index in range(arguments.layers)
            for head in range(arguments.heads)
        }
        if step in checkpoint_steps:
            for layer_index in range(arguments.layers):
                if not torch.equal(captured[layer_index][1], cache[layer_index][2]):
                    raise RuntimeError("forward hook does not match cached row map")
                before, after, gamma, bias, epsilon = captured[layer_index]
                for context in range(arguments.contexts):
                    for head in range(arguments.heads):
                        residuals = _gate_shape_residuals(
                            torch,
                            before[context, head],
                            after[context, head],
                            gamma,
                            bias,
                            epsilon,
                        )
                        gate_shape_map_backward_maximum = max(
                            gate_shape_map_backward_maximum,
                            residuals["map_backward_relative"],
                        )
                        gate_shape_energy_absolute_maximum = max(
                            gate_shape_energy_absolute_maximum,
                            residuals["quotient_energy_absolute"],
                        )
                        gate_shape_energy_symmetric_maximum = max(
                            gate_shape_energy_symmetric_maximum,
                            residuals["quotient_energy_symmetric_relative"],
                        )
            hook_identity_checks += 1
        recorder.record(step, rows)
        energies = np.asarray(
            [row_centered_energy(rows[name]) for name in map_ids], dtype=np.float64
        )
        return energies, rows

    start_step = 0
    if arguments.resume_from:
        resume_path = Path(arguments.resume_from)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if (
            checkpoint.get("schema_version") != "pldr-training-checkpoint-v3"
            or checkpoint.get("trajectory_id") != arguments.trajectory_id
            or checkpoint.get("seed") != arguments.seed
            or checkpoint.get("protocol_sha256") != protocol_digest
            or checkpoint.get("code_commit") != arguments.code_commit
        ):
            raise ValueError("resume checkpoint identity differs from this run")
        start_step = int(checkpoint["step"])
        if start_step >= arguments.steps:
            raise ValueError("resume checkpoint is already final")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        recorder.load_state_dict(checkpoint["recorder"])
        losses = [float(value) for value in checkpoint["losses"]]
        gate_shape_map_backward_maximum = float(
            checkpoint["gate_shape_map_backward_maximum"]
        )
        gate_shape_energy_absolute_maximum = float(
            checkpoint["gate_shape_energy_absolute_maximum"]
        )
        gate_shape_energy_symmetric_maximum = float(
            checkpoint["gate_shape_energy_symmetric_maximum"]
        )
        hook_identity_checks = int(checkpoint["hook_identity_checks"])
        plga_by_step = {int(key): value for key, value in checkpoint["plga_by_step"].items()}
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_random_state"]]
        )
        train_stream.skip(start_step * tokens_per_update)
    else:
        initial_energies, rows = observe(0)
        plga_by_step[0] = _plga_interventions(
            torch, np, functional, model, rows, protocol
        )
        _write_checkpoint(
            torch,
            checkpoint_dir / "checkpoint-step000000.pt",
            step=0,
            model=model,
            optimizer=optimizer,
            recorder=recorder,
            losses=losses,
            gate_shape_map_backward_maximum=gate_shape_map_backward_maximum,
            gate_shape_energy_absolute_maximum=gate_shape_energy_absolute_maximum,
            gate_shape_energy_symmetric_maximum=gate_shape_energy_symmetric_maximum,
            hook_identity_checks=hook_identity_checks,
            plga_by_step=plga_by_step,
            arguments=arguments,
            protocol_digest=protocol_digest,
        )

    model.train()
    for update in range(start_step + 1, arguments.steps + 1):
        values = train_stream.take(tokens_per_update)
        batch = torch.tensor(values, dtype=torch.long).reshape(
            arguments.batch_size, sequence_width
        ).to(device)
        logits, _, _, _ = model([batch[:, :-1], mask])
        loss = functional.cross_entropy(
            logits.reshape(-1, 257), batch[:, 1:].reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at update {update}")
        losses.append(float(loss.detach()))
        energies, rows = observe(update)
        if update in checkpoint_steps:
            plga_by_step[update] = _plga_interventions(
                torch, np, functional, model, rows, protocol
            )
            _write_checkpoint(
                torch,
                checkpoint_dir / f"checkpoint-step{update:06d}.pt",
                step=update,
                model=model,
                optimizer=optimizer,
                recorder=recorder,
                losses=losses,
                gate_shape_map_backward_maximum=gate_shape_map_backward_maximum,
                gate_shape_energy_absolute_maximum=gate_shape_energy_absolute_maximum,
                gate_shape_energy_symmetric_maximum=gate_shape_energy_symmetric_maximum,
                hook_identity_checks=hook_identity_checks,
                plga_by_step=plga_by_step,
                arguments=arguments,
                protocol_digest=protocol_digest,
            )
        model.train()
        if update == 1 or update % 2048 == 0 or update == arguments.steps:
            print(
                json.dumps(
                    {
                        "update": update,
                        "loss": losses[-1],
                        "energy_minimum": float(energies.min()),
                        "energy_median": float(np.median(energies)),
                        "energy_maximum": float(energies.max()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    tolerance = float(
        protocol["identity_tolerances"]["float32_gate_shape_map_backward_relative"]
    )
    if gate_shape_map_backward_maximum > tolerance:
        raise RuntimeError(
            "gate-shape map backward residual exceeds protocol tolerance: "
            f"{gate_shape_map_backward_maximum:.6e} > {tolerance:.6e}"
        )
    control_tolerance = float(protocol["identity_tolerances"]["constrained_plga_defect_norm"])
    if any(
        row["constrained_head_controls"]["maximum_constant_input_defect_norm"]
        > control_tolerance
        for row in plga_by_step.values()
    ):
        raise RuntimeError("constructed PLGA control exceeds protocol tolerance")

    output.parent.mkdir(parents=True, exist_ok=True)
    recorder.write_npz(output, trajectory_id=arguments.trajectory_id)
    np.savez_compressed(
        metrics_output,
        loss_steps=np.arange(1, arguments.steps + 1, dtype=np.int64),
        losses=np.asarray(losses, dtype=np.float64),
    )
    torch.cuda.synchronize()
    finished_at = _utc_now()
    wall_seconds = time.perf_counter() - wall_start
    with np.load(output, allow_pickle=False) as energy_record:
        all_energies = np.asarray(energy_record["energies"], dtype=np.float64)
        initial_energies = all_energies[0, 0]
    checkpoint_rows = [
        {
            "step": step,
            "path": os.path.relpath(
                (checkpoint_dir / f"checkpoint-step{step:06d}.pt").resolve(),
                metadata_output.parent.resolve(),
            ),
            "sha256": _sha256(checkpoint_dir / f"checkpoint-step{step:06d}.pt"),
            "size_bytes": (checkpoint_dir / f"checkpoint-step{step:06d}.pt").stat().st_size,
        }
        for step in sorted(checkpoint_steps)
    ]
    metadata = {
        "schema_version": "pldr-row-rg-energy-producer-metadata-v3",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_seconds": wall_seconds,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "logical_device": arguments.logical_device,
        "physical_device": arguments.physical_device,
        "cuda_visible_devices": visible,
        "visible_gpu_count": torch.cuda.device_count(),
        "trajectory_id": arguments.trajectory_id,
        "seed": arguments.seed,
        "code_commit": arguments.code_commit,
        "protocol": {
            "path": os.path.relpath(protocol_path, metadata_output.parent.resolve()),
            "canonical_sha256": protocol_digest,
            "file_sha256": _sha256(protocol_path),
        },
        "model": {
            "implementation": "PLDR_Model from pldr_model_v510.py",
            "layers": arguments.layers,
            "heads": arguments.heads,
            "dk": arguments.dk,
            "d_model": d_model,
            "contexts": arguments.contexts,
            "context_length": arguments.context_length,
            "vocabulary": 257,
            "parameter_count": parameter_count,
            "layernorm_epsilon": 1e-6,
            "plga_base_epsilon": 1e-9,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": arguments.learning_rate,
            "betas": [0.9, 0.95],
            "epsilon": 1e-5,
            "weight_decay": 0.1,
            "gradient_clip": "elementwise value 1.0",
            "schedule": "constant",
            "updates": arguments.steps,
        },
        "data": {
            "dataset": "tiiuae/falcon-refinedweb local read-only shard",
            "shard_path": str(shard),
            "shard_sha256": _sha256(shard),
            "tokens_per_update": tokens_per_update,
            "training": train_stream.metadata(),
            "probe": probe_data,
        },
        "sources": {
            "producer_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "reference_model_path": str(model_source),
            "reference_model_sha256": _sha256(model_source),
            "reference_attention_path": str(attention_source),
            "reference_attention_sha256": _sha256(attention_source),
            "recorder_path": "src/row_rgmap/recorder.py",
            "recorder_sha256": _sha256(ROOT / "src/row_rgmap/recorder.py"),
        },
        "capture": {
            "map_ids": map_ids,
            "states": arguments.steps + 1,
            "native_energy_dtype": "float64",
            "exact_zero_energy_count": int(np.sum(all_energies == 0.0)),
            "minimum_energy": float(all_energies.min()),
            "maximum_energy": float(all_energies.max()),
            "initial_energy_summary": _summary(np, initial_energies),
            "hook_identity_checks": hook_identity_checks,
            "gate_shape_maximum_map_backward_relative_residual": (
                gate_shape_map_backward_maximum
            ),
            "gate_shape_maximum_quotient_energy_absolute_residual": (
                gate_shape_energy_absolute_maximum
            ),
            "gate_shape_maximum_quotient_energy_symmetric_relative_residual": (
                gate_shape_energy_symmetric_maximum
            ),
        },
        "loss": {
            "initial_update_loss": losses[0],
            "final_update_loss": losses[-1],
            "minimum": min(losses),
            "median": float(np.median(losses)),
            "maximum": max(losses),
        },
        "plga_diagnostics_by_checkpoint": [
            {"step": step, "diagnostics": plga_by_step[step]}
            for step in sorted(plga_by_step)
        ],
        "checkpoints": checkpoint_rows,
        "peak_allocated_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "artifacts": {
            "energies": {
                "path": os.path.relpath(
                    output.resolve(), metadata_output.parent.resolve()
                ),
                "sha256": _sha256(output),
                "size_bytes": output.stat().st_size,
            },
            "metrics": {
                "path": os.path.relpath(
                    metrics_output.resolve(), metadata_output.parent.resolve()
                ),
                "sha256": _sha256(metrics_output),
                "size_bytes": metrics_output.stat().st_size,
            },
        },
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata = metadata_output.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_output)
    print(
        json.dumps(
            {
                "status": "complete",
                "trajectory_id": arguments.trajectory_id,
                "metadata": str(metadata_output),
                "energy_artifact": str(output),
                "wall_seconds": wall_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
