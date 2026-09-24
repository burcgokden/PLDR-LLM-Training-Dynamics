#!/usr/bin/env python3
"""Qualify crossing-safe endpoint identities, optionally on a real checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.func import functional_call


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.capture_chronological_confirmation import (  # noqa: E402
    optimizer_displacements,
)
from confirm.finite_increment import (  # noqa: E402
    affine_increment,
    gate_shape_state_margin,
    per_map_diameters,
    representative_crossing_qualification,
    two_variable_power_increment,
    write_json_atomic,
)
from confirm.row_map_live import (  # noqa: E402
    _causal_mask,
    _load_raw_checkpoint,
    _loss,
    _model_from_raw,
    _next_training_batch,
    _token_chunks,
)
from train_run import PROBE_RESERVE_CHUNKS  # noqa: E402


SCHEMA_VERSION = "pldr-finite-increment-qualification-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_maps(
    model: torch.nn.Module,
    token_batches: list[torch.Tensor],
    parameters: dict[str, torch.Tensor] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture normalized shapes and physical rows per context, layer, head."""

    shapes: list[list[torch.Tensor]] = [[] for _ in range(3)]
    outputs: list[list[torch.Tensor]] = [[] for _ in range(3)]
    handles = []
    for layer in range(3):
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
                module.eps + torch.mean(centered * centered, dim=-1, keepdim=True)
            )
            shapes[layer_index].append(shape.detach().cpu())
            outputs[layer_index].append(output.detach().cpu())

        handles.append(layernorm.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        for tokens in token_batches:
            inputs = tokens[:, :-1]
            dtype = next(model.parameters()).dtype
            mask = _causal_mask(inputs.shape[1], inputs.device, dtype)
            with torch.no_grad():
                if parameters is None:
                    model([inputs, mask])
                else:
                    functional_call(
                        model, parameters, ([inputs, mask],), strict=False
                    )
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if any(len(layer) != len(token_batches) for layer in shapes + outputs):
        raise RuntimeError("a row-map hook did not fire once per context")
    shape = torch.stack([
        torch.cat(shapes[layer], dim=0) for layer in range(3)
    ], dim=1)
    physical = torch.stack([
        torch.cat(outputs[layer], dim=0) for layer in range(3)
    ], dim=1)
    expected = (len(token_batches), 3, 4, 64, 64)
    if tuple(shape.shape) != expected or tuple(physical.shape) != expected:
        raise ValueError(
            f"captured row maps have {tuple(shape.shape)} and "
            f"{tuple(physical.shape)}, expected {expected}"
        )
    return shape.double().numpy(), physical.double().numpy()


def _parameter(mapping: dict[str, torch.Tensor], layer: int, name: str) -> np.ndarray:
    key = f"decoder.dec_layers.{layer}.mha1.plgatt_layer.{name}"
    if key not in mapping:
        raise KeyError(f"missing PLGA parameter {key}")
    return mapping[key].detach().double().cpu().numpy()


def _gate(mapping: dict[str, torch.Tensor], layer: int, name: str) -> np.ndarray:
    key = (
        f"decoder.dec_layers.{layer}.mha1.reslayerAs.7."
        f"layernormA.{name}"
    )
    if key not in mapping:
        raise KeyError(f"missing final LayerNorm parameter {key}")
    return mapping[key].detach().double().cpu().numpy()


def _plga_report(
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
    g_increment = (
        np.matmul(a1[None], h1 - h0)
        + np.matmul((a1 - a0)[None], h0)
        + (ba1 - ba0)[None]
    )
    direct_g_increment = g1 - g0
    affine = affine_increment(source_rows, endpoint_rows, w0, w1, b0, b1)
    scale = max(float(np.max(np.abs(direct_g_increment))), 1.0)
    return {
        "layer": layer,
        "strict_crossing_count": powers["strict_crossing_count"],
        "touching_zero_count": powers["touching_zero_count"],
        "minimum_power_base": powers["minimum_base"],
        "maximum_abs_z_secant": powers["maximum_abs_z_secant"],
        "maximum_abs_p_secant": powers["maximum_abs_p_secant"],
        "maximum_abs_multiplied_contribution": (
            powers["maximum_abs_multiplied_contribution"]
        ),
        "power_base_first_max_abs_residual": (
            powers["base_first_max_abs_residual"]
        ),
        "power_exponent_first_max_abs_residual": (
            powers["exponent_first_max_abs_residual"]
        ),
        "preactivation_telescope_max_abs_residual": (
            affine["maximum_abs_residual"]
        ),
        "plga_output_telescope_max_abs_residual": float(
            np.max(np.abs(g_increment - direct_g_increment))
        ),
        "plga_output_scale": scale,
    }


def _real_qualification(
    checkpoint_path: Path,
    tokens_path: Path,
    device_name: str,
    context_count: int,
) -> dict[str, Any]:
    started = time.monotonic()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    checkpoint = _load_raw_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu"
    )
    model = _model_from_raw(checkpoint, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(checkpoint["config"]["lr"]),
        betas=(0.9, 0.95),
        eps=1.0e-5,
        weight_decay=float(checkpoint["config"]["wd"]),
    )
    optimizer.load_state_dict(checkpoint["opt"])
    next_batch = _next_training_batch(checkpoint, tokens_path, device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(model, next_batch)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    names, values, directions, successors = optimizer_displacements(model, optimizer)
    source_parameters = dict(zip(names, values, strict=True))
    endpoint_parameters = dict(zip(names, successors, strict=True))

    data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    total_chunks = len(data) // int(checkpoint["config"]["ctx"])
    reserve_start = total_chunks - PROBE_RESERVE_CHUNKS
    indices = list(range(reserve_start, reserve_start + context_count))
    token_batches = _token_chunks(
        tokens_path,
        indices,
        int(checkpoint["config"]["ctx"]),
        device,
    )
    shape0, rows0 = _capture_maps(model, token_batches)
    shape1, rows1 = _capture_maps(model, token_batches, endpoint_parameters)

    optimizer.step()
    actual_parameters = dict(model.named_parameters())
    maximum_parameter_residual = max(
        float(torch.max(torch.abs(actual_parameters[name].detach() - endpoint_parameters[name])).cpu())
        for name in names
    )
    actual_shape, actual_rows = _capture_maps(model, token_batches)
    maximum_row_residual = float(np.max(np.abs(actual_rows - rows1)))
    maximum_shape_residual = float(np.max(np.abs(actual_shape - shape1)))

    layer_reports = []
    identity_tolerance = 2048.0 * np.finfo(np.float64).eps
    for layer in range(3):
        gamma0 = _gate(source_parameters, layer, "weight")
        gamma1 = _gate(endpoint_parameters, layer, "weight")
        map_reports = []
        for context in range(context_count):
            for head in range(4):
                result = gate_shape_state_margin(
                    shape0[context, layer, head],
                    shape1[context, layer, head],
                    gamma0,
                    gamma1,
                    decay_multiplier=(
                        1.0
                        - float(checkpoint["config"]["lr"])
                        * float(checkpoint["config"]["wd"])
                    ),
                )
                map_reports.append({
                    "context": context,
                    "head": head,
                    "source_diameter_squared": result["source_diameter_squared"],
                    "endpoint_diameter_squared": result["endpoint_diameter_squared"],
                    "minimum_state_margin": result["minimum_state_margin"],
                    "alpha_star_zero_forcing": result["alpha_star_zero_forcing"],
                    "strict_contraction_certified": (
                        result["strict_contraction_certified"]
                    ),
                    "state_bound_covers_endpoint": (
                        result["state_bound_covers_endpoint"]
                    ),
                    "gate_balance_max_abs_residual": (
                        result["gate_balance_max_abs_residual"]
                    ),
                    "shape_bound_minimum_slack": (
                        result["shape_bound_minimum_slack"]
                    ),
                })
        diameter = per_map_diameters(rows1[:, layer].reshape(
            context_count * 4, 64, 64
        ))
        plga = _plga_report(
            layer,
            rows0[:, layer],
            rows1[:, layer],
            source_parameters,
            endpoint_parameters,
        )
        layer_scale = max(
            max(row["source_diameter_squared"] for row in map_reports), 1.0
        )
        layer_reports.append({
            "layer": layer,
            "map_count": len(map_reports),
            "strict_contraction_certificate_count": sum(
                row["strict_contraction_certified"] for row in map_reports
            ),
            "realized_contraction_count": sum(
                row["endpoint_diameter_squared"] < row["source_diameter_squared"]
                for row in map_reports
            ),
            "minimum_state_margin": min(
                row["minimum_state_margin"] for row in map_reports
            ),
            "maximum_gate_balance_residual": max(
                row["gate_balance_max_abs_residual"] for row in map_reports
            ),
            "minimum_shape_bound_slack": min(
                row["shape_bound_minimum_slack"] for row in map_reports
            ),
            "all_state_bounds_cover": all(
                row["state_bound_covers_endpoint"] for row in map_reports
            ),
            "aggregate_endpoint_diameter_squared": (
                diameter["aggregate_diameter_squared"]
            ),
            "cross_map_pairs_formed": diameter["cross_map_pairs_formed"],
            "identity_scale": layer_scale,
            "plga": plga,
            "maps": map_reports,
        })

    identity_passed = all(
        row["all_state_bounds_cover"]
        and row["maximum_gate_balance_residual"]
        <= identity_tolerance * row["identity_scale"]
        and row["minimum_shape_bound_slack"]
        >= -identity_tolerance * row["identity_scale"]
        and row["plga"]["power_base_first_max_abs_residual"]
        <= identity_tolerance * row["plga"]["plga_output_scale"]
        and row["plga"]["power_exponent_first_max_abs_residual"]
        <= identity_tolerance * row["plga"]["plga_output_scale"]
        and row["plga"]["preactivation_telescope_max_abs_residual"]
        <= identity_tolerance * row["plga"]["plga_output_scale"]
        and row["plga"]["plga_output_telescope_max_abs_residual"]
        <= identity_tolerance * row["plga"]["plga_output_scale"]
        for row in layer_reports
    )
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak_allocated = 0
        peak_reserved = 0
    return {
        "executed": True,
        "passed": bool(
            identity_passed
            and maximum_parameter_residual == 0.0
            and maximum_row_residual == 0.0
            and maximum_shape_residual == 0.0
        ),
        "device": str(device),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "tokens_sha256": _sha256(tokens_path),
        "context_indices": indices,
        "context_count": context_count,
        "map_count": context_count * 3 * 4,
        "source_loss": float(loss.detach().cpu()),
        "native_clone_parameter_max_abs_residual": maximum_parameter_residual,
        "native_clone_row_max_abs_residual": maximum_row_residual,
        "native_clone_shape_max_abs_residual": maximum_shape_residual,
        "identity_tolerance_factor_float64_epsilon": 2048.0,
        "layers": layer_reports,
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": peak_allocated,
            "peak_gpu_reserved_bytes": peak_reserved,
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ) * (1024 if sys.platform != "darwin" else 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real-checkpoint", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--contexts", type=int, default=2)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "reason": None,
        "e0": None,
        "real": None,
    }
    exit_code = 0
    try:
        e0 = representative_crossing_qualification()
        report["e0"] = e0
        real = None
        if arguments.real_checkpoint is not None:
            if arguments.tokens is None:
                raise ValueError("a real checkpoint qualification needs --tokens")
            if arguments.contexts < 1:
                raise ValueError("--contexts must be positive")
            real = _real_qualification(
                arguments.real_checkpoint,
                arguments.tokens,
                arguments.device,
                arguments.contexts,
            )
        report["real"] = real
        passed = bool(e0["passed"] and (real is None or real["passed"]))
        report["status"] = "pass" if passed else "fail"
        report["reason"] = None if passed else "a qualification gate failed"
        exit_code = 0 if passed or not arguments.require_pass else 1
    except Exception as error:
        report["status"] = "error"
        report["reason"] = f"{type(error).__name__}: {error}"
        exit_code = 1
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "status": report["status"],
        "reason": report["reason"],
    }, sort_keys=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
