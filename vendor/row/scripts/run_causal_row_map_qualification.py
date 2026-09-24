#!/usr/bin/env python3
"""Qualify the post-norm chain and causal certificate on the real classes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.func import jacrev


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from pldr_model_v510 import ResLayerA  # noqa: E402
from confirm.causal_row_map import stream_registry_prediction  # noqa: E402
from confirm.composite_plga import plga_transfer  # noqa: E402
from confirm.postnorm_geometry import (  # noqa: E402
    affine_layernorm_jacobian,
    backward_path_metrics,
    layernorm_jacobian,
    verify_path_metric,
)


SCHEMA_VERSION = "pldr-causal-row-map-qualification-v1"


class ImplementedRowMap(nn.Module):
    """The implemented pre-norm plus eight post-norm metric learner."""

    def __init__(self, width: int, hidden: int, device: torch.device):
        super().__init__()
        self.pre = nn.LayerNorm(width, eps=1e-6, device=device)
        self.units = nn.ModuleList([
            ResLayerA(
                depth=width,
                A_dff=hidden,
                num_denseA=2,
                device=device,
            )
            for _ in range(8)
        ])

    @staticmethod
    def residual_branch(unit: ResLayerA, value: torch.Tensor) -> torch.Tensor:
        transformed = value
        for block in unit.denseAs:
            transformed = block(transformed)
        return transformed

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        state = self.pre(value)
        for unit in self.units:
            branch = self.residual_branch(unit, state)
            state = unit.layernormA(state + branch)
        return state

    def final_shape(self, value: torch.Tensor) -> torch.Tensor:
        state = self.pre(value)
        for index, unit in enumerate(self.units):
            branch = self.residual_branch(unit, state)
            layernorm_input = state + branch
            if index == len(self.units) - 1:
                centered = layernorm_input - torch.mean(layernorm_input)
                return centered / torch.sqrt(
                    unit.layernormA.eps + torch.mean(centered * centered))
            state = unit.layernormA(layernorm_input)
        raise AssertionError("the row map must contain a final unit")


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.linalg.norm(left)),
        float(np.linalg.norm(right)),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(left - right)) / scale


def _analytic_chains(
    model: ImplementedRowMap, source: torch.Tensor,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, float]],
    list[np.ndarray],
    list[np.ndarray],
]:
    value = source.detach()
    width = value.numel()
    L0 = affine_layernorm_jacobian(
        value.cpu().numpy(),
        model.pre.weight.detach().cpu().numpy(),
        model.pre.eps,
    )
    full = L0
    full_factors = [L0]
    shape_factors = [L0]
    state = model.pre(value)
    stage_rows = []
    shape_chain = None
    for index, unit in enumerate(model.units):
        branch_function = lambda local: model.residual_branch(unit, local)
        branch_jacobian = jacrev(branch_function)(state)
        residual = np.eye(width) + branch_jacobian.detach().cpu().numpy()
        branch = branch_function(state)
        layernorm_input = state + branch
        gamma = unit.layernormA.weight.detach().cpu().numpy()
        affine = affine_layernorm_jacobian(
            layernorm_input.detach().cpu().numpy(),
            gamma,
            unit.layernormA.eps,
        )
        normalized = layernorm_jacobian(
            layernorm_input.detach().cpu().numpy(), unit.layernormA.eps)
        incoming = full
        full = affine @ residual @ incoming
        full_factors.extend((residual, affine))
        if index == len(model.units) - 1:
            shape_chain = normalized @ residual @ incoming
            shape_factors.extend((residual, normalized))
        else:
            shape_factors.extend((residual, affine))
        stage_rows.append({
            "unit": index,
            "residual_operator_norm": float(np.linalg.norm(residual, 2)),
            "affine_layernorm_operator_norm": float(np.linalg.norm(affine, 2)),
            "composite_prefix_operator_norm": float(np.linalg.norm(full, 2)),
        })
        state = unit.layernormA(layernorm_input)
    if shape_chain is None:
        raise AssertionError("shape chain was not constructed")
    return full, shape_chain, stage_rows, full_factors, shape_factors


def qualify(arguments: argparse.Namespace) -> dict[str, object]:
    started = time.monotonic()
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(int(arguments.seed))
    model = ImplementedRowMap(
        int(arguments.width), int(arguments.hidden), device).to(
            dtype=torch.float64)
    model.eval()
    source = torch.randn(
        int(arguments.width), dtype=torch.float64, device=device)
    automatic_full = jacrev(model)(source).detach().cpu().numpy()
    automatic_shape = jacrev(model.final_shape)(source).detach().cpu().numpy()
    (
        analytic_full,
        analytic_shape,
        stages,
        full_factors,
        shape_factors,
    ) = _analytic_chains(model, source)
    full_residual = _relative(automatic_full, analytic_full)
    shape_residual = _relative(automatic_shape, analytic_shape)
    full_metrics, full_gains = backward_path_metrics(full_factors)
    shape_metrics, shape_gains = backward_path_metrics(shape_factors)
    full_metric_report = verify_path_metric(
        full_factors, full_metrics, full_gains)
    shape_metric_report = verify_path_metric(
        shape_factors, shape_metrics, shape_gains)

    rng = np.random.default_rng(int(arguments.seed) + 1)
    rows = rng.normal(size=(12, int(arguments.width)))
    jvp = -0.15 * rows
    radii = np.full(12, 1.0e-10)
    prediction = stream_registry_prediction(
        rows,
        jvp,
        radii,
        pair_chunk_size=17,
        successor_rows=0.85 * rows,
    )

    dimension = min(8, int(arguments.width))
    value = rng.normal(scale=0.15, size=(dimension, dimension))
    reference = np.broadcast_to(np.mean(value, axis=0), value.shape).copy()
    weight = rng.normal(scale=0.1, size=value.shape)
    bias = rng.normal(scale=0.05, size=value.shape)
    exponent = rng.uniform(-0.2, 0.6, size=value.shape)
    coupling = rng.normal(scale=0.1, size=value.shape)
    coupling_bias = rng.normal(scale=0.05, size=value.shape)
    transfer = plga_transfer(
        value,
        reference,
        weight,
        bias,
        exponent,
        coupling,
        coupling_bias,
    )

    checks = {
        "full_postnorm_chain_matches_autodiff": full_residual < 2.0e-11,
        "shape_postnorm_chain_matches_autodiff": shape_residual < 2.0e-11,
        "full_backward_path_metrics_hold": bool(
            full_metric_report["quadratic_inequalities_hold"]),
        "shape_backward_path_metrics_hold": bool(
            shape_metric_report["quadratic_inequalities_hold"]),
        "causal_contraction_is_source_state": (
            prediction["decision"] == "CONTRACTION_CERTIFIED"
            and prediction["decision_uses_successor"] is False
        ),
        "causal_pair_bounds_cover_successor": bool(
            prediction["coverage"]["all_pair_enclosures_hold"]),
        "composite_plga_secant_reconstructs": (
            float(transfer["identity_relative_residual"]) < 2.0e-11),
    }
    qualified = all(checks.values())
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "QUALIFIED" if qualified else "NOT_QUALIFIED",
        "checks": checks,
        "postnorm": {
            "width": int(arguments.width),
            "hidden": int(arguments.hidden),
            "residual_units": 8,
            "gated_blocks_per_unit": 2,
            "layernorm_epsilon": 1.0e-6,
            "full_chain_relative_residual": full_residual,
            "shape_chain_relative_residual": shape_residual,
            "stages": stages,
            "full_backward_path_metric": full_metric_report,
            "shape_backward_path_metric": shape_metric_report,
        },
        "causal_prediction": prediction,
        "plga": {
            "adjustment_epsilon": 1.0e-9,
            "identity_relative_residual": float(
                transfer["identity_relative_residual"]),
            "minimum_positive_base": float(transfer["minimum_positive_base"]),
            "negative_exponent_fraction": float(
                transfer["negative_exponent_fraction"]),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else platform.processor()
            ),
        },
        "resources": {
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
            "peak_gpu_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda" else 0
            ),
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                * (1024 if sys.platform != "darwin" else 1),
        },
    }
    return report


def write_report(path: str | Path, report: dict[str, object]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=12)
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    if arguments.width < 2 or arguments.hidden < 2:
        raise SystemExit("width and hidden dimensions must be at least two")
    report = qualify(arguments)
    write_report(arguments.output, report)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "status": report["status"],
    }, sort_keys=True))
    if arguments.require_pass and report["status"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
