#!/usr/bin/env python3
"""Run analytic prequalification for the Rev41 contrast-energy pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
ANALYSIS = ROOT / "experiments" / "analysis"
sys.path.insert(0, str(CONFIRM))
sys.path.insert(0, str(ANALYSIS))

from explicit_layernorm import affine_path_coefficients  # noqa: E402
from produce_validated_contrast_energy_jets import (  # noqa: E402
    INPUT_SCHEMA,
    SOURCE_SCHEMA,
    produce,
    sha256_path,
    write_npz,
)
from analyze_contrast_energy_confirmation import analyze  # noqa: E402
from composite_plga import (  # noqa: E402
    plga_transfer,
    restricted_operator_coefficients,
    row_center,
    row_variance_identity,
)


SCHEMA_VERSION = "pldr-contrast-energy-analytic-prequalification-v1"


def _torch_layernorm(x, gamma, beta, epsilon):
    centered = x - x.mean(dim=-1, keepdim=True)
    return gamma * centered / torch.sqrt(
        epsilon + (centered * centered).mean(dim=-1, keepdim=True)
    ) + beta


def _derivative_qualification(device: torch.device) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(41041)
    x_cpu = torch.randn((4, 64), generator=generator, dtype=torch.float64)
    direction_cpu = torch.randn((4, 64), generator=generator, dtype=torch.float64) / 5
    gamma_cpu = 0.75 + torch.rand((64,), generator=generator, dtype=torch.float64)
    beta_cpu = torch.randn((64,), generator=generator, dtype=torch.float64) / 7
    probe_cpu = torch.randn((4, 64), generator=generator, dtype=torch.float64)
    x = x_cpu.to(device)
    direction = direction_cpu.to(device)
    gamma = gamma_cpu.to(device)
    beta = beta_cpu.to(device)
    probe = probe_cpu.to(device)
    epsilon = 1.0e-6
    coefficients = affine_path_coefficients(
        x_cpu.numpy(), direction_cpu.numpy(), gamma_cpu.numpy(),
        beta_cpu.numpy(), epsilon=epsilon,
    )

    scalar = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    output = _torch_layernorm(x + scalar * direction, gamma, beta, epsilon)
    projected = torch.sum(output * probe)
    reverse = []
    current = projected
    for _ in range(3):
        current = torch.autograd.grad(current, scalar, create_graph=True)[0]
        reverse.append(float(current.detach().cpu()))
    analytic = [
        float(np.sum(coefficients[name] * probe_cpu.numpy()))
        for name in ("first", "second", "third")
    ]
    reverse_errors = [abs(left - right) for left, right in zip(
        analytic, reverse, strict=True)]

    def projected_value(position: float) -> float:
        with torch.no_grad():
            value = _torch_layernorm(
                x + position * direction, gamma, beta, epsilon)
            return float(torch.sum(value * probe).cpu())

    step1 = 2.0e-5
    step2 = 2.0e-4
    f0 = projected_value(0.0)
    richardson_first = (
        4.0 * (projected_value(step1 / 2) - projected_value(-step1 / 2))
        / step1
        - (projected_value(step1) - projected_value(-step1))
        / (2.0 * step1)
    ) / 3.0
    second_h = (
        projected_value(step2) - 2.0 * f0 + projected_value(-step2)
    ) / (step2 * step2)
    half = step2 / 2
    second_half = (
        projected_value(half) - 2.0 * f0 + projected_value(-half)
    ) / (half * half)
    richardson_second = (4.0 * second_half - second_h) / 3.0

    first_error = abs(richardson_first - analytic[0])
    second_error = abs(richardson_second - analytic[1])

    row0 = coefficients["value"][0]
    row1 = coefficients["value"][1]
    first0 = coefficients["first"][0]
    first1 = coefficients["first"][1]
    second0 = coefficients["second"][0]
    second1 = coefficients["second"][1]
    contrast = row0 - row1
    contrast_first = first0 - first1
    contrast_second = second0 - second1
    energy_analytic = [
        2.0 * float(np.dot(contrast, contrast_first)),
        2.0 * float(
            np.dot(contrast_first, contrast_first)
            + np.dot(contrast, contrast_second)
        ),
    ]
    scalar_energy = torch.zeros(
        (), dtype=torch.float64, device=device, requires_grad=True)
    pair_rows = _torch_layernorm(
        x[:2] + scalar_energy * direction[:2], gamma, beta, epsilon)
    energy = torch.sum((pair_rows[0] - pair_rows[1]) ** 2)
    energy_first = torch.autograd.grad(
        energy, scalar_energy, create_graph=True)[0]
    energy_second = torch.autograd.grad(
        energy_first, scalar_energy, create_graph=True)[0]
    energy_reverse = [
        float(energy_first.detach().cpu()),
        float(energy_second.detach().cpu()),
    ]
    energy_errors = [abs(left - right) for left, right in zip(
        energy_analytic, energy_reverse, strict=True)]
    passed = (
        max(reverse_errors) <= 5.0e-10
        and first_error <= 1.0e-7
        and second_error <= 2.0e-5
        and max(energy_errors) <= 5.0e-10
    )
    return {
        "passed": passed,
        "resolved_device": str(device),
        "reverse_over_reverse_absolute_errors": reverse_errors,
        "richardson_first_absolute_error": first_error,
        "richardson_second_absolute_error": second_error,
        "contrast_energy_reverse_absolute_errors": energy_errors,
    }


def _pipeline_qualification(directory: Path) -> dict:
    source = directory / "source.npz"
    physical = np.asarray([[[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]])
    physical_jvp = -0.25 * physical
    with source.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(SOURCE_SCHEMA),
            campaign_id=np.asarray(
                "pldr-contrast-energy-analytic-qualification-v1"),
            successor_evaluated=np.asarray(False),
            step_start=np.asarray(1000, dtype=np.int64),
            optimizer_step_after=np.asarray([1001], dtype=np.int64),
            normalized_rows=np.zeros((1, 3, 2), dtype=np.float64),
            gate=np.ones((1, 2), dtype=np.float64),
            physical_rows=physical,
            physical_row_jvp=physical_jvp,
            source_loss=np.asarray([1.0]),
            raw_gradient_sha256=np.asarray(["3" * 64]),
            clipped_gradient_sha256=np.asarray(["4" * 64]),
            optimizer_state_sha256=np.asarray(["5" * 64]),
            source_parameter_sha256=np.asarray(["6" * 64]),
            parameter_displacement_sha256=np.asarray(["7" * 64]),
            predicted_successor_parameter_sha256=np.asarray(["9" * 64]),
            successor_parameter_sha256=np.asarray([""]),
            full_parameter_update_bitwise_match=np.asarray(
                [], dtype=np.bool_),
            update_batch_sha256=np.asarray(["8" * 64]),
            data_cursor_before=np.asarray(0, dtype=np.int64),
            data_cursor_after=np.asarray(32, dtype=np.int64),
            update_batch_chunk_indices=np.arange(
                32, dtype=np.int64)[None, :],
            checkpoint_sha256=np.asarray("0" * 64),
            registry_sha256=np.asarray("1" * 64),
            lock_sha256=np.asarray(""),
            data_order_sha256=np.asarray("2" * 64),
            role=np.asarray("qualification"),
            seed=np.asarray(41041, dtype=np.int64),
            layer=np.asarray(0, dtype=np.int64),
        )
    source_digest = sha256_path(source)
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    source_energy = np.asarray([1.0, 9.0, 4.0])
    source_slope = -0.5 * source_energy
    exact_second = 0.125 * source_energy
    knots = np.asarray([0.0, 0.25, 0.5, 1.0])
    endpoints = np.repeat(exact_second[:, None], len(knots), axis=1)
    taylor = directory / "taylor-model.npz"
    with taylor.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(INPUT_SCHEMA),
            source_sha256=np.asarray(source_digest),
            successor_values_present=np.asarray(False),
            pairs=pairs,
            source_energy=source_energy,
            source_slope=source_slope,
            knots=knots,
            endpoint_second_lower=np.nextafter(endpoints, -np.inf),
            endpoint_second_upper=np.nextafter(endpoints, np.inf),
            third_modulus=np.full((3, 3), 1.0e-12),
            source_charge=np.full(3, 1.0e-14),
            arithmetic_charge=np.full(3, 1.0e-13),
            metadata_json=np.asarray(json.dumps({
                "model_order": 3,
                "interval_backend": "analytic_polynomial_fixture",
                "explicit_layernorm": True,
                "rope_interval_enclosure": True,
                "adaptive_subdivision_rule": "qualification_fixed_grid",
                "taylor_policy_sha256": "a" * 64,
                "backend_manifest_sha256": "b" * 64,
                "source_parameter_sha256": "6" * 64,
                "parameter_displacement_sha256": "7" * 64,
                "predicted_successor_parameter_sha256": "9" * 64,
                "anchor": 0,
                "trajectory": "qualification",
                "layer": 0,
            }, sort_keys=True, separators=(",", ":"))),
        )
    jets = directory / "jets.npz"
    write_npz(jets, produce(source, taylor))
    report = analyze(jets)
    exact_successor = 0.5625 * source_energy
    prediction = report["prediction"]
    passed = (
        prediction["decision"] == "contraction"
        and prediction["decision_uses_successor"] is False
        and np.all(exact_successor >= np.load(jets)["successor_lower"])
        and np.all(exact_successor <= np.load(jets)["successor_upper"])
    )
    return {
        "passed": bool(passed),
        "decision": prediction["decision"],
        "pair_count": prediction["pair_count"],
        "lower_multiplier": prediction["lower_multiplier"],
        "upper_multiplier": prediction["upper_multiplier"],
        "source_sha256": source_digest,
        "taylor_model_sha256": sha256_path(taylor),
        "jets_sha256": sha256_path(jets),
    }


def _plga_qualification() -> dict:
    rng = np.random.default_rng(41071)
    value = rng.normal(scale=0.2, size=(4, 4))
    occupied, reference = row_center(value)
    weight = rng.normal(scale=0.15, size=(4, 4))
    bias = rng.normal(scale=0.1, size=(4, 4))
    exponent = rng.uniform(-0.75, 1.25, size=(4, 4))
    coupling = rng.normal(scale=0.15, size=(4, 4))
    coupling_bias = rng.normal(scale=0.1, size=(4, 4))
    query = rng.normal(scale=0.2, size=(4, 4))
    key = rng.normal(scale=0.2, size=(4, 4))

    transfer = plga_transfer(
        value,
        reference,
        weight,
        bias,
        exponent,
        coupling,
        coupling_bias,
    )
    variance = row_variance_identity(value)
    coefficients = restricted_operator_coefficients(
        composite_secant_value=transfer["composite_secant"],
        weight=weight,
        coupling=coupling,
        query=query,
        key=key,
        iterations=300,
        seed=41072,
    )
    realized = np.asarray(transfer["realized_difference"])
    centered_realized = realized - np.mean(realized, axis=0, keepdims=True)
    occupied_norm = float(np.linalg.norm(occupied))
    realized_coefficient = (
        float(np.linalg.norm(centered_realized)) / occupied_norm
    )
    finite_coefficients = all(
        np.isfinite(float(value)) and float(value) >= 0.0
        for value in coefficients.values()
    )
    passed = (
        float(transfer["identity_relative_residual"]) <= 2.0e-13
        and float(transfer["minimum_positive_base"]) > 0.0
        and float(transfer["negative_exponent_fraction"]) > 0.0
        and float(variance["identity_relative_residual"]) <= 2.0e-13
        and float(variance["centered_frobenius_squared"])
        <= float(variance["diameter_upper_squared"]) + 2.0e-13
        and occupied_norm > 0.0
        and np.isfinite(realized_coefficient)
        and realized_coefficient
        <= coefficients["row_centered_composite_coefficient"] * (1.0 + 1.0e-10)
        and finite_coefficients
    )
    return {
        "passed": bool(passed),
        "composite_secant_identity_relative_residual": float(
            transfer["identity_relative_residual"]),
        "minimum_positive_base": float(
            transfer["minimum_positive_base"]),
        "negative_exponent_fraction": float(
            transfer["negative_exponent_fraction"]),
        "row_variance_identity_relative_residual": float(
            variance["identity_relative_residual"]),
        "realized_occupied_coefficient": realized_coefficient,
        "restricted_coefficients": coefficients,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    started = time.monotonic()
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA qualification but CUDA is unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    derivative = _derivative_qualification(device)
    with tempfile.TemporaryDirectory(prefix="pldr-contrast-q0-") as temporary:
        pipeline = _pipeline_qualification(Path(temporary))
    plga = _plga_qualification()
    peak_host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_host *= 1024
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "analytic_prequalification_not_campaign_confirmation",
        "campaign_execution_authorized": False,
        "passed": bool(
            derivative["passed"] and pipeline["passed"] and plga["passed"]),
        "derivatives": derivative,
        "pipeline": pipeline,
        "composite_plga": plga,
        "resources": {
            "wall_seconds": float(time.monotonic() - started),
            "peak_host_rss_bytes": peak_host,
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else 0
            ),
            "peak_gpu_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device))
                if device.type == "cuda" else 0
            ),
        },
    }
    target = Path(arguments.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n", encoding="utf-8")
    temporary.replace(target)
    if arguments.require_pass and not report["passed"]:
        raise SystemExit("contrast-energy analytic qualification failed")
    print(json.dumps({"output": str(target), "passed": report["passed"]}, sort_keys=True))


if __name__ == "__main__":
    main()
