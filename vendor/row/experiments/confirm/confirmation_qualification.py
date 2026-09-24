"""Pre-campaign qualification for measurement and record machinery."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from campaign_record import (
    RecordError, binding_for, strict_load, verify_binding,
)
from direct_certificate import deductive_output_order_bridge
from optimizer_transport import AdamWConstants, ordered_adamw_step
from check_certificate import check_small_certificate
from row_domain import disconnected_ball_counterexample
from schedule_clock import boundary_table, verify_training_schedule
from small_pldr_certificate import build_small_certificate
from tail_comparison import finite_trace_cannot_prove_infinite_tail
from validated_interval import RationalInterval, as_fraction, exp_enclosure
from validated_linear_algebra import (
    residual_aware_lyapunov_certificate,
    spectral_norm_upper,
)
from validated_row_map import certify_row_map
from renormalization_group import (
    apply_affine_cocycle,
    block_average_semigroup_residual,
    block_affine_cocycle,
    block_softmax_aggregation,
    metric_layernorm_scale_identity,
    query_density_coarse_graining,
)
from row_map_jacobian import qualify_synthetic
from qualification_full_stack import full_stack_fixture
from qualification_validated import (
    adversarial_fixture,
    validated_certificate_fixture,
)


def _schedule_fixture():
    from train_run import LinearWarmupCosineLRSchedule

    total = 40
    warmup = 8
    floor = 0.1
    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = LinearWarmupCosineLRSchedule(
        optimizer, total_steps=total, warmup_steps=warmup, alpha=floor)
    factors = []
    for _ in range(total):
        factors.append(float(scheduler.get_last_lr()[0]))
        optimizer.step()
        scheduler.step()
    clock_check = verify_training_schedule(
        factors, total_updates=total, warmup_updates=warmup, floor=floor,
    )

    q = np.array([0.80, 0.72, 0.65, 0.60], dtype=np.float64)
    w = np.array([0.10, 0.04, 0.02, 0.01], dtype=np.float64)
    initial = 0.9
    direct = initial
    for gain, forcing in zip(q, w):
        direct = gain * direct + forcing
    product = float(np.prod(q))
    convolution = 0.0
    for index, forcing in enumerate(w):
        convolution += float(forcing * np.prod(q[index + 1:]))
    unroll_error = abs(direct - (product * initial + convolution))
    maximum_error = max(clock_check["maximum_absolute_error"], unroll_error)
    return {
        "status": "PASS" if maximum_error <= 1e-14 else "FAIL",
        "maximum_schedule_endpoint_error":
            clock_check["maximum_absolute_error"],
        "nonautonomous_unroll_error": unroll_error,
        "warmup_step": warmup,
        "terminal_factor": factors[-1],
        "boundary_table": boundary_table(
            total_updates=total, warmup_updates=warmup,
            floor=floor, beta1=0.9),
    }


def _order_parameter_fixture():
    operator = np.diag(np.array([0.5, 0.25], dtype=np.float64))
    rows_a = np.array([
        [1.0, 0.5],
        [0.5, 1.5],
        [1.5, 1.0],
    ], dtype=np.float64)
    rows_b = np.array([
        [0.8, 0.4],
        [0.4, 1.2],
        [1.2, 0.8],
    ], dtype=np.float64)
    output_a = rows_a @ operator.T
    output_b = rows_b @ operator.T
    direct_upper = float(np.linalg.norm(operator, ord=2))
    bridge = deductive_output_order_bridge(
        output_a=output_a,
        output_b=output_b,
        rows_a=rows_a,
        rows_b=rows_b,
        direct_row_map_upper=direct_upper,
    )
    return {
        "status": "PASS" if bridge["bridge_holds"] else "FAIL",
        "source_order_parameter": bridge["source_order_parameter"],
        "bridge_upper": bridge["order_parameter_upper"],
        "bridge_ratio": bridge["order_parameter_bridge_ratio"],
        "direct_operator_norm": direct_upper,
    }


def _renormalization_fixture():
    queries = np.array([
        [1.0, 0.0, 0.5],
        [0.8, 0.2, 0.4],
        [0.1, 1.1, 0.3],
        [0.2, 0.9, 0.5],
        [1.2, 0.3, 0.7],
        [1.0, 0.5, 0.6],
        [0.4, 1.0, 0.8],
        [0.6, 0.8, 0.9],
    ], dtype=np.float64)
    density = query_density_coarse_graining(queries, 2)
    layernorm = metric_layernorm_scale_identity(
        density["coarse_normalized_density"],
        block_size=2,
        epsilon=1e-6,
        gamma=np.array([0.9, 1.1, 1.2]),
        beta=np.array([0.1, -0.2, 0.05]),
    )

    matrices = np.array([
        [[0.80, 0.10], [0.00, 0.70]],
        [[0.75, 0.00], [0.05, 0.65]],
        [[0.70, 0.05], [0.00, 0.60]],
        [[0.65, 0.00], [0.02, 0.55]],
    ], dtype=np.float64)
    forcings = np.array([
        [0.10, -0.02],
        [0.04, 0.01],
        [0.02, 0.03],
        [0.01, -0.01],
    ], dtype=np.float64)
    initial = np.array([0.9, -0.4], dtype=np.float64)
    fine_states = apply_affine_cocycle(matrices, forcings, initial)
    blocked_two = block_affine_cocycle(matrices, forcings, 2)
    blocked_states = apply_affine_cocycle(
        blocked_two["matrices"], blocked_two["forcings"], initial)
    temporal_error = float(np.max(np.abs(
        fine_states[::2] - blocked_states)))
    blocked_four = block_affine_cocycle(matrices, forcings, 4)
    twice_blocked = block_affine_cocycle(
        blocked_two["matrices"], blocked_two["forcings"], 2)
    semigroup_error = max(
        float(np.max(np.abs(
            blocked_four["matrices"] - twice_blocked["matrices"]))),
        float(np.max(np.abs(
            blocked_four["forcings"] - twice_blocked["forcings"]))),
    )
    token_semigroup_error = block_average_semigroup_residual(
        queries, 2, 2)
    softmax = block_softmax_aggregation(
        np.array([0.2, -0.1, 0.7, 0.8, -0.5, 0.3, 1.0, 0.6]),
        2,
    )
    positivity_error = max(
        0.0, -density["within_covariance_min_eigenvalue"])
    maximum_error = max(
        density["density_identity_residual"],
        density["trace_identity_residual"],
        layernorm["relative_residual"],
        temporal_error,
        semigroup_error,
        token_semigroup_error,
        softmax["aggregation_residual"],
        positivity_error,
    )
    return {
        "status": "PASS" if maximum_error <= 1e-13 else "FAIL",
        "density_identity_residual":
            density["density_identity_residual"],
        "within_covariance_min_eigenvalue":
            density["within_covariance_min_eigenvalue"],
        "layernorm_scale_identity_residual":
            layernorm["relative_residual"],
        "temporal_blocking_residual": temporal_error,
        "blocking_semigroup_residual": semigroup_error,
        "token_blocking_semigroup_residual": token_semigroup_error,
        "softmax_block_aggregation_residual":
            softmax["aggregation_residual"],
    }


def _optimizer_fixture():
    theta = np.array([1.0, -2.0, 0.25], dtype=np.float64)
    gradient = np.array([3.0, -0.25, 0.1], dtype=np.float64)
    parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.01, betas=(0.9, 0.95), eps=1e-5,
        weight_decay=0.1,
    )
    parameter.grad = torch.tensor(gradient, dtype=torch.float64)
    expected = ordered_adamw_step(
        theta, gradient, np.zeros_like(theta), np.zeros_like(theta),
        step=1, learning_rate=0.01,
        constants=AdamWConstants(clip_value=0.5),
    )
    torch.nn.utils.clip_grad_value_([parameter], 0.5)
    optimizer.step()
    error = float(np.max(np.abs(
        parameter.detach().numpy() - expected["theta"])))
    return {
        "status": "PASS" if error <= 1e-14 else "FAIL",
        "maximum_parameter_error": error,
        "replay_ledger_relative_residual":
            expected["ledger_relative_residual"],
    }


def _record_fixture():
    checks = {}
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        duplicate = directory / "duplicate.json"
        duplicate.write_text('{"x": 1, "x": 2}\n', encoding="utf-8")
        try:
            strict_load(duplicate)
        except RecordError:
            checks["duplicate_key_rejected"] = True
        else:
            checks["duplicate_key_rejected"] = False

        nonfinite = directory / "nonfinite.json"
        nonfinite.write_text('{"x": NaN}\n', encoding="utf-8")
        try:
            strict_load(nonfinite)
        except RecordError:
            checks["nonfinite_rejected"] = True
        else:
            checks["nonfinite_rejected"] = False

        payload = directory / "payload.bin"
        payload.write_bytes(b"row-map-confirmation-fixture")
        binding = binding_for(payload, directory)
        checks["fresh_binding_accepted"] = (
            verify_binding(binding, directory, "fixture") == payload)
        payload.write_bytes(b"changed")
        try:
            verify_binding(binding, directory, "fixture")
        except RecordError:
            checks["stale_binding_rejected"] = True
        else:
            checks["stale_binding_rejected"] = False
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }


def run_qualification(device="cpu"):
    jacobian = qualify_synthetic(device=device)
    optimizer = _optimizer_fixture()
    schedule = _schedule_fixture()
    observable = _order_parameter_fixture()
    renormalization = _renormalization_fixture()
    record = _record_fixture()
    validated = validated_certificate_fixture()
    full_stack = full_stack_fixture()
    adversarial = adversarial_fixture()
    passed = all(
        row["status"] == "PASS"
        for row in (
            jacobian, optimizer, schedule, observable, renormalization,
            record, validated, adversarial, full_stack,
        )
    )
    return {
        "qualification": "Q0",
        "status": "PASS" if passed else "FAIL",
        "jacobian": jacobian,
        "optimizer_transport": optimizer,
        "schedule": schedule,
        "order_parameter_bridge": observable,
        "renormalization_group": renormalization,
        "record_contract": record,
        "validated_certificate": validated,
        "adversarial": adversarial,
        "full_stack": full_stack,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": device,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_qualification(args.device)
    text = json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
