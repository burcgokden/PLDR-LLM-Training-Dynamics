#!/usr/bin/env python3
"""Run deterministic Q0 fixtures for the block-normal campaign."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analysis.analyze_block_normal_confirmation import (  # noqa: E402
    analyze_records,
    write_json_atomic,
)
from confirm.block_normal import (  # noqa: E402
    backward_metrics,
    block_prediction,
)
from confirm.block_normal_specs import (  # noqa: E402
    BLOCK_RECORD_SCHEMA,
    CAMPAIGN_ID,
    QUALIFICATION_SCHEMA,
    SCIENTIFIC_OUTCOMES,
    campaign_design,
    validate_design,
)
from confirm.checkpoint_retention import (  # noqa: E402
    in_memory_recovery_check,
    retention_record,
)
from confirm.evidence_seal import (  # noqa: E402
    seal_native,
    seal_source,
    write_new_json,
)
from confirm.finite_transport import (  # noqa: E402
    affine_increment,
    chronological_unroll,
    exact_energy_balance,
    named_transport_ledger,
    plga_multiplied_fields,
    product_increment,
)
from confirm.native_determinism import (  # noqa: E402
    native_and_conversion_report,
)
from confirm.normal_projection import (  # noqa: E402
    metric_projection,
    projection_valid,
    quadratic_defect_profile,
    sign_paired_perturbations,
)
from confirm.resource_executor import (  # noqa: E402
    ResourceCaps,
    run_capped,
)


def _device_probe(device_name: str) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA qualification requested without CUDA")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        value = torch.linspace(0.0, 1.0, 4096, device=device)
        checksum = float(torch.sum(value * value).cpu())
        torch.cuda.synchronize(device)
        return {
            "device": str(device),
            "checksum": checksum,
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        }
    value = torch.linspace(0.0, 1.0, 4096)
    return {
        "device": "cpu",
        "checksum": float(torch.sum(value * value)),
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
    }


def _fixture_record(prediction: dict[str, Any]) -> dict[str, Any]:
    now = time.time_ns()
    comparisons = [
        {
            "prediction": "physical_collapse",
            "map_id": "fixture-map-confirmed",
            "context": "fixture-c0",
            "layer": 0,
            "head": 0,
            "observed_lower": 0.39,
            "observed_upper": 0.40,
            "bound_lower": 0.50,
            "bound_upper": 0.51,
        },
        {
            "prediction": "physical_collapse",
            "map_id": "fixture-map-not-confirmed",
            "context": "fixture-c1",
            "layer": 1,
            "head": 1,
            "observed_lower": 0.70,
            "observed_upper": 0.71,
            "bound_lower": 0.49,
            "bound_upper": 0.50,
        },
        {
            "prediction": "normal_force_closure",
            "map_id": "fixture-map-unresolved",
            "context": "fixture-c2",
            "layer": 2,
            "head": 2,
            "observed_lower": 0.55,
            "observed_upper": 0.65,
            "bound_lower": 0.59,
            "bound_upper": 0.61,
        },
    ]
    return {
        "schema_version": BLOCK_RECORD_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "stage": "Q0",
        "trajectory": "qualification-fixture",
        "seed": 44001,
        "block_start": 0,
        "block_end": len(prediction["gains"]),
        "source_checkpoint_path": "/qualification/source-complete.pt",
        "source_checkpoint_sha256": "a" * 64,
        "prediction_created_at_ns": now,
        "native_endpoint_opened_at_ns": now + 1,
        "technical_valid": True,
        "source_values_only": True,
        "gains": prediction["gains"].tolist(),
        "quadratic_coefficients": (
            prediction["quadratic_coefficients"].tolist()
        ),
        "forces": prediction["forces"].tolist(),
        "radii": prediction["radii"].tolist(),
        "initial_normal_upper": 0.2,
        "prefix_products": prediction["prefix_products"].tolist(),
        "force_contributions": (
            prediction["force_contributions"].tolist()
        ),
        "normal_envelope": prediction["envelope"].tolist(),
        "comparisons": comparisons,
    }


def run(device_name: str) -> dict[str, Any]:
    started = time.monotonic()
    validate_design(campaign_design())
    rng = np.random.default_rng(44001)

    w0 = rng.normal(size=(5, 4))
    w1 = w0 + rng.normal(scale=0.01, size=w0.shape)
    x0 = rng.normal(size=4)
    x1 = x0 + rng.normal(scale=0.02, size=x0.shape)
    b0 = rng.normal(size=5)
    b1 = b0 + rng.normal(scale=0.01, size=b0.shape)
    affine = affine_increment(w0, w1, x0, x1, b0, b1)
    product = product_increment(
        rng.normal(size=7),
        rng.normal(size=7),
        rng.normal(size=7),
        rng.normal(size=7),
    )
    plga = plga_multiplied_fields(
        np.asarray([-2.0, -0.1, 0.0, 1.0]),
        np.asarray([1.5, 0.2, 0.0, 2.0]),
        np.asarray([0.7, 1.2, -0.4, 2.0]),
        np.asarray([1.1, 0.9, -0.4, 1.5]),
    )
    maps = [rng.normal(scale=0.1, size=(4, 4)) for _ in range(4)]
    sources = [rng.normal(scale=0.1, size=4) for _ in range(4)]
    unroll = chronological_unroll(maps, sources, rng.normal(size=4))

    rows0 = rng.normal(size=(3, 8, 6))
    components = {
        "upstream": rng.normal(scale=0.01, size=rows0.shape),
        "parameters": rng.normal(scale=0.01, size=rows0.shape),
        "layernorm": rng.normal(scale=0.01, size=rows0.shape),
        "plga": rng.normal(scale=0.01, size=rows0.shape),
    }
    rows1 = rows0 + sum(components.values(), start=np.zeros_like(rows0))
    transport = named_transport_ledger(rows0, rows1, components)
    energy = exact_energy_balance(rows0, rows1)

    projection = metric_projection(
        rng.normal(size=6),
        np.asarray([
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ]),
        np.zeros(2),
        np.diag(np.linspace(1.0, 2.0, 6)),
    )
    projection_ok = projection_valid(
        projection, absolute_tolerance=1.0e-10, required_rank=2
    )
    operator = np.asarray([
        [0.72, 0.08, 0.0],
        [0.0, 0.81, 0.04],
        [0.02, 0.0, 0.75],
    ])
    force = np.asarray([0.001, -0.002, 0.001])
    coefficient = 0.04

    def nonlinear(value):
        value = np.asarray(value, dtype=np.float64)
        return operator @ value + force + coefficient * value * value

    probes = sign_paired_perturbations(
        [1.0, -2.0, 1.0], 0.02
    )
    defect = quadratic_defect_profile(
        nonlinear, np.zeros(3), operator, force, probes
    )
    metrics = backward_metrics(
        [operator, operator * 1.02, operator * 0.98], np.eye(3)
    )
    quadratic = np.full(3, coefficient)
    forces = np.full(3, np.linalg.norm(force))
    radii = np.full(4, 1.0)
    prediction = block_prediction(
        metrics["gains"], quadratic, forces, radii, 0.2
    )
    record = _fixture_record(prediction)
    analysis = analyze_records([record])

    state = {
        "model": {"weight": rng.normal(size=(3, 3)).astype(np.float32)},
        "optimizer": {
            "first": rng.normal(size=3).astype(np.float32),
            "second": np.abs(rng.normal(size=3)).astype(np.float32),
        },
        "scheduler": {"step": np.asarray(1000, dtype=np.int64)},
        "rng": {"state": np.arange(8, dtype=np.uint64)},
    }
    native = native_and_conversion_report(
        state,
        copy.deepcopy(state),
        rows1.astype(np.float64),
        rows1.astype(np.float64),
    )
    recovery = in_memory_recovery_check(state, copy.deepcopy(state))
    device = _device_probe(device_name)

    with tempfile.TemporaryDirectory(prefix="pldr-block-normal-q0-") as raw:
        temporary = Path(raw)
        complete = temporary / "complete.pt"
        successor = temporary / "successor.pt"
        model_only = temporary / "model-only.pt"
        complete.write_bytes(b"complete-checkpoint-fixture")
        successor.write_bytes(b"native-successor-fixture")
        model_only.write_bytes(b"model-only-fixture")
        evidence_specification = {
            "stage": "Q1",
            "trajectory": "construction-seed7444",
            "seed": 7444,
            "block_start": 1000,
            "block_end": 1001,
            "operators": [operator.tolist()],
            "quadratic_coefficients": [coefficient],
            "forces": [float(np.linalg.norm(force))],
            "radii": [1.0, 1.0],
            "initial_normal_upper": 0.2,
            "comparison_bounds": [
                {
                    "comparison_id": "q0-map",
                    "prediction": "physical_collapse",
                    "map_id": "q0-map",
                    "context": "fixture-c0",
                    "layer": 0,
                    "head": 0,
                    "bound_lower": 0.4,
                    "bound_upper": 0.5,
                }
            ],
        }
        source_evidence = seal_source(
            evidence_specification, complete, created_at_ns=10
        )
        source_path = temporary / "q0.source.json"
        write_new_json(source_path, source_evidence)
        native_evidence = seal_native(
            source_path,
            successor,
            [{
                "comparison_id": "q0-map",
                "observed_lower": 0.2,
                "observed_upper": 0.3,
            }],
            opened_at_ns=11,
        )
        native_path = temporary / "q0.native.json"
        write_new_json(native_path, native_evidence)
        append_only_refusal = False
        try:
            write_new_json(source_path, source_evidence)
        except FileExistsError:
            append_only_refusal = True
        retention = retention_record(
            complete,
            model_only,
            open_descendants=["qualification-dependent"],
            recovery_report=recovery,
            maximum_complete_checkpoints=2,
            complete_checkpoint_count=2,
        )
        executor = run_capped(
            [sys.executable, "-c", "pass"],
            device="cpu",
            output_root=temporary / "executor",
            record_path=temporary / "executor-record.json",
            caps=ResourceCaps(
                wall_seconds=10.0,
                host_rss_bytes=2 * 1024**3,
                gpu_reserved_bytes=20 * 1024**3,
                output_bytes=1024**2,
                poll_seconds=0.01,
            ),
        )

    tolerance = 5.0e-12
    conditions = {
        "design_counts_and_budgets": True,
        "affine_endpoint_exact": (
            affine["maximum_abs_residual"] <= tolerance
        ),
        "product_endpoint_exact": (
            product["maximum_abs_residual"] <= tolerance
        ),
        "finite_source_unroll_exact": (
            unroll["maximum_abs_residual"] <= tolerance
        ),
        "centered_transport_exact": (
            transport["maximum_abs_transport_residual"] <= tolerance
        ),
        "centered_energy_exact": (
            energy["maximum_abs_residual"] <= tolerance
        ),
        "plga_both_orders_exact": (
            plga["maximum_abs_route_one_residual"] <= tolerance
            and plga["maximum_abs_route_two_residual"] <= tolerance
        ),
        "projection_primal_and_kkt": projection_ok,
        "quadratic_defect_resolved": bool(np.all(
            defect["defect_norms"]
            <= coefficient * defect["amplitudes"] ** 2 + tolerance
        )),
        "positive_scheduled_metrics": bool(
            np.all(metrics["minimum_metric_eigenvalues"] > 0.0)
        ),
        "lyapunov_residual_nonpositive": bool(
            np.all(metrics["lyapunov_residuals"] <= tolerance)
        ),
        "invariant_tube": prediction["tube_valid"],
        "all_classification_branches_fixture": (
            all(analysis["outcome_counts"][name] == 1
                for name in SCIENTIFIC_OUTCOMES)
        ),
        "native_bitwise_zero_radius": (
            native["native_radius_zero_from_bitwise_equality"]
        ),
        "native_conversion_separate": (
            not native["charges_combined"]
        ),
        "complete_state_recovery_fixture": recovery[
            "complete_reload_succeeded"
        ] and retention["recovery_complete"],
        "fixture_artifact_paths_distinct": retention["paths_distinct"],
        "append_only_evidence_seal": bool(
            append_only_refusal
            and native_evidence["opened_at_ns"]
            > source_evidence["created_at_ns"]
            and native_evidence["source_checkpoint_sha256"]
            == source_evidence["source_checkpoint_sha256"]
        ),
        "resource_executor_fixture_record": executor["technical_valid"],
    }
    passed = all(conditions.values())
    host_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    host_peak *= 1024 if sys.platform != "darwin" else 1
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "qualification_mode": "deterministic-fixture",
        "checkpoint_backed": False,
        "scientific_confirmation": False,
        "status": "passed" if passed else "failed",
        "technical_valid": passed,
        "device": device,
        "conditions": conditions,
        "fixture_outcome_counts": analysis["outcome_counts"],
        "maximum_residuals": {
            "affine": affine["maximum_abs_residual"],
            "product": product["maximum_abs_residual"],
            "finite_unroll": unroll["maximum_abs_residual"],
            "row_transport": transport[
                "maximum_abs_transport_residual"
            ],
            "energy": energy["maximum_abs_residual"],
            "plga_route_one": plga[
                "maximum_abs_route_one_residual"
            ],
            "plga_route_two": plga[
                "maximum_abs_route_two_residual"
            ],
            "projection_primal": projection["primal_residual_norm"],
            "projection_kkt": projection[
                "kkt_stationarity_residual_norm"
            ],
        },
        "block": {
            "gains": metrics["gains"],
            "lyapunov_residuals": metrics["lyapunov_residuals"],
            "effective_gains": prediction["effective_gains"],
            "prefix_products": prediction["prefix_products"],
            "force_contributions": prediction["force_contributions"],
            "envelope": prediction["envelope"],
        },
        "retention": retention,
        "resource_executor": executor,
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_host_rss_bytes": host_peak,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/pldr-block-normal-q0.json"),
    )
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = run(arguments.device)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "status": report["status"],
        "device": report["device"]["device"],
        "classification_branches": report["fixture_outcome_counts"],
        "output": str(arguments.output.resolve()),
    }, sort_keys=True))
    if arguments.require_pass and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
