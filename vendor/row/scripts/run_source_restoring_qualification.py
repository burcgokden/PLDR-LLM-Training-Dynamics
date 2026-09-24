#!/usr/bin/env python3
"""Run deterministic source-restoring fixtures and optional device checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from analysis.analyze_source_restoring_confirmation import (  # noqa: E402
    source_prediction_summary,
    write_json_atomic,
)
from confirm.source_restoring import (  # noqa: E402
    centered_row_geometry,
    cover_native_successor,
    pair_contrasts,
    source_multiplier_cocycle,
    source_restoring_certificate,
    symmetric_taylor_remainder,
    two_channel_spatial_bound,
)
from confirm.source_restoring_specs import CAMPAIGN_ID  # noqa: E402


QUALIFICATION_SCHEMA = "pldr-source-restoring-qualification-v1"


def fixture() -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(43001)
    source = rng.normal(size=(3, 8, 6))
    source -= np.mean(source, axis=1, keepdims=True)
    gate = np.zeros_like(source)
    gate[0] = -0.2 * source[0]
    gate[1] = 0.2 * source[1]
    shape = np.zeros_like(source)
    shape[2, :, 0] = -0.2 * source[2, :, 1]
    shape[2, :, 1] = 0.2 * source[2, :, 0]
    input_velocity = np.zeros_like(source)
    velocity = gate + shape
    endpoint = source + velocity
    _, velocity_contrast = pair_contrasts(velocity)
    curvature = 2.0 * np.einsum(
        "mpd,mpd->mp", velocity_contrast, velocity_contrast
    )
    remainder_lower, remainder_upper = symmetric_taylor_remainder(
        curvature, curvature.shape
    )
    _, endpoint_contrast = pair_contrasts(endpoint)
    certificate = source_restoring_certificate(
        source,
        gate,
        shape,
        input_velocity,
        remainder_lower=remainder_lower,
        remainder_upper=remainder_upper,
        endpoint_contrast_norm_upper=np.linalg.norm(
            endpoint_contrast, axis=-1
        ),
        native_contrast_radius=0.0,
    )
    return source, endpoint, certificate


def run(device_name: str) -> dict:
    started = time.monotonic()
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA qualification requested without CUDA")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
        probe = torch.arange(4096, dtype=torch.float64, device=device)
        probe = probe.square().sum()
        probe.cpu()
        torch.cuda.synchronize(device)
    source, endpoint, certificate = fixture()
    prediction = source_prediction_summary(certificate)
    coverage = cover_native_successor(certificate, endpoint)
    geometry = centered_row_geometry(source)
    spatial = two_channel_spatial_bound(
        [0.2, 0.2, 0.2],
        [0.01, 1.0, 0.5],
        [2.0, 2.0, 2.0],
        feature_dimension=source.shape[-1],
    )
    cocycle = source_multiplier_cocycle(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([
            [1.05, 0.9, 1.1],
            [0.8, 1.05, 0.85],
            [0.9, 0.9, 0.9],
        ]),
    )
    cuda = device.type == "cuda"
    passed = bool(
        geometry["valid"]
        and coverage["technical_valid"]
        and prediction["outcome_counts"] == {
            "confirmed": 1,
            "not_confirmed": 1,
            "unresolved": 1,
        }
        and cocycle["prefix_product"][1, 0] > 1.0
        and cocycle["prefix_product"][-1, 0] < 1.0
        and np.all(spatial["combined_bound"] >= 0.0)
    )
    return {
        "schema_version": QUALIFICATION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "passed" if passed else "failed",
        "technical_valid": passed,
        "device": str(device),
        "source_values_only_before_coverage": True,
        "scientific_outcome_counts": prediction["outcome_counts"],
        "coverage_fraction": coverage["coverage_fraction"],
        "centered_geometry_valid": geometry["valid"],
        "maximum_centering_identity_residual": geometry[
            "maximum_abs_identity_residual"
        ],
        "intermittent_expansion_fixture": {
            "first_multiplier": float(cocycle["prefix_product"][1, 0]),
            "final_product": float(cocycle["prefix_product"][-1, 0]),
        },
        "spatial_combined_bounds": [
            float(value) for value in spatial["combined_bound"]
        ],
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device.index)) if cuda else 0
            ),
            "peak_gpu_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device.index)) if cuda else 0
            ),
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ) * (1024 if sys.platform != "darwin" else 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = run(arguments.device)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "status": report["status"],
        "device": report["device"],
    }, sort_keys=True))
    if arguments.require_pass and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
