#!/usr/bin/env python3
"""Analyze hashed PLDR normal-trajectory ensembles under the RG protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIRM = HERE.parent / "confirm"
sys.path.insert(0, str(CONFIRM))

from rg_universality_protocol_specs import (  # noqa: E402
    BLOCK_SIZES,
    THRESHOLDS,
)
from transition_kernel_rg import (  # noqa: E402
    analyze_trajectory_ensemble,
    block_fluctuation_samples,
    estimate_affine_kernel,
    relative_kernel_residual,
    sliced_wasserstein_distance,
)


REQUIRED_KEYS = {
    "normal_paths",
    "hidden_labels",
    "anchor_labels",
    "model_width",
    "model_depth",
    "seed",
    "source_state_sha256",
}
HEX64 = re.compile(r"[0-9a-f]{64}")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar_integer(value, name):
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    result = int(array)
    if result < 1 or float(array) != result:
        raise ValueError(f"{name} must be a positive integer")
    return result


def load_trajectory_artifact(path):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != REQUIRED_KEYS:
            raise ValueError(
                "trajectory artifact keys disagree with the frozen schema")
        paths = np.asarray(payload["normal_paths"], dtype=float)
        labels = np.asarray(payload["hidden_labels"])
        width = _scalar_integer(payload["model_width"], "model_width")
        depth = _scalar_integer(payload["model_depth"], "model_depth")
        seed = _scalar_integer(payload["seed"], "seed")
        anchor_labels = np.asarray(payload["anchor_labels"])
        source_digests = np.asarray(payload["source_state_sha256"])
    if anchor_labels.shape != (paths.shape[0],):
        raise ValueError("anchor_labels must match the replica axis")
    unique_anchors = sorted(set(anchor_labels.tolist()), key=str)
    if source_digests.shape != (len(unique_anchors),):
        raise ValueError("source-state digests must match unique anchors")
    source_digests = [str(value) for value in source_digests.tolist()]
    if not all(HEX64.fullmatch(value) for value in source_digests):
        raise ValueError("source_state_sha256 contains an invalid digest")
    if not np.isfinite(paths).all():
        raise ValueError("normal_paths contains a nonfinite value")
    if labels.shape != paths.shape[:2]:
        raise ValueError("hidden_labels does not match normal_paths")
    return {
        "path": path,
        "sha256": _sha256(path),
        "normal_paths": paths,
        "hidden_labels": labels,
        "anchor_labels": anchor_labels,
        "model_width": width,
        "model_depth": depth,
        "seed": seed,
        "source_state_sha256": source_digests,
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {name: _jsonable(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _system_conditions(result):
    levels = result["levels"]
    later_levels = [row for row in levels if row["block_size"] > 1]
    memory = [row["direct_memory_spectral_radius"] for row in levels]
    final = levels[-1]["fluctuation"]
    first = levels[0]["fluctuation"]
    closure = result["projection_closure"]
    flow = result["continuous_flow"]
    stationarity = result["stationarity"]
    anchor = result["anchor_diagnostics"]
    innovation = result["innovation_dependence"]
    return {
        "empirical_kernel_semigroup": max(
            row["kernel_semigroup_residual"] for row in later_levels)
            <= THRESHOLDS["empirical_kernel_semigroup_relative"],
        "projection_prediction_closure":
            closure["maximum_prediction_defect"]
            <= THRESHOLDS["projection_prediction_defect"],
        "projection_innovation_closure":
            closure["maximum_innovation_sliced_wasserstein"]
            <= THRESHOLDS["projection_innovation_sliced_wasserstein"],
        "stationarity_mean_bounded":
            stationarity["standardized_mean_defect"]
            <= THRESHOLDS["stationarity_standardized_mean_defect"],
        "stationarity_covariance_bounded":
            stationarity["relative_covariance_defect"]
            <= THRESHOLDS["stationarity_relative_covariance_defect"],
        "anchor_kernel_heterogeneity_bounded":
            anchor["maximum_kernel_relative_defect"]
            <= THRESHOLDS["anchor_kernel_relative_defect"],
        "innovation_cross_covariance_bounded":
            innovation["maximum_cross_covariance_frobenius"]
            <= THRESHOLDS["innovation_cross_covariance_frobenius"],
        "innovation_conditional_dependence_bounded":
            innovation["previous_energy_conditional_sliced_wasserstein"]
            <= THRESHOLDS["innovation_conditional_sliced_wasserstein"],
        "memory_contracts": memory[-1] < memory[0],
        "continuous_flow_embeddable": flow["decision"] == "EMBEDDABLE",
        "embedding_reconstructs": (
            flow["decision"] == "EMBEDDABLE"
            and flow["embedding_reconstruction_residual"]
            <= THRESHOLDS["embedding_reconstruction_relative"]
        ),
        "generator_is_gapped": (
            flow["decision"] == "EMBEDDABLE"
            and flow["generator_stability_edge"] > 0.0
        ),
        "stationary_diffusion_nonnegative": (
            flow["decision"] == "EMBEDDABLE"
            and flow["diffusion_edge"] >= -1e-9
        ),
        "higher_cumulants_contract": (
            final["maximum_absolute_skewness"]
            + final["maximum_absolute_excess_kurtosis"]
            < first["maximum_absolute_skewness"]
            + first["maximum_absolute_excess_kurtosis"]
        ),
        "final_skewness_bounded":
            final["maximum_absolute_skewness"]
            <= THRESHOLDS["final_maximum_absolute_skewness"],
        "final_kurtosis_bounded":
            final["maximum_absolute_excess_kurtosis"]
            <= THRESHOLDS["final_maximum_absolute_excess_kurtosis"],
        "final_gaussian_distance_bounded":
            final["sliced_wasserstein_to_gaussian"]
            <= THRESHOLDS["final_sliced_wasserstein_to_gaussian"],
    }


def _anchor_diagnostics(artifact, global_analysis):
    paths = artifact["normal_paths"]
    labels = artifact["anchor_labels"]
    global_kernel = global_analysis["one_step"]
    rows = []
    for label in sorted(set(labels.tolist()), key=str):
        selected = paths[labels == label]
        if len(selected) < 8:
            raise ValueError("each anchor requires at least eight replicas")
        source = selected[:, :-1].reshape(-1, paths.shape[2])
        target = selected[:, 1:].reshape(-1, paths.shape[2])
        estimate = estimate_affine_kernel(source, target)
        rows.append({
            "anchor": str(label),
            "replicas": len(selected),
            "kernel_relative_defect": relative_kernel_residual(
                estimate, global_kernel),
        })
    if len(rows) < 2:
        raise ValueError("at least two anchors are required")
    return {
        "anchor_count": len(rows),
        "anchors": rows,
        "maximum_kernel_relative_defect": max(
            row["kernel_relative_defect"] for row in rows),
    }


def _cross_system_distances(artifacts):
    first_samples = []
    final_samples = []
    for artifact in artifacts:
        first_samples.append(block_fluctuation_samples(
            artifact["normal_paths"], BLOCK_SIZES[0])["samples"])
        final_samples.append(block_fluctuation_samples(
            artifact["normal_paths"], BLOCK_SIZES[-1])["samples"])
    rows = []
    for left in range(len(artifacts)):
        for right in range(left + 1, len(artifacts)):
            rows.append({
                "left_sha256": artifacts[left]["sha256"],
                "right_sha256": artifacts[right]["sha256"],
                "scale_1": sliced_wasserstein_distance(
                    first_samples[left], first_samples[right],
                    directions=96, seed=5100 + 100 * left + right),
                "scale_32": sliced_wasserstein_distance(
                    final_samples[left], final_samples[right],
                    directions=96, seed=6100 + 100 * left + right),
            })
    return rows


def analyze(paths, mode):
    if mode not in ("kernel-flow", "universality"):
        raise ValueError("mode must be kernel-flow or universality")
    artifacts = [load_trajectory_artifact(path) for path in paths]
    if len(artifacts) < 2:
        raise ValueError("at least two trajectory artifacts are required")
    identities = {
        (row["model_width"], row["model_depth"], row["seed"])
        for row in artifacts
    }
    if len(identities) != len(artifacts):
        raise ValueError("trajectory architecture and seed identities repeat")

    systems = []
    for artifact in artifacts:
        result = analyze_trajectory_ensemble(
            artifact["normal_paths"],
            artifact["hidden_labels"],
            BLOCK_SIZES,
        )
        result["anchor_diagnostics"] = _anchor_diagnostics(artifact, result)
        conditions = _system_conditions(result)
        systems.append({
            "artifact": {
                "path": str(artifact["path"]),
                "sha256": artifact["sha256"],
                "source_state_sha256": artifact["source_state_sha256"],
                "anchor_count": len(set(
                    artifact["anchor_labels"].tolist())),
                "model_width": artifact["model_width"],
                "model_depth": artifact["model_depth"],
                "seed": artifact["seed"],
            },
            "analysis": result,
            "conditions": conditions,
        })

    cross = _cross_system_distances(artifacts)
    architecture_seeds = {}
    for artifact in artifacts:
        cell = (artifact["model_width"], artifact["model_depth"])
        architecture_seeds.setdefault(cell, set()).add(artifact["seed"])
    kernel_conditions = {
        "at_least_two_systems": len(systems) >= 2,
        "every_system_kernel_and_flow_qualified": all(
            all(value for name, value in system["conditions"].items()
                if name not in {
                    "higher_cumulants_contract",
                    "final_skewness_bounded",
                    "final_kurtosis_bounded",
                    "final_gaussian_distance_bounded",
                })
            for system in systems
        ),
    }
    universality_conditions = {
        "three_architecture_cells": len(architecture_seeds) >= 3,
        "two_seeds_per_cell": (
            len(architecture_seeds) >= 3
            and all(len(seeds) >= 2 for seeds in architecture_seeds.values())
        ),
        "every_system_fixed_point_qualified": all(
            system["conditions"]["higher_cumulants_contract"]
            and system["conditions"]["final_skewness_bounded"]
            and system["conditions"]["final_kurtosis_bounded"]
            and system["conditions"]["final_gaussian_distance_bounded"]
            for system in systems
        ),
        "cross_system_fixed_point_bounded": (
            bool(cross)
            and max(row["scale_32"] for row in cross)
            <= THRESHOLDS["cross_system_final_sliced_wasserstein"]
        ),
        "cross_system_distance_contracts": (
            bool(cross)
            and sum(row["scale_32"] for row in cross)
            < sum(row["scale_1"] for row in cross)
        ),
    }
    required = dict(kernel_conditions)
    if mode == "universality":
        required.update(universality_conditions)
    decision = (
        "CONFIRMED_ON_REGISTERED_DOMAIN"
        if all(required.values()) else "NOT_CONFIRMED"
    )
    record = {
        "schema_version": "pldr-rg-universality-analysis-v1",
        "mode": mode,
        "block_sizes": list(BLOCK_SIZES),
        "thresholds": THRESHOLDS,
        "systems": systems,
        "cross_system_distances": cross,
        "kernel_flow_conditions": kernel_conditions,
        "universality_conditions": universality_conditions,
        "required_conditions": required,
        "decision": decision,
    }
    record = _jsonable(record)
    unsigned = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(unsigned).hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("kernel-flow", "universality"), required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = analyze(arguments.inputs, arguments.mode)
    path = Path(arguments.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result["decision"] != "CONFIRMED_ON_REGISTERED_DOMAIN":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
