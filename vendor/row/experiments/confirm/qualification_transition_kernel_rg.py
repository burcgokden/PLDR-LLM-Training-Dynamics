#!/usr/bin/env python3
"""Algebraic and statistical qualification for transition-kernel RG tools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from transition_kernel_rg import (
    analyze_trajectory_ensemble,
    block_affine_gaussian,
    gaussian_block_semigroup,
    ou_rg_flow,
    principal_ou_embedding,
    relative_kernel_residual,
    stationary_gaussian,
    whiten_stationary_kernel,
)


def _fixture():
    matrix = np.array([[0.75, 0.0], [0.0, 0.5]])
    bias = np.array([0.125, -0.25])
    stationary_covariance = np.array([[2.0, 0.0], [0.0, 0.5]])
    innovation_covariance = (
        stationary_covariance
        - matrix @ stationary_covariance @ matrix.T
    )
    return matrix, bias, innovation_covariance


def _sample_fixture(seed=31101, replicas=192, steps=64):
    generator = np.random.default_rng(seed)
    matrix = np.array([[0.75, 0.0], [0.0, 0.5]])
    covariance = np.eye(2) - matrix @ matrix.T
    noise_factor = np.linalg.cholesky(covariance)
    paths = np.empty((replicas, steps + 1, 2))
    paths[:, 0] = generator.normal(size=(replicas, 2))
    for step in range(steps):
        innovations = generator.normal(size=(replicas, 2)) @ noise_factor.T
        paths[:, step + 1] = paths[:, step] @ matrix.T + innovations
    labels = np.broadcast_to(
        (np.arange(replicas) % 2)[:, None], paths.shape[:2]).copy()
    return paths, labels


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


def qualify():
    started = time.perf_counter()
    matrix, bias, covariance = _fixture()
    semigroup = gaussian_block_semigroup(matrix, bias, covariance, 2, 4)
    direct = block_affine_gaussian(matrix, bias, covariance, 8)
    stationary = stationary_gaussian(matrix, bias, covariance)
    whitened = whiten_stationary_kernel(matrix, bias, covariance)
    embedding = principal_ou_embedding(whitened["matrix_whitened"])
    flow = ou_rg_flow(embedding["generator"], (1, 2, 4, 8))
    empirical = analyze_trajectory_ensemble(
        *_sample_fixture(), block_sizes=(1, 2, 4, 8, 16))
    memory = [row["memory_operator_norm"] for row in flow]
    conditions = {
        "exact_block_semigroup": semigroup["relative_residual"] < 1e-14,
        "stationary_mean_solved": stationary["mean_residual"] < 1e-14,
        "stationary_covariance_solved":
            stationary["covariance_residual"] < 1e-14,
        "whitening_identity": whitened[
            "stationary_covariance_identity_residual"] < 1e-14,
        "principal_flow_embeds":
            embedding["reconstruction_residual"] < 1e-12,
        "flow_is_gapped": embedding["stability_edge"] > 0.0,
        "memory_decreases": all(
            later < earlier for earlier, later in zip(memory, memory[1:])),
        "empirical_pipeline_complete":
            len(empirical["levels"]) == 5
            and len(empirical["projection_closure"]["groups"]) == 2,
        "stationarity_split_detected":
            empirical["stationarity"]["standardized_mean_defect"] < 0.1
            and empirical["stationarity"][
                "relative_covariance_defect"] < 0.1,
        "innovation_dependence_pipeline":
            empirical["innovation_dependence"][
                "maximum_cross_covariance_frobenius"] < 0.1
            and empirical["innovation_dependence"][
                "previous_energy_conditional_sliced_wasserstein"] < 0.1,
        "empirical_flow_identified":
            empirical["continuous_flow"]["decision"] == "EMBEDDABLE",
        "direct_level_matches": relative_kernel_residual(
            direct, semigroup["direct"]) < 1e-14,
    }
    record = {
        "schema_version": "pldr-transition-kernel-rg-q0-v1",
        "fixture": {
            "matrix": matrix,
            "bias": bias,
            "innovation_covariance": covariance,
        },
        "exact_semigroup": semigroup,
        "stationary": stationary,
        "embedding": embedding,
        "flow": flow,
        "empirical_summary": empirical,
        "resource_profile": {
            "device": "cpu",
            "wall_clock_seconds": time.perf_counter() - started,
            "replicas": 192,
            "steps_per_replica": 64,
        },
        "conditions": conditions,
        "decision": "QUALIFIED" if all(conditions.values()) else "REJECTED",
    }
    record = _jsonable(record)
    unsigned = json.dumps(
        record, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    record["record_sha256"] = hashlib.sha256(unsigned).hexdigest()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    arguments = parser.parse_args()
    result = qualify()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        path = Path(arguments.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if result["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
