"""Regression tests for the rev36 row-map confirmation analyzers."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.analyze_row_map_confirmation import analyze_blocks
from confirm.gate_shape import adamw_gate_block
from confirm.gate_shape_evidence import write_npz_atomic


def _qualification_record(device):
    dimension = 64
    gamma = np.linspace(-0.2, 0.2, dimension)
    first = np.linspace(0.01, 0.02, dimension)
    second = np.linspace(0.2, 0.3, dimension)
    gradient = np.linspace(-0.2, 0.2, dimension)
    fisher = 0.2 * np.eye(dimension)
    signed = 0.05 * np.eye(dimension)
    true_hessian = fisher + signed
    optimizer = {
        "learning_rate": 7.5e-4,
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-5,
        "optimizer_step": 24001,
        "weight_decay": 0.1,
        "clip_value": 1.0,
    }
    operator = adamw_gate_block(
        true_hessian, gamma, first, second, gradient, **optimizer)["operator"]
    return {
        "schema_version": np.asarray("pldr-row-map-live-v1"),
        "stage": np.asarray("Q"),
        "checkpoint_sha256": np.asarray("a" * 64),
        "run_name": np.asarray("w8-const-lr7.5e-4"),
        "seed": np.asarray(1234, dtype=np.int64),
        "device": np.asarray(device),
        "step": np.asarray(24000, dtype=np.int64),
        "layer": np.asarray(2, dtype=np.int64),
        "gamma": gamma,
        "first_moment": first,
        "second_moment": second,
        "raw_gradient": gradient,
        "true_hessian": true_hessian,
        "learning_rate": np.asarray(optimizer["learning_rate"]),
        "beta1": np.asarray(optimizer["beta1"]),
        "beta2": np.asarray(optimizer["beta2"]),
        "adam_epsilon": np.asarray(optimizer["epsilon"]),
        "weight_decay": np.asarray(optimizer["weight_decay"]),
        "clip_value": np.asarray(optimizer["clip_value"]),
        "optimizer_step_after": np.asarray(
            optimizer["optimizer_step"], dtype=np.int64),
        "analytic_operator": operator,
        "autodiff_operator": operator,
        "source_state_scale": np.ones(3 * dimension),
        "target_state_scale": np.ones(3 * dimension),
        "clipping_boundary_distance": np.asarray(0.8),
        "loss_value": np.asarray(4.25),
        "gradient_at_gate": gradient,
        "fisher": fisher,
        "sector_true_hessian": true_hessian,
        "signed_second_jet": signed,
        "force_at_zero": np.linspace(-0.1, 0.1, dimension),
        "loss_terms": np.asarray([4.25]),
        "gradient_at_gate_terms": gradient[None],
        "fisher_terms": fisher[None],
        "true_hessian_terms": true_hessian[None],
        "force_at_zero_terms": np.linspace(
            -0.1, 0.1, dimension)[None],
        "source_count_terms": np.asarray([255.0]),
    }


def _write_pair(tmp_path):
    left = tmp_path / "qualification-cuda-0.npz"
    right = tmp_path / "qualification-cuda-1.npz"
    write_npz_atomic(left, **_qualification_record("cuda:0"))
    write_npz_atomic(right, **_qualification_record("cuda:1"))
    return left, right


def test_block_analyzer_requires_and_replays_complete_two_device_q(tmp_path):
    left, right = _write_pair(tmp_path)
    report = analyze_blocks([left, right])
    assert all(report["checks"].values())
    qualification = report["results"]["qualification"]
    assert qualification["record_count"] == 2
    assert qualification["devices"] == ["cuda:0", "cuda:1"]
    assert qualification["maximum_cross_device_relative_residual"] == 0.0
    assert all(row["operator_dimension"] == 192
               for row in report["results"]["blocks"])


def test_block_analyzer_detects_a_device_specific_signed_jet_mutation(tmp_path):
    left, right = _write_pair(tmp_path)
    mutated = _qualification_record("cuda:1")
    mutated["signed_second_jet"] = mutated["signed_second_jet"].copy()
    mutated["signed_second_jet"][0, 0] += 0.1
    write_npz_atomic(right, **mutated)
    report = analyze_blocks([left, right])
    assert not report["checks"]["qualification_fisher_plus_signed_jet"]
    assert not report["checks"]["qualification_devices_agree"]


def test_block_analyzer_rejects_incomplete_qualification_schema(tmp_path):
    left, _right = _write_pair(tmp_path)
    raw = _qualification_record("cuda:1")
    raw.pop("force_at_zero")
    incomplete = tmp_path / "incomplete.npz"
    write_npz_atomic(incomplete, **raw)
    with pytest.raises(ValueError, match="qualification block omits fields"):
        analyze_blocks([left, incomplete])
