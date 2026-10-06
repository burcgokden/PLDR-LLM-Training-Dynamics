"""Regression tests for block-normal row-map confirmation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.analyze_block_normal_confirmation import analyze_records
from confirm.block_normal import (
    backward_metrics,
    block_prediction,
    classify_coverage,
)
from confirm.block_normal_specs import (
    BLOCK_RECORD_SCHEMA,
    CAMPAIGN_ID,
    REGISTRY,
    SCIENTIFIC_OUTCOMES,
    campaign_design,
    validate_design,
)
from confirm.checkpoint_retention import (
    in_memory_recovery_check,
    retention_record,
)
from confirm.evidence_seal import (
    seal_native,
    seal_source,
    write_new_json,
)
from confirm.finite_transport import (
    affine_increment,
    centered_energy,
    chronological_unroll,
    exact_energy_balance,
    named_transport_ledger,
    plga_multiplied_fields,
)
from confirm.native_determinism import native_and_conversion_report
from confirm.normal_projection import (
    metric_projection,
    projection_valid,
    quadratic_defect_profile,
    sign_paired_perturbations,
)
from scripts.freeze_block_normal_foundation import (
    ORDER_NAMES,
    build as build_foundation,
    validate as validate_foundation,
)
from scripts.run_block_normal_qualification import run as run_qualification
from scripts.stage_block_normal_confirmation import _stage_node


def test_campaign_counts_and_hard_budget_are_internal():
    design = campaign_design()
    validate_design(design)
    assert REGISTRY["map_count"] == 288
    assert REGISTRY["pairs_per_map"] == 2016
    assert REGISTRY["within_map_pair_count"] == 580_608
    assert sum(
        row["aggregate_gpu_hours_target"] for row in design["stages"]
    ) == pytest.approx(8.0)


def test_q0_is_explicitly_fixture_only():
    design = campaign_design()
    q0 = next(stage for stage in design["stages"] if stage["id"] == "Q0")
    assert q0["purpose"] == (
        "deterministic-kernel-schema-executor-recovery-fixtures"
    )
    report = run_qualification("cpu")
    assert report["qualification_mode"] == "deterministic-fixture"
    assert not report["checkpoint_backed"]
    assert not report["scientific_confirmation"]
    assert report["status"] == "passed"
    assert set(report["fixture_outcome_counts"]) == set(SCIENTIFIC_OUTCOMES)
    assert "all_classification_branches_fixture" in report["conditions"]
    assert "complete_state_recovery_fixture" in report["conditions"]
    assert "resource_executor_fixture_record" in report["conditions"]


def test_registration_foundation_is_digest_bound(tmp_path: Path):
    registry_path = tmp_path / "input-registry.json"
    order_dir = tmp_path / "input-orders"
    order_dir.mkdir()
    registry_path.write_text(json.dumps({
        "dataset_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "map_count": 288,
        "within_map_pair_count": 580_608,
        "construction": [{"id": index} for index in range(8)],
        "validation": [{"id": index} for index in range(16)],
    }), encoding="utf-8")
    for index, name in enumerate(ORDER_NAMES):
        (order_dir / name).write_bytes(bytes([index]) * 17)
    output = tmp_path / "foundation"
    build_foundation(registry_path, order_dir, output)
    validate_foundation(output)
    (output / "orders" / ORDER_NAMES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="file changed"):
        validate_foundation(output)


def test_assembly_nodes_are_cpu_accounted(tmp_path: Path):
    node = _stage_node(tmp_path, "python3", "Q1", ["q0-cpu"])
    assert node["device"] == "cpu"
    assert node["caps"]["gpu_reserved_bytes"] == 0


def test_finite_endpoint_transport_and_centered_energy_are_exact():
    rng = np.random.default_rng(44010)
    w0 = rng.normal(size=(4, 3))
    w1 = w0 + rng.normal(scale=0.1, size=w0.shape)
    x0 = rng.normal(size=3)
    x1 = x0 + rng.normal(scale=0.1, size=x0.shape)
    b0 = rng.normal(size=4)
    b1 = b0 + rng.normal(scale=0.1, size=b0.shape)
    affine = affine_increment(w0, w1, x0, x1, b0, b1)
    assert affine["maximum_abs_residual"] < 2.0e-15

    maps = [rng.normal(size=(3, 3)) for _ in range(5)]
    sources = [rng.normal(size=3) for _ in range(5)]
    unrolled = chronological_unroll(maps, sources, rng.normal(size=3))
    assert unrolled["maximum_abs_residual"] < 2.0e-13

    rows0 = rng.normal(size=(4, 7, 5))
    contributions = {
        "upstream": rng.normal(scale=0.01, size=rows0.shape),
        "parameters": rng.normal(scale=0.01, size=rows0.shape),
        "plga": rng.normal(scale=0.01, size=rows0.shape),
    }
    rows1 = rows0 + sum(contributions.values(), start=np.zeros_like(rows0))
    ledger = named_transport_ledger(rows0, rows1, contributions)
    balance = exact_energy_balance(rows0, rows1)
    geometry = centered_energy(rows0)
    assert ledger["maximum_abs_transport_residual"] < 5.0e-16
    assert balance["maximum_abs_residual"] < 2.0e-13
    assert np.max(np.abs(geometry["pair_identity_residual"])) < 2.0e-12


def test_plga_uses_true_diagonal_derivatives_and_both_orders():
    z0 = np.asarray([-1.0, 0.0, 2.0, 3.0])
    z1 = np.asarray([1.0, 0.0, 2.5, 3.0])
    p0 = np.asarray([0.8, -0.5, 1.2, 2.0])
    p1 = np.asarray([1.1, -0.5, 0.7, 2.0])
    result = plga_multiplied_fields(z0, z1, p0, p1)
    assert result["maximum_abs_route_one_residual"] < 3.0e-15
    assert result["maximum_abs_route_two_residual"] < 3.0e-15
    assert np.isfinite(result["route_z_then_p"]).all()
    assert np.isfinite(result["route_p_then_z"]).all()


def test_metric_projection_reports_primal_kkt_rank_and_metric():
    result = metric_projection(
        [1.0, 2.0, -1.0, 0.5],
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
        [0.0, 0.0],
        np.diag([1.0, 2.0, 3.0, 4.0]),
    )
    assert projection_valid(
        result, absolute_tolerance=1.0e-12, required_rank=2
    )
    with pytest.raises(ValueError, match="positive definite"):
        metric_projection([1.0], [[1.0]], [0.0], [[0.0]])


def test_quadratic_defect_and_ordered_forcing_are_not_averaged():
    operator = np.asarray([[0.8, 0.1], [0.0, 0.7]])
    force = np.asarray([0.01, -0.01])
    coefficient = 0.03

    def function(value):
        value = np.asarray(value)
        return operator @ value + force + coefficient * value * value

    probes = sign_paired_perturbations([1.0, -1.0], 0.1)
    profile = quadratic_defect_profile(
        function, np.zeros(2), operator, force, probes
    )
    assert np.all(
        profile["defect_norms"]
        <= coefficient * profile["amplitudes"] ** 2 + 1.0e-15
    )
    metric = backward_metrics(
        [operator, operator * 1.1, operator * 0.9], np.eye(2)
    )
    prediction = block_prediction(
        metric["gains"], [0.03, 0.03, 0.03],
        [0.01, 0.02, 0.01], [1.0, 1.0, 1.0, 1.0], 0.2,
    )
    assert prediction["tube_valid"]
    assert np.all(metric["lyapunov_residuals"] <= 1.0e-12)
    first_manual = (
        prediction["effective_gains"][0] * 0.2 + 0.01
    )
    second_manual = (
        prediction["effective_gains"][1] * first_manual + 0.02
    )
    assert prediction["envelope"][2] == pytest.approx(second_manual)
    assert prediction["force_contributions"][2, 0] == pytest.approx(
        prediction["effective_gains"][1] * 0.01
    )
    with pytest.raises(ValueError, match="terminal metric"):
        backward_metrics([operator], [[1.0, 0.0], [0.0, 0.0]])


def test_source_and_native_evidence_are_sealed_in_order(tmp_path: Path):
    source_checkpoint = tmp_path / "source.pt"
    native_checkpoint = tmp_path / "native.pt"
    source_checkpoint.write_bytes(b"source")
    native_checkpoint.write_bytes(b"native")
    specification = {
        "stage": "Q1",
        "trajectory": "construction-seed7444",
        "seed": 7444,
        "block_start": 1000,
        "block_end": 1001,
        "operators": [[[0.8, 0.0], [0.0, 0.7]]],
        "quadratic_coefficients": [0.02],
        "forces": [0.01],
        "radii": [1.0, 1.0],
        "initial_normal_upper": 0.1,
        "comparison_bounds": [{
            "comparison_id": "map-0-0-0",
            "prediction": "physical_collapse",
            "map_id": "map-0-0-0",
            "context": "context-0",
            "layer": 0,
            "head": 0,
            "bound_lower": 0.4,
            "bound_upper": 0.5,
        }],
    }
    source = seal_source(
        specification, source_checkpoint, created_at_ns=100
    )
    source_path = tmp_path / "edge.source.json"
    write_new_json(source_path, source)
    native = seal_native(
        source_path,
        native_checkpoint,
        [{
            "comparison_id": "map-0-0-0",
            "observed_lower": 0.2,
            "observed_upper": 0.3,
        }],
        opened_at_ns=101,
    )
    native_path = tmp_path / "edge.native.json"
    write_new_json(native_path, native)
    assert native["opened_at_ns"] > source["created_at_ns"]
    assert native["source_checkpoint_sha256"] == (
        source["source_checkpoint_sha256"]
    )
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_new_json(source_path, source)
    with pytest.raises(ValueError, match="after source sealing"):
        seal_native(
            source_path,
            native_checkpoint,
            [{
                "comparison_id": "map-0-0-0",
                "observed_lower": 0.2,
                "observed_upper": 0.3,
            }],
            opened_at_ns=100,
        )


def _record(prediction):
    comparisons = [
        {
            "prediction": "physical_collapse",
            "map_id": "a",
            "context": "c0",
            "layer": 0,
            "head": 0,
            "observed_lower": 0.2,
            "observed_upper": 0.3,
            "bound_lower": 0.4,
            "bound_upper": 0.5,
        },
        {
            "prediction": "physical_collapse",
            "map_id": "b",
            "context": "c1",
            "layer": 1,
            "head": 1,
            "observed_lower": 0.8,
            "observed_upper": 0.9,
            "bound_lower": 0.4,
            "bound_upper": 0.5,
        },
        {
            "prediction": "normal_force_closure",
            "map_id": "c",
            "context": "c2",
            "layer": 2,
            "head": 2,
            "observed_lower": 0.4,
            "observed_upper": 0.6,
            "bound_lower": 0.45,
            "bound_upper": 0.55,
        },
    ]
    return {
        "schema_version": BLOCK_RECORD_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "stage": "Q0",
        "trajectory": "fixture",
        "seed": 44010,
        "block_start": 0,
        "block_end": 2,
        "source_checkpoint_path": "/fixture/source.pt",
        "source_checkpoint_sha256": "a" * 64,
        "prediction_created_at_ns": 1,
        "native_endpoint_opened_at_ns": 2,
        "technical_valid": True,
        "source_values_only": True,
        "gains": prediction["gains"].tolist(),
        "quadratic_coefficients": (
            prediction["quadratic_coefficients"].tolist()
        ),
        "forces": prediction["forces"].tolist(),
        "radii": prediction["radii"].tolist(),
        "initial_normal_upper": 0.1,
        "prefix_products": prediction["prefix_products"].tolist(),
        "force_contributions": (
            prediction["force_contributions"].tolist()
        ),
        "normal_envelope": prediction["envelope"].tolist(),
        "comparisons": comparisons,
    }


def test_all_scientific_outcomes_are_retained_and_do_not_prune():
    prediction = block_prediction(
        [0.7, 0.8], [0.01, 0.01], [0.01, 0.02],
        [1.0, 1.0, 1.0], 0.1,
    )
    report = analyze_records([_record(prediction)])
    assert report["outcome_counts"] == {
        "confirmed": 1,
        "not_confirmed": 1,
        "unresolved": 1,
    }
    assert set(report["outcome_counts"]) == set(SCIENTIFIC_OUTCOMES)
    assert not report["scientific_outcomes_halted_descendants"]
    changed = _record(prediction)
    changed["prefix_products"][1] += 0.1
    with pytest.raises(ValueError, match="does not replay"):
        analyze_records([changed])


def test_native_and_conversion_radii_are_separate():
    rng = np.random.default_rng(44011)
    state = {"x": rng.normal(size=8).astype(np.float32)}
    rows = rng.normal(size=(2, 8, 4)).astype(np.float64)
    report = native_and_conversion_report(
        state, copy.deepcopy(state), rows, rows
    )
    assert report["native"]["bitwise_equal"]
    assert report["native_radius_zero_from_bitwise_equality"]
    assert not report["charges_combined"]
    assert np.any(report["conversion"]["centered_row_radius"] > 0.0)


def test_checkpoint_retention_never_overwrites_or_deletes(tmp_path: Path):
    complete = tmp_path / "complete.pt"
    model_only = tmp_path / "model-only.pt"
    complete.write_bytes(b"complete")
    model_only.write_bytes(b"model")
    state = {
        "model": {"x": np.asarray([1.0])},
        "optimizer": {"x": np.asarray([2.0])},
        "scheduler": {"x": np.asarray([3])},
        "rng": {"x": np.asarray([4])},
    }
    recovery = in_memory_recovery_check(state, copy.deepcopy(state))
    record = retention_record(
        complete,
        model_only,
        open_descendants=["child"],
        recovery_report=recovery,
        maximum_complete_checkpoints=2,
        complete_checkpoint_count=3,
    )
    assert record["recovery_complete"]
    assert not record["cleanup_authorized"]
    assert not record["deletion_performed"]
    with pytest.raises(ValueError, match="must not overwrite"):
        retention_record(
            complete,
            complete,
            open_descendants=[],
            recovery_report=recovery,
            maximum_complete_checkpoints=2,
            complete_checkpoint_count=3,
        )


def test_interval_classification_is_total():
    result = classify_coverage(
        [0.1, 0.7, 0.4],
        [0.2, 0.8, 0.6],
        [0.3, 0.4, 0.45],
        [0.4, 0.5, 0.55],
        tolerance=0.0,
    )
    assert result.tolist() == [
        "confirmed", "not_confirmed", "unresolved"
    ]
