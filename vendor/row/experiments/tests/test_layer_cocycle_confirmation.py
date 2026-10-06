"""Regression tests for the layer-resolved cocycle confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.layer_cocycle import (  # noqa: E402
    arithmetic_resolution,
    central_secant_prediction,
    layer_observables,
    operator_diagnostics,
    placebo_identifiability,
    plga_defect_bound,
    prediction_residual,
    row_preservation_residuals,
)
from confirm.layer_cocycle_live import _intervention_parameters  # noqa: E402
from confirm.layer_cocycle_specs import (  # noqa: E402
    CAMPAIGN_ID,
    GATE_OPERATOR_POLICY,
    INTERVENTION_ARMS,
    PERTURBATION_POLICY,
    campaign_design,
    validate_design,
)
from scripts.gen_layer_cocycle_protocols import (  # noqa: E402
    OUTPUT,
    generated_files,
    record_inventory,
)
from scripts.stage_layer_cocycle_confirmation import (  # noqa: E402
    bound_path,
    default_paths,
    launch_plan,
    source_import_closure,
)


def _minimal_analysis_bundle(root: Path) -> Path:
    record = root / "records" / "precision" / "A-step65536.npz"
    record.parent.mkdir(parents=True)
    float64_coordinate_energy = np.zeros((1, 3, 1, 2))
    float64_coordinate_energy[:, 0, :, :] = 1.0
    np.savez(
        record,
        float32_coordinate_energy=np.zeros((1, 3, 1, 2)),
        float64_coordinate_energy=float64_coordinate_energy,
        resolution_relative_discrepancy=np.full((1, 3, 1), 0.25),
    )
    (root / "protocol").mkdir()
    (root / "reports").mkdir()
    (root / "protocol" / "record-inventory.json").write_text(
        json.dumps({
            "precision": [{
                "trajectory": "A",
                "step": 65_536,
                "path": "records/precision/A-step65536.npz",
            }],
        }),
        encoding="utf-8",
    )
    result = {
        "precision": {"rows": [{
            "trajectory": "A",
            "layer": 0,
            "endpoint_median_energy": 1.0e-8,
            "late_log10_energy_gain": -2.0,
            "late_log10_shape_gain": -1.5,
            "late_log10_effective_gate_gain": -0.5,
            "small_at_endpoint": True,
        }]},
        "gate_operator": {"rows": [{
            "trajectory": "A",
            "layer": 0,
            "scaled_operator_norm": 2.0,
            "scaled_spectral_radius": 1.1,
            "scaled_nonnormality_ratio": 1.8,
            "scaled_invariance_defect": 4.0,
            "smaller_probe_maximum_relative_residual": 1.0e-4,
        }]},
        "intervention": {"rows": [{
            "trajectory": "A",
            "layer": 0,
            "placebo_response": 1.0e-6,
            "gate_update_removed_response": 1.0,
            "upstream_update_removed_response": 2.0,
            "joint_update_removed_response": 2.5,
            "placebo_fraction_of_smallest_treatment": 1.0e-6,
        }]},
        "perturbation": {"rows": [{
            "trajectory": "A",
            "layer": 0,
            "relative_prediction_residual": 0.2,
        }]},
        "plga": {"rows": [{
            "trajectory": "A",
            "layer": 0,
            "median_input_quotient_norm": 1.0e-8,
            "median_segment_response_norm": 0.5,
            "median_row_constant_defect_norm": 3.0,
            "median_output_quotient_norm": 3.5,
        }]},
        "resource": {
            "aggregate_gpu_device_hours": 0.5,
            "maximum_process_reserved_bytes": 2**30,
        },
    }
    (root / "reports" / "final-analysis.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    (root / "reports" / "EVIDENCE.sha256").write_text(
        "0" * 64 + "  records/example.npz\n", encoding="ascii"
    )
    (root / "protocol" / "source_binding.json").write_text(
        json.dumps({"git_commit": "1" * 40}), encoding="utf-8"
    )
    return root


def test_layer_external_paths_derive_from_explicit_binding_root(
    tmp_path: Path,
) -> None:
    binding_root = tmp_path.resolve()
    paths = default_paths(binding_root)
    assert all(path.is_relative_to(binding_root) for path in paths.values())
    assert paths["output"].name == "layer-resolved-cocycle-confirmation"
    assert paths["input_bundle"].parent.name == "mixed-collapse"
    assert paths["tokens"].parent.name == "refinedweb-538m-prefix-locked"


def test_layer_external_path_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes binding root"):
        bound_path(tmp_path.parent / "outside", tmp_path, "probe")


def test_design_fixes_population_tolerances_and_resource_stops() -> None:
    design = campaign_design()
    validate_design(design)
    assert design["campaign_id"] == CAMPAIGN_ID
    assert design["architecture"]["registered_maps"] == 288
    assert len(design["trajectories"]) == 3
    assert GATE_OPERATOR_POLICY["hessian_construction"] == (
        "weighted-basiswise-forward-over-reverse"
    )
    assert PERTURBATION_POLICY["maximum_validation_relative_residual"] == 0.25
    assert design["resource_budget"]["hard_aggregate_gpu_hours"] == 3.0


def test_inventory_and_launch_graph_are_population_complete() -> None:
    inventory = record_inventory()
    assert len(inventory["precision"]) == 6
    assert len(inventory["gate_operator"]) == 9
    assert len(inventory["intervention"]) == 9
    assert len(inventory["perturbation"]) == 9
    assert len(inventory["plga"]) == 3
    paths = []
    for category in ("precision", "gate_operator", "intervention", "plga"):
        paths.extend(item["path"] for item in inventory[category])
    for item in inventory["perturbation"]:
        paths.extend([item["prediction_path"], item["validation_path"]])
    assert len(paths) == len(set(paths)) == 45
    plan = launch_plan()
    assert len(plan["nodes"]) == 37
    assert plan["scientific_outcomes_control_execution"] is False
    position = {node["id"]: index for index, node in enumerate(plan["nodes"])}
    assert all(
        position[dependency] < position[node["id"]]
        for node in plan["nodes"]
        for dependency in node["depends_on"]
    )
    assert len(plan["nodes"][-1]["depends_on"]) == 36


def test_generated_protocols_and_checksum_file_are_current() -> None:
    files = generated_files()
    assert set(path.name for path in files) >= {
        "campaign-design.json",
        "record-inventory.json",
        "CHECKSUMS.sha256",
    }
    for relative, payload in files.items():
        assert (OUTPUT / relative).read_bytes() == payload
    lines = files[Path("CHECKSUMS.sha256")].decode("ascii").splitlines()
    for line in lines:
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256(files[Path(relative)]).hexdigest() == digest


def test_gate_shape_observable_factorization_is_exact_and_fail_capable() -> None:
    generator = np.random.default_rng(48)
    shape = generator.normal(size=(2, 3, 4, 5, 6))
    gates = generator.normal(size=(3, 6))
    physical = shape * gates[None, :, None, None, :]
    result = layer_observables(shape, physical, gates)
    assert result["factorization_relative_residual"] < 1.0e-14
    np.testing.assert_allclose(
        result["energy"],
        result["shape_energy"] * result["effective_squared_gate"],
    )
    broken = physical.copy()
    broken[0, 0, 0, 0, 0] += 1.0
    assert layer_observables(shape, broken, gates)[
        "factorization_relative_residual"
    ] > 0.0


def test_arithmetic_resolution_censors_below_the_observed_floor() -> None:
    high = np.asarray([1.0, 1.0e-10, 0.0])
    low = np.asarray([1.0 + 1.0e-8, 0.0, 1.0e-12])
    result = arithmetic_resolution(
        high, low, safety_factor=8.0, denominator_floor=1.0e-30
    )
    assert result["relative_resolved"].tolist() == [True, False, False]
    with pytest.raises(ValueError, match="resolution policy"):
        arithmetic_resolution(high, low, safety_factor=0.5, denominator_floor=1.0)


def test_scaled_operator_reports_nonnormal_gain_and_forcing() -> None:
    operator = np.asarray([[0.5, 3.0], [0.0, 0.5]])
    result = operator_diagnostics(
        operator, np.ones(2), np.ones(2), np.asarray([2.0, 0.0])
    )
    assert result["scaled_operator_norm"] > 1.0
    assert result["scaled_spectral_radius"] == pytest.approx(0.5)
    assert result["scaled_nonnormality_ratio"] > 2.0
    assert result["scaled_invariance_defect"] == pytest.approx(2.0)


def test_central_secant_prediction_has_a_fail_capable_validation_residual() -> None:
    native = np.zeros((2, 3, 3))
    plus = np.ones_like(native)
    minus = -np.ones_like(native)
    prediction = central_secant_prediction(native, plus, minus, 0.5)
    exact = prediction_residual(prediction, prediction, native)
    assert exact["relative_residual"] == 0.0
    perturbed = prediction.copy()
    perturbed[:, 0, 0] += 0.2
    assert prediction_residual(prediction, perturbed, native)[
        "relative_residual"
    ] > 0.0
    with pytest.raises(ValueError, match="strictly"):
        central_secant_prediction(native, plus, minus, 1.0)


def test_placebo_identifiability_passes_and_fails_by_trajectory_layer() -> None:
    placebo = np.full((3, 4), 0.01)
    treatments = np.full((3, 3, 4), 1.0)
    passed = placebo_identifiability(
        placebo, treatments, maximum_fraction=0.1, effect_floor=1.0e-12
    )
    assert passed["all_layers_identified"] is True
    placebo[1] = 0.2
    failed = placebo_identifiability(
        placebo, treatments, maximum_fraction=0.1, effect_floor=1.0e-12
    )
    assert failed["identified"].tolist() == [True, False, True]


def test_plga_sufficient_conditions_can_hold_or_fail() -> None:
    dimension = 4
    ones = np.ones((dimension, dimension))
    weight = np.eye(dimension)
    coupling = np.eye(dimension)
    exponent = 2.0 * ones
    bias = 3.0 * ones
    coupling_bias = -ones
    result = row_preservation_residuals(
        weight, bias, exponent, coupling, coupling_bias
    )
    assert all(value == pytest.approx(0.0) for value in result.values())
    weight[0, 0] += 1.0
    assert row_preservation_residuals(
        weight, bias, exponent, coupling, coupling_bias
    )["weight_ones_quotient"] > 0.0


def test_plga_quotient_defect_bound_checks_signed_slack() -> None:
    result = plga_defect_bound(
        np.asarray([2.0]),
        np.asarray([4.0]),
        np.asarray([1.0]),
        np.asarray([2.0]),
    )
    assert result["upper"].item() == pytest.approx(5.0)
    assert result["signed_slack"].item() == pytest.approx(1.0)
    assert result["covered"].item()
    with pytest.raises(ValueError, match="nonnegative"):
        plga_defect_bound(1.0, -1.0, 0.0, 1.0)


def test_intervention_parameter_selector_changes_only_the_selected_layer() -> None:
    selected_gate = "decoder.dec_layers.1.mha1.reslayerAs.7.layernormA.weight"
    other_gate = "decoder.dec_layers.0.mha1.reslayerAs.7.layernormA.weight"
    selected_upstream = "decoder.dec_layers.1.mha1.reslayerAs.0.weight"
    unrelated = "decoder.dec_layers.2.mha1.reslayerAs.0.weight"
    source = {
        selected_gate: torch.asarray([1.0]),
        other_gate: torch.asarray([2.0]),
        selected_upstream: torch.asarray([3.0]),
        unrelated: torch.asarray([4.0]),
    }
    native = {name: value + 0.5 for name, value in source.items()}
    result = _intervention_parameters(
        source,
        native,
        "joint_row_program_update_removed",
        layer=1,
        learning_rate=1.0e-3,
        weight_decay=0.1,
    )
    torch.testing.assert_close(result[selected_gate], source[selected_gate])
    torch.testing.assert_close(result[selected_upstream], source[selected_upstream])
    torch.testing.assert_close(result[other_gate], native[other_gate])
    torch.testing.assert_close(result[unrelated], native[unrelated])
    assert tuple(INTERVENTION_ARMS)[0] == "native"


def test_source_snapshot_closure_contains_all_campaign_entrypoints() -> None:
    relative = {path.relative_to(ROOT).as_posix() for path in source_import_closure()}
    assert {
        "experiments/confirm/layer_cocycle.py",
        "experiments/confirm/layer_cocycle_live.py",
        "experiments/confirm/layer_cocycle_specs.py",
        "experiments/analysis/analyze_layer_cocycle_confirmation.py",
        "scripts/execute_layer_cocycle_plan.py",
    } <= relative




