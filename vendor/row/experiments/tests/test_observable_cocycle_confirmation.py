"""Regression tests for the observable complete-state cocycle campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from analysis.analyze_observable_cocycle_confirmation import _direction_summary, _numerical_health
from confirm.observable_cocycle import (  # noqa: E402
    calibration_cell,
    centered_rows,
    directional_gain,
    relative_response_residual,
    select_layer_horizon,
)
from confirm.observable_cocycle_specs import (  # noqa: E402
    CAMPAIGN_ID,
    DECISION_POLICY,
    DIRECTION_CLASSES,
    HORIZON_GRID,
    RESOURCE_BUDGET,
    SOURCE_STEPS,
    campaign_design,
    validate_design,
)
from scripts.gen_observable_cocycle_protocols import (  # noqa: E402
    generated_files,
    record_inventory,
)
from scripts.stage_observable_cocycle_confirmation import (  # noqa: E402
    launch_plan,
    source_import_closure,
)


def test_design_fixes_full_state_scope_and_inference_limits() -> None:
    design = campaign_design()
    validate_design(design)
    assert design["campaign_id"] == CAMPAIGN_ID
    assert design["architecture"]["continuous_state_dimension"] == 61_868_742
    assert design["architecture"]["continuous_state_dimension"] == (
        3 * design["architecture"]["parameter_count"]
    )
    assert tuple(design["source_steps"]) == SOURCE_STEPS
    assert tuple(design["horizon_grid"]) == HORIZON_GRID
    assert DECISION_POLICY["operator_contraction_from_sampled_directions"] is False
    assert DECISION_POLICY["sampled_direction_implies_operator_norm"] is False
    assert DECISION_POLICY["finite_horizon_implies_asymptotic_collapse"] is False
    assert RESOURCE_BUDGET["hard_aggregate_reserved_device_hours"] == 3.0


def test_inventory_and_dag_enforce_construction_before_heldout() -> None:
    inventory = record_inventory()
    assert len(inventory["calibration"]) == 9
    assert len(inventory["prediction"]) == 18
    assert len(inventory["validation"]) == 18
    assert len(inventory["stratum"]) == 9
    plan = launch_plan()
    assert len(plan["nodes"]) == 57
    assert plan["scientific_outcomes_control_execution"] is False
    by_id = {node["id"]: node for node in plan["nodes"]}
    assert len(by_id) == len(plan["nodes"])
    positions = {node["id"]: index for index, node in enumerate(plan["nodes"])}
    assert all(
        positions[dependency] < positions[node["id"]]
        for node in plan["nodes"]
        for dependency in node["depends_on"]
    )
    assert len(by_id["construction-lock"]["depends_on"]) == 9
    assert all(
        node["depends_on"] == ["construction-lock"]
        for node in plan["nodes"]
        if node["stage"] == "prediction"
    )
    predictions = {
        node["id"]: node for node in plan["nodes"] if node["stage"] == "prediction"
    }
    assert len(predictions) == 18
    for node in plan["nodes"]:
        if node["stage"] == "validation":
            assert len(node["depends_on"]) == 1
            assert node["depends_on"][0] in predictions


def test_generated_protocol_checksums_are_self_consistent() -> None:
    files = generated_files()
    required = {
        "campaign-design.json",
        "record-inventory.json",
        "qualification.schema.json",
        "calibration.schema.json",
        "lock.schema.json",
        "prediction.schema.json",
        "validation.schema.json",
        "stratum.schema.json",
        "CHECKSUMS.sha256",
    }
    assert required <= {path.name for path in files}
    for line in files[Path("CHECKSUMS.sha256")].decode("ascii").splitlines():
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256(files[Path(relative)]).hexdigest() == digest
    design = json.loads(files[Path("campaign-design.json")])
    digest = design.pop("design_sha256")
    compact = json.dumps(
        design, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert hashlib.sha256(compact).hexdigest() == digest


def test_source_snapshot_closure_contains_all_campaign_entrypoints() -> None:
    relative = {
        path.relative_to(ROOT).as_posix() for path in source_import_closure()
    }
    assert {
        "experiments/confirm/observable_cocycle.py",
        "experiments/confirm/observable_cocycle_live.py",
        "experiments/confirm/observable_cocycle_specs.py",
        "experiments/analysis/analyze_observable_cocycle_confirmation.py",
        "scripts/execute_observable_cocycle_plan.py",
    } <= relative


def test_row_projection_residual_gain_and_zero_policy() -> None:
    value = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    projected = centered_rows(value)
    np.testing.assert_allclose(np.mean(projected, axis=-2), 0.0, atol=1.0e-15)
    residual = relative_response_residual(
        projected, projected.copy(), effect_floor=1.0e-10
    )
    assert residual["relative_residual"] == 0.0
    assert residual["effect_resolved"]
    unresolved = relative_response_residual(
        np.zeros(3), np.zeros(3), effect_floor=1.0e-8
    )
    assert not unresolved["effect_resolved"]
    gain = directional_gain(
        np.asarray([3.0, 4.0]),
        np.asarray([1.5, 2.0]),
        effect_floor=1.0e-8,
    )
    assert gain["visible_directional_gain"] == pytest.approx(0.5)
    assert gain["directionally_attenuating"]


def test_calibration_gate_requires_resolution_convergence_and_placebo() -> None:
    good = calibration_cell(
        larger_relative_residual=0.2,
        smaller_relative_residual=0.1,
        smaller_response_norm=1.0,
        placebo_norm=0.01,
    )
    assert good["qualified"]
    assert not calibration_cell(
        larger_relative_residual=0.1,
        smaller_relative_residual=0.1,
        smaller_response_norm=1.0,
        placebo_norm=0.01,
    )["qualified"]
    assert not calibration_cell(
        larger_relative_residual=0.4,
        smaller_relative_residual=0.3,
        smaller_response_norm=1.0,
        placebo_norm=0.01,
    )["qualified"]
    assert not calibration_cell(
        larger_relative_residual=0.2,
        smaller_relative_residual=0.1,
        smaller_response_norm=1.0,
        placebo_norm=0.2,
    )["qualified"]


def _selection_rows(qualified_horizons: set[int]) -> list[dict[str, object]]:
    return [
        {
            "source_step": step,
            "direction": direction,
            "horizon": horizon,
            "qualified": horizon in qualified_horizons,
        }
        for step in SOURCE_STEPS
        for direction in DIRECTION_CLASSES
        for horizon in HORIZON_GRID
    ]


def test_horizon_lock_uses_longest_jointly_qualified_and_explicit_fallback() -> None:
    selected = select_layer_horizon(_selection_rows({1, 2, 4}))
    assert selected["selected_horizon"] == 4
    assert selected["jointly_qualified"]
    fallback = select_layer_horizon(_selection_rows(set()))
    assert fallback["selected_horizon"] == 1
    assert fallback["fallback_used"]
    with pytest.raises(ValueError):
        select_layer_horizon(_selection_rows({1})[:-1])


def test_numerical_health_is_fail_capable() -> None:
    record = {
        "maximum_parameter_replay_relative_residual": np.asarray(1.0e-8),
        "maximum_first_moment_replay_relative_residual": np.asarray(2.0e-8),
        "maximum_second_moment_replay_relative_residual": np.asarray(3.0e-8),
        "minimum_clipping_boundary_distance": np.asarray(0.5),
        "resource_peak_gpu_reserved_bytes": np.asarray(2**30, dtype=np.int64),
    }
    assert _numerical_health(record)["valid"]
    record["minimum_clipping_boundary_distance"] = np.asarray(0.0)
    assert not _numerical_health(record)["valid"]


def _minimal_result() -> dict[str, object]:
    directions = []
    for index, direction in enumerate(DIRECTION_CLASSES):
        directions.append(
            {
                "direction": direction,
                "response_pass_count": index + 1,
                "total_count": 18,
                "median_relative_residual": 0.1,
                "maximum_relative_residual": 0.2,
                "minimum_visible_directional_gain": 0.5,
                "median_visible_directional_gain": 1.5,
                "maximum_visible_directional_gain": 2.5,
                "attenuating_count": 4,
            }
        )
    return {
        "state_dimension": 61_868_742,
        "construction": {
            "qualified_cell_count": 7,
            "total_cell_count": 135,
            "layers": {
                str(layer): {
                    "selected_horizon": 1,
                    "jointly_qualified": False,
                    "qualified_cell_count": layer,
                    "total_cell_count": 45,
                }
                for layer in range(3)
            },
        },
        "heldout": {
            "response_pass_count": 6,
            "total_count": 54,
            "attenuating_count": 12,
            "by_direction": directions,
        },
        "stratum": {
            "preserved_count": 0,
            "total_count": 9,
            "rows": [
                {
                    "trajectory": "A",
                    "layer": 0,
                    "source_row_quotient_norm": 0.0,
                    "observed_force_norm": 0.25,
                    "face_preserved": False,
                }
            ],
        },
        "resource": {
            "aggregate_reserved_device_hours": 0.5,
            "maximum_process_reserved_bytes": 2**30,
        },
    }


