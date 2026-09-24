"""Q0 fixtures for the complete-stack primary certificate."""

from __future__ import annotations

import numpy as np

from block_comparison import construct_block_comparison, weighted_gain
from full_stack import build_registry, coordinate
from full_stack_confirmation import (
    TIME_SEMANTICS,
    successor_graph_is_total,
)
from normal_closure import construct_dense_normal_closure
from program_state import (
    SCHEMA_VERSION as PROGRAM_SCHEMA,
    derived_positive_map,
    interval_adamw_successor,
)
from runtime_enclosure import enclose_runtime_array, verify_runtime_array
from validated_interval import RationalInterval


def _interval(lower, upper=None):
    if upper is None:
        upper = lower
    return RationalInterval(lower, upper).to_object()


def full_stack_fixture():
    registry = build_registry({
        0: [("c0", 2)],
        1: [("c0", 1)],
    })
    hidden_mode = construct_block_comparison(
        np.diag([0.5, 1.1]),
        [{"name": "complete", "start": 0, "stop": 2}],
        {"complete": np.eye(2)}, {"complete": np.eye(2)},
    )
    normal = construct_dense_normal_closure(
        stack_parameter_jacobian=np.eye(registry["dimension"]),
        normal_velocity_parameter_jacobian=(
            0.4 * np.eye(registry["dimension"])),
        stack_center=np.ones(registry["dimension"]),
        normal_velocity_center=(
            0.4 * np.ones(registry["dimension"])),
        construction_partition=["construction"],
        validation_partition=["validation"],
        runtime_roundoff_bounds={
            "right_inverse_residual": 1e-15,
            "closure_complement": 1e-15,
            "validation_center_residual": 1e-15,
            "validation_derivative_residual": 1e-15,
        },
        validation_cells=[{
            "cell_id": "validation",
            "stack": np.ones(registry["dimension"]).tolist(),
            "normal_velocity": (
                0.4 * np.ones(registry["dimension"])).tolist(),
            "stack_parameter_jacobian": np.eye(
                registry["dimension"]).tolist(),
            "normal_velocity_parameter_jacobian": (
                0.4 * np.eye(registry["dimension"])).tolist(),
            "cell_radius": 0.1,
            "second_derivative_bound": 0.0,
            "affine_residual_proof": True,
        }],
    )
    source = {
        "schema_version": PROGRAM_SCHEMA,
        "theta": [_interval("9/10", "11/10")],
        "first_moment": [_interval("1/20", "3/20")],
        "second_moment": [_interval("1/10", "1/5")],
        "optimizer_step": 4,
        "scheduler_phase": "stable",
        "learning_rate": _interval("1/100"),
        "cover_state": {"cell": "c0"},
        "intervention_state": {"arm": "sham"},
    }
    successor = interval_adamw_successor(
        source_cell=source,
        gradient_cell=[_interval("1/10", "1/5")],
        beta1="9/10", beta2="19/20", epsilon="1/1000",
        weight_decay="1/10", clip_value="1",
        intervention_cell=[_interval("0")],
        next_learning_rate=_interval("1/100"),
        next_scheduler_phase="stable",
        next_cover_state={"cell": "c1"},
        next_intervention_state={"arm": "sham"},
    )
    positive = derived_positive_map(successor, successor["image"])
    runtime_values = np.asarray([[1.0, -0.25], [0.5, 2.0]])
    runtime = enclose_runtime_array(runtime_values, 1e-15)
    checks = {
        "complete_registry_dimension": registry["dimension"] == 5,
        "complete_registry_last_coordinate":
            coordinate(registry, 4) == (1, "c0", 0, 0),
        "orthogonal_expansion_exposed":
            weighted_gain(hidden_mode["comparison_matrix"], [1.0]) > 1.0,
        "derivative_normal_right_inverse":
            normal["right_inverse_residual"] < 1.0,
        "controlled_tail_graph_totality": (
            successor_graph_is_total([{
                "source": "a", "target": "a",
            }])
            and not successor_graph_is_total([{
                "source": "a", "target": "b",
            }])
        ),
        "normal_validation_closed":
            normal["maximum_validation_residual_envelope"] < 1e-12,
        "actual_program_positive_map":
            len(positive["matrix"]) == 3 and len(positive["forcing"]) == 3,
        "runtime_enclosure_replays": verify_runtime_array(
            runtime, runtime_values),
        "time_semantics_complete": TIME_SEMANTICS == {
            "FINITE_IMPLEMENTED_SCHEDULE", "FROZEN_CHECKPOINT",
            "CONTROLLED_INFINITE_TAIL",
        },
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "registry_sha256": registry["registry_sha256"],
        "normal_record_sha256": normal["record_sha256"],
    }
