from __future__ import annotations

from pathlib import Path
import sys


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(CONFIRM))
sys.path.insert(0, str(ANALYSIS))

import analyze_full_stack_confirmation as ANALYZE
import full_stack_confirmation as FSC
import program_state as PS
from validated_interval import RationalInterval, rational_object


def _interval(lower, upper=None):
    if upper is None:
        upper = lower
    return RationalInterval(lower, upper).to_object()


def _state():
    return {
        "schema_version": PS.SCHEMA_VERSION,
        "theta": [_interval("9/10", "11/10"), _interval("-3/5", "-2/5")],
        "first_moment": [_interval("1/20", "3/20"), _interval("-1/10", "0")],
        "second_moment": [_interval("1/10", "1/5"), _interval("1/10", "1/5")],
        "optimizer_step": 4,
        "scheduler_phase": "stable_peak",
        "learning_rate": _interval("1/100"),
        "cover_state": {"cell": "c0"},
        "intervention_state": {"arm": "sham"},
    }


def _payload():
    update = {
        "source_cell": _state(),
        "gradient_cell": [
            _interval("1/10", "1/5"), _interval("-1/5", "-1/10")],
        "beta1": "9/10", "beta2": "19/20", "epsilon": "1/1000",
        "weight_decay": "1/10", "clip_value": "1",
        "intervention_cell": [_interval("0"), _interval("0")],
        "next_learning_rate": _interval("1/100"),
        "next_scheduler_phase": "stable_peak",
        "next_cover_state": {"cell": "c1"},
        "next_intervention_state": {"arm": "sham"},
    }
    target = PS.interval_adamw_successor(**update)["image"]
    block_names = ("z.c0", "m.c0")
    conversion_errors = [
        {"target": target_name, "source": source_name,
         "absolute_operator_error": 0.0}
        for target_name in block_names for source_name in block_names
    ]
    return {
        "schema_version": "pldr-full-stack-confirmation-input-v1",
        "time_semantics": "FINITE_IMPLEMENTED_SCHEDULE",
        "criterion": 1.0,
        "layer_cells": {"0": [{"cell_id": "c0", "width": 1}]},
        "normal_closure": {
            "stack_parameter_jacobian": [[1.0]],
            "normal_velocity_parameter_jacobian": [[0.5]],
            "stack_center": [1.0], "normal_velocity_center": [0.6],
            "construction_partition": ["construction-0"],
            "runtime_roundoff_bounds": {
                "right_inverse_residual": 1e-15,
                "closure_complement": 1e-15,
                "validation_center_residual": 1e-15,
                "validation_derivative_residual": 1e-15,
            },
            "validation_partition": ["validation-0"],
            "validation_cells": [{
                "cell_id": "validation-0", "stack": [1.1],
                "normal_velocity": [0.65],
                "stack_parameter_jacobian": [[1.0]],
                "normal_velocity_parameter_jacobian": [[0.5]],
                "cell_radius": 0.1, "second_derivative_bound": 0.0,
                "affine_residual_proof": True,
            }],
        },
        "runtime_primitives": [{
            "name": "complete_operator",
            "values": [[0.2, 0.0], [0.0, 0.2]],
            "absolute_errors": 1e-15,
        }],
        "edges": [{
            "edge_id": "ab", "source": "a", "target": "b",
            "program_update": update, "target_cell": target,
            "complete_operator": [[0.2, 0.0], [0.0, 0.2]],
            "block_slices": [
                {"name": "z.c0", "start": 0, "stop": 1},
                {"name": "m.c0", "start": 1, "stop": 2},
            ],
            "source_metrics": {"z.c0": [[1.0]], "m.c0": [[1.0]]},
            "target_metrics": {"z.c0": [[1.0]], "m.c0": [[1.0]]},
            "runtime_conversion_errors": conversion_errors,
            "forcing": [rational_object(0), rational_object(0)],
            "forcing_provenance": [
                "exact_program_interval_ledger", "exact_program_interval_ledger"],
        }],
        "path_length": 1, "positive_witness": None,
        "frozen_checkpoint": None,
        "validation_membership": [True],
        "cover_utility_margins": [0.25],
        "direct_row_map_uppers": [0.75],
    }


def test_full_stack_runner_and_analysis_use_only_primitive_certificates():
    record = FSC.build_confirmation(_payload())
    assert record["decision"] == "CONFIRMED"
    assert record["registry"]["dimension"] == 1
    assert record["trace_fitting_used"] is False
    assert record["caller_selected_radius_multipliers_used"] is False
    blocks = record["edges"][0]["complete_block_comparison"]["operator_blocks"]
    assert sum(row["is_off_diagonal"] for row in blocks) == 2
    analysis = ANALYZE.analyze([record])
    assert analysis["all_records_confirmed"] is True
    assert analysis["trace_refit_performed"] is False


def test_controlled_tail_requires_a_total_successor_graph():
    payload = _payload()
    payload["time_semantics"] = "CONTROLLED_INFINITE_TAIL"
    payload["positive_witness"] = {
        "weights": [1.0, 1.0],
        "kappa": 0.5,
    }
    record = FSC.build_confirmation(payload)
    assert record["positive_witness"]["maximum_ratio"] < 0.5
    assert record["conditions"][
        "controlled_tail_successor_graph_total"] is False
