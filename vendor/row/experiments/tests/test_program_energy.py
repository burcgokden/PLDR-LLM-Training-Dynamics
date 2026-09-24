from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

from full_stack import build_registry  # noqa: E402
from independent_energy_checker import check_record  # noqa: E402
from program_energy import (  # noqa: E402
    SCHEMA_VERSION,
    construct_program_energy_edge,
    exact_energy_increment,
)
from resource_gate import qualify_resources  # noqa: E402


def primitive_fixture():
    registry = build_registry({0: [("physical", 2)]})
    return {
        "schema_version": SCHEMA_VERSION,
        "edge_id": "tiny-exact-edge",
        "source_state_sha256": "1" * 64,
        "target_state_sha256": "2" * 64,
        "successor_owner_sha256": "3" * 64,
        "registry": registry,
        "source_stack": [1.0, 0.0, 0.0, 1.0],
        "target_stack": [0.75, 0.0, 0.0, 0.75],
        "actual_update_jvp": [-0.25, 0.0, 0.0, -0.25],
        "metric_blocks": [{
            "block_index": 0,
            "kind": "scaled_identity",
            "scale": 1.0,
        }],
        "primitive_errors": {
            name: {
                "observed_norm": 0.0,
                "certified_upper": 0.0,
                "derivation": "exact_binary_fixture",
            }
            for name in (
                "taylor_remainder", "row_cover_motion",
                "metric_conversion", "runtime_roundoff", "exogenous_input",
            )
        },
        "block_completeness": [
            [0, "physical", output, input_index]
            for output in range(2) for input_index in range(2)
        ],
        "construction_partition_sha256": "4" * 64,
        "validation_partition_sha256": "5" * 64,
    }


def test_exact_energy_identity_and_independent_replay():
    primitive = primitive_fixture()
    identity = exact_energy_increment(
        primitive["registry"], primitive["source_stack"],
        primitive["target_stack"], primitive["metric_blocks"],
    )
    assert identity["identity_enclosed"]
    record = construct_program_energy_edge(primitive)
    assert record["decision"] == "QUALIFIED"
    assert record["derived_comparison"]["q_squared"] == pytest.approx(0.5625)
    assert record["derived_comparison"]["negative_work_upper"] < 0
    assert check_record(record)["decision"] == "REPLAYED"


def test_comparison_is_derived_from_jvp_and_closed_charge():
    primitive = primitive_fixture()
    primitive["target_stack"] = [0.8125, 0.0, 0.0, 0.8125]
    primitive["primitive_errors"]["taylor_remainder"] = {
        "observed_norm": np.sqrt(2.0) / 16.0,
        "certified_upper": 0.09,
        "derivation": "directional_second_derivative_upper",
    }
    record = construct_program_energy_edge(primitive)
    observed_ratio = (
        record["energy_identity"]["target_energy"]
        / record["energy_identity"]["source_energy"]
    )
    assert record["derived_comparison"]["q_squared"] == pytest.approx(0.5625)
    assert record["derived_comparison"]["q_squared"] != pytest.approx(
        observed_ratio)
    assert record["derived_comparison"]["affine_source"] > 0
    assert check_record(record)["decision"] == "REPLAYED"


def test_unenclosed_remainder_and_missing_matrix_coordinate_reject():
    primitive = primitive_fixture()
    primitive["target_stack"][0] = 10.0
    assert construct_program_energy_edge(primitive)["decision"] == "REJECTED"
    primitive = primitive_fixture()
    primitive["block_completeness"].pop()
    assert construct_program_energy_edge(primitive)["decision"] == "REJECTED"


def test_large_dense_metric_and_dense_architecture_derivative_are_forbidden():
    registry = build_registry({0: [("large", 17)]})
    primitive = primitive_fixture()
    primitive["registry"] = registry
    primitive["source_stack"] = [0.0] * registry["dimension"]
    primitive["target_stack"] = [0.0] * registry["dimension"]
    primitive["actual_update_jvp"] = [0.0] * registry["dimension"]
    primitive["metric_blocks"] = [{
        "block_index": 0,
        "kind": "dense_tiny_fixture",
        "matrix": np.eye(registry["dimension"]).tolist(),
    }]
    primitive["block_completeness"] = [
        [0, "large", output, input_index]
        for output in range(17) for input_index in range(17)
    ]
    with pytest.raises(ValueError, match="tiny fixture"):
        construct_program_energy_edge(primitive)

    result = qualify_resources({
        "architecture": "w128_d08",
        "peak_gpu_bytes": 1,
        "peak_host_bytes": 1,
        "retained_bytes": 1,
        "projected_certificate_seconds": 1,
        "dense_derivative_requested": True,
    })
    assert result["decision"] == "INFEASIBLE"


def test_caller_cannot_smuggle_a_comparison_into_primitives():
    primitive = primitive_fixture()
    primitive["comparison_operator"] = [[0.0]]
    with pytest.raises(ValueError, match="missing or unknown"):
        construct_program_energy_edge(primitive)
