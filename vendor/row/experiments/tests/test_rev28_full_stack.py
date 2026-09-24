from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

import block_comparison as BC
import full_stack as FS
import normal_closure as NC
import program_state as PS
import runtime_enclosure as RE
import validated_interval as VI
from validated_interval import RationalInterval


def _interval(lower, upper=None):
    if upper is None:
        upper = lower
    return RationalInterval(lower, upper).to_object()


_NORMAL_ROUNDOFF = {
    "right_inverse_residual": 1e-15,
    "closure_complement": 1e-15,
    "validation_center_residual": 1e-15,
    "validation_derivative_residual": 1e-15,
}


def _state(*, step=4, second=("1/10", "1/5"), rate="1/100"):
    return {
        "schema_version": PS.SCHEMA_VERSION,
        "theta": [_interval("9/10", "11/10"), _interval("-3/5", "-2/5")],
        "first_moment": [_interval("1/20", "3/20"), _interval("-1/10", "0")],
        "second_moment": [_interval(*second), _interval(*second)],
        "optimizer_step": step,
        "scheduler_phase": "stable_peak",
        "learning_rate": _interval(rate),
        "cover_state": {"cell": "c0"},
        "intervention_state": {"arm": "sham"},
    }


def test_full_stack_registry_enumerates_every_matrix_coordinate():
    registry = FS.build_registry({
        0: [("c0", 2), ("c1", 2)],
        1: [("c0", 1)],
    })
    assert registry["dimension"] == 9
    assert FS.coordinate(registry, 0) == (0, "c0", 0, 0)
    assert FS.coordinate(registry, 3) == (0, "c0", 1, 1)
    assert FS.coordinate(registry, 8) == (1, "c0", 0, 0)
    matrices = {
        (0, "c0"): np.eye(2),
        (0, "c1"): np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        (1, "c0"): np.asarray([[5.0]]),
    }
    stack = FS.stack_jacobians(registry, matrices)
    replay = FS.unstack_jacobians(registry, stack)
    assert all(np.array_equal(replay[key], value) for key, value in matrices.items())
    with pytest.raises(ValueError, match="key mismatch"):
        FS.stack_jacobians(registry, {(0, "c0"): np.eye(2)})


def test_architecture_registry_rejects_fixed_invalid_or_sparse_layers():
    with pytest.raises(ValueError, match="every layer"):
        FS.build_registry({0: [("c0", 1)], 3: [("c0", 1)]})
    registry = FS.build_registry({index: [("c0", 1)] for index in range(4)})
    assert registry["criterion_layers"] == [0, 1, 2, 3]


def test_full_block_comparison_exposes_orthogonal_and_cross_layer_modes():
    orthogonal = np.diag([0.5, 1.1])
    one_block = [{"name": "complete", "start": 0, "stop": 2}]
    result = BC.construct_block_comparison(
        orthogonal, one_block,
        {"complete": np.eye(2)}, {"complete": np.eye(2)},
    )
    assert orthogonal[0, 0] < 1
    assert result["comparison_matrix"][0][0] > 1

    coupled = np.asarray([[0.4, 0.8], [0.8, 0.4]])
    blocks = [
        {"name": "layer0", "start": 0, "stop": 1},
        {"name": "layer1", "start": 1, "stop": 2},
    ]
    result = BC.construct_block_comparison(
        coupled, blocks,
        {"layer0": [[1.0]], "layer1": [[1.0]]},
        {"layer0": [[1.0]], "layer1": [[1.0]]},
    )
    matrix = np.asarray(result["comparison_matrix"])
    assert np.all(np.diag(matrix) < 1)
    assert BC.weighted_gain(matrix, [1.0, 1.0]) > 1
    with pytest.raises(ValueError, match="not componentwise"):
        BC.verify_positive_witness(matrix, [1.0, 1.0], 0.99)


def test_positive_affine_path_composition_is_ordered_and_complete():
    edges = [
        {"edge_id": "ab", "source": "a", "target": "b",
         "matrix": [[0.5, 0.1], [0.0, 0.4]], "forcing": [0.1, 0.2]},
        {"edge_id": "ba", "source": "b", "target": "a",
         "matrix": [[0.3, 0.0], [0.2, 0.3]], "forcing": [0.4, 0.1]},
    ]
    paths = BC.enumerate_path_compositions(edges, 2)
    assert {tuple(row["edge_ids"]) for row in paths} == {("ab", "ba"), ("ba", "ab")}
    first = next(row for row in paths if row["edge_ids"] == ["ab", "ba"])
    expected_matrix = np.asarray(edges[1]["matrix"]) @ np.asarray(edges[0]["matrix"])
    expected_source = (
        np.asarray(edges[1]["matrix"]) @ np.asarray(edges[0]["forcing"])
        + np.asarray(edges[1]["forcing"])
    )
    assert np.allclose(first["matrix"], expected_matrix)
    assert np.allclose(first["forcing"], expected_source)


def test_derivative_defined_normal_closure_freezes_forcing_and_partition():
    B = np.eye(2)
    DQ = np.diag([0.5, 0.25])
    z = np.asarray([1.0, 2.0])
    forcing = np.asarray([0.1, -0.2])
    q = DQ @ z + forcing
    validation_stack = np.asarray([1.2, 1.8])
    cells = [{
        "cell_id": "validation-0",
        "stack": validation_stack.tolist(),
        "normal_velocity": (DQ @ validation_stack + forcing).tolist(),
        "stack_parameter_jacobian": B.tolist(),
        "normal_velocity_parameter_jacobian": DQ.tolist(),
        "cell_radius": 0.1,
        "second_derivative_bound": 0.0,
        "affine_residual_proof": True,
    }]
    record = NC.construct_dense_normal_closure(
        stack_parameter_jacobian=B,
        normal_velocity_parameter_jacobian=DQ,
        stack_center=z,
        normal_velocity_center=q,
        construction_partition=["construction-0"],
        validation_partition=["validation-0"],
        validation_cells=cells,
        runtime_roundoff_bounds=_NORMAL_ROUNDOFF,
    )
    assert np.allclose(record["normal_operator"], DQ)
    assert np.allclose(record["reference_forcing"], forcing)
    assert record["right_inverse_residual"] < 1
    assert NC.check_dense_normal_closure(
        record,
        stack_parameter_jacobian=B,
        normal_velocity_parameter_jacobian=DQ,
        stack_center=z,
        normal_velocity_center=q,
        validation_cells=cells,
    )
    with pytest.raises(ValueError, match="overlap"):
        NC.construct_dense_normal_closure(
            stack_parameter_jacobian=B,
            normal_velocity_parameter_jacobian=DQ,
            stack_center=z,
            normal_velocity_center=q,
            construction_partition=["same"],
            validation_partition=["same"],
            validation_cells=cells,
            runtime_roundoff_bounds=_NORMAL_ROUNDOFF,
        )


def test_normal_closure_rejects_unproved_zero_nonlinear_residual():
    with pytest.raises(ValueError, match="exact affine"):
        NC.construct_dense_normal_closure(
            stack_parameter_jacobian=[[1.0]],
            normal_velocity_parameter_jacobian=[[0.5]],
            stack_center=[1.0], normal_velocity_center=[0.5],
            construction_partition=["fit"],
            validation_partition=["test"],
            validation_cells=[{
                "cell_id": "test", "stack": [1.0],
                "normal_velocity": [0.5],
                "stack_parameter_jacobian": [[1.0]],
                "normal_velocity_parameter_jacobian": [[0.5]],
                "cell_radius": 0.1, "second_derivative_bound": 0.0,
                "affine_residual_proof": False,
            }],
            runtime_roundoff_bounds=_NORMAL_ROUNDOFF,
        )


def test_runtime_to_rational_enclosure_contains_every_realized_primitive():
    values = np.asarray([[1.0, np.nextafter(1.0, np.inf)], [-0.25, 3.0]])
    record = RE.enclose_runtime_array(values, 2e-15)
    assert RE.verify_runtime_array(record, values)
    tampered = dict(record)
    tampered["intervals"] = list(record["intervals"])
    tampered["intervals"][0] = _interval("100", "101")
    with pytest.raises(ValueError, match="outside"):
        RE.verify_runtime_array(tampered, values)
    assert not RE.charged_gain(0.99, 0.02)["strict_contraction"]


def test_actual_program_state_generates_derivatives_and_positive_map():
    source = _state()
    successor = PS.interval_adamw_successor(
        source_cell=source,
        gradient_cell=[_interval("1/10", "1/5"), _interval("-1/5", "-1/10")],
        beta1="9/10", beta2="19/20", epsilon="1/1000",
        weight_decay="1/10", clip_value="1",
        intervention_cell=[_interval("0"), _interval("0")],
        next_learning_rate=_interval("1/100"),
        next_scheduler_phase="stable_peak",
        next_cover_state={"cell": "c1"},
        next_intervention_state={"arm": "sham"},
    )
    target = successor["image"]
    assert PS.check_program_state_inclusion(successor, target)
    image_intervals = (
        target["theta"] + target["first_moment"]
        + target["second_moment"]
    )
    for interval, center, radius in zip(
        image_intervals, successor["image_centers"],
        successor["image_radii"],
    ):
        interval = VI.RationalInterval.from_object(interval)
        center = VI.as_fraction(center)
        radius = VI.as_fraction(radius)
        assert radius >= max(abs(interval.lower - center),
                             abs(interval.upper - center))
    positive = PS.derived_positive_map(successor, target)
    assert len(positive["matrix"]) == 6
    assert len(positive["forcing"]) == 6
    assert positive["provenance"].startswith("actual_program")
    signature = inspect.signature(PS.interval_adamw_successor)
    assert "center_errors" not in signature.parameters
    assert "auxiliary_lipschitz" not in signature.parameters


def test_actual_program_state_rejects_clipping_and_sqrt_branch_crossings():
    with pytest.raises(ValueError, match="clipping boundary"):
        PS.interval_adamw_successor(
            source_cell=_state(),
            gradient_cell=[_interval("9/10", "11/10"), _interval("1/10")],
            beta1="9/10", beta2="19/20", epsilon="1/1000",
            weight_decay="1/10", clip_value="1",
            intervention_cell=[_interval("0"), _interval("0")],
            next_learning_rate=_interval("1/100"),
            next_scheduler_phase="stable_peak",
            next_cover_state={"cell": "c1"},
            next_intervention_state={"arm": "sham"},
        )
    with pytest.raises(ValueError, match="square-root branch point"):
        PS.interval_adamw_successor(
            source_cell=_state(second=("0", "0")),
            gradient_cell=[_interval("0"), _interval("0")],
            beta1="9/10", beta2="19/20", epsilon="1/1000",
            weight_decay="1/10", clip_value="1",
            intervention_cell=[_interval("0"), _interval("0")],
            next_learning_rate=_interval("1/100"),
            next_scheduler_phase="stable_peak",
            next_cover_state={"cell": "c1"},
            next_intervention_state={"arm": "sham"},
        )
