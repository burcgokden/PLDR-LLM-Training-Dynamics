from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import copy
import hashlib
import sys

import numpy as np
import pytest


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

from branch_resolved_state import (  # noqa: E402
    UPDATE_OPERATION_ORDER,
    bind_owned_successor,
    build_complete_state,
    tensor_record,
    verify_replayed_target,
)
from constructive_block_comparison import (  # noqa: E402
    bind_owned_edge,
    construct_comparison,
    construct_positive_witness,
    forced_affine_upper,
)
from directional_taylor import certify_directional_segment  # noqa: E402
from normal_stability import (  # noqa: E402
    certify_lifted_family,
    construct_normal_response,
    ordered_product_convolution,
)
from ordered_affine_blocking import (  # noqa: E402
    block_owned_edges,
    infinity_norm_affine_upper,
)


def normal_primitives():
    return {
        "loss_hessian_lower": "1",
        "loss_hessian_upper": "2",
        "downstream_normal_singular_lower": "3/4",
        "downstream_normal_norm_upper": "5/4",
        "full_hessian_residual_upper": "1/20",
        "branch_residual_upper": "1/100",
        "preconditioner_motion_upper": "1/100",
        "normal_force_terms": {
            "rope_frequency_remainder": "1/1000",
            "contextual_covariance": "1/500",
            "finite_batch": "1/1000",
            "nonstationarity": "1/2000",
            "intervention": "0",
        },
        "normal_hessian_lipschitz_upper": "1/5",
        "normal_tube_radius": "1/10",
    }


def test_normal_response_is_derived_from_primitive_edges():
    result = construct_normal_response(normal_primitives())
    assert result["conditions"]["normal_response_strictly_positive"]
    assert result["normal_response_lower"] == {
        "numerator": 201, "denominator": 400,
    }
    payload = normal_primitives()
    payload["full_hessian_residual_upper"] = "1"
    with pytest.raises(ValueError, match="lower edge"):
        construct_normal_response(payload)


def test_mesh_plus_modulus_encloses_between_point_spikes():
    result = certify_directional_segment(
        point_second_directional_uppers=["1", "1", "1"],
        third_directional_modulus="4",
        subdivisions=2,
        observed_remainder="1",
    )
    assert result["mesh_modulus_charge"] == {
        "numerator": 1, "denominator": 1,
    }
    assert result["observed_remainder_enclosed"]
    with pytest.raises(ValueError, match="subdivisions"):
        certify_directional_segment(
            point_second_directional_uppers=["1", "1"],
            third_directional_modulus="4",
            subdivisions=2,
        )


def comparison_fixture():
    return construct_comparison(
        block_names=["normal", "moment"],
        transformed_operator_norms={
            ("normal", "normal"): Fraction(1, 10),
            ("normal", "moment"): Fraction(1, 20),
            ("moment", "normal"): Fraction(1, 20),
            ("moment", "moment"): Fraction(1, 10),
        },
        forcing_norms={
            "normal": Fraction(1, 100),
            "moment": Fraction(1, 100),
        },
        structural_zeros=[],
    )


def test_complete_comparison_constructs_all_coefficients_and_witness():
    result = comparison_fixture()
    assert len(result["operator_blocks"]) == 4
    witness = construct_positive_witness(result["comparison_matrix"])
    assert witness["kappa"]["numerator"] < witness["kappa"]["denominator"]
    assert forced_affine_upper([[0]], [0], [1]) == (Fraction(1),)
    assert infinity_norm_affine_upper([[0]], [0], [1]) == 1


def test_owned_blocking_requires_a_resolved_temporal_chain():
    comparison = comparison_fixture()
    edge_one = bind_owned_edge(
        edge_id="e0", source_state_sha256="a" * 64,
        target_state_sha256="b" * 64, comparison_record=comparison)
    edge_two = bind_owned_edge(
        edge_id="e1", source_state_sha256="b" * 64,
        target_state_sha256="c" * 64, comparison_record=comparison)
    block = block_owned_edges([edge_one, edge_two])
    assert block["edge_count"] == 2
    with pytest.raises(ValueError, match="state chain"):
        block_owned_edges([edge_two, edge_one])


def test_exact_lifted_family_and_ordered_convolution():
    family = certify_lifted_family({
        "alpha": {"lower": "1/100", "upper": "101/10000"},
        "beta": {"lower": "9/10", "upper": "901/1000"},
        "normal_eigenvalue": {"lower": "1", "upper": "101/100"},
        "decay": {"lower": "1/1000", "upper": "11/10000"},
    })
    assert family["decision"] == "CERTIFIED"
    assert family["common_metric_certificate"]["strictly_contracting"]
    path = ordered_product_convolution(
        ["1/2", "1/2"], ["1/4", "0"], "1")
    assert path["final_upper"] == {"numerator": 3, "denominator": 8}


def test_ordered_convolution_allows_transient_expansion():
    path = ordered_product_convolution(
        ["3/2", "1/3"], ["0", "0"], "1")
    assert path["cumulative_gain"] == {"numerator": 1, "denominator": 2}
    assert path["path_contracts_without_force"]
    assert path["envelope"][1] == {"numerator": 3, "denominator": 2}


def state_fixture(step, model_value, moment_value):
    tensor = tensor_record("weight", np.array([model_value], dtype=np.float64))
    moment = tensor_record(
        "weight.exp_avg", np.array([moment_value], dtype=np.float64))
    row = tensor_record("layer0.row0", np.array([model_value], dtype=np.float64))
    return build_complete_state(
        model_tensors=[tensor],
        optimizer_tensors=[moment],
        optimizer_step=step,
        optimizer_groups={"betas": ["9/10", "19/20"], "epsilon": "1/100000"},
        scheduler_state={"phase": "warmup", "index": step},
        amp_state={"enabled": False},
        gradient_accumulation={"count": 0, "buffers": []},
        rng_states={"python": "p", "numpy": "n", "cpu": "c", "cuda": []},
        data_state={"dataset_sha256": "d" * 64, "batch_indices": [step]},
        intervention_state={"arm": "nominal"},
        branch_signature={"clip_mask": ["UNCLIPPED"], "phase": "warmup"},
        physical_rows=[row],
        row_registry={"rows": ["layer0.row0"]},
        model_config={"width": 1, "depth": 1},
        operation_order=list(UPDATE_OPERATION_ORDER),
        code_manifest={"sha256": "e" * 64},
    )


def test_complete_state_binds_moments_rows_and_independent_replay():
    source = state_fixture(0, 1.0, 0.0)
    target = state_fixture(1, 0.9, 0.1)
    edge = bind_owned_successor(
        edge_id="owned",
        source_state=source,
        target_state=target,
        update_direction_sha256=hashlib.sha256(b"delta").hexdigest(),
        constructor_sha256=hashlib.sha256(b"constructor").hexdigest(),
        operation_trace=UPDATE_OPERATION_ORDER,
    )
    replay = verify_replayed_target(edge, target)
    assert replay["decision"] == "REPLAYED"
    changed_moment = state_fixture(1, 0.9, 0.2)
    with pytest.raises(ValueError, match="regenerate"):
        verify_replayed_target(edge, changed_moment)


def test_branch_crossing_is_rejected_before_continuous_displacement():
    source = state_fixture(0, 1.0, 0.0)
    target = state_fixture(1, 0.9, 0.1)
    target_fields = {
        name: copy.deepcopy(target[name])
        for name in (
            "model_tensors", "optimizer_tensors", "optimizer_step",
            "optimizer_groups", "scheduler_state", "amp_state",
            "gradient_accumulation", "rng_states", "data_state",
            "intervention_state", "branch_signature", "physical_rows",
            "row_registry", "model_config", "operation_order", "code_manifest",
        )
    }
    target_fields["branch_signature"]["clip_mask"] = ["HIGH_CLIPPED"]
    crossed = build_complete_state(**target_fields)
    with pytest.raises(ValueError, match="split the edge"):
        bind_owned_successor(
            edge_id="crossed", source_state=source, target_state=crossed,
            update_direction_sha256="a" * 64,
            constructor_sha256="b" * 64,
            operation_trace=UPDATE_OPERATION_ORDER,
        )
