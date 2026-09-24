"""Regression tests for the comprehensive row-map confirmation path."""

from __future__ import annotations

import copy
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parent
sys.path.insert(0, str(EXP / "confirm"))
sys.path.insert(0, str(EXP / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(EXP))

import assemble_confirmation as AC  # noqa: E402
import campaign_design as CD  # noqa: E402
import campaign_record as R  # noqa: E402
import confirmation_analysis as A  # noqa: E402
import confirmation_qualification as Q  # noqa: E402
import direct_certificate as D  # noqa: E402
import estimator_manifest as EM  # noqa: E402
import gen_confirmation_protocols as G  # noqa: E402
import launch_confirmation as LC  # noqa: E402
import measure_direct_certificate as MD  # noqa: E402
import measure_optimizer_transport as MOT  # noqa: E402
import measure_rg_closure as MRG  # noqa: E402
import measurement_graph as MG  # noqa: E402
import optimizer_transport as O  # noqa: E402
import pathwise_contraction as PC  # noqa: E402
import renormalization_group as RG  # noqa: E402
import row_domain as RD  # noqa: E402
import row_map_jacobian as J  # noqa: E402
import run_confirmation_campaign as RCC  # noqa: E402
import validated_linear_algebra as VLA  # noqa: E402
import validated_row_map as VR  # noqa: E402
from pldr_model_v510 import PLDR_Model  # noqa: E402


def test_frozen_campaign_design_is_self_consistent():
    assert CD.validate_design()
    value = CD.design_object()
    assert value["resources"]["maximum_power_watts_per_gpu"] == 320
    assert value["e7_interventions"]["outcome_checkpoints"] == [
        9256, 11256, 13256,
    ]
    assert set(range(10000, 10011)).issubset(
        value["baseline"]["model_checkpoint_steps"])
    assert RCC._protocols_through("E6") == tuple(f"E{i}" for i in range(7))


def measurement_rows(protocol_id, overrides=None):
    overrides = overrides or {}
    rows = []
    for name in R.MEASUREMENT_NAMES[protocol_id]:
        value = float(overrides.get(name, 0.5))
        rows.append({
            "name": name,
            "value": value,
            "unit": "registered_unit",
            "method": "test_fixture",
            "status": "OBSERVED",
            "reason_code": None,
        })
    return rows


def test_live_measurement_graph_replays_and_binds_attached_artifact(tmp_path):
    measurement = {
        "schema_version": "live-fixture-v1",
        "measurements": measurement_rows("E6"),
    }
    manifest = EM.manifest_object(ROOT)
    graph, bindings = MG.build_measurement_graph(
        "E6", measurement, manifest, root=ROOT)
    assembled = AC.assemble("E6", graph, bindings, manifest)
    assert [row["name"] for row in assembled["measurements"]] == list(
        R.MEASUREMENT_NAMES["E6"])
    nodes = {row["name"]: row for row in graph["nodes"]}
    assert nodes["primary_predicted_entry_step"][
        "derivation_kind"] == "finite_horizon"
    assert nodes["primary_schedule_product_convolution_upper"][
        "derivation_kind"] == "positive_comparison"

    artifact = tmp_path / "measurement.json"
    artifact.write_text(R.strict_dumps(measurement), encoding="utf-8")
    assert LC._bound_measurement_artifact(
        graph, [artifact], "E6") == measurement

    tampered = copy.deepcopy(measurement)
    tampered["measurements"][0]["value"] += 1.0
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(R.strict_dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="graph-bound measurement"):
        LC._bound_measurement_artifact(graph, [tampered_path], "E6")


def record(protocol_id="E0", seed=1, step=10, arm="observational",
           measurements=None, block_size=None, architecture_cell=None,
           schedule_cell=None):
    return {
        "schema_version": R.SCHEMA_VERSION,
        "campaign_id": "campaign-fixture",
        "protocol_id": protocol_id,
        "run_id": f"run-{protocol_id.lower()}-{seed}-{arm}",
        "seed": seed,
        "unit_key": {
            "seed": seed,
            "checkpoint": step,
            "layer_bundle": None,
            "tensor_group": None,
            "certificate_box": None,
            "schedule_cell": schedule_cell,
            "architecture_cell": architecture_cell,
            "block_size": block_size,
            "arm": arm,
        },
        "clocks": {
            "global_optimizer_step": step,
            "segment_index": 0,
            "segment_local_step": step,
            "segment_global_offset": 0,
            "checkpoint_global_step": step,
        },
        "bindings": {
            name: {
                "path": f"fixture/{name}.bin",
                "sha256": f"{index + 1:x}" * 64,
            }
            for index, name in enumerate(R.BINDING_NAMES)
        },
        "tensor_groups": [{
            "name": "phi",
            "parameters": ["decoder.layer.0.phi.weight"],
        }],
        "intervention": {
            "arm": arm,
            "assigned_dose": None,
            "applied_after_optimizer": False,
            "description": "test fixture",
        },
        "measurements": (
            measurements if measurements is not None
            else measurement_rows(protocol_id)
        ),
        "artifacts": [],
        "environment": {
            "python": "3.14",
            "torch": "2.12.1",
            "numpy": "2.4.4",
            "device": "cpu",
            "dtype": "float64",
            "git_commit": "a" * 40,
            "source_tree_clean": True,
        },
    }


def bind_fixture_grid(registration, protocol_id, rows, base_units):
    registration[protocol_id]["sample_size"] = base_units
    registration[protocol_id]["expected_unit_keys"] = [
        copy.deepcopy(row["unit_key"]) for row in rows
    ]
    registration[protocol_id]["record_key_count"] = len(rows)


def test_q0_qualification_passes_all_fixtures():
    result = Q.run_qualification("cpu")
    assert result["status"] == "PASS"
    assert result["jacobian"]["pldr_basis_error"] < 1e-12
    assert (
        result["jacobian"]["mixed_parameter_row_derivative_error"]
        < 1e-12
    )
    assert result["jacobian"]["third_row_derivative_error"] < 1e-12
    assert result["optimizer_transport"]["maximum_parameter_error"] == 0.0
    assert result["schedule"]["nonautonomous_unroll_error"] < 1e-14
    assert result["order_parameter_bridge"]["bridge_ratio"] <= 1.0
    assert result["renormalization_group"][
        "density_identity_residual"] < 1e-13
    assert result["renormalization_group"][
        "temporal_blocking_residual"] < 1e-13
    assert result["renormalization_group"][
        "token_blocking_semigroup_residual"] < 1e-13
    assert result["renormalization_group"][
        "softmax_block_aggregation_residual"] < 1e-13
    assert all(result["record_contract"]["checks"].values())


def test_complete_row_jacobian_and_analytic_modulus():
    torch.manual_seed(7)
    model = PLDR_Model(
        num_layers=1, d_model=8, num_heads=2, dff=16,
        input_vocab_size=32, A_dff=8, num_reslayerA=1,
        num_denseA=1, max_seq_len=16, device="cpu",
    ).to(torch.float64)
    row = torch.randn(4, dtype=torch.float64)
    dense = J.dense_row_jacobian(model, 0, row)
    basis = J.basis_row_jacobian(model, 0, row)
    assert torch.linalg.matrix_norm(dense - basis, ord=2) < 1e-12
    analytic = J.analytic_row_map_modulus(model, 0, 3.0)
    assert analytic["certificate"].domain == "DECLARED_COMPACT_TUBE"
    assert analytic["derivative_modulus"] > 0.0
    cover = J.full_matrix_cover_upper(
        np.stack([basis.numpy()]), [1e-12], 1e-8,
        analytic["certificate"],
    )
    assert cover["direct_upper"] == pytest.approx(
        cover["grid_upper"] + cover["row_remainder"])
    assert cover["normalizer"] == 1.0


def _rational_float(value):
    return value["numerator"] / value["denominator"]


def test_joint_row_map_norm_certificate_encloses_live_derivatives():
    torch.manual_seed(23)
    model = PLDR_Model(
        num_layers=1, d_model=8, num_heads=2, dff=16,
        input_vocab_size=32, A_dff=8, num_reslayerA=1,
        num_denseA=1, max_seq_len=16, device="cpu",
    ).to(torch.float64)
    before = VR.model_row_map_specification(model, 0)
    row = 0.02 * torch.randn(4, dtype=torch.float64)
    radius = 0.01
    row_box = [
        {"lower": float(value - radius), "upper": float(value + radius)}
        for value in row
    ]
    with torch.no_grad():
        next(model.decoder.dec_layers[0].mha1.reslayerAs.parameters()).add_(
            1e-4)
    after = VR.model_row_map_specification(model, 0)
    certificate = VR.certify_joint_row_map_norms(
        before, after, row_box)
    assert certificate["residual_unit_bounds"]
    assert all(
        row["layernorm_denominator_lower"]["denominator"]
        <= VR.VALUE_BOUND_SCALE
        for row in certificate["residual_unit_bounds"]
    )
    first_upper = _rational_float(
        certificate["jacobian_operator_upper"])
    second_upper = _rational_float(
        certificate["derivative_lipschitz_upper"])
    third_upper = _rational_float(
        certificate["joint_third_derivative_upper"])

    jacobian_after = J.dense_row_jacobian(model, 0, row)
    assert torch.linalg.matrix_norm(
        jacobian_after, ord=2).item() <= first_upper

    hessians = torch.stack([
        torch.autograd.functional.hessian(
            lambda point, index=index: J._row_map_value(
                model, 0, point)[index],
            row,
        )
        for index in range(len(row))
    ])
    assert torch.linalg.vector_norm(hessians).item() <= second_upper

    direction = torch.randn_like(row)
    direction /= torch.linalg.vector_norm(direction)
    output_direction = torch.randn_like(row)
    output_direction /= torch.linalg.vector_norm(output_direction)
    scalar = torch.zeros((), dtype=torch.float64, requires_grad=True)
    value = output_direction @ J._row_map_value(
        model, 0, row + scalar * direction)
    first = torch.autograd.grad(value, scalar, create_graph=True)[0]
    second = torch.autograd.grad(first, scalar, create_graph=True)[0]
    third = torch.autograd.grad(second, scalar)[0]
    assert abs(third.item()) <= third_upper

    before_model = copy.deepcopy(model)
    with torch.no_grad():
        next(before_model.decoder.dec_layers[0].mha1.reslayerAs.parameters()
             ).sub_(1e-4)
    jacobian_before = J.dense_row_jacobian(before_model, 0, row)
    assert torch.linalg.vector_norm(
        jacobian_after - jacobian_before).item() <= second_upper


def test_ordered_adamw_replay_matches_torch_for_multiple_steps(tmp_path):
    theta = np.array([1.0, -2.0, 0.25], dtype=np.float64)
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    parameter = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.01, betas=(0.9, 0.95), eps=1e-5,
        weight_decay=0.1,
    )
    constants = O.AdamWConstants(clip_value=0.5)
    gradients = (
        np.array([3.0, -0.25, 0.1]),
        np.array([-0.4, 2.0, -0.75]),
        np.array([0.01, -0.02, 0.03]),
    )
    for step, gradient in enumerate(gradients, start=1):
        replay = O.ordered_adamw_step(
            theta, gradient, first, second, step=step,
            learning_rate=0.01, constants=constants)
        parameter.grad = torch.tensor(gradient, dtype=torch.float64)
        torch.nn.utils.clip_grad_value_([parameter], 0.5)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert np.allclose(
            parameter.detach().numpy(), replay["theta"],
            rtol=0.0, atol=2e-15,
        )
        theta = replay["theta"]
        first = replay["first_moment"]
        second = replay["second_moment"]

    scalar_constants = O.AdamWConstants(
        beta1=0.9, beta2=0.95, epsilon=1e-5,
        weight_decay=0.1, clip_value=0.5,
    )
    scalar_replay = O.ordered_adamw_step(
        np.array([1.0]), np.array([0.2]), np.array([0.0]),
        np.array([0.0]), step=1, learning_rate=0.01,
        constants=scalar_constants,
    )
    snapshot = {
        "schema_version": "pldr-adamw-transport-v2",
        "operation_order": [
            "averaged_raw_gradient", "value_clip", "first_moment",
            "second_moment", "bias_correction",
            "decoupled_decay_and_loss_update", "post_step_intervention",
        ],
        "learning_rate_applied": 0.01,
        "learning_rate_prepared_next": 0.009,
        "global_optimizer_step": 1,
        "blocks": {"criterion": [{
            "name": "theta",
            "raw_gradient": torch.tensor([0.2]),
            "clipped_gradient": torch.tensor(
                scalar_replay["clipped_gradient"]),
            "first_moment_before": torch.tensor([0.0]),
            "first_moment_after": torch.tensor(
                scalar_replay["first_moment"]),
            "second_moment_before": torch.tensor([0.0]),
            "second_moment_after": torch.tensor(
                scalar_replay["second_moment"]),
            "theta_before": torch.tensor([1.0]),
            "theta_after_optimizer": torch.tensor(scalar_replay["theta"]),
            "state_step_before": 0,
            "state_step_after": 1,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-5,
            "weight_decay": 0.1,
            "clip_value": 0.5,
        }]},
    }
    snapshot_path = tmp_path / "snapshot.pt"
    torch.save(snapshot, snapshot_path)
    certificate = {
        "alpha": 0.1,
        "beta1": 0.9,
        "lifted_state_before": [0.0, 0.0],
        "lifted_state_after": [0.0, 0.0],
        "lifted_matrix": [[0.0, 0.0], [0.0, 0.0]],
        "forcing": [0.0, 0.0],
        "normal_residual": [0.0],
        "decay_defect": [0.0],
        "row_motion_defect": [0.0],
        "intervention_defect": [0.0],
        "taylor_remainder": [0.0],
        "coordinate_motion_defect": [0.0],
        "recorded_complete_defect": [0.0, 0.0],
        "coordinate_motion_bound": 0.0,
        "taylor_remainder_bound": 0.0,
        "intervention_observed": [0.0],
        "intervention_expected": [0.0],
    }
    certificate_path = tmp_path / "certificate.json"
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    result = MOT.analyze_snapshot(snapshot_path, certificate_path)
    assert result["optimizer_scalar_residual"] < 1e-15
    measured = {
        row["name"]: row["value"] for row in result["measurements"]
    }
    assert measured["applied_learning_rate"] == pytest.approx(0.01)
    assert measured["schedule_increment"] == pytest.approx(-0.001)
    certificate["alpha"] = 0.2
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    mismatched = MOT.analyze_snapshot(snapshot_path, certificate_path)
    assert mismatched["optimizer_scalar_residual"] > 1e-3


def test_lifted_jury_and_quadratic_certificate():
    matrix = O.lifted_mode_matrix(
        curvature=1.0, learning_rate=0.01, beta1=0.9, step=20,
        preconditioner=1.0, weight_decay=0.0,
    )
    jury = O.jury_margins(matrix)
    assert jury["strict_schur"]
    assert jury["spectral_radius"] < 1.0
    identity = np.eye(2)
    operator = np.eye(4) - np.kron(matrix.T, matrix.T)
    metric = np.linalg.solve(
        operator, identity.reshape(-1, order="F")
    ).reshape((2, 2), order="F")
    metric = 0.5 * (metric + metric.T)
    band = O.lyapunov_band([matrix], metric)
    assert band["certified"]
    assert band["contraction_factor"] < 1.0


def test_constructed_metric_robust_radius_and_direct_horizon():
    nominal = np.array([[0.8, 0.1], [0.0, 0.7]])
    construction = D.solve_discrete_lyapunov(nominal)
    assert construction["relative_residual"] < 1e-12
    assert construction["metric_lower"] > 0.0
    family = [nominal, nominal + 1e-3 * np.eye(2)]
    robust = D.robust_family_certificate(
        nominal,
        family,
        construction["metric"],
        construction["nominal_q_bound"],
    )
    assert robust["robust_q"] < 1.0
    assert robust["maximum_exact_h_gain"] <= robust["robust_q"]
    assert D.normal_residual_envelope(
        displacement_norm=0.2,
        linear_residual_bound=0.3,
        second_derivative_bound=0.4,
    ) == pytest.approx(0.068)
    horizon = D.monotone_entry_horizon(
        q=0.8,
        r=0.7,
        initial_h_norm=1.0,
        forcing_constant=0.1,
        persistent_forcing=0.01,
        metric_lower=1.0,
        cover_floor=0.05,
        cover_transient_constant=0.1,
        cover_rate=0.6,
        criterion=0.5,
    )
    assert horizon["horizon"] >= horizon["monotonicity_start"]
    assert horizon["strict_margin"] > 0.0
    assert horizon["complete_persistent_floor"] == pytest.approx(0.1)
    schedule = D.nonautonomous_schedule_bounds(
        gains=[0.8, 0.7],
        disturbance_bounds=[0.1, 0.05],
        initial_h_norm=1.0,
        metric_lower=1.0,
        cover_floor=0.0,
        cover_transient_constant=0.0,
        cover_rate=0.5,
        criterion=0.8,
    )
    assert schedule["lifted_bounds"] == pytest.approx([1.0, 0.9, 0.68])
    assert schedule["cumulative_products"] == pytest.approx([1.0, 0.8, 0.56])
    assert schedule["first_sustained_relative_entry"] == 2
    geometric = MD.measure_e4({
        "metric": np.eye(2).tolist(),
        "forcing_vectors": [[0.1, 0.0], [0.05, 0.0]],
        "complete_defect_vectors": [[0.0, 0.0], [0.0, 0.0]],
        "complete_disturbance_vectors": [[0.1, 0.0], [0.05, 0.0]],
        "tail_offsets": [0, 1],
        "persistent_constant": 0.0,
        "geometric_constant": 0.1,
        "geometric_rate": 0.5,
        "comparison_matrix": [[0.5, 0.0], [0.0, 0.25]],
        "comparison_weights": [1.0, 1.0],
        "comparison_kappa": 0.5,
    })
    assert geometric["derived_disturbance_envelopes"] == [0.1, 0.05]
    geometric_measurements = {
        row["name"]: row["value"]
        for row in geometric["measurements"]
    }
    assert geometric_measurements["envelope_ratio"] == pytest.approx(1.0)
    assert geometric_measurements[
        "positive_comparison_witness_ratio"] == pytest.approx(1.0)
    finite = MD.measure_e6({
        "certificate_time": 100,
        "q": 0.9,
        "r": 0.8,
        "initial_h_norm": 100.0,
        "forcing_constant": 0.0,
        "persistent_forcing_constant": 0.0,
        "path_window_length": 1,
        "path_window_gain": 0.9,
        "block_forcing_constant": 0.0,
        "block_persistent_forcing_constant": 0.0,
        "block_rate": 0.8,
        "metric_lower": 1.0,
        "cover_floor": 0.0,
        "cover_transient_constant": 0.0,
        "cover_rate": 0.8,
        "criterion": 0.5,
        "observed_entry_step": 102,
        "direct_row_map_upper": 0.0,
        "sustained_entry_indicator": 0.0,
        "schedule_gains": [0.9],
        "schedule_disturbance_bounds": [0.0],
    })
    assert finite["predicted_entry_step"] == 102
    assert not finite["finite_entry"]
    assert finite["relative_horizon"]["reason_code"] == (
        "NO_MONOTONE_ENTRY_THROUGH_REGISTERED_TERMINAL")


def test_exact_convex_vertex_and_graph_window_certificates():
    vertices = [
        [[Fraction(1, 2), 0], [0, Fraction(1, 2)]],
        [[Fraction(3, 4), 0], [0, Fraction(1, 2)]],
    ]
    metric = [[1, 0], [0, 1]]
    certificate = VLA.convex_family_lyapunov_certificate(vertices, metric)
    assert VLA.check_convex_family_lyapunov_certificate(certificate)
    gain = Fraction(
        certificate["gain_upper"]["numerator"],
        certificate["gain_upper"]["denominator"],
    )
    assert gain < 1
    graph = VLA.graph_window_certificate([
        {
            "edge_id": "stay",
            "source": "tail",
            "target": "tail",
            "gain_upper": certificate["gain_upper"],
        },
    ], 3)
    assert VLA.check_graph_window_certificate(graph)
    maximum = Fraction(
        graph["maximum_window_gain"]["numerator"],
        graph["maximum_window_gain"]["denominator"],
    )
    assert maximum == gain ** 3

    expanding = VLA.convex_family_lyapunov_certificate(
        [[[1]]], [[1]], target_metric=[[4]], gain_squared=Fraction(5))
    restoring = VLA.convex_family_lyapunov_certificate(
        [[[Fraction(1, 10)]]], [[4]], target_metric=[[1]],
        gain_squared=Fraction(1, 100))
    assert not expanding["strictly_contracting"]
    assert restoring["strictly_contracting"]
    alternating = VLA.graph_window_certificate([
        {
            "edge_id": "expand",
            "source": "low",
            "target": "high",
            "gain_upper": expanding["gain_upper"],
        },
        {
            "edge_id": "restore",
            "source": "high",
            "target": "low",
            "gain_upper": restoring["gain_upper"],
        },
    ], 2)
    assert VLA.check_graph_window_certificate(alternating)

    cross_metric = PC.cross_metric_family_certificate(
        [[[Fraction(3, 5), 0], [0, Fraction(3, 5)]]],
        [[1, 0], [0, 1]],
        [[4, 0], [0, 4]],
    )
    assert VLA.check_convex_family_lyapunov_certificate(cross_metric)
    cross_gain = Fraction(
        cross_metric["gain_upper"]["numerator"],
        cross_metric["gain_upper"]["denominator"],
    )
    assert cross_gain > 1


def test_multiaffine_primitive_box_reconstructs_and_contracts():
    beta1 = 0.9
    primitive_bounds = {
        "alpha": [0.01, 0.0101],
        "decay": [0.001, 0.0011],
        "normal_eigenvalue": [0.5, 0.51],
    }
    _, corners = PC.multiaffine_corner_matrices(
        primitive_bounds, beta1)
    point = {
        "alpha": 0.01004,
        "decay": 0.00104,
        "normal_eigenvalue": 0.504,
    }
    weights = PC.multiaffine_weights(primitive_bounds, point)
    interior = D.lifted_block_matrix(
        np.asarray([[point["normal_eigenvalue"]]]),
        point["decay"],
        alpha=point["alpha"],
        beta1=beta1,
    )
    reconstructed = sum(
        (weight * corner for weight, corner in zip(weights, corners)),
        np.zeros((2, 2)),
    )
    assert weights.sum() == pytest.approx(1.0)
    assert np.min(weights) >= 0.0
    assert reconstructed == pytest.approx(interior)
    certificate = PC.structured_family_certificate(
        interior, [*corners, interior], beta1, window_length=4)
    assert certificate["strictly_contracting"]
    assert certificate["structured_q"] < 1.0
    assert certificate["maximum_window_gain"] < 1.0
    assert certificate["maximum_reconstruction_residual"] < 1e-12


def test_registered_chord_cover_avoids_ambient_box():
    cover = RD.cover_segment_hull(
        [[0, 0], [2, 0], [0, 2]],
        subdivisions_per_segment=2,
        maximum_boxes=6,
    )
    assert cover["status"] == "CERTIFIED"
    assert cover["box_count"] == 6
    assert cover["domain"]["type"] == "registered_pairwise_segment_hull"
    h = Fraction(
        cover["h_upper"]["numerator"],
        cover["h_upper"]["denominator"],
    )
    assert float(h) == pytest.approx(2 ** 0.5 / 2.0)
    infeasible = RD.cover_segment_hull(
        [[0, 0], [2, 0], [0, 2]],
        subdivisions_per_segment=3,
        maximum_boxes=8,
    )
    assert infeasible["status"] == "INFEASIBLE"


def test_row_domain_construction_and_validation_are_disjoint():
    assert RD.validate_construction_validation_split(
        ["construction-0", "construction-1"], ["validation-0"])
    with pytest.raises(ValueError, match="used to construct"):
        RD.validate_construction_validation_split(["shared"], ["shared"])
    hull = RD.interval_hull([[0, 0], [1, 1]])
    tiling = RD.tile_interval_hull(hull, 1)
    witnesses = RD.points_in_rational_boxes(
        [[Fraction(1, 2), Fraction(1, 2)], [2, 2]], tiling["boxes"])
    assert witnesses[0]["member"]
    assert not witnesses[1]["member"]


def test_deductive_output_order_parameter_bridge():
    operator = np.diag([0.5, 0.25])
    rows_a = np.array([[1.0, 0.5], [0.5, 1.5], [1.5, 1.0]])
    rows_b = 0.8 * rows_a
    bridge = D.deductive_output_order_bridge(
        output_a=rows_a @ operator.T,
        output_b=rows_b @ operator.T,
        rows_a=rows_a,
        rows_b=rows_b,
        direct_row_map_upper=np.linalg.norm(operator, ord=2),
    )
    assert bridge["bridge_holds"]
    assert bridge["order_parameter_bridge_ratio"] <= 1.0
    assert bridge["source_order_parameter"] == pytest.approx(
        bridge["source_rmse"] / bridge["tensor_mean_abs"])
    with pytest.raises(ValueError, match="mean_floor"):
        D.deductive_output_order_bridge(
            output_a=rows_a @ operator.T,
            output_b=rows_b @ operator.T,
            rows_a=rows_a,
            rows_b=rows_b,
            direct_row_map_upper=np.linalg.norm(operator, ord=2),
            mean_floor=10.0,
        )


def test_query_density_rg_identity_and_layernorm_rescaling():
    queries = np.array([
        [1.0, 0.0], [0.5, 0.5],
        [0.0, 1.0], [0.25, 0.75],
    ])
    result = RG.query_density_coarse_graining(queries, 2)
    assert result["density_identity_residual"] < 1e-15
    assert result["within_covariance_min_eigenvalue"] >= -1e-15
    assert result["within_covariance_trace"] == pytest.approx(
        result["mean_squared_within_block_deviation"])
    exact_blocks = np.repeat(np.array([[1.0, 0.0], [0.0, 1.0]]), 2, axis=0)
    exact = RG.query_density_coarse_graining(exact_blocks, 2)
    assert np.linalg.norm(exact["within_block_covariance"]) == 0.0
    layernorm = RG.metric_layernorm_scale_identity(
        result["coarse_normalized_density"],
        block_size=2,
        epsilon=1e-6,
        gamma=[0.9, 1.1],
        beta=[0.2, -0.1],
    )
    assert layernorm["relative_residual"] < 1e-13


def test_temporal_rg_blocking_semigroup_and_softmax_aggregation():
    matrices = np.array([
        [[0.8, 0.1], [0.0, 0.7]],
        [[0.7, 0.0], [0.1, 0.6]],
        [[0.6, 0.1], [0.0, 0.5]],
        [[0.5, 0.0], [0.1, 0.4]],
    ])
    forcings = np.array([
        [0.1, 0.0], [0.0, 0.1], [0.05, 0.0], [0.0, 0.02],
    ])
    initial = np.array([1.0, -0.5])
    fine = RG.apply_affine_cocycle(matrices, forcings, initial)
    blocked_two = RG.block_affine_cocycle(matrices, forcings, 2)
    coarse = RG.apply_affine_cocycle(
        blocked_two["matrices"], blocked_two["forcings"], initial)
    assert coarse == pytest.approx(fine[::2])
    blocked_four = RG.block_affine_cocycle(matrices, forcings, 4)
    twice = RG.block_affine_cocycle(
        blocked_two["matrices"], blocked_two["forcings"], 2)
    assert blocked_four["matrices"] == pytest.approx(twice["matrices"])
    assert blocked_four["forcings"] == pytest.approx(twice["forcings"])
    assert RG.block_average_semigroup_residual(
        np.arange(16, dtype=float).reshape(8, 2), 2, 2) == 0.0
    softmax = RG.block_softmax_aggregation(
        [0.2, -0.1, 0.7, 0.8, -0.5, 0.3], 2)
    assert softmax["aggregation_residual"] < 1e-15
    assert softmax["mean_logit_approximation_error"] > 0.0
    partial_causal_block = RG.block_softmax_aggregation(
        [0.2, -0.1, 0.7, 0.8, -0.5], 2)
    assert partial_causal_block["aggregation_residual"] < 1e-15
    assert len(partial_causal_block["fine_block_mass"]) == 3


def test_deductive_and_full_layer_rg_closure_bounds():
    queries = np.array([
        [1.0, 0.0], [0.8, 0.2],
        [0.1, 0.9], [0.2, 0.8],
    ])
    preliminary = RG.deductive_rg_closure_bound(
        rotated_queries=queries,
        block_size=2,
        layernorm_lipschitz=1.5,
        direct_row_map_upper=0.2,
        downstream_lipschitz=2.0,
    )
    bound = preliminary["deductive_closure_upper"]
    result = RG.deductive_rg_closure_bound(
        rotated_queries=queries,
        block_size=2,
        layernorm_lipschitz=1.5,
        direct_row_map_upper=0.2,
        downstream_lipschitz=2.0,
        observed_deductive_difference=0.5 * bound,
    )
    assert result["deductive_closure_holds"]
    assert result["deductive_closure_ratio"] == pytest.approx(0.5)
    telescope = RG.telescoping_closure_bound(
        local_defects=[0.1, 0.2, 0.3],
        downstream_lipschitz=[7.0, 2.0, 3.0],
    )
    assert telescope["transported_contributions"] == pytest.approx(
        [0.6, 0.6, 0.3])
    assert telescope["closure_upper"] == pytest.approx(1.5)


def test_rg_measurement_adapter_closes_registered_quantities():
    value = {
        "rotated_queries": [
            [1.0, 0.0], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8],
            [0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.4, 0.6],
        ],
        "block_size": 2,
        "repeat_block_size": 2,
        "metric_layernorm_epsilon": 1e-6,
        "metric_layernorm_gamma": [0.9, 1.1],
        "metric_layernorm_beta": [0.1, -0.1],
        "layernorm_lipschitz": 1.5,
        "direct_row_map_upper": 0.2,
        "deductive_downstream_lipschitz": 2.0,
        "observed_deductive_difference": 0.0,
        "attention_logits": [
            0.2, -0.1, 0.7, 0.8, -0.5, 0.3, 1.0, 0.6,
        ],
        "full_layer_other_defect_bounds": {
            "input_projection_rope": 0.1,
            "scores": 0.2,
            "mask_softmax": 0.1,
            "values_output": 0.2,
            "residual_ffn": 0.1,
        },
        "full_layer_stage_lipschitz": {
            "input_projection_rope": 1.0,
            "deductive_map": 1.1,
            "scores": 1.2,
            "mask_softmax": 1.1,
            "values_output": 1.2,
            "residual_ffn": 1.1,
        },
        "observed_full_layer_difference": 0.0,
    }
    result = MRG.measure(value)
    measured = {
        row["name"]: row["value"] for row in result["measurements"]
    }
    assert set(measured) == {
        name for name in R.MEASUREMENT_NAMES["E8"]
        if name.startswith("rg_")
    }
    assert measured["rg_density_identity_residual"] < 1e-13
    assert measured["rg_layernorm_scale_residual"] < 1e-13
    assert measured["rg_softmax_aggregation_residual"] < 1e-13
    assert measured["rg_repeated_block_semigroup_residual"] < 1e-13
    assert measured["rg_deductive_closure_ratio"] == 0.0
    assert measured["rg_full_layer_closure_ratio"] == 0.0


def test_e2_modes_are_derived_from_a_self_adjoint_normal_operator():
    normal = np.diag([1.0, 2.0])
    state = np.array([0.1, 0.2])
    value = {
        "normal_operator": normal.tolist(),
        "normal_velocity": (normal @ state).tolist(),
        "jacobian_state": state.tolist(),
        "reference_forcing": [0.0, 0.0],
        "normal_residual": [0.0, 0.0],
        "linearization_residual_norm": 0.1,
        "linearization_residual_bound": 0.2,
        "nonlinear_residual_norm": 0.1,
        "nonlinear_residual_bound": 0.2,
        "alpha": 0.1,
        "beta1": 0.9,
        "scalar_decay": 0.01,
    }
    result = MD.measure_e2(value)
    measured = {
        row["name"]: row["value"] for row in result["measurements"]
    }
    assert measured["normal_operator_self_adjoint_residual"] == 0.0
    assert measured["lower_normal_edge"] == pytest.approx(1.0)
    assert measured["upper_normal_edge"] == pytest.approx(2.0)
    assert measured["jury_margin_min"] > 0.0
    assert len(result["spectrum_derived_scalar_mode_matrices"]) == 2

    registration = copy.deepcopy(A.load_registration(
        G.OUT / "registration.json"))
    registration["E2"]["sample_size"] = 2
    rows = [
        record(
            "E2",
            seed=seed,
            measurements=measurement_rows("E2", measured),
        )
        for seed in (1, 2)
    ]
    bind_fixture_grid(registration, "E2", rows, 2)
    assert A.analyze(rows, registration)["experiments"]["E2"]["status"] \
        == "CONFIRMED"
    self_adjoint_row = next(
        row for row in rows[0]["measurements"]
        if row["name"] == "normal_operator_self_adjoint_residual"
    )
    self_adjoint_row["value"] = 1e-3
    assert A.analyze(rows, registration)["experiments"]["E2"]["status"] \
        == "NOT_CONFIRMED"


def test_componentwise_self_map_and_e3_measurement_adapter():
    budget = D.componentwise_self_map_budget(
        q=0.8,
        lifted_radius=1.0,
        disturbance_bound=0.1,
        next_lifted_radius=0.95,
        center_errors=[0.01, 0.02],
        lifted_lipschitz=[0.1, 0.1],
        auxiliary_lipschitz=[[0.2, 0.0], [0.0, 0.2]],
        auxiliary_radii=[0.1, 0.1],
        next_auxiliary_radii=[0.2, 0.2],
    )
    assert budget["certified"]
    path_budget = MD.measure_e5({
        "q": 1.2,
        "lifted_radius": 1.0,
        "disturbance_bound": 0.0,
        "next_lifted_radius": 1.2,
        "center_errors": [0.0],
        "lifted_lipschitz": [0.0],
        "auxiliary_lipschitz": [[0.0]],
        "auxiliary_radii": [0.0],
        "next_auxiliary_radii": [0.0],
        "initial_membership_margin": 0.0,
        "heldout_image_margins": [0.0],
        "family_box_enclosure_ratio": 1.0,
        "path_gains": [1.2, 0.5],
        "path_disturbance_bounds": [0.0, 0.0],
        "path_lifted_radii": [1.0, 1.2, 1.0],
        "certified_path_window_gain": 0.6,
    })
    assert path_budget["path_budget"]["certified"]
    assert path_budget["path_budget"]["block_lifted_slack"] \
        == pytest.approx(0.4)
    all_paths = MD._path_window_budget(
        gains=[[1.2, 0.5], [0.6, 0.9]],
        disturbances=[[0.0, 0.0], [0.0, 0.0]],
        radii=[[1.0, 1.2, 1.0], [1.0, 1.0, 1.0]],
        certified_window_gain=0.6,
    )
    assert all_paths["admissible_path_count"] == 2
    assert all_paths["edge_gain_products"] == pytest.approx([0.6, 0.54])
    assert all_paths["certified"]
    result = MD.measure_e3({
        "nominal_matrix": [[0.8, 0.1], [0.0, 0.7]],
        "q_matrix": [[1.0, 0.0], [0.0, 1.0]],
        "family_matrices": [
            [[0.8, 0.1], [0.0, 0.7]],
            [[0.801, 0.1], [0.0, 0.7]],
        ],
        "euclidean_interval_radius": 0.01,
    })
    measured = {
        row["name"]: row["value"] for row in result["measurements"]
    }
    assert measured["nominal_lyapunov_residual"] < 1e-12
    assert measured["metric_eigenvalue_min"] > 0.0
    assert measured["robust_q"] < 1.0
    with pytest.raises(ValueError, match="does not enclose"):
        MD.measure_e3({
            "nominal_matrix": [[0.8, 0.1], [0.0, 0.7]],
            "q_matrix": [[1.0, 0.0], [0.0, 1.0]],
            "family_matrices": [
                [[0.8, 0.1], [0.0, 0.7]],
                [[0.801, 0.1], [0.0, 0.7]],
            ],
            "euclidean_interval_radius": 1e-5,
        })


def test_record_contract_is_closed_and_clock_complete():
    value = record()
    assert R.validate_record(value) is value
    malformed = copy.deepcopy(value)
    malformed["unknown"] = 1
    with pytest.raises(R.RecordError, match="unknown"):
        R.validate_record(malformed)
    malformed = copy.deepcopy(value)
    malformed["clocks"]["global_optimizer_step"] += 1
    with pytest.raises(R.RecordError, match="global clock"):
        R.validate_record(malformed)
    malformed = copy.deepcopy(value)
    malformed["measurements"][0]["value"] = float("nan")
    with pytest.raises(R.RecordError, match="finite"):
        R.validate_record(malformed)


def test_strict_json_and_path_bindings(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a": 1, "a": 2}\n', encoding="utf-8")
    with pytest.raises(R.RecordError, match="duplicate"):
        R.strict_load(duplicate)
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"first")
    binding = R.binding_for(payload, tmp_path)
    assert R.verify_binding(binding, tmp_path, "payload") == payload
    payload.write_bytes(b"second")
    with pytest.raises(R.RecordError, match="stale"):
        R.verify_binding(binding, tmp_path, "payload")




def test_e0_analysis_uses_cover_and_joint_derivative_enclosures():
    registration = A.load_registration(
        G.OUT / "registration.json")
    registration = copy.deepcopy(registration)
    registration["E0"]["sample_size"] = 6
    rows = []
    for seed in range(6):
        rows.append(record(
            "E0", seed=seed + 1,
            measurements=measurement_rows("E0", {
                "direct_row_map_upper": 1.20 + seed * 0.005,
                "grid_jacobian_upper": 1.17,
                "cover_remainder": 0.03,
                "cover_identity_error": 1e-14,
                "cover_utility_margin": 0.02,
                "validation_domain_membership_indicator": 1.0,
                "heldout_jacobian_ratio": 0.80 + seed * 0.005,
                "mixed_derivative_enclosure_ratio": 0.75,
                "third_derivative_enclosure_ratio": 0.70,
                "numerical_error": 1e-12,
            }),
        ))
    bind_fixture_grid(registration, "E0", rows, 6)
    result = A.analyze(rows, registration)
    assert result["experiments"]["E0"]["status"] == "CONFIRMED"
    rows[0]["measurements"][0] = {
        **rows[0]["measurements"][0],
        "value": None,
        "status": "ERROR",
        "reason_code": "MEASUREMENT_ERROR",
    }
    assert A.analyze(rows, registration)["experiments"]["E0"]["status"] \
        == "INCOMPLETE"


def test_e6_confirms_a_valid_conservative_sufficient_horizon():
    registration = copy.deepcopy(A.load_registration(
        G.OUT / "registration.json"))
    registration["E6"]["sample_size"] = 2
    rows = []
    for seed, observed in ((1, 10.0), (2, 20.0)):
        rows.append(record(
            "E6", seed=seed,
            measurements=measurement_rows("E6", {
                "predicted_entry_step": 100.0,
                "observed_entry_step": observed,
                "entry_error": observed - 100.0,
                "residual_floor": 0.1,
                "direct_row_map_upper": 0.2,
                "criterion": 0.5,
                "sustained_entry_indicator": 1.0,
                "path_window_length": 4.0,
                "path_window_gain": 0.9,
                "block_product_convolution_upper": 0.3,
            }),
        ))
    bind_fixture_grid(registration, "E6", rows, 2)
    result = A.analyze(rows, registration)["experiments"]["E6"]
    assert result["status"] == "CONFIRMED"
    assert result["summary"]["median_absolute_relative_error"] > 0.5


def test_e8_uses_the_registered_physical_direct_criterion():
    registration = copy.deepcopy(A.load_registration(
        G.OUT / "registration.json"))
    registration["E8"]["sample_size"] = 12
    rows = []
    for seed in range(1, 13):
        common = {
            "model_width": 64.0,
            "model_depth": 4.0,
            "direct_row_map_upper": 1.5,
            "criterion": 2.0,
            "predicted_entry_step": 100.0,
            "observed_entry_step": 20.0,
            "heldout_loss": 0.5,
            "baseline_indicator": 0.0,
            "theorem_bound_ratio": 0.9,
            "maximum_learning_rate": 1e-3,
            "warmup_steps": 1000.0,
            "anneal_floor": 0.1,
            "source_order_parameter": 0.2,
            "source_rmse": 0.1,
            "tensor_mean_abs": 0.5,
            "row_input_spread": 1.0,
            "downstream_lipschitz": 1.0,
            "order_parameter_upper": 0.4,
            "order_parameter_bridge_ratio": 0.5,
            "avalanche_event_rate": 0.1,
            "avalanche_mean_size": 1.0,
            "avalanche_mean_duration": 2.0,
            "avalanche_mean_support": 3.0,
            "avalanche_tail_comparison": 0.2,
            "finite_size_scaling_error": 0.1,
            "rg_block_size": 2.0,
            "rg_density_identity_residual": 1e-14,
            "rg_layernorm_scale_residual": 1e-14,
            "rg_within_covariance_trace": 0.1,
            "rg_deductive_closure_upper": 0.2,
            "rg_deductive_closure_ratio": 0.5,
            "rg_softmax_aggregation_residual": 1e-14,
            "rg_repeated_block_semigroup_residual": 1e-14,
            "rg_full_layer_closure_upper": 0.4,
            "rg_full_layer_closure_ratio": 0.5,
        }
        for block_size in (2.0, 4.0, 8.0):
            scaled = {**common, "rg_block_size": block_size}
            rows.append(record(
                "E8", seed=seed, arm="theory",
                block_size=int(block_size), architecture_cell="fixture_arch",
                schedule_cell="fixture_schedule",
                measurements=measurement_rows("E8", {
                    **scaled, "model_score": 0.1,
                }),
            ))
            for index, arm in enumerate(
                ("constant_rate", "loss_only", "unconstrained_trend"),
                start=1,
            ):
                rows.append(record(
                    "E8", seed=seed, arm=arm,
                    block_size=int(block_size), architecture_cell="fixture_arch",
                    schedule_cell="fixture_schedule",
                    measurements=measurement_rows("E8", {
                        **scaled,
                        "baseline_indicator": 1.0,
                        "model_score": 0.3 + 0.1 * index,
                    }),
                ))
    bind_fixture_grid(registration, "E8", rows, 12)
    result = A.analyze(rows, registration)["experiments"]["E8"]
    assert result["status"] == "CONFIRMED"
    assert result["summary"]["maximum_direct_to_criterion_ratio"] == 0.75
    assert result["summary"][
        "maximum_order_parameter_bridge_ratio"] == 0.5
    assert result["summary"]["maximum_rg_deductive_closure_ratio"] == 0.5
    assert result["summary"]["maximum_rg_full_layer_closure_ratio"] == 0.5
    assert result["summary"]["complete_theorem_groups"] == 12
    assert result["summary"]["complete_rg_theory_rows"] == 36
    assert A.analyze(rows[:-1], registration)["experiments"]["E8"][
        "status"
    ] == "INCOMPLETE"


def test_empty_analysis_is_total():
    registration = A.load_registration(
        G.OUT / "registration.json")
    result = A.analyze([], registration)
    assert result["analysis_status"] == "NOT_RUN"
    assert all(
        row["status"] == "NOT_RUN"
        for row in result["experiments"].values()
    )
