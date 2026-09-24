"""Synthetic campaign-record tests for the rev36 analyzers."""

from __future__ import annotations

import numpy as np

from analysis.analyze_row_map_confirmation import (
    analyze_dynamics,
    analyze_interventions,
    analyze_loss,
    analyze_plga,
)
from confirm.gate_shape import adamw_gate_block, plga_secant_chain
from confirm.gate_shape_evidence import write_npz_atomic
from confirm.row_map_dynamics import dimensionless_state_scale


def _dynamics_archive(path, run_name, seed, layer):
    updates = 16
    dimension = 2
    learning_rate = 7.5e-4
    weight_decay = 0.1
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1e-5
    clip_value = 1.0
    gamma = np.asarray([0.2, -0.1])
    first = np.asarray([0.01, 0.02])
    second = np.asarray([0.2, 0.3])
    q = np.asarray([1.0, 2.0])
    rows = []
    for index in range(updates):
        step_after = 24001 + index
        raw_gradient = np.asarray([0.03, -0.02]) + index * 1e-5
        true_hessian = np.asarray([[0.2, 0.01], [0.01, 0.15]])
        block = adamw_gate_block(
            true_hessian, gamma, first, second, raw_gradient,
            learning_rate=learning_rate, beta1=beta1, beta2=beta2,
            epsilon=epsilon, optimizer_step=step_after,
            weight_decay=weight_decay, clip_value=clip_value)
        gamma_after = block["gamma"]
        q_after = q.copy()
        energy_before = float(np.sum(q * gamma * gamma))
        fixed_energy = float(np.sum(q * gamma_after * gamma_after))
        energy_after = float(np.sum(q_after * gamma_after * gamma_after))
        source_scale = dimensionless_state_scale(
            second / (1.0 - beta2 ** (step_after - 1)),
            epsilon=epsilon)
        target_scale = dimensionless_state_scale(
            block["second_moment"] / (1.0 - beta2 ** step_after),
            epsilon=epsilon)
        adaptive = (
            (1.0 - learning_rate * weight_decay) * gamma - gamma_after
        ) / learning_rate
        rows.append({
            "gamma_before": gamma.copy(),
            "gamma_optimizer_after": gamma_after.copy(),
            "gamma_after": gamma_after.copy(),
            "adaptive_direction": adaptive,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "q_before": q.copy(),
            "q_after": q_after.copy(),
            "energy_before": energy_before,
            "energy_after": energy_after,
            "gate_work": fixed_energy - energy_before,
            "shape_work": energy_after - fixed_energy,
            "loss": 4.0 - 0.01 * index,
            "q_directional": np.zeros(dimension),
            "q_remainder": np.zeros(dimension),
            "shape_jvp_source_residual": 0.0,
            "block_first_moment": first.copy(),
            "block_second_moment": second.copy(),
            "block_raw_gradient": raw_gradient,
            "block_true_hessian": true_hessian,
            "block_operator": block["operator"],
            "block_source_state_scale": source_scale,
            "block_target_state_scale": target_scale,
            "block_beta1": beta1,
            "block_beta2": beta2,
            "block_adam_epsilon": epsilon,
            "block_clip_value": clip_value,
            "block_optimizer_step_after": step_after,
            "block_gradient_replay_residual": 0.0,
            "block_clipping_boundary_distance": float(
                np.min(np.abs(np.abs(raw_gradient) - clip_value))),
        })
        gamma = gamma_after
        first = block["first_moment"]
        second = block["second_moment"]
        q = q_after
    payload = {
        "schema_version": np.asarray(
            "pldr-row-map-intervention-observables-v2"),
        "arm": np.asarray("baseline_continue"),
        "run_name": np.asarray(run_name),
        "seed": np.asarray(seed, dtype=np.int64),
        "layer": np.asarray(layer, dtype=np.int64),
        "capture_blocks": np.asarray(True),
        "local_step": np.arange(1, updates + 1, dtype=np.int64),
    }
    for name in rows[0]:
        payload[name] = np.asarray([row[name] for row in rows])
    write_npz_atomic(path, **payload)


def test_dynamics_analyzer_reconstructs_memory_blocks_and_products(tmp_path):
    paths = []
    runs = (
        ("w8-const-lr7.5e-4", 1234),
        ("w8-const-lr7.5e-4-s2", 2222),
        ("w8-const-lr7.5e-4-s3", 3333),
    )
    for run_name, seed in runs:
        for layer in range(3):
            path = tmp_path / f"{seed}-l{layer}.npz"
            _dynamics_archive(path, run_name, seed, layer)
            paths.append(path)
    report = analyze_dynamics(paths)
    assert report["all_checks_pass"]
    assert all(report["checks"].values())
    assert all(
        row["maximum_block_reconstruction_residual"] < 1e-12
        for row in report["results"]["windows"])


def test_loss_analyzer_rebuilds_nonpadding_weighted_aggregates(tmp_path):
    counts = np.asarray([1.0, 3.0])
    weights = counts / counts.sum()
    loss_terms = np.asarray([2.0, 4.0])
    gradient_terms = np.asarray([[1.0, -1.0], [3.0, 1.0]])
    fisher_terms = np.asarray([
        np.diag([0.2, 0.1]),
        np.diag([0.4, 0.3]),
    ])
    signed_terms = np.asarray([
        np.diag([0.05, -0.02]),
        np.diag([0.03, 0.01]),
    ])
    true_terms = fisher_terms + signed_terms
    force_terms = np.asarray([[0.1, -0.2], [0.3, 0.2]])
    fisher = np.sum(weights[:, None, None] * fisher_terms, axis=0)
    true = np.sum(weights[:, None, None] * true_terms, axis=0)
    force = np.sum(weights[:, None] * force_terms, axis=0)
    archive = tmp_path / "loss.npz"
    write_npz_atomic(
        archive,
        checkpoint_sha256=np.asarray("a" * 64),
        run_name=np.asarray("w8-const-lr7.5e-4"),
        seed=np.asarray(1234, dtype=np.int64),
        role=np.asarray("construction"),
        step=np.asarray(4000, dtype=np.int64),
        layer=np.asarray(0, dtype=np.int64),
        loss_value=np.asarray(float(weights @ loss_terms)),
        gradient_at_gate=np.sum(
            weights[:, None] * gradient_terms, axis=0),
        fisher=fisher,
        true_hessian=true,
        signed_second_jet=true - fisher,
        force_at_zero=force,
        loss_terms=loss_terms,
        gradient_at_gate_terms=gradient_terms,
        fisher_terms=fisher_terms,
        true_hessian_terms=true_terms,
        force_at_zero_terms=force_terms,
        source_count_terms=counts,
    )
    report = analyze_loss([archive])
    assert report["all_checks_pass"]
    row = report["results"]["loss_sectors"][0]
    assert row["total_nonpadding_sources"] == 4
    assert row["weighted_aggregate_relative_residual"] == 0.0


def _intervention_rows(arm):
    updates = 16
    lr = 7.5e-4
    wd = 0.1
    gamma = np.asarray([0.5, -0.3])
    q = np.asarray([1.0, 2.0])
    source_gamma = gamma.copy()
    source_q = q.copy()
    rows = []
    for index in range(updates):
        gamma_before = gamma.copy()
        q_before = q.copy()
        optimizer_after = 0.99 * gamma_before
        if arm == "freeze_final_gate":
            gamma_after = source_gamma.copy()
            q_after = 0.98 * q_before
        elif arm == "freeze_normalized_shape":
            gamma_after = optimizer_after.copy()
            q_after = source_q.copy()
        elif arm == "disable_final_gate_decay":
            gamma_after = optimizer_after + lr * wd * gamma_before
            q_after = 0.98 * q_before
        else:
            gamma_after = optimizer_after.copy()
            q_after = 0.98 * q_before
        energy_before = float(np.sum(q_before * gamma_before ** 2))
        fixed_energy = float(np.sum(q_before * gamma_after ** 2))
        energy_after = float(np.sum(q_after * gamma_after ** 2))
        rows.append({
            "gamma_before": gamma_before,
            "gamma_optimizer_after": optimizer_after,
            "gamma_after": gamma_after,
            "adaptive_direction": (
                (1.0 - lr * wd) * gamma_before - optimizer_after) / lr,
            "learning_rate": lr,
            "weight_decay": wd,
            "q_before": q_before,
            "q_after": q_after,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "gate_work": fixed_energy - energy_before,
            "shape_work": energy_after - fixed_energy,
            "loss": 4.0 - 0.01 * index,
        })
        gamma = gamma_after
        q = q_after
    return rows


def test_intervention_analyzer_requires_signal_retaining_controls(tmp_path):
    paths = []
    for arm in (
        "baseline_continue", "freeze_final_gate",
        "disable_final_gate_decay", "freeze_normalized_shape",
        "replay_physical_rows",
    ):
        rows = _intervention_rows(arm)
        path = tmp_path / f"{arm}.npz"
        payload = {
            "schema_version": np.asarray(
                "pldr-row-map-intervention-observables-v2"),
            "arm": np.asarray(arm),
            "run_name": np.asarray("w8-const-lr7.5e-4"),
            "seed": np.asarray(1234, dtype=np.int64),
            "layer": np.asarray(2, dtype=np.int64),
            "local_step": np.arange(1, 17, dtype=np.int64),
        }
        for name in rows[0]:
            payload[name] = np.asarray([row[name] for row in rows])
        write_npz_atomic(path, **payload)
        paths.append(path)
    report = analyze_interventions(paths)
    assert report["all_checks_pass"]
    assert report["checks"]["gate_freeze_retains_shape_signal"]
    assert report["checks"]["shape_freeze_retains_gate_signal"]



def test_plga_analyzer_uses_construction_bound_for_heldout_margins(tmp_path):
    paths = []
    runs = (
        ("w8-const-lr7.5e-4", 1234, "construction", 0.005),
        ("w8-const-lr7.5e-4-s2", 2222, "validation", 0.007),
        ("w8-const-lr7.5e-4-s3", 3333, "validation", 0.008),
    )
    weight = np.asarray([[0.8, 0.1], [-0.1, 0.7]])
    bias = np.asarray([[0.1, -0.2], [0.05, 0.15]])
    powers = np.asarray([[-0.4, 0.0], [0.6, 1.2]])
    coupling = np.asarray([[0.9, 0.05], [0.02, 0.8]])
    coupling_bias = np.zeros((2, 2))
    reference = np.asarray([1.0, 0.0, -1.0])
    for run_name, seed, role, logit_scale in runs:
        for pair_index in range(12):
            layer = pair_index // 4
            head = pair_index % 4
            left = np.asarray([[0.2, -0.1], [0.05, 0.3]])
            right = left + (pair_index + 1) * 1e-4
            replay = plga_secant_chain(
                left, right, weight, bias, powers, coupling, coupling_bias)
            perturbation = logit_scale * np.asarray([1.0, -0.4, -0.6])
            candidate = reference + perturbation
            path = tmp_path / f"{seed}-{pair_index}.npz"
            write_npz_atomic(
                path,
                stage=np.asarray("A"),
                checkpoint_sha256=np.asarray(f"{seed:064x}"[-64:]),
                run_name=np.asarray(run_name),
                seed=np.asarray(seed, dtype=np.int64),
                role=np.asarray(role),
                pair_id=np.asarray(f"plga-{role[0]}-{pair_index:02d}"),
                layer=np.asarray(layer, dtype=np.int64),
                head=np.asarray(head, dtype=np.int64),
                generator_left=left,
                generator_right=right,
                weight=weight,
                bias=bias,
                powers=powers,
                coupling=coupling,
                coupling_bias=coupling_bias,
                curvature_difference=replay["realized_difference"],
                secant_prediction=replay["predicted_difference"],
                reference_centered_logits=reference,
                candidate_centered_logits=candidate,
                reference_margin=np.asarray(1.0),
            )
            paths.append(path)
    report = analyze_plga(paths)
    assert report["all_checks_pass"]
    assert report["checks"]["complete_seed_owned_pair_registry"]
    assert report["checks"]["nonempty_heldout_sharp_margin_set"]
    assert report["results"]["qualified_validation_pair_count"] == 24
