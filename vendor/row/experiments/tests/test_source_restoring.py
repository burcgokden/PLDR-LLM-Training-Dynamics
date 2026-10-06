"""Regression tests for the source-restoring collapse certificate."""

from __future__ import annotations

import json
import numpy as np
import pytest

from confirm.build_source_restoring_remainder import (
    validated_taylor_remainder,
)
from analysis.analyze_source_restoring_confirmation import (
    AGGREGATE_SCHEMA,
    EDGE_SCHEMA,
    aggregate_chronological_edges,
    source_prediction_summary,
)
from confirm.source_restoring import (
    _directed_sum,
    centered_row_geometry,
    cover_native_successor,
    pair_contrasts,
    partial_timecourse,
    source_multiplier_cocycle,
    source_restoring_certificate,
    symmetric_taylor_remainder,
    two_channel_spatial_bound,
)
from confirm.confirmation_artifacts import (
    digest_object,
    sha256_path,
    validate_measurement_registry,
)
from confirm.source_restoring_live import (
    parameter_block,
    validate_source_checkpoint,
)
from confirm.source_restoring_specs import (
    CAMPAIGN_ID,
    OPTIMIZER,
    REGISTRY_SCHEMA,
    TRAJECTORIES,
    training_campaign_spec,
)


def _linear_fixture():
    rng = np.random.default_rng(43001)
    source = rng.normal(size=(3, 6, 4))
    source -= np.mean(source, axis=1, keepdims=True)
    gate = np.zeros_like(source)
    gate[0] = -0.2 * source[0]
    gate[1] = 0.2 * source[1]
    shape = np.zeros_like(source)
    shape[2, :, 0] = -0.2 * source[2, :, 1]
    shape[2, :, 1] = 0.2 * source[2, :, 0]
    upstream = np.zeros_like(source)
    velocity = gate + shape
    endpoint = source + velocity
    _, velocity_contrast = pair_contrasts(velocity)
    curvature = 2.0 * np.einsum(
        "mpd,mpd->mp", velocity_contrast, velocity_contrast
    )
    remainder_lower, remainder_upper = symmetric_taylor_remainder(
        curvature, curvature.shape
    )
    _, endpoint_contrast = pair_contrasts(endpoint)
    endpoint_norm = np.linalg.norm(endpoint_contrast, axis=-1)
    certificate = source_restoring_certificate(
        source,
        gate,
        shape,
        upstream,
        remainder_lower=remainder_lower,
        remainder_upper=remainder_upper,
        endpoint_contrast_norm_upper=endpoint_norm,
        native_contrast_radius=0.0,
    )
    return source, endpoint, certificate


def test_centered_geometry_checks_both_finite_norm_bounds():
    source, _endpoint, _certificate = _linear_fixture()
    result = centered_row_geometry(source)
    assert result["valid"]
    assert result["maximum_abs_identity_residual"] < 1.0e-11
    assert np.all(result["lower_equivalence_slack"] >= -1.0e-11)
    assert np.all(result["upper_equivalence_slack"] >= -1.0e-11)


def test_source_certificate_resolves_all_three_scientific_outcomes():
    _source, endpoint, certificate = _linear_fixture()
    assert certificate["successor_values_used"] is False
    assert certificate["cross_map_pairs_formed"] is False
    assert certificate["scientific_outcome"] == (
        "confirmed",
        "not_confirmed",
        "unresolved",
    )
    assert np.max(np.abs(certificate["work_additivity_residual"])) < 1.0e-13
    coverage = cover_native_successor(certificate, endpoint)
    assert coverage["technical_valid"]
    assert coverage["coverage_fraction"] == 1.0
    assert coverage["observed_outcome"] == (
        "contracted",
        "expanded",
        "expanded",
    )


def test_source_margin_exposes_every_deciding_pair_term():
    _source, _endpoint, certificate = _linear_fixture()
    for field in (
        "source_gap",
        "signed_gate_work",
        "signed_shape_work",
        "signed_input_work",
        "remainder_lower",
        "remainder_upper",
        "native_arithmetic_charge",
        "source_restoring_margin",
    ):
        assert np.asarray(certificate[field]).shape == (3, 15)
    reconstructed = _directed_sum((
        certificate["source_gap_lower"],
        -certificate["signed_total_work_upper"],
        -certificate["remainder_upper"],
        -certificate["native_arithmetic_charge"],
    ), -np.inf)
    assert np.array_equal(
        reconstructed, certificate["source_restoring_margin"]
    )


def test_bad_successor_is_a_technical_coverage_failure():
    _source, endpoint, certificate = _linear_fixture()
    changed = endpoint.copy()
    changed[0, 0] += 100.0
    coverage = cover_native_successor(certificate, changed)
    assert not coverage["technical_valid"]
    assert coverage["coverage_fraction"] < 1.0


def test_malformed_remainder_is_rejected_before_scientific_decision():
    source, _endpoint, _certificate = _linear_fixture()
    zeros = np.zeros_like(source)
    with pytest.raises(ValueError, match="lower endpoint exceeds"):
        source_restoring_certificate(
            source,
            zeros,
            zeros,
            zeros,
            remainder_lower=1.0,
            remainder_upper=-1.0,
            endpoint_contrast_norm_upper=1.0,
            native_contrast_radius=0.0,
        )


def test_source_cocycle_allows_intermittent_expansion():
    result = source_multiplier_cocycle(
        np.asarray([4.0]),
        np.asarray([[1.1], [0.8], [0.9]]),
    )
    assert result["prefix_product"][:, 0] == pytest.approx(
        [1.0, 1.1, 0.88, 0.792]
    )
    assert result["diameter_squared_bound"][-1, 0] == pytest.approx(3.168)
    assert result["cumulative_log_product"][1, 0] > 0.0
    assert result["cumulative_log_product"][-1, 0] < 0.0


def test_spatial_bound_keeps_gain_and_shape_channels_separate():
    result = two_channel_spatial_bound(
        np.asarray([0.2, 0.2]),
        np.asarray([0.01, 4.0]),
        np.asarray([2.0, 2.0]),
        feature_dimension=64,
    )
    assert result["occupied_chord_bound"][0] == pytest.approx(0.004)
    assert result["universal_gain_bound"][0] == pytest.approx(3.2)
    assert result["combined_bound"] == pytest.approx([0.004, 1.6])


def test_partial_timecourse_preserves_available_points():
    report = partial_timecourse(
        [100, 200],
        [[1.0], [0.9]],
        [100, 200, 300],
        reason="resource_cap",
    )
    assert not report["complete"]
    assert report["reason"] == "resource_cap"
    with pytest.raises(ValueError, match="needs a reason"):
        partial_timecourse([100], [[1.0]], [100, 200])


def test_complete_timecourse_rejects_a_stop_reason():
    report = partial_timecourse(
        [100, 200], [[1.0], [0.8]], [100, 200]
    )
    assert report["complete"]
    with pytest.raises(ValueError, match="cannot carry"):
        partial_timecourse(
            [100, 200],
            [[1.0], [0.8]],
            [100, 200],
            reason="none",
        )


def _edge(
    step, source, multiplier, upper, successor, outcome="unresolved"
):
    source_path = f"/checkpoints/step-{step}.pt"
    successor_path = f"/checkpoints/step-{step + 1}.pt"
    digest = lambda value: f"{value:064x}"
    return {
        "schema_version": EDGE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "step": step,
        "source_artifact_path": f"/artifacts/source-{step}.npz",
        "source_artifact_sha256": digest(100_000 + step),
        "registry_sha256": digest(200_000),
        "data_order_sha256": digest(300_000),
        "certificate_created_at_ns": step * 10 + 1,
        "prediction_created_at_ns": step * 10 + 2,
        "successor_opened_at_ns": step * 10 + 3,
        "source_checkpoint_path": source_path,
        "source_checkpoint_sha256": digest(step),
        "native_successor_checkpoint_path": successor_path,
        "native_successor_checkpoint_sha256": digest(step + 1),
        "prediction_sha256": digest(400_000 + step),
        "certificate_sha256": digest(500_000 + step),
        "technical_valid": True,
        "scientific_outcome": [outcome],
        "source_diameter_squared": [source],
        "source_diameter_squared_lower": [source],
        "source_diameter_squared_upper": [source],
        "source_upper_multiplier": [multiplier],
        "source_successor_diameter_squared_upper": [upper],
        "native_successor_diameter_squared": [successor],
        "native_successor_diameter_squared_lower": [successor],
        "native_successor_diameter_squared_upper": [successor],
        "certificate_coverage_fraction": 1.0,
        "native_peak_memory_reserved_bytes": 0,
    }


def test_zero_diameter_retains_absolute_bound_without_ratio():
    source = np.zeros((1, 3, 2), dtype=np.float64)
    gate = np.asarray([[[0.0, 0.0], [0.1, 0.0], [-0.1, 0.0]]])
    zeros = np.zeros_like(source)
    _, gate_contrast = pair_contrasts(gate)
    curvature = 2.0 * np.einsum(
        "mpd,mpd->mp", gate_contrast, gate_contrast
    )
    lower, upper = symmetric_taylor_remainder(curvature, curvature.shape)
    _, endpoint_contrast = pair_contrasts(source + gate)
    certificate = source_restoring_certificate(
        source,
        gate,
        zeros,
        zeros,
        remainder_lower=lower,
        remainder_upper=upper,
        endpoint_contrast_norm_upper=np.linalg.norm(
            endpoint_contrast, axis=-1
        ),
        native_contrast_radius=0.0,
    )
    prediction = source_prediction_summary(certificate)
    assert certificate["source_diameter_squared"][0] == 0.0
    assert np.isnan(certificate["source_upper_multiplier"][0])
    assert prediction["source_upper_multiplier"] == [None]
    assert prediction["source_successor_diameter_squared_upper"][0] > 0.0


def test_chronological_aggregate_resets_at_zero_diameter():
    first = _edge(10, 0.0, None, 1.0, 0.25)
    second = _edge(11, 0.25, 0.5, 0.125, 0.1)
    result = aggregate_chronological_edges([second, first])
    assert result["schema_version"] == AGGREGATE_SCHEMA
    assert result["all_prefix_bounds_cover_native"]
    assert not result["all_source_products_defined"]
    assert result["final_cumulative_log_product"] == [None]
    assert result["scientific_outcome"] == "unresolved"


def test_chronological_aggregate_checks_numerical_checkpoint_linkage():
    first = _edge(20, 1.0, 0.5, 0.5, 0.5)
    second = _edge(21, 0.6, 0.5, 0.3, 0.3)
    with pytest.raises(ValueError, match="row-map diameter"):
        aggregate_chronological_edges([first, second])


def test_scientific_outcome_does_not_prune_a_later_edge():
    first = _edge(30, 1.0, 1.1, 1.1, 1.1, "not_confirmed")
    second = _edge(31, 1.1, 0.5, 0.55, 0.55, "confirmed")
    result = aggregate_chronological_edges([first, second])
    assert result["edge_count"] == 2
    assert result["scientific_outcomes_halted_descendants"] is False
    assert result["scientific_outcome"] == "confirmed"


def test_parameter_partition_matches_the_three_source_work_blocks():
    final_norm = "decoder.dec_layers.1.mha1.reslayerAs.7.layernormA."
    assert parameter_block(final_norm + "weight", 1) == "gate"
    assert parameter_block(final_norm + "bias", 1) == "gate"
    assert (
        parameter_block("decoder.dec_layers.1.mha1.layernorm1.weight", 1)
        == "shape"
    )
    assert (
        parameter_block(
            "decoder.dec_layers.1.mha1.reslayerAs.6.layers.0.weight", 1
        )
        == "shape"
    )
    assert (
        parameter_block("decoder.dec_layers.0.mha1.layernorm1.weight", 1)
        == "input"
    )
    assert parameter_block("decoder.embed.weight", 1) == "input"


def test_directed_certificate_contains_every_nominal_reduction():
    _source, endpoint, certificate = _linear_fixture()
    assert np.all(
        certificate["source_energy_lower"]
        <= certificate["source_energy"]
    )
    assert np.all(
        certificate["source_energy"]
        <= certificate["source_energy_upper"]
    )
    assert np.all(
        certificate["source_diameter_squared_lower"]
        <= certificate["source_diameter_squared"]
    )
    assert np.all(
        certificate["source_diameter_squared"]
        <= certificate["source_diameter_squared_upper"]
    )
    positive = certificate["source_diameter_squared_lower"] > 0.0
    assert np.all(
        certificate["source_upper_multiplier"][positive]
        * certificate["source_diameter_squared_lower"][positive]
        >= certificate["predicted_diameter_squared_upper"][positive]
    )
    coverage = cover_native_successor(certificate, endpoint)
    assert np.all(coverage["native_energy_lower"] <= coverage["native_energy"])
    assert np.all(coverage["native_energy"] <= coverage["native_energy_upper"])


def test_validated_curvature_reduces_to_directed_taylor_endpoints():
    result = validated_taylor_remainder(
        [[0.0, 2.0]], [[1.0, 3.0]], [[0.0, 0.1]]
    )
    assert result["remainder_lower"][0, 0] == 0.0
    assert result["remainder_upper"][0, 0] == 0.0
    assert result["remainder_lower"][0, 1] < -1.0
    assert result["remainder_upper"][0, 1] > 1.0
    with pytest.raises(ValueError, match="malformed"):
        validated_taylor_remainder(
            [[-1.0]], [[1.0]], [[0.0]]
        )


def test_checkpoint_campaign_binding_is_content_addressed(tmp_path):
    tokens = tmp_path / "tokens.npy"
    tokens.write_bytes(np.arange(32, dtype=np.uint16).tobytes())
    contexts = [
        {"id": f"c{index:02d}", "chunk_index": index + 1}
        for index in range(8)
    ]
    validation = [
        {"id": f"v{index:02d}", "chunk_index": index + 1}
        for index in range(8, 24)
    ]
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "context_length": 256,
        "construction": contexts,
        "validation": validation,
        "dataset_sha256": sha256_path(tokens),
        "tokenizer_sha256": "1" * 64,
        "total_chunks": 25,
        "trainer_probe_reserve_chunks": 24,
        "trainer_training_limit": 1,
        "heads": [0, 1, 2, 3],
        "layers": [0, 1, 2],
        "rows_per_map": 64,
        "pairs_per_map": 2016,
        "pair_rule": "within-context-layer-head-all-unordered-pairs-v1",
        "cross_map_pairs_forbidden": True,
    }
    registry["registry_sha256"] = digest_object(registry)
    assert validate_measurement_registry(registry, require_nonempty=True)
    campaign = training_campaign_spec(registry["registry_sha256"])
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(campaign, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    trajectory = TRAJECTORIES[0]
    checkpoint = {
        "measurement_registry": registry,
        "data_state": {
            "dataset_sha256": registry["dataset_sha256"],
            "tokenizer_sha256": registry["tokenizer_sha256"],
            "campaign_binding": {
                "campaign_id": CAMPAIGN_ID,
                "campaign_spec_sha256": campaign["spec_sha256"],
                "campaign_spec_file_sha256": sha256_path(campaign_path),
                "campaign_spec_path": str(campaign_path),
            },
        },
        "config": {
            "layers": 3,
            "heads": 4,
            "dk": 64,
            "adff": 170,
            "batch": OPTIMIZER["batch_size"],
            "ctx": OPTIMIZER["context_length"],
            "lr": OPTIMIZER["learning_rate"],
            "warmup": OPTIMIZER["warmup_updates"],
            "optimizer": "adamw",
            "wd": OPTIMIZER["weight_decay"],
            "clip": OPTIMIZER["gradient_clip_value"],
            "const_lr": True,
            "probe_region": "global",
            "accum": 1,
            "seed": trajectory["seed"],
            "name": trajectory["name"],
        },
    }
    assert validate_source_checkpoint(
        checkpoint, registry, tokens
    )["role"] == "construction"
    checkpoint["data_state"]["campaign_binding"]["campaign_id"] = "changed"
    with pytest.raises(ValueError, match="identity or digest"):
        validate_source_checkpoint(checkpoint, registry, tokens)
