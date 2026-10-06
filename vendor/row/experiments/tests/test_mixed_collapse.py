"""Regression tests for the mixed-collapse campaign."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.analyze_mixed_collapse_confirmation import (  # noqa: E402
    dense_snapshot_updates,
    duhamel_diagnostics,
    exact_dense_diagnostics,
    factorization_residual,
    observed_block_diagnostics,
)
from analysis.summarize_mixed_manuscript_results import (  # noqa: E402
    _factorization_capture_point,
    _factorial_rows,
    _shape_gain_summary,
)
from confirm.confirmation_artifacts import digest_object  # noqa: E402
from confirm.mixed_collapse_specs import (  # noqa: E402
    BLOCK_ANCHORS,
    DENSE_CENTERS,
    INTERVENTION_MODES,
    INTERVENTION_UPDATES,
    REGISTRY,
    SOURCE_RADIUS_UPDATES,
    TERMINAL_UPDATE,
    TRAJECTORIES,
    campaign_design,
    full_snapshot_updates,
    validate_design,
)
from confirm.strict_schema import validate  # noqa: E402
from scripts.check_mixed_collapse_results import (  # noqa: E402
    _verify_factorization_outcomes,
    _verify_full_arm_reproduction,
    _verify_prefix_reproduction,
    _verify_radius_outcomes,
)
from scripts.gen_mixed_collapse_protocols import (  # noqa: E402
    generated_files,
    schemas,
)
from scripts.stage_mixed_collapse_confirmation import (  # noqa: E402
    _verify_source_snapshot,
    build_unique_extension_order,
    launch_plan,
)
from train_run import (  # noqa: E402
    final_gate_registry_required,
    final_gate_step_target,
    validate_data_order,
)


def test_design_has_fixed_horizon_all_maps_and_no_reuse() -> None:
    design = campaign_design()
    validate_design(design)
    assert len(full_snapshot_updates()) == TERMINAL_UPDATE // 32 + 1
    assert full_snapshot_updates()[:2] == [0, 32]
    assert full_snapshot_updates()[-2:] == [65_504, TERMINAL_UPDATE]
    assert len(dense_snapshot_updates()) == 5 * 65 + 33
    assert tuple(design["block_anchors"]) == BLOCK_ANCHORS
    assert tuple(design["dense_centers"]) == DENSE_CENTERS
    assert design["registry"]["map_count"] == 288
    policy = design["data_order_policy"]
    assert policy["globally_unique_chunk_indices"] is True
    assert policy["predecessor_training_chunks"] == 854_255
    assert policy["complete_training_chunks"] == TERMINAL_UPDATE * 32


def test_unique_extension_preserves_prefix_and_uses_only_new_indices() -> None:
    base = np.asarray([3, 0, 7, 2, 6, 1, 5, 4], dtype=np.int64)
    first = build_unique_extension_order(base, 10444, 20, 20)
    second = build_unique_extension_order(base, 11444, 20, 20)
    np.testing.assert_array_equal(first[:len(base)], base)
    np.testing.assert_array_equal(second[:len(base)], base)
    assert set(first[len(base):]) == set(range(len(base), 20))
    assert len(np.unique(first)) == len(first)
    assert not np.array_equal(first[len(base):], second[len(base):])
    np.testing.assert_array_equal(validate_data_order(first, 20 + 5_120), first)
    repeated = first.copy()
    repeated[-1] = repeated[0]
    with pytest.raises(ValueError, match="repeated"):
        validate_data_order(repeated, 20 + 5_120)


def test_launch_graph_contains_every_signed_arm_and_is_producer_first() -> None:
    plan = launch_plan()
    nodes = plan["nodes"]
    positions = {node["id"]: index for index, node in enumerate(nodes)}
    expected = (
        len(TRAJECTORIES)
        + len(TRAJECTORIES) * len(INTERVENTION_UPDATES) * len(INTERVENTION_MODES)
        + len(TRAJECTORIES) * len(SOURCE_RADIUS_UPDATES)
        + 1
    )
    assert len(nodes) == expected == 49
    assert len(positions) == expected
    assert all(
        positions[dependency] < positions[node["id"]]
        for node in nodes
        for dependency in node["depends_on"]
    )
    assert plan["scientific_outcomes_control_execution"] is False
    assert all(
        "--data_order_policy" not in node["command"]
        for node in nodes
    )
    assert sum(node["stage"] == "source-radius" for node in nodes) == 9


def test_coordinate_factorization_is_per_map_and_can_fail() -> None:
    generator = np.random.default_rng(47)
    shape = generator.uniform(0.05, 2.0, size=(288, 64))
    gates = generator.normal(size=(3, 64))
    map_gates = np.broadcast_to(
        gates[None, :, None, :], (24, 3, 4, 64)
    ).reshape(288, 64)
    energy = np.sum(shape * map_gates * map_gates, axis=1)
    point = {
        "energy": energy,
        "normalized_shape_coordinate_energy": shape,
        "layer_final_gate": gates,
    }
    residual = factorization_residual(point)
    assert residual.shape == (REGISTRY["map_count"],)
    assert float(np.max(residual)) < 5.0e-16
    absolute, supplemental_relative, captured_energy = (
        _factorization_capture_point(point)
    )
    assert float(np.max(absolute)) < 5.0e-13
    np.testing.assert_allclose(supplemental_relative, residual, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(captured_energy, energy)
    point["energy"] = energy.copy()
    point["energy"][17] *= 1.25
    residual = factorization_residual(point)
    assert residual[17] == pytest.approx(0.2)
    assert np.count_nonzero(residual > 1.0e-5) == 1
    absolute, supplemental_relative, captured_energy = (
        _factorization_capture_point(point)
    )
    assert supplemental_relative[17] == pytest.approx(0.2)
    assert absolute[17] == pytest.approx(0.25 * energy[17])
    assert captured_energy[17] == pytest.approx(1.25 * energy[17])


def test_fixed_blocks_report_excursion_and_positive_variation_separately() -> None:
    steps = np.arange(5, dtype=np.int64)
    energy = np.asarray([
        [1.0, 2.0],
        [3.0, 2.0],
        [2.0, 1.0],
        [4.0, 1.5],
        [1.0, 1.25],
    ])
    blocks = observed_block_diagnostics(steps, energy, anchors=(0, 2, 4))
    assert len(blocks) == 2
    assert blocks[0]["observed_excursion"]["maximum"] == pytest.approx(2.0)
    assert blocks[0]["observed_positive_variation"]["maximum"] == pytest.approx(2.0)
    assert blocks[1]["observed_excursion"]["maximum"] == pytest.approx(2.0)
    assert blocks[1]["interpretation"] == "cadence-lower-bound"


def test_dense_windows_include_both_sides_and_clip_at_terminal() -> None:
    ordinary = {
        step: np.asarray([1.0 + step, 65.0 - step])
        for step in range(65)
    }
    result = exact_dense_diagnostics(ordinary, centers=(32,))[0]
    assert (result["left"], result["center"], result["right"]) == (0, 32, 64)
    assert result["edge_count"] == 64
    assert result["exact_excursion"]["maximum"] == pytest.approx(64.0)

    terminal = {
        step: np.asarray([float(step - 65_503), 1.0])
        for step in range(65_504, 65_537)
    }
    clipped = exact_dense_diagnostics(terminal, centers=(65_536,))[0]
    assert clipped["left"] == 65_504
    assert clipped["right"] == TERMINAL_UPDATE
    assert clipped["edge_count"] == 32


def test_shape_gain_summary_retains_zero_sources_and_expansions() -> None:
    summary = _shape_gain_summary(
        [2.0, 0.0, 4.0],
        [1.0, 3.0, 8.0],
    )
    assert summary["map_count"] == 3
    assert summary["positive_source_count"] == 2
    assert summary["zero_source_count"] == 1
    assert summary["zero_source_reopened_count"] == 1
    assert summary["maps_with_shape_contraction"] == 1
    assert summary["maps_with_shape_expansion"] == 2
    assert summary["positive_source_gain"]["median"] == pytest.approx(1.25)
    assert summary["positive_source_gain"]["maximum"] == pytest.approx(2.0)


def test_adamw_duhamel_reconstructs_signed_and_absolute_forcing() -> None:
    generator = np.random.default_rng(11)
    gates = [generator.normal(size=(3, 64))]
    rates = np.asarray([np.nan, 0.1, 0.2, 0.05])
    innovations = [
        generator.normal(scale=0.01, size=(3, 64)) for _ in range(3)
    ]
    for rate, innovation in zip(rates[1:], innovations, strict=True):
        gates.append((1.0 - rate * 0.1) * gates[-1] - innovation)
    result = duhamel_diagnostics(
        np.stack(gates), rates, anchors=(1, 2, 3), terminal_step=3
    )
    assert result["maximum_reconstruction_residual"] < 1.0e-14
    assert result["maximum_absolute_bound_violation"] <= 1.0e-14
    assert result["previous_anchor"] == 2
    assert set(result["anchors"]) == {"0", "1", "2", "3"}


def test_mixed_observer_and_factorial_always_select_final_gates() -> None:
    assert final_gate_registry_required(1, False, "full")
    assert final_gate_registry_required(0, True, "full")
    assert final_gate_registry_required(0, False, "frozen")
    assert final_gate_registry_required(0, False, "decay_only")
    assert final_gate_registry_required(0, False, "adaptive_only")
    assert not final_gate_registry_required(0, False, "full")
    with pytest.raises(ValueError, match="unknown"):
        final_gate_registry_required(0, False, "hybrid")


def test_four_gate_modes_are_exact_adamw_component_identities() -> None:
    before = torch.tensor([2.0, -3.0], dtype=torch.float64)
    rate = 0.2
    decay = 0.1
    multiplier = 1.0 - rate * decay
    adaptive = torch.tensor([0.3, -0.1], dtype=torch.float64)
    full = multiplier * before - adaptive
    torch.testing.assert_close(
        final_gate_step_target(before, full, rate, decay, "full"), full
    )
    torch.testing.assert_close(
        final_gate_step_target(before, full, rate, decay, "frozen"), before
    )
    torch.testing.assert_close(
        final_gate_step_target(before, full, rate, decay, "decay_only"),
        multiplier * before,
    )
    torch.testing.assert_close(
        final_gate_step_target(before, full, rate, decay, "adaptive_only"),
        before - adaptive,
    )
    with pytest.raises(ValueError, match="unknown"):
        final_gate_step_target(before, full, rate, decay, "hybrid")


def test_generated_protocols_are_signed_strict_and_replayable() -> None:
    payload = json.loads(
        generated_files()[Path("training_campaign_spec.json")]
    )
    recorded = payload.pop("spec_sha256")
    assert recorded == digest_object(payload)
    for schema in schemas().values():
        pending = [schema]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if value.get("type") == "object":
                    assert value.get("additionalProperties") is False
                    assert set(value["required"]) == set(value["properties"])
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
    validate(
        [1, 2],
        {"type": "array", "items": {"type": "integer"},
         "minItems": 2, "maxItems": 2},
    )
    with pytest.raises(ValueError, match="more than 2"):
        validate(
            [1, 2, 3],
            {"type": "array", "items": {"type": "integer"},
             "minItems": 2, "maxItems": 2},
        )




def test_post_analysis_factorial_summary_uses_signed_units() -> None:
    effects = []
    modes = ("full", "frozen", "decay_only", "adaptive_only")
    for unit in range(9):
        for mode in modes:
            if mode == "full":
                ratio = 1.0
            elif mode == "frozen":
                ratio = 0.8 + 0.05 * unit
            elif mode == "decay_only":
                ratio = 1.1
            else:
                ratio = 0.9
            effects.append({
                "source_step": 4096,
                "mode": mode,
                "energy_ratio_to_control": ratio,
            })
    interactions = [
        {"source_step": 4096, "interaction_over_control": 0.02 * (unit - 4)}
        for unit in range(9)
    ]
    rows = _factorial_rows({
        "gate_factorial": {
            "effects": effects,
            "interactions": interactions,
        }
    })
    assert len(rows) == 1
    row = rows[0]
    assert row["modes"]["frozen"]["median"] == pytest.approx(1.0)
    assert row["modes"]["frozen"]["above_control"] == 4
    assert row["modes"]["decay_only"]["above_control"] == 9
    assert row["modes"]["adaptive_only"]["above_control"] == 0
    assert row["interaction_median"] == pytest.approx(0.0)


def test_result_gate_records_adverse_factorization_without_failure() -> None:
    adverse = {
        "construction_frozen_tolerance": 1.0e-5,
        "maximum_per_map_relative_residual": 4.0e-3,
        "timepoints_with_any_map_above_tolerance": 2,
        "map_timepoints_above_tolerance": 3,
    }
    _verify_factorization_outcomes(
        adverse, expected_timepoints=10, map_count=288
    )
    inconsistent = dict(
        adverse, maximum_per_map_relative_residual=1.0e-6
    )
    with pytest.raises(ValueError, match="disagrees"):
        _verify_factorization_outcomes(
            inconsistent, expected_timepoints=10, map_count=288
        )


def test_result_gate_records_adverse_radius_outcomes_without_failure() -> None:
    adverse = {
        "records": [
            {"radius_contract_closed": False, "energy_envelope_closed": True},
            {"radius_contract_closed": True, "energy_envelope_closed": False},
        ],
        "all_radius_contracts_closed": False,
        "all_energy_envelopes_closed": False,
    }
    _verify_radius_outcomes(adverse)
    inconsistent = dict(adverse, all_radius_contracts_closed=True)
    with pytest.raises(ValueError, match="disagrees"):
        _verify_radius_outcomes(inconsistent)


def test_result_gate_requires_full_arms_to_reproduce_native_paths(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    for trajectory in TRAJECTORIES:
        seed = int(trajectory["seed"])
        native_path = bundle / "runs" / str(trajectory["name"]) / "log.jsonl"
        native_path.parent.mkdir(parents=True)
        native_rows = []
        for source_step in INTERVENTION_UPDATES:
            endpoint = source_step + 64
            point = {"step": endpoint, "energy": [float(seed), float(endpoint)]}
            native_rows.append(json.dumps({
                "step": endpoint,
                "mixed_collapse_full_timepoint": point,
            }))
            control_path = (
                bundle / "runs" / f"gate-{seed}-{source_step}-full" / "log.jsonl"
            )
            control_path.parent.mkdir(parents=True)
            control_path.write_text(json.dumps({
                "step": endpoint,
                "mixed_collapse_full_timepoint": point,
            }) + "\n", encoding="utf-8")
        native_path.write_text("\n".join(native_rows) + "\n", encoding="utf-8")

    _verify_full_arm_reproduction(bundle)
    first = TRAJECTORIES[0]
    bad_path = (
        bundle / "runs"
        / f"gate-{first['seed']}-{INTERVENTION_UPDATES[0]}-full"
        / "log.jsonl"
    )
    endpoint = INTERVENTION_UPDATES[0] + 64
    bad_path.write_text(json.dumps({
        "step": endpoint,
        "mixed_collapse_full_timepoint": {
            "step": endpoint,
            "energy": [-1.0, float(endpoint)],
        },
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly reproduce"):
        _verify_full_arm_reproduction(bundle)


def test_result_gate_requires_exact_prefix_reproduction(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    reports = bundle / "reports"
    reports.mkdir(parents=True)
    for trajectory in TRAJECTORIES:
        for step in (16_384, 18_000):
            path = (
                reports
                / f"prefix-reproduction-{trajectory['name']}-{step}.json"
            )
            path.write_text(json.dumps({
                "exact_equal": True,
                "left": {"sha256": "1" * 64},
                "left_step": step,
                "mismatch_count": 0,
                "mismatches": [],
                "right": {"sha256": "2" * 64},
                "right_step": step,
                "schema_version": "pldr-checkpoint-exact-comparison-v1",
                "scopes": ["model", "opt", "scheduler", "rng_states"],
            }) + "\n", encoding="utf-8")

    _verify_prefix_reproduction(bundle)
    bad_path = (
        reports
        / f"prefix-reproduction-{TRAJECTORIES[0]['name']}-16384.json"
    )
    bad = json.loads(bad_path.read_text(encoding="utf-8"))
    bad["exact_equal"] = False
    bad["mismatch_count"] = 1
    bad["mismatches"] = ["model"]
    bad_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not exact"):
        _verify_prefix_reproduction(bundle)


def test_staged_source_snapshot_rejects_byte_drift(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    source = bundle / "source" / "experiments" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"value = 47\n")
    protocol = bundle / "protocol"
    protocol.mkdir()
    binding = {
        "schema_version": "pldr-source-snapshot-binding-v1",
        "source_files": {
            "experiments/fixture.py": hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
        },
    }
    (protocol / "source_binding.json").write_text(
        json.dumps(binding), encoding="utf-8"
    )
    _verify_source_snapshot(bundle)
    source.write_bytes(b"value = 48\n")
    with pytest.raises(ValueError, match="digest drifted"):
        _verify_source_snapshot(bundle)
