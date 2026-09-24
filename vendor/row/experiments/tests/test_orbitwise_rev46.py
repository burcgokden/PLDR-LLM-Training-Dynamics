"""Regression tests for the revision-46 orbitwise collapse kernels."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from analysis.analyze_orbitwise_confirmation import (  # noqa: E402
    _defined_summary,
    _source_gain,
)
from confirm.orbitwise_specs import (  # noqa: E402
    DENSE_CENTERS,
    PROBE_RESERVE_CHUNKS,
    RESOURCE_BUDGET,
    TRAJECTORIES,
    campaign_design,
    construction_checkpoint_updates,
    dense_snapshot_updates,
    full_snapshot_updates,
    source_edges,
    validate_design,
)
from confirm.confirmation_artifacts import digest_object  # noqa: E402
from confirm.source_resolved_energy import (  # noqa: E402
    effective_gate_shape,
    endpoint_gate_shape_decomposition,
    reopening_budget_envelope,
)
from confirm.strict_schema import validate  # noqa: E402
from scripts.gen_orbitwise_protocols import (  # noqa: E402
    generated_files,
    schemas,
)
from scripts.stage_orbitwise_confirmation import (  # noqa: E402
    _write_order,
    launch_plan,
)


def test_effective_gate_is_a_shape_weighted_convex_average():
    result = effective_gate_shape(
        [[1.0, 3.0], [2.0, 2.0]],
        [1.0, 2.0],
    )
    np.testing.assert_allclose(result["shape_energy"], [4.0, 4.0])
    np.testing.assert_allclose(result["physical_energy"], [13.0, 10.0])
    np.testing.assert_allclose(
        result["effective_gate_squared"], [3.25, 2.5]
    )
    assert np.all(
        result["gate_lower_energy"] <= result["physical_energy"]
    )
    assert np.all(
        result["physical_energy"] <= result["gate_upper_energy"]
    )
    np.testing.assert_allclose(
        np.sum(result["shape_occupancy"], axis=-1), 1.0
    )


def test_three_channel_gain_keeps_an_expanding_alignment_term():
    result = endpoint_gate_shape_decomposition(
        [1.0, 1.0],
        [0.05, 0.45],
        [1.0, 1.0],
        [0.2, 0.8],
    )
    assert result["shape_gain"] == pytest.approx(0.25)
    assert result["gate_only_gain"] == pytest.approx(0.34)
    assert result["alignment_gain"] > 1.0
    assert result["physical_gain"] < 1.0
    assert result["maximum_abs_gain_residual"] < 1.0e-15
    assert result["maximum_abs_log_gain_residual"] < 1.0e-15


def test_endpoint_decomposition_fails_closed_at_zero_denominators():
    with pytest.raises(ValueError, match="positive shape and gate"):
        endpoint_gate_shape_decomposition(
            [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]
        )


def test_reopening_budget_handles_zero_energy_and_large_relative_reopening():
    energy = np.asarray([4.0, 1.0, 0.0, 2.0, 1.0, 1.5])
    result = reopening_budget_envelope(energy)
    np.testing.assert_allclose(result["reopening"], [0.0, 0.0, 2.0, 0.0, 0.5])
    np.testing.assert_allclose(
        result["reopening_prefix"], [0.0, 0.0, 0.0, 2.0, 2.0, 2.5]
    )
    np.testing.assert_allclose(
        result["reopening_tail"], [2.5, 2.5, 2.5, 0.5, 0.5, 0.0]
    )
    assert result["minimum_envelope_slack"] >= 0.0
    assert np.all(result["energy"] <= result["energy_envelope"])


def test_reopening_budget_rejects_negative_energy():
    with pytest.raises(ValueError, match="nonnegative"):
        reopening_budget_envelope([1.0, -0.1])


def test_zero_source_gain_is_explicit_and_absolute_summary_survives():
    gain, defined = _source_gain(
        np.asarray([0.0, 3.0, 0.0]),
        np.asarray([0.0, 2.0, 4.0]),
        "test",
    )
    np.testing.assert_allclose(gain, [0.0, 1.5, 0.0])
    np.testing.assert_array_equal(defined, [False, True, True])
    assert _defined_summary(gain, defined)["maximum"] == pytest.approx(1.5)
    assert _defined_summary(gain, np.zeros(3, dtype=bool)) == {
        "minimum": 0.0,
        "median": 0.0,
        "p90": 0.0,
        "maximum": 0.0,
    }


def test_orbitwise_design_has_exact_full_dense_and_source_populations():
    design = campaign_design()
    validate_design(design)
    assert len(full_snapshot_updates()) == 564
    assert full_snapshot_updates()[-2:] == [17_984, 18_000]
    assert len(dense_snapshot_updates()) == 4 * 65
    assert len(source_edges()) == 8
    assert set(DENSE_CENTERS).issubset(construction_checkpoint_updates())
    assert len(construction_checkpoint_updates()) == 13
    assert [int(item["seed"]) for item in TRAJECTORIES] == [
        10444, 11444, 12444
    ]


def test_orbitwise_schemas_are_strict_and_separate_scientific_outcomes():
    values = schemas()
    source_link = values["source_link.schema.json"]
    assert "scientific_checks" in source_link["properties"]
    assert "source_reopening_bound_closed" not in (
        source_link["properties"]["checks"]["properties"]
    )
    for schema in values.values():
        pending = [schema]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                if node.get("type") == "object":
                    assert node.get("additionalProperties") is False
                    assert set(node["required"]) == set(node["properties"])
                pending.extend(node.values())
            elif isinstance(node, list):
                pending.extend(node)

    exact_pair = {"type": "array", "items": {"type": "integer"},
                  "minItems": 2, "maxItems": 2}
    validate([1, 2], exact_pair)
    with pytest.raises(ValueError, match="more than 2"):
        validate([1, 2, 3], exact_pair)


def test_training_campaign_signature_replays_in_the_native_trainer():
    payload = json.loads(
        generated_files()[Path("training_campaign_spec.json")]
    )
    recorded = payload.pop("spec_sha256")
    assert recorded == digest_object(payload)


def test_staged_data_order_excludes_the_reserved_probe_region(tmp_path):
    path = tmp_path / "order.npy"
    total_chunks = 600_000
    _write_order(path, 10444, total_chunks)
    order = np.load(path, allow_pickle=False)
    assert len(order) == total_chunks - PROBE_RESERVE_CHUNKS
    assert int(order.min()) == 0
    assert int(order.max()) == total_chunks - PROBE_RESERVE_CHUNKS - 1
    assert len(order) >= 18_000 * 32


def test_orbitwise_launch_graph_is_ordered_and_producer_first(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "protocol").mkdir(parents=True)
    tokens = tmp_path / "tokens.npy"
    tokenizer = tmp_path / "tokenizer.model"
    registry = bundle / "protocol" / "registry.json"
    for path, payload in (
        (tokens, b"tokens"),
        (tokenizer, b"tokenizer"),
        (registry, b"registry"),
    ):
        path.write_bytes(payload)
    orders = {}
    for trajectory in TRAJECTORIES:
        key = str(trajectory["order_key"])
        path = tmp_path / f"{key}.npy"
        path.write_bytes(key.encode("ascii"))
        orders[key] = path

    plan = launch_plan(bundle, tokens, tokenizer, registry, orders)
    nodes = plan["nodes"]
    positions = {
        node["id"]: index for index, node in enumerate(nodes)
    }
    assert len(nodes) == 59
    assert len(positions) == 59
    assert all(
        positions[dependency] < positions[node["id"]]
        for node in nodes
        for dependency in node["depends_on"]
    )
    assert all(
        positions[f"c2-predict-{left}-{right}"]
        < positions[f"c2-replay-{left}-{right}"]
        for left, right in source_edges()
    )
    assert all(
        positions[f"c3-seal-gate-{step}"]
        < positions[f"c3-control-{step}"]
        < positions[f"c3-gate-result-{step}"]
        for step in (4096, 8192, 16384)
    )
    assert (
        sum(node["projected_output_bytes"] for node in nodes)
        <= RESOURCE_BUDGET["persistent_output_cap_bytes"]
    )
    assert plan["scientific_outcomes_control_execution"] is False
