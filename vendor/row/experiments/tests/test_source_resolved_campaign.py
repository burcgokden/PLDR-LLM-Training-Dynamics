"""Campaign-graph and reduced-operator regressions."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.source_resolved_block_response import (  # noqa: E402
    _actions_finite,
    _normalize,
    _orthogonalize,
    _stable_vector_norm,
    _state_dot,
    _state_norm,
)
from confirm.source_resolved_rank_tube import _rank_record  # noqa: E402
from confirm.source_resolved_specs import (  # noqa: E402
    FROZEN_POLICY,
    REGISTRY,
    RESOURCE_BUDGET,
)
from scripts.stage_source_resolved_confirmation import (  # noqa: E402
    launch_plan,
)


def _state(values):
    return tuple(
        {"x": torch.tensor(block, dtype=torch.float64)}
        for block in values
    )


def test_lifted_state_orthogonalization_is_two_pass_and_normalized():
    first = _state(([1.0, 2.0], [0.5, -1.0], [3.0, 0.25]))
    second = _state(([-2.0, 1.0], [4.0, 0.5], [0.25, -3.0]))
    _normalize(first)
    _coefficients, residual = _orthogonalize(second, [first])
    assert residual > 0.0
    _normalize(second)
    np.testing.assert_allclose(
        _state_norm(first), 1.0, rtol=0.0, atol=1.0e-14
    )
    assert abs(_state_dot(first, second)) < 1.0e-14


def test_lifted_observable_norm_does_not_overflow_in_float32():
    value = torch.full((64, 64), 1.0e19, dtype=torch.float32)
    assert not torch.isfinite(torch.linalg.vector_norm(value))
    observed = _stable_vector_norm(value)
    assert np.isfinite(observed)
    np.testing.assert_allclose(observed, 64.0e19, rtol=1.0e-7)


def test_lifted_action_finiteness_is_a_schema_native_boolean():
    observed = _actions_finite([{"gain": 1.0}], np.eye(2))
    assert observed is True
    assert isinstance(observed, bool)
    rejected = _actions_finite([{"gain": float("inf")}], np.eye(2))
    assert rejected is False
    assert isinstance(rejected, bool)


def test_layernorm_image_rank_is_3969_and_not_full_adamw_rank():
    rng = np.random.default_rng(45111)
    raw = rng.normal(size=(64, 64))
    centered = raw - raw.mean(axis=-1, keepdims=True)
    shape = centered / np.sqrt(
        1.0e-6 + np.mean(centered * centered, axis=-1, keepdims=True)
    )
    contexts = [{"id": f"context-{index}"} for index in range(24)]
    record = _rank_record(0, contexts, shape, np.ones(64))
    assert record["operator_scope"] == "implemented-layernorm-image-contrast"
    assert record["constraint_dimension_exact"]
    assert record["target_row_count"] == 3969
    assert record["exact_algebraic_rank"] == 3969
    assert record["rank_threshold_pass"]

    deficient_gate = np.ones(64)
    deficient_gate[:2] = 0.0
    deficient = _rank_record(0, contexts, shape, deficient_gate)
    assert deficient["exact_algebraic_rank"] == 63 * 62
    assert not deficient["rank_threshold_pass"]


def test_launch_plan_is_producer_closed_and_uses_complete_terminal_files(
    tmp_path,
):
    bundle = tmp_path / "bundle"
    protocol = bundle / "protocol"
    protocol.mkdir(parents=True)
    dataset = tmp_path / "tokens.npy"
    tokenizer = tmp_path / "tokenizer.model"
    registry = protocol / "registry.json"
    for path, payload in (
        (dataset, b"tokens"),
        (tokenizer, b"tokenizer"),
        (registry, b"registry"),
    ):
        path.write_bytes(payload)
    orders = {}
    for name in ("construction", "heldout8444", "heldout9444"):
        path = tmp_path / f"{name}.npy"
        path.write_bytes(name.encode())
        orders[name] = path

    plan = launch_plan(
        bundle, dataset, tokenizer, registry, orders
    )
    nodes = plan["nodes"]
    identifiers = [node["id"] for node in nodes]
    positions = {
        identifier: index for index, identifier in enumerate(identifiers)
    }
    assert len(nodes) == 133
    assert len(set(identifiers)) == len(identifiers)
    assert all(
        all(positions[dependency] < index for dependency in node["depends_on"])
        for index, node in enumerate(nodes)
    )

    initial = {
        str(dataset), str(tokenizer), str(registry),
        *(str(path) for path in orders.values()),
        str(protocol / "campaign_design.json"),
        str(protocol / "training_campaign_spec.json"),
        *(str(bundle / f"preliminary/checkpoints/ckpt_{step}.pt")
          for step in range(1000, 1005)),
    }
    available = set(initial)
    for node in nodes:
        assert set(node["required_inputs"]).issubset(available)
        available.update(node["expected_outputs"])

    by_id = {node["id"]: node for node in nodes}
    continuation_outputs = by_id["q3-native-continuation"][
        "expected_outputs"
    ]
    assert len(continuation_outputs) == 12
    assert continuation_outputs[-1].endswith("ckpt_final.pt")
    continuation_command = by_id["q3-native-continuation"]["command"]
    schedule_index = continuation_command.index("--schedule_total_steps")
    assert continuation_command[schedule_index + 1] == "1025"
    assert by_id["q3-rank-tube-block"]["depends_on"] == [
        "q3-lifted-block-response"
    ]
    assert all(
        "ckpt_9000.pt" not in output
        for node in nodes for output in node["expected_outputs"]
    )

    projected = sum(node["projected_output_bytes"] for node in nodes)
    staged_pilot_allowance = 5 * 300 * 1024**2
    assert (
        projected + staged_pilot_allowance
        < RESOURCE_BUDGET["persistent_output_cap_bytes"]
    )


def test_frozen_policy_separates_interface_and_full_state_conditions():
    assert REGISTRY["independent_contrast_dimension_per_map"] == 63 * 63
    assert FROZEN_POLICY["rank_relative_threshold"] == 1.0e-8
    assert FROZEN_POLICY["reduced_krylov_rank"] == 8
    assert FROZEN_POLICY["orthogonal_audit_directions"] == 8
    assert FROZEN_POLICY["intervention_window_updates"] == {
        "pre-onset": 1000,
        "onset": 4000,
        "late": 8000,
    }
