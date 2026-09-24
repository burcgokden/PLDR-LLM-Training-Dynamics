"""Fixture-level tests for source-resolved live producers."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.source_resolved_live import qualify  # noqa: E402
from confirm.source_resolved_intervention import _lineage_digest  # noqa: E402
from confirm.source_resolved_observer import _json_list  # noqa: E402
from confirm.source_resolved_response import _direction_stream  # noqa: E402


PROTOCOL = ROOT / "experiments" / "protocols" / "source_resolved_confirmation"


def test_cpu_qualification_is_explicitly_fixture_only_and_passes():
    result = qualify("cpu", protocol_directory=PROTOCOL)
    assert result["mode"] == "fixture"
    assert result["checkpoint_binding_sha256"] is None
    assert not result["checks"]["checkpoint_content_opened"]
    assert result["technical_valid"]
    assert result["checks"]["schema_rejects_unknown_fields"]
    assert result["maximum_identity_residuals"]["energy_ledger"] < 1.0e-12


def test_full_state_energy_directions_are_constructed_lazily():
    random_directions = [
        ("random-audit", "random-0", {"x": torch.tensor([1.0])}),
        ("random-audit", "random-1", {"x": torch.tensor([2.0])}),
    ]
    constructed = []

    def energy_direction(index):
        constructed.append(index)
        return {"x": torch.tensor([float(index)])}

    directions = _direction_stream(
        random_directions, [7, 9], energy_direction
    )
    assert next(directions)[1] == "random-0"
    assert next(directions)[1] == "random-1"
    assert constructed == []
    assert next(directions)[1] == "energy-map-7"
    assert constructed == [7]
    assert random_directions == []
    assert next(directions)[1] == "energy-map-9"
    assert constructed == [7, 9]


def test_source_resolved_arrays_are_json_native():
    value = _json_list(torch.tensor([[1, 2], [3, 4]]).numpy())
    assert value == [[1, 2], [3, 4]]
    assert json.loads(json.dumps({"value": value})) == {"value": value}


def test_intervention_lineage_uses_canonical_state_digest():
    binding = {
        "checkpoint_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    assert _lineage_digest(binding) == "b" * 64
