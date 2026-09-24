"""Protocol and schema tests for source-resolved confirmation."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.source_resolved_specs import (  # noqa: E402
    RESOURCE_BUDGET,
    STAGES,
    campaign_design,
    snapshot_updates,
    validate_design,
)
from confirm.source_resolved_policy import _windows  # noqa: E402
from scripts.gen_source_resolved_protocols import schemas  # noqa: E402
from scripts.stage_source_resolved_confirmation import _immutable  # noqa: E402


def _assert_strict_objects(value):
    if isinstance(value, dict):
        if value.get("type") == "object":
            assert value.get("additionalProperties") is False
            assert set(value.get("required", [])) == set(
                value.get("properties", {})
            )
        for child in value.values():
            _assert_strict_objects(child)
    elif isinstance(value, list):
        for child in value:
            _assert_strict_objects(child)


def test_campaign_has_real_producer_for_every_scientific_stage():
    design = campaign_design()
    validate_design(design)
    assert [stage["id"] for stage in STAGES] == [
        "Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"
    ]
    assert all(stage["producer"] for stage in STAGES)
    assert sum(stage["aggregate_gpu_hours_target"] for stage in STAGES) == 7.8
    assert RESOURCE_BUDGET["hard_aggregate_gpu_hours"] == 10.0


def test_timecourse_cadence_is_complete_and_exact():
    updates = snapshot_updates()
    assert len(updates) == 564
    assert updates[:3] == [0, 16, 32]
    assert updates[-2:] == [8992, 9000]
    assert all(right - left == 16 for left, right in zip(
        updates[:-2], updates[1:-1], strict=True
    ))


def test_intervention_windows_need_not_be_snapshot_updates():
    updates = snapshot_updates()
    windows = _windows({"updates": updates})
    assert windows == {
        "pre-onset": 1000,
        "onset": 4000,
        "late": 8000,
    }
    assert 1000 not in updates


def test_staged_immutable_membership_ignores_generated_bytecode(tmp_path):
    assert _immutable(tmp_path / "source/module.py", tmp_path)
    assert not _immutable(
        tmp_path / "source/__pycache__/module.cpython-314.pyc", tmp_path
    )
    assert not _immutable(tmp_path / "source/module.pyo", tmp_path)


def test_every_scientific_schema_is_typed_and_rejects_unknown_fields():
    values = schemas()
    assert set(values) == {
        "checkpoint_binding.schema.json",
        "edge.schema.json",
        "mask_response.schema.json",
        "block_response.schema.json",
        "rank_tube.schema.json",
        "timepoint.schema.json",
        "timecourse.schema.json",
        "intervention_prediction.schema.json",
        "intervention.schema.json",
        "construction_policy.schema.json",
        "analysis.schema.json",
        "qualification.schema.json",
    }
    for value in values.values():
        assert value["$schema"].endswith("2020-12/schema")
        _assert_strict_objects(value)
