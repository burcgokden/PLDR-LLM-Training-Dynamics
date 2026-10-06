"""Evidence-binding and protocol regressions for the gate-shape program."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from confirm.gate_shape_evidence import (
    ledger_identity,
    load_npz,
    load_request,
    seal_request,
    write_json_atomic,
    write_npz_atomic,
)
from confirm.gate_shape_protocol_specs import (
    ARCHITECTURE,
    RESOURCE_CAPS,
    RAW_REQUIRED,
    STAGES,
    STATUS,
)
from train_run import canonical_event_record


def test_config_row_cannot_override_resolved_lineage_identity():
    row = canonical_event_record(
        "resolved-run", "resolved-lineage",
        {"event": "config", "run_id": "", "lineage_root": ""}, 0.0)
    assert row["run_id"] == "resolved-run"
    assert row["lineage_root"] == "resolved-lineage"


def test_realistic_continuation_ledger_has_one_resolved_lineage(tmp_path):
    ledger = tmp_path / "log.jsonl"
    rows = [
        canonical_event_record(
            "continuation", "source-lineage",
            {"event": "config", "run_id": "", "lineage_root": ""}, 0.0),
        canonical_event_record(
            "continuation", "source-lineage", {"step": 1}, 0.0),
        canonical_event_record(
            "continuation", "source-lineage",
            {"event": "final", "step": 8}, 0.0),
    ]
    ledger.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    identity = ledger_identity(ledger, expected_updates=8)
    assert identity["run_id"] == "continuation"
    assert identity["lineage_root"] == "source-lineage"


def test_npz_writer_enforces_real_npz_extension(tmp_path):
    archive = tmp_path / "raw.npz"
    write_npz_atomic(archive, value=np.arange(4, dtype=np.float64))
    assert np.array_equal(load_npz(archive, required=("value",))["value"], np.arange(4))
    with pytest.raises(ValueError, match=".npz"):
        write_npz_atomic(tmp_path / "raw.bin", value=np.zeros(1))


def test_request_rejects_self_pair_and_path_escape(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "one.npz").write_bytes(b"one")
    protocol = tmp_path / "q_protocol.json"
    write_json_atomic(protocol, {"stage": "Q"})
    with pytest.raises(ValueError, match="two evidence roles"):
        seal_request(
            "Q", "campaign", root,
            ["left=one.npz", "right=one.npz"], protocol)
    with pytest.raises(ValueError, match="normalized"):
        seal_request("Q", "campaign", root, ["left=../one.npz"], protocol)


def test_sealed_request_replays_and_detects_mutation(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    evidence = root / "fixture.npz"
    evidence.write_bytes(b"fixture-v1")
    protocol = tmp_path / "q_protocol.json"
    write_json_atomic(protocol, {"stage": "Q"})
    request = seal_request(
        "Q", "campaign", root, ["fixture=fixture.npz"], protocol)
    request_path = root / "request.json"
    write_json_atomic(request_path, request)
    loaded, resolved, loaded_protocol = load_request(request_path)
    assert loaded["stage"] == "Q"
    assert resolved["fixture"] == evidence.resolve()
    assert loaded_protocol == protocol.resolve()
    evidence.write_bytes(b"fixture-mutated")
    with pytest.raises(ValueError, match="changed"):
        load_request(request_path)


def test_protocol_is_low_dimensional_and_resource_feasible():
    assert ARCHITECTURE["gate_sector_dimension_per_layer"] == 16
    assert ARCHITECTURE["lifted_gate_dimension_per_layer"] == 48
    assert ARCHITECTURE["anchor_source_steps"] == [768, 3072, 5632]
    assert STATUS == "DRAFT_PENDING_TWO_DEVICE_QUALIFICATION"
    scientific = sum(stage["gpu_hour_cap"] for stage in STAGES)
    assert scientific == pytest.approx(2.45)
    assert scientific + RESOURCE_CAPS["engineering_reserve_gpu_hours"] < (
        RESOURCE_CAPS["hard_gpu_hours"])


def test_protocol_dependency_order_and_acceptance_semantics():
    stage_by_id = {stage["id"]: stage for stage in STAGES}
    assert list(stage_by_id) == ["Q", "T", "TR", "G", "C", "I", "A", "R"]
    assert stage_by_id["G"]["dependencies"] == ["T", "TR"]
    assert "global_gate_bound_encloses_rows" in (
        stage_by_id["Q"]["required_checks"])
    assert "capturing_block_contains_transient_expansion" in (
        stage_by_id["C"]["required_checks"])
    assert "construction_mechanism_label_replicates" in (
        stage_by_id["C"]["required_checks"])
    assert "all_five_registered_arms_complete" in (
        stage_by_id["I"]["required_checks"])
    assert "intervention_geometry_raw_archives" in RAW_REQUIRED["I"]
    assert "final_gate_freeze_has_zero_gate_work" in (
        stage_by_id["I"]["required_checks"])
    assert "target_energy_not_read_while_constructing_envelope" in (
        stage_by_id["C"]["required_checks"])
    assert "empty_qualified_set_supports_no_decision_claim" in (
        stage_by_id["A"]["required_checks"])
