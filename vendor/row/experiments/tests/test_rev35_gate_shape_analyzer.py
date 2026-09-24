"""Integration and mutation tests for the fresh gate-shape analyzer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis import analyze_gate_shape_confirmation as gate_analyzer
from analysis.analyze_gate_shape_confirmation import analyze_stage
from confirm.confirmation_artifacts import validate_measurement_registry
from confirm.gate_shape_evidence import (
    digest_object,
    seal_request,
    write_json_atomic,
    write_npz_atomic,
)
from confirm.gate_shape_live import qualify, realize_fixture


def _request(tmp_path, protocol, mutate=False):
    fixture = tmp_path / "fixture.npz"
    q0 = tmp_path / "q0.npz"
    q1 = tmp_path / "q1.npz"
    realize_fixture(SimpleNamespace(output=fixture))
    qualify(SimpleNamespace(fixture=fixture, device="cpu", output=q0))
    with np.load(q0, allow_pickle=False) as archive:
        raw0 = {name: archive[name].copy() for name in archive.files}
    raw0["device"] = np.asarray("cuda:0")
    raw1 = {name: value.copy() for name, value in raw0.items()}
    raw1["device"] = np.asarray("cuda:1")
    write_npz_atomic(q0, **raw0)
    write_npz_atomic(q1, **raw1)
    if mutate:
        with np.load(q1, allow_pickle=False) as archive:
            mutated = {name: archive[name].copy() for name in archive.files}
        mutated["row_output"][0, 0] += 0.25
        write_npz_atomic(q1, **mutated)
    request = seal_request(
        "Q", "rev35-test", tmp_path,
        [
            "fixture=fixture.npz",
            "qualification_cuda0=q0.npz",
            "qualification_cuda1=q1.npz",
        ],
        protocol,
    )
    path = tmp_path / "request.json"
    write_json_atomic(path, request)
    return path


def test_q_analyzer_recomputes_real_row_map_fixture(tmp_path):
    protocol = (
        Path(__file__).resolve().parents[1]
        / "protocols/gate_shape_confirmation/q_protocol.json")
    record = analyze_stage(_request(tmp_path, protocol))
    assert record["decision"] == "CONFIRMED"
    assert record["measurements"]["gate_factorization_replays"]["value"] < 1e-10


def test_q_analyzer_rejects_mutated_deciding_tensor(tmp_path):
    protocol = (
        Path(__file__).resolve().parents[1]
        / "protocols/gate_shape_confirmation/q_protocol.json")
    record = analyze_stage(_request(tmp_path, protocol, mutate=True))
    assert record["decision"] == "NOT_CONFIRMED"
    assert not record["measurements"]["gate_factorization_replays"]["passed"]
    assert not record["measurements"][
        "cross_device_arrays_agree_with_declared_tolerance"]["passed"]


def test_complete_checkpoint_accepts_signed_gate_shape_registry():
    registry = {
        "schema_version": "pldr-gate-shape-registry-v1",
        "context_length": 256,
        "construction": [{"id": "c0", "chunk_index": 10}],
        "validation": [{"id": "v0", "chunk_index": 20}],
        "application_pairs": [
            {"id": "a0", "left_chunk": 30, "right_chunk": 31}],
        "anchors": [768, 3072, 5632],
        "pair_rule": "adjacent-flattened-row-pairs-v1",
        "dataset_sha256": "1" * 64,
        "tokenizer_sha256": "2" * 64,
    }
    registry["registry_sha256"] = digest_object(registry)
    assert validate_measurement_registry(registry, require_nonempty=True)
    registry["anchors"] = [3072, 768]
    registry["registry_sha256"] = digest_object({
        key: value for key, value in registry.items()
        if key != "registry_sha256"
    })
    try:
        validate_measurement_registry(registry, require_nonempty=True)
    except ValueError as error:
        assert "geometry" in str(error)
    else:
        raise AssertionError("unsorted gate-shape anchors were accepted")


def test_rev35_decisions_are_not_producer_or_placeholder_verdicts():
    root = Path(__file__).resolve().parents[2]
    analyzer = (
        root / "experiments/analysis/analyze_gate_shape_confirmation.py"
    ).read_text(encoding="utf-8")
    application = (
        root / "experiments/confirm/gate_shape_application.py"
    ).read_text(encoding="utf-8")
    assert '"all_gate_shape_energies_recomputed": False' not in analyzer
    assert '"transient_expansion_permitted": True' not in analyzer
    assert "margin_qualified=np.asarray" not in application
    assert "decision_preserved=np.asarray" not in application
    protocol = json.loads((
        root / "experiments/protocols/gate_shape_confirmation/a_protocol.json"
    ).read_text(encoding="utf-8"))
    assert protocol["producer"].endswith("gate_shape_application.py")


def test_campaign_uses_the_audited_resource_record(monkeypatch):
    records = {}
    for stage in gate_analyzer.STAGES:
        records[stage["id"]] = {
            "campaign_id": "resource-authority",
            "stage": stage["id"],
            "decision": "CONFIRMED",
            "record_sha256": "a" * 64,
            "recomputed_gpu_hours": 99.0,
            "recomputed_wall_clock_hours": 99.0,
        }
    records["R"]["recomputed_gpu_hours"] = 1.25
    records["R"]["recomputed_wall_clock_hours"] = 2.5
    monkeypatch.setattr(
        gate_analyzer, "analyze_stage", lambda stage_id: records[stage_id])

    campaign = gate_analyzer.analyze_campaign(list(records))

    assert campaign["decision"] == "CONFIRMED"
    assert campaign["recomputed_gpu_hours"] == 1.25
    assert campaign["recomputed_wall_clock_hours"] == 2.5
