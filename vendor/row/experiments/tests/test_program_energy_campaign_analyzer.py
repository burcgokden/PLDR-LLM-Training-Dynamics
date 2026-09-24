from __future__ import annotations

from pathlib import Path
import sys

import pytest


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(CONFIRM))

from analyze_program_energy_confirmation import analyze  # noqa: E402
from confirmation_artifacts import seal_record, write_json_atomic  # noqa: E402
from program_energy_protocol_specs import STAGES  # noqa: E402
from run_program_energy_campaign import (  # noqa: E402
    gated_replication_command,
    plan,
)


def _stage_record(stage):
    return seal_record({
        "schema_version": "pldr-composite-observability-stage-record-v3",
        "campaign_id": "aggregate-test",
        "stage": stage["id"],
        "time_semantics": stage["time_semantics"][0],
        "request_sha256": stage["id"].lower().ljust(64, "0"),
        "resolved_artifacts": [],
        "derived_measurements": {
            name: {
                "value": 1,
                "passed": True,
                "derivation": "test_recomputed_quantity",
            }
            for name in stage["required_checks"]
        },
        "resource_metadata": {
            "recomputed_gpu_hours": stage["gpu_hour_cap"],
            "checks": {"stage_gpu_hours": True},
            "passed": True,
        },
        "decision": "CONFIRMED",
    })


def _records(tmp_path, omit=None):
    paths = []
    for stage in STAGES:
        if stage["id"] == omit:
            continue
        path = tmp_path / f"{stage['id'].lower()}.json"
        write_json_atomic(path, _stage_record(stage))
        paths.append(path)
    return paths


def test_complete_typed_campaign_reaches_replay_stage(tmp_path):
    result = analyze(_records(tmp_path))
    assert result["stages"]["R"]["verdict"] == "CONFIRMED"
    assert result["theory_status"]["normal_attraction_confirmed"]
    assert result["theory_status"]["optimizer_corridor_confirmed"]
    assert result["theory_status"]["same_source_bridge_confirmed"]
    assert result["theory_status"]["complete_campaign_replayed"]
    assert result["resource_summary"]["within_hard_cap"]
    assert result["decision"] == "CONFIRMED"


def test_missing_dependency_blocks_replay_promotion(tmp_path):
    result = analyze(_records(tmp_path, omit="I"))
    assert result["stages"]["I"]["verdict"] == "PENDING"
    assert not result["stages"]["R"]["dependencies_satisfied"]
    assert result["stages"]["R"]["verdict"] == "NOT_CONFIRMED"
    assert result["decision"] == "PENDING"


def test_tampered_stage_record_is_rejected(tmp_path):
    record = _stage_record(STAGES[0])
    record["decision"] = "NOT_CONFIRMED"
    path = tmp_path / "tampered.json"
    write_json_atomic(path, record)
    with pytest.raises(ValueError, match="digest"):
        analyze([path])


def test_primary_training_plan_withholds_gated_replication():
    result = plan(
        "tokens.bin", "tokenizer.model", "registry.json", "campaign",
        ("cuda:0",),
    )
    assert len(result["training_commands"]) == 1
    assert "seed-3201-primary" in result["training_commands"][0]
    assert "seed-3202-replication" not in result["training_commands"][0]
    gated = result["gated_replication"]
    assert gated["release_condition"] == "CONFIRMED_Q_T_N_SEALED_RECORDS"
    assert [row["seed"] for row in gated["architecture_runs"]] == [3202]


def test_replication_command_requires_confirmed_sealed_qtn(tmp_path):
    records = _records(tmp_path)
    gate_records = [
        path for path in records if path.stem.upper() in {"Q", "T", "N"}
    ]
    result = gated_replication_command(
        records=gate_records,
        tokens="tokens.bin",
        tokenizer="tokenizer.model",
        registry="registry.json",
        output_root="campaign",
    )
    assert result["required_confirmed_stages"] == ["Q", "T", "N"]
    assert "seed-3202-replication" in result["training_command"]

    missing_t = [path for path in gate_records if path.stem.upper() != "T"]
    with pytest.raises(ValueError, match="confirmed sealed Q, T, and N"):
        gated_replication_command(
            records=missing_t,
            tokens="tokens.bin",
            tokenizer="tokenizer.model",
            registry="registry.json",
            output_root="campaign",
        )
