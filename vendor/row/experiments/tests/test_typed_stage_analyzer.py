from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "experiments" / "analysis"
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(CONFIRM))

from analyze_normal_stability_stage import analyze_stage  # noqa: E402
from confirmation_artifacts import seal_record, write_json_atomic  # noqa: E402
from qualification_normal_stability import qualify  # noqa: E402
from run_program_energy_campaign import seal_request  # noqa: E402


@pytest.fixture(scope="module")
def q_record():
    return qualify("cpu")


def fixture(tmp_path, q_record):
    protocol_source = (
        ROOT / "experiments" / "protocols"
        / "program_bound_confirmation" / "q_protocol.json"
    )
    shutil.copy2(protocol_source, tmp_path / "q_protocol.json")
    write_json_atomic(tmp_path / "q_raw.json", q_record)
    (tmp_path / "source.bin").write_bytes(b"source-state-binding")
    (tmp_path / "target.bin").write_bytes(b"target-state-binding")
    write_json_atomic(tmp_path / "partition.json", {
        "schema_version": "pldr-confirmation-partitions-v3",
        "construction_ids": ["construction-0"],
        "validation_ids": ["validation-0"],
        "application_ids": ["application-0"],
        "dataset_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "measurement_registry_sha256": "c" * 64,
    })
    write_json_atomic(tmp_path / "resource.json", {
        "schema_version": "pldr-resource-metadata-v2",
        "work_units": 1,
        "gpu_device_seconds": [],
        "peak_gpu_bytes_each": 0,
        "peak_host_bytes": 1024,
        "retained_artifact_bytes": 1024,
        "wall_clock_seconds": 1,
    })
    bindings = [
        "stage_protocol=q_protocol.json",
        "raw_observations=q_raw.json",
        "source_complete_state=source.bin",
        "target_complete_state=target.bin",
        "partition_manifest=partition.json",
        "resource_metadata=resource.json",
    ]
    request = seal_request(
        stage="Q",
        campaign_id="strict-q-test",
        time_semantics="FINITE_IMPLEMENTED_SCHEDULE",
        artifact_root=tmp_path,
        bindings=bindings,
    )
    return request, bindings


def test_stage_analyzer_recomputes_q_from_bound_raw_record(
        tmp_path, q_record):
    request, _ = fixture(tmp_path, q_record)
    result = analyze_stage(
        stage_id="Q", request=request, artifact_root=tmp_path)
    assert result["decision"] == "CONFIRMED"
    assert result["resource_metadata"]["passed"]
    assert all(
        value["passed"] for value in result["derived_measurements"].values())


def test_caller_pass_injection_and_zero_resource_are_rejected(
        tmp_path, q_record):
    _, bindings = fixture(tmp_path, q_record)
    injected = copy.deepcopy(q_record)
    injected["caller_pass"] = True
    write_json_atomic(tmp_path / "q_raw.json", seal_record(injected))
    request = seal_request(
        stage="Q",
        campaign_id="strict-q-test",
        time_semantics="FINITE_IMPLEMENTED_SCHEDULE",
        artifact_root=tmp_path,
        bindings=bindings,
    )
    with pytest.raises(ValueError, match="missing or unknown fields"):
        analyze_stage(
            stage_id="Q", request=request, artifact_root=tmp_path)

    write_json_atomic(tmp_path / "q_raw.json", q_record)
    resource = json.loads((tmp_path / "resource.json").read_text())
    resource["peak_host_bytes"] = 0
    write_json_atomic(tmp_path / "resource.json", resource)
    request = seal_request(
        stage="Q",
        campaign_id="strict-q-test",
        time_semantics="FINITE_IMPLEMENTED_SCHEDULE",
        artifact_root=tmp_path,
        bindings=bindings,
    )
    with pytest.raises(ValueError, match="zero required resources"):
        analyze_stage(
            stage_id="Q", request=request, artifact_root=tmp_path)

def test_resealed_per_row_taylor_fabrication_is_not_confirmed(
        tmp_path, q_record):
    _, bindings = fixture(tmp_path, q_record)
    fabricated = copy.deepcopy(q_record)
    first = fabricated["jvp_and_taylor"]["taylor_rows"][0]
    first["certified_remainder_upper"] = (
        first["observed_remainder"] / 2)
    write_json_atomic(tmp_path / "q_raw.json", seal_record(fabricated))
    request = seal_request(
        stage="Q",
        campaign_id="strict-q-test",
        time_semantics="FINITE_IMPLEMENTED_SCHEDULE",
        artifact_root=tmp_path,
        bindings=bindings,
    )
    result = analyze_stage(
        stage_id="Q", request=request, artifact_root=tmp_path)
    assert result["decision"] == "NOT_CONFIRMED"
    assert not result["derived_measurements"][
        "taylor_remainder_enclosed"]["passed"]
