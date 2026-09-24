from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

from artifact_binding import REQUEST_SCHEMA_VERSION, sha256_path  # noqa: E402
from independent_energy_checker import check_record  # noqa: E402
from program_energy_confirmation import construct_from_request  # noqa: E402
from test_program_energy import primitive_fixture  # noqa: E402


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def bound_fixture(tmp_path, *, frozen=False):
    protocol = {
        "schema_version": "pldr-program-energy-stage-protocol-v1",
        "stage": "C2",
        "time_semantics": [
            "FROZEN_CHECKPOINT" if frozen else "FINITE_IMPLEMENTED_SCHEDULE"
        ],
        "request_artifact_roles": [
            "protocol", "primitive_bundle", "measurement_constructor",
            "source_checkpoint", "target_checkpoint",
        ],
        "constructor": "experiments/confirm/program_energy_confirmation.py",
        "measurement_constructor_sha256": None,
        "rg_requirement": "test the owned affine RG edge",
        "rg_required_checks": ["every_fine_edge_owned"],
    }
    protocol_path = tmp_path / "protocol.json"
    owner_path = tmp_path / "measurement_constructor.py"
    source_path = tmp_path / "source.pt"
    target_path = source_path if frozen else tmp_path / "target.pt"
    primitive_path = tmp_path / "primitives.json"
    _write_json(protocol_path, protocol)
    owner_path.write_text("# immutable measurement owner\n", encoding="utf-8")
    protocol["measurement_constructor_sha256"] = sha256_path(owner_path)
    _write_json(protocol_path, protocol)
    source_path.write_bytes(b"source")
    if not frozen:
        target_path.write_bytes(b"target")
    primitive = primitive_fixture()
    primitive["source_state_sha256"] = sha256_path(source_path)
    primitive["target_state_sha256"] = sha256_path(target_path)
    primitive["successor_owner_sha256"] = sha256_path(owner_path)
    if frozen:
        primitive["target_stack"] = list(primitive["source_stack"])
        primitive["actual_update_jvp"] = [0.0] * 4
    _write_json(primitive_path, primitive)
    paths = {
        "protocol": protocol_path,
        "primitive_bundle": primitive_path,
        "measurement_constructor": owner_path,
        "source_checkpoint": source_path,
        "target_checkpoint": target_path,
    }
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "campaign_id": "test-campaign",
        "stage": "C2",
        "time_semantics": (
            "FROZEN_CHECKPOINT" if frozen else "FINITE_IMPLEMENTED_SCHEDULE"
        ),
        "artifacts": [
            {"role": role, "path": path.name, "sha256": sha256_path(path)}
            for role, path in paths.items()
        ],
    }
    return request, paths


def test_closed_request_constructs_and_replays(tmp_path):
    request, _ = bound_fixture(tmp_path)
    record = construct_from_request(request, tmp_path)
    assert record["decision"] == "QUALIFIED"
    assert record["conditions"]["time_semantics_bound"]
    assert check_record(record)["decision"] == "REPLAYED"


def test_frozen_semantics_requires_and_accepts_exactly_unchanged_state(tmp_path):
    request, _ = bound_fixture(tmp_path, frozen=True)
    record = construct_from_request(request, tmp_path)
    assert record["decision"] == "QUALIFIED"
    request["time_semantics"] = "FINITE_IMPLEMENTED_SCHEDULE"
    with pytest.raises(ValueError, match="disagrees"):
        construct_from_request(request, tmp_path)


def test_hash_mismatch_missing_file_and_caller_attestation_reject(tmp_path):
    request, paths = bound_fixture(tmp_path)
    bad = json.loads(json.dumps(request))
    bad["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        construct_from_request(bad, tmp_path)

    paths["target_checkpoint"].unlink()
    with pytest.raises(FileNotFoundError):
        construct_from_request(request, tmp_path)

    request, _ = bound_fixture(tmp_path)
    request["forcing"] = [0.0]
    with pytest.raises(ValueError, match="missing or unknown"):
        construct_from_request(request, tmp_path)


def test_wrong_constructor_owner_rejects(tmp_path):
    request, paths = bound_fixture(tmp_path)
    paths["measurement_constructor"].write_text("# changed owner\n", encoding="utf-8")
    for row in request["artifacts"]:
        if row["role"] == "measurement_constructor":
            row["sha256"] = sha256_path(paths["measurement_constructor"])
    with pytest.raises(ValueError, match="frozen protocol"):
        construct_from_request(request, tmp_path)
