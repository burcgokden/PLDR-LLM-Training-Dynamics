#!/usr/bin/env python3
"""Raw-artifact-to-energy-certificate production entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from artifact_binding import validate_request, verify_artifacts
from campaign_record import strict_dumps
from program_energy import construct_program_energy_edge, digest_object


def _load(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(strict_dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def construct_from_request(request, artifact_root):
    """Construct one edge from a closed request containing only file bindings."""

    validate_request(request)
    protocol_rows = [
        row for row in request["artifacts"] if row.get("role") == "protocol"
    ]
    if len(protocol_rows) != 1:
        raise ValueError("request must bind exactly one protocol")
    preliminary = verify_artifacts(
        artifact_root, protocol_rows, {"protocol"})
    protocol = _load(preliminary["protocol"]["resolved_path"])
    required_protocol = {
        "schema_version", "stage", "time_semantics",
        "request_artifact_roles", "constructor",
        "measurement_constructor_sha256", "rg_requirement",
        "rg_required_checks",
    }
    if not isinstance(protocol, dict) or set(protocol) != required_protocol:
        raise ValueError("frozen protocol has missing or unknown fields")
    if protocol["schema_version"] != "pldr-program-energy-stage-protocol-v1":
        raise ValueError("unknown stage protocol")
    if (
        request["stage"] != protocol["stage"]
        or request["time_semantics"] not in protocol["time_semantics"]
    ):
        raise ValueError("request stage or time semantics disagrees with protocol")
    if (
        not isinstance(protocol["rg_requirement"], str)
        or not protocol["rg_requirement"]
        or not isinstance(protocol["rg_required_checks"], list)
        or not protocol["rg_required_checks"]
        or any(not isinstance(value, str) or not value
               for value in protocol["rg_required_checks"])
        or len(set(protocol["rg_required_checks"]))
            != len(protocol["rg_required_checks"])
    ):
        raise ValueError("frozen protocol has malformed RG obligations")
    roles = set(protocol["request_artifact_roles"])
    bindings = verify_artifacts(artifact_root, request["artifacts"], roles)
    if protocol["constructor"] != (
        "experiments/confirm/program_energy_confirmation.py"
    ):
        raise ValueError("protocol does not name this production constructor")
    if (
        bindings["measurement_constructor"]["sha256"]
        != protocol["measurement_constructor_sha256"]
    ):
        raise ValueError("measurement constructor disagrees with frozen protocol")

    primitives = _load(bindings["primitive_bundle"]["resolved_path"])
    if (
        primitives.get("successor_owner_sha256")
        != bindings["measurement_constructor"]["sha256"]
    ):
        raise ValueError("primitive bundle is not owned by its bound constructor")
    if (
        primitives.get("source_state_sha256")
        != bindings["source_checkpoint"]["sha256"]
        or primitives.get("target_state_sha256")
        != bindings["target_checkpoint"]["sha256"]
    ):
        raise ValueError("primitive source or target state binding is inconsistent")
    edge = construct_program_energy_edge(primitives)
    frozen = request["time_semantics"] == "FROZEN_CHECKPOINT"
    if frozen:
        semantic_binding = (
            primitives["source_state_sha256"]
            == primitives["target_state_sha256"]
            and primitives["source_stack"] == primitives["target_stack"]
            and all(float(value) == 0.0
                    for value in primitives["actual_update_jvp"])
        )
    else:
        semantic_binding = (
            primitives["source_state_sha256"]
            != primitives["target_state_sha256"]
        )
    edge["conditions"]["time_semantics_bound"] = semantic_binding
    edge["decision"] = (
        "QUALIFIED" if all(edge["conditions"].values()) else "REJECTED")
    edge["request_binding"] = {
        "campaign_id": request["campaign_id"],
        "stage": request["stage"],
        "time_semantics": request["time_semantics"],
        "artifacts": {
            role: {
                "path": row["path"],
                "sha256": row["sha256"],
            }
            for role, row in sorted(bindings.items())
        },
    }
    edge.pop("record_sha256")
    edge["record_sha256"] = digest_object(edge)
    return edge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    record = construct_from_request(
        _load(arguments.request), arguments.artifact_root)
    _write(arguments.output, record)
    print(arguments.output)


if __name__ == "__main__":
    main()
