#!/usr/bin/env python3
"""Run strict native producers for the orbitwise confirmation campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.orbitwise_source import (  # noqa: E402
    link_source_prediction,
    produce_gate_result,
    produce_source_prediction,
    seal_gate_prediction,
)
from confirm.orbitwise_specs import (  # noqa: E402
    CAMPAIGN_ID,
    QUALIFICATION_SCHEMA,
    campaign_design,
    validate_design,
)
from confirm.source_resolved_energy import (  # noqa: E402
    effective_gate_shape,
    endpoint_gate_shape_decomposition,
    reopening_budget_envelope,
)
from confirm.source_resolved_live import write_json_atomic  # noqa: E402
from confirm.strict_schema import load_json, validate  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "experiments" / "protocols" / "orbitwise_confirmation"
)
DEFAULT_BINDING_PROTOCOL = (
    ROOT / "experiments" / "protocols" / "source_resolved_confirmation"
)


def qualify(device_name: str, protocol_directory: Path) -> dict:
    """Exercise the new exact kernels, strict schema, and requested device."""

    design = campaign_design()
    validate_design(design)
    shape0 = np.asarray([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    shape1 = np.asarray([[0.8, 1.7, 2.0], [0.4, 1.2, 2.1]])
    gate0 = np.asarray([[1.0, 0.8, 1.2], [1.0, 0.9, 1.1]])
    gate1 = np.asarray([[0.6, 0.7, 0.9], [0.7, 0.8, 0.9]])
    effective = effective_gate_shape(shape0, gate0)
    decomposition = endpoint_gate_shape_decomposition(
        shape0, shape1, gate0, gate1
    )
    reopening = reopening_budget_envelope(
        np.asarray([[2.0, 0.0], [0.0, 4.0], [1.0, 3.0], [1.5, 5.0]])
    )
    bounds_pass = bool(
        np.all(effective["physical_energy"] >= effective["gate_lower_energy"])
        and np.all(
            effective["physical_energy"] <= effective["gate_upper_energy"]
        )
    )
    schema = load_json(protocol_directory / "qualification.schema.json")
    schema_rejects = False
    try:
        validate({"unexpected": True}, schema)
    except ValueError:
        schema_rejects = True

    device = torch.device(device_name)
    peak_allocated = 0
    peak_reserved = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    tensor = torch.arange(1024, device=device, dtype=torch.float64)
    device_exercised = tensor.device == device and float(tensor.sum()) > 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        device_exercised = (
            device_exercised and peak_allocated > 0 and peak_reserved > 0
        )
    checks = {
        "design_canonical": True,
        "effective_gate_bounds": bounds_pass,
        "three_channel_identity": (
            decomposition["maximum_abs_gain_residual"] < 1.0e-12
        ),
        "reopening_envelope": (
            reopening["minimum_envelope_slack"] >= -1.0e-12
        ),
        "zero_energy_reopening": bool(
            reopening["reopening"][0, 1] == 4.0
        ),
        "schema_rejects_unknown_field": schema_rejects,
        "requested_device_exercised": bool(device_exercised),
    }
    result = {
        "schema_version": QUALIFICATION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "device": str(device),
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(result, schema)
    return result


def _common_protocols(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protocol-dir", type=Path, default=DEFAULT_PROTOCOL
    )
    parser.add_argument(
        "--binding-protocol-dir",
        type=Path,
        default=DEFAULT_BINDING_PROTOCOL,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    qualification = subparsers.add_parser("qualify")
    qualification.add_argument("--device", default="cpu")
    qualification.add_argument("--output", type=Path, required=True)
    qualification.add_argument("--require-pass", action="store_true")
    _common_protocols(qualification)

    predict = subparsers.add_parser("source-predict")
    predict.add_argument("--source-binding", type=Path, required=True)
    predict.add_argument("--registry", type=Path, required=True)
    predict.add_argument("--tokens", type=Path, required=True)
    predict.add_argument("--data-order", type=Path, required=True)
    predict.add_argument("--device", required=True)
    predict.add_argument("--output", type=Path, required=True)
    _common_protocols(predict)

    link = subparsers.add_parser("source-link")
    link.add_argument("--prediction", type=Path, required=True)
    link.add_argument("--source-binding", type=Path, required=True)
    link.add_argument("--endpoint-binding", type=Path, required=True)
    link.add_argument("--registry", type=Path, required=True)
    link.add_argument("--tokens", type=Path, required=True)
    link.add_argument("--data-order", type=Path, required=True)
    link.add_argument("--device", required=True)
    link.add_argument("--output", type=Path, required=True)
    _common_protocols(link)

    gate = subparsers.add_parser("seal-gate")
    gate.add_argument("--source-binding", type=Path, required=True)
    gate.add_argument("--output", type=Path, required=True)
    _common_protocols(gate)

    gate_result = subparsers.add_parser("gate-result")
    gate_result.add_argument("--prediction", type=Path, required=True)
    gate_result.add_argument("--control-log", type=Path, required=True)
    gate_result.add_argument("--frozen-log", type=Path, required=True)
    gate_result.add_argument("--registry", type=Path, required=True)
    gate_result.add_argument("--output", type=Path, required=True)
    _common_protocols(gate_result)

    arguments = parser.parse_args()
    command = sys.argv[:]
    protocol = arguments.protocol_dir.resolve()
    binding_protocol = arguments.binding_protocol_dir.resolve()
    if arguments.command == "qualify":
        result = qualify(arguments.device, protocol)
    elif arguments.command == "source-predict":
        result = produce_source_prediction(
            arguments.source_binding,
            arguments.registry,
            arguments.tokens,
            arguments.data_order,
            arguments.device,
            binding_protocol_directory=binding_protocol,
            orbitwise_protocol_directory=protocol,
            producer_command=command,
        )
    elif arguments.command == "source-link":
        result = link_source_prediction(
            arguments.prediction,
            arguments.source_binding,
            arguments.endpoint_binding,
            arguments.registry,
            arguments.tokens,
            arguments.data_order,
            arguments.device,
            binding_protocol_directory=binding_protocol,
            orbitwise_protocol_directory=protocol,
            producer_command=command,
        )
    elif arguments.command == "seal-gate":
        result = seal_gate_prediction(
            arguments.source_binding,
            binding_protocol_directory=binding_protocol,
            orbitwise_protocol_directory=protocol,
        )
    else:
        result = produce_gate_result(
            arguments.prediction,
            arguments.control_log,
            arguments.frozen_log,
            arguments.registry,
            orbitwise_protocol_directory=protocol,
        )
    write_json_atomic(arguments.output, result)
    print(json.dumps({
        "command": arguments.command,
        "output": str(arguments.output.resolve()),
        "technical_valid": result["technical_valid"],
    }, sort_keys=True))
    if (
        arguments.command == "qualify"
        and arguments.require_pass
        and not result["technical_valid"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
