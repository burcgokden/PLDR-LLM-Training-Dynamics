#!/usr/bin/env python3
"""Plan, qualify, and analyze the separate RG-universality campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "analysis"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ANALYSIS))

from analyze_rg_universality import analyze  # noqa: E402
from qualification_transition_kernel_rg import qualify  # noqa: E402
from rg_universality_protocol_specs import (  # noqa: E402
    ARCHITECTURE_RUNS,
    BLOCK_SIZES,
    RESOURCE_CAPS,
    STAGES,
    THRESHOLDS,
)


def plan():
    return {
        "schema_version": "pldr-rg-universality-execution-plan-v1",
        "program": "separate_transition_kernel_rg_and_universality",
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": list(STAGES),
        "block_sizes": list(BLOCK_SIZES),
        "thresholds": THRESHOLDS,
        "architecture_runs": list(ARCHITECTURE_RUNS),
        "resource_caps": RESOURCE_CAPS,
        "stop_before_universality_rule": (
            "R2 closure and R3 embeddability must both confirm before U1 or U2"
        ),
    }


def _write(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output")
    qualify_parser = subparsers.add_parser("qualify")
    qualify_parser.add_argument("--output")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument(
        "--mode", choices=("kernel-flow", "universality"), required=True)
    analyze_parser.add_argument("--inputs", nargs="+", required=True)
    analyze_parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.command == "plan":
        _write(arguments.output, plan())
        return
    if arguments.command == "qualify":
        result = qualify()
        _write(arguments.output, result)
        if result["decision"] != "QUALIFIED":
            raise SystemExit(1)
        return
    result = analyze(arguments.inputs, arguments.mode)
    _write(arguments.output, result)
    if result["decision"] != "CONFIRMED_ON_REGISTERED_DOMAIN":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
