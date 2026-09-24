#!/usr/bin/env python3
"""CLI for the source-frozen cross-arithmetic row-radius producer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import write_json_atomic  # noqa: E402
from confirm.source_frozen_radius import produce  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--data-order", type=Path, required=True)
    parser.add_argument("--prediction-device", required=True)
    parser.add_argument("--comparison-device", default="cpu")
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = produce(
        arguments.checkpoint,
        arguments.registry,
        arguments.tokens,
        arguments.data_order,
        arguments.prediction_device,
        arguments.comparison_device,
        protocol_directory=arguments.protocol_dir,
    )
    write_json_atomic(arguments.output, result)
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "technical_valid": result["technical_valid"],
        "radius_contract_closed": result["radius_contract_closed"],
        "energy_envelope_closed": result["energy_envelope_closed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
