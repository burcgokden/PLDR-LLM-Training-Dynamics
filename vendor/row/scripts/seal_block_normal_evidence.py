#!/usr/bin/env python3
"""Seal source-only block predictions and later native endpoint records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.evidence_seal import (  # noqa: E402
    load_json_object,
    seal_native,
    seal_source,
    write_new_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    source_parser = subparsers.add_parser("source")
    source_parser.add_argument("--spec", type=Path, required=True)
    source_parser.add_argument("--checkpoint", type=Path, required=True)
    source_parser.add_argument("--output", type=Path, required=True)

    native_parser = subparsers.add_parser("native")
    native_parser.add_argument("--source", type=Path, required=True)
    native_parser.add_argument("--checkpoint", type=Path, required=True)
    native_parser.add_argument("--observations", type=Path, required=True)
    native_parser.add_argument("--output", type=Path, required=True)

    arguments = parser.parse_args()
    if arguments.mode == "source":
        if not arguments.output.name.endswith(".source.json"):
            raise ValueError("source evidence output must end in .source.json")
        artifact = seal_source(
            load_json_object(arguments.spec), arguments.checkpoint
        )
    else:
        if not arguments.output.name.endswith(".native.json"):
            raise ValueError("native evidence output must end in .native.json")
        raw = json.loads(arguments.observations.read_text(encoding="utf-8"))
        observations = (
            raw["observations"]
            if isinstance(raw, dict) and set(raw) == {"observations"}
            else raw
        )
        artifact = seal_native(
            arguments.source, arguments.checkpoint, observations
        )
    write_new_json(arguments.output, artifact)
    print(json.dumps({
        "mode": arguments.mode,
        "output": str(arguments.output.resolve()),
        "comparison_count": len(
            artifact.get("comparison_bounds", artifact.get("observations", []))
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
