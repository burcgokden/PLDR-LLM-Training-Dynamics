#!/usr/bin/env python3
"""CLI for strict raw-artifact Q/T/N/I/A/R analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from confirmation_artifacts import write_json_atomic  # noqa: E402
from composite_confirmation_analysis import analyze_stage  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    with Path(arguments.request).open("r", encoding="utf-8") as stream:
        request = json.load(stream)
    result = analyze_stage(
        stage_id=arguments.stage,
        request=request,
        artifact_root=arguments.artifact_root,
    )
    write_json_atomic(arguments.output, result)
    if result["decision"] != "CONFIRMED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
