#!/usr/bin/env python3
"""Generate the archive-bound row-map collapse confirmation protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from row_map_confirmation_specs import (  # noqa: E402
    ARCHITECTURE,
    CHECKPOINTS,
    CONFIRMATION_STATEMENTS,
    FIXED_CONSTANTS,
    INTERVENTIONS,
    INTERVENTION_SELECTION,
    RESOURCE_CAPS,
    RUNS,
    SCHEMA_VERSION,
    STAGES,
    STATUS,
)


OUTPUT = ROOT / "experiments" / "protocols" / "row_map_collapse_confirmation"


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def expected_files():
    files = {}
    files[Path("campaign_design.json")] = _json({
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "architecture": ARCHITECTURE,
        "runs": list(RUNS),
        "checkpoints": CHECKPOINTS,
        "fixed_constants": FIXED_CONSTANTS,
        "resource_caps": RESOURCE_CAPS,
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": list(STAGES),
        "confirmation_statements": list(CONFIRMATION_STATEMENTS),
        "interventions": INTERVENTIONS,
        "intervention_selection": INTERVENTION_SELECTION,
        "ownership_rule": (
            "Seed 1234 and construction contexts own selections and bounds. "
            "Seeds 2222 and 3333 and disjoint contexts test them unchanged."
        ),
    })
    for stage in STAGES:
        files[Path(f"{stage['id'].lower()}_protocol.json")] = _json({
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "stage": stage,
            "fixed_constants": FIXED_CONSTANTS,
            "producer": (
                "experiments/confirm/row_map_plga_live.py"
                if stage["id"] == "A"
                else "experiments/confirm/row_map_live.py"),
            "analyzer": "experiments/analysis/analyze_row_map_confirmation.py",
            "scientific_arrays": "npz_without_pickle",
        })
    files[Path("record_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PLDR row-map confirmation stage record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "campaign_id", "stage", "status", "inputs",
            "outputs", "checks", "resources", "record_sha256",
        ],
        "properties": {
            "schema_version": {"const": "pldr-row-map-stage-record-v1"},
            "campaign_id": {"type": "string", "minLength": 1},
            "stage": {"enum": [stage["id"] for stage in STAGES]},
            "status": {"enum": ["PASS", "FAIL", "BLOCKED"]},
            "inputs": {"type": "array"},
            "outputs": {"type": "array"},
            "checks": {"type": "object"},
            "resources": {"type": "object"},
            "record_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    })
    files[Path("registry_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PLDR row-map context registry",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "context_length", "construction_chunks",
            "validation_chunks", "application_pairs", "graph_rule", "checkpoint_steps",
            "construction_seed", "validation_seeds", "tokens_sha256",
            "tokenizer_sha256", "registry_sha256",
        ],
        "properties": {
            "schema_version": {"const": "pldr-row-map-registry-v1"},
            "context_length": {"const": 256},
            "construction_chunks": {
                "type": "array", "minItems": 8, "maxItems": 8,
                "uniqueItems": True},
            "validation_chunks": {
                "type": "array", "minItems": 16, "maxItems": 16,
                "uniqueItems": True},
            "application_pairs": {
                "type": "array", "minItems": 24, "maxItems": 24},
            "graph_rule": {
                "const": "unit-weight-chain-on-flattened-registered-rows-v1"},
            "checkpoint_steps": {"const": [1000, 2000, 4000, 8000, 16000, 24000]},
            "construction_seed": {"const": 1234},
            "validation_seeds": {"const": [2222, 3333]},
            "tokens_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "tokenizer_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "registry_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    })
    files[Path("README.md")] = (
        "# Row-map collapse confirmation protocols\n\n"
        "This generated directory specifies the Q/G/H/D/I/A/R campaign on "
        "the three retained 20.6M-parameter PLDR trajectories. Geometry uses "
        "all six model checkpoints. Exact AdamW successors and continuations "
        "use the 24,000-update checkpoints because those are the retained "
        "states that include optimizer moments. No missing moments are "
        "estimated or imputed.\n\n"
        "All safety factors, the factor-three materiality threshold, the "
        "zero policy, construction-only intervention selection, carried "
        "resource budget, and float64 residual tolerances are fixed in "
        "`campaign_design.json` before execution. Q is a hard gate.\n"
    ).encode("utf-8")
    checksums = "".join(
        f"{_digest(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda item: item.as_posix())
    )
    files[Path("CHECKSUMS.sha256")] = checksums.encode("ascii")
    return files


def generate(check=False):
    expected = expected_files()
    actual = {
        path.relative_to(OUTPUT)
        for path in OUTPUT.rglob("*") if path.is_file()
    } if OUTPUT.is_dir() else set()
    stale = [
        path.as_posix() for path, payload in expected.items()
        if not (OUTPUT / path).is_file()
        or (OUTPUT / path).read_bytes() != payload
    ]
    unknown = sorted(actual - set(expected), key=lambda value: value.as_posix())
    if check:
        if stale or unknown:
            raise SystemExit(
                "row-map protocols are stale: "
                + ", ".join(stale + [f"unknown:{path}" for path in unknown]))
        print(f"row-map protocols: verified ({len(expected)} files)")
        return
    if unknown:
        raise SystemExit(
            "refusing to overwrite protocol directory with unknown files: "
            + ", ".join(path.as_posix() for path in unknown))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for relative, payload in expected.items():
        destination = OUTPUT / relative
        destination.write_bytes(payload)
    print(f"row-map protocols: wrote {len(expected)} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    generate(parser.parse_args().check)


if __name__ == "__main__":
    main()
