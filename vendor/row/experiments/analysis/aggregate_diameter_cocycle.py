#!/usr/bin/env python3
"""Aggregate immutable edge reports into chronological diameter cocycles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "confirm"))

from contrast_energy import aggregate_cocycle  # noqa: E402


REPORT_SCHEMA = "pldr-contrast-energy-analysis-v1"


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_edge(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    prediction = value.get("prediction", {}) if isinstance(value, dict) else {}
    if (
        value.get("schema_version") != REPORT_SCHEMA
        or prediction.get("decision_uses_successor") is not False
        or prediction.get("lower_multiplier") is None
        or prediction.get("upper_multiplier") is None
    ):
        raise ValueError(f"{source} is not a ratio-bearing source report")
    metadata = value.get("metadata", {})
    required = {
        "trajectory", "role", "seed", "layer", "global_step_before",
        "global_step_after", "registry_sha256", "lock_sha256",
    }
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise ValueError(f"{source} has incomplete source metadata")
    step_start = int(metadata["global_step_before"])
    step_stop = int(metadata["global_step_after"])
    if step_stop != step_start + 1:
        raise ValueError(f"{source} is not a certified one-update edge")
    return {
        "path": str(source),
        "sha256": sha256_path(source),
        "step_start": step_start,
        "step_stop": step_stop,
        "trajectory": str(metadata["trajectory"]),
        "role": str(metadata["role"]),
        "seed": int(metadata["seed"]),
        "layer": int(metadata["layer"]),
        "registry_sha256": str(metadata["registry_sha256"]),
        "lock_sha256": str(metadata["lock_sha256"]),
        "lower_multiplier": float(prediction["lower_multiplier"]),
        "upper_multiplier": float(prediction["upper_multiplier"]),
        "decision": prediction["decision"],
    }


def _contiguous_blocks(
    edges: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    blocks: list[list[dict[str, Any]]] = []
    for edge in edges:
        if not blocks or edge["step_start"] != blocks[-1][-1]["step_stop"]:
            blocks.append([edge])
        else:
            blocks[-1].append(edge)
    return blocks


def aggregate(paths: list[str | Path]) -> dict[str, Any]:
    edges = [load_edge(path) for path in paths]
    keys = [
        (edge["trajectory"], edge["layer"], edge["registry_sha256"])
        for edge in edges
    ]
    groups = {}
    for key in sorted(set(keys), key=lambda value: (str(value[0]), value[1:])):
        selected = sorted(
            (
                edge for edge, edge_key in zip(edges, keys, strict=True)
                if edge_key == key
            ),
            key=lambda edge: edge["step_start"],
        )
        starts = [edge["step_start"] for edge in selected]
        if len(starts) != len(set(starts)) or any(
            right <= left for left, right in zip(starts, starts[1:])
        ):
            raise ValueError("cocycle edges are not strictly chronological")
        blocks = _contiguous_blocks(selected)
        block_reports = [
            {
                "step_start": block[0]["step_start"],
                "step_stop": block[-1]["step_stop"],
                "edges": block,
                "analysis": aggregate_cocycle(block),
            }
            for block in blocks
        ]
        groups[f"{key[0]}:layer{key[1]}"] = {
            "trajectory": key[0],
            "layer": key[1],
            "registry_sha256": key[2],
            "sampled_edge_count": len(selected),
            "contiguous_block_count": len(block_reports),
            "longest_contiguous_block": max(map(len, blocks)),
            "blocks": block_reports,
        }
    return {
        "schema_version": "pldr-diameter-cocycle-campaign-v1",
        "input_count": len(edges),
        "groups": groups,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = aggregate(arguments.reports)
    target = Path(arguments.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({
        "groups": len(report["groups"]), "output": str(target)
    }, sort_keys=True))


if __name__ == "__main__":
    main()
