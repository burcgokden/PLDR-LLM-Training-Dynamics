#!/usr/bin/env python3
"""Generate manuscript tables from the sealed layer-cocycle analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = (
    ROOT.parent
    / "experiment-data"
    / "manuscript-revisions"
    / "rev48"
    / "layer-resolved-cocycle-confirmation"
)
OUTPUT = ROOT / "docs" / "figures"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
        coefficient, exponent = f"{value:.3e}".split("e")
        return rf"{coefficient}\mathord{{\times}}10^{{{int(exponent)}}}"
    return f"{value:.4f}"


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _breakable_monospace(value: str, width: int = 8) -> str:
    chunks = [value[index : index + width] for index in range(0, len(value), width)]
    return r"\texttt{" + r"\allowbreak{}".join(chunks) + "}"


def _precision_supplement(
    bundle: Path, inventory: dict[str, Any]
) -> tuple[dict[tuple[str, int], dict[str, Any]], int]:
    rows = {}
    false_zero_total = 0
    for item in inventory["precision"]:
        if int(item["step"]) != 65_536:
            continue
        with np.load(bundle / item["path"], allow_pickle=False) as record:
            false_zero = (
                (record["float32_coordinate_energy"] == 0.0)
                & (record["float64_coordinate_energy"] > 0.0)
            )
            false_zero_total += int(np.count_nonzero(false_zero))
            for layer in range(3):
                discrepancy = record["resolution_relative_discrepancy"][
                    :, layer, :
                ]
                rows[str(item["trajectory"]), layer] = {
                    "median_relative_error": float(np.median(discrepancy)),
                    "maximum_relative_error": float(np.max(discrepancy)),
                    "false_coordinate_zeros": int(
                        np.count_nonzero(false_zero[:, layer, :, :])
                    ),
                }
    return rows, false_zero_total










