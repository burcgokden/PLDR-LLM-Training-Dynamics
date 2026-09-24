#!/usr/bin/env python3
"""Generate the separate transition-kernel RG confirmation bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from rg_universality_protocol_specs import (  # noqa: E402
    ARCHITECTURE_RUNS,
    BLOCK_SIZES,
    RESOURCE_CAPS,
    STAGES,
    THRESHOLDS,
)


OUT = ROOT / "experiments" / "protocols" / "rg_universality_confirmation"
TABLE = ROOT / "docs" / "figures" / "rg_universality_protocol_table.tex"
OWNERS = (
    "experiments/confirm/transition_kernel_rg.py",
    "experiments/confirm/qualification_transition_kernel_rg.py",
    "experiments/confirm/rg_universality_protocol_specs.py",
    "experiments/analysis/analyze_rg_universality.py",
    "experiments/confirm/run_rg_universality_campaign.py",
)


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_path(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _tex(value):
    return (
        str(value).replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )








