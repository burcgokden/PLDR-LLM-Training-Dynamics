#!/usr/bin/env python3
"""Generate the frozen Q/T/N/I/A/R confirmation protocol bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
ANALYSIS = ROOT / "experiments" / "analysis"
sys.path.insert(0, str(CONFIRM))
sys.path.insert(0, str(ANALYSIS))

from composite_confirmation_analysis import RAW_ARRAYS  # noqa: E402
from program_energy_protocol_specs import (  # noqa: E402
    ARCHITECTURE_RUNS,
    ARTIFACT_ROLES,
    BLOCKING_QUALIFICATION,
    RESOURCE_CAPS,
    STAGES,
)


OUT = ROOT / "experiments" / "protocols" / "program_bound_confirmation"
TABLE = ROOT / "docs" / "figures" / "program_energy_protocol_table.tex"
ANALYZER = (
    ROOT / "experiments" / "analysis"
    / "composite_confirmation_analysis.py"
)
OWNERS = (
    "experiments/train_run.py",
    "experiments/confirm/qualification_normal_stability.py",
    "experiments/confirm/confirmation_artifacts.py",
    "experiments/analysis/composite_confirmation_analysis.py",
    "experiments/confirm/capture_confirmation_stage.py",
    "experiments/confirm/live_confirmation_producers.py",
    "experiments/confirm/program_energy_protocol_specs_v3.py",
    "experiments/confirm/validated_row_map.py",
    "experiments/confirm/validated_directional_taylor_mixed.py",
    "experiments/confirm/validated_interval.py",
    "experiments/confirm/branch_resolved_state.py",
    "experiments/confirm/normal_stability.py",
    "experiments/confirm/constructive_block_comparison.py",
    "experiments/confirm/same_source_bridge.py",
)


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
        "utf-8")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tex(value):
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )








