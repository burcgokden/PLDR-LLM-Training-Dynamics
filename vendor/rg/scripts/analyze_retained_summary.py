#!/usr/bin/env python3
"""Reduce the supplied retained block census to a guarded RG flow summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import (  # noqa: E402
    analyze_retained_summary,
    file_sha256,
)


def _format_integer(value: int) -> str:
    return f"{value:,}"


OUTCOME_TEXT = {
    "AGGREGATE_MEDIANS_BELOW_ONE_WITH_HETEROGENEOUS_CELLS":
        "aggregate medians below one with heterogeneous cell flow",
    "NO_REGISTERED_AGGREGATE_MEDIAN_CONTRACTION":
        "no registered aggregate median contraction",
}


def _format_optional(value: float | None, specification: str) -> str:
    return format(value, specification) if value is not None else r"\text{not applicable}"


def _format_optional_percent(value: float | None) -> str:
    return f"{100.0 * value:.1f}\\%" if value is not None else r"\text{not applicable}"





