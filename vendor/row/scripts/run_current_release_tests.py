#!/usr/bin/env python3
"""Run the full current-release Python suite with one exact deselection."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
MANUSCRIPT = ROOT / "docs"
CONFIRM = EXPERIMENTS / "confirm"
sys.path.insert(0, str(CONFIRM))

DESELECTED_NODE = (
    "tests/test_gates_w6.py::"
    "test_corrected_wave5_report_matches_manual_application"
)




def environment_requires_reexec(
    current: Mapping[str, str],
    desired: dict[str, str],
) -> bool:
    """Return whether any missing, changed, or extra variable remains."""

    return dict(current) != desired


class DeselectionContractError(RuntimeError):
    """The current test gate did not deselect its one declared node."""


class TestExecutionContractError(RuntimeError):
    """The current test gate returned without executing a test call."""


def validate_exact_deselection(nodeids: list[str] | tuple[str, ...]) -> None:
    observed = tuple(nodeids)
    expected = (DESELECTED_NODE,)
    if observed != expected:
        raise DeselectionContractError(
            "current release requires exactly one deselected node; "
            f"expected={expected!r}, observed={observed!r}"
        )


def validate_nonzero_execution(executed_test_calls: int) -> None:
    if (
        isinstance(executed_test_calls, bool)
        or not isinstance(executed_test_calls, int)
        or executed_test_calls <= 0
    ):
        raise TestExecutionContractError(
            "current release requires at least one executed pytest call")


class ExactDeselectionGuard:
    """Observe pytest deselection and reject missing, renamed, or extra nodes."""

    def __init__(self) -> None:
        self.nodeids: list[str] = []
        self.executed_test_calls = 0

    def pytest_deselected(self, items) -> None:
        self.nodeids.extend(item.nodeid for item in items)

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call":
            self.executed_test_calls += 1

    def pytest_collection_finish(self, session) -> None:
        del session
        try:
            validate_exact_deselection(self.nodeids)
        except DeselectionContractError as error:
            import pytest

            raise pytest.UsageError(str(error)) from error








