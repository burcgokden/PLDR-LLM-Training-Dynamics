#!/usr/bin/env python3
"""Reject Lean escape hatches and enforce the declared axiom allowlist."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from collections import defaultdict
import subprocess
import sys
import tempfile
import time


ROOT = Path(
    os.environ.get("ROW_RGMAP_REPO_ROOT", Path(__file__).resolve().parents[1])
).resolve()
sys.path.insert(0, str(ROOT))



FORBIDDEN_NAMES = (
    "sorry",
    "admit",
    "axiom",
    "unsafe",
    "implemented_by",
    "native_decide",
)
FORBIDDEN = re.compile(r"\b(" + "|".join(FORBIDDEN_NAMES) + r")\b")
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
AXIOM_BLOCK = re.compile(
    r"'(?P<name>[^']+)' depends on axioms: \[(?P<axioms>[^]]*)\]",
    re.DOTALL,
)
NO_AXIOM_LINE = re.compile(
    r"^'(?P<name>[^']+)' does not depend on any axioms$",
    re.MULTILINE,
)


def build_lean(lake: str = "lake") -> dict[str, object]:
    """Build current sources before any compiled declaration is inspected."""

    started = time.perf_counter()
    completed = subprocess.run(
        [lake, "build"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            "Lean build failed before the formal audit:\n"
            + completed.stdout
            + completed.stderr
        )
    return {
        "command": [lake, "build"],
        "completed": True,
        "wall_seconds": elapsed,
        "toolchain": (ROOT / "lean-toolchain").read_text(encoding="utf-8").strip(),
    }


def strip_lean_comments(text: str) -> str:
    result: list[str] = []
    index = 0
    depth = 0
    while index < len(text):
        pair = text[index : index + 2]
        if pair == "/-":
            depth += 1
            index += 2
        elif pair == "-/" and depth:
            depth -= 1
            index += 2
        elif pair == "--" and depth == 0:
            end = text.find("\n", index)
            if end < 0:
                break
            result.append("\n")
            index = end + 1
        else:
            if depth == 0:
                result.append(text[index])
            elif text[index] == "\n":
                result.append("\n")
            index += 1
    if depth:
        raise ValueError("unterminated Lean block comment")
    return "".join(result)


def forbidden_escape_hatches(text: str) -> list[str]:
    return sorted(set(FORBIDDEN.findall(strip_lean_comments(text))))


def check_escape_hatches(manifest: dict) -> None:
    for row in manifest["modules"]:
        module = row["module"]
        path = ROOT / (module.replace(".", "/") + ".lean")
        found = forbidden_escape_hatches(path.read_text(encoding="utf-8"))
        if found:
            raise ValueError(f"forbidden Lean escape hatches in {module}: {found}")






def equal_type_digest_groups(manifest: dict) -> list[dict[str, object]]:
    """Report declarations whose normalized elaborated statement types coincide."""

    groups: dict[str, list[str]] = defaultdict(list)
    for row in manifest["lean_type_fingerprint"]["declarations"]:
        groups[row["elaborated_type_sha256"]].append(row["fully_qualified_name"])
    return [
        {"elaborated_type_sha256": digest, "declarations": sorted(names)}
        for digest, names in sorted(groups.items())
        if len(names) > 1
    ]





