#!/usr/bin/env python3
"""Validate and optionally axiom-audit the historical Lean archive."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import tempfile



ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SOURCE = ROOT / "PldrLlmCurvatureSandpile"
ACTIVE_ROOT = ROOT / "PldrLlmCurvatureSandpile.lean"
ARCHIVE = ROOT / "archive" / "historical" / "lean-library"
ARCHIVE_SOURCE = ARCHIVE / "PldrLlmCurvatureSandpile"
ARCHIVE_ROOT = ARCHIVE / "PldrLlmCurvatureSandpile.lean"
ARCHIVE_MANIFEST = ARCHIVE / "ARCHIVE_MANIFEST.sha256"

EXPECTED_ARCHIVE_MODULES = 93
EXPECTED_ARCHIVE_THEOREMS = 765
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

IMPORT = re.compile(
    r"^import PldrLlmCurvatureSandpile\.([A-Za-z0-9_]+)$",
    re.MULTILINE,
)
NAMESPACE = re.compile(r"namespace\s+([A-Za-z0-9_'.]+)")
END_NAMESPACE = re.compile(r"end\s+([A-Za-z0-9_'.]+)")
THEOREM = re.compile(r"(?:theorem|lemma)\s+([A-Za-z0-9_']+)")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_paths(source):
    return {
        path.stem: path
        for path in sorted(source.glob("*.lean"))
    }


def root_imports(path):
    return set(IMPORT.findall(path.read_text(encoding="utf-8")))


def module_imports(paths):
    return {
        name: set(IMPORT.findall(path.read_text(encoding="utf-8")))
        for name, path in paths.items()
    }


def archive_files():
    files = []
    for path in sorted(ARCHIVE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ARCHIVE)
        if ".lake" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix == ".pyc":
            continue
        if path == ARCHIVE_MANIFEST:
            continue
        files.append(path)
    return files


def expected_manifest():
    return {
        str(path.relative_to(ARCHIVE)): sha256(path)
        for path in archive_files()
    }


def write_manifest():
    entries = expected_manifest()
    with ARCHIVE_MANIFEST.open("w", encoding="utf-8") as target:
        for relative, digest in sorted(entries.items()):
            target.write(f"{digest}  {relative}\n")
    print(
        f"Lean archive manifest: {len(entries)} files written to "
        f"{ARCHIVE_MANIFEST.relative_to(ROOT)}"
    )


def read_manifest():
    if not ARCHIVE_MANIFEST.is_file():
        raise RuntimeError("archive manifest is missing")
    entries = {}
    for line in ARCHIVE_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"malformed archive manifest line: {line}")
        if relative in entries:
            raise RuntimeError(f"duplicate archive manifest path: {relative}")
        entries[relative] = digest
    return entries


def validate_manifest():
    recorded = read_manifest()
    expected = expected_manifest()
    if recorded != expected:
        missing = sorted(set(expected) - set(recorded))
        extra = sorted(set(recorded) - set(expected))
        changed = sorted(
            relative
            for relative in set(expected) & set(recorded)
            if expected[relative] != recorded[relative]
        )
        raise RuntimeError(
            "archive manifest mismatch; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def theorem_names(paths):
    names = []
    for path in sorted(paths.values()):
        stack = []
        for line in path.read_text(encoding="utf-8").splitlines():
            match = NAMESPACE.match(line)
            if match:
                stack.append(match.group(1))
                continue
            match = END_NAMESPACE.match(line)
            if match and stack and stack[-1] == match.group(1):
                stack.pop()
                continue
            match = THEOREM.match(line)
            if match:
                names.append(".".join(stack + [match.group(1)]))
    return names




def audit_axioms(paths, names):
    probe = "import PldrLlmCurvatureSandpile\n" + "".join(
        f"#print axioms {name}\n" for name in names
    )
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".lean",
        dir=ARCHIVE,
        delete=False,
        encoding="utf-8",
    ) as target:
        target.write(probe)
        probe_path = Path(target.name)
    try:
        result = subprocess.run(
            [
                "lake",
                "--dir",
                str(ARCHIVE),
                "env",
                "lean",
                str(probe_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
    finally:
        probe_path.unlink()
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("archived axiom probe failed to elaborate")

    joined = []
    for line in result.stdout.splitlines():
        if line[:1].isspace() and joined:
            joined[-1] += " " + line.strip()
        else:
            joined.append(line)

    checked = 0
    forbidden = []
    for line in joined:
        match = re.match(r"'(.+)' depends on axioms: \[(.*)\]", line)
        if match:
            checked += 1
            axioms = {
                value.strip()
                for value in match.group(2).split(",")
                if value.strip()
            }
            extra = sorted(axioms - ALLOWED_AXIOMS)
            if extra:
                forbidden.append((match.group(1), extra))
        elif "does not depend on any axioms" in line:
            checked += 1

    if checked != len(names):
        raise RuntimeError(
            f"only {checked} of {len(names)} archived axiom probes "
            "produced recognizable output"
        )
    if forbidden:
        raise RuntimeError(f"forbidden archived axioms: {forbidden}")
    print(
        f"archived axiom audit: {checked} declarations checked; "
        f"allowed={sorted(ALLOWED_AXIOMS)}"
    )




