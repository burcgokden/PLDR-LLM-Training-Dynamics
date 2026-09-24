#!/usr/bin/env python3
"""Axiom audit for every theorem and lemma imported by the manuscript root."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
LEAN_ROOT = ROOT / "PldrLlmCurvatureSandpile.lean"
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
IMPORT_RE = re.compile(
    r"^import PldrLlmCurvatureSandpile\.([A-Za-z0-9_']+)\s*$",
    re.MULTILINE,
)
DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]+\]\s*)*"
    r"(?:(?:noncomputable|private|protected)\s+)*"
    r"(?:theorem|lemma)\s+([A-Za-z0-9_']+)",
)


def active_modules():
    source = LEAN_ROOT.read_text(encoding="utf-8")
    modules = IMPORT_RE.findall(source)
    if not modules or len(modules) != len(set(modules)):
        raise SystemExit("axiom audit: active root imports are empty or duplicated")
    return modules


def source_hygiene(modules):
    hidden = []
    finite_maximum = []
    hidden_pattern = re.compile(r"\(_h[^)\s:]*\s*:")
    maximum_pattern = re.compile(
        r"(?m)^\s*(?:noncomputable\s+)?def\s+finiteMaximum\b"
    )
    for module in modules:
        path = ROOT / "PldrLlmCurvatureSandpile" / f"{module}.lean"
        source = path.read_text(encoding="utf-8")
        for match in hidden_pattern.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            hidden.append(f"{module}.lean:{line}")
        for match in maximum_pattern.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            finite_maximum.append(f"{module}.lean:{line}")
    if hidden:
        raise SystemExit(
            "axiom audit: underscore-prefixed theorem hypotheses remain: "
            + ", ".join(hidden)
        )
    if len(finite_maximum) != 1:
        raise SystemExit(
            "axiom audit: expected one active finiteMaximum definition; found "
            + repr(finite_maximum)
        )
    print(
        "axiom audit: source hygiene OK "
        f"(0 hidden hypotheses, finiteMaximum at {finite_maximum[0]})"
    )


def declarations(modules):
    names = []
    for module in modules:
        path = ROOT / "PldrLlmCurvatureSandpile" / f"{module}.lean"
        if not path.is_file():
            raise SystemExit(f"axiom audit: missing active module {path}")
        stack = []
        pending_attributes = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("@[") and not re.search(
                r"\]\s*(?:theorem|lemma)\s+", stripped
            ):
                pending_attributes = True
                continue
            namespace = re.match(r"namespace\s+([A-Za-z0-9_'.]+)", stripped)
            if namespace:
                stack.append(namespace.group(1))
                pending_attributes = False
                continue
            ending = re.match(r"end\s+([A-Za-z0-9_'.]+)", stripped)
            if ending and stack and stack[-1] == ending.group(1):
                stack.pop()
                pending_attributes = False
                continue
            match = DECL_RE.match(line)
            if match:
                names.append(".".join(stack + [match.group(1)]))
            pending_attributes = False
    if not names:
        raise SystemExit("axiom audit: no theorems found in active modules")
    if len(names) != len(set(names)):
        raise SystemExit("axiom audit: duplicate qualified declaration")
    return names


def main():
    modules = active_modules()
    source_hygiene(modules)
    names = declarations(modules)
    probe = "import PldrLlmCurvatureSandpile\n" + "".join(
        f"#print axioms {name}\n" for name in names
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".lean", dir=ROOT, delete=False
    ) as stream:
        stream.write(probe)
        path = Path(stream.name)
    try:
        result = subprocess.run(
            ["lake", "env", "lean", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
    finally:
        path.unlink()
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit("axiom audit: probe failed to elaborate")

    joined = []
    for line in result.stdout.splitlines():
        if line[:1].isspace() and joined:
            joined[-1] += " " + line.strip()
        else:
            joined.append(line)
    checked = 0
    bad = []
    for line in joined:
        match = re.match(r"'(.+)' depends on axioms: \[(.*)\]", line)
        if match:
            checked += 1
            axioms = {
                value.strip()
                for value in match.group(2).split(",")
                if value.strip()
            }
            extra = sorted(axioms - ALLOWED)
            if extra:
                bad.append((match.group(1), extra))
        elif "does not depend on any axioms" in line:
            checked += 1

    print(
        f"axiom audit: {checked} declarations checked in "
        f"{len(modules)} active modules; allowed: {sorted(ALLOWED)}"
    )
    if checked != len(names):
        print(result.stdout)
        raise SystemExit(
            f"axiom audit: recognized {checked} of {len(names)} probes"
        )
    if bad:
        for name, axioms in bad:
            print(f"FORBIDDEN axiom dependency in {name}: {axioms}")
        raise SystemExit(1)
    print("axiom audit: OK")


if __name__ == "__main__":
    main()
