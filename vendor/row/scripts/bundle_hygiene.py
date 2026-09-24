"""Canonical source-bundle path hygiene."""

from __future__ import annotations

from pathlib import Path


EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})


def is_excluded_path(relative: Path) -> bool:
    """Return whether a relative payload path is interpreter bytecode."""

    return (
        "__pycache__" in relative.parts
        or relative.suffix.lower() in EXCLUDED_SUFFIXES
    )


def assert_clean_tree(root: Path) -> None:
    """Reject bytecode files or cache directories anywhere below the root."""

    offenders = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if is_excluded_path(path.relative_to(root))
    )
    if offenders:
        raise ValueError(
            "bundle contains excluded interpreter bytecode: "
            + ", ".join(offenders)
        )


def relative_payload_files(root: Path) -> list[Path]:
    """Enumerate regular payload files with canonical bytecode exclusion."""

    return sorted(
        relative
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
        for relative in (path.relative_to(root),)
        if not is_excluded_path(relative)
    )
