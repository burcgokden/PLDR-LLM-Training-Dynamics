"""Qualification and launch-provenance checks for protocol v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .analysis import file_sha256, verify_record_seal


_REPOSITORY_PREFIXES = ("scripts/", "src/", "configs/")


def derive_seed(master_seed: int, domain: str) -> int:
    """Derive one domain-separated unsigned 32-bit seed."""

    if not isinstance(master_seed, int) or isinstance(master_seed, bool) or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    if not isinstance(domain, str) or not domain:
        raise ValueError("seed domain must be a nonempty string")
    payload = (
        b"pldr-row-rg-seed-v1\x00"
        + str(master_seed).encode("ascii")
        + b"\x00"
        + domain.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def validate_seed_registry(protocol: dict[str, Any]) -> dict[str, int]:
    """Recompute and validate every predeclared protocol seed."""

    randomness = protocol.get("randomness")
    if not isinstance(randomness, dict):
        raise ValueError("protocol randomness registry is missing")
    master_seed = randomness.get("master_seed")
    domains = randomness.get("domains")
    observed = randomness.get("derived_seeds")
    if (
        not isinstance(domains, list)
        or not domains
        or len(set(domains)) != len(domains)
        or not all(isinstance(domain, str) and domain for domain in domains)
        or not isinstance(observed, dict)
        or set(observed) != set(domains)
    ):
        raise ValueError("protocol seed domains or registry are invalid")
    expected = {domain: derive_seed(master_seed, domain) for domain in domains}
    if observed != expected:
        raise ValueError("protocol derived-seed registry does not replay")
    return expected



def repository_relative_source(raw_path: str, repository_root: Path) -> str | None:
    """Return a stable repository-relative source identity when available.

    Historical qualification records stored absolute worktree paths. Their
    suffix after a standard source-directory name remains an unambiguous
    repository identity and can be checked at the producing Git commit.
    """

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        normalized = candidate.as_posix().removeprefix("./")
        if normalized.startswith(_REPOSITORY_PREFIXES):
            return normalized
    else:
        try:
            return candidate.resolve().relative_to(repository_root.resolve()).as_posix()
        except ValueError:
            pass
    normalized = candidate.as_posix()
    for prefix in _REPOSITORY_PREFIXES:
        marker = f"/{prefix}"
        if marker in normalized:
            return prefix + normalized.split(marker, 1)[1]
    return None


def git_blob_descriptor(
    repository_root: Path, commit: str, repository_path: str
) -> tuple[str, int]:
    """Hash one source blob at a named commit, independently of the worktree."""

    prefix = subprocess.check_output(
        ["git", "rev-parse", "--show-prefix"], cwd=repository_root, text=True
    ).strip()
    result = subprocess.run(
        ["git", "show", f"{commit}:{prefix}{repository_path}"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest(), len(result.stdout)


def validate_source_descriptor(
    source: dict[str, Any],
    *,
    repository_root: Path | None,
    source_commit: str | None,
) -> None:
    """Validate a qualification source from Git when possible, else by file."""

    raw_path = source.get("repository_path", source.get("path", ""))
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("qualification imported-source identity is missing")
    commit = source.get("code_commit", source_commit)
    repository_path = None
    if repository_root is not None:
        repository_path = source.get("repository_path")
        if not isinstance(repository_path, str) or not repository_path:
            repository_path = repository_relative_source(raw_path, repository_root)
    if repository_root is not None and isinstance(commit, str) and repository_path:
        digest, size = git_blob_descriptor(repository_root, commit, repository_path)
    else:
        path = Path(raw_path)
        if repository_root is not None and not path.is_absolute():
            path = repository_root / path
        if not path.is_file():
            raise ValueError("qualification imported-source file is unavailable")
        digest, size = file_sha256(path), path.stat().st_size
    if source.get("sha256") != digest or source.get("size_bytes") != size:
        raise ValueError("qualification imported-source digest mismatch")


def validate_qualification(
    descriptor: dict[str, Any],
    parent: Path,
    protocol_path: Path,
    protocol: dict[str, Any],
    *,
    repository_root: Path | None = None,
    source_commit: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate the exact passing preflight record registered by the run."""

    raw_path = descriptor.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("run configuration has no qualification path")
    candidate = Path(raw_path)
    path = candidate.resolve() if candidate.is_absolute() else (parent / candidate).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"registered qualification record is missing: {path}")
    if descriptor.get("file_sha256") != file_sha256(path):
        raise ValueError("qualification file digest differs from the run configuration")
    record = json.loads(path.read_text(encoding="utf-8"))
    verify_record_seal(record)
    if (
        record.get("schema_version")
        != "pldr-row-rg-confirmation-v3-qualification-v1"
        or record.get("record_sha256") != descriptor.get("record_sha256")
        or record.get("protocol_id") != protocol.get("protocol_id")
        or record.get("protocol_file_sha256") != file_sha256(protocol_path)
        or not record.get("all_launch_gates_passed")
        or not record.get("gates")
        or not all(record["gates"].values())
    ):
        raise ValueError("qualification identity or launch gates are invalid")
    sources = record.get("qualification_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("qualification record has no imported-source registry")
    for source in sources:
        validate_source_descriptor(
            source,
            repository_root=repository_root,
            source_commit=record.get("code_commit", source_commit),
        )
    return path, record
