#!/usr/bin/env python3
"""Stage the source-restoring confirmation campaign with provenance."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import tempfile
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = (
    ROOT.parent
    / "experiment-data"
    / "campaigns"
    / "finite-increment"
    / "finite-increment-collapse-confirmation"
)
DEFAULT_OUTPUT = (
    ROOT.parent
    / "experiment-data"
    / "campaigns"
    / "source-restoring"
    / "source-restoring-collapse-confirmation"
)
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import digest_object, sha256_path  # noqa: E402
from confirm.source_restoring_specs import (  # noqa: E402
    CAMPAIGN_ID,
    DESIGN_SCHEMA,
    REGISTRY_SCHEMA,
    training_campaign_spec,
)


SOURCE_FILES = (
    "experiments/train_run.py",
    "experiments/pldr_model_v510.py",
    "experiments/power_law_attention_layer_v510.py",
    "experiments/confirm/confirmation_artifacts.py",
    "experiments/confirm/finite_increment_live.py",
    "experiments/confirm/capture_chronological_confirmation.py",
    "experiments/confirm/source_restoring.py",
    "experiments/confirm/source_restoring_specs.py",
    "experiments/confirm/source_restoring_live.py",
    "experiments/confirm/build_source_restoring_remainder.py",
    "experiments/confirm/capture_source_restoring_source.py",
    "experiments/confirm/produce_source_restoring_certificate.py",
    "experiments/confirm/cover_source_restoring_successor.py",
    "experiments/analysis/analyze_source_restoring_confirmation.py",
    "scripts/run_source_restoring_qualification.py",
    "scripts/gen_source_restoring_protocols.py",
    "scripts/stage_source_restoring_confirmation.py",
    "scripts/retain_model_only_checkpoint.py",
)


def _resolve_import(module: str) -> Path | None:
    parts = module.split(".")
    experiments = ROOT / "experiments"
    candidates = [
        ROOT.joinpath(*parts).with_suffix(".py"),
        ROOT.joinpath(*parts, "__init__.py"),
        experiments.joinpath(*parts).with_suffix(".py"),
        experiments.joinpath(*parts, "__init__.py"),
        experiments.joinpath("confirm", *parts).with_suffix(".py"),
        experiments.joinpath("analysis", *parts).with_suffix(".py"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def source_import_closure() -> list[Path]:
    """Return every repository-local module needed by staged entry points."""

    pending = [ROOT / relative for relative in SOURCE_FILES]
    selected: set[Path] = set()
    while pending:
        path = pending.pop().resolve()
        if path in selected:
            continue
        try:
            path.relative_to(ROOT)
        except ValueError as error:
            raise ValueError("a source entry escapes the repository") from error
        if not path.is_file():
            raise FileNotFoundError(f"source closure is missing {path}")
        selected.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
            ):
                modules.append(node.module)
        for module in modules:
            resolved = _resolve_import(module)
            if resolved is not None and resolved not in selected:
                pending.append(resolved)
    return sorted(selected)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def is_mutable_or_transient(path: Path, root: Path) -> bool:
    """Identify runtime products outside the immutable staged bundle."""

    parts = path.relative_to(root).parts
    return (
        "reports" in parts or "__pycache__" in parts or path.suffix == ".pyc"
    )


def build(parent: Path, output: Path) -> None:
    parent = parent.resolve()
    output = output.resolve()
    if not (parent / "MANIFEST.sha256").is_file():
        raise FileNotFoundError("the frozen parent campaign is missing")
    protocol_source = (
        ROOT
        / "experiments"
        / "protocols"
        / "source_restoring_confirmation"
    )
    if not (protocol_source / "CHECKSUMS.sha256").is_file():
        raise FileNotFoundError("generate source-restoring protocols first")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("orders", "protocol", "source", "reports"):
        (output / directory).mkdir(exist_ok=True)

    order_rows = []
    for source in sorted((parent / "orders").glob("*.npy")):
        target = output / "orders" / source.name
        copy_file(source, target)
        order_rows.append({
            "path": str(target.relative_to(output)),
            "sha256": sha256_path(target),
            "parent_path": str(source),
            "parent_sha256": sha256_path(source),
        })

    old_registry_path = parent / "protocol" / "registry.json"
    registry = json.loads(old_registry_path.read_text(encoding="utf-8"))
    registry["schema_version"] = REGISTRY_SCHEMA
    registry["campaign_id"] = CAMPAIGN_ID
    registry.pop("registry_sha256", None)
    registry["registry_sha256"] = digest_object(registry)
    (output / "protocol" / "registry.json").write_bytes(canonical(registry))
    (output / "protocol" / "training_campaign_spec.json").write_bytes(
        canonical(training_campaign_spec(registry["registry_sha256"]))
    )
    copy_file(
        protocol_source / "campaign_design.json",
        output / "protocol" / "campaign_design.json",
    )
    for name in (
        "training_campaign_schema.json",
        "source_artifact_schema.json",
        "validated_curvature_schema.json",
        "remainder_enclosure_schema.json",
        "edge_record_schema.json",
    ):
        copy_file(protocol_source / name, output / "protocol" / name)
    (output / "protocol" / "order_manifest.json").write_bytes(canonical({
        "schema_version": "pldr-source-restoring-order-manifest-v1",
        "campaign_id": CAMPAIGN_ID,
        "orders": order_rows,
    }))

    source_rows = []
    for source in source_import_closure():
        relative = source.relative_to(ROOT).as_posix()
        if not source.is_file():
            raise FileNotFoundError(f"source closure is missing {relative}")
        target = output / "source" / relative
        copy_file(source, target)
        source_rows.append({
            "path": str(target.relative_to(output)),
            "sha256": sha256_path(target),
            "bytes": target.stat().st_size,
        })
    source_manifest = {
        "schema_version": "pldr-source-restoring-source-manifest-v1",
        "campaign_id": CAMPAIGN_ID,
        "files": source_rows,
    }
    source_manifest["manifest_sha256"] = digest_object(source_manifest)
    (output / "source" / "SOURCE_MANIFEST.json").write_bytes(
        canonical(source_manifest)
    )

    launch = {
        "schema_version": "pldr-source-restoring-launch-v1",
        "campaign_id": CAMPAIGN_ID,
        "environment": {
            "TOKENS": "absolute path to refinedweb_tokens.npy",
            "TOKENIZER": "absolute path to tokenizer.model",
            "BUNDLE": "absolute path to this staged bundle",
        },
        "checkpoint_step_sets": {
            "construction": list(range(1000, 1025)),
            "heldout": [
                step + offset
                for step in (1000, 3000, 5000, 7000, 9000)
                for offset in (0, 1)
            ],
            "independent": list(range(1000, 1025)),
        },
        "commands": {
            "train": (
                "python3 source/experiments/train_run.py --name NAME "
                "--run-id RUN_ID --lineage-root RUN_ID --lr 7.5e-4 "
                "--warmup 250 --const_lr --steps 9001 --batch 32 --ctx 256 "
                "--layers 3 --heads 4 --dk 64 --adff 170 --seed SEED "
                "--device DEVICE --tokens TOKENS --tok_model TOKENIZER "
                "--outdir RUNS --probe_every 100 --sharp_every 1000000 "
                "--sharp_pre_every 1000000 --sharp_full 0 --ckpt_steps , "
                "--optimizer_ckpt_steps OPTIMIZER_CHECKPOINT_STEPS "
                "--confirmation_registry protocol/registry.json "
                "--campaign_spec protocol/training_campaign_spec.json "
                "--data_order ORDER --probe_region global --skip_generation"
            ),
            "qualification": (
                "python3 source/scripts/run_source_restoring_qualification.py "
                "--device DEVICE --output REPORT --require-pass"
            ),
            "source": (
                "python3 source/experiments/confirm/"
                "capture_source_restoring_source.py --checkpoint SOURCE_CHECKPOINT "
                "--registry protocol/registry.json --tokens TOKENS "
                "--data-order ORDER --device DEVICE --output SOURCE_NPZ"
            ),
            "remainder": (
                "python3 source/experiments/confirm/"
                "build_source_restoring_remainder.py --source SOURCE_NPZ "
                "--validated-curvature VALIDATED_CURVATURE_NPZ "
                "--output REMAINDER_NPZ"
            ),
            "certificate": (
                "python3 source/experiments/confirm/"
                "produce_source_restoring_certificate.py --source SOURCE_NPZ "
                "--remainder REMAINDER_NPZ --certificate CERTIFICATE_NPZ "
                "--prediction PREDICTION_JSON"
            ),
            "coverage": (
                "python3 source/experiments/confirm/"
                "cover_source_restoring_successor.py "
                "--source-checkpoint SOURCE_CHECKPOINT "
                "--successor-checkpoint NATIVE_NEXT_CHECKPOINT "
                "--registry protocol/registry.json --tokens TOKENS "
                "--certificate CERTIFICATE_NPZ --prediction PREDICTION_JSON "
                "--device DEVICE --output EDGE_JSON"
            ),
            "aggregate": (
                "python3 source/experiments/analysis/"
                "analyze_source_restoring_confirmation.py --edges EDGE_JSONS "
                "--output AGGREGATE_JSON"
            ),
        },
        "ordering_constraints": [
            "bind every checkpoint to the training campaign and registry",
            "launch trajectory roles serially under the persistent storage cap",
            "stream checkpoint descendants before retention cleanup",
            "freeze source artifact before remainder enclosure",
            "freeze certificate and prediction before opening native successor",
            "link native successor digest to the next edge source",
            "retain complete checkpoints until all descendants close",
        ],
    }
    (output / "protocol" / "launch_template.json").write_bytes(
        canonical(launch)
    )
    provenance = {
        "schema_version": "pldr-source-restoring-stage-provenance-v1",
        "campaign_id": CAMPAIGN_ID,
        "design_schema": DESIGN_SCHEMA,
        "parent_campaign_path": str(parent),
        "parent_manifest_sha256": sha256_path(parent / "MANIFEST.sha256"),
        "parent_registry_path": str(old_registry_path),
        "parent_registry_sha256": sha256_path(old_registry_path),
        "inherited_token_selection_only": True,
        "source_certificate_family": "source-restoring-v1",
    }
    (output / "protocol" / "PROVENANCE.json").write_bytes(
        canonical(provenance)
    )
    (output / "README.md").write_text(
        "# Source-restoring collapse confirmation\n\n"
        "This staged bundle inherits only frozen token selections and PCG64 "
        "orders. Its source-state certificate, trainer-native coverage path, "
        "and chronological decision logic are the source-restoring v1 "
        "interfaces in `source/`.\n",
        encoding="utf-8",
    )
    members = sorted(
        path
        for path in output.rglob("*")
        if (
            path.is_file()
            and path.name != "MANIFEST.sha256"
            and not is_mutable_or_transient(path, output)
        )
    )
    manifest = "".join(
        f"{sha256_path(path)}  {path.relative_to(output).as_posix()}\n"
        for path in members
    )
    (output / "MANIFEST.sha256").write_text(manifest, encoding="ascii")


def verify(parent: Path, output: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix="source-restoring-stage-"
    ) as temporary:
        expected = Path(temporary) / "bundle"
        build(parent, expected)
        expected_files = {
            path.relative_to(expected): path
            for path in expected.rglob("*")
            if path.is_file()
        }
        actual_files = (
            {
                path.relative_to(output): path
                for path in output.rglob("*")
                if path.is_file()
                and not is_mutable_or_transient(path, output)
            }
            if output.is_dir()
            else {}
        )
        if set(expected_files) != set(actual_files):
            raise ValueError("source-restoring staged membership changed")
        for relative in expected_files:
            if (
                expected_files[relative].read_bytes()
                != actual_files[relative].read_bytes()
            ):
                raise ValueError(
                    f"source-restoring staged byte drift: {relative}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        verify(arguments.parent, arguments.output)
        print(
            "source-restoring staging: verified "
            f"{arguments.output.resolve()}"
        )
    else:
        build(arguments.parent, arguments.output)
        print(
            "source-restoring staging: wrote "
            f"{arguments.output.resolve()}"
        )


if __name__ == "__main__":
    main()
