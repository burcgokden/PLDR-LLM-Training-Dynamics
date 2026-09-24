#!/usr/bin/env python3
"""Stage a self-checking row-map confirmation bundle in experiment-data."""

from __future__ import annotations
from companion_paths import legacy_path, data_root, resolve_row_arguments

import argparse
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from gate_shape_evidence import canonical_json  # noqa: E402
from row_map_live import create_registry, inventory  # noqa: E402
from run_row_map_confirmation import execution_plan  # noqa: E402


DEFAULT_OUTPUT = Path(
    legacy_path('/pldr-data/row/rev36/row-map-collapse-confirmation'))
DEFAULT_RUN_ROOT = Path(
    data_root('row') / 'inputs/runs')
DEFAULT_TOKENS = ROOT / "experiments" / "data" / "refinedweb_tokens.npy"
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"
PROTOCOL = ROOT / "experiments" / "protocols" / "row_map_collapse_confirmation"

SOURCE_FILES = (
    "experiments/confirm/row_map_confirmation_specs.py",
    "experiments/confirm/row_map_dynamics.py",
    "experiments/confirm/row_map_live.py",
    "experiments/confirm/row_map_plga_live.py",
    "experiments/confirm/run_row_map_confirmation.py",
    "experiments/confirm/gate_shape.py",
    "experiments/confirm/gate_shape_evidence.py",
    "experiments/analysis/analyze_row_map_confirmation.py",
    "experiments/pldr_model_v510.py",
    "experiments/power_law_attention_layer_v510.py",
    "scripts/execute_row_map_plan.py",
    "scripts/gen_row_map_protocols.py",
    "scripts/stage_row_map_confirmation.py",
    "requirements.txt",
)


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def expected_files(output, run_root, tokens, tokenizer):
    files = {}
    for source_name in SOURCE_FILES:
        source = ROOT / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        files[Path("code") / source_name] = source.read_bytes()
    for source in sorted(PROTOCOL.rglob("*")):
        if source.is_file():
            files[Path("protocols") / source.relative_to(PROTOCOL)] = (
                source.read_bytes())
    with tempfile.TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        inventory_path = temporary / "inventory.json"
        registry_path = temporary / "registry.json"
        inventory(SimpleNamespace(run_root=run_root, output=inventory_path))
        create_registry(SimpleNamespace(
            tokens=tokens,
            tokenizer=tokenizer,
            construction_contexts=8,
            validation_contexts=16,
            reserved_start=-1,
            output=registry_path,
        ))
        files[Path("preflight/inventory.json")] = inventory_path.read_bytes()
        files[Path("preflight/registry.json")] = registry_path.read_bytes()
    plan = execution_plan(SimpleNamespace(
        run_root=run_root,
        tokens=tokens,
        tokenizer=tokenizer,
        output_root=output / "results",
        devices="cuda:0,cuda:1",
    ))
    files[Path("execution_plan.json")] = canonical_json(plan) + b"\n"
    selection = {
        "schema_version": "pldr-row-map-intervention-selection-request-v1",
        "selection_status": "GENERATE_FROM_SEALED_CONSTRUCTION_GEOMETRY",
        "base_plan_sha256": plan["plan_sha256"],
        "geometry_analysis": str(
            output / "results/R/geometry-analysis.json"),
        "generator_command": "select-interventions",
    }
    files[Path("intervention_selection.template.json")] = (
        canonical_json(selection) + b"\n")
    readme = f"""# Row-map collapse confirmation execution bundle

This bundle binds the retained 20.6M-parameter checkpoints, token archive,
tokenizer, generated protocol, executable source snapshots, and a two-device
job graph. The preflight inventory records that geometry is available at six
nodes per run and complete AdamW moments are retained at update 24,000.

Execute the complete base plan from repository root `{ROOT}`:

```
python3 scripts/execute_row_map_plan.py \\
  --plan {output / 'execution_plan.json'} \\
  --stages setup,Q,G,H,D,A \\
  --record-root {output / 'records'} \\
  --summary {output / 'summaries/base.json'}
```

Q is a hard gate inside that command. Every later stage stops if the
two-device qualification analysis fails. Each scientific analysis runs after
its owning GPU stage, so construction geometry is sealed at
`{output / 'results/R/geometry-analysis.json'}`.

Generate the construction-only layer selection and the budget-bound
intervention plan:

```
python3 experiments/confirm/run_row_map_confirmation.py select-interventions \\
  --base-plan {output / 'execution_plan.json'} \\
  --geometry-analysis {output / 'results/R/geometry-analysis.json'} \\
  --output {output / 'intervention_selection.json'}

python3 experiments/confirm/run_row_map_confirmation.py interventions \\
  --selection {output / 'intervention_selection.json'} \\
  --base-plan {output / 'execution_plan.json'} \\
  --prior-summary {output / 'summaries/base.json'} \\
  --geometry-analysis {output / 'results/R/geometry-analysis.json'} \\
  --run-root {run_root} \\
  --tokens {tokens} \\
  --registry {output / 'results/registry.json'} \\
  --output-root {output / 'results/I'} \\
  --analysis-root {output / 'results/R'} \\
  --output {output / 'intervention_plan.json'}
```

Run the paired arms and final reconstruction:

```
python3 scripts/execute_row_map_plan.py \\
  --plan {output / 'intervention_plan.json'} \\
  --stages I,R \\
  --record-root {output / 'records'} \\
  --summary {output / 'summaries/interventions.json'}
```

The intervention plan subtracts base usage from the 2.4 aggregate GPU-hour
and two-hour wall-clock hard stops. Fixed scientific constants and the
construction-selected layer set cannot be enlarged after validation opens.
"""
    files[Path("README.md")] = readme.encode("utf-8")
    checksums = "".join(
        f"{_digest(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda value: value.as_posix())
    )
    files[Path("CHECKSUMS.sha256")] = checksums.encode("ascii")
    return files


def stage(arguments):
    output = Path(arguments.output).resolve()
    run_root = Path(arguments.run_root).resolve()
    tokens = Path(arguments.tokens).resolve()
    tokenizer = Path(arguments.tokenizer).resolve()
    expected = expected_files(output, run_root, tokens, tokenizer)
    actual = {
        path.relative_to(output)
        for path in output.rglob("*") if path.is_file()
    } if output.is_dir() else set()
    unknown = sorted(actual - set(expected), key=lambda value: value.as_posix())
    stale = [
        path for path, payload in expected.items()
        if not (output / path).is_file()
        or (output / path).read_bytes() != payload
    ]
    if arguments.check:
        if stale or unknown:
            raise SystemExit(
                "staged row-map bundle is stale: "
                + ", ".join(
                    [path.as_posix() for path in stale]
                    + [f"unknown:{path.as_posix()}" for path in unknown]))
        print(f"row-map stage: verified ({len(expected)} files)")
        return
    if unknown:
        raise SystemExit(
            "refusing to overwrite staged directory with unknown files: "
            + ", ".join(path.as_posix() for path in unknown))
    for relative, payload in expected.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    print(f"row-map stage: wrote {len(expected)} files to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--tokens", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Report inputs and outputs without acquisition")
    arguments = parser.parse_args()
    if not resolve_row_arguments(arguments, parser, defaults={'run_root': DEFAULT_RUN_ROOT, 'tokens': DEFAULT_TOKENS, 'tokenizer': DEFAULT_TOKENIZER},
            identities={'tokens': globals().get('DATASET_SHA256'), 'tokenizer': globals().get('TOKENIZER_SHA256')}):
        return
    stage(arguments)


if __name__ == "__main__":
    main()
