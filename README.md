# PLDR-LLM Training Dynamics

Scientific code for **Training and Inference Dynamics of PLDR-LLMs: Row-Map
Collapse, Renormalization, and Predictive Reduction**, by Burc Gokden.

Code: https://github.com/burcgokden/PLDR-LLM-Training-Dynamics

Numerical evidence: https://huggingface.co/datasets/fromthesky/pldr-llm-training-dynamics-data

This repository provides selected Lean developments, scientific
implementations, numerical reducers, experiment producers, protocols and tests.
No private manuscript checkout is required. [The chapter guide](docs/CHAPTERS.md)
links the monograph's subjects to the stable scientific namespaces.

## Checks

The scientific suites were checked with Python 3.14.6 and the dependencies in
`requirements-checks.txt`. The evidence reader uses only the Python standard
library and supports Python 3.11 or later. Native acquisition has additional
producer-specific dependencies and requires supplied model/corpus assets.

```sh
python3 -m pip install -r requirements-checks.txt
python3 scripts/verify_scientific_manifest.py
python3 scripts/check_formal_manifest.py
python3 scripts/run_scientific_checks.py
python3 scripts/check_process_boundaries.py
python3 scripts/check_campaign_budget.py
python3 scripts/check_resource_execution.py
lake build
python3 scripts/check_lean.py
```

Lean is pinned to `leanprover/lean4:v4.33.0-rc1` and Mathlib to
`9c0c555bde5a8277cd36dc4dc6dfe2a5a77a2b11`. Lake obtains the public pinned
packages. `lake exe cache get` can supply the dependency cache before building.
`check_lean.py --dependency-cache PATH` can use an existing matching third-party
cache and rebuilds all owned modules from source. It exports the referenced
types and permits only `propext`, `Classical.choice` and `Quot.sound` as
transitive axiom dependencies, with negative controls. The statement manifest
records partial coverage; compilation does not establish all analytical or
native-program assumptions in the monograph.

The test runner discovers the shipped scientific tests. Publication builders,
renderers and their dependent fixtures are excluded explicitly in
`provenance/export-disposition.json`, rather than counted as passes. Mixed
source modules retain separable numerical functions. Only the documented
entry points are supported as command-line workflows. Older source utilities
have narrower support, recorded in `provenance/supported-programs.json`.

The scientific manifest binds prepared file contents independently of Git
history. Its base checkout head is not the identity of uncommitted changes.
Validation output is written beneath `validation/` and `build/`.

## Numerical evidence

Clone the dataset to any directory. All readers and verification code reside
here; the dataset contains only records, indexes, dictionaries and metadata.

```sh
python3 scripts/verify_evidence.py --data-repo /path/to/data-repo
python3 scripts/read_evidence.py --data-repo /path/to/data-repo --list
python3 scripts/verify_evidence.py --data-repo /path/to/data-repo --extract /tmp/pldr-evidence
```

The dataset preserves reported numerical values, complete outcome grids,
controls, source-selection identities and uncertainty summaries. Local path
metadata is normalized and exported records have new hashes. The coverage
index distinguishes numerical summaries, printed table values and raw inputs
that are not included. Integrity checking is not independent replication.

## Scientific execution

```sh
python3 scripts/run_source.py rg scripts/run_synthetic_rg.py --output /tmp/synthetic-rg.json
python3 scripts/run_source.py model scripts/check_consuming_bridge.py --output /tmp/consuming.json
python3 scripts/run_source.py model scripts/run_cache_state_transfer.py --validate-worker
```

`--data-root PATH` on `run_source.py` sets `PLDR_DATA_ROOT`; family inputs then
resolve beneath `FAMILY/research/`. `MODEL_RG_DATA_ROOT`, `PLDR_ROW_DATA_ROOT`
and `PLDR_RG_DATA_ROOT` select family roots directly. `PLDR_REFINEDWEB_ROOT`
selects a read-only corpus. The virtual names `/pldr-data/` and `/pldr-code/`
are acquisition roles resolved by `companion_paths`, not machine mounts.
Large arrays, weights, optimizer states and corpus tokens are external inputs.

[Execution instructions](docs/EXECUTION.md) specify the thirty-cell raw-cache
reconstruction and its hash-bound input graph. [Resource accounting](docs/RESOURCE_EXECUTION.md)
distinguishes bounded process supervision from cumulative campaign admission.
Bounded dispatch checks do not run model forwards or training. Single-pass
consumption of distinct registered RefinedWeb target blocks is the primary
training law. Repeated-corpus controls retain their separate interpretation.

License and attribution notices accompany the retained materials. The root
license does not override third-party terms. Pinned native architecture and
tokenizer assets retain their upstream identities in `vendor/native/`.

Runtime diagnostics and native cache-position semantics are documented in
[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md).
