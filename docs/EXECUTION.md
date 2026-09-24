# Scientific execution contract

Run `scripts/run_source.py FAMILY scripts/PROGRAM.py` at the repository root.
The launcher preserves family imports, caller data settings and an optional
`PLDR_READ_GUARD` in child processes. The supported-interface manifest states
whether a route is checked for argument dispatch, numerical execution or raw
reconstruction. A dispatch check does not establish a complete acquisition.

Four row stages support `--help` and `--dry-run` without a corpus. Explicit
`--tokens`, `--tokenizer` and `--run-root` override `PLDR_ROW_TOKENS`,
`PLDR_ROW_TOKENIZER` and `PLDR_ROW_RUN_ROOT`. Execution requires the frozen
hashes, schema and block ranges. A new tokenization is a different asset.

## Thirty-cell cache reconstruction

Protocol `rg-cache-state-transfer-v1`, admission `cache-state-contract-v2`,
and study `cache-state-confirmation-20260916` define the retained route.
The compact dataset provides numerical results, not the large raw arrays.
Supply the completed raw study and every external file bound by its protocol.
The location manifest has schema `pldr-acquisition-locations-v1`, a canonical
absolute `logical_root` taken from the immutable acquisition, a `study`
record with `identity` and current `path`, and a `files` array. Each file
contains its exact acquisition `identity`, actual regular-file `path` and
`sha256`. These locations are caller configuration and are not distributed.
The logical root can name any admitted acquisition namespace. No private
machine root is built into the interface.

```sh
sh vendor/model/scripts/workspace-wrappers/analyze-cache-state-transfer.sh RAW-STUDY FRESH-OUTPUT INPUT-LOCATIONS.json
python3 scripts/check_cache_relocation.py --source-study RAW-STUDY --workspace FRESH-WORKSPACE
```

The wrapper writes `analysis.json` and `verification.json`. The second command
copies the complete raw graph, blocks original-workspace reads, checks source
and relocated hashes, requires exact equality of every one of the thirty
scientific cell dictionaries and runs the independent verifier. Allow about
3 GB of temporary input space and up to fifteen minutes per bounded CPU
subprocess. No native forward or training update is performed. Missing aliases,
changed bytes, duplicate destinations and symlinks are rejected. Metadata
normalization of public summaries does not make them raw reconstruction inputs.

The full cache outcome grid retains eighteen recalibrated aggregate passes,
six zero-control transfer passes and six positive-control initial-cache
transfer failures, along with context-level exceptions. These are finite
paired results under the declared panel and initialization law.

Native acquisition requires separately admitted states, token arrays and
protocols. `cache-state-transfer.sh FRESH-STUDY` uses `MODEL_RG_DATA_ROOT`.
Protocol admission and resource caps remain mandatory. Training uses distinct
registered target blocks; calibration and assessment roles remain separate.
The shipped publication-independent tests and bounded resource regressions
perform no new scientific acquisition.

`python3 scripts/check_scientific_standalone.py --data-repo DATA-REPO`
checks manifested code and data copies with their original locations blocked.
Optional stage-predecessor tests need the separately declared raw observations;
missing raw fixtures are explicit skips. Publication-only fixtures are excluded
from this scientific suite and are not counted as passes.
