#!/bin/sh
set -eu
code=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
study=${1:?Supply the completed study path}
records=${2:?Supply a fresh analysis-output directory}
if [ "$#" -ge 3 ]; then export PLDR_INPUT_MANIFEST="$3"; fi
mkdir "$records"
export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 "$code/scripts/run_source.py" model scripts/analyze_cache_state_transfer.py --study "$study" --output "$records/analysis.json"
python3 "$code/scripts/run_source.py" model scripts/verify_cache_state_transfer.py --study "$study" --analysis "$records/analysis.json" --output "$records/verification.json"
