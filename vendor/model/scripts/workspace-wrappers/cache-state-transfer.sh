#!/bin/sh
set -eu
code=$(CDPATH= cd -- "$(dirname -- "$0")/../../../.." && pwd)
study=${1:?Supply a fresh study beneath MODEL_RG_DATA_ROOT}
: "${MODEL_RG_DATA_ROOT:?Set the prepared raw-input and experiment root}"
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
python3 "$code/scripts/run_source.py" model scripts/run_cache_state_transfer.py prepare --study "$study" "$@"
python3 "$code/scripts/run_source.py" model scripts/run_cache_state_transfer.py run --study "$study"
python3 "$code/scripts/run_source.py" model scripts/analyze_cache_state_transfer.py --study "$study" --output "$study/analysis.json"
python3 "$code/scripts/run_source.py" model scripts/verify_cache_state_transfer.py --study "$study" --analysis "$study/analysis.json" --output "$study/verification.json"
