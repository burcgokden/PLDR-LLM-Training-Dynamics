#!/bin/sh
set -eu
rg_default_repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
rg_repo=${MODEL_RG_REPO:-$rg_default_repo}
cd "$rg_repo"
export PYTHONPATH="$PWD/../..:$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4
python scripts/verify_training.py --data-root "${MODEL_RG_DATA_ROOT:?Set MODEL_RG_DATA_ROOT to the raw input root}" "$@"
