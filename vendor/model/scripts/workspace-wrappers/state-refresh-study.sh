#!/bin/sh
# Complete, fresh-output path from native qualification to verified results.
set -eu
rg_default_repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
rg_repo=${MODEL_RG_REPO:-$rg_default_repo}
rg_data=${MODEL_RG_DATA_ROOT:?Set MODEL_RG_DATA_ROOT to the raw input root}
rg_python=${MODEL_RG_PYTHON:-python}
rg_phase=${1:?Use qualify, freeze, run, analyze, verify, resource, or all}
rg_study=${2:?Supply a fresh study root with qualification/ and confirmation/ children}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$rg_repo/../..:$rg_repo/src:$rg_repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
cd "$rg_repo"
case "$rg_phase" in
  qualify)
    "$rg_python" scripts/refresh_study.py prepare --root "$rg_data" --study "$rg_study/qualification" --qualification
    "$rg_python" scripts/refresh_study.py run --root "$rg_data" --study "$rg_study/qualification"
    "$rg_python" scripts/analyze_refresh.py --study "$rg_study/qualification"
    "$rg_python" scripts/verify_refresh.py --study "$rg_study/qualification"
    "$rg_python" scripts/verify_refresh_summary.py --study "$rg_study/qualification"
    ;;
  freeze)
    "$rg_python" scripts/refresh_study.py prepare --root "$rg_data" --study "$rg_study/confirmation"
    ;;
  run)
    "$rg_python" scripts/refresh_study.py run --root "$rg_data" --study "$rg_study/confirmation"
    ;;
  analyze)
    "$rg_python" scripts/analyze_refresh.py --study "$rg_study/confirmation"
    ;;
  verify)
    "$rg_python" scripts/verify_refresh.py --study "$rg_study/confirmation"
    "$rg_python" scripts/verify_refresh_summary.py --study "$rg_study/confirmation"
    ;;
  resource)
    "$rg_python" scripts/resource_coupling_study.py run --study "$rg_study/resource-coupling" --native "$rg_study/confirmation"
    "$rg_python" scripts/resource_coupling_study.py verify --study "$rg_study/resource-coupling"
    ;;
  all)
    for rg_stage in qualify freeze run analyze verify resource; do
      sh "$rg_repo/scripts/workspace-wrappers/state-refresh-study.sh" "$rg_stage" "$rg_study"
    done
    ;;
  *)
    echo 'Unknown phase' >&2
    exit 2
    ;;
esac
