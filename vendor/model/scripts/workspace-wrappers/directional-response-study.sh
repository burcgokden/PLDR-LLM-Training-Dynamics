#!/bin/sh
# Complete pipeline. Every experiment destination must be new.
set -eu
rg_default_repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
rg_repo=${MODEL_RG_REPO:-$rg_default_repo}
rg_data=${MODEL_RG_DATA_ROOT:?Set MODEL_RG_DATA_ROOT to the raw input root}
rg_python=${MODEL_RG_PYTHON:-python}
rg_phase=${1:?Use qualify, develop, freeze, run, analyze, verify, or all}
rg_study=${2:?Supply a fresh study root}
rg_amplitude=${MODEL_RG_DIRECTIONAL_AMPLITUDE:-0.00003}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$rg_repo/../..:$rg_repo/src:$rg_repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
cd "$rg_repo"
case "$rg_phase" in
  develop|freeze|run)
    "$rg_python" scripts/confirmation_status.py require-qualification \
      --producer "$rg_repo/scripts/directional_study.py" \
      --verifier "$rg_repo/scripts/verify_directional.py" \
      --protocol "$rg_study/qualification/protocol.json" \
      --evidence "$rg_study/qualification/verification.json"
    ;;
esac
case "$rg_phase" in
  qualify|develop)
    if [ "$rg_phase" = qualify ]; then rg_kind=qualification; else rg_kind=development; fi
    "$rg_python" scripts/directional_study.py prepare --root "$rg_data" --study "$rg_study/$rg_kind" --kind "$rg_kind" --amplitude "$rg_amplitude"
    "$rg_python" scripts/directional_study.py run --root "$rg_data" --study "$rg_study/$rg_kind"
    "$rg_python" scripts/analyze_directional.py --study "$rg_study/$rg_kind"
    "$rg_python" scripts/verify_directional.py --study "$rg_study/$rg_kind"
    if [ "$rg_kind" = qualification ]; then
      "$rg_python" scripts/analyze_pulse_parity.py --study "$rg_study/$rg_kind"
    fi
    ;;
  freeze)
    "$rg_python" scripts/directional_study.py prepare --root "$rg_data" --study "$rg_study/confirmation" --kind confirmation --amplitude "$rg_amplitude"
    ;;
  run)
    "$rg_python" scripts/directional_study.py run --root "$rg_data" --study "$rg_study/confirmation"
    ;;
  analyze)
    "$rg_python" scripts/analyze_directional.py --study "$rg_study/confirmation"
    ;;
  verify)
    "$rg_python" scripts/verify_directional.py --study "$rg_study/confirmation"
    "$rg_python" scripts/analyze_pulse_parity.py --study "$rg_study/confirmation"
    ;;
  all)
    for rg_stage in qualify develop freeze run analyze verify; do
      sh "$rg_repo/scripts/workspace-wrappers/directional-response-study.sh" "$rg_stage" "$rg_study"
    done
    ;;
  *) echo 'Unknown phase' >&2; exit 2 ;;
esac
