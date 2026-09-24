"""Small executable-contract regression independent of large native assets."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
from model_rg.qualification import QualificationError, stage_of, source_names

@pytest.mark.parametrize('record', [{}, {'status':'passed'}, {'schema':'onepass-refresh-protocol-v1','status':'frozen','qualification':1},
    {'schema':'onepass-directional-v1','status':'frozen','kind':'qualification','qualification':True}])
def test_ambiguous_or_missing_schema_is_rejected(record):
    with pytest.raises(QualificationError):stage_of(record)

@pytest.mark.parametrize('stage,required',[('P1','scripts/verify_refresh_summary.py'),('P3','scripts/analyze_pulse_parity.py')])
def test_transitive_and_terminal_dependencies_are_resolved(stage,required):
    repo=Path(__file__).resolve().parents[1];names=source_names(repo,stage)
    assert {'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/qualification.py',required}<=set(names)

def test_optimized_python_cannot_disable_schema_guard():
    code='from model_rg.qualification import stage_of; stage_of({"status":"passed"})'
    result=subprocess.run([sys.executable,'-O','-c',code],capture_output=True,text=True)
    assert result.returncode!=0 and 'QualificationError' in result.stderr
