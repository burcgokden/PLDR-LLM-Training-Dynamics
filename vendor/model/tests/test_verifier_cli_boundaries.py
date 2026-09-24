"""Actual CLI boundaries reject nonfinite claims even with Python optimization."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'


@pytest.mark.parametrize('mode',[[],['-O']])
@pytest.mark.parametrize('token',['NaN','Infinity','1e999'])
@pytest.mark.parametrize('route',['critical','collective','metric'])
def test_nonfinite_cli_has_no_certificate(tmp_path,mode,token,route):
    root=tmp_path/'input';root.mkdir();(root/'protocol.json').write_text('{}')
    corrupt='{"claim":'+token+'}'
    analysis=root/'analysis.json';analysis.write_text(corrupt)
    output=tmp_path/'must-not-exist.json'
    if route=='critical':script='verify_critical_onepass.py';args=['--study',str(root),'--analysis',str(analysis)]
    elif route=='collective':script='verify_critical_collectives.py';args=['--study',str(root),'--analysis',str(analysis)]
    elif route=='metric':
        script='verify_metric_studies.py';(root/'metric-budget.json').write_text(corrupt)
        args=['--metric-root',str(root),'--matched-analysis',str(analysis),'--matched-refinement',str(analysis)]
    else:script='verify_replica_release.py';args=['--inventory',str(analysis)]
    result=subprocess.run([sys.executable,*mode,str(SCRIPTS/script),*args,'--output',str(output)],capture_output=True,text=True)
    assert result.returncode!=0 and 'Nonfinite numerical scalar' in result.stderr
    assert not output.exists()
