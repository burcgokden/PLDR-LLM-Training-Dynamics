"""Nonfinite claims and reconstructed arithmetic must never earn a certificate."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from numerical_validation import discrepancy, array_discrepancy, load_json_strict
from numerical_claims import finite_greater
import verify_optimizer_compact
import verify_critical_onepass
import verify_critical_collectives
import verify_metric_studies


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf'),True,'1'])
@pytest.mark.parametrize('side',['actual','claimed'])
def test_scalar_sides(bad,side):
    with pytest.raises(ValueError):
        discrepancy(bad if side=='actual' else 1.,bad if side=='claimed' else 1.,'fixture')


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
@pytest.mark.parametrize('side',['actual','claimed'])
def test_array_sides(bad,side):
    with pytest.raises(ValueError):
        array_discrepancy([bad] if side=='actual' else [1.],[bad] if side=='claimed' else [1.],'fixture')


@pytest.mark.parametrize('text',['{"x": NaN}','{"x": Infinity}','{"x": -Infinity}','{"x": 1e999}'])
def test_strict_json(text):
    with pytest.raises(ValueError):load_json_strict(text)


def test_undefined_and_exact_zero():
    assert load_json_strict('{"variance":0,"relative":null}')==dict(variance=0,relative=None)
    assert discrepancy(0.,0.,'zero')==0.
    assert not finite_greater(0.,1.,'finite')
    assert discrepancy(1.,1.+1e-13,'tolerance')<1e-12
    for actual,bound in [(float('nan'),1.),(1.,float('nan'))]:
        with pytest.raises(ValueError):finite_greater(actual,bound,'bound')
    with pytest.raises(ValueError):discrepancy(1e308,-1e308,'overflow')
    with pytest.raises(ValueError):array_discrepancy([1e308],[-1e308],'overflow')


@pytest.fixture
def optimizer_fixture(tmp_path):
    source=Path(__file__).resolve().parents[1]/'tests/fixtures'
    target=tmp_path/'generated';target.mkdir()
    for name in ['optimizer-transport-protocol.json','optimizer-transport-analysis.json']:
        (target/name).write_bytes((source/name).read_bytes())
    return tmp_path


def test_optimizer_positive(optimizer_fixture):
    assert verify_optimizer_compact.verify(optimizer_fixture)['complete_cells']==16


@pytest.mark.parametrize('location',['claim','raw'])
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-float('inf')])
def test_optimizer_full_route(optimizer_fixture,location,bad):
    path=optimizer_fixture/'generated/optimizer-transport-analysis.json';r=json.loads(path.read_text())
    if location=='claim':r['rows'][0]['incoming_odd_rms']=bad
    else:
        key=next(iter(r['compact']['paths']));r['compact']['paths'][key][0][0]=bad
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):verify_optimizer_compact.verify(optimizer_fixture)


@pytest.mark.parametrize('mode',[[],['-O']])
def test_optimizer_optimized_cli(optimizer_fixture,mode):
    path=optimizer_fixture/'generated/optimizer-transport-analysis.json';r=json.loads(path.read_text())
    r['rows'][0]['incoming_odd_rms']=float('nan');path.write_text(json.dumps(r))
    command='import sys;sys.path.insert(0,sys.argv[1]);from verify_optimizer_compact import verify;verify(sys.argv[2])'
    result=subprocess.run([sys.executable,*mode,'-c',command,str(Path(verify_optimizer_compact.__file__).parent),str(optimizer_fixture)],capture_output=True,text=True)
    assert result.returncode!=0 and 'Nonfinite numerical scalar' in result.stderr



