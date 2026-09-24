"""Deferred routes must never create an execution directory or dispatch work."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from release_index import execute


@pytest.mark.parametrize('route',['P2','P4'])
def test_unqualified_routes_fail_before_output(tmp_path,route):
    index={'routes':[{'id':route,'executable':False,'commands':[['forbidden']]}]}
    output=tmp_path/'run'
    with pytest.raises(ValueError,match='Unqualified'):execute(index,route,output)
    assert not output.exists()


def test_unknown_route_fails_before_output(tmp_path):
    with pytest.raises(ValueError,match='Unknown'):execute({'routes':[]},'absent',tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_selected_empirical_route_requires_external_protocol_before_output(tmp_path):
    index={'routes':[{'id':'raw','executable':True,'commands':[['forbidden']],
            'protocol_identity':{str(tmp_path/'missing.json'):{'sha256':'absent'}}}]}
    with pytest.raises(FileNotFoundError,match='Required external protocol'):
        execute(index,'raw',tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_finite_control_route_has_no_native_data_dependency(tmp_path):
    index={'routes':[{'id':'finite','executable':True,'commands':[['{python}','-c','print(2+2)']],
                      'protocol_identity':{}}]}
    execute(index,'finite',tmp_path/'run')
    assert (tmp_path/'run/command-0.log').read_text().strip()=='4'
