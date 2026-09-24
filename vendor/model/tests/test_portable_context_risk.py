"""The portable risk entry point must respect the complete-package boundary."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pytest
from scripts.reproduce_context_risk_observations import reproduce


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_metadata_stub_is_not_a_complete_package(tmp_path):
    (tmp_path/'INDEX.json').write_text('{}')
    (tmp_path/'PACKAGE_KIND.json').write_text(json.dumps(dict(kind='external_observed_data_metadata',
        external_index_sha256=sha(tmp_path/'INDEX.json'),complete_local_root='/unavailable')))
    with pytest.raises(ValueError,match='complete observation package'):reproduce(tmp_path,tmp_path/'never.json')
    assert not (tmp_path/'never.json').exists()


@pytest.mark.parametrize('expected',['../outside.json','/tmp/outside-context-risk.json'])
def test_unindexed_context_record_cannot_escape_package(tmp_path,expected):
    names=['README.md','reference.npz','code/reproduce_categorical_observations.py',
           'refinement-expected.json','context-expected.json','observation.npz']
    files={}
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
        files[name]=sha(path)
    (tmp_path/'INDEX.json').write_text(json.dumps(dict(files=files,observations=[dict(path='observation.npz')],
        context_risk=[dict(family='context',expected=expected,contexts=64)])))
    with pytest.raises(ValueError,match='contained package index'):reproduce(tmp_path,tmp_path/'never.json')
    assert not (tmp_path/'never.json').exists()
