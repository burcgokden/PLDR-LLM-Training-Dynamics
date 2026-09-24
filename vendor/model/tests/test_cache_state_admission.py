"""Coverage and common-entry-point regressions for state-transfer admission."""
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import cache_state_contract as contract
import run_cache_state_transfer as producer


@pytest.mark.parametrize('text', [
    '{"schema":"x","schema":"x"}', '{"nested":{"a":1,"a":2}}',
    '{"x":NaN}', '{"x":Infinity}', '{"x":1e9999}'])
def test_strict_json_rejects_ambiguous_or_nonfinite(tmp_path, text):
    path = tmp_path/'protocol.json'; path.write_text(text)
    with pytest.raises(ValueError):
        contract.read(path)


def test_schema_inventory_has_every_independently_enumerated_role():
    expected = {'caches.npz'}
    for prefix in ['initial-L32', 'initial-L64', 'initial-L128',
                   'g0-L32', 'g0-L64', 'g0-L128',
                   'g1.5-L32', 'g1.5-L64', 'g1.5-L128']:
        expected.update(prefix+suffix for suffix in ['-native.npy', '-recalibrated.npy', '-operators.npz'])
        if not prefix.startswith('initial'):
            expected.add(prefix+'-initial_cache.npy')
    assert len(expected) == 34
    assert contract.artifacts() == expected
    p = {'source_sha256': dict.fromkeys(contract.OLD_SOURCES),
         'jobs': [{'run_id': f'h{n}-s{s}'} for n in [8,24] for s in range(915101,915107)]}
    assert len(contract.required_members(p)) == 431


@pytest.mark.parametrize('bad', [{}, {'unexpected': '0'*64},
    {'protocol.json': '0'*64}, {'./protocol.json': '0'*64}])
def test_incomplete_observation_roles_never_admitted(bad):
    p = {'source_sha256': dict.fromkeys(contract.OLD_SOURCES),
         'jobs': [{'run_id': f'h{n}-s{s}'} for n in [8,24] for s in range(915101,915107)]}
    with pytest.raises(ValueError, match='observation roles'):
        contract.analysis_roles({'checked_sha256': bad}, p)


@pytest.mark.parametrize('name', ['../outside', './protocol.json', '/tmp/file', 'a/../protocol.json'])
def test_path_alias_or_escape_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        contract.local(tmp_path, name)


def test_symlink_escape_rejected(tmp_path):
    study = tmp_path/'study'; study.mkdir()
    outside = tmp_path/'data'; outside.write_text('retained bytes')
    (study/'inputs.npz').symlink_to(outside)
    with pytest.raises(ValueError):
        contract.local(study, 'inputs.npz')


@pytest.mark.parametrize('actual,expected', [(True,1), (False,0), ([True],[1]), ({'x':True},{'x':1})])
def test_bool_is_not_a_scientific_count(actual, expected):
    with pytest.raises(ValueError):
        contract.same(actual,expected,'count')


@pytest.mark.parametrize('route', ['queue','worker'])
@pytest.mark.parametrize('schema', ['unknown','cache-state-transfer-v1','cache-state-transfer-v2'])
def test_both_routes_refuse_invalid_design_before_output_or_construction(tmp_path, monkeypatch, route, schema):
    study = tmp_path/'study'; study.mkdir()
    (study/'protocol.json').write_text(json.dumps({'schema':schema}))
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Construction or process launch before admission')
    monkeypatch.setattr(producer,'TrainingModel',forbidden)
    monkeypatch.setattr(producer.subprocess,'run',forbidden)
    with pytest.raises(ValueError):
        if route == 'worker':
            producer.worker(study,0,'cuda:0')
        else:
            producer.run(study)
    assert calls == []
    assert {p.name for p in study.iterdir()} == {'protocol.json'}
