"""Design admission and independent evidence checks, including rehashed corruption."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from scripts import finetuning_design as producer
from scripts import finetuning_verification_design as independent


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def short(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, 'ROOT', tmp_path)
    cases = []
    for h in [4, 8, 14]:
        folder = tmp_path / 'scheduled-training-feasible-20260908/runs' / f'compact-reference1-h{h}-s640101'
        folder.mkdir(parents=True)
        state = folder / 'final-training-state.pt'; state.write_bytes(b'fixture only')
        profile = dict(name='reference1', heads=h, epsilon=1e-5)
        manifest = dict(status='complete', completed_step=40960, arguments=dict(heads=h, seed=640101, stream_seed=640001),
                        data_law=dict(stream_seed=640001, consumed_blocks=1310720, completed_input_tokens=83886080),
                        effective_batch_size=32, microbatch_size=32, recipe=profile, checkpoint_sha256=producer.sha(state))
        put(folder / 'manifest.json', manifest)
        cases.append(dict(name=f'h{h}-s640101', heads=h, seed=640101, profile=profile,
                          manifest=str(folder / 'manifest.json'), manifest_sha256=producer.sha(folder / 'manifest.json'),
                          state=str(state), state_sha256=producer.sha(state)))
    return dict(schema='pldr-finetuning-study-v1', stage='qualification', horizon=2, times=[0, 1, 2],
                batch_size=32, arms=producer.expected_arms(), cases=cases, root=str(tmp_path), data=str(tmp_path / 'data'),
                runtime=dict(microbatch=32, dtype='float32'), budgets=dict(max_worker_seconds=10800))


def test_real_design_has_two_independent_acceptors(short):
    producer.check_design(short)
    independent.design(short)


@pytest.mark.parametrize('edit', [
    lambda s:s.update(stage='unrecognized'), lambda s:s.update(horizon=1, times=[0,1]),
    lambda s:s.update(batch_size=64), lambda s:s.update(batch_size=32.0),
    lambda s:s.update(cases=s['cases'][:-1]), lambda s:s['cases'].append(s['cases'][0]),
    lambda s:s['cases'][0]['profile'].update(epsilon=1e-8),
    lambda s:s['cases'][0].update(state=s['cases'][1]['state']),
    lambda s:s['arms'][0].update(role='replay'), lambda s:s['arms'][0].update(rho=.5),
    lambda s:s['arms'][0].update(reset=0), lambda s:s.update(times=[0,2]),
])
def test_declared_design_mutations_refused_independently(short, edit):
    value=copy.deepcopy(short);edit(value)
    for check in [producer.check_design, independent.design]:
        with pytest.raises((ValueError, KeyError, TypeError)):
            check(value)


def terminal_fixture(study, spec):
    put(study/'protocol.json',spec);signature=producer.sha(study/'protocol.json')
    put(study/'launcher.json',dict(status='complete',protocol_sha256=signature,
        results=[dict(case=c['name'],returncode=0) for c in spec['cases']]))
    for c in spec['cases']:
        folder=study/c['name'];folder.mkdir()
        ids,raw=independent.panel('qualification',0,2)
        np.savez(folder/'initial-observation.npz',indices=ids,raw_indices=raw)
        arms=copy.deepcopy(spec['arms']);general=next(a for a in arms if a['name']=='general')
        arms += [dict(general,name='general-replay',role='replay'),dict(general,name='general-unobserved',role='unobserved replay')]
        records=[]
        for arm in arms:
            path=folder/arm['name'];path.mkdir()
            np.save(path/'blocks.npy',np.arange(64).reshape(2,32))
            np.savez(path/'training.npz',losses=np.ones(2),gradient_norms=np.ones(2),rates=np.ones((2,2)))
            obs=[];times=[2] if arm['role']=='unobserved replay' else [1,2]
            for t in times:
                p=path/f'observation-{t:04d}.npz';np.savez(p,indices=ids,raw_indices=raw)
                obs.append(dict(time=t,file=str(p),sha256=producer.sha(p)))
            record=dict(arm=arm,steps=2,observations=obs,state_digests={str(t):'test-digest' for t in times},
                        checkpoint=None,checkpoint_sha256=None,blocks_sha256=producer.sha(path/'blocks.npy'),
                        training_sha256=producer.sha(path/'training.npz'))
            put(path/'result.json',dict(status='complete',**record));records.append(record)
        put(folder/'results.json',dict(status='complete',case=c,records=records,replay_bitwise=True,
            protocol_sha256=signature,initial_observation_sha256=producer.sha(folder/'initial-observation.npz')))


@pytest.mark.parametrize('mutation',['none','shortened','batch','trace','panel','raw_panel','duplicate_record','arm_meaning'])
def test_terminal_checks_rehashed_mutations(short,tmp_path,mutation):
    study=tmp_path/'terminal';terminal_fixture(study,short)
    folder=study/short['cases'][0]['name'];case=json.loads((folder/'results.json').read_text())
    record=case['records'][0];arm=folder/record['arm']['name']
    if mutation=='shortened':record['steps']=1
    if mutation=='batch':
        np.save(arm/'blocks.npy',np.arange(128).reshape(2,64))
        record['blocks_sha256']=producer.sha(arm/'blocks.npy')
    if mutation=='trace':
        np.savez(arm/'training.npz',losses=np.ones(2),gradient_norms=np.ones(1),rates=np.ones((2,2)))
        record['training_sha256']=producer.sha(arm/'training.npz')
    if mutation in ['panel','raw_panel']:
        p=Path(record['observations'][0]['file']);ids,raw=independent.panel('qualification',1,2)
        np.savez(p,indices=ids[::-1] if mutation=='panel' else ids,raw_indices=raw[::-1] if mutation=='raw_panel' else raw)
        record['observations'][0]['sha256']=producer.sha(p)
    if mutation=='duplicate_record':case['records'].append(copy.deepcopy(record))
    if mutation=='arm_meaning':record['arm']['role']='replay'
    # Local checksums and both copies are repaired: refusal must follow semantics.
    put(arm/'result.json',dict(status='complete',**record));put(folder/'results.json',case)
    if mutation=='none':
        assert len(independent.terminal_inventory(study,short))>200
    else:
        with pytest.raises((ValueError,KeyError,TypeError)):
            independent.terminal_inventory(study,short)
