"""Independent reconstruction of the primary design and terminal file inventory.

Uses no producer, model adapter, observer, or producer-side design contract.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def equal(a, b, label):
    if json.dumps(a, sort_keys=True, allow_nan=False) != json.dumps(b, sort_keys=True, allow_nan=False):
        raise ValueError('Independent design mismatch: ' + label)


def design(spec):
    if spec.get('schema') != 'pldr-finetuning-study-v1' or spec.get('stage') not in ('qualification', 'assessment'):
        raise ValueError('Independent design: invalid schema/stage')
    short = spec['stage'] == 'qualification'
    equal(spec['horizon'], 2 if short else 512, 'horizon')
    equal(spec['times'], [0, 1, 2] if short else [0, 16, 64, 128, 256, 512], 'observation times')
    equal(spec['batch_size'], 32, 'batch size')
    arms = []
    for name, rho in [('mix0000', 0.), ('mix0125', .125), ('mix0250', .25), ('mix0500', .5),
                      ('mix0750', .75), ('mix0875', .875), ('mix1000', 1.), ('general', None)]:
        arms.append(dict(name=name, rho=rho, group='full', reset=False, role='primary'))
    for name, rho in [('narrative', 0.), ('technical', 1.), ('general', None)]:
        arms.append(dict(name='reset-' + name, rho=rho, group='full', reset=True, role='optimizer control'))
    for group in ('generator', 'body'):
        for name, rho in [('narrative', 0.), ('technical', 1.)]:
            arms.append(dict(name=group + '-' + name, rho=rho, group=group, reset=False, role='component control'))
    equal(spec['arms'], arms, 'arm meanings')
    expected = [[h, s, f'h{h}-s{s}'] for h in (4, 8, 14)
                for s in ([640101] if short else [640101, 640102, 640103, 640104])]
    equal([[c['heads'], c['seed'], c['name']] for c in spec['cases']], expected, 'case identities')
    for c in spec['cases']:
        folder = Path(spec['root']) / 'scheduled-training-feasible-20260908/runs' / f"compact-reference1-h{c['heads']}-s{c['seed']}"
        equal(c['manifest'], str(folder / 'manifest.json'), 'incoming manifest path')
        equal(c['state'], str(folder / 'final-training-state.pt'), 'incoming state path')
        m = json.loads(Path(c['manifest']).read_text())
        equal(sha(c['manifest']), c['manifest_sha256'], 'incoming manifest hash')
        equal([m['status'], m['completed_step'], m['arguments']['heads'], m['arguments']['seed'],
               m['arguments']['stream_seed'], m['data_law']['consumed_blocks'],
               m['data_law']['completed_input_tokens'], m['effective_batch_size']],
              ['complete', 40960, c['heads'], c['seed'], 640001, 1310720, 83886080, 32], 'incoming design')
        equal(c['profile'], m['recipe'], 'incoming profile')
        equal(c['state_sha256'], m['checkpoint_sha256'], 'incoming checkpoint')
    return arms


def panel(stage, timepoint, scientific_horizon):
    offsets = [0, 80, 160, 240, 320, 400]
    if stage == 'qualification':
        ids = [j + k for j in offsets for k in range(3)]
    elif timepoint in (0, scientific_horizon):
        ids = list(range(480))
    else:
        ids = [j + k for j in offsets for k in range(16)]
    return np.asarray(ids, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


def observation_panel(archive, spec, timepoint):
    ids, raw = panel(spec['stage'], timepoint, spec['horizon'])
    if not np.issubdtype(archive['indices'].dtype, np.integer) or not np.array_equal(archive['indices'], ids):
        raise ValueError('Independent design: reordered, missing or substituted observation panel')
    if not np.issubdtype(archive['raw_indices'].dtype, np.integer) or not np.array_equal(archive['raw_indices'], raw):
        raise ValueError('Independent design: raw panel identity')


def case_records(spec, case, result):
    equal(result['status'], 'complete', 'case completion')
    equal(result['case'], case, 'case record identity')
    expected = [a for a in spec['arms'] if a['role'] == 'primary' or case['seed'] == 640101]
    general = next(a for a in expected if a['name'] == 'general')
    expected.append(dict(general, name='general-replay', role='replay'))
    if spec['stage'] == 'qualification':
        expected.append(dict(general, name='general-unobserved', role='unobserved replay'))
    equal([r['arm'] for r in result['records']], expected, 'exact record inventory')
    for record in result['records']:
        role = record['arm']['role']
        horizon = min(16, spec['horizon']) if role == 'replay' else spec['horizon']
        equal(record['steps'], horizon, 'registered path length')
        times = [horizon] if role == 'unobserved replay' else [t for t in spec['times'] if 0 < t <= horizon]
        equal([r['time'] for r in record['observations']], times, 'ordered observation grid')
        equal(sorted(record['state_digests']), sorted(map(str, times)), 'state digest time grid')
    return result['records']


def trace_shapes(blocks, trace, horizon):
    if blocks.shape != (horizon, 32) or not np.issubdtype(blocks.dtype, np.integer):
        raise ValueError('Independent design: actual batch shape/type')
    for key, shape in [('losses', (horizon,)), ('gradient_norms', (horizon,)), ('rates', (horizon, 2))]:
        if trace[key].shape != shape or not np.isfinite(trace[key]).all():
            raise ValueError('Independent design: trace shape or nonfinite ' + key)


def terminal_inventory(study, spec):
    """Check design, roles, lengths, panels and hashes, without native execution."""
    study = Path(study).resolve()
    design(spec)
    checked = {}
    def check(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            equal(digest, expected, 'terminal hash: ' + str(path))
        checked[str(path)] = digest
    check(study / 'protocol.json')
    launcher_path = study / 'launcher.json'
    launcher = json.loads(launcher_path.read_text())
    equal(launcher['status'], 'complete', 'launcher completion')
    equal(launcher['protocol_sha256'], checked[str(study / 'protocol.json')], 'launcher protocol')
    equal(sorted((r['case'], r['returncode']) for r in launcher['results']),
          sorted((c['name'], 0) for c in spec['cases']), 'launcher cases')
    check(launcher_path)
    for c in spec['cases']:
        folder = study / c['name']
        r = json.loads((folder / 'results.json').read_text())
        check(folder / 'results.json')
        equal(r['protocol_sha256'], checked[str(study / 'protocol.json')], 'case protocol')
        equal(r['replay_bitwise'], True, 'replay completion')
        records = case_records(spec, c, r)
        check(folder / 'initial-observation.npz', r['initial_observation_sha256'])
        with np.load(folder / 'initial-observation.npz') as f:
            observation_panel(f, spec, 0)
        for record in records:
            arm_folder = folder / record['arm']['name']
            check(arm_folder / 'result.json')
            equal(json.loads((arm_folder / 'result.json').read_text()), dict(status='complete', **record), 'arm record copy')
            check(arm_folder / 'blocks.npy', record['blocks_sha256'])
            check(arm_folder / 'training.npz', record['training_sha256'])
            with np.load(arm_folder / 'training.npz') as trace:
                trace_shapes(np.load(arm_folder / 'blocks.npy'), trace, record['steps'])
            for obs in record['observations']:
                equal(obs['file'], str(arm_folder / f"observation-{obs['time']:04d}.npz"), 'observation path')
                check(obs['file'], obs['sha256'])
                with np.load(obs['file']) as f:
                    observation_panel(f, spec, obs['time'])
            need_state = (spec['stage'] == 'assessment' and record['arm']['role'] == 'primary'
                          and record['arm']['rho'] in (None, 0., .5, 1.))
            if need_state:
                equal(record['checkpoint'], str(arm_folder / 'final-state.pt'), 'checkpoint path')
                check(record['checkpoint'], record['checkpoint_sha256'])
            else:
                equal([record['checkpoint'], record['checkpoint_sha256']], [None, None], 'checkpoint role')
    return checked
