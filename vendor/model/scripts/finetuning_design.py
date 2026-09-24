"""Canonical primary-study admission contract. Read-only until admission succeeds.

This producer-side specification is deliberately separate from the independent
terminal reconstruction in finetuning_verification_design.py.
"""
from companion_paths import legacy_path
import hashlib
import json
from pathlib import Path

ROOT = Path(legacy_path('/pldr-data/model'))


def require(condition, message):
    if not condition:
        raise ValueError('Fine-tuning design: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_arms():
    result = [dict(name=f'mix{int(r * 1000):04d}', rho=r, group='full', reset=False,
                   role='primary') for r in [0., .125, .25, .5, .75, .875, 1.]]
    result.append(dict(name='general', rho=None, group='full', reset=False, role='primary'))
    for rho, tag in [(0., 'narrative'), (1., 'technical'), (None, 'general')]:
        result.append(dict(name='reset-' + tag, rho=rho, group='full', reset=True,
                           role='optimizer control'))
    for group in ['generator', 'body']:
        for rho, tag in [(0., 'narrative'), (1., 'technical')]:
            result.append(dict(name=group + '-' + tag, rho=rho, group=group,
                               reset=False, role='component control'))
    return result


def check_design(spec):
    """Check scientific fields and manifest identities without writing files."""
    require(spec.get('schema') == 'pldr-finetuning-study-v1', 'schema')
    stage = spec.get('stage')
    require(stage in ('qualification', 'assessment'), 'unknown stage')
    short = stage == 'qualification'
    require(type(spec.get('horizon')) is int and spec['horizon'] == (2 if short else 512), 'horizon')
    require(canonical(spec.get('times')) == canonical([0, 1, 2] if short else [0, 16, 64, 128, 256, 512]), 'time grid')
    require(type(spec.get('batch_size')) is int and spec['batch_size'] == 32, 'batch size')
    require(canonical(spec.get('arms')) == canonical(expected_arms()), 'arm inventory or semantics')
    require(Path(spec['root']).resolve() == ROOT, 'experiment root')
    require(Path(spec['data']).resolve().is_relative_to(ROOT), 'data location')
    seeds = [640101] if short else list(range(640101, 640105))
    grid = [(h, s, f'h{h}-s{s}') for h in [4, 8, 14] for s in seeds]
    cases = spec.get('cases')
    require(isinstance(cases, list), 'case list')
    require([(c['heads'], c['seed'], c['name']) for c in cases] == grid, 'case grid')
    for c in cases:
        require(type(c['heads']) is int and type(c['seed']) is int, 'case types')
        folder = ROOT / 'scheduled-training-feasible-20260908/runs' / f"compact-reference1-h{c['heads']}-s{c['seed']}"
        require(Path(c['state']) == folder / 'final-training-state.pt' and
                Path(c['manifest']) == folder / 'manifest.json', 'incoming paths')
        require(sha(c['manifest']) == c['manifest_sha256'], 'manifest identity')
        m = json.loads(Path(c['manifest']).read_text())
        require(m['status'] == 'complete' and m['completed_step'] == 40960, 'incoming age/completion')
        require(m['arguments']['heads'] == c['heads'] and m['arguments']['seed'] == c['seed'], 'incoming case')
        require(m['arguments']['stream_seed'] == 640001 and m['data_law']['stream_seed'] == 640001,
                'incoming stream')
        require(m['data_law']['consumed_blocks'] == 1310720 and
                m['data_law']['completed_input_tokens'] == 83886080, 'incoming resource')
        require(m['effective_batch_size'] == 32 and m['microbatch_size'] == 32, 'incoming batching')
        require(canonical(c['profile']) == canonical(m['recipe']), 'profile/manifest mismatch')
        require(c['state_sha256'] == m['checkpoint_sha256'], 'checkpoint/manifest mismatch')
        require(c['profile']['name'] == 'reference1' and c['profile']['heads'] == c['heads'], 'recipe family')
    require(spec['runtime']['microbatch'] == 32 and spec['runtime']['dtype'] == 'float32', 'arithmetic')
    require(type(spec['budgets']['max_worker_seconds']) is int and
            0 < spec['budgets']['max_worker_seconds'] <= 10800, 'worker budget')
    if short:
        require('qualification' not in spec, 'qualification cannot depend on itself')
    else:
        require(isinstance(spec.get('qualification'), dict), 'missing qualification')
    return spec


def check_correspondence(spec, short, verification):
    check_design(spec)
    check_design(short)
    require(spec['stage'] == 'assessment' and short['stage'] == 'qualification', 'stage correspondence')
    require(verification.get('schema') == 'pldr-finetuning-verification-v1' and
            verification.get('status') == 'passed' and verification.get('stage') == 'qualification',
            'qualification verification schema/stage/status')
    for key in ['sources', 'runtime', 'inputs', 'data', 'root', 'arms', 'batch_size',
                'law', 'intervention', 'numeric']:
        require(canonical(spec[key]) == canonical(short[key]), 'qualification correspondence: ' + key)
    require(canonical(short['cases']) == canonical([c for c in spec['cases'] if c['seed'] == 640101]),
            'qualification case reduction')


def check_qualification(spec, repo):
    if spec['stage'] == 'qualification':
        return
    bound = spec['qualification']
    path = Path(bound['path'])
    require(sha(path) == bound['sha256'], 'qualification hash')
    result = json.loads(path.read_text())
    protocol = path.parent / 'protocol.json'
    short = json.loads(protocol.read_text())
    check_correspondence(spec, short, result)
    require(result['protocol_sha256'] == sha(protocol), 'qualification protocol hash')
    require(result['verifier_sha256'] == sha(Path(repo) / 'scripts/verify_finetuning.py'), 'current verifier')
    require(result.get('design_verifier_sha256') == sha(Path(repo) / 'scripts/finetuning_verification_design.py'),
            'current independent design verifier')
    require(result['update_counts'] == {'primary': 48, 'optimizer control': 18,
            'component control': 24, 'replay': 6, 'unobserved replay': 6}, 'qualification update inventory')
    # Reconstruct required terminal files independently; a detached status cannot qualify execution.
    from finetuning_verification_design import terminal_inventory
    required = terminal_inventory(path.parent, short)
    require(set(required).issubset(result['verified_files']), 'unbound qualification terminals')
    for name, digest in result['verified_files'].items():
        require(sha(name) == digest, 'qualification terminal changed: ' + name)
    for name, digest in required.items():
        require(result['verified_files'][name] == digest, 'terminal inventory identity')
    for name, digest in short['sources'].items():
        require(sha(path.parent / 'executed-source' / name) == digest, 'qualification source snapshot')
