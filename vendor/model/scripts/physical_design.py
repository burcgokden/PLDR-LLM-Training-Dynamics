"""Physical experiment admission. Validation performs no writes or native work."""
from companion_paths import configured_path
from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from model_rg.provenance import sha256

ROOT = Path(configured_path('data:model'))
ASSETS = ROOT / 'assets/PLDR-LLM-v51-SOC-110M-1'
SOURCE_NAMES = ['scripts/physical_study.py', 'scripts/physical_design.py',
    'scripts/run_physical_queue.py', 'scripts/qualify_physical_native.py',
    'src/model_rg/physical_native.py', 'src/model_rg/training.py',
    'src/model_rg/native.py', 'src/model_rg/lattice.py', 'src/model_rg/provenance.py']
OPTIMIZER = dict(name='AdamW', betas=[.9, .95], epsilon=1e-8,
    weight_decay=.01, gradient_clip_norm=1., memory='zero at physical origin')
SCHEDULE = dict(name='linear-cosine', warmup=128, floor=.2,
    clock='zero-based update before optimizer.step')


def require(condition, message):
    if not condition:
        raise ValueError('Physical design: ' + message)


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)


def equal(x, y, message):
    require(canonical(x) == canonical(y), message)


def sources():
    return {n: sha256(REPO / n) for n in SOURCE_NAMES}


def assets():
    # Include every local Python dependency, configuration and tokenizer asset.
    return {str(p): sha256(p) for p in sorted(ASSETS.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.json', '.model')}


def runtime():
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        cuda=torch.version.cuda, dtype='float32', tf32=False, threads=4,
        dependencies={n: importlib.metadata.version(n)
                      for n in ['transformers', 'sentencepiece', 'safetensors']})


def stage_design(stage):
    require(stage in ('qualification', 'development', 'confirmation'), 'unknown stage')
    if stage == 'qualification':
        steps, checkpoints = 12, [0, 12]
        grid = [(h, 640101, a) for h in [4, 8]
                for a in ['full', 'frozen-generator', 'shuffled']]
    elif stage == 'development':
        steps, checkpoints = 2048, [0, 512, 2048]
        grid = [(h, 640101, 'full') for h in [4, 8]]
    else:
        steps, checkpoints = 16384, [0, 512, 2048, 16384]
        grid = [(h, s, 'full') for h in [4, 8] for s in [640101, 640102, 640103]]
        grid += [(4, 640101, a) for a in ['frozen-generator', 'shuffled']]
    return steps, checkpoints, grid


def schedule_factor(step, steps):
    require(type(step) is int and 0 <= step <= steps, 'schedule clock')
    return ((step + 1) / 128 if step < 128 else
            .2 + .8 * (1 + math.cos(math.pi * (step - 128) / max(1, steps - 128))) / 2)


def check_design(spec):
    require(spec.get('schema') == 'physical-native-study-v2', 'schema')
    steps, checkpoints, grid = stage_design(spec.get('stage'))
    require(type(spec.get('steps')) is int and spec['steps'] == steps, 'horizon')
    equal(spec.get('checkpoints'), checkpoints, 'checkpoint grid')
    require(type(spec.get('batch_size')) is int and spec['batch_size'] == 32, 'batch size')
    equal(spec.get('learning_rate'), .0003, 'learning rate')
    equal(spec.get('optimizer'), OPTIMIZER, 'optimizer')
    equal(spec.get('schedule'), SCHEDULE, 'schedule')
    require(spec.get('status') == 'frozen', 'protocol status')
    cases = spec.get('cases')
    require(isinstance(cases, list), 'case list')
    actual = [(c['heads'], c['seed'], c['arm']) for c in cases]
    equal(actual, grid, 'case inventory')
    for c in cases:
        require(set(c) == {'name', 'heads', 'seed', 'arm', 'state', 'state_sha256'}, 'case keys')
        require(type(c['heads']) is int and type(c['seed']) is int, 'case types')
        equal(c['name'], f"h{c['heads']}-s{c['seed']}-{c['arm']}", 'case name')
        expected = ROOT / f"scheduled-training-feasible-20260908/runs/compact-reference1-h{c['heads']}-s{c['seed']}/final-training-state.pt"
        require(Path(c['state']) == expected, 'incoming state path')
    equal(spec.get('runtime'), runtime(), 'runtime')
    equal(spec.get('sources'), sources(), 'producer sources')
    equal(spec.get('native_assets'), assets(), 'native/tokenizer assets')
    require(set(spec.get('qualifications', {})) == {'4', '8'}, 'qualification inventory')
    return spec


def check_corpus(path, expected_hash, development=False):
    path = Path(path)
    require(path.resolve().is_relative_to(ROOT), 'corpus location')
    require(sha256(path / 'manifest.json') == expected_hash, 'corpus manifest')
    ds = json.loads((path / 'manifest.json').read_text())
    require(ds['status'] == 'complete' and ds['schema'] == 'physical-corpus-v1', 'corpus status/schema')
    expected = [(q, L, r) for q in [2, 3] for L in [4, 8, 16]
                for r in ([.85, 1., 1.15] if development else [.85, .925, 1., 1.075, 1.15])]
    equal([(c['q'], c['L'], c['temperature_ratio']) for c in ds['cells']], expected, 'physical grid')
    for i, c in enumerate(ds['cells']):
        require(type(c['id']) is int and c['id'] == i, 'cell identity')
        equal(c['shape'], [8 if development else 16, 2048, c['L'], c['L']], 'corpus shape')
        require(c['chains'] == c['shape'][0] and c['samples_per_chain'] == 2048 and c['dtype'] == 'uint8', 'corpus layout')
        require(sha256(c['path']) == c['sha256'], 'corpus array')
        x = np.memmap(c['path'], mode='r', dtype=np.uint8, shape=tuple(c['shape']))
        require(int(x.max()) < c['q'], 'spin alphabet')
    return ds


def check_draws(draws, spec, ds):
    require(set(draws) == {'cell', 'site', 'index'}, 'draw keys')
    n = spec['steps']
    for k, shape in [('cell', (n,)), ('site', (n,)), ('index', (n, 32))]:
        require(draws[k].dtype == np.dtype('int64') and draws[k].shape == shape, 'draw dtype/shape: ' + k)
    require(np.all((draws['cell'] >= 0) & (draws['cell'] < len(ds['cells']))), 'cell range')
    for c in ds['cells']:
        mask = draws['cell'] == c['id']; idx = draws['index'][mask].ravel()
        available = (6 if spec['stage'] == 'development' else c['chains']) * c['samples_per_chain']
        require(np.all((draws['site'][mask] >= 0) & (draws['site'][mask] < c['L'] ** 2)), 'site range')
        require(np.all((idx >= 0) & (idx < available)), 'sample range')
        require(len(np.unique(idx)) == len(idx), 'repeated configuration identity')


def check_qualifications(spec):
    for h in [4, 8]:
        bound = spec['qualifications'][str(h)]
        require(set(bound) == {'path', 'sha256'}, 'qualification binding')
        p = Path(bound['path'])
        require(sha256(p) == bound['sha256'], 'qualification digest')
        q = json.loads(p.read_text())
        require(q.get('status') == 'passed' and type(q.get('heads')) is int and q['heads'] == h, 'qualification width/status')
        equal(q.get('sources'), spec['sources'], 'qualification source')
        equal(q.get('native_assets'), spec['native_assets'], 'qualification native assets')
        equal(q.get('runtime'), spec['runtime'], 'qualification runtime')
        if spec['stage'] == 'qualification':
            require(q.get('schema') == 'physical-parity-qualification-v2', 'parity qualification schema')
            require(q.get('qualification_updates') == 12, 'parity update inventory')
            require(0 <= q['max_logit_error'] <= 2e-5 and 0 <= q['relative_gradient_error'] <= 2e-5, 'parity tolerance')
            require(q['incoming_state_sha256'] == next(c['state_sha256'] for c in spec['cases'] if c['heads'] == h), 'parity incoming state')
        else:
            require(q.get('schema') == 'physical-branch-qualification-v2', 'branch qualification schema')
            equal(q.get('branches'), ['full', 'frozen-generator', 'shuffled'], 'qualified branches')
            require(q.get('qualification_updates') == 36, 'branch update inventory')
            require(q.get('verifier_sha256') == sha256(REPO / 'scripts/verify_physical_execution.py'), 'branch verifier')
            short_path = Path(q['protocol'])
            require(sha256(short_path) == q['protocol_sha256'], 'branch protocol')
            short = json.loads(short_path.read_text()); check_design(short)
            require(short['stage'] == 'qualification', 'branch reduction stage')
            for key in ['native_assets', 'sources', 'runtime', 'optimizer', 'schedule', 'learning_rate', 'batch_size']:
                equal(short[key], spec[key], 'qualified reduction: ' + key)
            required = {str(short_path.parent / f'h{h}-s640101-{a}' / 'manifest.json')
                        for a in ['full', 'frozen-generator', 'shuffled']}
            require(required.issubset(q.get('verified_files', {})), 'branch terminal inventory')
            for name, digest in q['verified_files'].items():
                require(sha256(name) == digest, 'branch terminal identity')
            for a in ['full', 'frozen-generator', 'shuffled']:
                verify_completion(short_path.parent, short, f'h{h}-s640101-{a}')


@dataclass(frozen=True)
class ValidatedExecution:
    spec: dict
    data: dict
    development: dict
    draws: dict
    case: dict | None


def validate_spec(spec, draws, name=None):
    check_design(spec)
    case = None
    if name is not None:
        matches = [c for c in spec['cases'] if c['name'] == name]
        require(len(matches) == 1, 'worker case')
        case = matches[0]
    check_qualifications(spec)
    for c in spec['cases']:
        require(sha256(c['state']) == c['state_sha256'], 'incoming state digest')
    ds = check_corpus(spec['training_data'], spec['data_manifest_sha256'], spec['stage'] == 'development')
    dev = check_corpus(spec['development_data'], spec['development_manifest_sha256'], True)
    check_draws(draws, spec, ds)
    return ValidatedExecution(spec, ds, dev, draws, case)


def admit(study, name=None):
    study = Path(study)
    spec = json.loads((study / 'protocol.json').read_text())
    # Fail on design before opening arrays or creating worker directories.
    check_design(spec)
    with np.load(study / 'draws.npz', allow_pickle=False) as z:
        draws = {k: z[k] for k in z.files}
    execution = validate_spec(spec, draws, name)
    require(sha256(study / 'draws.npz') == spec['draws_sha256'], 'draw digest')
    return execution


def verify_completion(study, spec, name):
    folder = Path(study) / name
    m = json.loads((folder / 'manifest.json').read_text())
    case = next(c for c in spec['cases'] if c['name'] == name)
    require(m['status'] == 'complete' and m['completed_updates'] == spec['steps'], 'resume completion')
    equal(m['case'], case, 'resume case')
    require(m['protocol_sha256'] == sha256(Path(study) / 'protocol.json'), 'resume protocol')
    equal([c['step'] for c in m['checkpoints']], spec['checkpoints'][1:], 'resume checkpoints')
    require('training-trace.npy' in m['files'] and f"observation-{spec['steps']:05d}.npz" in m['files'], 'resume inventory')
    for n, digest in m['files'].items():
        require(Path(n).name == n and sha256(folder / n) == digest, 'resume file identity')
    for c in m['checkpoints']:
        require(Path(c['path']).resolve() == (folder / f"state-{c['step']:05d}.pt").resolve(), 'resume state path')
        require(sha256(c['path']) == c['sha256'], 'resume state identity')
    return m
