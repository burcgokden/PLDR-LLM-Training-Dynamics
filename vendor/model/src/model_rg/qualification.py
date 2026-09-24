"""Fail-closed correspondence between short native qualification and execution.

Hashes are integrity/correspondence checks, not attestations or scientific proofs.
Both supported native schemas normalize here; no indispensable check uses assert.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np
from model_rg.provenance import sha256

VERSION = 'native-execution-correspondence-v1'
SCHEMAS = {'onepass-refresh-protocol-v1': 'P1', 'onepass-directional-v1': 'P3'}
REQUIRED = {
    'P1': ['scripts/refresh_study.py', 'scripts/analyze_refresh.py',
           'scripts/verify_refresh.py', 'scripts/verify_refresh_summary.py',
           'scripts/workspace-wrappers/state-refresh-study.sh'],
    'P3': ['scripts/directional_study.py', 'scripts/analyze_directional.py',
           'scripts/verify_directional.py', 'scripts/analyze_pulse_parity.py',
           'scripts/workspace-wrappers/directional-response-study.sh'],
}
TERMINALS = {'P1': [('verification.json', 'scripts/verify_refresh.py'),
                    ('summary-verification.json', 'scripts/verify_refresh_summary.py')],
             'P3': [('verification.json', 'scripts/verify_directional.py'),
                    ('parity-analysis.json', 'scripts/analyze_pulse_parity.py')]}
SETTINGS = {'P1': ['primary_steps', 'fit_replicates', 'adaptation', 'calibration',
                   'budgets', 'batch_size', 'bootstrap_samples', 'bootstrap_seed',
                   'regularization', 'gate', 'primary', 'secondary', 'controls'],
            'P3': ['amplitude', 'arms', 'batch_size', 'direction', 'observation',
                   'primary', 'secondary', 'gate']}

class QualificationError(ValueError):
    """An execution has no current complete qualification correspondence."""

def need(condition, message):
    if not condition:
        raise QualificationError(message)

def read(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise QualificationError('Missing or malformed record: '+str(path)) from exc
    need(isinstance(value, dict), 'Expected an object: '+str(path))
    return value

def stage_of(spec):
    need(spec.get('schema') in SCHEMAS, 'Unknown native qualification schema')
    stage = SCHEMAS[spec['schema']]
    if stage == 'P1':
        need(type(spec.get('qualification')) is bool and 'kind' not in spec,
             'Ambiguous P1 qualification identity')
    else:
        need(spec.get('kind') in ['qualification', 'development', 'confirmation']
             and 'qualification' not in spec, 'Ambiguous P3 qualification identity')
    need(spec.get('status') == 'frozen', 'Protocol must be frozen')
    return stage

def is_short(spec):
    return spec.get('qualification') is True or spec.get('kind') == 'qualification'

def runtime():
    packages = ['numpy', 'scipy', 'torch', 'transformers', 'safetensors',
                'sentencepiece', 'pyarrow', 'torchtune', 'einops']
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    import torch
    return dict(python=platform.python_version(), executable_sha256=sha256(sys.executable),
                packages=versions, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                numerical_policy=dict(parameters='float32', observations='float64',
                    tf32=False, threads=4, attention='eager', dropout=0.,
                    clipping='global_l2_1', objective='full_vocabulary_next_token',
                    restoration='parameters_both_Adam_moments_scheduler_counters',
                    source_law='distinct_blocks_without_replacement_per_path'))

def source_names(repo, stage, inherited=()):
    """Resolve local static imports recursively, plus explicit dynamic entry roots."""
    repo = Path(repo).resolve()
    found = set(inherited) | set(REQUIRED[stage]) | {'src/model_rg/qualification.py',
            'scripts/confirmation_status.py', 'src/model_rg/training.py',
            'src/model_rg/native.py', 'scripts/measure_law_closure.py'}
    for init in (repo/'src/model_rg').glob('__init__.py'):
        found.add(init.relative_to(repo).as_posix())
    pending = list(found)
    while pending:
        name = pending.pop()
        path = repo/name
        need(path.is_file() and path.resolve().is_relative_to(repo), 'Unknown source: '+name)
        if path.suffix != '.py':
            continue
        tree = ast.parse(path.read_text())
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
                modules += [node.module+'.'+a.name for a in node.names]
        for module in modules:
            choices = [repo/'src'/Path(*module.split('.')).with_suffix('.py'),
                       repo/'scripts'/Path(*module.split('.')).with_suffix('.py')]
            for dep in choices:
                if dep.is_file():
                    rel = dep.relative_to(repo).as_posix()
                    if rel not in found:
                        found.add(rel); pending.append(rel)
    return sorted(found)

def check_hashes(bindings, base=None):
    need(isinstance(bindings, dict) and bindings, 'Missing binding inventory')
    for name, digest in bindings.items():
        path = (Path(base)/name) if base is not None else Path(name)
        if base is not None:
            need(not Path(name).is_absolute() and '..' not in Path(name).parts,
                 'Noncanonical repository identity: '+name)
        need(isinstance(digest, str) and len(digest) == 64 and path.is_file(),
             'Missing bound artifact: '+str(path))
        need(sha256(path) == digest, 'Changed bound artifact: '+str(path))

def assets(root, spec):
    root = Path(root).resolve(); old = root/'outer-transfer-20260911'
    roles = {f'parent/{p}': old/p for p in ['protocol.json', 'data/manifest.json',
              'data-verification.json', 'data/panels.npz', 'data/corpus-0.npy', 'data/corpus-1.npy']}
    native = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    for path in native.iterdir():
        if path.suffix in ['.py', '.json', '.model']:
            roles['native/'+path.name] = path
    # Every endpoint is identified even though short qualification samples two.
    prior = read(old/'protocol.json')
    for case in prior['cases']:
        for name in ['incoming-state.pt', 'results.json']:
            roles['state/'+case['name']+'/'+name] = old/'runs'/case['name']/name
    if stage_of(spec) == 'P1':
        roles['archive/forecast'] = old/'archive-forecast.json'
    return {role: dict(path=str(path), sha256=sha256(path)) for role, path in sorted(roles.items())}

def check_design(spec, root, study):
    stage = stage_of(spec); short = is_short(spec)
    prior = read(Path(root)/'outer-transfer-20260911/protocol.json')
    expected = prior['cases']
    if short:
        expected = [c for c in expected if c['corpus'] == 0 and c['identity'] == 0]
    need(spec.get('cases') == expected, 'Unsupported case coverage or model/optimizer recipe')
    need(spec.get('batch_size') == 32, 'Changed batch size')
    need(spec.get('root') == str(Path(root).resolve()), 'Wrong source root')
    need(spec.get('parent_study') == str(Path(root).resolve()/'outer-transfer-20260911'), 'Wrong parent study')
    if stage == 'P1':
        h, n = (4, 4) if short else (64, 32)
        expected_times = [0, 1, 2, 3, 4] if short else [0, 1, 4, 16, 64]
        need(spec.get('horizon') == h and spec.get('successor_updates') == h
             and spec.get('assessment') == n and spec.get('horizons') == expected_times,
             'Unallowed P1 short/full design')
        for key, value in dict(primary_steps=2048, fit_replicates=2, adaptation=16,
                               calibration=8, budgets=[4, 8, 16]).items():
            need(spec.get(key) == value, 'Changed P1 operation: '+key)
    else:
        h, n = (8, 2) if short else ((128, 1) if spec['kind'] == 'development' else (128, 4))
        need(spec.get('horizon') == h and spec.get('source_replicates') == n,
             'Unallowed P3 short/full design')
        need(spec.get('times') == sorted(set([0, 1, 2]+list(range(4, h+1, 4)))), 'Wrong observation grid')
        amplitude = spec.get('amplitude')
        need(type(amplitude) in [int, float] and np.isfinite(amplitude) and 0 < amplitude < .01,
             'Invalid pulse amplitude')
    for case in expected:
        path = Path(study)/'sampling'/(case['name']+'.npz')
        need(spec['inputs'].get(str(path)) == sha256(path), 'Sampling arrays must be frozen')
        with np.load(path, allow_pickle=False) as data:
            order = np.random.default_rng(case['stream_seed']).permutation(524288)
            need(np.array_equal(data['primary'], order[:65536]), 'Wrong consumed prefix')
            if stage == 'P1':
                seed = 115110000+case['heads']*1000+case['corpus']*100+case['identity']+(10000000 if short else 0)
                rng = np.random.default_rng(seed); pool = order[65536:]
                advance = rng.choice(pool, 32*h, replace=False).reshape(h, 32)
                need(np.array_equal(data['successor'], advance), 'Changed successor source law')
                depleted = pool[~np.isin(pool, advance.ravel())]
                for name, pool in [('parent', pool), ('later', depleted)]:
                    want = np.stack([rng.choice(pool, 32*h, replace=False).reshape(h, 32) for _ in range(48+n)])
                    need(np.array_equal(data[name], want), 'Changed conditional draw: '+name)
            else:
                seed = 121200000+{'qualification': 0, 'development': 1000000, 'confirmation': 2000000}[spec['kind']]+case['heads']*1000+case['corpus']*100+case['identity']
                rng = np.random.default_rng(seed)
                want = np.stack([rng.choice(order[65536:], 32*h, replace=False).reshape(h, 32) for _ in range(n)])
                need(int(data['seed']) == seed and np.array_equal(data['blocks'], want), 'Changed paired source law')

def freeze_contract(spec, repo, root, study):
    """Called after draws are saved, before the protocol is written or executed."""
    stage = stage_of(spec)
    check_design(spec, root, study)
    need(all(key in spec for key in SETTINGS[stage]), 'Missing scientific settings')
    return dict(version=VERSION, stage=stage, runtime=runtime(), assets=assets(root, spec),
        settings={key: spec[key] for key in SETTINGS[stage]},
        permitted_reductions=dict(version='native-short-design-v1',
            fields=['horizon', 'derived_observation_grid', 'cohort_size', 'case_coverage', 'declared_source_seed'],
            coverage='Two width endpoints at corpus 0/identity 0; other endpoints are bound inputs, not qualified model replicates.'))

def validate_frozen(spec, repo, root, study):
    repo = Path(repo).resolve(); study = Path(study).resolve()
    need(spec == read(study/'protocol.json'), 'Worker arguments differ from frozen protocol')
    stage = stage_of(spec)
    contract = spec.get('execution_contract')
    need(isinstance(contract, dict) and contract.get('version') == VERSION, 'Missing current execution contract')
    need(contract.get('stage') == stage, 'Unrelated execution contract')
    required = source_names(repo, stage)
    need(set(required) <= set(spec.get('sources', {})), 'Missing scientific dependency')
    check_hashes(spec['sources'], repo)
    check_hashes(spec['sources'], study/'executed-source')
    need(read(study/'executed-source/manifest.json').get('sources') == spec['sources'], 'Snapshot inventory differs')
    check_hashes(spec.get('inputs'))
    need(contract.get('runtime') == runtime(), 'Changed runtime or numerical policy')
    need(contract.get('settings') == {key: spec.get(key) for key in SETTINGS[stage]}, 'Changed scientific settings')
    need(contract.get('assets') == assets(root, spec), 'Changed native/source/state asset')
    check_design(spec, root, study)
    return stage

@dataclass(frozen=True)
class Qualification:
    stage: str
    protocol: str
    protocol_sha256: str
    terminal: dict
    contract: dict

    def record(self):
        return dict(version=VERSION, stage=self.stage, protocol=self.protocol,
                    protocol_sha256=self.protocol_sha256, terminal=self.terminal,
                    contract=self.contract)

def validate_qualification(protocol, repo, expected_stage=None):
    protocol = Path(protocol).resolve(); study = protocol.parent
    spec = read(protocol); stage = stage_of(spec)
    need(is_short(spec), 'Evidence is not a short qualification')
    need(expected_stage is None or stage == expected_stage, 'Unrelated qualification stage')
    validate_frozen(spec, repo, spec['root'], study)
    terminal = {}
    for filename, source in TERMINALS[stage]:
        path = study/filename; record = read(path)
        need(record.get('status') == 'passed', 'Incomplete terminal evidence: '+filename)
        key = 'analyzer_sha256' if filename == 'parity-analysis.json' else 'verifier_sha256'
        need(record.get(key) == sha256(Path(repo)/source), 'Changed terminal program: '+source)
        links = record.get('checked_sha256', record.get('inputs_sha256'))
        check_hashes(links)
        if filename == 'parity-analysis.json':
            need(record.get('protocol_sha256') == sha256(protocol)
                 and record.get('native_verification_sha256') == sha256(study/'verification.json'), 'Stale parity chain')
        else:
            need(links.get(str(protocol)) == sha256(protocol), 'Terminal evidence does not bind protocol')
        if filename == 'verification.json':
            need((record.get('qualification') is True) if stage == 'P1' else record.get('kind') == 'qualification',
                 'Terminal stage identity mismatch')
            need(set(spec['inputs']) <= set(links), 'Incomplete primary input checks')
            for name, digest in spec['sources'].items():
                need(links.get(str(study/'executed-source'/name)) == digest,
                     'Unchecked canonical snapshot dependency: '+name)
            count=len(spec['cases']); h=spec['horizon']
            if stage == 'P1':
                need(record.get('schema') == 'onepass-refresh-verification-v1', 'Unknown P1 terminal schema')
                paths=count*2*(48+spec['assessment'])
                need(record.get('branches') == paths and record.get('qualification_updates') == count*h+paths*h
                     and record.get('scientific_updates') == 0 and record.get('replay_updates') == count*2*h,
                     'Incomplete P1 execution ledger')
                required=[study/'analysis.json',study/'scores.npz']
                for case in spec['cases']:
                    out=study/'runs'/case['name']
                    required += [out/'results.json',out/'successor-state.pt',out/'successor-transition.npz']
                    for state in ['parent','successor']:
                        required += [out/state/'results.json',out/state/'paths.npy']
                        required += [out/state/f'branch-{b:03d}.npz' for b in range(48+spec['assessment'])]
                        required += [out/state/f'{kind}-{rep}.json' for kind in ['fit','calibration'] for rep in range(2)]
            else:
                need(record.get('cases') == count and record.get('arms_per_source') == 9
                     and record.get('primary_cells') == count*2*spec['source_replicates']
                     and record.get('qualification_updates') == count*9*spec['source_replicates']*h
                     and record.get('scientific_updates') == 0 and record.get('replay_updates') == count*h,
                     'Incomplete P3 execution ledger')
                required=[study/'analysis.json',study/'run-status.json',study/'executed-source/manifest.json']
                for case in spec['cases']:
                    out=study/'runs'/case['name']
                    required += [out/'results.json',out/'directions.json']
                    required += [out/f"source{source}-{arm['name']}.npz" for source in range(spec['source_replicates']) for arm in spec['arms']]
            need(all(str(path) in links for path in required), 'Incomplete native output evidence chain')
        elif filename == 'summary-verification.json':
            need(links.get(str(study/'verification.json')) == sha256(study/'verification.json'), 'Stale secondary P1 chain')
        if filename == 'summary-verification.json':
            need(all(type(record.get(k)) is int and record[k]>0 for k in ['mean_predictions','budget_gates','costs','calibrated_sets']),
                 'Incomplete P1 summary outcomes')
        if filename == 'parity-analysis.json':
            need(record.get('schema') == 'finite-pulse-parity-v1'
                 and len(record.get('rows',[])) == len(spec['cases'])*4*spec['source_replicates']
                 and len(record.get('covariance_terms',[])) == len(spec['cases'])*8,
                 'Incomplete P3 parity outcomes')
        terminal[str(path)] = sha256(path)
    return Qualification(stage, str(protocol), sha256(protocol), terminal, spec['execution_contract'])

def admit(spec, repo, root, study):
    """Guard launch and every direct worker, before output directories/updates."""
    stage = validate_frozen(spec, repo, root, study)
    if is_short(spec):
        return
    record = spec.get('qualification_binding')
    need(isinstance(record, dict) and record.get('version') == VERSION, 'Missing bound qualification')
    qualified = validate_qualification(record.get('protocol', ''), repo, stage)
    need(record == qualified.record(), 'Qualification chain changed after preparation')
    need(spec['execution_contract'] == qualified.contract, 'Scientific execution is not qualified')
