"""Versioned, complete admission for the fixed cache-state experiment.

Acquisition snapshots are immutable. The one retained v1 protocol is read-only;
new acquisitions use v2 implementation metadata with the same scientific design.
This module checks identities and geometry, never predictive target success.
"""
from companion_paths import configured_path
from acquisition_paths import logical_root, resolve_recorded
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from model_rg.provenance import sha256

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(configured_path('data:model'))
PARENT = ROOT/'critical-onepass-refinement-20260914'
PROBES = ROOT/'controlled-study-20260905/data/short'
CORPUS = ROOT/'data/refinedweb-onepass-524288'
NATIVE = ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
DESIGN = 'rg-cache-state-transfer-v1'
CONTRACT = 'cache-state-contract-v3'
HISTORICAL_PROTOCOL = 'faacd92b5e44ebc3755750d1bf6e740be65b163bb64d7ef51872de2ba39d94cb'
STATES = ['initial', 'g0', 'g1.5']
LENGTHS = [32, 64, 128]
SEEDS = list(range(915101, 915107))
CALLS = dict(calibration=216, native_assessment=864,
             recalibrated_assessment=864, initial_cache_assessment=576,
             qualification=504)
OLD_SOURCES = ['scripts/run_cache_state_transfer.py', 'scripts/context_categorical_study.py',
    'scripts/numerical_validation.py', 'src/model_rg/provenance.py', 'src/model_rg/training.py',
    'src/model_rg/native.py', 'src/model_rg/criticality.py', 'src/model_rg/variance_family.py',
    'src/model_rg/inference_interventions.py']
V2_SOURCES = OLD_SOURCES + ['scripts/cache_state_contract.py']
SOURCES = V2_SOURCES + ['scripts/context_reservations.py','scripts/context_execution_contract.py']
ANALYSIS_SOURCES = ['scripts/analyze_cache_state_transfer.py', 'scripts/analyze_operator_cache.py',
    'scripts/numerical_validation.py', 'scripts/cache_risk_validation.py',
    'src/model_rg/provenance.py', 'scripts/cache_state_contract.py', 'scripts/acquisition_paths.py']
VERIFIER_SOURCES = ['scripts/verify_cache_state_transfer.py', 'scripts/verify_cache_risk_study.py',
    'scripts/numerical_validation.py', 'src/model_rg/provenance.py', 'scripts/cache_state_contract.py', 'scripts/acquisition_paths.py']
BASE_KEYS = set('cache_rule calibration_contexts comparison conditioning context_rows contexts '
    'design_id disk_budget_bytes document_hashes expected_calls hypotheses input_sha256 jobs '
    'memory_ceiling_bytes new_training_replicas parent prefix_lengths schema selection_sha256 '
    'source_sha256 states status target_exclusion targets training_updates vocabulary worker_hour_cap'.split())


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    def floating(value):
        number = float(value)
        if not math.isfinite(number):
            bad(value)
        return number
    return json.loads(resolve_recorded(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=bad, parse_float=floating)


def same(actual, expected, label):
    # JSON distinguishes booleans from numbers; Python equality alone does not.
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError('Invalid ' + label)
    if isinstance(expected, dict):
        for key in expected:
            same(actual[key], expected[key], label + '/' + key)
    elif isinstance(expected, list):
        for a, b in zip(actual, expected):
            same(a, b, label)


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError('Incomplete or unexpected ' + label)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def hash_value(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Malformed SHA256')


def local(study, name):
    if not isinstance(name, str) or str(Path(name)) != name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Noncanonical local member')
    path = study/name
    if path.resolve() != path.absolute() or not path.resolve().is_relative_to(study.resolve()):
        raise ValueError('Local path alias or symlink escape')
    return path


def sources(mapping, expected, root):
    keys(mapping, expected, 'source roles')
    for name, value in mapping.items():
        hash_value(value)
        if sha256(local(root, name)) != value:
            raise ValueError('Changed source: ' + name)


def artifacts():
    names = {'caches.npz'}
    for state in STATES:
        for length in LENGTHS:
            names.update(f'{state}-L{length}-{role}' for role in
                         ['native.npy', 'recalibrated.npy', 'operators.npz'])
            if state != 'initial':
                names.add(f'{state}-L{length}-initial_cache.npy')
    return names


def cells():
    return [(n, s, l, policy) for n in [8, 24] for s in STATES for l in LENGTHS
            for policy in (['recalibrated'] if s == 'initial' else ['recalibrated', 'initial_cache'])]


def external_roles(p):
    root = logical_root(ROOT)
    parent = root/'critical-onepass-refinement-20260914'
    probes = root/'controlled-study-20260905/data/short'
    corpus = root/'data/refinedweb-onepass-524288'
    native = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    fixed = {str(x) for x in [probes/'tokens.npy', probes/'records.json', corpus/'records.json',
                             parent/'protocol.json', native/'modeling_pldrllm.py', native/'configuration_pldrllm.py']}
    fixed.update(str(parent/'runs'/e['run_id']/'manifest.json')
                 for j in p['jobs'] for e in j['endpoints'].values())
    if p['schema'] in ['cache-state-transfer-v2','cache-state-transfer-v3']:
        exclusions = p['excluded_context_protocols']
        if not isinstance(exclusions, list) or exclusions != sorted(set(exclusions)):
            raise ValueError('Noncanonical exclusion roles')
    else:
        # The immutable protocol hash pins this explicit historical set.
        exclusions = sorted(set(p['input_sha256']) - fixed)
    for name in exclusions:
        path = Path(name)
        if path.parent.parent != root or path.name != 'protocol.json' or str(path) != name:
            raise ValueError('Foreign exclusion path')
    return fixed | set(exclusions), exclusions


def validate_protocol(study, *, acquisition=False, index=None):
    study = Path(study).resolve()
    p = read(local(study, 'protocol.json'))
    identity_root = logical_root(ROOT)
    identity_parent = identity_root/'critical-onepass-refinement-20260914'
    identity_probes = identity_root/'controlled-study-20260905/data/short'
    identity_corpus = identity_root/'data/refinedweb-onepass-524288'
    version = p.get('schema')
    if version not in ['cache-state-transfer-v1', 'cache-state-transfer-v2','cache-state-transfer-v3']:
        raise ValueError('Unsupported state-transfer schema')
    if version == 'cache-state-transfer-v1':
        if acquisition or sha256(study/'protocol.json') != HISTORICAL_PROTOCOL:
            raise ValueError('Only the pinned immutable v1 acquisition may be reconstructed')
        required_sources = OLD_SOURCES
        keys(p, BASE_KEYS, 'protocol members')
    elif version == 'cache-state-transfer-v2':
        if acquisition: raise ValueError('New acquisitions require shared reservation contract v3')
        required_sources = V2_SOURCES
        keys(p, BASE_KEYS | {'implementation', 'excluded_context_protocols'}, 'protocol members')
        same(p['implementation'], 'cache-state-contract-v2', 'implementation')
    else:
        required_sources = SOURCES
        keys(p, BASE_KEYS | {'implementation', 'excluded_context_protocols','reservation_sha256'}, 'protocol members')
        same(p['implementation'], CONTRACT, 'implementation')
        from context_reservations import validate_receipt
        validate_receipt(study,p)
    for key, value in dict(design_id=DESIGN, states=STATES, prefix_lengths=LENGTHS,
            contexts=64, calibration_contexts=16, vocabulary=32000, training_updates=0,
            new_training_replicas=0, expected_calls=CALLS, worker_hour_cap=2,
            memory_ceiling_bytes=22*1024**3, disk_budget_bytes=4*1024**3,
            parent=str(identity_parent), status='frozen_before_acquisition',
            targets=dict(relative_centered_rms=.25, mean_kl=.03),
            target_exclusion='At length L, only tokens [:L] enter the model; token L is an external NLL target.').items():
        same(p[key], value, key)
    for key in ['cache_rule', 'comparison', 'conditioning']:
        if not isinstance(p[key], str) or not p[key].strip():
            raise ValueError('Missing design description')
    if not isinstance(p['hypotheses'], list) or not p['hypotheses'] or not all(isinstance(x, str) and x for x in p['hypotheses']):
        raise ValueError('Invalid hypothesis descriptions')
    if not isinstance(p['jobs'], list) or len(p['jobs']) != 12:
        raise ValueError('Incomplete job grid')
    grid = []
    for job in p['jobs']:
        keys(job, ['run_id', 'heads', 'seed', 'shared_seed', 'initial_parameter_sha256',
                   'initial_shared_sha256', 'endpoints'], 'job members')
        n, seed = job['heads'], job['seed']
        if type(n) is not int or n not in [8, 24] or type(seed) is not int or seed not in SEEDS:
            raise ValueError('Foreign initialization')
        same(job['run_id'], f'h{n}-s{seed}', 'run identity')
        same(job['shared_seed'], 9152510, 'shared initialization')
        hash_value(job['initial_parameter_sha256']); hash_value(job['initial_shared_sha256'])
        keys(job['endpoints'], ['g0', 'g1.5'], 'paired endpoints')
        for state, endpoint in job['endpoints'].items():
            keys(endpoint, ['run_id', 'checkpoint_sha256'], 'endpoint members')
            same(endpoint['run_id'], f'e1-h{n}-{state}-s{seed}', 'endpoint identity')
            hash_value(endpoint['checkpoint_sha256'])
        grid.append((n, seed))
    same(grid, [(n, s) for n in [8, 24] for s in SEEDS], 'ordered paired jobs')
    if index is not None and (type(index) is not int or not 0 <= index < 12):
        raise ValueError('Worker index outside design')
    keys(p['source_sha256'], required_sources, 'acquisition source roles')
    sources(p['source_sha256'], required_sources, study/'executed-source')
    if acquisition:
        sources(p['source_sha256'], required_sources, REPO)
    hash_value(p['selection_sha256'])
    if sha256(local(study, 'inputs.npz')) != p['selection_sha256']:
        raise ValueError('Changed selected inputs')
    with np.load(study/'inputs.npz', allow_pickle=False) as z:
        if set(z.files) != {'blocks', 'rows'} or len(z.files) != 2:
            raise ValueError('Wrong input members')
        blocks, rows = z['blocks'], z['rows']
    if blocks.shape != (80, 129) or blocks.dtype.kind not in 'iu' or np.any(blocks < 0) or np.any(blocks >= 32000):
        raise ValueError('Invalid token geometry or range')
    if rows.shape != (80,) or rows.dtype.kind not in 'iu' or np.any(rows < 0) or not np.array_equal(rows, np.arange(rows[0], rows[0]+80)):
        raise ValueError('Invalid source rows')
    same(p['context_rows'], rows.tolist(), 'source rows')
    if not isinstance(p['document_hashes'], list) or len(set(p['document_hashes'])) != 80:
        raise ValueError('Repeated or incomplete document identities')
    for value in p['document_hashes']:
        hash_value(value)
    required, exclusions = external_roles(p)
    keys(p['input_sha256'], required, 'external input roles')
    for name, value in p['input_sha256'].items():
        path = Path(name)
        if str(path) != name or not path.is_absolute() or '..' in path.parts:
            raise ValueError('External path alias')
        hash_value(value)
        resolve_recorded(path, value)
    records = read(identity_probes/'records.json')
    if rows[-1] >= len(records):
        raise ValueError('Source rows outside retained pool')
    same(p['document_hashes'], [records[i]['content_sha256'] for i in rows], 'source document identities')
    original = np.load(resolve_recorded(identity_probes/'tokens.npy'), mmap_mode='r', allow_pickle=False)
    if not np.array_equal(blocks, original[rows, :129]):
        raise ValueError('Selected blocks differ from source')
    excluded = {r['content_sha256'] for r in read(identity_corpus/'records.json')}
    excluded.update(r['content_sha256'] for r in records[1024:1152])
    for name in exclusions:
        panel = read(name)
        if not isinstance(panel.get('document_hashes'), list):
            raise ValueError('Missing exclusion document role')
        excluded.update(panel['document_hashes'])
    named_replay = version == 'cache-state-transfer-v3' and read(study/'reservation.json')['role'] == 'named-panel-reproduction'
    if excluded.intersection(p['document_hashes']) and not named_replay:
        raise ValueError('Repeated assessment document')
    parent = read(identity_parent/'protocol.json')
    for job in p['jobs']:
        for state, endpoint in job['endpoints'].items():
            manifest = read(identity_parent/'runs'/endpoint['run_id']/'manifest.json')
            incoming = [j for j in parent['jobs'] if j['run_id'] == endpoint['run_id']]
            if len(incoming) != 1 or manifest['job'] != incoming[0] or manifest['status'] != 'complete':
                raise ValueError('Foreign or incomplete parent state')
            for key in ['heads', 'seed', 'shared_seed']:
                same(incoming[0][key], job[key], 'parent ' + key)
            if incoming[0]['control'] != float(state[1:]) or manifest['artifacts']['final-state.pt'] != endpoint['checkpoint_sha256']:
                raise ValueError('Foreign endpoint or checkpoint')
            for key in ['initial_parameter_sha256', 'initial_shared_sha256']:
                same(manifest[key], job[key], 'paired ' + key)
    if acquisition and index is not None:
        for endpoint in p['jobs'][index]['endpoints'].values():
            if sha256(PARENT/'runs'/endpoint['run_id']/'final-state.pt') != endpoint['checkpoint_sha256']:
                raise ValueError('Changed incoming checkpoint payload')
    if acquisition:
        elapsed = 0.
        for path in (study/'runs').glob('*/manifest.json'):
            m = read(path)
            duration = m.get('elapsed_seconds', 0.)
            if type(duration) not in [int, float] or not 0 <= duration <= 600:
                raise ValueError('Invalid existing worker duration')
            elapsed += duration
        if elapsed + 600 > p['worker_hour_cap']*3600:
            raise ValueError('Worker budget exhausted')
        if sum(path.stat().st_size for path in study.rglob('*') if path.is_file()) >= p['disk_budget_bytes']:
            raise ValueError('Disk budget exhausted')
    return p


def inventory(study, p):
    """Derive exact roles, then verify acquisition hashes before reduction."""
    study = Path(study).resolve()
    checked = {'protocol.json': sha256(local(study, 'protocol.json')),
               'inputs.npz': p['selection_sha256']}
    checked.update({'executed-source/'+n: h for n, h in p['source_sha256'].items()})
    counts = {k: 0 for k in CALLS}; seconds = 0.; peak = 0
    for job in p['jobs']:
        prefix = 'runs/'+job['run_id']+'/'
        path = local(study, prefix+'manifest.json'); m = read(path)
        keys(m, ['artifacts', 'calls', 'device', 'elapsed_seconds', 'initial_parameter_sha256',
                 'initial_shared_sha256', 'job', 'longest_batch_one', 'peak_allocated_bytes',
                 'protocol_sha256', 'qualification', 'status', 'training_updates'], 'run manifest')
        for key, expected in dict(status='complete', job=job, protocol_sha256=checked['protocol.json'],
                training_updates=0, longest_batch_one=True,
                initial_parameter_sha256=job['initial_parameter_sha256'],
                initial_shared_sha256=job['initial_shared_sha256'],
                calls={k: v//12 for k, v in CALLS.items()},
                qualification=[dict(state=s, length=l, own_replay=True, restoration=True,
                                    native_generator_calls=40, cached_generator_calls=0)
                               for s in STATES for l in LENGTHS]).items():
            same(m[key], expected, 'manifest '+key)
        if m['device'] not in ['cuda:0', 'cuda:1']:
            raise ValueError('Foreign device')
        if type(m['elapsed_seconds']) not in [float, int] or not 0 < m['elapsed_seconds'] <= 600:
            raise ValueError('Worker duration outside budget')
        if type(m['peak_allocated_bytes']) is not int or not 0 < m['peak_allocated_bytes'] < p['memory_ceiling_bytes']:
            raise ValueError('Memory outside budget')
        keys(m['artifacts'], artifacts(), 'observation artifact roles')
        if {f.name for f in path.parent.iterdir()} != artifacts() | {'manifest.json'}:
            raise ValueError('Unlisted or missing run member')
        checked[prefix+'manifest.json'] = sha256(path)
        for name, h in m['artifacts'].items():
            hash_value(h)
            checked[prefix+name] = h
        for k in counts:
            counts[k] += m['calls'][k]
        seconds += m['elapsed_seconds']; peak = max(peak, m['peak_allocated_bytes'])
    for name, value in checked.items():
        if sha256(local(study, name)) != value:
            raise ValueError('Changed acquisition artifact: ' + name)
    same(counts, CALLS, 'total calls')
    if seconds > p['worker_hour_cap']*3600:
        raise ValueError('Total worker time exceeded')
    return checked, counts, seconds, peak


def validate_arrays(study, p):
    def finite(value, shape, label):
        if value.shape != shape or value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
            raise ValueError('Invalid finite observation: ' + label)
    for job in p['jobs']:
        folder = study/'runs'/job['run_id']; n = job['heads']
        with np.load(folder/'caches.npz', allow_pickle=False) as z:
            expected = {f'{s}-L{l}' for s in STATES for l in LENGTHS}
            if set(z.files) != expected or len(z.files) != len(expected):
                raise ValueError('Wrong cache members')
            for name in z.files:
                finite(z[name], (5, 3, 1, n, 64, 64), name)
        for state in STATES:
            for length in LENGTHS:
                prefix = f'{state}-L{length}-'
                policies = ['recalibrated'] if state == 'initial' else ['recalibrated', 'initial_cache']
                for role in ['native'] + policies:
                    finite(np.load(folder/(prefix+role+'.npy'), allow_pickle=False, mmap_mode='r'),
                           (64, 32000), prefix+role)
                with np.load(folder/(prefix+'operators.npz'), allow_pickle=False) as z:
                    expected = {'mean', 'powers', 'scatter'} | {q+'-'+k for q in policies for k in ['risk', 'displacement']}
                    if set(z.files) != expected or len(z.files) != len(expected):
                        raise ValueError('Wrong operator members')
                    for name in z.files:
                        value = z[name]
                        finite(value, (5, n, 64, 64) if name == 'mean' else (5, 64) if name == 'powers' else (5,), name)
                        if name != 'mean' and np.any(value < 0):
                            raise ValueError('Negative squared observation')


def required_members(p):
    members = {'protocol.json', 'inputs.npz'}
    members.update('executed-source/'+n for n in p['source_sha256'])
    for job in p['jobs']:
        prefix = 'runs/'+job['run_id']+'/'
        members.update(prefix+n for n in artifacts() | {'manifest.json'})
    return members


def analysis_roles(a, p):
    keys(a.get('checked_sha256'), required_members(p), 'analysis observation roles')
    sources(a.get('analysis_sources_sha256'), ANALYSIS_SOURCES, REPO)


def validate_analysis(a, p, checked):
    same(a['status'], 'passed', 'analysis status')
    same(a['schema'], 'cache-state-transfer-analysis-v2', 'analysis schema')
    same(a['contract'], CONTRACT, 'analysis contract')
    same(a['protocol_sha256'], checked['protocol.json'], 'analysis protocol')
    sources(a['analysis_sources_sha256'], ANALYSIS_SOURCES, REPO)
    same(a['checked_sha256'], checked, 'complete analysis coverage')
    same(a['coverage_sha256'], digest(checked), 'coverage digest')
    same(a['external_sha256'], p['input_sha256'], 'external coverage')
    same(a['total_cells'], 30, 'cell count')
    same(a['training_updates'], 0, 'training count')
    same(a['new_training_replicas'], 0, 'replica count')
    same(a['calls'], CALLS, 'analysis calls')
    same([(c['heads'], c['state'], c['length'], c['policy']) for c in a['cells']], cells(), 'cell identities')
    for c in a['cells']:
        same(c['seeds'], SEEDS, 'cell seeds')
        same(c['contexts'], 64, 'cell contexts')
        same(c['updates'], 0 if c['state'] == 'initial' else 4096, 'cell age')
        same(c['control'], None if c['state'] == 'initial' else float(c['state'][1:]), 'cell control')


def validate_certificate(v, a, analysis, checked):
    for key, value in dict(status='passed', schema='cache-state-transfer-independent-v2',
            contract=CONTRACT, design_id=DESIGN, cells=30, calls=CALLS, training_updates=0,
            analysis_sha256=sha256(analysis), protocol_sha256=checked['protocol.json'],
            coverage_sha256=digest(checked), checked_sha256=checked,
            cell_identities=[list(x) for x in cells()], tolerances=dict(predictive_atol=4e-12,
            predictive_rtol=2e-10, operator_atol=4e-12, operator_rtol=1e-12)).items():
        same(v[key], value, 'certificate '+key)
    sources(v['verification_sources_sha256'], VERIFIER_SOURCES, REPO)
    same(v['external_sha256'], a['external_sha256'], 'certificate external coverage')
