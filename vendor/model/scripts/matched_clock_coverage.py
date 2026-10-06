"""Versioned, portable census for an explicitly admitted observed clock family.

Admission is independent of submitted result rows. The bundled protocol is a
normalized census specification with its own byte digest. PROTOCOL_SHA256
continues to bind the original acquisition bytes, without authorizing a rewrite.
Large final states are native replay assets, outside the CPU reduction census.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path

from model_rg.provenance import sha256
from numerical_validation import load_json_strict

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'scripts/contracts/matched-clock-observation-v1.json'
BUNDLED_DESIGN_SHA256 = '39f17ff5484b89138ca9a47e381cfd19972d49a40ff90cf82f3fb38d6d99bcdb'
PROTOCOL_SHA256 = '71396258762b83c5a87727d9b5aea913bb3f6216b4eeafb4fc81b7d20a0c4788'


def require(condition, message):
    if not condition:
        raise ValueError('Matched-clock coverage: ' + message)


def read(path):
    text = Path(path).read_text()
    load_json_strict(text)  # finite JSON numbers, including exponent overflow
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {name: sha256(ROOT / name) for name in [
        'scripts/matched_clock_coverage.py', 'scripts/numerical_validation.py',
        'scripts/contracts/matched-clock-observation-v1.json']}


def admitted_protocol(study=None):
    require(sha256(PROTOCOL) == BUNDLED_DESIGN_SHA256, 'changed admitted design')
    p = read(PROTOCOL)
    if study is not None:
        require(sha256(Path(study) / 'protocol.json') == PROTOCOL_SHA256,
                'unregistered observation protocol')
        p = read(Path(study) / 'protocol.json')  # original bytes were authenticated above
    return p


def number(value, label, integer=False, nonnegative=False):
    require(type(value) is int if integer else type(value) in (int, float), label + ' type')
    require(math.isfinite(value), label + ' finite')
    if nonnegative:
        require(value >= 0, label + ' nonnegative')


def keys(row, expected, label):
    require(type(row) is dict and set(row) == set(expected), label + ' schema')


def digest(value):
    require(type(value) is str and len(value) == 64 and set(value) <= set('0123456789abcdef'), 'digest type')


def census(rows, columns, expected, label):
    require(type(rows) is list, label + ' list')
    found = []
    for row in rows:
        require(type(row) is dict, label + ' object')
        for key in columns:
            require(key in row, label + ' missing ' + key)
            v = row[key]
            if key in ('heads', 'seed', 'steps', 'update'):
                number(v, label + ' ' + key, integer=True)
            elif key in ('control', 'age'):
                if not (key == 'control' and v is None):
                    number(v, label + ' ' + key)
            else:
                require(type(v) is str, label + ' string ' + key)
        found.append(tuple(row[k] for k in columns))
    require(len(found) == len(set(found)), label + ' duplicate key')
    require(set(found) == set(expected), label + ' missing or unexpected key')


def design(p):
    f = p['family']
    fields = list(itertools.product(f['heads'], f['controls'], p['ages']))
    cache = []
    for n, g in p['cache_conditions']:
        if g == 0:
            cache.append((n, None, 0, 'state'))
        cache.extend((n, g, t, policy) for t in p['ages'] if t != 0
                     for policy in ('state', 'initial'))
    jobs = {(j['heads'], j['control'], j['seed']): j for j in p['jobs']}
    required = {'protocol.json', 'selection.npz', 'reservation.json', 'qualification/manifest.json'}
    for j in p['jobs']:
        prefix = 'runs/' + j['run_id'] + '/'
        required.update({prefix + 'manifest.json', prefix + 'training.npz'})
        required.update(prefix + f'age-{int(t*j["steps"])}.npz' for t in p['ages'])
    return fields, cache, jobs, required


def seed_binding(p, hashes, n, g, age, seed):
    jobs = design(p)[2]
    j = jobs[n, 0. if g is None else g, seed]
    path = 'runs/' + j['run_id'] + f'/age-{int(age*j["steps"])}.npz'
    return dict(seed=seed, run_id=j['run_id'], observation_path=path, observation_sha256=hashes[path])


def inventory(study):
    """Check all required observation roles before any reduction is emitted."""
    study = Path(study)
    p = admitted_protocol(study)
    _, _, jobs, required = design(p)
    actual_runs = {x.name for x in (study / 'runs').iterdir()}
    require(actual_runs == {j['run_id'] for j in jobs.values()}, 'run directory census')
    hashes = {name: sha256(study / name) for name in sorted(required)}
    require(hashes['selection.npz'] == p['selection_sha256'], 'selection identity')
    require(hashes['reservation.json'] == p['reservation_sha256'], 'reservation identity')
    for j in jobs.values():
        folder = study / 'runs' / j['run_id']
        m = read(folder / 'manifest.json')
        require(m['status'] == 'complete' and canonical(m['job']) == canonical(j) and
                m['protocol_sha256'] == PROTOCOL_SHA256, 'trajectory identity')
        keys(m['calls'], ['training','observation'], 'call ledger')
        for key in m['calls']:number(m['calls'][key], 'calls '+key, integer=True, nonnegative=True)
        number(m['updates'], 'updates', integer=True)
        require(m['updates'] == j['steps'], 'trajectory updates')
        names = {f'age-{int(t*j["steps"])}.npz' for t in p['ages']} | {'training.npz', 'final-state.pt'}
        require(set(m['artifacts']) == names, 'artifact role census')
        # Replay states may be stored separately in an observation-only deposit.
        observed = {x.name for x in folder.iterdir() if x.suffix in ('.npz', '.pt', '.json')}
        require(observed - {'final-state.pt'} == (names - {'final-state.pt'}) | {'manifest.json'}, 'unexpected run observation')
        for name, h in m['artifacts'].items():
            digest(h)
            if name != 'final-state.pt':
                require(hashes['runs/' + j['run_id'] + '/' + name] == h, 'artifact binding')
    for name, h in p['source_sha256'].items():
        require(sha256(study / 'executed-source' / name) == h, 'executed source ' + name)
    return p, hashes


def validate_analysis(a, p=None):
    p = p or admitted_protocol()
    require(p == admitted_protocol(), 'admitted protocol required')
    keys(a, ['status', 'schema', 'protocol_sha256', 'checked_sha256', 'scientific_updates',
             'scientific_observation_forwards', 'qualification_updates', 'qualification_forwards',
             'native_forward_calls', 'worker_seconds', 'peak_allocated_bytes', 'trajectories',
             'fields', 'cache_cells', 'maximum_identity_error', 'analysis_source_sha256',
             'coverage_sources_sha256', 'scope'], 'analysis')
    require(a['status'] == 'passed' and a['schema'] == 'matched-clock-analysis-v2', 'analysis status/version')
    require(a['protocol_sha256'] == PROTOCOL_SHA256, 'analysis protocol')
    require(a['coverage_sources_sha256'] == source_hashes(), 'coverage sources')
    digest(a['analysis_source_sha256'])
    require(type(a['scope']) is str, 'scope type')
    fields, cache, jobs, required = design(p)
    hashes = a['checked_sha256']
    keys(hashes, required, 'observation inventory')
    for h in hashes.values():
        digest(h)
    for name, h in [('protocol.json', PROTOCOL_SHA256), ('selection.npz', p['selection_sha256']),
                    ('reservation.json', p['reservation_sha256'])]:
        require(hashes[name] == h, 'design observation ' + name)
    census(a['trajectories'], ['heads', 'control', 'seed'], jobs, 'trajectories')
    for t in a['trajectories']:
        j = jobs[t['heads'], t['control'], t['seed']]
        keys(t, set(j) | {'initial_parameter_sha256', 'clipped_fraction', 'seconds', 'peak_allocated_bytes'}, 'trajectory')
        require(all(type(t[k]) is type(j[k]) and t[k] == j[k] for k in j), 'trajectory job binding')
        digest(t['initial_parameter_sha256'])
        number(t['clipped_fraction'], 'clipped fraction', nonnegative=True)
        require(t['clipped_fraction'] <= 1, 'clipped fraction range')
        number(t['seconds'], 'worker seconds', nonnegative=True)
        number(t['peak_allocated_bytes'], 'memory', integer=True, nonnegative=True)
    census(a['fields'], ['heads', 'control', 'age'], fields, 'fields')
    field_values = ['row_mean', 'row_susceptibility', 'operator_amplitude', 'operator_susceptibility',
                    'predictive_susceptibility', 'mean_nll', 'maximum_adaptive_force']
    for f in a['fields']:
        keys(f, ['heads', 'control', 'age', 'update', 'row_seed_means', 'seed_nll', 'seed_bindings', *field_values], 'field')
        number(f['update'], 'update', integer=True)
        require(f['update'] == int(f['age'] * 128 * f['heads']), 'field age binding')
        bindings = [seed_binding(p, hashes, f['heads'], f['control'], f['age'], s) for s in p['family']['seeds']]
        require(type(f['seed_bindings']) is list and canonical(f['seed_bindings']) == canonical(bindings), 'field seed binding/order')
        for key in ['row_seed_means', 'seed_nll']:
            require(type(f[key]) is list and len(f[key]) == len(bindings), 'field seed count')
            for v in f[key]:
                number(v, key)
        for key in field_values:
            number(f[key], key, nonnegative=True)
    census(a['cache_cells'], ['heads', 'control', 'age', 'policy'], cache, 'cache')
    budget_values = ['risk', 'scatter', 'displacement', 'mean_drift', 'initial_displacement', 'signed_cross']
    cache_values = ['relative_centered_rms', 'mean_kl', 'native_nll', 'cached_nll', 'mean_nll_change',
                    'maximum_context_rms', 'native_variance', 'residual_variance']
    bindings = []
    for c in a['cache_cells']:
        keys(c, ['heads', 'control', 'age', 'policy', 'contexts_over_target', 'per_context_rms',
                 'aggregate_targets_met', 'operator_budget', *cache_values], 'cache cell')
        require(type(c['aggregate_targets_met']) is bool, 'target decision type')
        number(c['contexts_over_target'], 'exceptions', integer=True, nonnegative=True)
        require(c['contexts_over_target'] <= p['contexts'], 'exception count')
        for key in cache_values:
            number(c[key], key, nonnegative=key not in ('mean_kl', 'mean_nll_change'))
        require(c['native_variance'] > 0, 'positive reference variance')
        require(type(c['per_context_rms']) is list and len(c['per_context_rms']) == p['contexts'], 'context count')
        for r in c['per_context_rms']:
            number(r, 'context RMS', nonnegative=True)
        require(c['contexts_over_target'] == sum(r > p['targets']['centered_rms'] for r in c['per_context_rms']), 'context decisions')
        require(c['aggregate_targets_met'] == (c['relative_centered_rms'] <= p['targets']['centered_rms'] and c['mean_kl'] <= p['targets']['mean_kl']), 'aggregate decision')
        bs = c['operator_budget']
        require(type(bs) is list and len(bs) == len(p['family']['seeds']), 'budget count')
        for b, seed in zip(bs, p['family']['seeds'], strict=True):
            expected = seed_binding(p, hashes, c['heads'], c['control'], c['age'], seed)
            keys(b, set(expected) | set(budget_values), 'budget')
            require(canonical({k: b[k] for k in expected}) == canonical(expected), 'budget seed/run/observation binding')
            for key in budget_values:
                number(b[key], key, nonnegative=key != 'signed_cross')
            bindings.append(dict(cell=[c[k] for k in ['heads', 'control', 'age', 'policy']], **expected))
    for key, value in dict(scientific_updates=76800, scientific_observation_forwards=2400,
                           qualification_updates=3, qualification_forwards=6, native_forward_calls=79206).items():
        number(a[key], key, integer=True)
        require(a[key] == value, 'workload ' + key)
    for key in ['worker_seconds', 'peak_allocated_bytes', 'maximum_identity_error']:
        number(a[key], key, integer=key == 'peak_allocated_bytes', nonnegative=True)
    require(abs(a['worker_seconds'] - sum(t['seconds'] for t in a['trajectories'])) < 1e-8, 'worker ledger')
    require(a['peak_allocated_bytes'] == max(t['peak_allocated_bytes'] for t in a['trajectories']), 'peak ledger')
    return dict(schema='matched-clock-coverage-v1', protocol_sha256=PROTOCOL_SHA256,
                field_keys=[list(x) for x in fields], cache_keys=[list(x) for x in cache],
                seeds=p['family']['seeds'], runs=p['jobs'], budget_bindings=bindings,
                observation_sha256=hashes, executed_source_sha256=p['source_sha256'],
                replay_assets_scope='Final optimizer states are excluded from observation-only reconstruction.')


def validate_certificate(v, a, analysis_path):
    cov = validate_analysis(a)
    keys(v, ['status','schema','fields','cache_cells','maximum_difference','analysis_sha256','protocol_sha256',
             'verifier_sha256','native_frozen_population_controls','scope','coverage','coverage_sha256',
             'coverage_sources_sha256'], 'certificate')
    require(type(v['scope']) is str, 'certificate scope')
    controls=v['native_frozen_population_controls']
    census(controls,['heads','control','age'],itertools.product([8,24],[0.,1.5],[0.,1.]),'finite population controls')
    for row in controls:
        keys(row,['heads','control','age','maximum_difference','scaled_covariance'],'finite population control')
        number(row['maximum_difference'],'reference discrepancy',nonnegative=True)
        matrix=row['scaled_covariance']
        require(type(matrix) is list and len(matrix)==2 and all(type(x) is list and len(x)==2 for x in matrix),'reference covariance shape')
        for line in matrix:
            for x in line:number(x,'reference covariance')
    require(v.get('status') == 'passed' and v.get('schema') == 'matched-clock-independent-v2', 'certificate version/status')
    require(v.get('analysis_sha256') == sha256(analysis_path) and v.get('protocol_sha256') == PROTOCOL_SHA256, 'certificate input binding')
    require(v.get('coverage') == cov and v.get('coverage_sha256') == canonical(cov), 'certificate census binding')
    require(v.get('coverage_sources_sha256') == source_hashes(), 'certificate contract sources')
    require(v.get('verifier_sha256') == sha256(ROOT / 'scripts/verify_matched_clock.py'), 'certificate verifier source')
    require(a['analysis_source_sha256'] == sha256(ROOT / 'scripts/analyze_matched_clock.py'), 'analysis source')
    require(type(v.get('fields')) is int and v['fields'] == len(cov['field_keys']) and
            type(v.get('cache_cells')) is int and v['cache_cells'] == len(cov['cache_keys']), 'certificate counts')
    number(v.get('maximum_difference'), 'reconstruction discrepancy', nonnegative=True)
    require(v['maximum_difference'] <= 3e-10, 'reconstruction tolerance')
    return cov


def validate_records(records, generated=None):
    """Run at both publication boundaries, before creating any output."""
    records = Path(records)
    ap = records / 'matched-clock-analysis.json'
    vp = records / 'matched-clock-verification.json'
    a, v = read(ap), read(vp)
    cov = validate_certificate(v, a, ap)
    if generated is not None:
        generated = Path(generated)
        r = read(generated / 'matched-clock-rendering.json')
        require(r.get('status') == 'passed' and r.get('coverage_sha256') == canonical(cov), 'rendered census')
        require(r['analysis_sha256'] == sha256(ap) and r['verification_sha256'] == sha256(vp), 'rendered inputs')
        require(r['renderer_sha256'] == sha256(ROOT / 'scripts/render_matched_clock.py'), 'renderer source')
        required = {'matched-clock-endpoints.tex', 'matched-clock-details.tex', 'matched-clock.pdf', 'matched-clock-nll.pdf', 'matched-clock-summary.json'}
        keys(r['outputs'], required, 'rendered output inventory')
        for name, h in r['outputs'].items():
            require(sha256(generated / name) == h, 'rendered artifact ' + name)
    return cov
