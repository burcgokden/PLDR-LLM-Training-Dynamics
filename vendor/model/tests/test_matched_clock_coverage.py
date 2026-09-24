"""Portable synthetic census controls; native values are tested by raw replay."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from matched_clock_coverage import (admitted_protocol, design, seed_binding,
    source_hashes, validate_analysis, PROTOCOL_SHA256, read)


def synthetic_summary():
    p = admitted_protocol()
    fs, cs, jobs, required = design(p)
    hashes = {name: 'a'*64 for name in required}
    hashes.update({'protocol.json': PROTOCOL_SHA256, 'selection.npz': p['selection_sha256'],
                   'reservation.json': p['reservation_sha256']})
    fields = []
    for n, g, t in fs:
        fields.append(dict(heads=n, control=g, age=t, update=int(t*128*n), row_mean=.5,
            row_susceptibility=0., operator_amplitude=1., operator_susceptibility=0.,
            predictive_susceptibility=1., mean_nll=10., maximum_adaptive_force=1.,
            row_seed_means=[.5]*3, seed_nll=[10.]*3,
            seed_bindings=[seed_binding(p, hashes, n, g, t, s) for s in p['family']['seeds']]))
    cells = []
    for n, g, t, policy in cs:
        budgets = [dict(**seed_binding(p, hashes, n, g, t, s), risk=0., scatter=0.,
                        displacement=0., mean_drift=0., initial_displacement=0., signed_cross=0.)
                   for s in p['family']['seeds']]
        cells.append(dict(heads=n, control=g, age=t, policy=policy, relative_centered_rms=0.,
            mean_kl=0., native_nll=10., cached_nll=10., mean_nll_change=0., maximum_context_rms=0.,
            native_variance=1., residual_variance=0., contexts_over_target=0, per_context_rms=[0.]*64,
            aggregate_targets_met=True, operator_budget=budgets))
    return dict(status='passed', schema='matched-clock-analysis-v2', protocol_sha256=PROTOCOL_SHA256,
        checked_sha256=hashes, scientific_updates=76800, scientific_observation_forwards=2400,
        qualification_updates=3, qualification_forwards=6, native_forward_calls=79206,
        worker_seconds=48., peak_allocated_bytes=1, trajectories=[dict(**j,
            initial_parameter_sha256='a'*64, clipped_fraction=0., seconds=1., peak_allocated_bytes=1)
            for j in jobs.values()], fields=fields, cache_cells=cells, maximum_identity_error=0.,
        analysis_source_sha256='a'*64, coverage_sources_sha256=source_hashes(), scope='Synthetic schema fixture only.')


def mutations():
    return [
        ('duplicate_field', lambda a: a['fields'].__setitem__(1, copy.deepcopy(a['fields'][0]))),
        ('missing_field', lambda a: a['fields'].pop()),
        ('extra_field', lambda a: a['fields'].append(copy.deepcopy(a['fields'][0]))),
        ('duplicate_cache', lambda a: a['cache_cells'].__setitem__(1, copy.deepcopy(a['cache_cells'][0]))),
        ('missing_cache', lambda a: a['cache_cells'].pop()),
        ('extra_cache', lambda a: a['cache_cells'].append(copy.deepcopy(a['cache_cells'][0]))),
        ('duplicate_trajectory', lambda a: a['trajectories'].__setitem__(1, copy.deepcopy(a['trajectories'][0]))),
        ('missing_trajectory', lambda a: a['trajectories'].pop()),
        ('run_identity', lambda a: a['trajectories'][0].__setitem__('run_id', 'foreign')),
        ('bool_head', lambda a: a['fields'][0].__setitem__('heads', True)),
        ('bool_control', lambda a: a['fields'][0].__setitem__('control', False)),
        ('bool_age', lambda a: a['fields'][0].__setitem__('age', False)),
        ('wrong_age', lambda a: a['fields'][0].__setitem__('age', .75)),
        ('wrong_update', lambda a: a['fields'][0].__setitem__('update', 1)),
        ('missing_seed', lambda a: a['cache_cells'][0]['operator_budget'].pop()),
        ('extra_seed', lambda a: a['cache_cells'][0]['operator_budget'].append(copy.deepcopy(a['cache_cells'][0]['operator_budget'][0]))),
        ('duplicate_seed', lambda a: a['cache_cells'][0]['operator_budget'][1].update(a['cache_cells'][0]['operator_budget'][0])),
        ('seed_order', lambda a: a['cache_cells'][0]['operator_budget'].reverse()),
        ('bool_seed', lambda a: a['cache_cells'][0]['operator_budget'][0].__setitem__('seed', True)),
        ('budget_run', lambda a: a['cache_cells'][0]['operator_budget'][0].__setitem__('run_id', 'foreign')),
        ('budget_observation', lambda a: a['cache_cells'][0]['operator_budget'][0].__setitem__('observation_path', 'foreign')),
        ('budget_hash', lambda a: a['cache_cells'][0]['operator_budget'][0].__setitem__('observation_sha256', 'b'*64)),
        ('field_seed_order', lambda a: a['fields'][0]['seed_bindings'].reverse()),
        ('field_seed_count', lambda a: a['fields'][0]['row_seed_means'].pop()),
        ('missing_observation', lambda a: a['checked_sha256'].pop('qualification/manifest.json')),
        ('extra_observation', lambda a: a['checked_sha256'].__setitem__('foreign.npz', 'a'*64)),
        ('invalid_digest', lambda a: a['checked_sha256'].__setitem__('qualification/manifest.json', 'x'*64)),
        ('wrong_protocol', lambda a: a.__setitem__('protocol_sha256', 'a'*64)),
        ('nonfinite_field', lambda a: a['fields'][0].__setitem__('row_mean', float('nan'))),
        ('bool_scalar', lambda a: a['fields'][0].__setitem__('row_mean', True)),
        ('unexpected_field', lambda a: a['fields'][0].__setitem__('undocumented', 1)),
        ('unexpected_summary', lambda a: a.__setitem__('undocumented', 1)),
        ('context_count', lambda a: a['cache_cells'][0]['per_context_rms'].pop()),
        ('bool_decision', lambda a: a['cache_cells'][0].__setitem__('aggregate_targets_met', 1)),
        ('bool_exceptions', lambda a: a['cache_cells'][0].__setitem__('contexts_over_target', False)),
        ('false_decision', lambda a: a['cache_cells'][0].__setitem__('aggregate_targets_met', False)),
        ('zero_variance', lambda a: a['cache_cells'][0].__setitem__('native_variance', 0.)),
        ('missing_contract', lambda a: a.pop('coverage_sources_sha256')),
        ('old_schema', lambda a: a.__setitem__('schema', 'matched-clock-analysis-v1')),
    ]


def test_complete_census():
    c = validate_analysis(synthetic_summary())
    assert (len(c['field_keys']), len(c['cache_keys']), len(c['budget_bindings']),
            len(c['runs']), len(c['observation_sha256'])) == (64, 26, 78, 48, 292)


@pytest.mark.parametrize('name,mutate', mutations(), ids=[n for n, _ in mutations()])
def test_reject_incomplete_or_illegal_summary(name, mutate):
    a = synthetic_summary()
    mutate(a)
    with pytest.raises((ValueError, KeyError)):
        validate_analysis(a)


@pytest.mark.parametrize('text', ['{"x": 1, "x": 2}', '{"x": {"seed": 1, "seed": 2}}',
                                  '{"x": NaN}', '{"x": 1e999}'])
def test_json_boundary(tmp_path, text):
    p = tmp_path / 'bad.json'
    p.write_text(text)
    with pytest.raises(ValueError):
        read(p)
