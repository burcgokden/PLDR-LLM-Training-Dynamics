"""Strict publication contract: strict redundant aggregates of independently checked cells.

This module does not import the production analysis reducer. The verifier also
compares its summaries against separately reconstructed direct-pair cells.
"""
from collections import defaultdict
from numerical_validation import finite_scalar, discrepancy

FIELDS = ('relative_centered_rms', 'relative_uncentered_rms',
          'retained_variance_fraction', 'mean_emission_kl',
          'sufficient_kl_relative_bound')
COUNTS = ('scale_cells', 'adjacent_scale_cells', 'observed_paths',
          'replicas_per_cell', 'contexts', 'path_context_kl_chains')


def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Invalid integer ' + label)
    return value


def summaries(cells, plan):
    groups = defaultdict(list)
    seen = set()
    for row in cells:
        key = (integer(row['heads'], 'heads', 1),
               finite_scalar(row['control'], 'control'),
               integer(row['time'], 'time'),
               integer(row['retained_tokens'], 'retained_tokens', 1))
        if key in seen:
            raise ValueError('Duplicate cell')
        seen.add(key)
        if type(row['target_met']) is not bool:
            raise ValueError('Invalid target Boolean')
        for field in FIELDS:
            finite_scalar(row[field], field)
        if row['target_met'] != (row['relative_centered_rms'] <= plan['target']):
            raise ValueError('Target decision differs')
        groups[key[2:]].append(row)
    expected = {(t, k) for t in plan['times'] for k in plan['sizes']}
    if len(expected) != len(plan['times']) * len(plan['sizes']) or set(groups) != expected:
        raise ValueError('Wrong summary group coverage')
    result = []
    for t in plan['times']:
        for k in plan['sizes']:
            rows = groups[t, k]
            item = dict(time=t, retained_tokens=k, cells=len(rows),
                        target_met=sum(r['target_met'] for r in rows))
            for field in FIELDS:
                item[field + '_range'] = [min(r[field] for r in rows),
                                         max(r[field] for r in rows)]
            result.append(item)
    return result


def compare_summaries(claimed, actual, atol=0.):
    if not isinstance(claimed, list) or len(claimed) != len(actual):
        raise ValueError('Wrong summary length')
    keys = set()
    by_key = {}
    for row in claimed:
        key = (integer(row['time'], 'summary time'),
               integer(row['retained_tokens'], 'summary tokens', 1))
        if key in keys:
            raise ValueError('Duplicate summary group')
        keys.add(key)
        by_key[key] = row
    if keys != {(r['time'], r['retained_tokens']) for r in actual}:
        raise ValueError('Wrong summary groups')
    for truth in actual:
        row = by_key[truth['time'], truth['retained_tokens']]
        if set(row) != set(truth):
            raise ValueError('Unexpected or missing summary field')
        for field in ('cells', 'target_met'):
            if integer(row[field], 'summary ' + field) != truth[field]:
                raise ValueError('Summary integer differs: ' + field)
        for field in FIELDS:
            values = row[field + '_range']
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError('Invalid summary range ' + field)
            for a, b in zip(truth[field + '_range'], values):
                discrepancy(a, b, 'summary ' + field, atol=atol)


def validate(record, plan):
    canonical = summaries(record['cells'], plan)
    compare_summaries(record['summary'], canonical)
    rows, edges = record['cells'], record['edges']
    for field in COUNTS:
        integer(record[field], field, 1)
    replicas = record['replicas_per_cell']
    contexts = record['contexts']
    pairs = {(r['heads'], r['control']) for r in rows}
    truth = dict(scale_cells=len(rows), adjacent_scale_cells=len(edges),
                 observed_paths=len(pairs) * replicas, replicas_per_cell=6,
                 contexts=integer(plan['contexts'], 'protocol contexts', 1),
                 path_context_kl_chains=len(edges) * replicas * contexts)
    for field in COUNTS:
        if record[field] != truth[field]:
            raise ValueError('Aggregate count differs: ' + field)
    for row in rows + edges:
        if integer(row['contexts'], 'cell contexts', 1) != contexts:
            raise ValueError('Cell context denominator differs')
        ids = row['seed_ids']
        if len(ids) != replicas or len(set(ids)) != replicas:
            raise ValueError('Cell replica identities differ')
        for seed in ids:
            integer(seed, 'seed')
    errors = [finite_scalar(r['maximum_algebra_error'], 'cell error') for r in rows]
    for row in edges:
        if integer(row['path_context_chains'], 'edge chain count', 1) != replicas * contexts:
            raise ValueError('Edge chain denominator differs')
        errors += [finite_scalar(row[k], k) for k in ('kl_chain_max_error', 'absorption_max_error')]
    if any(e < 0 for e in errors):
        raise ValueError('Negative algebra discrepancy')
    discrepancy(max(errors), record['maximum_algebra_error'], 'aggregate algebra maximum', atol=0.)
    return canonical
