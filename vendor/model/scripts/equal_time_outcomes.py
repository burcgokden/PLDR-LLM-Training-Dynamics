"""Machine-readable completeness contract for the declared physical-block outcomes.

The acquisition protocol stated its secondary in prose. The incoming-energy
normalization was made explicit during post-acquisition reconstruction; this
contract does not retrospectively claim a frozen mathematical convention.
"""
import math

SCHEMA = 'equal-time-outcomes-v1'
CELL_FIELDS = ('increment', 'finite_cross', 'quadratic', 'matrix_derivative',
               'matrix_derivative_absolute_error', 'step_squared_energy',
               'block_squared_energy', 'signed_cross_time_energy')
ENDPOINT_FIELDS = ('increment', 'finite_cross', 'quadratic', 'matrix_derivative',
                   'derivative_error', 'target_nll_before', 'target_nll_after')
OUTCOMES = {
    'finite_row_increment': ['increment'],
    'chronological_composition': ['composition_error', 'energy_composition_error'],
    'matrix_directional_derivative_error': ['matrix_derivative_absolute_error'],
    'finite_cross_and_quadratic': ['finite_cross', 'quadratic'],
    'signed_cross_time_matrix_energy': ['step_squared_energy', 'block_squared_energy', 'signed_cross_time_energy'],
    'target_nll': ['target_nll_before', 'target_nll_after'],
}
NORMALIZATION = 'Per coordinate and aligned block, divide by incoming Frobenius energy before averaging over all fixed contexts, layers, heads and aligned blocks.'


def validate_outcomes(a, protocol):
    if 'signed cross-time matrix energy' not in protocol['secondary']:
        raise ValueError('Declared protocol outcome is missing')
    if a.get('outcome_contract') != dict(schema=SCHEMA, outcomes=OUTCOMES,
            normalization=NORMALIZATION, convention_status='explicit_post_acquisition'):
        raise ValueError('Incomplete declared outcome contract')
    jobs={j['run_id']:j for j in protocol['jobs']}
    if len(jobs)!=len(protocol['jobs']):raise ValueError('Duplicate protocol identity')
    for group, fields in [('endpoints', ENDPOINT_FIELDS), ('cells', CELL_FIELDS)]:
        rows=a[group]; seen=set()
        for row in rows:
            key=(row['run_id'],row['factor']) if group=='cells' else row['run_id']
            if key in seen:raise ValueError('Duplicate outcome identity')
            seen.add(key)
            if row['run_id'] not in jobs or any(row.get(k)!=v or isinstance(row.get(k),bool) for k,v in jobs[row['run_id']].items()):
                raise ValueError('Foreign outcome identity')
            for field in fields:
                value=row.get(field)
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError('Missing or nonfinite declared outcome: '+field)
            if group=='cells':
                if type(row['factor']) is not int:raise ValueError('Physical block factor must be an integer')
                s,b,x=(row[k] for k in CELL_FIELDS[-3:])
                tol=3e-12*max(1,abs(s),abs(b),abs(x))
                m=row['heads']*row['factor']
                if s<0 or b<0 or abs(b-s-x)>tol or x < -s-tol or x>(m-1)*s+tol:
                    raise ValueError('Temporal energy domain, identity or bounds failed')
        want={(j,f) for j in jobs for f in protocol['physical_block_fractions']} if group=='cells' else set(jobs)
        if seen!=want:raise ValueError('Missing declared outcome cells')
    return dict(schema=SCHEMA, status='passed', paths=len(jobs), cells=len(a['cells']),
                outcomes=OUTCOMES, normalization=NORMALIZATION,
                convention_status='explicit_post_acquisition')
