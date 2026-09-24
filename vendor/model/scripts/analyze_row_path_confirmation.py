#!/usr/bin/env python3
"""Complete saved-matrix analysis of finite cross terms and chronological blocking."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict


def transport(a, b):
    """Float64 finite transport; matrix derivative uses the incoming denominator."""
    if a.shape != b.shape or a.ndim < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid matrix pair')
    d = b-a
    e = np.square(a).sum((-2, -1)); en = np.square(b).sum((-2, -1))
    if np.any(e <= 0) or np.any(en <= 0):
        raise ValueError('Row quotient needs nonzero endpoint energy')
    pa = a-a.mean(-2, keepdims=True); pd = d-d.mean(-2, keepdims=True)
    u = np.square(pa).sum((-2, -1))/e
    un = np.square(b-b.mean(-2, keepdims=True)).sum((-2, -1))/en
    de = np.square(d).sum((-2, -1))
    l = 2*(pa*pd).sum((-2,-1))-2*u*(a*d).sum((-2,-1))
    q = np.square(pd).sum((-2,-1))-u*de
    delta = np.sqrt(de/e)
    bound = np.full(delta.shape, np.nan)
    small = delta < 1
    bound[small] = (3*delta[small]**2+delta[small]**3)/(1-delta[small])**2
    return dict(increment=un-u, finite_cross=l/en, quadratic=q/en,
                matrix_derivative=l/e, delta=delta, error_bound=bound,
                relative_displacement_energy=de/e, incoming_energy=e)


def analyze(study):
    p = load_json_strict((study/'protocol.json').read_text())
    expected = {(n,g,s) for n in [4,8,14,24] for g in [1.5,2.] for s in [9163402,9163403]}
    if p['schema'] != 'row-path-confirmation-v1' or len(p['jobs']) != 16 or {(j['heads'],j['control'],j['seed']) for j in p['jobs']} != expected or p['blocks'] != [1,2,4,8]:
        raise ValueError('Incomplete frozen path census')
    ph = sha256(study/'protocol.json')
    checked = {'protocol.json': ph}
    with np.load(Path(p['panel_reuse']['parent'])/'selection.npz') as z:
        order = z['order']
    cells, endpoints = [], []
    errors = dict(finite_identity=0., chronological_composition=0., path_error_budget=0.)
    seconds, peak, one_step_pairs, blocked_pairs = 0., 0, 0, 0
    for job in p['jobs']:
        folder = study/'runs'/job['run_id']; path = folder/'observations.npz'
        m = load_json_strict((folder/'manifest.json').read_text())
        if m['status'] != 'complete' or m['job'] != job or m['protocol_sha256'] != ph or (m['scientific_updates'],m['native_forwards']) != (8,17) or m['artifacts'] != {'observations.npz':sha256(path)}:
            raise ValueError('Incomplete native path')
        for f in [path, folder/'manifest.json']:
            checked[str(f.relative_to(study))] = sha256(f)
        with np.load(path) as z:
            a = z['A'].astype(np.float64); nll = z['target_nll'].astype(float)
            if not np.array_equal(z['document_rows'].ravel(), order[32*job['steps']:32*(job['steps']+8)]) or not np.array_equal(z['optimizer_steps'],np.arange(job['steps'],job['steps']+9)):
                raise ValueError('Wrong source history or optimizer clock')
        if a.shape != (9,8,5,job['heads'],64,64) or nll.shape != (9,8) or not np.isfinite(nll).all():
            raise ValueError('Wrong observation geometry')
        step = transport(a[:-1],a[1:])
        one_step_pairs += step['increment'].size
        # Pathwise sufficient budget for a recorded sequence of matrix derivatives.
        cumulative_error = np.abs(np.cumsum(step['increment']-step['matrix_derivative'],axis=0))
        cumulative_bound = np.cumsum(step['error_bound'],axis=0)
        finite = np.isfinite(cumulative_bound)
        if np.any(finite):
            errors['path_error_budget'] = max(errors['path_error_budget'],float(np.max(cumulative_error[finite]-cumulative_bound[finite])))
        endpoints.append(dict(**job,increment=float(step['increment'].sum(0).mean()),
            accumulated_finite_cross=float(step['finite_cross'].sum(0).mean()),
            accumulated_quadratic=float(step['quadratic'].sum(0).mean()),
            accumulated_matrix_derivative=float(step['matrix_derivative'].sum(0).mean()),
            derivative_error=float(abs((step['increment']-step['matrix_derivative']).sum(0).mean())),
            maximum_inner_derivative_error=float(cumulative_error[-1].max()),
            maximum_one_step_delta=float(step['delta'].max()),
            one_step_bound_applicable=int(np.isfinite(step['error_bound']).sum()),
            one_step_pairs=int(step['increment'].size),
            accumulated_bound_mean=float(cumulative_bound[-1].mean()) if np.isfinite(cumulative_bound[-1]).all() else None,
            target_nll_before=float(nll[0].mean()),target_nll_after=float(nll[-1].mean())))
        for block in [1,2,4,8]:
            starts = np.arange(0,8,block)
            t = transport(a[starts],a[starts+block])
            blocked_pairs += t['increment'].size
            chronological = np.stack([step['finite_cross'][k:k+block].sum(0)+step['quadratic'][k:k+block].sum(0) for k in starts])
            diagonal_energy = np.stack([np.square(np.diff(a[k:k+block+1],axis=0)).sum((0,-2,-1)) for k in starts])
            total_energy = np.square(a[starts+block]-a[starts]).sum((-2,-1))
            signed_cross = (total_energy-diagonal_energy)/t['incoming_energy']
            identity_error = float(np.max(np.abs(t['increment']-t['finite_cross']-t['quadratic'])))
            composition_error = float(np.max(np.abs(t['increment']-chronological)))
            errors['finite_identity'] = max(errors['finite_identity'],identity_error)
            errors['chronological_composition'] = max(errors['chronological_composition'],composition_error)
            cells.append(dict(**job,block=block,aligned_blocks=len(starts),pairs=int(t['increment'].size),
                increment=float(t['increment'].mean()),finite_cross=float(t['finite_cross'].mean()),
                quadratic=float(t['quadratic'].mean()),matrix_derivative=float(t['matrix_derivative'].mean()),
                matrix_derivative_absolute_error=float(np.abs(t['increment']-t['matrix_derivative']).mean()),
                relative_displacement_energy=float(t['relative_displacement_energy'].mean()),
                signed_cross_time_energy=float(signed_cross.mean()),
                maximum_delta=float(t['delta'].max()),identity_error=identity_error,composition_error=composition_error))
        seconds += m['elapsed_seconds']; peak = max(peak,m['peak_allocated_bytes'])
    if any(v > 3e-12 for v in errors.values()):
        raise ValueError('Finite transport check failed')
    return dict(status='passed',schema='row-path-analysis-v1',protocol_sha256=ph,
        analysis_source_sha256=sha256(__file__),checked_sha256=checked,
        cells=cells,endpoints=endpoints,maximum_errors=errors,scientific_updates=128,
        native_forward_calls=272,paths=16,initialization_identities=2,
        one_step_pairs=one_step_pairs,all_aligned_block_pairs=blocked_pairs,
        worker_seconds=seconds,peak_allocated_bytes=peak,
        scope='All frozen paths and aligned scales. Matrix pairs are inner observations; finite errors are deterministic diagnostics on recorded float32 matrices reduced in float64. No new population replicas, parameter derivatives, or fitted successor forecasts.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=analyze(args.study);write_json(args.output,result)
    print({k:v for k,v in result.items() if k not in ['cells','endpoints','checked_sha256']})
