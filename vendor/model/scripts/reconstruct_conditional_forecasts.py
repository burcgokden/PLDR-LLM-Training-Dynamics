"""Re-execute the supplied independent conditional-data reduction without changing its estimands."""
from pathlib import Path
from itertools import product
import hashlib
import json
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
ROOT = Path(args.root).resolve()
OUT = Path(args.output).resolve()
if OUT.exists(): raise FileExistsError(OUT)
study = ROOT/'law-closure-20260910'
checked = {}

def read_json(path):
    path = Path(path)
    content = path.read_bytes()
    checked[str(path)] = hashlib.sha256(content).hexdigest()
    return json.loads(content)

raw_by_stage = {}
panels = 0
max_loss_error = 0.
protocol = read_json(study/'protocol.json')
order = np.random.default_rng(641001).permutation(4194304)
positions = np.empty_like(order)
positions[order] = np.arange(len(order))
for stage in ['', 'state-law-calibration', 'state-law-validation']:
    folder = study/stage
    spec = read_json(folder/'protocol.json')
    cases = {}
    for case in spec['cases']:
        run = folder/'runs'/case['name']
        meta = read_json(run/'results.json')
        with np.load(run/'sampling.npz') as saved:
            blocks, crops = saved['block_ids'].copy(), saved['evaluation_crops'].copy()
        assert blocks.shape == (spec['branches'], 64, 32)
        assert np.all(positions[blocks] >= case['step']*32)
        assert all(len(set(x.flatten())) == 2048 for x in blocks)
        rg = np.random.default_rng(spec['branch_seed']+case['seed']*37+case['heads']*101+case['step'])
        for ids in blocks:
            expected = rg.choice(order[case['step']*32:], 2048, replace=False)
            np.testing.assert_array_equal(ids.ravel(), expected)
        q, risk = [], []
        for branch in meta['records']:
            path = run/branch['raw']
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == branch['sha256']
            checked[str(path)] = digest
            with np.load(path) as saved:
                assert np.isfinite(saved['losses']).all()
                q.append(saved['q'])
                risk.append(saved['nll'].mean(1))
                np.testing.assert_allclose(q[-1][:,0], risk[-1], atol=4e-13, rtol=0)
                if branch['branch'] == 0:
                    for j, logits in enumerate(saved['logits']):
                        z = logits.astype(np.longdouble)
                        normalizer = np.logaddexp.reduce(z, axis=1)
                        nll = normalizer-z[np.arange(len(z)), crops[:,64]]
                        discrepancy = np.max(np.abs(nll-saved['nll'][j]))
                        max_loss_error = max(max_loss_error, float(discrepancy))
                        assert discrepancy < 5e-12
                        panels += 1
        q, risk = np.array(q), np.array(risk)
        np.testing.assert_array_equal(q[:,0], np.repeat(q[:1,0], len(q), axis=0))
        cases[case['name']] = dict(q=q, risk=risk, case=case, start=meta['started_at'])
    raw_by_stage[stage] = cases
    print('Reconstructed', stage or 'development', len(cases), 'states', flush=True)

# Refit every original candidate by augmented least squares (independent of the
# producer's normal-equation solve), then evaluate each untouched identity.
frozen = read_json(study/'frozen-fit.json')
records = []
max_refit_error = 0.
for heads, step, variant in product([4,14], [8192,32768], protocol['analysis']['variants']):
    train = np.concatenate([raw_by_stage[''][f'h{heads}-s{seed}-t{step}']['q'] for seed in [640101,640102]])
    dimension = 29 if variant == 'optimizer_augmented' else 25
    train = train[:,:,:dimension]
    mu = train.reshape(-1, dimension).mean(0)
    floor = 1e-4*np.maximum(1, abs(mu)); floor[0] = .01
    scale = np.maximum(train.reshape(-1,dimension).std(0), floor)
    normalized = (train-mu)/scale
    matrices, offsets, norms = [], [], []
    for j in range(4):
        X, Y = normalized[:,j], normalized[:,j+1]
        Xc = X-X.mean(0)
        if variant in ['identity','translation']:
            A = np.eye(dimension)
        else:
            delta = Y-X
            augmented = np.vstack((Xc, np.eye(dimension)))
            targets = np.vstack((delta-delta.mean(0), np.zeros((dimension,dimension))))
            coefficient = np.linalg.lstsq(augmented, targets, rcond=None)[0]
            u, s, vt = np.linalg.svd(np.eye(dimension)+coefficient)
            A = (u*np.minimum(s,1)) @ vt
        b = np.zeros(dimension) if variant == 'identity' else (Y-X @ A).mean(0)
        matrices.append(A); offsets.append(b); norms.append(np.linalg.norm(A,2))
    prior = frozen['maps'][f'h{heads}-t{step}-{variant}']
    error = max(np.max(abs(np.array(matrices)-prior['matrices'])), np.max(abs(np.array(offsets)-prior['offsets'])))
    max_refit_error = max(max_refit_error, float(error))
    assert error < 3e-9
    for seed in [640103,640104]:
        name = f'h{heads}-s{seed}-t{step}'
        values = raw_by_stage[''][name]['q'][:,:,:dimension]
        u = (values-mu)/scale
        predicted = u[:,0].copy()
        bound = np.zeros(len(values))
        for j,horizon in enumerate([1,4,16,64]):
            residual = u[:,j+1] - (u[:,j] @ matrices[j]+offsets[j])
            bound = norms[j]*bound+np.linalg.norm(residual,axis=1)
            predicted = predicted @ matrices[j]+offsets[j]
            assert np.all(np.linalg.norm(predicted-u[:,j+1],axis=1) <= bound+1e-8)
            risk_error = (predicted[:,0]*scale[0]+mu[0])-values[:,j+1,0]
            records.append(dict(case=name, variant=variant, horizon=horizon,
                mean_error=float(risk_error.mean()), mean_target_met=bool(abs(risk_error.mean())<=.01),
                risk_bound=float(bound.mean()*scale[0])))
summary = []
for variant in protocol['analysis']['variants']:
    rows = [r for r in records if r['variant']==variant]
    last = [r for r in rows if r['horizon']==64]
    summary.append(dict(variant=variant, cells=len(rows), target_met=sum(r['mean_target_met'] for r in rows),
        horizon64_target_met=sum(r['mean_target_met'] for r in last),
        mean_absolute_error=float(np.mean([abs(r['mean_error']) for r in rows])),
        horizon64_mean_absolute_error=float(np.mean([abs(r['mean_error']) for r in last])),
        maximum_absolute_error=max(abs(r['mean_error']) for r in rows),
        horizon64_bounds=[min(r['risk_bound'] for r in last),max(r['risk_bound'] for r in last)]))

design = read_json(study/'state-law-design.json')
frozen_tubes = read_json(study/'state-law-tubes.json')
published = read_json(study/'state-law-results.json')
states = []
for case in design['cases']:
    name = case['name']
    dev = raw_by_stage[''][name]['risk']
    cal = raw_by_stage['state-law-calibration'][name]['risk']
    val = raw_by_stage['state-law-validation'][name]['risk']
    center = dev[:,1:].mean(0)
    scale = np.maximum(np.sqrt(((dev[:,1:]-center)**2).sum(0)/15), .01)
    radius = max(np.max(np.abs(path[1:]-center)/scale) for path in cal)
    lower, upper = center-radius*scale, center+radius*scale
    covered = np.all((val[:,1:]>=lower) & (val[:,1:]<=upper),axis=1)
    intervals = val[:,1:]-val[:,:-1]
    # Population covariance by the pair-difference formula.
    differences = intervals[:,None,:]-intervals[None,:,:]
    covariance = np.einsum('abi,abj->ij',differences,differences)/(2*len(val)**2)
    final_differences = val[:,-1,None]-val[None,:,-1]
    variance = np.mean(final_differences**2)/2
    np.testing.assert_allclose(covariance.sum(),variance,atol=2e-14,rtol=1e-12)
    saved = next(r for r in published['records'] if r['case']==name)
    assert int(covered.sum()) == saved['covered']
    np.testing.assert_allclose(covariance,saved['interval_covariance'],atol=2e-14,rtol=1e-12)
    states.append(dict(case=name, covered=int(covered.sum()), branches=len(val),
        half_width_64=float(radius*scale[-1]),
        mean_error_64=float(center[-1]-val[:,-1].mean()),
        final_variance=float(variance),diagonal_sum=float(np.trace(covariance)),
        signed_cross_sum=float(covariance.sum()-np.trace(covariance)),
        cross_fraction=float((covariance.sum()-np.trace(covariance))/variance)))
assert sum(r['covered'] for r in states)==117


result = dict(status='passed', implementation='Re-execution of the supplied independent reduction method; not an additional independent implementation.', checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), conditional_branches=448, full_vocabulary_panels=panels, maximum_longdouble_NLL_error=max_loss_error, maximum_independent_refit_error=max_refit_error, candidate_outcomes=summary, candidate_cells=records, state_law=states, validation_covered=sum(r['covered'] for r in states), inputs_sha256=checked)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['candidate_cells','inputs_sha256','state_law']},indent=2),flush=True)
