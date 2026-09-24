#!/usr/bin/env python3
"""Independent SciPy/pair-distance reduction of every cache-risk observation cell."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
REPO = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def strict_json(text):
    def reject(value):
        raise ValueError('Nonfinite JSON value: '+str(value))
    def floating(value):
        value=float(value)
        if not np.isfinite(value): reject(value)
        return value
    return json.loads(text, parse_constant=reject, parse_float=floating)


def finite(value, label, shape=None):
    value=np.asarray(value)
    if value.dtype.kind not in 'iuf' or value.size == 0:
        raise ValueError('Invalid numerical dtype or empty array: '+label)
    if shape is not None and value.shape != shape:
        raise ValueError('Invalid shape: '+label)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite array: '+label)
    with np.errstate(over='ignore', invalid='ignore'):
        value=value.astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite float64 conversion: '+label)
    return value


def strict_difference(a,b,label,atol=2e-12,rtol=0.):
    a=finite(a,'reconstructed '+label)
    b=finite(b,'stored '+label,a.shape)
    with np.errstate(over='ignore',invalid='ignore'):
        errors=finite(np.abs(a-b),'discrepancy '+label)
        tolerances=finite(atol+rtol*np.abs(a),'tolerance '+label)
    if np.any(errors>tolerances):
        raise ValueError('Independent reduction differs: '+label)
    return float(errors.max())


def operator_preflight(study,p):
    # These dimensions belong to this producer's frozen native architecture.
    count=0; files=0
    cache_keys={f"C{a['calibration_length']}-P{a['panel']}-M{a['size']}" for a in p['arms']}
    for job in p['jobs']:
        folder=study/'runs'/job['run_id']
        with np.load(folder/'caches.npz',allow_pickle=False) as ca:
            if set(ca.files)!=cache_keys: raise ValueError('Invalid cache member set')
            for key in ca.files:
                finite(ca[key],'cache '+key,(5,3,1,job['heads'],64,64)); count+=1
            files+=1
        for length in p['prefix_lengths']:
            arms=[a for a in p['arms'] if a['length']==length]
            expected={'mean','powers','scatter'} | {a['name']+s for a in arms for s in ['-risk','-bias']}
            with np.load(folder/f'operators-L{length}.npz',allow_pickle=False) as op:
                if set(op.files)!=expected: raise ValueError('Invalid operator member set')
                for key in op.files:
                    shape=(5,job['heads'],64,64) if key=='mean' else (5,p['contexts']) if key=='powers' else (5,)
                    value=finite(op[key],'operator '+key,shape); count+=1
                    if key!='mean' and np.any(value<0): raise ValueError('Negative squared statistic: '+key)
                files+=1
    return dict(files=files, arrays=count, nonfinite_arrays=0)


def verify(study, analysis, output):
    if output.exists(): raise FileExistsError(output)
    p = strict_json((study/'protocol.json').read_text()); result = strict_json(analysis.read_text())
    if result['protocol_sha256'] != digest(study/'protocol.json') or result['status'] != 'passed':
        raise ValueError('Foreign or incomplete reduction')
    if result.get('schema') != 'cache-risk-analysis-v2' or result['analysis_source_sha256'] != digest(REPO/'scripts/analyze_cache_risk_study.py'):
        raise ValueError('Current primary reduction required')
    sources = ['scripts/analyze_cache_risk_study.py', 'scripts/cache_risk_validation.py', 'scripts/numerical_validation.py', 'scripts/analyze_operator_cache.py', 'src/model_rg/provenance.py']
    if result.get('analysis_sources_sha256') != {n: digest(REPO/n) for n in sources}:
        raise ValueError('Changed analysis dependencies')
    if p['schema'] != 'cache-risk-v1' or p['status'] != 'frozen_before_acquisition' or p['contexts'] != 64 or p['prefix_lengths'] != [32, 64, 128]:
        raise ValueError('Invalid frozen geometry')
    for name, expected in result['checked_sha256'].items():
        path = (study/name).resolve()
        if not path.is_relative_to(study) or digest(path) != expected: raise ValueError('Changed or escaping input '+name)
    operator_scan=operator_preflight(study,p)
    with np.load(study/'inputs.npz',allow_pickle=False) as a:
        blocks = a['blocks']
        if blocks.shape != (128, 129) or blocks.dtype.kind not in 'iu' or np.any(blocks < 0) or np.any(blocks >= p['vocabulary']):
            raise ValueError('Invalid token blocks')
        targets = blocks[64:]
    maximum = 0.; opmax = 0.; counts = {k: 0 for k in p['expected_calls']}
    def compare(a, b):
        nonlocal maximum
        error = strict_difference(a,b,'predictive',atol=4e-12,rtol=2e-10)
        maximum = max(maximum,error)
    for job in p['jobs']:
        relative = 'runs/'+job['run_id']+'/manifest.json'
        m = strict_json((study/relative).read_text())
        if m['status'] != 'complete' or m['job'] != job or m['protocol_sha256'] != result['protocol_sha256'] or m['training_updates'] != 0:
            raise ValueError('Invalid acquisition manifest')
        if result['checked_sha256'].get(relative) != digest(study/relative):
            raise ValueError('Unbound acquisition manifest')
        for name, expected_hash in m['artifacts'].items():
            relative = 'runs/'+job['run_id']+'/'+name
            if result['checked_sha256'].get(relative) != expected_hash or digest(study/relative) != expected_hash:
                raise ValueError('Unbound observation')
        for k in counts: counts[k] += m['calls'][k]
    if counts != p['expected_calls'] or counts != result['calls']: raise ValueError('Call ledger mismatch')
    expected = {(h, g, a['name']) for h in [8, 24] for g in [0, 1.5] for a in p['arms']}
    actual = {(c['heads'], c['control'], c['name']) for c in result['cells']}
    if actual != expected or len(result['cells']) != len(expected): raise ValueError('Incomplete cell grid')
    for cell in result['cells']:
        jobs = sorted([j for j in p['jobs'] if (j['heads'], j['control']) == (cell['heads'], cell['control'])], key=lambda j: j['seed'])
        if len(jobs) != 6 or len({j['seed'] for j in jobs}) != 6:
            raise ValueError('Incomplete replica grid')
        arm = next(a for a in p['arms'] if a['name'] == cell['name'])
        if any(cell[k] != value for k, value in arm.items()):
            raise ValueError('Changed arm coordinates')
        x = []; y = []; kl = []; nll = []; nllq = []
        risks=[]; scatters=[]; biases=[]
        for job in jobs:
            folder = study/'runs'/job['run_id']; length = cell['length']
            lp = finite(np.load(folder/f'native-L{length}.npy', allow_pickle=False), 'native logits', (64, p['vocabulary']))
            lq = finite(np.load(folder/(cell['name']+'.npy'), allow_pickle=False), 'cached logits', (64, p['vocabulary']))
            lp -= logsumexp(lp, axis=-1, keepdims=True)
            lq -= logsumexp(lq, axis=-1, keepdims=True)
            x.append(2*np.exp(lp/2)); y.append(2*np.exp(lq/2)); kl.append(np.sum(np.exp(lp)*(lp-lq), axis=-1))
            nll.append(-lp[np.arange(64), targets[:, length]]); nllq.append(-lq[np.arange(64), targets[:, length]])
            with np.load(folder/f'operators-L{length}.npz') as op, np.load(folder/'caches.npz') as ca:
                mean = finite(op['mean'], 'operator mean'); cg = finite(ca[f"C{cell['calibration_length']}-P{cell['panel']}-M{cell['size']}"][:, 2, 0], 'cache G')
                scatter = op['powers'].mean(1)-np.mean(mean**2, axis=(1, 2, 3))
                risk = op['powers'].mean(1)-2*np.mean(mean*cg, axis=(1, 2, 3))+np.mean(cg**2, axis=(1, 2, 3))
                bias = np.mean((mean-cg)**2,axis=(1,2,3))
                for label, actual_value, stored_value in [
                    ('scatter',scatter,op['scatter']),
                    ('risk',risk,op[cell['name']+'-risk']),
                    ('displacement',bias,op[cell['name']+'-bias']),
                    ('risk identity',risk,scatter+bias)]:
                    err=strict_difference(actual_value,stored_value,label)
                    opmax=max(opmax,err)
                risks.append(risk); scatters.append(scatter); biases.append(bias)
        x, y = np.stack(x), np.stack(y); e = x-y; s = len(jobs)
        def pairvar(a):
            return sum(np.sum((a[i]-a[j])**2, axis=-1) for i in range(s) for j in range(i))/(s*(s-1))
        vx, vy, ve = pairvar(x), pairvar(y), pairvar(e)
        if np.any(finite(vx, 'native variance') <= 0):
            raise ValueError('Relative context risk undefined')
        risk = np.sqrt(ve/vx); uncentered = np.mean(np.sum(e*e, axis=-1), axis=0)
        cross = (vx-vy-ve)/2
        independent = dict(native_variance=vx.mean(), cached_variance=vy.mean(), residual_variance=ve.mean(),
            relative_centered_rms=np.sqrt(ve.sum()/vx.sum()), retained_variance_fraction=vy.sum()/vx.sum(),
            signed_cross_covariance=cross.mean(), uncentered_energy=uncentered.mean(),
            mean_bias_energy=np.sum(e.mean(0)**2, axis=-1).mean(),
            mean_kl=np.mean(kl), maximum_kl=np.max(kl), native_nll=np.mean(nll), cached_nll=np.mean(nllq),
            mean_nll_change=np.mean(np.array(nllq)-nll), per_seed_mean_kl=np.mean(kl, axis=1),
            per_context_rms=risk, maximum_context_rms=risk.max(), contexts_over_target=np.sum(risk>.25),
            per_context_native_variance=vx, per_context_residual_variance=ve,
            per_seed_context_kl=kl, per_seed_context_nll_change=np.array(nllq)-nll)
        independent.update(operator_risk_by_seed_layer=risks,
            operator_scatter_by_seed_layer=scatters,operator_displacement_by_seed_layer=biases,
            mean_operator_risk=np.mean(risks),mean_operator_scatter=np.mean(scatters),
            mean_operator_displacement=np.mean(biases))
        for key, value in independent.items(): compare(value, cell[key])
        if cell['targets_met'] != bool(independent['relative_centered_rms'] <= .25 and independent['mean_kl'] <= .03):
            raise ValueError('Target classification differs')
    answer = dict(status='passed', schema='cache-risk-independent-v2', operator_scan=operator_scan, cells=len(result['cells']),
        maximum_predictive_difference=maximum, maximum_operator_moment_difference=opmax,
        calls=counts, training_updates=0, analysis_sha256=digest(analysis),
        protocol_sha256=digest(study/'protocol.json'), verifier_sha256=digest(__file__),
        analysis_sources_sha256=result['analysis_sources_sha256'], checked_sha256=result['checked_sha256'],
        methods='Independent SciPy log-sum-exp, pair-distance variances and raw operator sufficient moments; no production reducer import.')
    payload = json.dumps(answer, indent=2, allow_nan=False)+'\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.verification-', dir=output.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(payload)
        os.link(name, output)
    finally:
        os.unlink(name)
    print(json.dumps({k: v for k, v in answer.items() if k != 'checked_sha256'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True); parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); verify(a.study.resolve(), a.analysis.resolve(), a.output.resolve())
