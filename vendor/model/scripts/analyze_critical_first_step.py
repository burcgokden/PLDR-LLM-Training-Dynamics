#!/usr/bin/env python3
"""Independently reduce the first shared force and its native Adam displacement."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json

REPO=Path(__file__).resolve().parents[1]


def variance(x):
    return float(np.mean((x-x.mean(0,keepdims=True))**2)*len(x)/(len(x)-1))


def pairs(x):
    total=0.
    for i in range(len(x)):
        for j in range(i):total+=float(np.mean((x[i]-x[j])**2))
    return total/(len(x)*(len(x)-1))


def analyze(root,output):
    if output.exists():raise FileExistsError(output)
    p=json.loads((root/'protocol.json').read_text())
    if p['schema']!='critical-first-step-replay-v1' or p['role']!='qualification' or p['scientific_updates']!=0:
        raise ValueError('Wrong reconstruction role')
    checked={str(root/'protocol.json'):sha256(root/'protocol.json')}
    for name,digest in p['source_sha256'].items():
        if sha256(REPO/name)!=digest:raise ValueError('Changed acquisition source')
        checked[str(REPO/name)]=digest
    for name,digest in p['input_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Changed parent evidence')
        checked[name]=digest
    groups=defaultdict(list);updates=0
    for job in p['jobs']:
        folder=root/'runs'/job['run_id'];m=json.loads((folder/'manifest.json').read_text())
        if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=sha256(root/'protocol.json'):
            raise ValueError('Incomplete or unbound first-step reconstruction')
        if m['role']!='qualification' or m['scientific_updates']!=0 or m['replay_updates']!=64:
            raise ValueError('Reconstruction updates were relabeled')
        raw=folder/'first-step.npz'
        if sha256(raw)!=m['raw_sha256']:raise ValueError('Changed first-step arrays')
        checked[str(raw)]=sha256(raw);checked[str(folder/'manifest.json')]=sha256(folder/'manifest.json')
        native_manifest=folder/'native-replay/manifest.json'
        if sha256(native_manifest)!=m['native_manifest_sha256']:raise ValueError('Changed native replay manifest')
        checked[str(native_manifest)]=sha256(native_manifest)
        replay=json.loads(native_manifest.read_text())
        for name,digest in replay['artifacts'].items():
            path=native_manifest.parent/name
            if sha256(path)!=digest:raise ValueError('Changed native replay artifact')
            checked[str(path)]=digest
        with np.load(native_manifest.parent/'observations.npz') as current,np.load(Path(p['study'])/'runs'/job['run_id']/'observations.npz') as original:
            for key in ['blocks','training_loss','gradient_norm']:
                np.testing.assert_array_equal(current[key],original[key][:64])
            for key in ['heads','nll_path']:
                positions=[int(np.flatnonzero(original['steps']==t)[0]) for t in current['steps']]
                np.testing.assert_array_equal(current[key],original[key][positions])
            np.testing.assert_array_equal(current['logits_0'],original['logits_0'])
        groups[(job['environment'],job['heads'])].append((job['seed'],m,raw))
        updates+=m['replay_updates']
    parent=json.loads((Path(p['study'])/'protocol.json').read_text());seeds=sorted(parent['design']['seeds'])
    cells=[];maximum_pair_error=maximum_formula_error=0.;common_initial=None;common_names=None
    for (e,n),rows in sorted(groups.items()):
        rows.sort(key=lambda r:r[0])
        if [s for s,_,_ in rows]!=seeds:raise ValueError('The full declared seed ensemble is required')
        files=[np.load(path) for _,_,path in rows]
        try:
            data={key:np.stack([a[key].astype(float) for a in files]) for key in files[0].files}
            initial=data['initial_shared'];after=data['after_shared'];h=data['clipped_gradient']
            for value in initial:np.testing.assert_array_equal(value,initial[0])
            names=rows[0][1]['first_step']['names']
            if any(m['first_step']['names']!=names for _,m,_ in rows):raise ValueError('Shared coordinate identities differ')
            if common_initial is None:common_initial=initial[0].copy();common_names=names
            np.testing.assert_array_equal(initial[0],common_initial)
            if names!=common_names:raise ValueError('The shared parameter dimension changed with width')
            metadata=rows[0][1]['first_step'];eta=metadata['rate'];decay=metadata['decay'];eps=metadata['epsilon']
            if any(any(m['first_step'][key]!=metadata[key] for key in ['rate','decay','epsilon']) for _,m,_ in rows):
                raise ValueError('Optimizer convention differs within a cell')
            force=h/(np.abs(h)+eps);sign=np.sign(h);delta=after-initial
            prediction=initial*(1-eta*decay)-eta*force
            normalized=np.abs(after-prediction)/(1+np.abs(initial))
            error=float(normalized.max());maximum_formula_error=max(maximum_formula_error,error)
            if error>8*np.finfo(np.float32).eps:raise ValueError('First Adam update exceeds the declared arithmetic envelope')
            sign_error=float(np.mean(np.abs(force-sign)))
            values={key:variance(x) for key,x in [('raw_gradient',data['raw_gradient']),('clipped_gradient',h),
                ('normalized_force',force),('sign_force',sign),('actual_shared_step',delta)]}
            for key,x in [('normalized_force',force),('actual_shared_step',delta)]:
                residual=abs(pairs(x)-values[key]);maximum_pair_error=max(maximum_pair_error,residual)
                if residual>2e-12*(1+abs(values[key])):raise ValueError('Independent shared variance reconstruction differs')
            bound=4*len(seeds)/(len(seeds)-1)*sign_error
            if abs(values['normalized_force']-values['sign_force'])>bound+2e-12:
                raise ValueError('Finite empirical sign-force bound failed')
            remainder=delta+eta*(decay*initial+force)
            remainder_variance=variance(remainder)
            ideal_variance=eta**2*values['normalized_force']
            covariance_error=abs(values['actual_shared_step']-ideal_variance)
            covariance_bound=2*np.sqrt(ideal_variance*remainder_variance)+remainder_variance
            tolerance=64*np.finfo(float).eps*(values['actual_shared_step']+ideal_variance+covariance_bound)
            if covariance_error>covariance_bound+tolerance:
                raise ValueError('The measured arithmetic covariance budget failed')
            cells.append(dict(environment=e,heads=n,seeds=seeds,shared_parameter_count=initial.shape[1],
                rate=eta,epsilon=eps,per_coordinate_variances=values,
                shared_step_susceptibility=n*values['actual_shared_step'],
                step_variance_divided_by_rate_squared=values['actual_shared_step']/eta**2,
                arithmetic_remainder_variance=remainder_variance,
                arithmetic_covariance_error=covariance_error,arithmetic_covariance_bound=float(covariance_bound),
                arithmetic_rms_relative_to_ideal_fluctuation=float(np.sqrt(remainder_variance/ideal_variance)) if ideal_variance else None,
                normalized_force_to_sign_l1=sign_error,empirical_variance_difference_bound=bound,
                fraction_clipped_magnitude_over_100_epsilon=float(np.mean(np.abs(h)>100*eps)),
                maximum_normalized_first_update_error=error,
                global_gradient_norm_by_seed=[m['first_step']['global_gradient_norm'] for _,m,_ in rows],
                analysis_role='exploratory_native_mechanism'))
        finally:
            for f in files:f.close()
    if updates!=p['replay_updates']:raise ValueError('Replay cost inventory differs')
    result=dict(schema='critical-first-step-analysis-v1',status='complete',study=str(root),cells=cells,
        scientific_updates=0,replay_updates=updates,reconstructed_paths=len(p['jobs']),maximum_pairwise_variance_error=maximum_pair_error,
        maximum_normalized_first_update_error=maximum_formula_error,all_recorded_prefixes_bitwise=True,
        checked_sha256=checked,analyzer_sha256=sha256(__file__),
        units='Variance averaged over the fixed-dimensional shared residual metric parameters; the common-sector susceptibility multiplies this quantity by the number of heads.',
        scope='First-update normalization and reconstruction of already counted trajectories. Finite common-state fluctuations do not establish a critical limit or an intrinsic relaxation exponent.')
    write_json(output,result)
    print(json.dumps({key:result[key] for key in ['status','scientific_updates','replay_updates','maximum_normalized_first_update_error']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    analyze(a.study.resolve(),a.output.resolve())
