#!/usr/bin/env python3
"""Reduce native shared paths into exact mean and block-covariance budgets.

Replica covariance is conditional on one fixed source order. The covariance
of recorded block increments is not a claim about fresh minibatch noise,
stationarity, diffusion, or a fluctuation-dissipation identity.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from model_rg.provenance import sha256,write_json

REPO=Path(__file__).resolve().parents[1]


def trace_variance(x):
    return float(np.sum((x-x.mean(0,keepdims=True))**2)/((len(x)-1)*x.shape[-1]))


def pair_variance(x):
    return float(sum(np.mean((x[i]-x[j])**2) for i in range(len(x)) for j in range(i))/(len(x)*(len(x)-1)))


def path_budget(x,steps,rate,decay):
    """x has shape [complete initialization, recorded time, shared coordinate]."""
    x=np.asarray(x,dtype=np.float64);steps=np.asarray(steps,dtype=np.int64)
    if x.ndim!=3 or len(x)<2 or x.shape[1]!=len(steps) or steps[0]!=0 or np.any(np.diff(steps)<=0):
        raise ValueError('Invalid complete shared path ensemble')
    if not np.isfinite(x).all() or not 0<=rate*decay<1:raise ValueError('Invalid shared path values or decay')
    for value in x[:,0]:np.testing.assert_array_equal(value,x[0,0])
    sample=len(x);dimension=x.shape[-1];logq=float(np.log1p(-rate*decay))
    cells=[];maximum_pair_error=maximum_mean_budget_error=0.
    for k,t in enumerate(steps):
        displacement=x[:,k]-np.exp(logq*t)*x[:,0]
        variance=trace_variance(displacement);raw=float(np.mean(displacement**2))
        mean_square=float(np.mean(displacement.mean(0)**2))
        unbiased_mean_square=mean_square-variance/sample
        error=abs(pair_variance(displacement)-variance)
        maximum_pair_error=max(maximum_pair_error,error)
        budget_error=abs(unbiased_mean_square+variance-raw)
        maximum_mean_budget_error=max(maximum_mean_budget_error,budget_error)
        if error>2e-11*max(variance,1e-30) or budget_error>2e-11*max(raw,1e-30):
            raise ValueError('Independent shared path variance or mean budget differs')
        cells.append(dict(step=int(t),per_coordinate_variance=variance,
            mean_squared_decay_corrected_displacement=raw,
            squared_empirical_mean_displacement=mean_square,
            unbiased_squared_mean_displacement=unbiased_mean_square,
            variance_divided_by_rate_squared=variance/rate**2 if rate else None,
            variance_divided_by_rate_squared_time=variance/(rate**2*t) if rate and t else None))
    # Blocks have the predeclared 256-update cadence. The separate first update
    # is excluded from this partition so that block widths are not mixed.
    indices=np.flatnonzero((steps%256)==0)
    if len(indices)<2 or steps[indices[-1]]!=steps[-1]:raise ValueError('Missing complete fixed-cadence blocks')
    t=steps[indices];a=x[:,indices]
    block_decay=np.exp(logq*np.diff(t))
    increments=a[:,1:]-block_decay[None,:,None]*a[:,:-1]
    centered=increments-increments.mean(0,keepdims=True)
    gram=sum(z@z.T for z in centered)/((sample-1)*dimension)
    diagonal=np.diag(gram);eigenvalues=np.linalg.eigvalsh(gram)
    if eigenvalues[0]<-2e-11*max(float(eigenvalues[-1]),1e-30):raise ValueError('Block covariance is not positive semidefinite')
    budgets=[];maximum_block_error=maximum_telescope_error=0.
    for k,final in enumerate(t[1:],1):
        weights=np.exp(logq*(final-t[1:k+1]))
        direct=next(c['per_coordinate_variance'] for c in cells if c['step']==final)
        covariance=float(weights@gram[:k,:k]@weights)
        diag=float(np.sum(weights**2*diagonal[:k]));offdiag=covariance-diag
        reconstructed=np.einsum('k,skd->sd',weights,increments[:,:k],optimize=True)
        observed=a[:,k]-np.exp(logq*final)*a[:,0]
        telescope=float(np.max(np.abs(reconstructed-observed)))
        error=abs(covariance-direct);maximum_block_error=max(maximum_block_error,error)
        maximum_telescope_error=max(maximum_telescope_error,telescope)
        cancellation_scale=float(np.sum(np.abs(weights[:,None]*gram[:k,:k]*weights[None,:])))
        if error>3e-10*max(direct,cancellation_scale,1e-30) or telescope>3e-12*max(1.,float(np.max(np.abs(a)))):
            raise ValueError('Shared block covariance fails independent endpoint reconstruction')
        budgets.append(dict(step=int(final),measured_variance=direct,reconstructed_variance=covariance,
            weighted_diagonal_contribution=diag,signed_off_diagonal_contribution=offdiag,
            off_diagonal_fraction=offdiag/direct if direct else None,
            maximum_coordinate_telescope_error=telescope))
    denominator=np.sqrt(np.maximum(diagonal[:,None]*diagonal[None,:],0))
    correlations=np.divide(gram,denominator,out=np.zeros_like(gram),where=denominator>0)
    return dict(path_statistics=cells,block_times=t.tolist(),block_covariance_matrix=gram.tolist(),
        block_correlation_matrix=correlations.tolist(),block_budgets=budgets,
        minimum_block_covariance_eigenvalue=float(eigenvalues[0]),
        maximum_pairwise_variance_error=maximum_pair_error,
        maximum_mean_budget_error=maximum_mean_budget_error,
        maximum_block_variance_error=maximum_block_error,
        maximum_coordinate_telescope_error=maximum_telescope_error)


def first_force_budget(initial,after,raw,clipped,rate,decay,epsilon):
    force=clipped/(np.abs(clipped)+epsilon);sign=np.sign(clipped)
    delta=after-initial;remainder=delta+rate*(decay*initial+force)
    vf=trace_variance(force);actual=trace_variance(delta);ve=trace_variance(remainder)
    ideal=rate**2*vf;bound=2*np.sqrt(ideal*ve)+ve
    if abs(actual-ideal)>bound+64*np.finfo(float).eps*(actual+ideal+bound):
        raise ValueError('First-force arithmetic covariance bound failed')
    d=float(np.mean(np.abs(force-sign)));sign_bound=4*len(force)/(len(force)-1)*d
    if abs(vf-trace_variance(sign))>sign_bound+2e-12:raise ValueError('First-force sign comparison failed')
    return dict(raw_gradient_variance=trace_variance(raw),clipped_gradient_variance=trace_variance(clipped),
        normalized_force_variance=vf,sign_force_variance=trace_variance(sign),actual_step_variance=actual,
        ideal_step_variance=ideal,arithmetic_remainder_variance=ve,arithmetic_covariance_bound=float(bound),
        normalized_force_to_sign_l1=d,empirical_sign_variance_bound=sign_bound,
        fraction_clipped_magnitude_over_100_epsilon=float(np.mean(np.abs(clipped)>100*epsilon)))


def analyze(study,output):
    if output.exists():raise FileExistsError(output)
    p=json.loads((study/'protocol.json').read_text());ph=sha256(study/'protocol.json')
    if p.get('shared_paths',{}).get('schema')!='native-shared-parameter-path-v1':raise ValueError('Missing recorded shared paths')
    checked={str(study/'protocol.json'):ph};groups=defaultdict(list)
    for name,digest in p['source_sha256'].items():
        if sha256(REPO/name)!=digest:raise ValueError('Changed acquisition source')
        checked[str(REPO/name)]=digest
    q=json.loads((study/'qualification/verification.json').read_text())
    if q['status']!='passed' or not q.get('uninstrumented_vs_instrumented_bitwise') or q['protocol_sha256']!=ph:
        raise ValueError('Missing passive-observer native qualification')
    for name,digest in q['checked_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Changed native qualification')
        checked[name]=digest
    checked[str(study/'qualification/verification.json')]=sha256(study/'qualification/verification.json')
    excluded=[];updates=0
    for job in p['jobs']:
        folder=study/'runs'/job['run_id'];m=json.loads((folder/'manifest.json').read_text())
        if m['job']!=job or m['protocol_sha256']!=ph or m['role']!='scientific' or not m.get('shared_path_recorded'):
            raise ValueError('Unbound native shared path')
        checked[str(folder/'manifest.json')]=sha256(folder/'manifest.json')
        for name,digest in m['artifacts'].items():
            if sha256(folder/name)!=digest:raise ValueError('Changed native shared artifact')
            checked[str(folder/name)]=digest
        updates+=m['scientific_updates']
        if m['status']!='complete':excluded.append(job['run_id']);continue
        groups[(job['environment'],job['heads'],job['control'])].append((job['seed'],folder,m))
    cells=[];unresolved=[];initial_shared=None;coordinate_names=None;gradient_identities={}
    for (e,n,g),rows in sorted(groups.items()):
        rows.sort()
        if [s for s,_,_ in rows]!=sorted(p['design']['seeds']):
            unresolved.append(dict(environment=e,heads=n,control=g,reason='Incomplete initialization ensemble'));continue
        parameters=[];raws=[];clipped=[];metadata=None;steps=None
        for seed,folder,m in rows:
            meta=json.loads((folder/'shared-metadata.json').read_text())
            if meta['protocol_sha256']!=ph or meta['job']!=m['job'] or meta['role']!='scientific':raise ValueError('Unbound shared metadata')
            times=np.load(folder/'shared-steps.npy');x=np.load(folder/'shared-parameters.npy',mmap_mode='r')
            if x.shape!=(len(times),meta['shared_parameter_count']) or x.dtype!=np.float32 or list(times)!=meta['steps']:
                raise ValueError('Shared shape or recorded cadence differs')
            expected=sorted({0,1,m['completed_steps'],*range(256,m['completed_steps']+1,256)})
            np.testing.assert_array_equal(times,expected)
            if metadata is None:metadata=meta;steps=times
            for key in ['names','shapes','rate','weight_decay','epsilon','betas']:
                if meta[key]!=metadata[key]:raise ValueError('Shared coordinate or optimizer conventions differ')
            if initial_shared is None:initial_shared=np.array(x[0]);coordinate_names=meta['names']
            np.testing.assert_array_equal(x[0],initial_shared)
            if coordinate_names!=meta['names']:raise ValueError('Shared coordinate dimension changed with width')
            state=torch.load(folder/'final-state.pt',map_location='cpu',mmap=True,weights_only=False)
            endpoint=np.concatenate([state['model'][name].reshape(-1).numpy() for name in meta['names']])
            np.testing.assert_array_equal(x[-1],endpoint);del state,endpoint
            parameters.append(np.asarray(x,dtype=float))
            with np.load(folder/'first-gradient.npz') as arrays:
                current={name:arrays[name] for name in ['raw_gradient','clipped_gradient']}
                key=(e,n,seed)
                if key not in gradient_identities:gradient_identities[key]=current
                else:
                    for name in current:np.testing.assert_array_equal(current[name],gradient_identities[key][name])
                raws.append(current['raw_gradient'].astype(float));clipped.append(current['clipped_gradient'].astype(float))
        x=np.stack(parameters);del parameters
        budget=path_budget(x,steps,metadata['rate'],metadata['weight_decay'])
        first_budget=first_force_budget(x[:,0],x[:,1],np.stack(raws),np.stack(clipped),metadata['rate'],metadata['weight_decay'],metadata['epsilon'])
        cells.append(dict(environment=e,heads=n,control=g,seed_ids=[s for s,_,_ in rows],shared_parameter_count=x.shape[-1],
            rate=metadata['rate'],weight_decay=metadata['weight_decay'],first_force=first_budget,**budget))
        print('SHARED PATH',e,n,g,flush=True)
    result=dict(schema='critical-shared-path-analysis-v1',status='complete',study=str(study),cells=cells,
        scientific_updates=updates,additional_training_updates=0,excluded_paths=excluded,unresolved_cells=unresolved,
        checked_sha256=checked,source_sha256={'scripts/analyze_critical_shared_paths.py':sha256(__file__)},
        all_shared_endpoints_equal_native_state=True,all_initial_gradients_control_paired=True,
        units='Covariance traces and mean-square displacements divided by the fixed shared-parameter dimension; sample covariance divisor S-1.',
        covariance_scope='Complete 256-update block increments conditioned on a fixed source and shared initial state. Off-diagonal contributions are signed and cadence dependent. No stationarity, independent minibatch noise or fluctuation-dissipation law is asserted.',
        mean_scope='The estimate of squared population mean displacement subtracts V/S and may be negative. Its sum with V is exactly the observed mean squared displacement.')
    write_json(output,result)
    print(json.dumps({'status':'complete','cells':len(cells),'additional_training_updates':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.study.resolve(),args.output.resolve())
