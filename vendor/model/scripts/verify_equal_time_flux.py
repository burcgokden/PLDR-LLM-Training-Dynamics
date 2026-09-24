#!/usr/bin/env python3
"""Independent common-row-energy reconstruction of every finite native transition."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict
from equal_time_flux_contract import inventory
from equal_time_outcomes import validate_outcomes


def reduce(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite matrices')
    d=y-x;n=x.shape[-2]
    e=np.einsum('...ij,...ij->...',x,x);en=np.einsum('...ij,...ij->...',y,y)
    if np.any(e<=0) or np.any(en<=0):raise ValueError('Nonpositive endpoint energy')
    # An independent expression uses total energy minus common-row energy.
    sx=x.sum(-2);sy=y.sum(-2);sd=d.sum(-2)
    common=np.einsum('...j,...j->...',sx,sx)/n
    common_y=np.einsum('...j,...j->...',sy,sy)/n
    u=1-common/e;un=1-common_y/en
    dd=np.einsum('...ij,...ij->...',d,d);xd=np.einsum('...ij,...ij->...',x,d)
    cross=2*((1-u)*xd-np.einsum('...j,...j->...',sx,sd)/n)
    quad=(1-u)*dd-np.einsum('...j,...j->...',sd,sd)/n
    delta=np.sqrt(dd/e)
    bound=np.full_like(delta,np.nan);small=delta<1
    bound[small]=(3*delta[small]**2+delta[small]**3)/(1-delta[small])**2
    return dict(increment=un-u,finite_cross=cross/en,quadratic=quad/en,matrix_derivative=cross/e,error_bound=bound,delta=delta,ratio=en/e)


def verify(study,analysis):
    a=load_json_strict(analysis.read_text())
    from analyze_equal_time_flux import analyze as reconstruct
    if a != reconstruct(study): raise ValueError('Complete deterministic analysis differs')
    p,manifests,checked=inventory(study)
    if a['status']!='passed' or a['schema']!='equal-time-flux-analysis-v2' or a['checked_sha256']!=checked:raise ValueError('Changed analysis binding')
    repo=Path(__file__).resolve().parents[1]
    for n,h in a['source_sha256'].items():
        if sha256(repo/n)!=h:raise ValueError('Changed analysis source')
    want={j['run_id'] for j in p['jobs']};ends={x['run_id']:x for x in a['endpoints']}
    cells={(x['run_id'],x['factor']):x for x in a['cells']}
    if len(a['endpoints'])!=16 or set(ends)!=want or len(a['cells'])!=64 or set(cells)!={(r,b) for r in want for b in [1,2,4,8]}:raise ValueError('Incomplete or duplicate result keys')
    maximum=0.;pairs=0;minimum=float('inf');old_replay=0.;temporal_max=0.;temporal_pairs=0
    completeness=validate_outcomes(a,p)
    def close(actual,expected):
        nonlocal maximum
        if isinstance(actual,bool) or not np.isfinite(actual):raise ValueError('Invalid numeric value')
        d=abs(actual-expected);maximum=max(maximum,d)
        if d>3e-12*max(1,abs(expected)):raise ValueError(f'Independent reduction differs: {actual}, {expected}')
    def count(row,k,n):
        if type(row[k]) is not int or row[k]!=n:raise ValueError('Wrong count '+k)
    old=study.parent/'row-path-confirmation-20260917'
    replay_checked={}
    for j in p['jobs']:
        t=8*j['heads'];folder=study/'runs'/j['run_id'];raw=np.load(folder/'matrices.npy',mmap_mode='r');ep=ends[j['run_id']]
        for row in [ep]+[cells[j['run_id'],f] for f in [1,2,4,8]]:
            if any(row.get(k)!=v or isinstance(row.get(k),bool) for k,v in j.items()):raise ValueError('Mislabeled outcome')
        previous=old/'runs'/j['run_id']/'observations.npz';replay_checked[str(previous)]=sha256(previous)
        with np.load(previous) as z:
            discrepancy=float(np.max(abs(raw[:9].astype(float)-z['A'].astype(float))))
            old_replay=max(old_replay,discrepancy)
        # Same scientific prefix, with a directly reported numerical replay diagnostic.
        values=[reduce(raw[k],raw[k+1]) for k in range(t)]
        shape=values[0]['increment'].shape;inner=int(np.prod(shape));pairs+=t*inner
        minimum=min(minimum,min(float(z['ratio'].min()) for z in values))
        for name in ['increment','finite_cross','quadratic','matrix_derivative']:
            close(ep[name],sum(v[name] for v in values).mean())
        difference=sum(v['increment']-v['matrix_derivative'] for v in values)
        close(ep['derivative_error'],abs(difference.mean()))
        applicable=sum(int(np.sum(v['delta']<1)) for v in values)
        count(ep,'updates',t);count(ep,'one_step_pairs',t*inner);count(ep,'bound_applicable',applicable)
        close(ep['physical_duration'],1/16);close(ep['maximum_delta'],max(float(v['delta'].max()) for v in values))
        if applicable==t*inner:
            close(ep['accumulated_bound_mean'],sum(v['error_bound'] for v in values).mean())
            if np.any(abs(difference)>sum(v['error_bound'] for v in values)+3e-12):raise ValueError('Path error bound fails')
        elif ep['accumulated_bound_mean'] is not None:raise ValueError('Unavailable bound is not null')
        with np.load(folder/'observations.npz') as z:
            close(ep['target_nll_before'],z['target_nll'][0].astype(float).mean());close(ep['target_nll_after'],z['target_nll'][-1].astype(float).mean())
        for factor in [1,2,4,8]:
            block=factor*j['heads'];cell=cells[j['run_id'],factor];v=[reduce(raw[k],raw[k+block]) for k in range(0,t,block)]
            pairs+=len(v)*inner;minimum=min(minimum,min(float(z['ratio'].min()) for z in v))
            count(cell,'block',block);count(cell,'aligned_blocks',t//block);count(cell,'pairs',len(v)*inner);close(cell['physical_duration'],factor/128)
            for name in ['increment','finite_cross','quadratic','matrix_derivative']:close(cell[name],np.mean([z[name].mean() for z in v]))
            close(cell['matrix_derivative_absolute_error'],np.mean([abs(z['increment']-z['matrix_derivative']).mean() for z in v]))
            # Independent ordered pair accumulation: never compute X by B-S.
            temporal=[]
            for start in range(0,t,block):
                origin=raw[start].astype(float)
                denominator=np.einsum('...ij,...ij->...',origin,origin)
                accumulated=np.zeros_like(origin);diagonal=np.zeros_like(denominator);cross=np.zeros_like(denominator)
                for k in range(start,start+block):
                    increment=raw[k+1].astype(float)-raw[k].astype(float)
                    cross+=2*np.einsum('...ij,...ij->...',accumulated,increment)
                    diagonal+=np.einsum('...ij,...ij->...',increment,increment)
                    accumulated+=increment
                endpoint=np.einsum('...ij,...ij->...',accumulated,accumulated)
                scale=np.maximum(1,np.maximum(abs(endpoint/denominator),abs(diagonal/denominator)))
                residual=np.max(abs((endpoint-diagonal-cross)/denominator)/scale)
                temporal_max=max(temporal_max,float(residual));temporal_pairs+=inner
                if residual>3e-12 or np.any(cross < -diagonal-3e-12*denominator) or np.any(cross>(block-1)*diagonal+3e-12*denominator):
                    raise ValueError('Independent pointwise temporal energy failed')
                temporal.append((np.mean(diagonal/denominator),np.mean(endpoint/denominator),np.mean(cross/denominator)))
            for key,value in zip(['step_squared_energy','block_squared_energy','signed_cross_time_energy'],np.mean(temporal,axis=0)):
                close(cell[key],value)
            for key in ['identity_error','composition_error','energy_composition_error']:
                if not 0<=cell[key]<3e-12:raise ValueError('Invalid arithmetic residual')
        print('VERIFIED',j['run_id'],flush=True)
    for k,n in dict(paths=16,initialization_identities=2,new_pretraining_identities=0,executed_updates=1600,scientific_updates=1472,replay_updates=128,native_forwards=3216,pairs=pairs,synthetic_perturbation_pairs=1090560).items():count(a,k,n)
    close(a['minimum_energy_ratio'],minimum);close(a['worker_seconds'],sum(m['elapsed_seconds'] for m in manifests));close(a['peak_allocated_bytes'],max(m['peak_allocated_bytes'] for m in manifests))
    if set(a['maximum_errors'])!={'identity','composition','path_bound','synthetic_bound'} or any(not 0<=x<3e-12 for x in a['maximum_errors'].values()):raise ValueError('Invalid global residuals')
    return dict(status='passed',schema='equal-time-flux-verification-v2',analysis_sha256=sha256(analysis),verifier_sha256=sha256(__file__),
        contract_sha256=sha256(Path(__file__).with_name('equal_time_flux_contract.py')),outcome_contract_sha256=sha256(Path(__file__).with_name('equal_time_outcomes.py')),
        outcome_completeness=completeness,temporal_coordinate_blocks=temporal_pairs,maximum_temporal_scaled_residual=temporal_max,paths=16,cells=64,pairs=pairs,
        maximum_independent_difference=maximum,maximum_prefix_matrix_replay_difference=old_replay,replayed_observation_sha256=replay_checked,
        verified_scope='Complete declared outcomes, identities, counts and analytic bounds. Independent common-row-energy and ordered pair-product reconstruction. Synthetic perturbations are sensitivity controls only.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    write_json(a.output,verify(a.study.resolve(),a.analysis.resolve()))
