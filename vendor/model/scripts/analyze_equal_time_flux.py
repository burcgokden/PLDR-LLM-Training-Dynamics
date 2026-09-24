#!/usr/bin/env python3
"""Reconstruct every equal-time path and aligned physical block from native matrices."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json
from equal_time_flux_contract import inventory
from analyze_row_path_confirmation import transport
from equal_time_outcomes import SCHEMA, OUTCOMES, NORMALIZATION, validate_outcomes


def coordinates(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    z=transport(x,y);e=z['incoming_energy'];en=np.square(y).sum((-2,-1))
    u=np.square(x-x.mean(-2,keepdims=True)).sum((-2,-1))/e
    c=en/e-1; a=z['matrix_derivative']; b=z['quadratic']*en/e
    return z,u,a,b,c


def analyze(study):
    p, manifests, checked=inventory(study)
    endpoints=[];cells=[];pair_count=0;max_identity=0.;max_compose=0.;max_bound=0.;min_ratio=float('inf')
    rng=np.random.default_rng(2026091737);synthetic_error=0.;synthetic_pairs=0
    for j,m in zip(p['jobs'],manifests):
        t=8*j['heads']; folder=study/'runs'/j['run_id'];raw=np.load(folder/'matrices.npy',mmap_mode='r')
        sums={k:np.zeros(raw.shape[1:4]) for k in ['increment','finite_cross','quadratic','matrix_derivative','error_bound']}
        bound_applicable=0;delta_max=0.;local=[];step_energy=[]
        for k in range(t):
            z,u,a,b,c=coordinates(raw[k],raw[k+1]);pair_count+=u.size
            ratio=1+c; min_ratio=min(min_ratio,float(ratio.min()))
            exact=(a+b)/ratio
            max_identity=max(max_identity,float(np.max(abs(exact-z['increment']))))
            for name in sums:sums[name]+=z[name]
            bound_applicable+=int(np.sum(z['delta']<1));delta_max=max(delta_max,float(z['delta'].max()))
            # Frozen sensitivity control, not an incoming forecast or learned closure.
            ah=a+1e-4*rng.normal(size=a.shape);bh=b+1e-4*rng.normal(size=b.shape)
            ch=ratio*np.exp(1e-4*rng.normal(size=c.shape))-1;fh=(ah+bh)/(1+ch)
            budget=(abs(a-ah)+abs(b-bh)+abs(fh)*abs(c-ch))/ratio
            synthetic_error=max(synthetic_error,float(np.max(abs(exact-fh)-budget)));synthetic_pairs+=u.size
            q=a+b+u*c
            local.append((q,c,z['increment']))
            step_energy.append(np.square(raw[k+1].astype(float)-raw[k].astype(float)).sum((-2,-1)))
        with np.load(folder/'observations.npz') as z:nll=z['target_nll'].astype(float)
        all_small=bound_applicable==t*np.prod(raw.shape[1:4])
        if all_small:max_bound=max(max_bound,float(np.max(abs(sums['increment']-sums['matrix_derivative'])-sums['error_bound'])))
        endpoints.append(dict(j,updates=t,physical_duration=t/(128*j['heads']),
            **{name:float(value.mean()) for name,value in sums.items() if name!='error_bound'},
            derivative_error=float(abs((sums['increment']-sums['matrix_derivative']).mean())),
            accumulated_bound_mean=float(sums['error_bound'].mean()) if all_small else None,
            bound_applicable=bound_applicable,one_step_pairs=int(t*np.prod(raw.shape[1:4])),maximum_delta=delta_max,
            target_nll_before=float(nll[0].mean()),target_nll_after=float(nll[-1].mean())))
        for factor in p['physical_block_fractions']:
            block=factor*j['heads'];values=[];composition=0.;identity=0.;decomp=[];temporal=[]
            for start in range(0,t,block):
                z,u,a,b,c=coordinates(raw[start],raw[start+block]);pair_count+=u.size
                min_ratio=min(min_ratio,float((1+c).min()))
                identity=max(identity,float(np.max(abs(z['increment']-(a+b)/(1+c)))))
                qsum=np.zeros_like(c);energy_ratio=np.ones_like(c);direct=np.zeros_like(c)
                for qk,ck,inc in local[start:start+block]:
                    qsum+=energy_ratio*qk;energy_ratio*=1+ck;direct+=inc
                composition=max(composition,float(np.max(abs((u+qsum)/energy_ratio-u-z['increment']))),float(np.max(abs(direct-z['increment']))))
                values.append(z);decomp.append(float(np.max(abs(energy_ratio-(1+c)))))
                e=z['incoming_energy']
                s=np.sum(step_energy[start:start+block],axis=0)/e
                benergy=np.square(raw[start+block].astype(float)-raw[start].astype(float)).sum((-2,-1))/e
                x=benergy-s
                tolerance=3e-12*np.maximum(1,np.maximum(abs(s),abs(x)))
                if np.any(x < -s-tolerance) or np.any(x>(block-1)*s+tolerance):raise ValueError('Pointwise temporal bounds failed')
                temporal.append((float(s.mean()),float(benergy.mean()),float(x.mean())))
            max_identity=max(max_identity,identity);max_compose=max(max_compose,composition)
            cells.append(dict(j,block=block,factor=factor,physical_duration=block/(128*j['heads']),aligned_blocks=t//block,
                pairs=int((t//block)*np.prod(raw.shape[1:4])),
                **{key:float(np.mean([v[key].mean() for v in values])) for key in ['increment','finite_cross','quadratic','matrix_derivative']},
                matrix_derivative_absolute_error=float(np.mean([abs(v['increment']-v['matrix_derivative']).mean() for v in values])),
                identity_error=identity,composition_error=composition,energy_composition_error=max(decomp),
                **dict(zip(['step_squared_energy','block_squared_energy','signed_cross_time_energy'],np.mean(temporal,axis=0).tolist()))))
        print('ANALYZED',j['run_id'],flush=True)
    if max(max_identity,max_compose,max_bound,synthetic_error)>3e-12:raise ValueError('Finite transport or budget failed')
    sources=['scripts/equal_time_outcomes.py','scripts/analyze_equal_time_flux.py','scripts/equal_time_flux_contract.py','scripts/analyze_row_path_confirmation.py']
    result=dict(status='passed',schema='equal-time-flux-analysis-v2',protocol_sha256=checked['protocol.json'],checked_sha256=checked,
        source_sha256={n:sha256(Path(__file__).resolve().parents[1]/n) for n in sources},endpoints=endpoints,cells=cells,
        paths=16,initialization_identities=2,new_pretraining_identities=0,executed_updates=1600,scientific_updates=1472,replay_updates=128,
        native_forwards=3216,pairs=pair_count,synthetic_perturbation_pairs=synthetic_pairs,
        maximum_errors=dict(identity=max_identity,composition=max_compose,path_bound=max_bound,synthetic_bound=synthetic_error),
        minimum_energy_ratio=min_ratio,worker_seconds=sum(m['elapsed_seconds'] for m in manifests),peak_allocated_bytes=max(m['peak_allocated_bytes'] for m in manifests),
        scope='Exact finite transport, signed temporal energy and synthetic sensitivity. Reused parent identities and contexts; no fitted incoming forecast, free reduced rollout or thermodynamic inference.',
        outcome_contract=dict(schema=SCHEMA,outcomes=OUTCOMES,normalization=NORMALIZATION,convention_status='explicit_post_acquisition'))
    validate_outcomes(result,p)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    write_json(a.output,analyze(a.study.resolve()))
