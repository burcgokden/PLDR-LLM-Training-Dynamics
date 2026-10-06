#!/usr/bin/env python3
"""Reconstruct endpoint convention sensitivity and every native paired effect."""
from companion_paths import configured_path
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256,write_json
from model_rg.potential_symmetry import summarize_allocations
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(configured_path('data:model'))
LABELS={'reference1':'Strong collapse','subcritical1':'Partial collapse','early-h2':'Early 2 heads','early-h8':'Early 8 heads'}


def kl(a,b):
    la=a.astype(float)-logsumexp(a.astype(float),axis=-1,keepdims=True)
    lb=b.astype(float)-logsumexp(b.astype(float),axis=-1,keepdims=True)
    return np.sum(np.exp(la)*(la-lb),axis=-1)


def nll(logits,targets):
    x=logits.astype(float);return logsumexp(x,axis=-1)-x[np.arange(len(x)),targets]


def panels(study):
    base=ROOT/'potential-avalanche-20260913';results=[]
    for kind,folder,start in [('early','runs',256),('continuation','continuations',0)]:
        for p in sorted((base/folder).glob('*/activity.npz')):
            m=json.loads((p.parent/'manifest.json').read_text())
            if sha256(p)!=m['artifacts']['activity.npz']:raise ValueError('Changed activity')
            z=np.load(p);b=z['logbase'][start:,0];power=z['power'][start:]
            results.append(dict(name=p.parent.name,kind=kind,raw_sha256=sha256(p),raw=str(p),
                manifest_sha256=sha256(p.parent/'manifest.json'),**summarize_allocations(b,power)))
    if len(results)!=22:raise ValueError('Incomplete coordinate family')
    out=study/'analysis';out.mkdir(exist_ok=True)
    write_json(out/'endpoint-panels.json',dict(status='complete',panels=results,
        analysis_sources={str(p.relative_to(REPO)):sha256(p) for p in [Path(__file__).resolve(),REPO/'src/model_rg/potential_symmetry.py']},
        scope='First fixed probe, 128 prespecified entries per layer/head. Float64 reconstruction of saved float32 coordinates.'))
    return results


def factorial(study):
    spec=json.loads((study/'protocol.json').read_text());records=[]
    for case in spec['cases']:
        path=study/'runs'/case['name'];mf=path/'manifest.json';m=json.loads(mf.read_text())
        if m['status']!='complete' or m['profile'] or m['protocol_sha256']!=sha256(study/'protocol.json'):raise ValueError('Incomplete bound scientific case')
        branches={};arrays={}
        for name,record in m['branches'].items():
            p=path/(name+'.npz')
            if sha256(p)!=record['artifact_sha256']:raise ValueError('Changed raw branch')
            z=np.load(p);arrays[name]=z
            if name=='replay':continue
            f=z['fields'][1:,0];tensor=z['tensor_statistics'][1:,0];sym=z['symmetric'][1:,0]
            act=np.sqrt(np.mean(f[...,0]**2,axis=(1,2)))
            sympower=sym[...,0].sum();symbase=sym[...,1].sum();symcross=sym[...,2].sum()
            initial=nll(z['initial_native_logits'],z['targets']);intervened=nll(z['initial_intervened_logits'],z['targets']);end=nll(z['final_logits'],z['targets'])
            stepkl=kl(z['step_logits'][:-1],z['step_logits'][1:])
            np.testing.assert_allclose(stepkl,z['predictive_step_kl'][1:],atol=3e-13,rtol=2e-7)
            native_base=z['logbase'].astype(float)
            # Passive raising of the offset on this unchanged recorded path.
            passive_b=np.log(np.exp(native_base)+1e-6-record['offset'])
            power=z['power'].astype(float)[:,None]
            passive=np.diff(passive_b*power,axis=0)[:,0]
            raw=np.diff(native_base*power,axis=0)[:,0]
            branches[name]=dict(offset=record['offset'],reset_first_moment=record['reset_first_moment'],
                integrated_activity=float(act.sum()),integrated_curvature=float(np.sqrt(np.mean(tensor[...,4,1]**2,axis=(1,2))).sum()),
                summed_step_kl=float(stepkl[:,0].sum()),nll_initial=float(initial.mean()),nll_intervened=float(intervened.mean()),nll_final=float(end.mean()),
                nll_change=float((end-initial).mean()),immediate_offset_kl=float(kl(z['initial_native_logits'],z['initial_intervened_logits']).mean()),
                immediate_offset_q_rms=record['initial_logpotential_jump_rms'],
                symmetric_base_fraction=float(symbase/(symbase+sympower)),symmetric_signed_cross_fraction=float(symcross/(symbase+sympower)),
                symmetric_max_identity_residual=float(sym[...,4].max()),
                initial_row_ratio=float(np.mean(z['fields'][0,0,...,8])),final_row_ratio=float(np.mean(z['fields'][-1,0,...,8])),
                passive_panel_energy_ratio=float(np.sum(passive**2)/np.sum(raw**2)),
                source=str(p),source_sha256=sha256(p))
        for key in ['rows','offsets','lr','loss','final_logits']:
            if not np.array_equal(arrays['native_keep'][key],arrays['replay'][key]):raise ValueError('Replay mismatch '+key)
        if m['branches']['native_keep']['final_state_sha256']!=m['branches']['replay']['final_state_sha256']:raise ValueError('State replay mismatch')
        metrics=['integrated_activity','integrated_curvature','summed_step_kl','nll_final'];effects={}
        for metric in metrics:
            y={n:b[metric] for n,b in branches.items()}
            effects[metric]=dict(offset_retained=y['raised_keep']-y['native_keep'],offset_reset=y['raised_reset']-y['native_reset'],
                reset_native=y['native_reset']-y['native_keep'],reset_raised=y['raised_reset']-y['raised_keep'],
                interaction=y['raised_reset']-y['raised_keep']-y['native_reset']+y['native_keep'])
            if metric!='nll_final':effects[metric].update(offset_ratio_retained=y['raised_keep']/y['native_keep'],offset_ratio_reset=y['raised_reset']/y['native_reset'],reset_ratio_native=y['native_reset']/y['native_keep'])
        records.append(dict(case=case,branches=branches,effects=effects,replay_bitwise=True,manifest_sha256=sha256(mf),
            runtime_seconds=m['runtime_seconds'],max_cuda_memory_bytes=m['max_cuda_memory_bytes']))
        print(case['name'],{k:v for k,v in effects.items()},flush=True)
    result=dict(status='complete',cases=records,scientific_paths=16,scientific_updates=2048,scientific_input_token_presentations=2048*32*64,
        replay_updates=512,profile_updates=128,protocol_sha256=sha256(study/'protocol.json'),analysis_sha256=sha256(__file__),
        units='Four fixed states with paired branch interventions; no population confidence intervals or phase assignment.')
    write_json(study/'analysis/factorial-summary.json',result)
    return result




