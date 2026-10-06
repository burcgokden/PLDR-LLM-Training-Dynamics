#!/usr/bin/env python3
"""Development-frozen categorical observations on complete native predictions.

This is an offline observation reduction. It acquires full native logits and
makes no saving claim for the forward pass or the optimizer successor law.
"""
from companion_paths import configured_path
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256,write_json
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(configured_path('data:model'))
COARSE=ROOT/'critical-onepass-coarse-20260914'
INDEPENDENT=ROOT/'critical-onepass-refinement-20260914'


def load_endpoint(study,job):
    folder=study/'runs'/job['run_id'];mp=folder/'manifest.json';m=json.loads(mp.read_text())
    if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=sha256(study/'protocol.json'):
        raise ValueError('Uncompleted or foreign native endpoint')
    path=folder/'observations.npz';digest=sha256(path)
    if digest!=m['artifacts']['observations.npz']:raise ValueError('Changed native observation')
    with np.load(path,allow_pickle=False) as data:z=data[f'logits_{job["steps"]}'].astype(float)
    if not np.isfinite(z).all():raise ValueError('Nonfinite native logits')
    return z,{str(mp):sha256(mp),str(path):digest}


def prepare(output):
    if not output.resolve().is_relative_to(ROOT) or output.resolve()==ROOT:raise ValueError('Use the experiment root')
    output.mkdir(parents=True,exist_ok=False)
    p=json.loads((COARSE/'protocol.json').read_text());rows=[];checked={str(COARSE/'protocol.json'):sha256(COARSE/'protocol.json')}
    # One fixed reference at the largest development width and zero generator
    # control. This choice is explicit and shared across every test width.
    jobs=[j for j in p['jobs'] if j['heads']==14 and j['control']==0 and j['environment']==0]
    if len(jobs)!=6:raise ValueError('Expected six development initializations')
    for job in jobs:
        z,ids=load_endpoint(COARSE,job);checked.update(ids)
        rows.append(np.exp(z-logsumexp(z,axis=-1,keepdims=True)))
    reference=np.mean(rows,axis=0)
    order=np.argsort(-reference.mean(0),kind='stable')
    np.savez_compressed(output/'reference.npz',reference=reference,order=order)
    write_json(output/'protocol.json',dict(schema='categorical-visibility-v1',
        development_run_ids=[j['run_id'] for j in jobs],development_checked_sha256=checked,
        evaluation_study=str(INDEPENDENT),evaluation_protocol_sha256=sha256(INDEPENDENT/'protocol.json'),
        retained_tokens=[128,512,2048,8192,16384],reference_sha256=sha256(output/'reference.npz'),
        source_sha256=sha256(__file__),relative_centered_rms_target=.25,
        primary_endpoint='All declared widths, controls and complete seeds at their final native step.',
        units='Full vocabulary-summed Hellinger embedding variance, averaged over eight fixed contexts.',
        resource_scope='Offline acquisition of full native predictions; no training-state or successor-law compression.',
        selection_scope='Dictionary frozen using the development cohort; independent-family endpoints are retained observations reused for this specified comparison.'))
    print('Dictionary frozen',flush=True)


def variance(x):
    centered=x-x.mean(0,keepdims=True)
    return float(np.sum(centered*centered)/(len(x)-1)/x.shape[1])


def analyze(output):
    started=time.monotonic();spec=json.loads((output/'protocol.json').read_text())
    if (output/'analysis.json').exists():raise FileExistsError(output/'analysis.json')
    if spec['source_sha256']!=sha256(__file__) or spec['reference_sha256']!=sha256(output/'reference.npz'):
        raise ValueError('Frozen dictionary or source changed')
    p=json.loads((INDEPENDENT/'protocol.json').read_text())
    if sha256(INDEPENDENT/'protocol.json')!=spec['evaluation_protocol_sha256']:raise ValueError('Evaluation family changed')
    with np.load(output/'reference.npz') as data:ref=data['reference'];order=data['order']
    checked={str(output/'protocol.json'):sha256(output/'protocol.json'),str(output/'reference.npz'):sha256(output/'reference.npz')}
    seeds=sorted(p['design']['seeds']);cells=[];max_residual=0.
    for n in sorted(p['design']['heads']):
        for g in sorted(p['design']['controls']):
            jobs=sorted([j for j in p['jobs'] if j['heads']==n and j['control']==g],key=lambda j:j['seed'])
            if [j['seed'] for j in jobs]!=seeds:raise ValueError('Incomplete initialization denominator')
            logits=[]
            for job in jobs:
                z,ids=load_endpoint(INDEPENDENT,job);checked.update(ids);logits.append(z)
            z=np.stack(logits);lp=z-logsumexp(z,axis=-1,keepdims=True);prob=np.exp(lp);phi=2*np.sqrt(prob)
            vy=variance(phi)
            for k in spec['retained_tokens']:
                keep=order[:k];tail=order[k:];qtail=prob[:,:,tail].sum(-1)
                rtail=ref[:,tail]/ref[:,tail].sum(-1,keepdims=True)
                q=np.concatenate([prob[:,:,keep],qtail[...,None]],axis=-1)
                surrogate=np.empty_like(prob);surrogate[:,:,keep]=prob[:,:,keep]
                surrogate[:,:,tail]=qtail[...,None]*rtail[None]
                psi=2*np.sqrt(surrogate);residual=phi-psi
                vx=variance(2*np.sqrt(q));vhat=variance(psi);vr=variance(residual)
                covariance=float(np.sum((psi-psi.mean(0))*(residual-residual.mean(0)))/(len(seeds)-1)/phi.shape[1])
                budget=2*np.sqrt(vx*vr)+vr;defect=vy-vx
                error=max(abs(vhat-vx),abs(defect-(2*covariance+vr)))
                max_residual=max(max_residual,error)
                if defect < -2e-12 or defect>budget+2e-12 or error>2e-12:raise ValueError('Visibility identity failed')
                kl=np.sum(prob*(lp-np.log(surrogate)),axis=-1)
                tail_kl=np.sum(prob[:,:,tail]*(lp[:,:,tail]-np.log(qtail)[...,None]-np.log(rtail)[None]),axis=-1)
                if np.max(np.abs(kl-tail_kl))>2e-12:raise ValueError('Conditional-tail KL identity failed')
                cells.append(dict(heads=n,control=g,retained_tokens=k,coarse_dimension=k+1,vocabulary_size=prob.shape[-1],
                    seed_ids=seeds,native_variance=vy,coarse_variance=vx,retained_variance_fraction=vx/vy,
                    centered_remainder_variance=vr,relative_centered_rms=float(np.sqrt(vr/vy)),
                    relative_uncentered_rms=float(np.sqrt(np.mean(np.sum(residual*residual,axis=-1))/vy)),
                    covariance_defect=defect,variance_budget=budget,signed_cross_covariance=covariance,
                    mean_emission_kl=float(kl.mean()),mean_tail_mass=float(qtail.mean()),
                    native_susceptibility=n*vy,coarse_susceptibility=n*vx,
                    target_met=bool(np.sqrt(vr/vy)<=spec['relative_centered_rms_target']),
                    identity_residual=error))
    summary=[]
    for k in spec['retained_tokens']:
        rows=[r for r in cells if r['retained_tokens']==k]
        summary.append(dict(retained_tokens=k,cells=len(rows),target_met=sum(r['target_met'] for r in rows),
            relative_centered_rms_range=[min(r['relative_centered_rms'] for r in rows),max(r['relative_centered_rms'] for r in rows)],
            retained_variance_fraction_range=[min(r['retained_variance_fraction'] for r in rows),max(r['retained_variance_fraction'] for r in rows)],
            emission_kl_range=[min(r['mean_emission_kl'] for r in rows),max(r['mean_emission_kl'] for r in rows)]))
    write_json(output/'analysis.json',dict(status='complete',schema='categorical-visibility-analysis-v1',cells=cells,summary=summary,
        observed_paths=len(p['jobs']),scientific_updates=0,additional_native_forward_calls=0,
        maximum_identity_residual=max_residual,analysis_seconds=time.monotonic()-started,checked_sha256=checked,
        reference_bytes=(output/'reference.npz').stat().st_size,source_sha256=sha256(__file__),
        cost='Retains a context-dependent full-vocabulary tail dictionary, acquired full-vocabulary logits and native training state. Dimension ratios describe stored observations only.',
        interpretation='Finite predictive observation visibility, conditional on the fixed corpus and initial shared generator. No economical autonomous closure or critical exponent is asserted.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','analyze']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();(prepare if a.action=='prepare' else analyze)(a.output.resolve())
