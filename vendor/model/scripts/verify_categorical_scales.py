#!/usr/bin/env python3
"""Independent direct-pair reconstruction of every categorical scale cell."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict, finite_array, finite_scalar, discrepancy
from categorical_summary_contract import validate, summaries, compare_summaries, integer


def verify(analysis,output):
    if output.exists():raise FileExistsError(output)
    result=load_json_strict(analysis.read_text());plan=load_json_strict((analysis.parent/'protocol.json').read_text())
    if result['status']!='complete':raise ValueError('Incomplete analysis')
    canonical=validate(result,plan)
    reconstructed=[]
    checked={}
    def bind(path,expected=None):
        digest=sha256(path)
        if expected is not None and digest!=expected:raise ValueError('Changed input '+str(path))
        checked[str(path)]=digest
    bind(analysis);bind(analysis.parent/'protocol.json')
    for path,digest in result['checked_sha256'].items():bind(Path(path),digest)
    study=Path(plan['study']);spec=load_json_strict((study/'protocol.json').read_text())
    fresh=spec['schema']=='context-categorical-v1'
    dictionary=study/'inputs.npz' if fresh else Path(plan['dictionary'])/'reference.npz'
    bind(dictionary)
    with np.load(dictionary) as a:reference=a['reference'];order=a['order']
    finite_array(reference,'reference')
    if np.any(reference<=0) or not np.array_equal(np.sort(order),np.arange(len(order))):raise ValueError('Invalid dictionary')
    groups=defaultdict(list)
    for job in spec['jobs']:groups[job['heads'],job['control']].append(job)
    index={(r['heads'],r['control'],r['time'],r['retained_tokens']):r for r in result['cells']}
    if len(index)!=len(result['cells']):raise ValueError('Duplicate cell')
    seen=set();max_error=0.;chain_error=0.;chain_checks=0
    edge_index={(r['heads'],r['control'],r['time'],r['coarse_tokens'],r['fine_tokens']):r for r in result['edges']}
    if len(edge_index)!=len(result['edges']):raise ValueError('Duplicate scale edge')
    seen_edges=set(); calls=qualification=logical_bytes=compressed_bytes=0; worker_seconds=0.
    absorption_error=0.; risk_error=0.; isometry_error=0.; variance_error=0.
    for (n,g),jobs in sorted(groups.items()):
        jobs.sort(key=lambda j:j['seed'])
        if len(jobs)!=6:raise ValueError('Missing replica')
        values={t:[] for t in plan['times']}
        for job in jobs:
            folder=study/'runs'/job['run_id'];m=load_json_strict((folder/'manifest.json').read_text())
            bind(folder/'manifest.json')
            if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=sha256(study/'protocol.json'):raise ValueError('Incomplete or foreign native state')
            if fresh:
                calls+=integer(m['native_forward_calls'],'native forwards')
                qualification+=integer(m['qualification_forward_calls'],'qualification forwards')
                worker_seconds+=finite_scalar(m['elapsed_seconds'],'worker seconds')
                if m['elapsed_seconds']<0 or integer(m['training_updates'],'updates')!=0:raise ValueError('Invalid observation cost')
            path=folder/'observations.npz';bind(path,m['observations_sha256'] if fresh else m['artifacts']['observations.npz'])
            if fresh:compressed_bytes+=path.stat().st_size
            with np.load(path) as a:
                if fresh:
                    if a['logits'].dtype!=np.float32:raise ValueError('Expected native float32 logits')
                    logical_bytes+=a['logits'].nbytes
                for t in plan['times']:
                    z=finite_array(a['logits' if fresh else f'logits_{t}'],'logits').astype(float)
                    p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);values[t].append(p)
        for t,ps in values.items():
            p=np.stack(ps);S,C,V=p.shape
            if S!=6 or C!=plan['contexts'] or V!=len(order):raise ValueError('Observed panel dimensions differ')
            ref=np.broadcast_to(reference,(C,V));full=2*np.sqrt(p)
            def lift(a,k):
                q=a.copy();tail=order[k:]
                q[:,:,tail]=a[:,:,tail].sum(-1,keepdims=True)*(ref[:,tail]/ref[:,tail].sum(-1,keepdims=True))[None]
                return q
            def pair(x,y=None):
                if y is None:y=x
                return sum(float(np.sum((x[i]-x[j])*(y[i]-y[j]))) for i in range(S) for j in range(i))/(S*(S-1)*C)
            previous=None
            for k in plan['sizes']:
                row=index[n,g,t,k];seen.add((n,g,t,k))
                if row['seed_ids']!=[j['seed'] for j in jobs] or row['contexts']!=C:raise ValueError('Cell conditioning differs')
                keep=order[:k];tail=order[k:];tailmass=p[:,:,tail].sum(-1)
                q=p.copy();q[:,:,tail]=tailmass[...,None]*(ref[:,tail]/ref[:,tail].sum(-1,keepdims=True))[None]
                lifted=2*np.sqrt(q);residual=full-lifted
                reduced=np.concatenate([full[:,:,keep],2*np.sqrt(tailmass)[...,None]],axis=-1)
                vp=pair(full);vc=pair(reduced);ve=pair(residual)
                if vp<=0:raise ValueError('Undefined relative native variance')
                kl=np.sum(p*np.log(p/q),axis=-1)
                u=float(np.mean(np.sum(residual**2,axis=-1)));b=float(np.mean(np.sum(residual.mean(0)**2,axis=-1)))
                claims=dict(native_variance=vp,coarse_variance=vc,centered_remainder_variance=ve,
                    relative_centered_rms=float(np.sqrt(ve/vp)),retained_variance_fraction=vc/vp,
                    mean_emission_kl=float(kl.mean()),uncentered_energy=u,mean_bias_energy=b,
                    signed_cross_covariance=pair(lifted,residual),relative_uncentered_rms=float(np.sqrt(u/vp)),
                    sufficient_kl_relative_bound=float(np.sqrt(4*S/(S-1)*kl.mean()/vp)))
                for key,actual in claims.items():max_error=max(max_error,discrepancy(actual,row[key],key,atol=3e-12))
                risk_error=max(risk_error,discrepancy(ve,S/(S-1)*(u-b),'risk identity',atol=3e-12))
                vl=pair(lifted);cross_cov=pair(lifted,residual)
                isometry_error=max(isometry_error,discrepancy(vl,vc,'lift isometry',atol=3e-12))
                variance_error=max(variance_error,discrepancy(vp-vl,2*cross_cov+ve,'variance budget',atol=3e-12))
                primitive=max(abs(vl-vc),abs((vp-vl)-(2*cross_cov+ve)),abs(ve-S/(S-1)*(u-b)))
                discrepancy(primitive,row['maximum_algebra_error'],'recorded cell algebra residual',atol=3e-12)
                if u>4*float(kl.mean())+3e-12 or vc>vp+3e-12 or np.min(kl)<-3e-12:raise ValueError('Finite risk/channel bound failed')
                if type(row['target_met']) is not bool or row['target_met']!=(claims['relative_centered_rms']<=plan['target']):raise ValueError('Target decision differs')
                reconstructed.append(dict(heads=n,control=g,time=t,retained_tokens=k,
                    target_met=bool(claims['relative_centered_rms']<=plan['target']),**claims))
                if previous is not None:
                    oldk,oldq,oldkl,oldv=previous;cross=np.sum(q*np.log(q/oldq),axis=-1)
                    residual_kl=oldkl-kl-cross;finite_array(residual_kl,'KL chain')
                    err=float(np.max(np.abs(residual_kl)));chain_error=max(chain_error,err);chain_checks+=S*C
                    if err>3e-12 or vc<oldv-3e-12:raise ValueError('Nested law failed')
                    edgekey=n,g,t,oldk,k;edge=edge_index[edgekey];seen_edges.add(edgekey)
                    absorption=max(float(np.max(np.abs(lift(q,oldk)-oldq))),float(np.max(np.abs(lift(oldq,k)-oldq))))
                    absorption_error=max(absorption_error,absorption)
                    if absorption>3e-12:raise ValueError('Absorption failed')
                    discrepancy(err,edge['kl_chain_max_error'],'recorded KL residual',atol=3e-12)
                    discrepancy(absorption,edge['absorption_max_error'],'recorded absorption residual',atol=3e-12)
                    if edge['seed_ids']!=[j['seed'] for j in jobs] or edge['contexts']!=C:raise ValueError('Edge conditioning differs')
                    discrepancy(vc-oldv,edge['variance_gain'],'nested variance gain',atol=3e-12)
                    discrepancy(float((oldkl-kl).mean()),edge['mean_kl_gain'],'nested KL gain',atol=3e-12)
                previous=k,q,kl,vc
    if seen!=set(index) or seen_edges!=set(edge_index):raise ValueError('Incomplete scale coverage')
    if chain_checks!=result['path_context_kl_chains']:raise ValueError('Wrong chain denominator')
    compare_summaries(canonical,summaries(reconstructed,plan),atol=3e-12)
    metadata=dict(acquisition_native_forward_calls=calls,qualification_forward_calls=qualification,
        acquisition_worker_seconds=worker_seconds,compressed_observation_bytes=compressed_bytes,logical_logit_bytes=logical_bytes,
        training_updates=0,analysis_native_forward_calls=0)
    for name,value in metadata.items():
        if name!='acquisition_worker_seconds':integer(result[name],name)
        discrepancy(value,result[name],'acquisition metadata '+name,atol=0.)
    if fresh and (calls!=spec['expected_forward_calls'] or qualification!=spec['qualification_forward_calls']):raise ValueError('Protocol forward counts differ')
    support=Path(__file__).parent/'numerical_validation.py'
    contract=Path(__file__).parent/'categorical_summary_contract.py'
    write_json(output,dict(status='passed',schema='categorical-scale-verification-v2',summary_groups=len(canonical),reconstructed_summary=canonical,cells=len(seen),
        adjacent_scale_cells=len(seen_edges),observed_paths=len(spec['jobs']),path_context_kl_chains=chain_checks,
        maximum_absolute_error=max_error,maximum_kl_chain_error=chain_error,
        maximum_absorption_error=absorption_error,maximum_risk_error=risk_error,
        maximum_lift_isometry_error=isometry_error,maximum_variance_budget_error=variance_error,
        verified_metadata=metadata,
        checked_sha256=checked,verifier_sha256=sha256(__file__),support_sha256={str(support):sha256(support),str(contract):sha256(contract)},
        training_updates=0,native_forward_calls=0,scope='Independent direct-pair cell, summary, KL-chain, absorption and risk reconstruction; costs reconciled to bound manifests and observation arrays. Worker times are recorded wall-clock metadata, not remeasured training costs. Float64 checks are not interval enclosures.'))
    print(len(seen),'cells verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.analysis,a.output)
