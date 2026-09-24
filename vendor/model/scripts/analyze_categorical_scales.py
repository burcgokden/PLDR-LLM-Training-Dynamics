#!/usr/bin/env python3
"""Complete finite categorical scale and bias/variance budgets, with frozen inputs."""
import argparse
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict, finite_array, discrepancy


def probabilities(logits):
    z=finite_array(logits,'native logits').astype(float)
    p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    if np.any(p<=0):raise ValueError('Positive categorical probabilities required')
    return p


def variance(x):
    if x.ndim!=3 or len(x)<2 or x.shape[1]<1:raise ValueError('Invalid replica/context panel')
    finite_array(x,'replica values')
    return float(np.sum((x-x.mean(0,keepdims=True))**2)/((len(x)-1)*x.shape[1]))


def reduce_panel(p,reference,order,sizes):
    finite_array(p,'probabilities');finite_array(reference,'reference')
    if p.ndim!=3 or np.any(p<=0) or np.any(reference<=0):raise ValueError('Positive categorical panel required')
    if reference.ndim==1:reference=np.broadcast_to(reference,p.shape[1:])
    if reference.shape!=p.shape[1:] or not np.array_equal(np.sort(order),np.arange(p.shape[-1])):
        raise ValueError('Reference or vocabulary ordering differs')
    if not np.allclose(p.sum(-1),1,rtol=0,atol=1e-13):raise ValueError('Probabilities must sum to one')
    if sorted(set(sizes))!=sizes or not all(0<k<p.shape[-1] for k in sizes):raise ValueError('Invalid nested scales')
    s,c,_=p.shape;full=2*np.sqrt(p);vp=variance(full)
    if vp<=0:raise ValueError('Relative criterion undefined at zero native variance')
    def lift(a,k):
        q=a.copy();tail=order[k:]
        q[:,:,tail]=a[:,:,tail].sum(-1,keepdims=True)*(reference[:,tail]/reference[:,tail].sum(-1,keepdims=True))[None]
        return q
    def kl(a,b):return np.sum(a*np.log(a/b),axis=-1)
    cells=[];levels=[];edges=[]
    for k in sizes:
        q=lift(p,k);phi=2*np.sqrt(q);e=full-phi
        coarse=np.concatenate([p[:,:,order[:k]],p[:,:,order[k:]].sum(-1,keepdims=True)],axis=-1)
        vc=variance(2*np.sqrt(coarse));ve=variance(e);vl=variance(phi)
        u=float(np.mean(np.sum(e*e,axis=-1)));bias=float(np.mean(np.sum(e.mean(0)**2,axis=-1)))
        d=kl(p,q);cross=float(np.sum((phi-phi.mean(0))*(e-e.mean(0)))/((s-1)*c))
        errors=[discrepancy(vl,vc,'lift isometry',atol=3e-12),
                discrepancy(vp-vl,2*cross+ve,'signed variance',atol=3e-12),
                discrepancy(ve,s/(s-1)*(u-bias),'centered risk',atol=3e-12)]
        if np.min(d)<-3e-12 or u>4*float(d.mean())+3e-12 or vc>vp+3e-12:
            raise ValueError('Finite channel/KL bound failed')
        cells.append(dict(retained_tokens=k,native_variance=vp,coarse_variance=vc,
            centered_remainder_variance=ve,relative_centered_rms=float(np.sqrt(ve/vp)),
            retained_variance_fraction=vc/vp,mean_emission_kl=float(d.mean()),
            uncentered_energy=u,mean_bias_energy=bias,signed_cross_covariance=cross,
            relative_uncentered_rms=float(np.sqrt(u/vp)),
            sufficient_kl_relative_bound=float(np.sqrt(4*s/(s-1)*d.mean()/vp)),
            target_met=bool(np.sqrt(ve/vp)<=.25),maximum_algebra_error=max(errors)))
        levels.append((k,q,vc,d))
    for (k,qk,vk,dk),(ell,ql,vl,dl) in zip(levels,levels[1:]):
        chain=float(np.max(np.abs(dk-dl-kl(ql,qk))))
        absorption=max(float(np.max(np.abs(lift(ql,k)-qk))),float(np.max(np.abs(lift(qk,ell)-qk))))
        finite_array([chain,absorption,vl-vk],'nested residuals')
        if max(chain,absorption)>3e-12 or vl<vk-3e-12 or np.min(dk-dl)<-3e-12:
            raise ValueError('Nested scale identity failed')
        edges.append(dict(coarse_tokens=k,fine_tokens=ell,kl_chain_max_error=chain,
            absorption_max_error=absorption,variance_gain=vl-vk,mean_kl_gain=float((dk-dl).mean()),
            path_context_chains=s*c))
    return cells,edges


def analyze(study,output,dictionary=None,times=None):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);started=time.monotonic();checked={}
    def bind(path,expected=None):
        h=sha256(path)
        if expected is not None and h!=expected:raise ValueError('Changed input '+str(path))
        checked[str(path)]=h
    spec=load_json_strict((study/'protocol.json').read_text());bind(study/'protocol.json')
    context_mode=spec.get('schema')=='context-categorical-v1'
    if context_mode:
        bind(study/'inputs.npz',spec['selection_sha256'])
        with np.load(study/'inputs.npz') as a:ref=a['reference'];order=a['order']
        sizes=spec['sizes'];times=[4096];contexts=spec['contexts'];jobs=spec['jobs']
    else:
        d=load_json_strict((dictionary/'protocol.json').read_text())
        bind(dictionary/'protocol.json');bind(dictionary/'reference.npz',d['reference_sha256'])
        if sha256(study/'protocol.json')!=d['evaluation_protocol_sha256']:raise ValueError('Wrong evaluation study')
        for path,h in d['development_checked_sha256'].items():bind(Path(path),h)
        with np.load(dictionary/'reference.npz') as a:ref=a['reference'];order=a['order']
        sizes=d['retained_tokens'];contexts=ref.shape[0];jobs=spec['jobs']
    if not times:raise ValueError('Explicit saved times required')
    protocol=dict(schema='categorical-scale-analysis-v1',study=str(study),dictionary=str(dictionary) if dictionary else None,
        times=times,sizes=sizes,contexts=contexts,absolute_tolerance=3e-12,target=.25,
        source_sha256={str(Path(__file__).resolve()):sha256(__file__)},
        selection='All declared width/control cells and all six complete initialization identities; no outcome selection.',
        new_training_updates=0,analysis_forward_calls=0)
    write_json(output/'protocol.json',protocol);bind(output/'protocol.json')
    groups=defaultdict(list)
    for job in jobs:groups[job['heads'],job['control']].append(job)
    cells=[];edges=[];calls=qualification=0;worker_seconds=0.;bytes_acquired=0;logical_bytes=0
    for (n,g),group in sorted(groups.items()):
        group.sort(key=lambda j:j['seed']);saved={t:[] for t in times}
        if len(group)!=6:raise ValueError('Incomplete six-replica cell')
        for job in group:
            folder=study/'runs'/job['run_id'];mp=folder/'manifest.json';m=load_json_strict(mp.read_text());bind(mp)
            if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=sha256(study/'protocol.json'):
                raise ValueError('Incomplete or foreign observation')
            path=folder/'observations.npz'
            bind(path,m['observations_sha256'] if context_mode else m['artifacts']['observations.npz'])
            with np.load(path,allow_pickle=False) as a:
                if context_mode:logical_bytes+=a['logits'].nbytes
                for t in times:saved[t].append(probabilities(a['logits' if context_mode else f'logits_{t}']))
            if context_mode:
                calls+=m['native_forward_calls'];qualification+=m['qualification_forward_calls'];worker_seconds+=m['elapsed_seconds']
                bytes_acquired+=path.stat().st_size
        for t in times:
            panel=np.stack(saved[t]);cs,es=reduce_panel(panel,ref,order,sizes)
            meta=dict(heads=n,control=g,time=t,seed_ids=[j['seed'] for j in group],contexts=contexts)
            cells.extend(dict(meta,**row) for row in cs);edges.extend(dict(meta,**row) for row in es)
        print(n,g,'checked',flush=True)
    summary=[]
    for t in times:
        for k in sizes:
            rows=[r for r in cells if r['time']==t and r['retained_tokens']==k]
            row=dict(time=t,retained_tokens=k,cells=len(rows),target_met=sum(r['target_met'] for r in rows))
            for key in ['relative_centered_rms','relative_uncentered_rms','retained_variance_fraction','mean_emission_kl','sufficient_kl_relative_bound']:
                row[key+'_range']=[min(r[key] for r in rows),max(r[key] for r in rows)]
            summary.append(row)
    result=dict(status='complete',schema='categorical-scale-analysis-v1',cells=cells,edges=edges,summary=summary,
        observed_paths=len(jobs),replicas_per_cell=6,contexts=contexts,scale_cells=len(cells),
        adjacent_scale_cells=len(edges),path_context_kl_chains=sum(r['path_context_chains'] for r in edges),
        maximum_algebra_error=max([r['maximum_algebra_error'] for r in cells]+[max(r['kl_chain_max_error'],r['absorption_max_error']) for r in edges]),
        training_updates=0,analysis_native_forward_calls=0,acquisition_native_forward_calls=calls,
        qualification_forward_calls=qualification,acquisition_worker_seconds=worker_seconds,
        compressed_observation_bytes=bytes_acquired,logical_logit_bytes=logical_bytes,
        source_sha256=sha256(__file__),checked_sha256=checked,analysis_seconds=time.monotonic()-started,
        scope='Finite observation scales; times and contexts share complete native training replicas. No autonomous optimizer closure or native exponent is inferred.')
    write_json(output/'analysis.json',result)
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--dictionary',type=Path)
    p.add_argument('--times',nargs='+',type=int)
    a=p.parse_args();analyze(a.study,a.output,a.dictionary,a.times)
