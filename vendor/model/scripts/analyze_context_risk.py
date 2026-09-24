"""Independently reconstruct context-resolved risk from complete saved logits.

Uses NumPy and strict input guards, without a production numerical reducer.
All resolutions and contexts are retained. Counts describe a fixed panel of
shared trained replicas, not independent prompt failure trials.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import platform
import time
import numpy as np
from numerical_validation import load_json_strict, finite_array, discrepancy


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(x):
    n = len(x)
    return sum(np.sum((x[i]-x[j])**2,axis=-1)
               for i in range(n) for j in range(i)) / (n*(n-1))


def centered(x):
    return np.sum((x-x.mean(axis=0))**2,axis=(0,2))/(len(x)-1)



def context_metrics(native, residual, target):
    """Positive-variance domain with no floor or hidden zero-variance convention."""
    native=finite_array(native,'native context variance').astype(float)
    residual=finite_array(residual,'context residual variance').astype(float)
    require(native.ndim==1 and native.size>0 and native.shape==residual.shape,
            'Invalid context variance shape')
    require(np.all(native>0) and np.all(residual>=0),'Undefined or negative context variance')
    require(type(target) in (int,float) and np.isfinite(target) and target>0,'Invalid target')
    native_total=native.sum();residual_total=residual.sum()
    finite_array([native_total,residual_total],'variance totals')
    rel=np.sqrt(residual/native);weights=native/native_total
    aggregate=float(np.sqrt(residual_total/native_total))
    finite_array(rel,'relative context RMS');finite_array([aggregate],'aggregate RMS')
    mass=float(weights[rel>target].sum());bound=min(1.,aggregate**2/target**2)
    require(mass<=bound+3e-12,'Context tail bound failed')
    discrepancy(aggregate**2,float(np.dot(weights,rel**2)),'weighted context identity',atol=3e-12)
    return dict(aggregate_rms=aggregate,per_context_rms=rel.tolist(),
                weighted_mass_over_target=mass,weighted_mass_markov_bound=bound,
                contexts_over_target=int(np.sum(rel>target)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--published-analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); start=time.perf_counter(); bound={}
    if args.output.exists():raise FileExistsError(args.output)
    def bind(f):
        h=sha(f);bound[str(f)]=h;return h
    protocol_path=args.study/'protocol.json'
    protocol=load_json_strict(protocol_path.read_text());protocol_hash=bind(protocol_path)
    contexts=protocol['contexts']
    require(type(contexts) is int and contexts>=8 and contexts%8==0,'Invalid context count')
    require(protocol['schema']=='context-categorical-v1','Unexpected study schema')
    plan_path=args.study/'risk-plan.json'
    if plan_path.exists():
        plan=load_json_strict(plan_path.read_text());bind(plan_path)
        require(plan['protocol_sha256']==protocol_hash and plan['contexts']==contexts and
                plan['primary_resolution']==8192 and plan['target']==.25 and
                plan['resolutions']==protocol['sizes'],'Changed context-risk plan')
    inputs=args.study/'inputs.npz'
    require(bind(inputs)==protocol['selection_sha256'],'Changed frozen inputs')
    with np.load(inputs) as a:
        ref=a['reference'].astype(np.float64);order=a['order'].copy();blocks=a['blocks']
    require(np.isfinite(ref).all() and (ref>0).all(),'Invalid reference')
    require(np.isclose(ref.sum(),1,rtol=0,atol=1e-12),'Unnormalized reference')
    require(np.array_equal(np.sort(order),np.arange(len(ref))),'Invalid vocabulary order')
    require(blocks.shape==(protocol['contexts'],65),'Unexpected frozen input shape')
    require(len(protocol['document_hashes'])==contexts and len(set(protocol['document_hashes']))==contexts,'Repeated or missing documents')
    require(protocol['sizes']==[128,512,2048,8192,16384] and protocol['target']==.25,
            'The frozen scale or target changed')
    grouped=defaultdict(list)
    for job in protocol['jobs']:grouped[job['heads'],job['control']].append(job)
    require(set(grouped)=={(8,0),(8,1.5),(24,0),(24,1.5)},'Width/control inventory differs')
    rows=[];max_pair_error=0.;max_identity_error=0.;chains=0;max_chain_error=0.
    for (heads,control),jobs in sorted(grouped.items()):
        jobs=sorted(jobs,key=lambda j:j['seed']);prob=[]
        require(len(jobs)==6 and len({j['seed'] for j in jobs})==6,'Replica inventory mismatch')
        for job in jobs:
            root=args.study/'runs'/job['run_id']
            mfile=root/'manifest.json';manifest=load_json_strict(mfile.read_text());bind(mfile)
            require(manifest['status']=='complete' and manifest['job']==job,'Manifest job mismatch')
            require(manifest['protocol_sha256']==protocol_hash,'Manifest protocol mismatch')
            file=root/'observations.npz'
            require(bind(file)==manifest['observations_sha256'],'Observation hash mismatch')
            with np.load(file) as a:z=a['logits'].astype(np.float64)
            require(z.shape==(contexts,protocol['vocabulary']) and np.isfinite(z).all(),'Invalid logits')
            u=np.exp(z-z.max(axis=-1,keepdims=True));prob.append(u/u.sum(axis=-1,keepdims=True))
        prob=np.stack(prob);full=2*np.sqrt(prob);vp=pairs(full)
        require(np.isfinite(vp).all() and (vp>0).all(),'Zero or invalid native variance')
        max_pair_error=max(max_pair_error,float(np.max(np.abs(vp-centered(full)))))
        previous=None
        for size in protocol['sizes']:
            q=prob.copy();tail=order[size:]
            q[:,:,tail]=prob[:,:,tail].sum(axis=-1,keepdims=True)*ref[tail]/ref[tail].sum()
            require(np.isfinite(q).all() and (q>0).all(),'Invalid lifted distribution')
            require(np.max(np.abs(q.sum(axis=-1)-1))<1e-12,'Projection lost mass')
            lifted=2*np.sqrt(q);e=full-lifted
            vq=pairs(lifted);ve=pairs(e)
            ebar=e.mean(axis=0);qbar=lifted.mean(axis=0)
            cross=np.sum((lifted-qbar)*(e-ebar),axis=(0,2))/5
            energy=np.mean(np.sum(e**2,axis=-1),axis=0)
            bias=np.sum(ebar**2,axis=-1)
            kl=np.sum(prob*(np.log(prob)-np.log(q)),axis=-1)
            metrics=context_metrics(vp,ve,.25)
            rel=np.asarray(metrics['per_context_rms']);agg=metrics['aggregate_rms']
            weights=vp/vp.sum()
            max_pair_error=max(max_pair_error,float(np.max(np.abs(ve-centered(e)))),
                               float(np.max(np.abs(vq-centered(lifted)))))
            max_identity_error=max(max_identity_error,
                float(np.max(np.abs(vp-vq-ve-2*cross))),
                float(np.max(np.abs(ve-6/5*(energy-bias)))),
                abs(agg**2-float(np.sum(weights*rel**2))))
            if previous is not None:
                oldq,oldkl=previous
                stepkl=np.sum(q*(np.log(q)-np.log(oldq)),axis=-1)
                max_chain_error=max(max_chain_error,float(np.max(np.abs(oldkl-kl-stepkl))))
                chains+=kl.size
            previous=(q,kl)
            row=dict(heads=heads,control=control,retained_tokens=size,contexts=contexts,replicas=6,
                aggregate_rms=agg,aggregate_retention=float(vq.sum()/vp.sum()),
                native_variance_mean=float(vp.mean()),residual_variance_mean=float(ve.mean()),
                coarse_variance_mean=float(vq.mean()),signed_cross_covariance=float(cross.mean()),
                uncentered_energy=float(energy.mean()),mean_bias_energy=float(bias.mean()),
                mean_emission_kl=float(kl.mean()),
                per_context_rms_min=float(rel.min()),per_context_rms_median=float(np.median(rel)),
                per_context_rms_max=float(rel.max()),contexts_over_target=int(np.sum(rel>.25)),
                weighted_mass_over_target=float(weights[rel>.25].sum()),
                weighted_mass_markov_bound=min(1.,agg**2/.25**2),
                native_variance=vp.tolist(),residual_variance=ve.tolist(),per_context_rms=rel.tolist())
            require(row['weighted_mass_over_target']<=row['weighted_mass_markov_bound']+1e-12,
                    'Weighted exceedance bound failed')
            rows.append(row)
    # Compare only after the independent numerical reconstruction is complete.
    published=load_json_strict(args.published_analysis.read_text());bind(args.published_analysis)
    lookup={(c['heads'],c['control'],c['retained_tokens']):c for c in published['cells']}
    require(len(lookup)==len(rows)==20,'Published cell inventory mismatch')
    mapping={'aggregate_rms':'relative_centered_rms','aggregate_retention':'retained_variance_fraction',
             'native_variance_mean':'native_variance','residual_variance_mean':'centered_remainder_variance',
             'coarse_variance_mean':'coarse_variance','mean_emission_kl':'mean_emission_kl',
             'signed_cross_covariance':'signed_cross_covariance','uncentered_energy':'uncentered_energy',
             'mean_bias_energy':'mean_bias_energy'}
    max_published=0.
    for row in rows:
        old=lookup[row['heads'],row['control'],row['retained_tokens']]
        for k,v in mapping.items():
            max_published=max(max_published,discrepancy(row[k],old[v],k,atol=3e-12))
    errors=[max_pair_error,max_identity_error,max_chain_error,max_published]
    require(np.isfinite(errors).all() and max(errors)<3e-12,'Numerical check failed')
    result=dict(schema='context-disaggregation-v1',status='passed',
        rows=rows,resolution_cells=len(rows),context_resolution_cells=sum(r['contexts'] for r in rows),
        path_context_kl_chains=chains,maximum_pair_centering_discrepancy=max_pair_error,
        maximum_identity_residual=max_identity_error,maximum_kl_chain_residual=max_chain_error,
        maximum_published_cell_discrepancy=max_published,
        tolerance=3e-12,checked_sha256=bound,source_sha256=sha(Path(__file__)),
        support_sha256={str(Path(__file__).with_name('numerical_validation.py').resolve()):sha(Path(__file__).with_name('numerical_validation.py'))},
        elapsed_seconds=time.perf_counter()-start,python=platform.python_version(),numpy=np.__version__,
        training_updates=0,native_forward_calls=0,new_training_replicas=0,
        scope='Saved numerical observations only; historical training states not rehashed or resumed.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','checked_sha256']},indent=2))
    for r in rows:
        if r['retained_tokens']==8192:
            print(json.dumps({k:v for k,v in r.items() if not isinstance(v,list)}))


if __name__=='__main__':
    main()
