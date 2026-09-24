#!/usr/bin/env python
"""Finite source responses, untouched-mixture forecasts and conditional size diagnostics.

No critical exponent is inferred from a finite covariance slope. The four body
initializations are the conditional statistical units; documents are observation
coordinates. All registered scientific cells are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def probabilities(z):
    a=np.asarray(z,dtype=np.float64);a=a-a.max(-1,keepdims=True)
    return np.exp(a)/np.exp(a).sum(-1,keepdims=True)


def fisher_product(x,y,p):
    """Per-document categorical Fisher inner product of two logit changes."""
    return (p*x*y).sum(-1)-(p*x).sum(-1)*(p*y).sum(-1)


def fisher_gram(changes,p):
    a=np.stack(changes)
    return np.stack([np.stack([fisher_product(x,y,p) for y in a],-1) for x in a],-2)


def matrix_summary(g):
    eig=np.linalg.eigvalsh(g)
    if eig[0]<-1e-9*max(1.,eig[-1]):raise ValueError('Nonpositive response Gram')
    eig=np.maximum(eig,0)
    return dict(gram=g.tolist(),eigenvalues=eig[::-1].tolist(),
                second_to_first=float(eig[0]/eig[-1]) if eig[-1]>0 else None,
                cosine=float(g[0,1]/np.sqrt(g[0,0]*g[1,1])) if g[0,0]*g[1,1]>0 else None,
                second_above_reporting_floor=bool(np.sqrt(eig[0])>5e-7))


def psd_inverse_sqrt(g):
    w,v=np.linalg.eigh(g)
    if w[0]<=1e-13*w[-1]:return None
    return (v/np.sqrt(w))@v.T


def canonical_overlap(g,h,cross):
    a=psd_inverse_sqrt(g);b=psd_inverse_sqrt(h)
    if a is None or b is None:return None
    values=np.linalg.svd(a@cross@b,compute_uv=False)
    if values[0]>1+1e-8:raise ValueError('Incompatible subspace inner products')
    return np.minimum(values,1.).tolist()


def risk(z,targets):
    z=np.asarray(z,dtype=np.float64);m=z.max(-1)
    return np.log(np.exp(z-m[:,None]).sum(-1))+m-z[np.arange(len(z)),targets]


def cohort_masks(indices,labels,split):
    result={'all':np.ones(len(indices),bool),'heldout':split[indices]==1,'calibration':split[indices]==0}
    for label,name in [(1,'technical'),(0,'narrative'),(-1,'unassigned')]:
        result[name]=(labels[indices]==label)
        result[name+'-heldout']=(labels[indices]==label)&(split[indices]==1)
    fixed=(indices % 80)<16
    for name,mask in list(result.items()):result[name+'-fixed']=mask & fixed
    return result


def load(path,keys=None,indices=None):
    with np.load(path) as f:
        allids=f['indices'];pos=None
        if indices is not None:
            pos=np.searchsorted(allids,indices)
            if not np.array_equal(allids[pos],indices):raise ValueError('Observation panels disagree')
        return {k:(f[k] if pos is None else f[k][pos]) for k in (f.files if keys is None else keys)}


def outcomes(study,spec,labels,split,crops):
    rows=[];response=[];forecasts=[];alignment=[];cache={};input_files={}
    def bound(path,**kw):
        path=Path(path)
        if str(path) not in input_files:input_files[str(path)]=sha(path)
        return load(path,**kw)
    for case in spec['cases']:
        name=case['name'];heads=case['heads'];folder=study/name
        result=json.loads((folder/'results.json').read_text())
        if result['status']!='complete':raise ValueError('Case incomplete')
        records=[r for r in result['records'] if r['arm']['role'] not in ['replay','unobserved replay']]
        baseline=bound(folder/'initial-observation.npz',keys=['indices','logits32','logits64','common','energy','total','paired_sse','reference_sum','reference_sumsq','attention_entropy'])
        for time in spec['times']:
            if time==0:jobs=[('incoming',None,baseline)]
            else:jobs=[(r['arm']['name'],r['arm'],bound(next(o['file'] for o in r['observations'] if o['time']==time),
                keys=['indices','logits32','logits64','common','energy','total','paired_sse','reference_sum','reference_sumsq','attention_entropy'])) for r in records]
            # Temporary arrays are released after each saved-time panel.
            byarm={}
            for arm,arm_spec,f in jobs:
                ids=f['indices'];masks=cohort_masks(ids,labels,split)
                risks={p:risk(f['logits'+str(p)],crops[ids,p]) for p in [32,64]}
                common=f['common'];headmean=common.mean(2)
                entropy=f['attention_entropy'].mean(2)
                cache[(heads,case['seed'],arm,time)]={'ids':ids,'common':headmean,'attention':entropy,'risk':risks[64]}
                byarm[arm]=f
                for cohort,mask in masks.items():
                    n=int(mask.sum())
                    if not n:continue
                    entries=n*5*heads*4096
                    sse=f['paired_sse'][mask].sum((0,1,2));ref=f['reference_sum'][mask].sum((0,1,2))
                    refsq=f['reference_sumsq'][mask].sum((0,1,2));mean=ref/entries
                    rmse=np.sqrt(sse/entries)
                    rows.append(dict(case=name,heads=heads,seed=case['seed'],arm=arm,time=time,cohort=cohort,
                        documents=n,risk32=float(risks[32][mask].mean()),risk64=float(risks[64][mask].mean()),
                        row_energy=float(f['energy'][mask].mean()),total_energy=float(f['total'][mask].mean()),
                        row_fraction=float(f['energy'][mask].sum()/f['total'][mask].sum()),
                        common_rms=float(np.sqrt(np.square(common[mask]).mean())),
                        attention_entropy=float(entropy[mask].mean()),paired_rmse=rmse.tolist(),reference_mean=mean.tolist(),
                        order_mean=[float(r/abs(m)) if m!=0 else None for r,m in zip(rmse,mean)],
                        order_rms=[float(np.sqrt(x/y)) if y>0 else None for x,y in zip(sse,refsq)]))
            if time==0:continue
            ids=byarm['general']['indices'];p0=probabilities(baseline['logits64'][ids])
            masks=cohort_masks(ids,labels,split)
            base=byarm['general']['logits64'].astype(np.float64)
            left=byarm['mix0000']['logits64'].astype(np.float64)-base
            right=byarm['mix1000']['logits64'].astype(np.float64)-base
            middle=byarm['mix0500']['logits64'].astype(np.float64)-base
            g=fisher_gram([left,right],p0)
            commonbase=byarm['general']['common']
            dl=(byarm['mix0000']['common']-commonbase).reshape(len(ids),-1)
            dr=(byarm['mix1000']['common']-commonbase).reshape(len(ids),-1)
            cg=np.stack([np.stack([(dl*x).mean(-1),(dr*x).mean(-1)],-1) for x in [dl,dr]],-2)
            for cohort,mask in masks.items():
                response.append(dict(case=name,heads=heads,seed=case['seed'],time=time,cohort=cohort,
                    documents=int(mask.sum()),predictive=matrix_summary(g[mask].mean(0)),common=matrix_summary(cg[mask].mean(0))))
            for rho in [.125,.25,.75,.875]:
                actual=byarm[f'mix{int(1000*rho):04d}']['logits64'].astype(np.float64)-base
                linear=(1-rho)*left+rho*right
                quad=linear+4*rho*(1-rho)*(middle-(left+right)/2)
                signal=np.maximum(0.,fisher_product(actual,actual,p0))
                e1=np.maximum(0.,fisher_product(actual-linear,actual-linear,p0))
                e2=np.maximum(0.,fisher_product(actual-quad,actual-quad,p0))
                for cohort,mask in masks.items():
                    den=float(signal[mask].mean());a=float(e1[mask].mean());b=float(e2[mask].mean())
                    forecasts.append(dict(case=name,heads=heads,seed=case['seed'],time=time,rho=rho,cohort=cohort,
                        signal_rms=float(np.sqrt(den)),affine_rms=float(np.sqrt(a)),quadratic_rms=float(np.sqrt(b)),
                        affine_relative=float(np.sqrt(a/den)) if den>0 else None,
                        quadratic_relative=float(np.sqrt(b/den)) if den>0 else None,
                        improvement=bool(b<a)))
            del jobs,byarm,base,left,right,middle,p0,g,cg
        # Compare fixed 96-document source subspaces at early and final times.
        fixed=np.array([j+k for j in [0,80,160,240,320,400] for k in range(16)])
        p0=probabilities(baseline['logits64'][fixed]);masks=cohort_masks(fixed,labels,split)
        def contrasts(t):
            paths={r['arm']['name']:next(o['file'] for o in r['observations'] if o['time']==t)
                   for r in records if r['arm']['name'] in ['mix0000','mix1000','general']}
            a={n:bound(p,keys=['logits64'],indices=fixed)['logits64'].astype(np.float64) for n,p in paths.items()}
            return [a[n]-a['general'] for n in ['mix0000','mix1000']]
        final=contrasts(spec['horizon']);gf=fisher_gram(final,p0)
        for t in [t for t in spec['times'] if t not in [0,spec['horizon']]]:
            early=contrasts(t);ge=fisher_gram(early,p0)
            cross=np.stack([np.stack([fisher_product(x,y,p0) for y in final],-1) for x in early],-2)
            for cohort,mask in masks.items():
                alignment.append(dict(case=name,heads=heads,seed=case['seed'],early_time=t,final_time=spec['horizon'],cohort=cohort,
                    early_gram=ge[mask].mean(0).tolist(),final_gram=gf[mask].mean(0).tolist(),cross_gram=cross[mask].mean(0).tolist(),
                    canonical_cosines=canonical_overlap(ge[mask].mean(0),gf[mask].mean(0),cross[mask].mean(0))))
        print('analyzed',name,flush=True)
    return rows,response,forecasts,alignment,cache,input_files


def susceptibilities(cache,spec,labels,split):
    result=[]
    for heads in [4,8,14]:
        seeds=sorted(c['seed'] for c in spec['cases'] if c['heads']==heads)
        if len(seeds)!=4:raise ValueError('Size diagnostics require four complete body initializations')
        for time in spec['times']:
            arms=['incoming'] if time==0 else [a['name'] for a in spec['arms'] if a['role']=='primary']
            for arm in arms:
                parts=[cache[(heads,s,arm,time)] for s in seeds];ids=parts[0]['ids']
                if any(not np.array_equal(p['ids'],ids) for p in parts):raise ValueError('Seed panels differ')
                for cohort,mask in cohort_masks(ids,labels,split).items():
                    if not mask.any():continue
                    c=np.stack([p['common'][mask] for p in parts]);c=c-c.mean(0)
                    h=np.stack([p['attention'][mask] for p in parts]);h=h-h.mean(0)
                    r=np.stack([p['risk'][mask] for p in parts]);r=r-r.mean(0)
                    # Covariance is centered across initializations at each input.
                    cov=np.einsum('scld,scmd->lm',c,c)/(3*mask.sum()*64)
                    eig=np.linalg.eigvalsh(cov)
                    common=float(heads*np.trace(cov)/5)
                    result.append(dict(heads=heads,arm=arm,time=time,cohort=cohort,documents=int(mask.sum()),
                        common_chi=common,common_intensive=common/heads,
                        attention_chi=float(heads*np.square(h).sum()/(3*mask.sum()*5)),
                        risk_chi=float(heads*np.square(r).sum()/(3*mask.sum())),
                        layer_covariance=cov.tolist(),layer_covariance_eigenvalues=eig[::-1].tolist(),
                        cross_layer_fraction=float((np.abs(cov).sum()-np.abs(np.diag(cov)).sum())/np.abs(cov).sum()) if np.abs(cov).sum()>0 else None))
    return result


def size_diagnostics(susceptibility,spec):
    out=[]
    for t in spec['times']:
        arms=['incoming'] if t==0 else [a['name'] for a in spec['arms'] if a['role']=='primary']
        for arm in arms:
            for cohort in ['all','heldout','heldout-fixed','technical-heldout','narrative-heldout','unassigned-heldout']:
                rows=sorted([r for r in susceptibility if r['time']==t and r['arm']==arm and r['cohort']==cohort],key=lambda r:r['heads'])
                for field in ['common_chi','attention_chi','risk_chi']:
                    values=np.array([r[field] for r in rows]);sizes=np.array([r['heads'] for r in rows])
                    if len(rows)!=3 or np.any(values<0):raise ValueError('Incomplete nonnegative size panel')
                    slopes=[float(np.log(values[i+1]/values[i])/np.log(sizes[i+1]/sizes[i])) if values[i]>0 and values[i+1]>0 else None for i in range(2)]
                    endpoint=float(np.log(values[2]/values[0])/np.log(14/4)) if values[0]>0 and values[2]>0 else None
                    predicted=float(values[0]*(8/4)**endpoint) if endpoint is not None else None
                    out.append(dict(time=t,arm=arm,cohort=cohort,observable=field,values=values.tolist(),
                        adjacent_slopes=slopes,endpoint_slope=endpoint,
                        middle_prediction=predicted,middle_relative_error=float(abs(predicted-values[1])/values[1]) if predicted is not None and values[1]>0 else None,
                        interpretation='Finite size secants conditional on the fixed corpus, optimizer schedule and initial metric learner; not critical exponents.'))
    return out


def analyze(study):
    study=Path(study).resolve();spec=json.loads((study/'protocol.json').read_text())
    if spec['stage']!='assessment':raise ValueError('Scientific assessment required')
    native=json.loads((study/'verification.json').read_text())
    if native['status']!='passed' or native['protocol_sha256']!=sha(study/'protocol.json'):raise ValueError('Independent verification required')
    with np.load(Path(spec['data'])/'evaluation.npz') as f:labels=f['labels'];split=f['split'];crops=f['crops']
    rows,response,forecasts,alignment,cache,inputs=outcomes(study,spec,labels,split,crops)
    susceptibility=susceptibilities(cache,spec,labels,split)
    result=dict(schema='pldr-finetuning-analysis-v1',status='complete',protocol_sha256=sha(study/'protocol.json'),
        verification_sha256=sha(study/'verification.json'),analyzer_sha256=sha(__file__),
        primary_paths=96,control_paths=21,scientific_updates=59904,replay_updates=192,
        statistical_units=spec['statistical_units'],outcomes=rows,responses=response,mixture_forecasts=forecasts,
        time_alignment=alignment,susceptibilities=susceptibility,size_diagnostics=size_diagnostics(susceptibility,spec),
        input_sha256=inputs,scope='Complete registered panel. Conditional body-seed covariance at fixed input, categorical Fisher source contrasts and interpolation scored only at untouched mixture values. Source directions and descriptive size slopes are not critical modes or critical exponents.')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    value=analyze(a.study);Path(a.output).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print('Complete fine-tuning analysis:',a.output)
