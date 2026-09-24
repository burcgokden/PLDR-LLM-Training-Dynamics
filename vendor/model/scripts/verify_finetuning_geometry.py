#!/usr/bin/env python
"""Independent full-vocabulary reconstruction of source-response geometry.

Uses explicitly centered, square-root-weighted logit vectors. Imports no
producer, primary verifier, analyzer, renderer or model helper.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--return-study',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    study=Path(a.study);returned=Path(a.return_study);spec=json.loads((study/'protocol.json').read_text())
    analysis=json.loads((study/'analysis.json').read_text());rdata=json.loads((returned/'analysis.json').read_text());rspec=json.loads((returned/'protocol.json').read_text())
    checked={};errors={};counts={};cache={}
    def check(path):
        path=Path(path)
        if str(path) not in checked:checked[str(path)]=sha(path)
    for folder in [study,returned]:
        for n in ['analysis.json','protocol.json','verification.json']:check(folder/n)
    def read(path,keys):
        check(path)
        with np.load(path) as f:return {k:f[k] for k in keys}
    def compare(tag,x,y):
        x=np.asarray(x);y=np.asarray(y)
        if x.shape!=y.shape:raise ValueError('Geometry shape mismatch: '+tag)
        err=float(np.max(np.abs(x-y)));errors[tag]=max(errors.get(tag,0.),err);counts[tag]=counts.get(tag,0)+1
        if not np.allclose(x,y,atol=3e-12,rtol=3e-9):raise ValueError('Independent geometry mismatch: '+tag)
    with np.load(Path(spec['data'])/'evaluation.npz') as f:labels=f['labels'];split=f['split'];crops=f['crops']
    def masks(ids):
        d={'all':np.ones(len(ids),bool),'heldout':split[ids]==1,'calibration':split[ids]==0}
        for k,name in [(1,'technical'),(0,'narrative'),(-1,'unassigned')]:
            d[name]=labels[ids]==k;d[name+'-heldout']=(labels[ids]==k)&(split[ids]==1)
        for name,v in list(d.items()):d[name+'-fixed']=v&((ids%80)<16)
        return d
    def prob(z):
        e=np.exp(z.astype(np.float64)-z.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)
    def embed(z,p):
        z=z.astype(np.float64);return np.sqrt(p)*(z-(p*z).sum(-1,keepdims=True))
    def grams(x,y):
        return np.stack([np.stack([np.sum(u*v,axis=-1) for v in y],axis=-1) for u in x],axis=-2)
    row_indices={}
    def getrow(rows,**keys):
        names=tuple(keys);tag=(id(rows),names)
        if tag not in row_indices:
            index={tuple(r[k] for k in names):r for r in rows}
            if len(index)!=len(rows):raise ValueError('Duplicate geometry row')
            row_indices[tag]=index
        key=tuple(keys[k] for k in names)
        if key not in row_indices[tag]:raise ValueError('Missing geometry row')
        return row_indices[tag][key]
    for case in spec['cases']:
        name=case['name'];base=study/name;initial=read(base/'initial-observation.npz',['logits64','indices','common','attention_entropy'])
        pre=prob(initial['logits64']);initial_d=None;frames={}
        for t in spec['times']:
            names=['incoming'] if t==0 else [r['name'] for r in spec['arms'] if r['role']=='primary']
            panel={n:initial if t==0 else read(base/n/f'observation-{t:04d}.npz',['indices','logits64','common','attention_entropy']) for n in names}
            ids=panel[names[0]]['indices'];m=masks(ids)
            for n,f in panel.items():
                z=f['logits64'].astype(np.float64);p=prob(z)
                risk=-np.log(p[np.arange(len(ids)),crops[ids,64]])
                cache[(case['heads'],case['seed'],n,t)]={'ids':ids,'common':f['common'].mean(2),'entropy':f['attention_entropy'].mean(2),'risk':risk}
                for cohort,mask in m.items():
                    row=getrow(analysis['outcomes'],case=name,arm=n,time=t,cohort=cohort)
                    compare('risk64',row['risk64'],risk[mask].mean())
            if t==0:continue
            p=pre[ids];general=panel['general']['logits64'].astype(np.float64)
            raw=[panel[n]['logits64'].astype(np.float64)-general for n in ['mix0000','mix1000']]
            x=[embed(u,p) for u in raw];g=grams(x,x)
            frames[t]=[u[(ids%80)<16] for u in x]
            metric=[(panel[n]['common']-panel['general']['common']).reshape(len(ids),-1) for n in ['mix0000','mix1000']]
            mg=grams(metric,metric)/metric[0].shape[-1]
            for cohort,mask in m.items():
                row=getrow(analysis['responses'],case=name,time=t,cohort=cohort)
                compare('predictive_gram',row['predictive']['gram'],g[mask].mean(0));compare('common_gram',row['common']['gram'],mg[mask].mean(0))
            middle=panel['mix0500']['logits64'].astype(np.float64)-general
            for rho in [.125,.25,.75,.875]:
                target=panel[f'mix{int(rho*1000):04d}']['logits64'].astype(np.float64)-general
                linear=(1-rho)*raw[0]+rho*raw[1];quadratic=linear+4*rho*(1-rho)*(middle-(raw[0]+raw[1])/2)
                energy=[np.square(embed(z,p)).sum(-1) for z in [target,target-linear,target-quadratic]]
                for cohort,mask in m.items():
                    row=getrow(analysis['mixture_forecasts'],case=name,time=t,rho=rho,cohort=cohort)
                    for key,e in zip(['signal_rms','affine_rms','quadratic_rms'],energy):compare(key,row[key],np.sqrt(e[mask].mean()))
            if t==512:
                initial_d=raw
                dictionary_names=['mix0000','mix1000','mix0125','mix0250','mix0500','mix0750','mix0875']
                dictionary_vectors=[embed(panel[n]['logits64'].astype(np.float64)-general,p) for n in dictionary_names]
                dictionary_gram=grams(dictionary_vectors,dictionary_vectors)
        fixed=np.array([j+k for j in [0,80,160,240,320,400] for k in range(16)])
        for t in [16,64,128,256]:
            for cohort,mask in masks(fixed).items():
                row=getrow(analysis['time_alignment'],case=name,early_time=t,cohort=cohort)
                for key,x,y in [('early_gram',frames[t],frames[t]),('final_gram',frames[512],frames[512]),('cross_gram',frames[t],frames[512])]:
                    compare('alignment_'+key,row[key],grams(x,y)[mask].mean(0))
        # Independently reconstruct every return Gram, with the same incoming Fisher weights.
        result=json.loads((returned/name/'results.json').read_text());records={r['name']:r for r in result['records'] if r['role']=='scientific'}
        for t in rspec['times'][1:]:
            panel={n:read(next(o['file'] for o in r['observations'] if o['time']==t),['indices','logits64']) for n,r in records.items()}
            ids=panel['general']['indices'];p=pre[ids];m=masks(ids)
            x=[embed(z[ids],p) for z in initial_d]
            y=[embed(panel[n]['logits64'].astype(np.float64)-panel['general']['logits64'],p) for n in ['mix0000','mix1000']]
            g=grams(x,x);h=grams(y,y);c=grams(x,y)
            dg=dictionary_gram[ids];dc=grams([z[ids] for z in dictionary_vectors],y)
            for cohort,mask in m.items():
                row=getrow(rdata['rows'],case=name,time=t,cohort=cohort)
                calname='calibration-fixed' if cohort.endswith('-fixed') else 'calibration';cm=m[calname]
                if row['calibration_cohort']!=calname or row['calibration_documents']!=int(cm.sum()):raise ValueError('Calibration panel budget changed')
                for key,mat in [('incoming_gram',g),('current_gram',h),('cross_gram',c)]:compare('return_'+key,row[key],mat[mask].mean(0))
                compare('return_calibration_gram',row['calibration_gram'],g[cm].mean(0))
                compare('return_calibration_cross',row['calibration_cross'],c[cm].mean(0))
                q=row['expanded']
                if q['names']!=dictionary_names:raise ValueError('Expanded dictionary changed')
                compare('expanded_gram',q['incoming_gram'],dg[mask].mean(0));compare('expanded_cross',q['cross_gram'],dc[mask].mean(0))
                compare('expanded_calibration_gram',q['calibration_gram'],dg[cm].mean(0))
                compare('expanded_calibration_cross',q['calibration_cross'],dc[cm].mean(0))
        print('verified geometry',name,flush=True)
    for row in analysis['susceptibilities']:
        h=row['heads'];t=row['time'];arm=row['arm'];parts=[cache[(h,s,arm,t)] for s in [640101,640102,640103,640104]]
        mask=masks(parts[0]['ids'])[row['cohort']];n=int(mask.sum())
        x=np.array([p['common'][mask] for p in parts]);x=x-x.mean(0)
        # Explicit layer-pair loops use the sample divisor at each fixed input.
        cov=np.array([[np.sum(x[:,:,l,:]*x[:,:,k,:])/(3*n*64) for k in range(5)] for l in range(5)])
        compare('layer_covariance',row['layer_covariance'],cov)
        compare('common_chi',row['common_chi'],h*np.trace(cov)/5)
        for field,key in [('entropy','attention_chi'),('risk','risk_chi')]:
            y=np.array([p[field][mask] for p in parts]);value=h*np.var(y,axis=0,ddof=1).mean()
            compare(key,row[key],value)
    value=dict(status='passed',schema='pldr-finetuning-geometry-verification-v1',verifier_sha256=sha(__file__),
               verified_files=checked,reconstructed_rows=counts,maximum_errors=errors,
               scope='Independent square-root-weighted full-vocabulary reconstruction of every primary response Gram and untouched-mixture error, every source-return Gram for both nested dictionaries, and all input-conditional seed susceptibilities. Control risks and tensor reductions are separately checked by the native artifact verifiers.')
    Path(a.output).write_text(json.dumps(value,indent=2)+'\n');print('Complete independent geometry verification')


if __name__=='__main__':main()
