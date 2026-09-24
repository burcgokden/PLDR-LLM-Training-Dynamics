#!/usr/bin/env python
"""Calibrate finite source-span transport and test it on untouched documents."""
import argparse
import json
from pathlib import Path
import numpy as np
try:
    from .analyze_finetuning import sha,probabilities,fisher_gram,fisher_product,cohort_masks,canonical_overlap,load
except ImportError:
    from analyze_finetuning import sha,probabilities,fisher_gram,fisher_product,cohort_masks,canonical_overlap,load


def transport(g0cal,crosscal,g0,gt,cross):
    eigen=np.linalg.eigvalsh(g0cal)
    if eigen[0]<=1e-12*eigen[-1]:raise ValueError('Incoming calibration span unresolved')
    matrix=np.linalg.solve(g0cal,crosscal)
    residual=gt-matrix.T@cross-cross.T@matrix+matrix.T@g0@matrix
    residual=(residual+residual.T)/2
    if np.linalg.eigvalsh(residual)[0]<-1e-9*max(1.,np.trace(gt)):raise ValueError('Residual Gram is not positive')
    error=float(np.sqrt(max(0.,np.trace(residual))))
    incoming=np.linalg.eigvalsh(g0)[::-1];current=np.linalg.eigvalsh(gt)[::-1]
    singular=np.linalg.svd(matrix,compute_uv=False)
    lower=np.maximum(0.,singular[-1]*np.sqrt(np.maximum(incoming,0))-error)
    upper=singular[0]*np.sqrt(np.maximum(incoming,0))+error
    if np.any(np.sqrt(np.maximum(current,0))<lower-1e-8) or np.any(np.sqrt(np.maximum(current,0))>upper+1e-8):raise ValueError('Finite singular-value transfer inequality failed')
    unchanged=gt-cross-cross.T+g0
    return dict(matrix=matrix.tolist(),singular_values=singular.tolist(),incoming_eigenvalues=incoming.tolist(),current_eigenvalues=current.tolist(),
        residual_gram=residual.tolist(),residual_rms=error,relative_residual=float(error/np.sqrt(np.trace(gt))),
        residual_to_weak_incoming=float(error/np.sqrt(incoming[-1])),
        unchanged_rms=float(np.sqrt(max(0.,np.trace(unchanged)))),lower_singular_bounds=lower.tolist(),upper_singular_bounds=upper.tolist(),
        weak_lower_bound_positive=bool(lower[-1]>0),canonical_cosines=canonical_overlap(g0,gt,cross))



def dictionary_transport(gcal,ccal,g,gt,cross):
    eigen=np.linalg.eigvalsh(gcal)
    if eigen[0]<=1e-12*eigen[-1]:raise ValueError('Expanded calibration dictionary unresolved')
    matrix=np.linalg.solve(gcal,ccal)
    residual=gt-matrix.T@cross-cross.T@matrix+matrix.T@g@matrix
    residual=(residual+residual.T)/2
    if np.linalg.eigvalsh(residual)[0]<-1e-9*max(1.,np.trace(gt)):raise ValueError('Expanded residual not positive')
    error=float(np.sqrt(max(0.,np.trace(residual))))
    return dict(matrix=matrix.tolist(),residual_gram=residual.tolist(),residual_rms=error,
                relative_residual=float(error/np.sqrt(np.trace(gt))),calibration_eigenvalues=eigen[::-1].tolist(),
                calibration_condition=float(eigen[-1]/eigen[0]))


def analyze(study):
    study=Path(study).resolve();spec=json.loads((study/'protocol.json').read_text())
    v=json.loads((study/'verification.json').read_text())
    if spec['stage']!='assessment' or v['status']!='passed' or v['protocol_sha256']!=sha(study/'protocol.json'):raise ValueError('Verified assessment required')
    parent=Path(spec['parent_study']);inputs={}
    with np.load(Path(spec['data'])/'evaluation.npz') as f:labels=f['labels'];split=f['split']
    def bound(path,**kw):
        if str(path) not in inputs:inputs[str(path)]=sha(path)
        return load(path,**kw)
    rows=[]
    for case in spec['cases']:
        folder=study/case['name'];records=json.loads((folder/'results.json').read_text())['records']
        records={r['name']:r for r in records if r['role']=='scientific'}
        p0full=probabilities(bound(parent/case['name']/'initial-observation.npz',keys=['logits64'])['logits64'])
        initial={n:bound(folder/n/'observation-0000.npz',keys=['logits64'])['logits64'].astype(np.float64) for n in records}
        initial_d=[initial[n]-initial['general'] for n in ['mix0000','mix1000']]
        dictionary_names=['mix0000','mix1000','mix0125','mix0250','mix0500','mix0750','mix0875']
        dictionary=initial_d+[bound(parent/case['name']/n/'observation-0512.npz',keys=['logits64'])['logits64'].astype(np.float64)-initial['general'] for n in dictionary_names[2:]]
        dictionary_gram=fisher_gram(dictionary,p0full)

        for t in spec['times'][1:]:
            current={n:bound(next(o['file'] for o in r['observations'] if o['time']==t),keys=['indices','logits64']) for n,r in records.items()}
            ids=current['general']['indices'];p0=p0full[ids]
            d0=[d[ids] for d in initial_d]
            dt=[current[n]['logits64'].astype(np.float64)-current['general']['logits64'] for n in ['mix0000','mix1000']]
            g0=fisher_gram(d0,p0);gt=fisher_gram(dt,p0)
            cross=np.stack([np.stack([fisher_product(x,y,p0) for y in dt],-1) for x in d0],-2)
            masks=cohort_masks(ids,labels,split)
            dg=dictionary_gram[ids]
            dc=np.stack([np.stack([fisher_product(x[ids],y,p0) for y in dt],-1) for x in dictionary],-2)
            for cohort,mask in masks.items():
                calname='calibration-fixed' if cohort.endswith('-fixed') else 'calibration'
                cal=masks[calname];gcal=g0[cal].mean(0);ccal=cross[cal].mean(0)
                dcal=dg[cal].mean(0);dccal=dc[cal].mean(0)
                rows.append(dict(case=case['name'],heads=case['heads'],seed=case['seed'],time=t,cohort=cohort,documents=int(mask.sum()),calibration_cohort=calname,calibration_documents=int(cal.sum()),
                    calibration_gram=gcal.tolist(),calibration_cross=ccal.tolist(),incoming_gram=g0[mask].mean(0).tolist(),current_gram=gt[mask].mean(0).tolist(),cross_gram=cross[mask].mean(0).tolist(),
                    expanded=dict(names=dictionary_names,calibration_gram=dcal.tolist(),calibration_cross=dccal.tolist(),incoming_gram=dg[mask].mean(0).tolist(),cross_gram=dc[mask].mean(0).tolist(),
                        **dictionary_transport(dcal,dccal,dg[mask].mean(0),gt[mask].mean(0),dc[mask].mean(0))),
                    **transport(gcal,ccal,g0[mask].mean(0),gt[mask].mean(0),cross[mask].mean(0))))
        print('analyzed return',case['name'],flush=True)
    return dict(schema='pldr-source-return-analysis-v1',status='complete',protocol_sha256=sha(study/'protocol.json'),verification_sha256=sha(study/'verification.json'),analyzer_sha256=sha(__file__),
                geometry_source_sha256=sha(Path(__file__).with_name('analyze_finetuning.py')),scientific_paths=36,scientific_updates=9216,replay_updates=192,rows=rows,input_sha256=inputs,
                scope='Two-source finite transport calibrated at each recorded state on calibration documents and tested on untouched documents. This is state-conditioned observation transport, not autonomous prediction of future coefficients or an asymptotic inheritance theorem.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    v=analyze(a.study);Path(a.output).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    print('Complete source-return analysis:',a.output)
