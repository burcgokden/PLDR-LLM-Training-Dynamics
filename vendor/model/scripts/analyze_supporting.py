#!/usr/bin/env python
"""Complete numerical evidence from retained finite-call and initialization runs."""
from companion_paths import required_input, acquisition_identity
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.linalg as la
from analyze import response_analysis, projection
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from model_rg.training import mixture_covariance


def stats(x):
    return dict(median=float(np.median(x)),q90=float(np.quantile(x,.9)),maximum=float(np.max(x)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    executed=root/'executed';extra=root/required_input('analyze-supporting-input-1')
    inputs=[root/'analysis/main/results.json',root/'analysis/main/analysis-manifest.json',root/'data/refinedweb-4608/tokens.npy']
    for run in executed.iterdir():
        inputs += [p for p in run.iterdir() if p.suffix in ['.npz','.json']]
    inputs+=list(extra.glob('*.json'))+list(extra.glob('scan64*.npz'))
    bind_run(out,inputs,vars(a))
    result=dict(schema='complete-supporting-evidence-v1',inference={},initialization={},inventory=[])
    old=json.loads((root/'analysis/main/results.json').read_text())
    for cp in [1,2]:
        record=old[f'soc{cp}']
        for name,dirname in [('response',f'response-replication-soc{cp}-s128'),('pilot_response',f'response-soc{cp}-s128')]:
            r=response_analysis(executed/dirname);d=np.load(executed/dirname/'response.npz')
            q=np.einsum('di,nij,dj->nd',d['directions'],d['fisher'],d['directions']);split=len(q)//2
            for k,row in enumerate(r['local_prediction']):
                err=abs(d['kl'][split:,k]/(.5*d['amplitudes'][k]**2*q[split:])-1)
                row['q90_relative_error']=float(np.quantile(err,.9));row['maximum_relative_error']=float(err.max())
            r['finite_difference_check_q90']=float(np.quantile(d['half_step_direction_error'],.9))
            r['curvature_cv_by_direction']=(q.std(0)/q.mean(0)).tolist()
            H=d['fisher'];tr=np.trace(H,axis1=1,axis2=2)
            r['median_context_effective_rank']=float(np.median(tr**2/np.sum(H*H,axis=(1,2))))
            mean=H.mean(0);r['mean_matrix_effective_rank']=float(np.trace(mean)**2/np.sum(mean*mean))
            record[name]=r
        record['matrix_variation']=json.loads((extra/f'matrix-variation-soc{cp}.json').read_text())
        record['native_validation']=json.loads((executed/f'native-validation-soc{cp}/result.json').read_text())
        x=np.load(executed/f'segments-soc{cp}-s64/segments.npz')['features'].astype(float)
        y,_=projection(x[:512].reshape(-1,x.shape[-1]),x[512:]);y-=y.mean(0,keepdims=True)
        def cv(y):return np.einsum('nti,ntj->ij',y-y.mean(0),y-y.mean(0))/(len(y)*y.shape[1])
        c0=cv(y);e=la.eigvalsh(c0);rows=[]
        fit=y[:2048];cf=cv(fit);test=y[2048:]
        for b in [1,2,4,8]:
            chi=cv(y.reshape(len(y),8//b,b,32).sum(2)/np.sqrt(b))
            vals=la.eigvalsh(chi,c0)
            chif=cv(fit.reshape(len(fit),8//b,b,32).sum(2)/np.sqrt(b))
            vv=la.eigh(chif,cf)[1][:,-1]
            chit=cv(test.reshape(len(test),8//b,b,32).sum(2)/np.sqrt(b))
            rows.append(dict(b=b,Lambda=float(vals[-1]),selected_direction=float(vv@chit@vv/(vv@cv(test)@vv))))
        record['generalized_susceptibility']=dict(rows=rows,rank=32,ridge=0,eigen_min=float(e.min()),eigen_max=float(e.max()))
        result['inference'][str(cp)]=record
    tokens=np.load(root/'data/refinedweb-4608/tokens.npy');targets=tokens[3584:4096,64]
    pool=np.bincount(tokens[:3072].ravel(),minlength=32000);pool_nll=-np.log((pool+.5)/(pool.sum()+16000))[targets]
    for h in [2,4,8,14]:
        fields=[];runs=[]
        for seed in [12001,12002,12003]:
            run=executed/f'training-h{h}-s{seed}';d=np.load(run/'training.npz');f=d['features'][-1].astype(float);fields.append(f)
            sampling=np.load(run/'sampling.npz');yy=tokens[sampling['rows'][:256],sampling['offsets'][:256]+64]
            counts=np.bincount(yy.ravel(),minlength=32000);matched=-np.log((counts+.5)/(counts.sum()+16000))[targets]
            response=executed/f'training-response-h{h}-s{seed}';r=np.load(response/'response.npz')
            errors=abs(r['kl']/(.5*r['epsilon'][:,None]**2*r['curvature'][None])-1)
            z=r['baseline_logits'];p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            difference=r['central_derivative']-r['logit_derivative'];difference-=(p*difference).sum(-1,keepdims=True)
            fd=np.sqrt((p*difference**2).sum(-1)/r['curvature'])
            dual=json.loads((executed/f'training-derivative-h{h}-s{seed}/manifest.json').read_text())
            scans={name:json.loads((extra/f'{name}-h{h}-s{seed}.json').read_text()) for name in ['scan','scan64']}
            runs.append(dict(width=64*h,seed=seed,nll=float(f[:,-1].mean()),entropy=float(f[:,-2].mean()),
                entropy_sd=float(f[:,-2].std()),matched_unigram=float(matched.mean()),pool_unigram=float(pool_nll.mean()),
                response=[dict(epsilon=float(e),**stats(err)) for e,err in zip(r['epsilon'],errors)],
                secant_fisher=stats(fd),duality=dual,precision_scans=scans))
        within,between,total=mixture_covariance(np.array(fields));den=np.diag(total)
        frac=np.divide(np.diag(between),den,out=np.zeros_like(den),where=den>0)
        result['initialization'][str(64*h)]=dict(runs=runs,between_fractions=frac.tolist(),
            fraction_median=float(np.median(frac)),fraction_max=float(frac.max()),fraction_over_half=int((frac>.5).sum()),
            entropy_fraction=float(frac[-2]),nll_fraction=float(frac[-1]),nll_shared_ratio_b128=float(1+127*frac[-1]))
    for run in sorted(executed.iterdir()):
        if not run.is_dir():continue
        files={p.name:sha256(p) for p in run.iterdir() if p.suffix in ['.json','.npz']}
        role='implementation verification' if any(k in run.name for k in ['validation','benchmark','derivative']) else 'retained measurement'
        if run.name.startswith('response-soc'):role='exploratory response sample'
        result['inventory'].append(dict(run_id=run.name,role=role,files=files,
            launcher_note='Artifact exists independently of the response launcher ledger; no missing historical log reconstructed.' if run.name=='training-response-h14-s12001' else None))
    write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json')))
    print(out/'results.json')


if __name__=='__main__':main()
