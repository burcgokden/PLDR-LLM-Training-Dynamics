#!/usr/bin/env python
"""Analyze complete controlled runs; document and trajectory units remain separate."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def summary(x):
    x=np.asarray(x,dtype=float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),q90=float(np.quantile(x,.9)),maximum=float(x.max()),minimum=float(x.min()))


def interval(x,seed=630121,iterations=1000):
    """Equal-shard stratified document bootstrap; input row order cycles 16 strata."""
    x=np.asarray(x);rng=np.random.default_rng(seed);n=len(x)
    strata=[np.arange(k,n,16) for k in range(16)]
    means=[x[np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in strata])].mean(0) for _ in range(iterations)]
    return np.quantile(means,[.025,.975],axis=0).tolist()


def covariance(y):
    y=y-y.mean(0,keepdims=True)
    return np.einsum('nti,ntj->ij',y,y)/(len(y)*y.shape[1])


def lag_covariances(y,extent):
    y=y[:,:extent];y=y-y.mean(0,keepdims=True)
    n,t,q=y.shape
    return np.stack([np.einsum('nti,ntj->ij',y[:,:t-r],y[:,r:])/(n*(t-r)) for r in range(t)])


def fit_covariance_models(cov):
    q=cov.shape[-1];I=np.eye(q);r=np.arange(1,len(cov));target=(cov[1:]+cov[1:].transpose(0,2,1))/2
    best={}
    for name in ['shared_ar','ar','power']:
        candidates=[]
        for value in (np.linspace(0,.995,200) if name!='power' else np.linspace(.025,2,160)):
            if name=='shared_ar':
                coeff=value**r;u=1-coeff
                b=np.einsum('r,rij->ij',u,target-coeff[:,None,None]*I)/(u@u)
                eig,vec=la.eigh(b);b=(vec*np.clip(eig,0,1))@vec.T
                pred=b[None]+coeff[:,None,None]*(I-b)[None]
                params=dict(rho=float(value),B=b.tolist())
            elif name=='ar':
                pred=(value**r)[:,None,None]*I;params=dict(rho=float(value))
            else:
                # Completely monotone (1+r)^(-gamma), plus white variance: PSD at all extents.
                shape=(1+r)**(-value)
                amp=float(np.clip(np.einsum('r,rii->',shape,target)/(q*(shape@shape)),0,1))
                pred=amp*shape[:,None,None]*I;params=dict(gamma=float(value),amplitude=amp)
            loss=float(np.mean((pred-target)**2));candidates.append((loss,params))
        loss,params=min(candidates,key=lambda x:x[0]);best[name]=dict(parameters=params,fit_mse=loss)
    return best


def model_covariance(name,params,extent,q):
    I=np.eye(q);r=np.arange(extent)
    if name=='shared_ar':
        B=np.array(params['B']);return B[None]+(params['rho']**r)[:,None,None]*(I-B)[None]
    if name=='ar':return (params['rho']**r)[:,None,None]*I
    c=params['amplitude']*(1+r)**(-params['gamma']);c[0]=1
    return c[:,None,None]*I


def weighted_covariance(cov,weights):
    t=len(weights);out=(weights@weights)*cov[0]
    for r in range(1,t):out=out+(weights[:-r]@weights[r:])*(cov[r]+cov[r].T)
    return out


def long_analysis(study,out):
    results={};fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    for cp in [1,2]:
        run=study/'runs'/f'long-soc{cp}';x=np.load(run/'segments.npz')['features'].astype(float)
        cal=x[:256].reshape(-1,x.shape[-1]);scale=cal.std(0);scale=np.maximum(scale,1e-10)
        rng=np.random.default_rng(630081);projection=rng.normal(size=(x.shape[-1],16))/np.sqrt(x.shape[-1])
        y=(x-cal.mean(0))/scale@projection
        fit=y[256:512,:16];c0=covariance(fit);e,v=la.eigh(c0)
        if e.min()<=1e-10*e.max():raise RuntimeError('Unresolved projection covariance')
        whitening=(v/np.sqrt(e))@v.T;y=y@whitening
        fitted=fit_covariance_models(lag_covariances(y[256:512],16))
        test=y[512:];test-=test.mean(0,keepdims=True);c0test=covariance(test)
        er,vr=la.eigh(c0test);rows=[]
        # Choose direction on independent fit documents, evaluated at every test extent.
        f64=y[256:512];f64-=f64.mean(0,keepdims=True)
        cf=covariance(f64.sum(1)[:,None,:]/8)
        ef,vf=la.eigh(cf,covariance(f64));direction=vf[:,-1]
        all_lags=lag_covariances(test,64)
        for b in [1,2,4,8,16,32,64]:
            sums=test.reshape(len(test),64//b,b,16).sum(2)/np.sqrt(b)
            chi=covariance(sums);ev=la.eigvalsh(chi,c0test)
            ratio=np.diag(chi)/np.diag(c0test)
            permutation=np.stack([test[rng.permutation(len(test)),t] for t in range(64)],1)
            null=covariance(permutation.reshape(len(test),64//b,b,16).sum(2)/np.sqrt(b))
            row=dict(b=b,median_ratio=float(np.median(ratio)),trace_ratio=float(np.trace(chi)/np.trace(c0test)),
                Lambda=float(ev[-1]),selected_direction=float(direction@chi@direction/(direction@c0test@direction)),
                permutation_trace_ratio=float(np.trace(null)/np.trace(c0test)),predictions={})
            if b>=2:
                w=np.r_[np.ones(b//2),-np.ones(b//2)]/np.sqrt(b)
                contrast=np.einsum('ntbi,b->nti',test.reshape(len(test),64//b,b,16),w)
                cc=covariance(contrast);row['contrast_trace_ratio']=float(np.trace(cc)/np.trace(c0test))
            for name,model in fitted.items():
                kernel=model_covariance(name,model['parameters'],b,16)
                pred=weighted_covariance(kernel,np.ones(b)/np.sqrt(b))
                pp=dict(trace_ratio=float(np.trace(pred)/16),absolute_covariance_relative_error=float(la.norm(pred-chi)/la.norm(chi)))
                if b>=2:pp['contrast_trace_ratio']=float(np.trace(weighted_covariance(kernel,w))/16)
                row['predictions'][name]=pp
            rows.append(row)
        # Document-bootstrap uncertainty of directional trace ratio and contrast at b=64.
        def boot_stat(ids):
            sample=test[ids];c=covariance(sample)
            plus=covariance(sample.sum(1)[:,None,:]/8)
            minus=covariance((sample[:,:32].sum(1)-sample[:,32:].sum(1))[:,None,:]/8)
            return [np.trace(plus)/np.trace(c),np.trace(minus)/np.trace(c)]
        boots=[];strata=[np.arange(k,512,16) for k in range(16)]
        for _ in range(500):boots.append(boot_stat(np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])))
        ci=np.quantile(boots,[.025,.975],axis=0)
        results[str(cp)]=dict(rows=rows,models=fitted,C0_eigen_range=[float(er.min()),float(er.max())],
            fit_C0_eigen_range=[float(e.min()),float(e.max())],ridge=0,rank=16,
            b64_trace_interval=ci[:,0].tolist(),b64_contrast_interval=ci[:,1].tolist(),
            normalized_lag_trace=[float(np.trace(c)/np.trace(c0test)) for c in all_lags],
            role='Models fitted to 256 documents and first 16 positions; evaluated on 512 other documents up to 64 positions. Selected direction uses all 64 fit positions.',
            raw_sha256=sha256(run/'segments.npz'))
        np.savez_compressed(out/f'long-projection-soc{cp}.npz',mean=cal.mean(0),scale=scale,projection=projection,
                            whitening=whitening,selected_direction=direction,C0=c0test,lag_covariances=all_lags)
        for ax,key in [(axes[cp-1,0],'trace_ratio'),(axes[cp-1,1],'contrast_trace_ratio')]:
            rr=[r for r in rows if key in r];bb=[r['b'] for r in rr]
            ax.plot(bb,[r[key] for r in rr],'ko-',label='fresh documents')
            for name in fitted:ax.plot(bb,[r['predictions'][name][key] for r in rr],'.--',label=name.replace('_',' + '))
            ax.set(xscale='log',xlabel='64-token segments b',ylabel=key.replace('_',' '),title=f'Checkpoint {cp}')
            ax.grid(alpha=.2)
        axes[cp-1,0].legend(fontsize=7)
    fig.savefig(out/'long-correlations.pdf');plt.close(fig)
    return results


def training_analysis(study,out,prefix="family",heads=(2,4,8,14),seeds=(630101,630102,630103)):
    results={};fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for h in heads:
        runs=[];fields=[]
        for seed in seeds:
            rid=f'{prefix if h!=2 else "family"}-h{h}-s{seed}';run=study/'runs'/rid
            meta=json.loads((run/'manifest.json').read_text());d=np.load(run/'measurements.npz')
            if sha256(run/'measurements.npz')!=meta['raw_sha256']:raise RuntimeError('Training hash mismatch')
            stages=[]
            for t in [0,128,512,1024,2048]:
                f=d[f'evaluation_{t}_features'];sf=d[f'evaluation_{t}_shuffled_features']
                delta=f[:,25]-d[f'evaluation_{t}_matched_unigram_nll'];shuffle=sf[:,25]-f[:,25]
                stages.append(dict(step=t,nll=float(f[:,25].mean()),entropy=float(f[:,24].mean()),
                    entropy_sd=float(f[:,24].std()),matched_unigram=float(d[f'evaluation_{t}_matched_unigram_nll'].mean()),
                    full_pool_unigram=float(d[f'evaluation_{t}_pool_unigram_nll'].mean()),
                    model_minus_matched=float(delta.mean()),model_minus_matched_interval=interval(delta),
                    shuffle_minus_intact=float(shuffle.mean()),shuffle_interval=interval(shuffle),
                    matrix_dispersion=summary(d[f'matrix_dispersion_{t}']),
                    row_ratio=summary(f[:,26:31]),g_rms=summary(f[:,31:36])))
            branch={};base=d['branch_continue']
            for name in ['continue','reset_moments','freeze_generator','reset_and_freeze']:
                b=d['branch_'+name];diff=b-base
                branch[name]=dict(nll_mean=b[:,:,25].mean(1).tolist(),entropy_mean=b[:,:,24].mean(1).tolist(),
                    row_ratio_mean=b[:,:,26:31].mean((1,2)).tolist(),
                    final_nll_difference=float(diff[-1,:,25].mean()),nll_difference_interval=interval(diff[-1,:,25]),
                    final_entropy_difference=float(diff[-1,:,24].mean()))
            r=dict(run_id=rid,width=64*h,seed=seed,parameters=meta['parameters'],stages=stages,branches=branch,
                steps=d['steps'].tolist(),path=d['mean_fields'].tolist(),optimizer_statistics=d['optimizer_statistics'].tolist(),
                raw_sha256=meta['raw_sha256'],seconds=meta['seconds'],peak_cuda_gb=meta['peak_cuda_gb'])
            runs.append(r);fields.append(d['evaluation_2048_features'])
        x=np.array(fields,dtype=float);means=x.mean(1);res=x-means[:,None];grand=means.mean(0)
        within=np.einsum('sni,snj->ij',res,res)/(3*x.shape[1]);between=(means-grand).T@(means-grand)/3
        denom=np.diag(within+between);fraction=np.divide(np.diag(between),denom,out=np.zeros_like(denom),where=denom>0)
        results[str(64*h)]=dict(runs=runs,between_fractions=fraction.tolist(),fraction_summary=summary(fraction),
                              entropy_between_fraction=float(fraction[24]),nll_between_fraction=float(fraction[25]))
        tt=[x['step'] for x in runs[0]['stages']]
        nll=np.array([[x['nll'] for x in r['stages']] for r in runs]);g=np.array([[x['matrix_dispersion']['median'] for x in r['stages']] for r in runs])
        axes[0].plot(tt,nll.mean(0),'o-',label=f'{64*h}');axes[0].fill_between(tt,nll.min(0),nll.max(0),alpha=.12)
        axes[1].semilogy(tt,g.mean(0),'o-',label=f'{64*h}')
        for r in runs:axes[2].scatter(64*h,r['branches']['reset_moments']['final_nll_difference'],s=28)
    axes[0].set(xlabel='Optimizer updates',ylabel='Fresh-document NLL');axes[0].legend(title='Width')
    axes[1].set(xlabel='Optimizer updates',ylabel='Median relative operator dispersion')
    axes[2].axhline(0,color='gray',lw=1);axes[2].set(xlabel='Width',ylabel='Moment reset: paired NLL difference')
    fig.savefig(out/'controlled-training.pdf');plt.close(fig)
    return results


def drift_analysis(study,out):
    results={}
    for run in sorted((study/'runs').glob('drift-smooth-family-*/manifest.json')):
        d=np.load(run.parent/'drift.npz');e=d['epsilon'];q=d['q'];c=d['cubic']
        quadratic=.5*e[None,:,None]**2*q[:,None,:]
        cubic=quadratic+e[None,:,None]**3*c[:,None,:]/6
        a1=e[None,:,None]*d['first'][:,None,:]
        a2=a1+.5*e[None,:,None]**2*d['second'][:,None,:]
        rows=[]
        for k,epsilon in enumerate(e):
            actual=d['dnll'][:,k].mean(1);pred=a2[:,k].mean(1)
            se=np.sqrt(actual[8:].var(ddof=1)/len(actual[8:])+pred[:8].var(ddof=1)/8)
            diff=actual[8:].mean()-pred[:8].mean()
            rows.append(dict(epsilon=float(epsilon),quadratic_relative_error=summary(abs(d['kl'][:,k]/quadratic[:,k]-1)),
                cubic_relative_error=summary(abs(d['kl'][:,k]/quadratic[:,k]-cubic[:,k]/quadratic[:,k])),
                nll_first_abs_error=summary(abs(d['dnll'][:,k]-a1[:,k])),
                nll_second_abs_error=summary(abs(d['dnll'][:,k]-a2[:,k])),
                calibration_prediction=float(pred[:8].mean()),test_mean_increment=float(actual[8:].mean()),
                batch_standard_error=float(se),calibration_test_difference=float(diff),
                discrepancy_in_standard_errors=float(abs(diff)/se) if se>0 else 0.,
                test_prediction_mean=float(pred[8:].mean()),test_taylor_bias=float((actual[8:]-pred[8:]).mean())))
        results[run.parent.name]=dict(rows=rows,raw_sha256=sha256(run.parent/'drift.npz'),
            numerical_checks=json.loads(run.read_text())['numerical_checks'])
    return results


def quality_analysis(study,out):
    results={}
    for path in sorted((study/'runs').glob('quality-*/manifest.json')):
        d=np.load(path.parent/'quality.npz');m=json.loads(path.read_text())
        rows={}
        for mode in m['modes']:
            delta=d[mode+'_nll']-d['intact_nll']
            rows[mode]=dict(nll_mean=float(d[mode+'_nll'].mean()),nll_difference_mean=float(delta.mean()),
                entropy_mean=float(d[mode+'_entropy'].mean()),kl=summary(d[mode+'_kl']),
                max_logit_difference=float(d[mode+'_max_logit_difference'].max()),argmax_changes=int(d[mode+'_argmax_changed'].sum()))
        results[path.parent.name]=dict(modes=rows,raw_sha256=m['raw_sha256'],
            uncertainty='Finite-cohort paired point estimates; the cyclic context derangement is not treated as independent document replication.')
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--parts',default='training,long,drift');a=ap.parse_args()
    study=Path(a.root)/'controlled-study-20260905';out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    inputs=[study/'protocol.json']
    for run in (study/'runs').iterdir():
        if not (run/'manifest.json').exists():continue
        for p in run.iterdir():
            if p.suffix in ['.json','.npz']:inputs.append(p)
    bind_run(out,inputs,vars(a));result=dict(schema='controlled-analysis-v1')
    for part in a.parts.split(','):
        result[part]={'training':training_analysis,'long':long_analysis,'drift':drift_analysis,'normalization':lambda s,o:training_analysis(s,o,'normalized'),'quality':quality_analysis,'seed_confirmation':lambda s,o:training_analysis(s,o,'normalized-confirmation',(14,),(630111,630112,630113))}[part](study,out)
    if 'drift' in result:
        runs=list(result['drift'].values())
        local=[next(row for row in run['rows'] if row['epsilon']==1e-5) for run in runs]
        result['drift_validation']=dict(
            primal_cast_difference=summary([r['numerical_checks']['primal_max_abs_error'] for r in runs]),
            second_derivative_relative_error=summary([x for r in runs for x in r['numerical_checks']['second_derivative_relative_error']]),
            epsilon=1e-5,
            maximum_quadratic_relative_error=max(r['quadratic_relative_error']['maximum'] for r in local),
            maximum_cubic_relative_error=max(r['cubic_relative_error']['maximum'] for r in local),
            maximum_standardized_ensemble_difference=max(r['discrepancy_in_standard_errors'] for r in local))
    write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json'),
               figures={p.name:sha256(p) for p in out.glob('*.pdf')}))
    print(out/'results.json')


if __name__=='__main__':main()
