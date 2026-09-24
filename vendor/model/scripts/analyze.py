#!/usr/bin/env python
"""Recompute manuscript numbers from completed runs, with document bootstrap."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from model_rg.rg import block_sum, connected_moments
from model_rg.provenance import sha256, source_manifest, write_json, environment


def projection(cal, x):
    mean=cal.mean(0);scale=cal.std(0)
    if np.any(scale<1e-10):raise ValueError("Unresolved feature variance")
    rng=np.random.default_rng(773)
    v=rng.normal(size=(cal.shape[-1],32));v/=np.linalg.norm(v,axis=0)
    projected_cal=(cal-mean)/scale@v
    v/=projected_cal.std(0)[None,:]
    return (x-mean)/scale@v, {"mean":mean,"scale":scale,"projection":v}


def bootstrap_median(values, rng, count=1000):
    # First dimension is always a complete independent document or block.
    v=np.asarray(values)
    out=[]
    for _ in range(count):out.append(np.median(v[rng.integers(len(v),size=len(v))]))
    return np.quantile(out,[.025,.975]).tolist()


def feature_analysis(path, figure_dir):
    data=np.load(path/"features.npz")
    x=data['features'];cal=x[:512];test=x[512:]
    y,proj=projection(cal,test)
    results=[]
    rng=np.random.default_rng(884)
    for b in [1,2,4,8,16,32]:
        z=block_sum(y,b);m=connected_moments(z)
        estimates=[]
        # Resample entire blocks to keep correlations among projections.
        for _ in range(1000):
            mb=connected_moments(z[rng.integers(len(z),size=len(z))])
            estimates.append([np.median(mb['variance']),np.median(abs(mb['skewness'])),np.median(abs(mb['excess_kurtosis']))])
        ci=np.quantile(estimates,[.025,.975],axis=0)
        results.append({"b":b,"blocks":len(z),"median_variance":float(np.median(m['variance'])),
          "median_abs_skewness":float(np.median(abs(m['skewness']))),
          "median_abs_excess":float(np.median(abs(m['excess_kurtosis']))),
          "bootstrap_interval":ci.tolist(),
          "gaussian_skewness_sampling_sd":float(np.sqrt(6/len(z))),
          "gaussian_excess_sampling_sd":float(np.sqrt(24/len(z)))})
    b=np.array([v['b'] for v in results]);logb=np.log(b)
    variance=np.array([connected_moments(block_sum(y,int(i)))['variance'] for i in b])
    slopes=np.polyfit(logb,np.log(variance),1)[0]
    hs=.5+slopes/2
    hboot=[]
    # All scales in a replicate derive from the same resampled documents.
    for _ in range(1000):
        z=y[rng.integers(len(y),size=len(y))]
        v=np.array([block_sum(z,int(i)).var(0) for i in b])
        hboot.append(np.median(.5+np.polyfit(logb,np.log(v),1)[0]/2))
    cov=np.cov((test-cal.mean(0))/cal.std(0),rowvar=False)
    gcv=data['g_rms'].std(0)/data['g_rms'].mean(0)
    output={"documents":len(x),"confirmation_documents":len(test),"features":x.shape[1],
      "flow":results,"H_descriptive_median":float(np.median(hs)),
      "H_document_bootstrap_interval":np.quantile(hboot,[.025,.975]).tolist(),
      "H_projection_range":[float(hs.min()),float(hs.max())],
      "next_token_nll":float(test[:,-1].mean()),
      "feature_covariance_off_diagonal_fraction":float(np.linalg.norm(cov-np.diag(np.diag(cov)))/np.linalg.norm(cov)),
      "g_norm_coefficient_variation_median":float(np.median(gcv)),"g_norm_coefficient_variation_max":float(gcv.max()),
      "feature_artifact_sha256":sha256(path/"features.npz")}
    np.savez_compressed(figure_dir/(path.name+"-projection.npz"),**proj)
    return output


def response_analysis(path):
    d=np.load(path/"response.npz")
    h=d['fisher'];v=d['directions'];amps=d['amplitudes'];actual=d['kl']
    split=len(h)//2
    q=np.einsum('di,nij,dj->nd',v,h,v)
    pred=.5*amps[None,:,None]**2*q[:,None,:]
    diagonal_q=np.einsum('di,ni,di->nd',v,np.diagonal(h,axis1=1,axis2=2),v)
    ratio=actual/pred
    rng=np.random.default_rng(885)
    local=[]
    for i,amp in enumerate(amps):
        errs=abs(ratio[split:,i]-1)
        local.append({"amplitude":float(amp),"median_relative_error":float(np.median(errs)),
                      "relative_error_interval":bootstrap_median(errs,rng),
                      "q90_relative_error":float(np.quantile(errs,.9)),"maximum_relative_error":float(errs.max()),
                      "median_ratio":float(np.median(ratio[split:,i])),
                      "median_diagonal_relative_error":float(np.median(abs(actual[split:,i]/(.5*amp**2*diagonal_q[split:])-1)))})
    hb=h[:split].mean(0);eig=np.linalg.eigvalsh(hb)
    calc_q=q[:split].mean(0)
    transfer=actual[split:,0].mean(0)/(.5*amps[0]**2*calc_q)
    ci=[]
    for _ in range(1000):
        ic=rng.integers(split,size=split);it=rng.integers(split,len(h),size=len(h)-split)
        ci.append(actual[it,0].mean(0)/(.5*amps[0]**2*q[ic].mean(0)))
    transfer_ci=np.quantile(ci,[.025,.975],axis=0)
    meta=json.loads((path/'manifest.json').read_text())
    trace=np.trace(h,axis1=1,axis2=2)
    return {"documents":len(h),"calibration_documents":split,"confirmation_documents":len(h)-split,
        "local_prediction":local,"mean_response_eigen_min":float(eig[0]),"mean_response_eigen_max":float(eig[-1]),
        "off_diagonal_fraction":float(np.linalg.norm(hb-np.diag(np.diag(hb)))/np.linalg.norm(hb)),
        "dataset_transfer_ratios":transfer.tolist(),"dataset_transfer_intervals":transfer_ci.tolist(),
        "dataset_transfer_intervals_cover_one":int(((transfer_ci[0]<=1)&(transfer_ci[1]>=1)).sum()),
        "trace_mean":float(trace.mean()),"trace_median":float(np.median(trace)),
        "trace_quantiles":np.quantile(trace,[.05,.25,.5,.75,.95]).tolist(),
        "finite_difference_check_median":float(np.median(d['half_step_direction_error'])),
        "finite_difference_check_max":float(d['half_step_direction_error'].max()),
        "autodiff_relative_error_max":max(c['relative_fisher_error'] for c in meta['autodiff_checks']),
        "response_artifact_sha256":sha256(path/'response.npz')}


def segment_analysis(path):
    x=np.load(path/'segments.npz')['features'].astype(np.float64)
    # A shared feature map preserves position-to-position covariance.
    y,_=projection(x[:512].reshape(-1,x.shape[-1]),x[512:])
    n,t,q=y.shape
    y-=y.mean(0,keepdims=True)  # covariance at each fixed position, no position-mean drift
    c0=np.mean(y.var(0,ddof=1),0)
    rng=np.random.default_rng(886)
    independent=np.stack([y[rng.permutation(n),s] for s in range(t)],axis=1)
    rows=[]
    for b in [1,2,4,8]:
        grouped=y.reshape(n,t//b,b,q).sum(2)/np.sqrt(b)
        control=independent.reshape(n,t//b,b,q).sum(2)/np.sqrt(b)
        # Average the covariance of aligned blocks; documents are sample units.
        ratio=grouped.var(0,ddof=1).mean(0)/c0
        null=control.var(0,ddof=1).mean(0)/c0
        boots=[]
        for _ in range(1000):
            ids=rng.integers(n,size=n)
            c=y[ids].var(0,ddof=1).mean(0)
            boots.append(np.median(grouped[ids].var(0,ddof=1).mean(0)/c))
        rows.append({"b":b,"median_susceptibility_ratio":float(np.median(ratio)),
                     "document_bootstrap_interval":np.quantile(boots,[.025,.975]).tolist(),
                     "projection_range":[float(ratio.min()),float(ratio.max())],
                     "permuted_control_median":float(np.median(null))})
    # Independent matrix route to the exact cross-position covariance identity.
    big=np.cov(y.reshape(n,t*q),rowvar=False)
    tensor=big.reshape(t,q,t,q)
    predicted=sum(tensor[i,:,j,:] for i in range(t) for j in range(t))/t
    measured=np.cov(y.sum(1)/np.sqrt(t),rowvar=False)
    return {"confirmation_documents":n,"projections":q,"flow":rows,
            "covariance_identity_relative_residual":float(np.linalg.norm(predicted-measured)/np.linalg.norm(measured)),
            "segments_artifact_sha256":sha256(path/'segments.npz')}


def plots(results, out):
    colors=['#1565a7','#b34623']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.3))
    for i in [1,2]:
        r=results[f'soc{i}']['features'];f=r['flow'];b=[d['b'] for d in f]
        axes[0].plot(b,[d['median_variance'] for d in f],'o-',color=colors[i-1],label=f'Checkpoint {i}')
        axes[1].plot(b,[d['median_abs_skewness'] for d in f],'o-',color=colors[i-1])
        axes[2].plot(b,[d['median_abs_excess'] for d in f],'o-',color=colors[i-1])
    axes[0].axhline(1,color='gray',ls=':',lw=1)
    for ax,label in zip(axes,['Projected variance','Median absolute skewness','Median absolute excess kurtosis']):
        ax.set_xscale('log',base=2);ax.set_xlabel('Independent document block size b');ax.set_ylabel(label);ax.grid(alpha=.15)
    axes[0].legend(frameon=False);fig.tight_layout();fig.savefig(out/'document-flow.pdf');fig.savefig(out/'document-flow.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.3))
    for i in [1,2]:
        rr=results[f'soc{i}']['response']['local_prediction'];amp=[x['amplitude'] for x in rr]
        axes[0].loglog(amp,[x['median_relative_error'] for x in rr],'o-',color=colors[i-1],label=f'Checkpoint {i}')
        axes[0].loglog(amp,[x['median_diagonal_relative_error'] for x in rr],'s:',color=colors[i-1],alpha=.6)
        sr=results[f'soc{i}']['segments']['flow'];b=[x['b'] for x in sr]
        axes[1].plot(b,[x['median_susceptibility_ratio'] for x in sr],'o-',color=colors[i-1],label=f'Checkpoint {i}')
        axes[1].plot(b,[x['permuted_control_median'] for x in sr],':',color=colors[i-1])
    axes[0].set_xlabel('RMS fractional head perturbation');axes[0].set_ylabel('Median relative KL prediction error')
    axes[1].set_xlabel('Adjacent 64-token segments per block');axes[1].set_ylabel('Susceptibility / fine-scale susceptibility')
    axes[1].set_xscale('log',base=2)
    for ax in axes:ax.grid(alpha=.15);ax.legend(frameon=False)
    fig.tight_layout();fig.savefig(out/'response-and-segments.pdf');fig.savefig(out/'response-and-segments.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.data_root)/'executed';out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    results={}
    for i in [1,2]:
        print('Analyzing checkpoint',i,flush=True)
        results[f'soc{i}']={
            'features':feature_analysis(root/f'features-soc{i}-s128',out),
            'response':response_analysis(root/f'response-replication-soc{i}-s128'),
            'segments':segment_analysis(root/f'segments-soc{i}-s64')}
        pilot=root/f'response-soc{i}-s128'
        if pilot.exists():results[f'soc{i}']['pilot_response']=response_analysis(pilot)
    write_json(out/'results.json',results)
    plots(results,out)
    write_json(out/'analysis-manifest.json',{'schema':'model-rg-analysis-v1','source_files':source_manifest(),'arguments':vars(args),'environment':environment(),
                                          'input_files':{str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file() and p.suffix in ['.json','.npz']},
                                          'results_sha256':sha256(out/'results.json')})
    print(json.dumps({k:{'H':v['features']['H_descriptive_median'],'response_error':v['response']['local_prediction'][0]['median_relative_error'],
                         'segment_b8':v['segments']['flow'][-1]['median_susceptibility_ratio']} for k,v in results.items()},indent=2))


if __name__=='__main__':main()
