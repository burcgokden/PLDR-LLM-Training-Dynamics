#!/usr/bin/env python
"""Analyze completed training trajectories and exact empirical mixture laws."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_rg.training import mixture_covariance
from model_rg.provenance import sha256, source_manifest, write_json, environment


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True)
    ap.add_argument('--output',required=True);args=ap.parse_args()
    root=Path(args.data_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    heads=[2,4,8,14];seeds=[12001,12002,12003];blocks=np.array([1,2,4,8,16,32,64,128])
    rng=np.random.default_rng(76123);count=16384
    result={'schema':'model-rg-training-analysis-v1','heads':heads,'seeds':seeds,
            'empirical_law':'Uniform over the three trained seeds and 512 fixed probe documents, with replacement.',
            'monte_carlo_blocks':count,'monte_carlo_seed':76123,'widths':{}}
    bindings={};mcarrays={}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.0),constrained_layout=True)
    fig2,ax2=plt.subplots(figsize=(6.3,4.0),constrained_layout=True)
    for color,h in zip(plt.cm.viridis(np.linspace(.1,.85,4)),heads):
        metas=[];arrays=[];responses=[];response_metas=[];derivative_metas=[]
        for seed in seeds:
            run=root/'executed'/f'training-h{h}-s{seed}'
            meta=json.loads((run/'manifest.json').read_text())
            if sha256(run/'training.npz')!=meta['raw_sha256']:raise RuntimeError('Training hash mismatch')
            metas.append(meta);arrays.append(dict(np.load(run/'training.npz')))
            bindings[str(run.relative_to(root))]=sha256(run/'manifest.json')
            rr=root/'executed'/f'training-response-h{h}-s{seed}'
            rm=json.loads((rr/'manifest.json').read_text())
            if sha256(rr/'response.npz')!=rm['raw_sha256']:raise RuntimeError('Response hash mismatch')
            responses.append(dict(np.load(rr/'response.npz')));response_metas.append(rm)
            bindings[str(rr.relative_to(root))]=sha256(rr/'manifest.json')
            dr=root/'executed'/f'training-derivative-h{h}-s{seed}'
            dm=json.loads((dr/'manifest.json').read_text());derivative_metas.append(dm)
            if sha256(dr/'duality.npz')!=dm['raw_sha256']:raise RuntimeError('Derivative hash mismatch')
            bindings[str(dr.relative_to(root))]=sha256(dr/'manifest.json')
        x=np.stack([a['features'] for a in arrays]).astype(np.float64)
        steps=arrays[0]['steps'];within=[];between=[];total=[];residual=[]
        for k in range(len(steps)):
            a,b,c=mixture_covariance(x[:,k]);within.append(a);between.append(b);total.append(c)
            residual.append(float(np.linalg.norm(c-a-b)/np.linalg.norm(c)))
        np.savez_compressed(out/f'covariances-h{h}.npz',within=np.stack(within),between=np.stack(between),
                            total=np.stack(total),steps=steps)
        nll=x[:,:,:,-1].mean(-1)
        axes[0].plot(steps,nll.mean(0),marker='o',color=color,label=f'{64*h} ({metas[0]["parameters"]/1e6:.1f}M)')
        axes[0].fill_between(steps,nll.min(0),nll.max(0),color=color,alpha=.13)
        # Predictive entropy is a fixed, common observable for every width.
        y=x[:,-1,:,-2];a=float(within[-1][-2,-2]);b=float(between[-1][-2,-2]);c=a+b
        grand=y.mean();shared=[];independent=[];exact=[];standard_errors=[]
        for block in blocks:
            state=rng.integers(0,3,size=count)
            doc=rng.integers(0,y.shape[1],size=(count,int(block)))
            shared_draw=(y[state[:,None],doc]-grand).sum(1)/np.sqrt(block)
            state_independent=rng.integers(0,3,size=doc.shape)
            independent_draw=(y[state_independent,doc]-grand).sum(1)/np.sqrt(block)
            mcarrays[f'h{h}_b{block}_shared']=shared_draw
            mcarrays[f'h{h}_b{block}_independent']=independent_draw
            shared.append(float(shared_draw.var(ddof=1)));independent.append(float(independent_draw.var(ddof=1)))
            exact.append(a+int(block)*b)
            cent=shared_draw-shared_draw.mean()
            standard_errors.append(float(np.sqrt(max(0,np.mean(cent**4)-np.mean(cent**2)**2)/count)))
        shared=np.array(shared);independent=np.array(independent);exact=np.array(exact)
        axes[1].plot(blocks,exact/c,color=color,linewidth=1.4)
        axes[1].scatter(blocks,shared/c,color=color,s=20)
        axes[1].scatter(blocks,independent/c,color=color,marker='x',s=20)
        rel=[]
        for r in responses:
            pred=.5*r['epsilon'][:,None]**2*r['curvature'][None,:]
            rel.append(np.abs(r['kl']/pred-1))
        rel=np.stack(rel);median=np.median(rel,axis=(0,2));eps=responses[0]['epsilon']
        ax2.plot(eps,100*median,marker='o',color=color,label=f'width {64*h}')
        record={'width':64*h,'parameters':metas[0]['parameters'],'steps':steps.tolist(),
            'initial_nll_by_seed':nll[:,0].tolist(),'final_nll_by_seed':nll[:,-1].tolist(),
            'mean_nll_by_step':nll.mean(0).tolist(),'max_covariance_relative_residual':max(residual),
            'entropy_within_by_step':[float(z[-2,-2]) for z in within],
            'entropy_between_by_step':[float(z[-2,-2]) for z in between],
            'blocks':blocks.tolist(),'shared_variance':shared.tolist(),
            'shared_prediction':exact.tolist(),'independent_variance':independent.tolist(),
            'independent_prediction':c,'shared_mc_standard_error':standard_errors,
            'max_shared_relative_mc_error':float(np.max(abs(shared/exact-1))),
            'max_independent_relative_mc_error':float(np.max(abs(independent/c-1))),
            'shared_ratio_b128':float(exact[-1]/c),'between_fraction':b/c,
            'training_seconds':[m['seconds'] for m in metas],
            'peak_training_cuda_gb':max(m['peak_cuda_gb'] for m in metas),
            'temporal_parameter_error':max(m['temporal_blocking']['parameter_max_abs_error'] for m in metas),
            'temporal_optimizer_error':max(m['temporal_blocking']['optimizer_max_abs_error'] for m in metas),
            'response_epsilon':eps.tolist(),'response_median_relative_error':median.tolist(),
            'max_derivative_fisher_error':max(m['max_directional_derivative_fisher_error'] for m in response_metas),
            'max_normalized_duality_error':max(m['max_normalized_duality_error'] for m in derivative_metas),
            'max_ad_primal_difference':max(m['baseline_max_abs_error'] for m in derivative_metas),
            'response_seconds':[m['seconds'] for m in response_metas],
            'peak_response_cuda_gb':max(m['peak_cuda_gb'] for m in response_metas)}
        denominator=np.diag(total[-1]);fraction=np.divide(np.diag(between[-1]),denominator,out=np.zeros_like(denominator),where=denominator>0)
        record['between_coordinate_fractions']=fraction.tolist()
        record['nll_shared_ratio_b128']=float(1+127*fraction[-1])
        record['initial_nll_mean']=float(nll[:,0].mean())
        record['final_nll_mean']=float(nll[:,-1].mean())
        record['response_by_seed']=[{'seed':int(seed),'median':np.median(err,axis=1).tolist(),'q90':np.quantile(err,.9,axis=1).tolist(),'maximum':err.max(1).tolist()} for seed,err in zip(seeds,rel)]
        result['widths'][str(64*h)]=record
    axes[0].set(xlabel='Optimizer step',ylabel='Probe next-token NLL (nats)')
    axes[0].legend(title='Width (parameters)',fontsize=8,title_fontsize=8)
    axes[1].set(xscale='log',yscale='log',xlabel='Independent contexts per block',
                ylabel='Entropy susceptibility / one-context variance')
    axes[1].set_xticks([1,4,16,64,128],[1,4,16,64,128])
    axes[1].text(.04,.95,'Lines: shared-checkpoint prediction\nCircles: shared-checkpoint sampling\nCrosses: independently redrawn checkpoints',
                 transform=axes[1].transAxes,va='top',fontsize=8)
    fig.savefig(out/'training-and-mixture.pdf');fig.savefig(out/'training-and-mixture.png',dpi=160)
    ax2.set(xscale='log',yscale='log',xlabel='Fraction of the actual optimizer update',
            ylabel='Median relative KL prediction error (%)')
    ax2.legend(fontsize=8);fig2.savefig(out/'training-response.pdf');fig2.savefig(out/'training-response.png',dpi=160)
    np.savez_compressed(out/'mixture-sampling.npz',**mcarrays)
    write_json(out/'results.json',result)
    write_json(out/'analysis-manifest.json',{'source_files':source_manifest(),'arguments':vars(args),'environment':environment(),'input_manifests':bindings,
        'results_sha256':sha256(out/'results.json'),
        'derived_arrays':{p.name:sha256(p) for p in out.glob('*.npz')}})
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
