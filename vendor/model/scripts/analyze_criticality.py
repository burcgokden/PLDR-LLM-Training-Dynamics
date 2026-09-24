#!/usr/bin/env python
"""Analyze conditional replica susceptibility without pooling document replicates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_rg.controlled import bind_run
from model_rg.criticality import conditional_covariances
from model_rg.provenance import sha256, write_json


def summary(values):
    x = np.asarray(values, dtype=float)
    return dict(mean=float(x.mean()), median=float(np.median(x)), minimum=float(x.min()),
                maximum=float(x.max()), q90=float(np.quantile(x,.9)),
                sd=float(x.std(ddof=1)) if x.size>1 else 0.)


def susceptibility(values):
    chi, diagonal, context = conditional_covariances(values)
    eigen, vectors = la.eigh(diagonal)
    threshold = max(float(eigen[-1])*1e-10, 1e-18)
    keep = eigen > threshold
    rank = int(keep.sum())
    generalized = []
    if rank:
        whitening = vectors[:,keep] / np.sqrt(eigen[keep])
        generalized = la.eigvalsh(whitening.T@chi@whitening).tolist()
        if generalized[-1] > values.shape[-1] + 1e-5 or generalized[0] < -1e-5:
            raise AssertionError('Uniform-head covariance violates its finite-sample bounds')
    scale = values.shape[2]
    trace = float(np.trace(chi)/scale)
    diag_trace = float(np.trace(diagonal)/scale)
    return dict(trace=trace, diagonal_trace=diag_trace,
                enhancement=trace/diag_trace if diag_trace>0 else None,
                maximum_generalized=generalized[-1] if rank else None,
                generalized_eigenvalues=generalized, resolved_rank=rank,
                diagonal_eigenvalues=eigen.tolist(), numerical_threshold=threshold, ridge=0,
                off_diagonal_trace=(trace-diag_trace)/(values.shape[-1]-1) if values.shape[-1]>1 else None,
                document_sector_trace=float(np.trace(context)/scale),
                covariance=chi.tolist(), head_diagonal=diagonal.tolist(), document_covariance=context.tolist())


def jackknife_trace(values):
    if len(values)<3:
        return None
    estimates=np.array([susceptibility(np.delete(values,s,axis=0))['trace'] for s in range(len(values))])
    error=np.sqrt((len(values)-1)/len(values)*np.sum((estimates-estimates.mean())**2))
    return dict(leave_one_seed=estimates.tolist(), standard_error=float(error),
                interpretation='Descriptive leave-one-training-seed jackknife; four seeds do not provide a calibrated population interval')


def width_fits(heads, y):
    """Small-sample diagnostics, not an identification of a scaling exponent."""
    x=np.array(heads,dtype=float);y=np.array(y,dtype=float)
    scale=float(y.mean())
    if scale == 0:
        return {name:dict(parameters=[0.,0.],fitted=y.tolist(),relative_rmse=0.,
                          leave_one_width_predictions=y.tolist(),leave_one_width_relative_rmse=0.,
                          optimization_scale=0.,exponent_identified=False)
                for name in ['regular_inverse_width','power']}
    normalized=y/scale
    definitions={
        'regular_inverse_width': (lambda p,n:p[0]+p[1]/n,[max(float(normalized[-1]),1e-6),0.],([0.,-np.inf],[np.inf,np.inf])),
        'power': (lambda p,n:p[0]*n**p[1],[1.,0.],([0.,-3.],[np.inf,3.])),
    }
    results={}
    for name,(prediction,initial,bounds) in definitions.items():
        fitted=least_squares(lambda p:prediction(p,x)-normalized,initial,bounds=bounds)
        loo=[]
        for k in range(len(x)):
            mask=np.arange(len(x))!=k
            fit=least_squares(lambda p:prediction(p,x[mask])-normalized[mask],initial,bounds=bounds)
            loo.append(float(scale*prediction(fit.x,x[k])))
        parameters=fitted.x.copy()
        parameters[0]*=scale
        if name=='regular_inverse_width':parameters[1]*=scale
        predicted=scale*prediction(fitted.x,x)
        results[name]=dict(parameters=parameters.tolist(), fitted=predicted.tolist(),
                           relative_rmse=float(np.sqrt(np.mean((predicted-y)**2))/scale),
                           leave_one_width_predictions=loo,
                           leave_one_width_relative_rmse=float(np.sqrt(np.mean((np.array(loo)-y)**2))/scale),
                           optimization_scale=scale)
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--pattern',default='scan-*')
    parser.add_argument('--include-pattern',action='append',default=[])
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--reuse-reference',action='store_true',help='Reuse the exactly identical h=2 scan for the variance family')
    args=parser.parse_args()
    study=Path(args.root)/'criticality-study-20260905'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifests=sorted((study/'runs').glob(args.pattern+'/manifest.json'))
    for pattern in args.include_pattern:
        manifests += sorted((study/'runs').glob(pattern+'/manifest.json'))
    manifests = sorted(set(manifests))
    if args.reuse_reference:
        manifests += sorted((study/'runs').glob('scan-h2-*/manifest.json'))
        manifests = sorted(set(manifests))
    if not manifests:
        raise RuntimeError('No complete manifests match the declared pattern')
    inputs=[p for m in manifests for p in [m,m.parent/'measurements.npz',m.parent/'sampling.npz']]
    bind_run(out,inputs,vars(args))
    inventory=[];groups=defaultdict(list)
    for path in manifests:
        meta=json.loads(path.read_text())
        if sha256(path.parent/'measurements.npz') != meta['raw_sha256']:
            raise AssertionError('Raw measurement digest mismatch')
        inventory.append(dict(run_id=path.parent.name,status=meta['status'],manifest_sha256=sha256(path)))
        if meta['status']!='complete':
            continue
        a=meta['arguments']
        normalization=a.get('normalization','fan_in')
        if args.reuse_reference and a['heads']==2:
            normalization='variance'
        key=(normalization,a['heads'],a['multiplier'],a.get('stream_seed'),a.get('shared_seed'),meta['completed_step'])
        groups[key].append((path,meta))
    results={};condition_rows=[]
    for key,entries in sorted(groups.items()):
        normalization,h,g,stream,shared,horizon=key
        entries.sort(key=lambda e:e[1]['arguments']['seed'])
        if len({m['arguments']['seed'] for p,m in entries}) != len(entries):
            raise RuntimeError(f'Duplicate training seed in condition: {key}')
        if len(entries)<4 and not args.allow_partial:
            raise RuntimeError(f'Incomplete training-seed group: {key}')
        if len(entries)<2:
            continue
        arrays=[np.load(p.parent/'measurements.npz') for p,m in entries]
        common_steps=sorted(set.intersection(*[set(m['milestones']) for p,m in entries]))
        stages=[]
        for t in common_steps:
            if not all(f'heads_{t}' in a for a in arrays):
                continue
            heads=np.stack([a[f'heads_{t}'] for a in arrays]).astype(float)
            stage=dict(step=t,
                nll_by_seed=[float(a[f'fields_{t}'][:,25].mean()) for a in arrays],
                entropy_by_seed=[float(a[f'fields_{t}'][:,24].mean()) for a in arrays],
                context_kl_by_seed=[float(a[f'context_kl_{t}'].mean()) for a in arrays],
                context_loss_by_seed=[float(a[f'context_loss_{t}'].mean()) for a in arrays],
                matched_nll_by_seed=[float(a[f'matched_nll_{t}'].mean()) for a in arrays],
                operator_dispersion_by_seed=[float(np.median(a[f'operator_dispersion_{t}'])) for a in arrays],
                fields={})
            for label,index in [('attention_entropy',0),('row_energy',2)]:
                values=heads[:,:,:,:,index]
                stat=susceptibility(values)
                stat['jackknife']=jackknife_trace(values)
                stat['mean_by_seed']=values.mean((1,2,3)).tolist()
                stage['fields'][label]=stat
            stages.append(stage)
        name=f'{normalization}-h{h}-g{g:g}-stream{stream}-shared{shared}-t{horizon}'
        final=stages[-1]
        source=[]
        for a in arrays:
            norm=np.sqrt(a['source_half_curvature'])
            source.append(dict(curvature=summary(a['source_half_curvature']),
                               absolute_secant_difference=summary(a['source_secant_discrepancy']),
                               relative_secant_difference=summary(a['source_secant_discrepancy']/np.maximum(norm,1e-14))))
        results[name]=dict(normalization=normalization,heads=h,width=64*h,multiplier=g,stream_seed=stream,shared_seed=shared,horizon=horizon,
                           seeds=[m['arguments']['seed'] for p,m in entries],
                           run_ids=[p.parent.name for p,m in entries],stages=stages,source_controls=source,
                           runtime_seconds=[m['seconds'] for p,m in entries],
                           peak_cuda_gb=[m['peak_cuda_gb'] for p,m in entries])
        condition_rows.append(dict(condition=name,normalization=normalization,stream_seed=stream,shared_seed=shared,horizon=horizon,heads=h,width=64*h,multiplier=g,
                                    nll=summary(final['nll_by_seed']),context_kl=summary(final['context_kl_by_seed']),
                                    entropy=final['fields']['attention_entropy'],row_energy=final['fields']['row_energy']))
    fits={}
    environments={(r['normalization'],r['stream_seed'],r['shared_seed'],r['horizon']) for r in results.values()}
    for environment in sorted(environments):
        selected=[r for r in results.values() if (r['normalization'],r['stream_seed'],r['shared_seed'],r['horizon'])==environment]
        for g in sorted({r['multiplier'] for r in selected}):
            rows=sorted([r for r in selected if r['multiplier']==g],key=lambda r:r['heads'])
            if len(rows)<4:
                continue
            for field in ['attention_entropy','row_energy']:
                values=[r['stages'][-1]['fields'][field]['trace'] for r in rows]
                fits[f'{environment}-g{g:g}-{field}']=dict(heads=[r['heads'] for r in rows],values=values,
                    models=width_fits([r['heads'] for r in rows],values),
                    role='Four-size exploratory comparison; neither fit certifies a thermodynamic limit or a critical exponent')
    if len(environments)>1:
        raise RuntimeError('Use separate analyses for distinct normalization, environment, or horizon conditions')
    figure,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for h in sorted({row['heads'] for row in condition_rows}):
        rows=sorted([r for r in condition_rows if r['heads']==h],key=lambda r:r['multiplier'])
        x=[r['multiplier'] for r in rows]
        for ax,getter in [(axes[0,0],lambda r:r['nll']['mean']),
                          (axes[0,1],lambda r:r['context_kl']['mean']),
                          (axes[0,2],lambda r:r['entropy']['trace']),
                          (axes[1,0],lambda r:r['entropy']['enhancement']),
                          (axes[1,1],lambda r:r['row_energy']['trace']),
                          (axes[1,2],lambda r:r['row_energy']['enhancement'])]:
            ax.plot(x,[getter(r) for r in rows],'o-',label=str(64*h))
            if min(x)>0 and max(x)/min(x)<=4:
                ax.set_xscale('linear');ax.set_xticks(x)
            else:
                ax.set_xscale('symlog',linthresh=.25)
            ax.set_xlabel('Generator update multiplier')
            ax.grid(alpha=.2)
    for ax,title in zip(axes.ravel(),['NLL','Predictive context KL','Conditional entropy susceptibility',
                                     'Entropy head-correlation enhancement','Conditional row susceptibility','Row head-correlation enhancement']):
        ax.set_ylabel(title)
    axes[0,0].legend(title='Width',fontsize=8)
    figure.savefig(out/'criticality-scan.pdf');plt.close(figure)
    complete=dict(schema='criticality-analysis-v1',inventory=inventory,conditions=results,
                   condition_rows=condition_rows,width_fit_diagnostics=fits,
                   statistical_unit='Training initialization conditional on recorded shared generator and complete sampler history; fixed evaluation cohort',
                   uncertainty='Seed-level spread and descriptive jackknife; no independent replicas inferred from heads, contexts, or time checkpoints')
    write_json(out/'results.json',complete)
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json'),
                                        figures={p.name:sha256(p) for p in out.glob('*.pdf')}))
    print(json.dumps(dict(conditions=len(results),runs=len(inventory),output=str(out/'results.json')),indent=2))


if __name__=='__main__':
    main()
