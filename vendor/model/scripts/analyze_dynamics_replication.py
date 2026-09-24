#!/usr/bin/env python
"""Analyze the recorded sixteen-seed family with whole-seed uncertainty."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_rg.controlled import bind_run
from model_rg.criticality import conditional_covariances
from model_rg.provenance import sha256, write_json
from analyze_dynamics import summarize


def bootstrap_trace(values, counts):
    """Evaluate complete-seed resamples using an independently derived Gram formula."""
    seeds,contexts,layers,heads=values.shape
    collective=values.astype(float).mean(-1).reshape(seeds,-1)
    # A common context-wise translation leaves every resampled covariance invariant.
    # Remove it before the Gram products to preserve tiny fluctuations near a constant.
    collective-=collective.mean(0,keepdims=True)
    gram=collective@collective.T/(contexts*layers)
    traces=heads/(seeds-1)*(counts@np.diag(gram)-np.einsum('bi,ij,bj->b',counts,gram,counts)/seeds)
    return np.maximum(traces,0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root);study=root/'criticality-dynamics-20260906'
    protocol_path=study/'protocols/replication.json';protocol=json.loads(protocol_path.read_text())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    paths=[study/'runs'/job['run_id'] for job in protocol['jobs']]+[Path(p) for p in protocol['additional_bound_runs']]
    groups=defaultdict(list);inputs=[protocol_path];metas={}
    for path in paths:
        meta=json.loads((path/'manifest.json').read_text());a=meta['arguments'];metas[path.name]=meta
        if meta['status']!='complete':raise AssertionError('An explicit treatment is required for every scientific outcome')
        if a['multiplier']!=1 or a['steps']!=8192 or a['shared_seed']!=640011 or a['stream_seed']!=640001:raise AssertionError('Family changed')
        if a['heads']!=2 and a['normalization']!='variance':raise AssertionError('Normalization changed')
        # Both conventions are exactly the identity at the N=2 reference architecture.
        for filename,key in [('measurements.npz','raw_sha256'),('sampling.npz','sampling_sha256')]:
            if sha256(path/filename)!=meta[key]:raise AssertionError('Changed replication input')
        inputs.extend([path/'manifest.json',path/'measurements.npz',path/'sampling.npz'])
        groups[a['heads']].append(path)
    if set(groups)!={2,4,8,14,24}:raise AssertionError('Width inventory changed')
    bind_run(out,inputs,vars(args));rows=[];drifts=[];raw_results={}
    for heads,paths in sorted(groups.items()):
        paths.sort(key=lambda p:metas[p.name]['arguments']['seed'])
        seeds=[metas[p.name]['arguments']['seed'] for p in paths];n=len(seeds)
        if n!=(4 if heads==24 else 16) or len(set(seeds))!=n:raise AssertionError('Independent seed inventory changed')
        if seeds!=list(range(640101,640101+n)):raise AssertionError('Prespecified seed identities changed')
        data=[np.load(p/'measurements.npz') for p in paths]
        # Common resampling identities across the matched-width sixteen-seed family.
        generator=np.random.default_rng(640811)
        counts=generator.multinomial(n,np.full(n,1/n),size=5000)
        raw_results[f'bootstrap_counts_N{heads}']=counts
        by_step={}
        for step in [2048,4096,8192]:
            values=np.stack([d[f'heads_{step}'] for d in data]).astype(np.float64);by_step[step]=values
            row=dict(heads=heads,width=64*heads,multiplier=1,step=step,seeds=seeds,runs=[p.name for p in paths],
                nll_by_seed=[float(d[f'fields_{step}'][:,25].astype(np.float64).mean()) for d in data],
                context_kl_by_seed=[float(d[f'context_kl_{step}'].mean()) for d in data])
            for name,index in [('entropy',0),('row_energy',2)]:
                field=values[...,index];summary=summarize(field)
                original=bootstrap_trace(field,np.ones((1,n)))
                np.testing.assert_allclose(original[0],summary['trace'],rtol=1e-10,atol=1e-12)
                boot=bootstrap_trace(field,counts);raw_results[f'chi_{name}_N{heads}_t{step}']=boot
                summary['whole_seed_bootstrap']=dict(replicates=5000,seed=640811,percentile_95=np.quantile(boot,[.025,.975]).tolist(),
                    standard_error=float(boot.std(ddof=1)),role='Finite-ensemble bootstrap uncertainty; fixed contexts and times are never resampled as independent training realizations. Percentile intervals do not certify a thermodynamic limit.')
                row[name]=summary
            rows.append(row)
        for start,end in [(2048,4096),(4096,8192)]:
            entry=dict(heads=heads,start=start,end=end,seeds=seeds)
            for name,index in [('entropy',0),('row_energy',2)]:
                delta=(by_step[end][...,index]-by_step[start][...,index]).mean((1,2,3)).astype(float)
                boot=counts@delta/n
                entry[name]=dict(difference_by_seed=delta.tolist(),mean_difference=float(delta.mean()),
                    paired_seed_bootstrap_percentile_95=np.quantile(boot,[.025,.975]).tolist())
            drifts.append(entry)
    np.savez_compressed(out/'bootstrap.npz',**raw_results)
    results=dict(schema='width-time-replication-analysis-v1',rows=rows,paired_drifts=drifts,
        interpretation='Sixteen independent initialization seeds at N=2,4,8,14; four at N=24. Shared initialization and the complete minibatch history are fixed. N=2 reuses the exact identity reference convention. Replicate uncertainty does not include alternative shared initializations or training streams. These finite times do not establish stationarity, a critical parameter, or critical exponents.')
    write_json(out/'results.json',results)
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),constrained_layout=True)
    for step,color in [(2048,'C0'),(4096,'C1'),(8192,'C2')]:
        selected=[r for r in rows if r['step']==step];ns=[r['heads'] for r in selected]
        axes[0].plot(ns,[r['row_energy']['mean'] for r in selected],'o-',color=color,label=str(step))
        for ax,name in zip(axes[1:],['row_energy','entropy']):
            point=np.array([r[name]['trace'] for r in selected]);interval=np.array([r[name]['whole_seed_bootstrap']['percentile_95'] for r in selected])
            ax.plot(ns,point,'o-',color=color,label=str(step))
            ax.vlines(ns,interval[:,0],interval[:,1],color=color,alpha=.6)
    for ax,label in zip(axes,['Mean row energy','Conditional row susceptibility','Conditional entropy susceptibility']):
        ax.set_xscale('log');ax.set_xticks([2,4,8,14,24],labels=['2','4','8','14','24']);ax.set_xlabel('Head count N');ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(title='Updates',fontsize=7)
    for ax in axes[1:]:ax.set_yscale('log')
    fig.savefig(out/'replicated-width-time.pdf');plt.close(fig)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),results_sha256=sha256(out/'results.json'),
        raw_sha256=sha256(out/'bootstrap.npz'),figures={'replicated-width-time.pdf':sha256(out/'replicated-width-time.pdf')}))
    print('Completed',len(rows),'width-time estimates and',len(drifts),'paired drift comparisons')


if __name__=='__main__':main()
