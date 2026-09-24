#!/usr/bin/env python
"""Analyze completed width-time conditions without pooling seeds with contexts or time."""
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
from model_rg.provenance import sha256,write_json


def summarize(values):
    values=np.asarray(values,dtype=np.float64)
    susceptibility,microscopic,context=conditional_covariances(values)
    seeds,contexts,layers,heads=values.shape
    trace=float(np.trace(susceptibility)/layers)
    collective=values.mean(-1)
    centered=collective-collective.mean(0,keepdims=True)
    second=np.mean(centered**2,axis=0);fourth=np.mean(centered**4,axis=0)
    omission=[]
    if seeds>=3:
        for seed in range(seeds):
            selected=np.delete(values,seed,axis=0)
            omission.append(float(np.trace(conditional_covariances(selected)[0])/layers))
    se=float(np.sqrt((seeds-1)/seeds*np.sum((np.array(omission)-np.mean(omission))**2))) if omission else None
    reference=float(np.trace(microscopic)/layers)
    return dict(trace=trace,mean=float(values.mean()),mean_by_seed=values.mean((1,2,3)).tolist(),
                susceptibility=susceptibility.tolist(),microscopic=microscopic.tolist(),
                enhancement=trace/reference if reference>0 else None,
                eigenvalues=np.linalg.eigvalsh(susceptibility).tolist(),
                seed_jackknife=dict(standard_error=se,leave_one_seed=omission),
                centered_second_moment=second.tolist(),centered_fourth_moment=fourth.tolist(),
                moment_role='Raw central moments over the measured seeds at each fixed context and decoder; not unbiased population cumulants or a critical Binder crossing.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--study',default='criticality-dynamics-20260906');parser.add_argument('--pattern',default='pilot-*')
    parser.add_argument('--output',required=True);parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args();study=Path(args.root)/args.study;out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list);inputs=[];inventory=[]
    for folder in sorted((study/'runs').glob(args.pattern)):
        path=folder/'manifest.json'
        if not path.exists():
            inventory.append(dict(run_id=folder.name,status='running_or_incomplete'))
            if not args.allow_partial:raise AssertionError('An included run is incomplete')
            continue
        meta=json.loads(path.read_text());inventory.append(dict(run_id=folder.name,status=meta['status']))
        inputs.extend([path,folder/'measurements.npz',folder/'sampling.npz'])
        if meta['status']!='complete':continue
        for filename,key in [('measurements.npz','raw_sha256'),('sampling.npz','sampling_sha256')]:
            if sha256(folder/filename)!=meta[key]:raise AssertionError('A measurement changed')
        a=meta['arguments'];key=(a['heads'],a['multiplier'],a['shared_seed'],a['stream_seed'])
        groups[key].append((folder,meta))
    bind_run(out,inputs,vars(args));rows=[];paths=[];condition_models=[]
    for (heads,g,shared,stream),members in sorted(groups.items()):
        if len(members)<4 and args.allow_partial:continue
        if len(members)<4:raise AssertionError('Incomplete four-seed condition')
        members.sort(key=lambda x:x[1]['arguments']['seed'])
        seeds=[meta['arguments']['seed'] for _,meta in members]
        if len(seeds)!=len(set(seeds)):raise AssertionError('Repeated seed at one condition')
        arrays=[np.load(folder/'measurements.npz') for folder,_ in members]
        milestones=sorted(set.intersection(*[set(meta['milestones']) for _,meta in members]))
        common_times=sorted(set.intersection(*[set(raw['steps'].tolist()) for raw in arrays]))
        dense=np.stack([raw['dense_heads'][np.searchsorted(raw['steps'],common_times)] for raw in arrays]).astype(np.float64)
        trace_path={};mean_path={}
        for name,index in [('entropy',0),('row_energy',2)]:
            mean_path[name]=dense[...,index].mean((2,3,4)).tolist()
            trace_path[name]=[float(np.trace(conditional_covariances(dense[:,i,...,index])[0])/5) for i in range(len(common_times))]
        paths.append(dict(heads=heads,multiplier=g,shared_seed=shared,stream_seed=stream,seeds=seeds,
                          steps=common_times,mean_by_seed=mean_path,conditional_trace=trace_path,
                          cohort=[512,528],role='Dense repeated observation on sixteen fixed contexts; no stationary-law or independent-time assertion.'))
        for step in milestones:
            fields=np.stack([raw[f'heads_{step}'] for raw in arrays])
            row=dict(heads=heads,width=64*heads,multiplier=g,shared_seed=shared,stream_seed=stream,seeds=seeds,step=step,
                entropy=summarize(fields[...,0]),row_energy=summarize(fields[...,2]),
                nll_by_seed=[float(raw[f'fields_{step}'][:,25].astype(np.float64).mean()) for raw in arrays],
                context_kl_by_seed=[float(raw[f'context_kl_{step}'].mean()) for raw in arrays])
            rows.append(row)
        condition_models.append(dict(heads=heads,multiplier=g,seeds=seeds,runs=[folder.name for folder,_ in members]))
    if not rows:raise AssertionError('No complete condition is available')
    result=dict(schema='width-time-analysis-v1',arguments=vars(args),rows=rows,dense_paths=paths,conditions=condition_models,inventory=inventory,
        interpretation='Four or more initialization seeds per condition with fixed shared initialization and batch history. Contexts and time points are paired observations. Grid peaks, central moments and temporal profiles are finite descriptive measurements, not critical exponents or stationary distributions.')
    write_json(out/'results.json',result)
    widths=sorted({row['heads'] for row in rows});fig,axes=plt.subplots(2,len(widths),figsize=(4*len(widths),6),squeeze=False,constrained_layout=True)
    for column,heads in enumerate(widths):
        for p in paths:
            if p['heads']!=heads:continue
            for row,(field,kind) in enumerate([('row_energy','mean'),('row_energy','susceptibility')]):
                values=np.mean(p['mean_by_seed'][field],axis=0) if kind=='mean' else p['conditional_trace'][field]
                axes[row,column].plot(p['steps'],values,label='g='+str(p['multiplier']))
            axes[0,column].set_title('N='+str(heads));axes[1,column].set_xlabel('Training updates')
        axes[0,column].set_ylabel('Mean row energy, fixed dense cohort')
        axes[1,column].set_ylabel('Conditional row susceptibility')
        for ax in axes[:,column]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'width-time.pdf');plt.close(fig)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),results_sha256=sha256(out/'results.json'),figures={'width-time.pdf':sha256(out/'width-time.pdf')}))
    print('Complete analyzed conditions',len(condition_models),'rows',len(rows),flush=True)


if __name__=='__main__':main()
