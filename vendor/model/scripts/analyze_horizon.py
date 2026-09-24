#!/usr/bin/env python
"""Compare conditional fluctuations across recorded continuations of the same seeds."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from analyze_criticality import susceptibility, jackknife_trace


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    study=Path(args.root)/'criticality-study-20260905';out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted((study/'runs').glob('horizon-*/manifest.json'))
    if len(paths)!=8:raise RuntimeError('All eight horizon continuations are required')
    bind_run(out,[p for m in paths for p in [m,m.parent/'measurements.npz',m.parent/'sampling.npz']],vars(args))
    rows=[]
    for h in [2,14]:
        group=[p for p in paths if json.loads(p.read_text())['arguments']['heads']==h]
        if len(group)!=4:raise AssertionError('Horizon seed coverage')
        arrays=[];metadata=[]
        for path in group:
            meta=json.loads(path.read_text());metadata.append(meta)
            if meta['status']!='complete':raise RuntimeError('A numerical failure must be assessed explicitly')
            if sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('Raw hash mismatch')
            arrays.append(np.load(path.parent/'measurements.npz'))
        for t in [2048,4096,8192]:
            heads=np.stack([a[f'heads_{t}'] for a in arrays]).astype(float)
            fields={}
            for label,index in [('attention_entropy',0),('row_energy',2)]:
                values=heads[...,index];stat=susceptibility(values);stat['jackknife']=jackknife_trace(values)
                stat['mean_by_seed']=values.mean((1,2,3)).tolist();fields[label]=stat
            rows.append(dict(heads=h,step=t,seeds=[m['arguments']['seed'] for m in metadata],
                             run_ids=[p.parent.name for p in group],fields=fields,
                             nll_by_seed=[float(a[f'fields_{t}'][:,25].mean()) for a in arrays],
                             context_kl_by_seed=[float(a[f'context_kl_{t}'].mean()) for a in arrays],
                             operator_rms_by_seed=heads[...,3].mean((1,2,3)).tolist()))
    figure,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    getters=[lambda r:np.mean(r['nll_by_seed']),lambda r:np.mean(r['context_kl_by_seed']),
             lambda r:r['fields']['attention_entropy']['trace'],
             lambda r:np.mean(r['fields']['row_energy']['mean_by_seed']),
             lambda r:r['fields']['row_energy']['trace'],lambda r:r['fields']['row_energy']['enhancement']]
    for h in [2,14]:
        selected=[r for r in rows if r['heads']==h]
        for ax,get in zip(axes.ravel(),getters):
            ax.plot([r['step'] for r in selected],[get(r) for r in selected],'o-',label=str(64*h))
            ax.set_xscale('log',base=2);ax.set_xlabel('Training updates');ax.grid(alpha=.2)
    for ax,title in zip(axes.ravel(),['NLL','Predictive context KL','Conditional entropy susceptibility',
                                     'Mean normalized row energy','Conditional row susceptibility','Row correlation enhancement']):ax.set_ylabel(title)
    axes[0,0].legend(title='Width')
    figure.savefig(out/'horizon-dependence.pdf');plt.close(figure)
    write_json(out/'results.json',dict(schema='horizon-analysis-v1',rows=rows,
        interpretation='Repeated measurements on eight original initialization trajectories, at fixed generator rate and continued identical sampler histories. No new independent seeds or stationary law are inferred from the time points.'))
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json'),figures={p.name:sha256(p) for p in out.glob('*.pdf')}))


if __name__=='__main__':main()
