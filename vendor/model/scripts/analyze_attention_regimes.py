#!/usr/bin/env python
"""Connect microscopic saturation diagnostics with completed predictive responses."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    study=Path(args.root)/'criticality-study-20260905';out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted((study/'attention-regimes').glob('*/manifest.json'))
    if len(paths)!=140:raise RuntimeError('All 140 unique checkpoint diagnostics are required')
    inputs=[p for m in paths for p in [m,m.parent/'measurements.npz',study/'runs'/m.parent.name/'measurements.npz']]
    bind_run(out,inputs,vars(args));rows=[];groups=defaultdict(list)
    for path in paths:
        meta=json.loads(path.read_text());a=meta['condition']
        if sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('Raw hash mismatch')
        data=np.load(study/'runs'/path.parent.name/'measurements.npz')
        row=dict(parent=path.parent.name,normalization=a.get('normalization','fan_in'),heads=a['heads'],
                 multiplier=a['multiplier'],seed=a['seed'],diagnostics=meta['summary'],
                 predictive_curvature=float(data['source_half_curvature'].mean()),
                 predictive_context_kl=float(data['context_kl_2048'].mean()),
                 normalized_entropy=float(data['heads_2048'][:,:,:,0].mean()),
                 row_energy=float(data['heads_2048'][:,:,:,2].mean()),
                 operator_rms=float(data['heads_2048'][:,:,:,3].mean()))
        rows.append(row);groups[(row['normalization'],a['heads'],a['multiplier'])].append(row)
    for row in rows:
        if row['heads']==2:groups[('variance',2,row['multiplier'])].append(row)
    conditions=[]
    for (family,h,g),members in sorted(groups.items()):
        if len(members)!=4:raise AssertionError('Missing training realizations')
        record=dict(normalization=family,heads=h,multiplier=g,parents=[r['parent'] for r in members],statistics={})
        getters={key:lambda r,k=key:r[k] for key in ['predictive_curvature','predictive_context_kl','normalized_entropy','row_energy','operator_rms']}
        getters.update({key:lambda r,k=key:r['diagnostics'][k] for key in members[0]['diagnostics']})
        for key,get in getters.items():
            values=[get(r) for r in members]
            record['statistics'][key]=dict(mean=float(np.mean(values)),minimum=float(min(values)),maximum=float(max(values)),by_seed=values)
        conditions.append(record)
    figure,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    metrics=['fraction_pmax_above_099','mean_jacobian_trace','predictive_curvature']
    for row,family in enumerate(['fan_in','variance']):
        for heads in [2,4,8,14]:
            selected=sorted([r for r in conditions if r['normalization']==family and r['heads']==heads],key=lambda r:r['multiplier'])
            for col,metric in enumerate(metrics):
                ax=axes[row,col]
                ax.plot([r['multiplier'] for r in selected],[r['statistics'][metric]['mean'] for r in selected],'o-',label=str(64*heads))
                ax.set_xscale('symlog',linthresh=.25);ax.set_xlabel('Generator update multiplier');ax.grid(alpha=.2)
                ax.set_title('Extra fan-in initialization' if family=='fan_in' else 'Shape-aware initialization')
                ax.set_ylabel({'fraction_pmax_above_099':'Fraction with max attention > 0.99','mean_jacobian_trace':'Mean attention Jacobian trace','predictive_curvature':'Fisher squared norm of gain secant'}[metric])
                if metric=='predictive_curvature':ax.set_yscale('symlog',linthresh=1e-6)
    axes[0,0].legend(title='Width',fontsize=8)
    figure.savefig(out/'attention-regimes.pdf');plt.close(figure)
    write_json(out/'results.json',dict(schema='attention-regime-analysis-v1',unique_checkpoints=len(rows),runs=rows,conditions=conditions,
        conditioning='The same four wide initializations under fixed shared generator initialization and external batch history.',
        reference_reuse='The heads=2 trajectories are exactly identical in the two initialization families and appear once in the unique checkpoint inventory.',
        interpretation='Local last-query saturation is compared with the Fisher-weighted squared norm of the recorded full predictive gain secant at amplitude 0.005. Two-amplitude discrepancies are retained in the scan analysis. Context independence is a separate measurement.'))
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json'),figures={p.name:sha256(p) for p in out.glob('*.pdf')}))


if __name__=='__main__':main()
