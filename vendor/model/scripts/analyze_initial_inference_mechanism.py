#!/usr/bin/env python
"""Summarize the complete five-state inference mechanism panel."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;out=study/'analysis/initial-inference-mechanism'
    out.mkdir(parents=True,exist_ok=False)
    prefix_selection=study/'protocols/prefix-mechanism-initial-selection.json'
    cases=json.loads(prefix_selection.read_text())['cases'];inputs=[prefix_selection]
    checks=[study/'verification/prefix-mechanism-initial.json',study/'verification/row-input-transport.json']
    for case in cases:
        suffix=case['name'].removeprefix('prefix-')
        checks.append(study/'verification/reasoning'/('reasoning-'+suffix+'.json'))
        for name in [case['name'],'row-transport-'+suffix,'reasoning-'+suffix]:
            folder=study/'measurements'/name;inputs.extend([folder/'manifest.json',folder/'results.json'])
    for path in checks:
        data=json.loads(path.read_text())
        if data['status']!='passed':raise AssertionError('A required mechanism verification is incomplete')
    inputs+=checks;bind_run(out,inputs,vars(a))
    records=[]
    for case in cases:
        name=case['name'].removeprefix('prefix-')
        pr=json.loads((study/'measurements'/case['name']/'results.json').read_text())
        tr=json.loads((study/'measurements'/('row-transport-'+name)/'results.json').read_text())
        rr=json.loads((study/'measurements'/('reasoning-'+name)/'results.json').read_text())
        vr=json.loads((study/'verification/reasoning'/('reasoning-'+name+'.json')).read_text())
        raw_mode={m['mode']:m for m in rr['modes']};native=raw_mode['native'];native_items={e['id']:e for e in native['examples']}
        modes=[]
        for m in vr['summaries']:
            mode=m['mode'];changed=raw_mode[mode];eps=[];certified=0;flips=0;positive=0
            for item in changed['examples']:
                ref=native_items[item['id']];error=float(np.max(np.abs(np.array(item['scores'])-ref['scores'])))
                eps.append(error);flips+=item['prediction']!=ref['prediction'];positive+=ref['margin']>0
                if ref['margin']>2*error:
                    certified+=1
                    if item['prediction']!=item['correct']:raise AssertionError('An observed score bound did not preserve a certified margin')
                if abs(item['margin']-ref['margin'])>2*error+1e-12:raise AssertionError('The exact candidate-score margin inequality failed')
            modes.append(dict(**m,paired_external_nll_change=m['external_target_nll']-native['external_target_nll'],
                answer_decision_changes=flips,strictly_correct_native=positive,certified_preserved_correct=certified,
                mean_maximum_candidate_score_error=float(np.mean(eps)),maximum_candidate_score_error=max(eps)))
        prefix=[]
        for variant in ['prefix','target_swap','suffix_swap']:
            selected=[r for r in pr['conditions'] if r['variant']==variant and r['length']<64]
            prefix.append(dict(variant=variant,mean_predictive_kl=float(np.mean([r['mean_predictive_kl'] for r in selected])),
                mean_relative_G_difference=float(np.mean([r['mean_relative_G_difference'] for r in selected])),
                mean_nll_change=float(np.mean([r['nll_change'] for r in selected]))))
        records.append(dict(name=name,prefix=prefix,modes=modes,
            row_transport_factors=tr['overall_row_factor'],suffix_transport_factors=tr['overall_suffix_factor']))
    labels=['Released 1','Released 2','Constant g=1\n32,768','Constant g=0\n32,768','Constant g=1\n131,072']
    colors=['#1f77b4','#2ca02c','#ff7f0e','#555555','#9467bd']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained')
    for row,label,color in zip(records,labels,colors,strict=True):
        axes[0].scatter(row['row_transport_factors'],row['suffix_transport_factors'],label=label.replace('\n',' '),color=color,s=34)
    axes[0].plot([0,1.1],[0,1.1],color='#bbbbbb',linewidth=1,zorder=0)
    axes[0].set_xscale('symlog',linthresh=1e-7);axes[0].set_yscale('symlog',linthresh=1e-7)
    axes[0].set_xlabel('Row RMS transport factor');axes[0].set_ylabel('Suffix RMS transport factor')
    axes[0].set_title('Shared row maps: five decoder paths per state')
    axes[0].legend(fontsize=7,loc='upper left',frameon=False)
    styles={'native':('o','#222222','Native'),'calibration_G':('s','#1f77b4','Calibrated G'),
            'permuted_G':('^','#d62728','Head-permuted G'),'row_projection':('x','#2ca02c','Row projection')}
    for j,(mode,(marker,color,label)) in enumerate(styles.items()):
        values=[next(m for m in r['modes'] if m['mode']==mode)['external_target_nll'] for r in records]
        axes[1].scatter(np.arange(5)+(j-1.5)*.12,values,marker=marker,color=color,label=label,s=35)
    axes[1].set_xticks(range(5),labels,fontsize=8);axes[1].set_ylabel('Proper external-target NLL (nats)')
    axes[1].set_title('Fixed learned content and operator interventions')
    axes[1].legend(fontsize=8,frameon=False)
    fig.savefig(out/'operator-mechanism.pdf');fig.savefig(out/'operator-mechanism.png',dpi=180);plt.close(fig)
    write_json(out/'results.json',dict(status='complete',states=records,
        scope='Five explicitly selected descriptive states. Each panel uses its declared heldout finite context/task law. The same-state intervention preserves weights; different checkpoints do not constitute a randomized comparison of pretraining-data exposure. Zero transport factors are exact zeros in this CPU float32 observation, not analytical global contraction certificates. Accuracy and every task mode are retained in full. This summary does not establish critical exponents or broad multi-fact reasoning.'))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),figures={n:sha256(out/n) for n in ['operator-mechanism.pdf','operator-mechanism.png']}))
    print('Complete five-state inference mechanism summary',str(out),flush=True)


if __name__=='__main__':main()
