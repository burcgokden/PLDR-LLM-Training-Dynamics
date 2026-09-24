#!/usr/bin/env python
"""Summarize executed component interventions and augmented-response controls."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model_rg.criticality import generator_parameter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def log_probabilities(logits):
    values=logits.astype(float);values-=values.max(-1,keepdims=True)
    return values-np.log(np.exp(values).sum(-1,keepdims=True))


def augmented_distances(reference,other):
    names=list(reference['model'])
    if names!=list(other['model']):raise AssertionError('Paired state parameter identities changed')
    ordered=[n for n in names if generator_parameter(n)]+[n for n in names if not generator_parameter(n)]
    identities=[]
    for a,b in zip(reference['optimizer']['param_groups'],other['optimizer']['param_groups'],strict=True):
        if a['params']!=b['params']:raise AssertionError('Paired optimizer order changed')
        identities.extend(a['params'])
    if len(ordered)!=len(identities):raise AssertionError('Unmapped paired optimizer coordinates')
    sums={label:{state:dict(coordinates=0,reference_squared_norm=0.,squared_distance=0.) for state in ['weights','first_moment','second_moment']} for label in ['shared_metric','head_power','remaining']}
    for name,index in zip(ordered,identities,strict=True):
        label='shared_metric' if 'reslayerAs' in name else ('head_power' if generator_parameter(name) else 'remaining')
        for field,key in [('weights',None),('first_moment','exp_avg'),('second_moment','exp_avg_sq')]:
            a=(reference['model'][name] if key is None else reference['optimizer']['state'][index][key]).numpy().astype(float)
            b=(other['model'][name] if key is None else other['optimizer']['state'][index][key]).numpy().astype(float)
            if a.shape!=b.shape:raise AssertionError('Paired state shape changed')
            record=sums[label][field];record['coordinates']+=a.size
            record['reference_squared_norm']+=float(np.sum(a*a));record['squared_distance']+=float(np.sum((b-a)**2))
    for group in sums.values():
        for r in group.values():
            r['distance_rms']=np.sqrt(r['squared_distance']/r['coordinates'])
            r['reference_rms']=np.sqrt(r['reference_squared_norm']/r['coordinates'])
            r['relative_distance']=np.sqrt(r['squared_distance']/r['reference_squared_norm']) if r['reference_squared_norm']>0 else None
    return sums


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(4);study=Path(args.root)/'criticality-dynamics-20260906';out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    inputs=[]
    def read(folder):
        meta=json.loads((folder/'manifest.json').read_text())
        if meta['status']!='complete':raise AssertionError('A declared control has an unresolved outcome')
        inputs.append(folder/'manifest.json')
        for filename,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256')]:
            if key in meta:
                if sha256(folder/filename)!=meta[key]:raise AssertionError('Changed diagnostic evidence')
                inputs.append(folder/filename)
        return json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else meta
    protocol_path=study/'protocols/tangent-replication.json';spec=json.loads(protocol_path.read_text());inputs.append(protocol_path)
    tangent=[]
    resolution_path=study/'protocols/tangent-resolution-selected.json'
    resolution_spec=json.loads(resolution_path.read_text());inputs.append(resolution_path)
    resolution_cases={c['name']:c for c in resolution_spec['cases']}
    for case in spec['cases']:
        folder=study/'analysis'/case['name'];result=read(folder);records=[]
        measured=[dict(r,grid='primary') for r in result['records']]
        if case['name'] in resolution_cases:
            extension=read(study/'analysis'/resolution_cases[case['name']]['output_name'])
            measured += [dict(r,grid='resolution') for r in extension['records']]
        for r in measured:
            components=r['augmented_relative_errors']
            score=max([r['maximum_predictive_relative_error'],*components]) if all(x is not None for x in components) else None
            records.append(dict(**r,joint_relative_error=score))
        windows=[]
        # Repeated radii remain in records and use their worst error for window selection.
        grouped=[]
        for radius in sorted({r['amplitude'] for r in records},reverse=True):
            scores=[r['joint_relative_error'] for r in records if r['amplitude']==radius]
            grouped.append(dict(amplitude=radius,joint_relative_error=max(scores) if all(v is not None for v in scores) else None))
        for first,second in zip(grouped,grouped[1:]):
            if first['joint_relative_error'] is not None and second['joint_relative_error'] is not None and max(first['joint_relative_error'],second['joint_relative_error'])<=.05:
                windows.append([second['amplitude'],first['amplitude']])
        tangent.append(dict(name=case['name'],heads=result['condition']['heads'],seed=result['condition']['seed'],step=result['step'],direction=case['direction'],
            records=records,adjacent_5pct_windows=windows,initial_source_curvature=result['initial_source_curvature'],updated_source_curvature=result['updated_source_curvature'],
            source_curvature_gain=float(np.sum(result['updated_source_curvature'])/np.sum(result['initial_source_curvature'])),
            native_optimizer_max_parameter_error=result['native_optimizer_max_parameter_error'],native_optimizer_max_moment_error=result['native_optimizer_max_moment_error'],
            clip_factor=result['update_statistics']['clip_factor']))
    arithmetic=[]
    path=study/'protocols/transport-arithmetic.json';inputs.append(path)
    for case in json.loads(path.read_text())['cases']:
        r=read(study/'analysis'/case['name']);arithmetic.append(dict(name=case['name'],heads=r['condition']['heads'],seed=r['condition']['seed'],step=r['step'],
            initial_max_kl=max(r['initial_arithmetic_kl']),updated_max_kl=max(r['updated_arithmetic_kl']),
            initial_kl=r['initial_arithmetic_kl'],updated_kl=r['updated_arithmetic_kl']))
    crossings=[]
    for name in ['h4-g2-t2048','h4-g2-t8192']:
        r=read(study/'analysis'/('shared-crossing-'+name));kl=np.array(r['mean_predictive_kl']);off=kl[~np.eye(4,dtype=bool)]
        crossings.append(dict(name=name,step=r['case']['step'],summaries=r['summaries'],off_diagonal_mean_kl_range=[float(off.min()),float(off.max())]))
    returns=[]
    path=study/'protocols/collective-return.json';inputs.append(path);spec=json.loads(path.read_text())
    for case in spec['cases']:
        folder=study/'runs'/case['name'];meta=read(folder);raw=np.load(folder/'measurements.npz')
        item=dict(name=case['name'],seed=case['seed'],steps=raw['probe_steps'].tolist(),branches={})
        baseline=raw['base_heads'];base_fields=raw['base_fields']
        saved={r['branch']:folder/r['checkpoint'] for r in meta['branches']};inputs.extend(saved.values())
        base_state=torch.load(saved['base'],map_location='cpu',mmap=True,weights_only=True)
        for branch in spec['branches']:
            h=raw[branch+'_heads'];f=raw[branch+'_fields'];record={}
            for name,index in [('entropy',0),('row_energy',2)]:
                q=h[...,index].mean(-1);delta=q-baseline[...,index].mean(-1);rms=np.sqrt(np.mean(delta.astype(float)**2,axis=(1,2)))
                record[name]=dict(mean=q.mean((1,2)).tolist(),paired_rms=rms.tolist(),
                    relative_paired_rms=(rms/rms[0]).tolist() if rms[0]>0 else None)
            record['nll']=f[:,:,25].mean(1).tolist();record['paired_nll']=(f[:,:,25]-base_fields[:,:,25]).mean(1).tolist()
            emissions={}
            for stage in ['initial','final']:
                base_lp=log_probabilities(raw['base_'+stage+'_logits']);branch_lp=log_probabilities(raw[branch+'_'+stage+'_logits'])
                kl=np.sum(np.exp(base_lp)*(base_lp-branch_lp),axis=-1)
                if kl.min() < -1e-12:raise AssertionError('Negative paired predictive KL')
                emissions[stage]=dict(kl_by_context=kl.tolist(),mean_kl=float(kl.mean()),maximum_kl=float(kl.max()))
            record['predictive_emission']=emissions
            state=torch.load(saved[branch],map_location='cpu',mmap=True,weights_only=True)
            record['augmented_endpoint_distance']=augmented_distances(base_state,state);del state
            item['branches'][branch]=record
        returns.append(item);del base_state
        raw.close()
    bind_run(out,inputs,vars(args))
    result=dict(schema='dynamics-mechanism-controls-v2',tangent=tangent,arithmetic=arithmetic,crossings=crossings,collective_returns=returns,
        interpretation='Tangent windows refer to the expanded smooth64 program and require both full augmented state and predictive errors below five percent at two neighboring measured amplitudes. Native arithmetic gaps are separate controls. Crossed models form a finite product intervention law, and return branches are paired finite displacements, without an identified critical coordinate.')
    write_json(out/'results.json',result)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    for item in returns:
        for branch,style in [('early_shared_weights','-'),('early_shared_weights_moments','--'),('frozen_early_shared_weights',':')]:
            r=item['branches'][branch];label=branch.replace('_',' ') if item is returns[0] else None
            axes[0].plot(np.array(item['steps'])-8192,r['row_energy']['mean'],style,label=label,alpha=.7)
            ratio=r['row_energy']['relative_paired_rms']
            if ratio is not None:axes[1].plot(np.array(item['steps'])-8192,ratio,style,alpha=.7)
    axes[0].set_ylabel('Mean row energy');axes[1].set_ylabel('Row displacement RMS / initial RMS')
    for ax in axes:ax.set_xlabel('Updates after component intervention');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(out/'collective-return.pdf');plt.close(fig)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),results_sha256=sha256(out/'results.json'),figures={'collective-return.pdf':sha256(out/'collective-return.pdf')}))
    print('Completed controls:',len(tangent),'tangents,',len(arithmetic),'arithmetic pairs,',len(crossings),'crossings,',len(returns),'return parents')


if __name__=='__main__':main()
