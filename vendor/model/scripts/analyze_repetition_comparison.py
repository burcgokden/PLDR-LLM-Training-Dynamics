#!/usr/bin/env python
"""Analyze complete matched panels under their two explicitly different data laws."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256,write_json


def coordinates(path,precision):
    with np.load(Path(path).parent/'measurements.npz') as raw:
        f,h,c,e=(raw[precision+'_'+name].astype(float) for name in ['fields','heads','centroids','energies'])
        return dict(row_ratio=h[...,2].mean(-1),row_energy=e[...,0].mean(-1),
            total_energy=e[...,1].mean(-1),common_centroid=c.mean(-2),
            nll=f[:,25],entropy=f[:,24],logit_projection=f[:,16:24],context_kl=raw[precision+'_context_kl'].astype(float))


def summary(x,y,n):
    s=len(x);xm=x.reshape(s,-1).mean(1);ym=y.reshape(s,-1).mean(1);difference=ym-xm
    result=dict(repeated_mean=float(xm.mean()),onepass_mean=float(ym.mean()),paired_mean=float(difference.mean()),
        repeated_seed_means=xm.tolist(),onepass_seed_means=ym.tolist(),paired_seed_means=difference.tolist(),
        seed_count=s,repeated_susceptibility=None,onepass_susceptibility=None,paired_error_susceptibility=None,
        paired_mean_percentiles=None,delete_one_paired_means=None,susceptibility_difference_percentiles=None)
    if s>1:
        result.update(repeated_susceptibility=float(n*np.var(x,axis=0,ddof=1).mean()),
            onepass_susceptibility=float(n*np.var(y,axis=0,ddof=1).mean()),
            paired_error_susceptibility=float(n*np.var(y-x,axis=0,ddof=1).mean()))
        indices=np.asarray(list(itertools.product(range(s),repeat=s)),dtype=int)
        means=difference[indices].mean(1)
        # Centered seed Gram matrices retain all context/coordinate covariance.
        xf=(x-x.mean(0)).reshape(s,-1);yf=(y-y.mean(0)).reshape(s,-1)
        gx=xf@xf.T/xf.shape[1];gy=yf@yf.T/yf.shape[1]
        weights=np.stack([np.bincount(row,minlength=s) for row in indices]).astype(float)
        covariance=lambda g:n*(weights@np.diag(g)-np.einsum('bi,ij,bj->b',weights,g,weights)/s)/(s-1)
        chi_difference=covariance(gy)-covariance(gx)
        result.update(paired_mean_percentiles=np.quantile(means,[.025,.975]).tolist(),
            delete_one_paired_means=[float(np.delete(difference,i).mean()) for i in range(s)],
            susceptibility_difference_percentiles=np.quantile(chi_difference,[.025,.975]).tolist())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--panel',choices=['reference_schedule','constant','complete'],required=True)
    a=p.parse_args();root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/repetition-analysis-selection.json';spec=json.loads(selection.read_text())
    pairs=[v for v in spec['pairs'] if a.panel=='complete' or v['family']==a.panel]
    for path,digest in spec['inputs_sha256'].items():
        if sha256(path)!=digest:raise AssertionError('A bound repetition input changed')
    for name,digest in spec['producer_sources'].items():
        if sha256(repo/name)!=digest:raise AssertionError('A frozen repetition analyzer changed')
    inputs=[selection];groups=defaultdict(list);states=[];cached={}
    if a.panel=='complete':
        complete_observations=study/'launcher-onepass-observations.json'
        if json.loads(complete_observations.read_text())['status']!='complete':
            raise AssertionError('Final claim decisions require the completed compact single-pass observations')
        inputs.append(complete_observations)
    def load(path):
        path=Path(path)
        if str(path) not in cached:cached[str(path)]=json.loads(path.read_text())
        return cached[str(path)]
    for pair in pairs:
        for side in ['old','new']:
            item=pair[side];path=Path(item['collective']);meta=load(path);proof=load(item['verification'])
            if meta['status']!='complete' or proof['status']!='passed':raise AssertionError('Complete verified collectives required')
            if side=='old' and pair['family']=='constant':
                records=[q for q in proof['observations'] if q['name']==path.parent.name]
                if len(records)!=1 or records[0]['manifest_sha256']!=sha256(path):
                    raise AssertionError('The complete constant reconstruction does not bind this observation')
            elif proof['checked_sha256'][str(path)]!=sha256(path):
                raise AssertionError('The independent observation check differs')
            if sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('Raw collective changed')
            inputs.extend([path,path.parent/'measurements.npz',Path(item['verification'])])
            if item.get('risk'):
                rm=load(item['risk']);rp=Path(item['risk']).parent
                if rm['status']!='complete' or sha256(rp/'results.json')!=rm['results_sha256']:
                    raise AssertionError('Incomplete frozen risk')
                inputs.extend([Path(item['risk']),rp/'results.json',rp/'measurements.npz'])
                if sha256(rp/'measurements.npz')!=rm['raw_sha256']:raise AssertionError('Frozen risk changed')
                if item.get('risk_verification'):
                    if load(item['risk_verification'])['status']!='passed':raise AssertionError('Risk verification missing')
                    inputs.append(Path(item['risk_verification']))
        oldparent=load(pair['old']['parent_manifest']);newparent=load(pair['new']['case']['parent_manifest'])
        if oldparent['initial_parameter_sha256']!=newparent['initial_parameter_sha256']:
            raise AssertionError('The data-law contrast does not share its exact initial parameters')
        if newparent['status']!='complete' or load(pair['new']['native_verification'])['status']!='complete':
            raise AssertionError('The selected single-pass native path is incomplete')
        inputs.extend([Path(pair['old']['parent_manifest']),Path(pair['new']['case']['parent_manifest']),
            Path(pair['new']['native_verification'])])
        key=pair['family'],pair['recipe'],pair['heads'],pair['step'];groups[key].append(pair)
        states.append(dict(name=pair['name'],initial_parameter_sha256=newparent['initial_parameter_sha256']))
    out=study/'analysis'/('repetition-'+a.panel);out.mkdir(parents=True,exist_ok=False);bind_run(out,inputs,vars(a))
    conditions=[];path_values=defaultdict(list)
    for key,items in sorted(groups.items()):
        family,recipe,n,t=key;items=sorted(items,key=lambda q:q['seed']);seeds=[q['seed'] for q in items]
        allowed=[pr for pr in spec['precision_preference'] if all(pr in q['precisions'] for q in items)]
        for precision in allowed:
            old=[coordinates(q['old']['collective'],precision) for q in items]
            new=[coordinates(q['new']['collective'],precision) for q in items]
            metrics={name:summary(np.stack([q[name] for q in old]),np.stack([q[name] for q in new]),n)
                     for name in spec['fields']}
            row=dict(family=family,recipe=recipe,heads=n,step=t,precision=precision,seeds=seeds,observables=metrics)
            floor=spec['row_log_floor']
            log_difference=(np.log10(np.maximum(metrics['row_energy']['onepass_seed_means'],floor))-
                np.log10(np.maximum(metrics['row_energy']['repeated_seed_means'],floor)))
            relative={name:(abs(metrics[name]['onepass_susceptibility']-metrics[name]['repeated_susceptibility'])/
                metrics[name]['repeated_susceptibility'] if metrics[name]['repeated_susceptibility'] is not None and
                metrics[name]['repeated_susceptibility']>1e-10 else None) for name in spec['susceptibility_fields']}
            row.update(paired_log10_row_energy=log_difference.tolist(),mean_absolute_log10_row_energy=float(np.abs(log_difference).mean()),
                susceptibility_relative_changes=relative,screened_fields=[])
            if abs(metrics['nll']['paired_mean'])>.1:row['screened_fields'].append('nll')
            if row['mean_absolute_log10_row_energy']>.5:row['screened_fields'].append('row_energy')
            row['screened_fields'].extend(name for name,value in relative.items() if value is not None and value>.25)
            row['screened_fields']=sorted(set(row['screened_fields']))
            conditions.append(row)
            if precision=='float32':
                for i,seed in enumerate(seeds):
                    path_values[family,recipe,n,seed].append(dict(step=t,
                        repeated_row=metrics['row_ratio']['repeated_seed_means'][i],onepass_row=metrics['row_ratio']['onepass_seed_means'][i],
                        repeated_nll=metrics['nll']['repeated_seed_means'][i],onepass_nll=metrics['nll']['onepass_seed_means'][i]))
        print(family,recipe,n,t,'paired held-out NLL',metrics['nll']['paired_mean'],flush=True)
    paths=[]
    for (family,recipe,n,seed),values in sorted(path_values.items()):
        values=sorted(values,key=lambda q:q['step']);persistence=[]
        for threshold in spec['collapse_thresholds']:
            record=dict(threshold=threshold)
            for label in ['repeated','onepass']:
                flag=np.asarray([q[label+'_row']<=threshold for q in values])
                first=int(np.argmax(flag)) if flag.any() else None
                record[label]=dict(ever_below=bool(flag.any()),final_below=bool(flag[-1]),
                    first_observed_below=values[first]['step'] if first is not None else None,
                    returned_above=bool(np.any(~flag[first:])) if first is not None else False)
            record['reversal']=any(record['repeated'][key]!=record['onepass'][key] for key in
                                  ['ever_below','final_below','returned_above'])
            persistence.append(record)
        baseline=next((q for q in values if q['step']==32768),None);late=None
        if baseline is not None and values[-1]['step']>32768:
            late={label:values[-1][label+'_nll']-baseline[label+'_nll'] for label in ['repeated','onepass']}
            late['direction_reversal']=bool(np.sign(late['repeated'])!=np.sign(late['onepass']))
        paths.append(dict(family=family,recipe=recipe,heads=n,seed=seed,values=values,
            persistence=persistence,late_loss=late))
    risks=[]
    for pair in pairs:
        new=load(Path(pair['new']['risk']).parent/'results.json');row=dict(name=pair['name'],
            new_seen_training=new['cohorts']['training'],new_heldout=new['cohorts']['heldout'],old=None)
        if pair['old'].get('risk'):
            old=load(Path(pair['old']['risk']).parent/'results.json');row['old']=old['cohorts']
        risks.append(row)
    active=sorted({field for row in conditions if row['precision']=='float32' for field in row['screened_fields']})
    if any(q['late_loss'] and q['late_loss']['direction_reversal'] for q in paths):active.append('late_loss_direction')
    if any(r['reversal'] for q in paths for r in q['persistence']):active.append('sampled_persistence')
    decisions=[]
    if a.panel=='complete':
        for claim in spec['claims']:
            hits=sorted(set(claim.get('affected_by',[]))&set(active))
            disposition=('retain_stated_scope' if claim['kind'] in ['analytical','fixed_checkpoint'] else
                'conditional_requirements_not_established_by_data_screen' if claim['kind']=='conditional_physics' else
                'use_completed_singlepass_replication' if claim['id'] in ['common_variance_scaling','late_loss_and_row_return','recipe_regime_selection'] else
                'restrict_existing_coefficients_to_repeated_data_law')
            decisions.append(dict(**claim,screened_fields=hits,disposition=disposition,
                additional_replication_required_for_transfer=disposition=='restrict_existing_coefficients_to_repeated_data_law'))
    result=dict(schema='paired-data-law-analysis-v1',status='complete',panel=a.panel,states=states,
        pairs=len(pairs),groups=len(groups),conditions=conditions,paths=paths,risks=risks,
        active_primary_screens=sorted(set(active)),claim_decisions=decisions,
        interpretation='A comparison of the declared larger single-pass data law and the retained small repeated law. '
        'Native initialization is byte-identical and held-out contexts match. Differences include corpus size/content '
        'and stream structure. Screening flags are not causal attribution to repetition frequency alone, criticality '
        'tests, equivalence tests or confidence-coverage claims.',training_risk_scope=spec['training_risk_scope'],
        uncertainty=spec['uncertainty'],threshold_scope=spec['threshold_scope'])
    write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),panel=a.panel,pairs=len(pairs),groups=len(groups)))


if __name__=='__main__':main()
