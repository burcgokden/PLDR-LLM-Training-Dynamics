#!/usr/bin/env python
"""Reconstruct paired means, covariance, resampling and data-law screens."""
import argparse
from collections import defaultdict
import itertools
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256,write_json


def fields(path,precision):
    with np.load(Path(path).parent/'measurements.npz') as raw:
        f=raw[precision+'_fields'].astype(float);h=raw[precision+'_heads'].astype(float)
        c=raw[precision+'_centroids'].astype(float);e=raw[precision+'_energies'].astype(float)
        return {'row_ratio':np.sum(h[...,2],axis=2)/h.shape[2],
            'row_energy':np.sum(e[...,0],axis=2)/e.shape[2],
            'total_energy':np.sum(e[...,1],axis=2)/e.shape[2],
            'common_centroid':np.sum(c,axis=2)/c.shape[2],
            'nll':f[:,25], 'entropy':f[:,24], 'logit_projection':f[:,16:24],
            'context_kl':raw[precision+'_context_kl'].astype(float)}


def independent_summary(x,y,n):
    s=len(x);x=x.reshape(s,-1);y=y.reshape(s,-1)
    xm=np.sum(x,axis=1)/x.shape[1];ym=np.sum(y,axis=1)/y.shape[1];d=ym-xm
    row=dict(repeated_mean=float(np.sum(xm)/s),onepass_mean=float(np.sum(ym)/s),paired_mean=float(np.sum(d)/s),
        repeated_seed_means=xm.tolist(),onepass_seed_means=ym.tolist(),paired_seed_means=d.tolist(),seed_count=s,
        repeated_susceptibility=None,onepass_susceptibility=None,paired_error_susceptibility=None,
        paired_mean_percentiles=None,delete_one_paired_means=None,susceptibility_difference_percentiles=None)
    if s>1:
        dx=np.zeros((s,s));dy=np.zeros((s,s));de=np.zeros((s,s));error=y-x
        for i in range(s):
            for j in range(i+1,s):
                dx[i,j]=np.mean((x[i]-x[j])**2);dy[i,j]=np.mean((y[i]-y[j])**2)
                de[i,j]=np.mean((error[i]-error[j])**2)
        coefficient=n/(s*(s-1))
        row.update(repeated_susceptibility=float(coefficient*dx.sum()),
            onepass_susceptibility=float(coefficient*dy.sum()),paired_error_susceptibility=float(coefficient*de.sum()))
        means=[];contrasts=[]
        for sample in itertools.product(range(s),repeat=s):
            weights=[sample.count(i) for i in range(s)]
            means.append(sum(d[i] for i in sample)/s)
            contrasts.append(coefficient*sum(weights[i]*weights[j]*(dy[i,j]-dx[i,j])
                                            for i in range(s) for j in range(i+1,s)))
        row.update(paired_mean_percentiles=np.quantile(means,[.025,.975]).tolist(),
            delete_one_paired_means=[float(sum(d[j] for j in range(s) if j!=i)/(s-1)) for i in range(s)],
            susceptibility_difference_percentiles=np.quantile(contrasts,[.025,.975]).tolist())
    return row


def compare_metric(actual,expected,x,y):
    if set(actual)!=set(expected):raise AssertionError('The paired statistic inventory changed')
    # Independent centered-Gram and pair-distance programs may cancel
    # differently in degenerate empirical resamples. Use the complete field
    # scale rather than the possibly zero lower bootstrap percentile.
    s=len(x);dimension=x[0].size;eps=np.finfo(float).eps;operations=16*s*dimension+128
    gamma=operations*eps/(1-operations*eps)
    mean_scale=max(float(np.sqrt(np.mean(x*x))+np.sqrt(np.mean(y*y))),1e-300)
    chi_scale=max([abs(expected[k]) for k in ['repeated_susceptibility','onepass_susceptibility',
        'paired_error_susceptibility'] if expected[k] is not None]+[1e-300])
    maximum=0.;ratio=0.;values=0
    for key,want in expected.items():
        got=actual[key]
        if want is None:
            if got is not None:raise AssertionError('Unidentified uncertainty was imputed')
            continue
        if key=='seed_count':
            if got!=want:raise AssertionError('The whole-initialization count changed')
            continue
        want=np.asarray(want);got=np.asarray(got)
        if got.shape!=want.shape or not np.isfinite(got).all():raise AssertionError('Paired statistic shape changed')
        allowance=gamma*(chi_scale if 'susceptibility' in key else mean_scale)
        difference=np.abs(got-want);error=float(np.max(difference));maximum=max(maximum,error)
        if finite_greater(error, allowance, 'scripts/verify_repetition_comparison.py:77'):raise AssertionError('Independent paired statistic differs: '+key)
        ratio=max(ratio,error/allowance if allowance else 0.);values+=want.size
    return dict(values=values,maximum_absolute_difference=maximum,maximum_allowance_fraction=ratio,
                operation_count=operations,roundoff_gamma=gamma)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--panel',choices=['reference_schedule','constant','complete'],required=True);a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/repetition-analysis-selection.json';spec=load_json_strict(selection.read_text())
    folder=study/'analysis'/('repetition-'+a.panel);out=study/'verification'/('repetition-'+a.panel+'.json')
    if out.exists():raise FileExistsError(out)
    checked={}
    def check(path,digest=None):
        path=Path(path).resolve();key=str(path)
        if key not in checked:checked[key]=sha256(path)
        if digest is not None and checked[key]!=digest:raise AssertionError('Changed data-law analysis input: '+key)
    check(selection)
    for path,digest in spec['inputs_sha256'].items():check(path,digest)
    for name,digest in spec['producer_sources'].items():check(repo/name,digest)
    check(folder/'manifest.json');meta=load_json_strict((folder/'manifest.json').read_text())
    if meta['status']!='complete' or meta['panel']!=a.panel:raise AssertionError('Incomplete analysis panel')
    for name,key in [('binding.json','binding_sha256'),('results.json','results_sha256')]:check(folder/name,meta[key])
    binding=load_json_strict((folder/'binding.json').read_text())
    for path,digest in binding['inputs'].items():check(path,digest)
    for name,digest in binding['source_files'].items():check(folder/'source'/name,digest)
    report=load_json_strict((folder/'results.json').read_text());pairs=[q for q in spec['pairs'] if a.panel=='complete' or q['family']==a.panel]
    groups=defaultdict(list)
    for pair in pairs:groups[pair['family'],pair['recipe'],pair['heads'],pair['step']].append(pair)
    if report['pairs']!=len(pairs) or report['groups']!=len(groups) or len(report['states'])!=len(pairs):
        raise AssertionError('The selected paired panel is incomplete')
    for pair,state_record in zip(pairs,report['states'],strict=True):
        old=load_json_strict(Path(pair['old']['parent_manifest']).read_text())
        new=load_json_strict(Path(pair['new']['case']['parent_manifest']).read_text())
        if old['initial_parameter_sha256']!=new['initial_parameter_sha256']:
            raise AssertionError('A paired initialization differs')
        if state_record!=dict(name=pair['name'],initial_parameter_sha256=new['initial_parameter_sha256']):
            raise AssertionError('The exact paired-state identity changed')
    if a.panel=='complete':
        all_observations=study/'launcher-onepass-observations.json'
        check(all_observations,binding['inputs'][str(all_observations)])
        if load_json_strict(all_observations.read_text())['status']!='complete':
            raise AssertionError('Final decisions precede the complete compact family')
    actual={(q['family'],q['recipe'],q['heads'],q['step'],q['precision']):q for q in report['conditions']}
    expected_keys=set();records=[];path_values=defaultdict(list);active=set()
    for key,items in sorted(groups.items()):
        family,recipe,n,t=key;items=sorted(items,key=lambda q:q['seed']);seeds=[q['seed'] for q in items]
        for precision in spec['precision_preference']:
            if not all(precision in q['precisions'] for q in items):continue
            expected_keys.add((*key,precision));row=actual[*key,precision]
            if row['seeds']!=seeds or set(row['observables'])!=set(spec['fields']):raise AssertionError('A field or initialization was dropped')
            old=[fields(q['old']['collective'],precision) for q in items];new=[fields(q['new']['collective'],precision) for q in items]
            expected={};proofs={}
            for name in spec['fields']:
                x=np.stack([q[name] for q in old]);y=np.stack([q[name] for q in new])
                expected[name]=independent_summary(x,y,n)
                proofs[name]=compare_metric(row['observables'][name],expected[name],x,y)
            logs=(np.log10(np.maximum(expected['row_energy']['onepass_seed_means'],spec['row_log_floor']))-
                np.log10(np.maximum(expected['row_energy']['repeated_seed_means'],spec['row_log_floor'])))
            np.testing.assert_allclose(row['paired_log10_row_energy'],logs,rtol=1e-12,atol=1e-12)
            if finite_greater(abs(row['mean_absolute_log10_row_energy']-np.abs(logs).mean()), 1e-12, 'scripts/verify_repetition_comparison.py:139'):raise AssertionError('Row-energy screen changed')
            screens=set()
            if finite_greater(abs(expected['nll']['paired_mean']), .1, 'scripts/verify_repetition_comparison.py:141'):screens.add('nll')
            if finite_greater(float(np.abs(logs).mean()), .5, 'scripts/verify_repetition_comparison.py:142'):screens.add('row_energy')
            for name in spec['susceptibility_fields']:
                denominator=expected[name]['repeated_susceptibility'];value=None
                if denominator is not None and denominator>1e-10:
                    value=abs(expected[name]['onepass_susceptibility']-denominator)/denominator
                    if value>.25:screens.add(name)
                observed=row['susceptibility_relative_changes'][name]
                if value is None:
                    if observed is not None:raise AssertionError('Unavailable relative susceptibility was imputed')
                else:np.testing.assert_allclose(observed,value,rtol=2e-10,atol=1e-12)
            if row['screened_fields']!=sorted(screens):raise AssertionError('A replication screen changed')
            if precision=='float32':
                active.update(screens)
                for i,seed in enumerate(seeds):
                    path_values[family,recipe,n,seed].append(dict(step=t,
                        repeated_row=expected['row_ratio']['repeated_seed_means'][i],onepass_row=expected['row_ratio']['onepass_seed_means'][i],
                        repeated_nll=expected['nll']['repeated_seed_means'][i],onepass_nll=expected['nll']['onepass_seed_means'][i]))
            records.append(dict(condition=[*key,precision],statistics=proofs))
    if set(actual)!=expected_keys or len(actual)!=len(report['conditions']):raise AssertionError('The paired precision inventory changed')
    observed_paths={(q['family'],q['recipe'],q['heads'],q['seed']):q for q in report['paths']}
    if set(observed_paths)!=set(path_values):raise AssertionError('A selected path was dropped')
    for key,values in path_values.items():
        row=observed_paths[key];values=sorted(values,key=lambda q:q['step'])
        for x,y in zip(row['values'],values,strict=True):
            if x['step']!=y['step']:raise AssertionError('A paired path time changed')
            for name in ['repeated_row','onepass_row','repeated_nll','onepass_nll']:
                np.testing.assert_allclose(x[name],y[name],rtol=2e-12,atol=1e-15)
        for item,threshold in zip(row['persistence'],spec['collapse_thresholds'],strict=True):
            if item['threshold']!=threshold:raise AssertionError('A selected row threshold changed')
            for side in ['repeated','onepass']:
                flags=[v[side+'_row']<=threshold for v in values];first=flags.index(True) if any(flags) else None
                expected=dict(ever_below=any(flags),final_below=flags[-1],
                    first_observed_below=values[first]['step'] if first is not None else None,
                    returned_above=any(not v for v in flags[first:]) if first is not None else False)
                if item[side]!=expected:raise AssertionError('Sampled collapse persistence changed')
            reversal=any(item['repeated'][k]!=item['onepass'][k] for k in ['ever_below','final_below','returned_above'])
            if item['reversal']!=reversal:raise AssertionError('A persistence reversal changed')
            if reversal:active.add('sampled_persistence')
        early=next((v for v in values if v['step']==32768),None)
        if early is not None and values[-1]['step']>32768:
            change={side:values[-1][side+'_nll']-early[side+'_nll'] for side in ['repeated','onepass']}
            for side in change:np.testing.assert_allclose(row['late_loss'][side],change[side],rtol=1e-12,atol=1e-12)
            reversal=bool(np.sign(change['repeated'])!=np.sign(change['onepass']))
            if row['late_loss']['direction_reversal']!=reversal:raise AssertionError('Late-loss direction changed')
            if reversal:active.add('late_loss_direction')
        elif row['late_loss'] is not None:raise AssertionError('An unobserved late loss was imputed')
    if report['active_primary_screens']!=sorted(active):raise AssertionError('The combined replication decision screens changed')
    for pair,row in zip(pairs,report['risks'],strict=True):
        new=load_json_strict((Path(pair['new']['risk']).parent/'results.json').read_text())
        if row['name']!=pair['name'] or row['new_seen_training']!=new['cohorts']['training'] or row['new_heldout']!=new['cohorts']['heldout']:
            raise AssertionError('A risk-cohort identity changed')
        old=load_json_strict((Path(pair['old']['risk']).parent/'results.json').read_text())['cohorts'] if pair['old'].get('risk') else None
        if row['old']!=old:raise AssertionError('Repeated training-law risk changed or was imputed')
    if a.panel=='complete':
        if len(report['claim_decisions'])!=len(spec['claims']):raise AssertionError('A claim lacks its required disposition')
        for claim,decision in zip(spec['claims'],report['claim_decisions'],strict=True):
            if any(decision[k]!=v for k,v in claim.items()):raise AssertionError('A prior scientific claim was silently changed')
            if decision['screened_fields']!=sorted(set(claim.get('affected_by',[]))&active):raise AssertionError('A claim-specific screen changed')
            retain=['exact_rg_identities','released_checkpoint_inference']
            repeat=['common_variance_scaling','late_loss_and_row_return','recipe_regime_selection']
            expected=('retain_stated_scope' if claim['id'] in retain else
                'use_completed_singlepass_replication' if claim['id'] in repeat else
                'conditional_requirements_not_established_by_data_screen' if claim['id']=='critical_exponents_and_soc' else
                'restrict_existing_coefficients_to_repeated_data_law')
            if decision['disposition']!=expected or decision['additional_replication_required_for_transfer']!=(expected=='restrict_existing_coefficients_to_repeated_data_law'):
                raise AssertionError('A required replication or restriction disposition changed')

    elif report['claim_decisions']:raise AssertionError('Final dispositions require both complete panels')
    write_json(out,dict(status='passed',panel=a.panel,pairs=len(pairs),groups=len(groups),records=records,
        active_primary_screens=sorted(active),checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='Independent whole-initialization pair-distance covariance, exact resampling, paired means, saved-state '
        'persistence, held-out drift and replication-screen reconstruction. These checks do not establish criticality.'))
    print('Complete paired data-law panel independently verified',a.panel,flush=True)


if __name__=='__main__':main()
