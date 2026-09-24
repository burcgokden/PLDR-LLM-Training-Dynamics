#!/usr/bin/env python
"""Check the full intervention panel with independent scalar summation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256,write_json
from verify_scheduled_study import Evidence


def rebuild(folder):
    with np.load(folder/'measurements.npz') as raw:
        mask=raw['mask'];fields=['native_full_nll','native_proper_nll','full_nll64',
            'proper_nll64','score_error','forward_kl','oscillation']
        counts=[sum(bool(v) for v in row) for row in mask];total=sum(counts)
        contextual={};means={}
        for name in fields:
            contextual[name]=[math.fsum(float(v) for v,keep in zip(row,m,strict=True) if keep)/n
                for row,m,n in zip(raw[name],mask,counts,strict=True)]
            means[name]=math.fsum(float(v) for v,keep in zip(raw[name].flat,mask.flat,strict=True) if keep)/total
        return dict(valid_targets=total,contexts=len(mask),positions=mask.shape[1],targets_per_context=counts,
            context_means=contextual,means=means,
            maximum_forward_kl=max(map(float,raw['forward_kl'].flat)),
            maximum_oscillation=max(map(float,raw['oscillation'].flat)),
            maximum_absolute_score_difference=max(abs(float(v)) for v in raw['score_error'].flat))


def compare(actual,expected):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise AssertionError('Summary field coverage differs')
        for k,v in expected.items():compare(actual[k],v)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise AssertionError('Summary cohort coverage differs')
        for x,y in zip(actual,expected,strict=True):compare(x,y)
    elif isinstance(expected,float):
        if not math.isclose(actual,expected,rel_tol=2e-13,abs_tol=2e-13):
            raise AssertionError('Independent scalar intervention reduction differs')
    elif actual!=expected:raise AssertionError('Summary identity differs')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    out=study/'verification/decoder-prefix-intervention-summary.json'
    if out.exists():raise FileExistsError(out)
    files=Evidence(repo);selection=study/'protocols/decoder-prefix-intervention-selection.json'
    spec=files.load(selection);folder=study/'analysis/decoder-prefix-interventions';files.artifact(folder)
    result=files.load(folder/'results.json')
    if (result['status'],result['selected_states'],result['selected_interventions'],
            result['targets_per_state'],result['additional_independent_training_identities'])!=('complete',2,16,2048,0):
        raise AssertionError('A complete paired intervention panel is required')
    if result['selection_sha256']!=files.check(selection) or [r['case'] for r in result['records']]!=spec['cases']:
        raise AssertionError('Intervention selection changed')
    native={};checked_contexts=0
    for recipe in ['reference1','subcritical1']:
        candidates=[c for c in spec['cases'] if c['recipe']==recipe]
        paths={c['baseline_folder'] for c in candidates}
        if len(paths)!=1 or len(candidates)!=8:raise AssertionError('Unmatched native reference')
        path=Path(next(iter(paths)));files.artifact(path)
        files.proof(study/'verification/prefix-risk'/(path.name+'.json'),'verify_prefix_risk_comparison.py')
        native[recipe]=rebuild(path)
        compare(result['baselines'][recipe],dict(folder=str(path),summary=native[recipe]))
    for record,c in zip(result['records'],spec['cases'],strict=True):
        path=study/'measurements/decoder-prefix-risk'/c['name'];files.artifact(path)
        proof_path=study/'verification/decoder-prefix-risk'/(c['name']+'.json')
        proof=files.proof(proof_path,'verify_decoder_prefix_intervention.py')
        if proof['case']!=c or record['folder']!=str(path):raise AssertionError('Wrong intervention state')
        compare(record['verification'],files.record(proof_path))
        value=rebuild(path);base=native[c['recipe']];compare(record['summary'],value)
        compare(record['minus_native'],{name:value['means'][name]-base['means'][name] for name in value['means']})
        compare(record['proper_risk_change_by_context'],[x-y for x,y in
            zip(value['context_means']['proper_nll64'],base['context_means']['proper_nll64'],strict=True)])
        checked_contexts+=len(value['targets_per_context'])
    # A corrupted aggregate must be rejected by the same comparison routine.
    mutations=0
    for key in ['valid_targets','maximum_forward_kl','maximum_absolute_score_difference']:
        corrupt=dict(value);corrupt[key]+=1
        try:compare(corrupt,value)
        except AssertionError:mutations+=1
    if mutations!=3:raise AssertionError('A changed scalar summary escaped reconstruction')
    write_json(out,dict(status='passed',selected_states=2,selected_interventions=16,
        contextual_contrasts=checked_contexts,baseline_and_intervention_targets=18*2048,
        corrupted_summaries_rejected=mutations,checked_sha256=files.checked,
        verifier_sha256=sha256(__file__),verifier_sources={
            'scripts/verify_scheduled_study.py':sha256(repo/'scripts/verify_scheduled_study.py')},
        scope='Every selected intervention and both native references, including independent scalar '
        'reconstruction of all context means, aggregate risks and native-paired changes. '
        'The per-observation proofs separately reconstruct native logits, risks and intervention controls.'))
    print('All 16 decoder intervention summaries independently reconstructed',flush=True)


if __name__=='__main__':main()
