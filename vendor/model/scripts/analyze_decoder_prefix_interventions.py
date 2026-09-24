#!/usr/bin/env python
"""Summarize every selected decoder intervention on the fixed causal cohort."""
import argparse
import json
from pathlib import Path

import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from verify_scheduled_study import Evidence


def summarize(folder):
    with np.load(folder/'measurements.npz') as z:
        mask=z['mask']; count=mask.sum(axis=1)
        fields={name:z[name].astype(float) for name in
                ['native_full_nll','native_proper_nll','full_nll64','proper_nll64',
                 'score_error','forward_kl','oscillation']}
        contextual={name:(np.sum(value*mask,axis=1)/count).tolist()
                    for name,value in fields.items()}
        means={name:float(np.sum(value*mask)/mask.sum()) for name,value in fields.items()}
        return dict(valid_targets=int(mask.sum()),contexts=len(mask),positions=mask.shape[1],
            targets_per_context=count.tolist(),context_means=contextual,means=means,
            maximum_forward_kl=float(fields['forward_kl'].max()),
            maximum_oscillation=float(fields['oscillation'].max()),
            maximum_absolute_score_difference=float(np.abs(fields['score_error']).max()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/decoder-prefix-intervention-selection.json'
    spec=json.loads(selection.read_text());cases=spec['cases']
    expected={*(f'fixed_decoder_{i}' for i in range(1,6)),'fixed_all','row_decoder_4','row_all'}
    if len(cases)!=16 or any({c['mode'] for c in cases if c['recipe']==recipe}!=expected
                             for recipe in ['reference1','subcritical1']):
        raise AssertionError('The complete two-recipe, eight-intervention panel is required')
    # Fail before expensive hashing when any selected observation is unfinished.
    for c in cases:
        path=study/'verification/decoder-prefix-risk'/(c['name']+'.json')
        if not path.exists():raise AssertionError('Selected intervention is unfinished: '+c['name'])
    out=study/'analysis/decoder-prefix-interventions'
    if out.exists():raise FileExistsError(out)
    files=Evidence(repo);files.load(selection)
    for name,digest in spec['producer_sources'].items():files.check(repo/name,digest)
    baselines={};records=[]
    for c in cases:
        base=Path(c['baseline_folder']);recipe=c['recipe']
        if recipe not in baselines:
            files.proof(study/'verification/prefix-risk'/(base.name+'.json'),
                        'verify_prefix_risk_comparison.py')
            files.artifact(base)
            baselines[recipe]=dict(folder=str(base),summary=summarize(base))
        elif baselines[recipe]['folder']!=str(base):raise AssertionError('Unmatched native baseline')
        folder=study/'measurements/decoder-prefix-risk'/c['name']
        proof_path=study/'verification/decoder-prefix-risk'/(c['name']+'.json')
        proof=files.proof(proof_path,'verify_decoder_prefix_intervention.py')
        if proof['case']!=c or proof['native_risks_replayed_bytewise']!=4096:
            raise AssertionError('Incomplete native intervention reconstruction')
        files.artifact(folder);summary=summarize(folder);native=baselines[recipe]['summary']
        delta={name:summary['means'][name]-native['means'][name] for name in summary['means']}
        records.append(dict(case=c,folder=str(folder),verification=files.record(proof_path),
            summary=summary,minus_native=delta,
            proper_risk_change_by_context=(np.array(summary['context_means']['proper_nll64'])-
                np.array(native['context_means']['proper_nll64'])).tolist()))
    bind_run(out,list(map(Path,files.checked)),vars(a))
    write_json(out/'results.json',dict(schema='decoder-prefix-intervention-summary-v1',status='complete',
        selection_sha256=sha256(selection),baselines=baselines,records=records,
        selected_states=2,selected_interventions=16,targets_per_state=2048,
        additional_independent_training_identities=0,
        scope='All selected interventions on two completed single-pass endpoints and a fixed 32-context, '
        '64-position cohort. Targets within a context and interventions on a model are paired observations, '
        'not independent training replicas. Operator replacement can change prediction quality while '
        'reducing the full-window to proper-prefix defect. No confidence interval, critical exponent, '
        'uniform input-domain bound or population reasoning claim is inferred.'))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),selection_sha256=sha256(selection)))
    print('Complete decoder intervention panel summarized: 16 interventions, two native baselines',flush=True)


if __name__=='__main__':main()
