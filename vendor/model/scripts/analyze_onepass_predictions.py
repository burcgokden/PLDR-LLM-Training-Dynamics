#!/usr/bin/env python
"""Summarize every single-pass risk, prefix, generation and task intervention."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

from onepass_analysis import group_key, group_record, empirical_mean
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def task_summary(modes, cohort):
    native={x['id']:x for x in modes['native']['examples']}; summaries=[]
    for name,mode in modes.items():
        groups=defaultdict(list); normalized=defaultdict(list); pairs=defaultdict(dict); errors=[]
        changed=certified=positive=0
        for item in mode['examples']:
            reference=native[item['id']]; definition=cohort[item['id']]
            groups[item['task']].append(item['prediction']==item['correct'])
            normalized[item['task']].append(item['token_mean_prediction']==item['correct'])
            error=max(abs(x-y) for x,y in zip(item['scores'],reference['scores'],strict=True)); errors.append(error)
            changed+=item['prediction']!=reference['prediction']; positive+=reference['margin']>0
            certified+=reference['margin']>2*error
            if reference['margin']>2*error and item['prediction']!=item['correct']:
                raise AssertionError('A certified task margin was lost')
            if 'pair' in definition: pairs[definition['pair']][definition['variant']]=(item,definition)
        responses=defaultdict(list); correctness=defaultdict(list)
        for pair in pairs.values():
            if set(pair)!={0,1}: raise AssertionError('A task counterfactual pair is incomplete')
            left,definition=pair[0];right,_=pair[1]
            if (left['correct'],right['correct'])!=(0,1): raise AssertionError('Counterfactual answer orientation changed')
            response=(right['scores'][1]-right['scores'][0])-(left['scores'][1]-left['scores'][0])
            responses[definition['depth']].append(response)
            correctness[definition['depth']].append(left['prediction']==0 and right['prediction']==1)
        summaries.append(dict(mode=name,accuracy={k:float(np.mean(v)) for k,v in sorted(groups.items())},
            token_mean_accuracy={k:float(np.mean(v)) for k,v in sorted(normalized.items())},
            external_target_nll=mode['external_target_nll'],paired_external_nll_change=mode['external_target_nll']-modes['native']['external_target_nll'],
            answer_decision_changes=changed,strictly_correct_native=positive,certified_preserved_correct=certified,
            maximum_candidate_score_error=max(errors),mean_maximum_candidate_score_error=float(np.mean(errors)),
            premise_responses={str(depth):dict(pairs=len(values),mean_signed_response=float(np.mean(values)),
                positive_response_fraction=float(np.mean(np.array(values)>0)),both_correct_fraction=float(np.mean(correctness[depth])))
                for depth,values in sorted(responses.items())}))
    return summaries


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    study=Path(a.root).resolve()/a.study; inputs=[]; verified={}
    analysis_protocol=study/'protocols/onepass-analysis-selection.json'
    contract=json.loads(analysis_protocol.read_text());inputs.append(analysis_protocol)
    for name,digest in contract['producer_sources'].items():
        if sha256(Path(__file__).resolve().parents[1]/name)!=digest:
            raise AssertionError('A recorded scheduled prediction analysis implementation changed')

    def load(path):
        path=Path(path);inputs.append(path);return json.loads(path.read_text())

    def proof(path,case=None):
        record=load(path)
        if record['status']!='passed' or (case is not None and record['case']!=case):
            raise AssertionError('Every selected scheduled observation must be independently verified')
        for name,digest in record['checked_sha256'].items():
            if name in verified and verified[name]!=digest: raise AssertionError('Conflicting verification bindings')
            verified[name]=digest
        return record

    def result(folder):
        folder=Path(folder);meta=load(folder/'manifest.json')
        if meta['status']!='complete': raise AssertionError('An incomplete observation entered the analysis')
        for name,key in [('manifest.json',None),('results.json','results_sha256')]:
            path=folder/name;signature=sha256(path)
            if verified[str(path)]!=signature or (key is not None and meta[key]!=signature):
                raise AssertionError('A verified result changed')
        return load(folder/'results.json')

    selection=load(study/'protocols/onepass-observation-selection.json')
    prefix_selection=load(study/'protocols/scheduled-prefix-selection.json')
    reasoning_selection=load(study/'protocols/reasoning-mechanism-selection.json')
    cohort_path=Path(reasoning_selection['cohort']);cohort={x['id']:x for x in load(cohort_path)['examples']}
    prefixes={(c['run_id'],c['step']):c for c in prefix_selection['cases']}
    reasoning={(c['run_id'],c['step']):c for c in reasoning_selection['cases']}
    if len(selection['cases'])!=284 or len(prefixes)!=284 or len(reasoning)!=94:
        raise AssertionError('The complete scheduled prediction selection is required')
    records=[]
    for case in selection['cases']:
        key=(case['run_id'],case['step']);prefix_case=prefixes[key]
        proof(study/'verification/observations'/(case['name']+'.json'),case)
        proof(study/'verification/prefix'/(prefix_case['name']+'.json'),prefix_case)
        risk=result(study/'measurements'/('risk-'+case['name'].removeprefix('cpu-')))
        prefix=result(study/'measurements'/prefix_case['name']);scalars={}
        for cohort_name,values in risk['cohorts'].items():
            for field in ['last_nll','last_entropy','all_token_weighted_nll','all_token_weighted_entropy']:
                scalars[cohort_name+'_'+field]=values[field]
        for objective in ['last_nll','all_token_weighted_nll']:
            scalars['generalization_gap_'+objective]=risk['cohorts']['heldout'][objective]-risk['cohorts']['training'][objective]
        scalars['preceding_online_loss']=risk['online_window']['preupdate_mean_loss']
        proper=[r for r in prefix['conditions'] if r['variant']=='prefix' and r['length']<64]
        if [r['length'] for r in proper]!=[16,32,48]: raise AssertionError('The prefix risk panel changed')
        scalars['parallel_nll_prefix_panel']=float(np.mean([r['full_window_nll'] for r in proper]))
        scalars['proper_nll_prefix_panel']=float(np.mean([r['nll'] for r in proper]))
        for variant in ['prefix','target_swap','suffix_swap']:
            rows=[r for r in prefix['conditions'] if r['variant']==variant and r['length']<64]
            for field in ['nll_change','mean_predictive_kl','mean_relative_G_difference','mean_row_ratio']:
                scalars[variant+'_'+field]=float(np.mean([r[field] for r in rows]))
            scalars[variant+'_maximum_logit_difference']=max(r['maximum_logit_difference'] for r in rows)
        record=dict(case=case,scalars=scalars,prefix_conditions=prefix['conditions'],risk=risk['cohorts'])
        if case['inference_stability']:
            inferred=result(study/'measurements'/('inference-'+case['name'].removeprefix('cpu-')))
            for tensor,comparisons in inferred['comparisons'].items():
                for comparison,values in comparisons.items():
                    for field in ['rmse','mean_normalized_rmse','rms_normalized_rmse','mean_magnitude_to_rms']:
                        scalars[f'generation_{tensor}_{comparison}_{field}']=values[field]
            scalars['mean_generated_length']=sum(inferred['generated_tokens'])/200
            scalars['eos_stop_fraction']=sum(inferred['eos_stops'])/200
            task_case=reasoning[key];proof(study/'verification/reasoning'/(task_case['name']+'.json'),task_case)
            task=result(study/'measurements'/task_case['name']);modes={m['mode']:m for m in task['modes']}
            record['task_interventions']=task_summary(modes,cohort)
            for mode in record['task_interventions']:
                for metric in ['accuracy','token_mean_accuracy']:
                    for task_name,value in mode[metric].items(): scalars[f'task_{mode["mode"]}_{metric}_{task_name}']=value
                for field in ['external_target_nll','paired_external_nll_change','answer_decision_changes','strictly_correct_native',
                              'certified_preserved_correct','maximum_candidate_score_error','mean_maximum_candidate_score_error']:
                    scalars[f'task_{mode["mode"]}_{field}']=mode[field]
                for depth,values in mode['premise_responses'].items():
                    for field in ['mean_signed_response','positive_response_fraction','both_correct_fraction']:
                        scalars[f'task_{mode["mode"]}_depth{depth}_{field}']=values[field]
        records.append(record)
    if sha256(cohort_path) != verified[str(cohort_path)]:
        raise AssertionError('The verified counterfactual task cohort changed')
    groups=defaultdict(dict)
    for record in records:
        c=record['case'];key=group_key(c)
        if c['seed'] in groups[key]:raise AssertionError('Duplicate drive and initialization identity')
        groups[key][c['seed']]=record
    expected={tuple(r[k] for k in contract['group_keys']):r['seeds'] for r in contract['groups']}
    if {k:sorted(v) for k,v in groups.items()}!=expected:
        raise AssertionError('The complete single-pass group coverage changed')
    summaries=[]
    for key,seeds in sorted(groups.items()):
        names=set().union(*(r['scalars'] for r in seeds.values()));observations={}
        for name in sorted(names):
            selected=[s for s in sorted(seeds) if name in seeds[s]['scalars']]
            unselected=[s for s in sorted(seeds) if s not in selected]
            if unselected and not (name.startswith('generation_') or name.startswith('task_') or name in ['mean_generated_length','eos_stop_fraction']):
                raise AssertionError('A required core prediction observation is missing')
            observations[name]=dict(**empirical_mean([seeds[s]['scalars'][name] for s in selected]),
                seed_ids=selected,unselected_seed_ids=unselected)
        summaries.append(dict(**group_record(key),seeds=sorted(seeds),observations=observations))
    contrasts=[]
    for definition in contract['recipe_contrasts']:
        left=groups[tuple(definition['first_key'])];right=groups[tuple(definition['second_key'])]
        seeds=definition['seeds']
        if sorted(left)!=seeds or sorted(right)!=seeds:
            raise AssertionError('The paired recipe initialization coverage changed')
        names=set().union(*(r['scalars'] for r in left.values())) & set().union(*(r['scalars'] for r in right.values()))
        differences={}
        for field in sorted(names):
            selected=[s for s in seeds if field in left[s]['scalars'] and field in right[s]['scalars']]
            values=[]
            for seed in selected:
                x=left[seed]['scalars'][field];y=right[seed]['scalars'][field]
                values.append(x-y if x is not None and y is not None else None)
            differences[field]=dict(**empirical_mean(values),seed_ids=selected,
                unselected_seed_ids=[s for s in seeds if s not in selected])
        contrasts.append(dict(**definition,observations=differences))
    out=study/'analysis/onepass-predictions';out.mkdir(parents=True,exist_ok=False)
    bind_run(out,list(dict.fromkeys(inputs)),vars(a))
    write_json(out/'results.json',dict(status='complete',schema='onepass-prediction-analysis-v1',states=records,
        conditions=summaries,paired_recipe_contrasts=contrasts,selected_states=284,task_and_generation_states=94,
        scope=contract['predictive_laws'],uncertainty=contract['uncertainty'],
        coverage=contract['endpoint_coverage'],undefined_ratios=contract['undefined_ratios'],
        inference='Predictive entropy and stopping statistics do not define coherence. Task-margin retention and correct counterfactual premise response are separate observations. No finite task statistic alone establishes a thermodynamic exponent or endogenous approach to a critical surface.'))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),states=284,conditions=80))
    print('Complete single-pass prediction panel:',len(records),'states and',len(contrasts),'paired recipe conditions',flush=True)


if __name__=='__main__':main()
