#!/usr/bin/env python
"""Summarize every scheduled risk, prefix, generation and task intervention."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


RESAMPLES = np.array(list(itertools.product(range(4), repeat=4)), dtype=int)


def empirical_mean(values):
    if len(values) != 4: raise AssertionError('Four whole-initialization identities are required')
    record = dict(seed_values=values, defined_seeds=sum(v is not None for v in values))
    if any(v is None for v in values):
        return dict(**record,mean=None,empirical_percentiles=None,leave_one_out=None)
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all(): raise AssertionError('Nonfinite scheduled summary')
    return dict(**record,mean=float(x.mean()),empirical_percentiles=np.quantile(x[RESAMPLES].mean(-1),[.025,.975]).tolist(),
        leave_one_out=[float(np.delete(x,i).mean()) for i in range(4)])


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
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    study=Path(a.root).resolve()/a.study; inputs=[]; verified={}
    analysis_protocol=study/'protocols/scheduled-analysis-selection.json'
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

    selection=load(study/'protocols/regime-observation-selection.json')
    prefix_selection=load(study/'protocols/scheduled-prefix-selection.json')
    reasoning_selection=load(study/'protocols/reasoning-mechanism-selection.json')
    cohort_path=study/'data/reasoning-cohort/cohort.json';cohort={x['id']:x for x in load(cohort_path)['examples']}
    prefixes={(c['recipe'],c['heads'],c['seed'],c['step']):c for c in prefix_selection['cases']}
    reasoning={(c['recipe'],c['heads'],c['seed'],c['step']):c for c in reasoning_selection['cases'] if c.get('role')=='scheduled_endpoint'}
    if len(selection['cases'])!=320 or len(prefixes)!=320 or len(reasoning)!=80:
        raise AssertionError('The complete scheduled prediction selection is required')
    records=[]
    for case in selection['cases']:
        key=(case['recipe'],case['heads'],case['seed'],case['step']);prefix_case=prefixes[key]
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
        c=record['case'];groups[c['recipe'],c['heads'],c['step']][c['seed']]=record
    summaries=[]
    for (recipe,n,step),seeds in sorted(groups.items()):
        if sorted(seeds)!=list(range(640101,640105)): raise AssertionError('A scheduled condition omits a selected initialization')
        names=set(seeds[640101]['scalars'])
        if any(set(row['scalars'])!=names for row in seeds.values()): raise AssertionError('A seed has different observation coverage')
        summaries.append(dict(recipe=recipe,heads=n,step=step,seeds=sorted(seeds),
            observations={name:empirical_mean([seeds[s]['scalars'][name] for s in sorted(seeds)]) for name in sorted(names)}))
    contrasts=[]; pairs=[(near,below) for near in ['reference1','reference2'] for below in ['subcritical1','subcritical2']]
    for near,below in pairs:
        for n in [4,14]:
            times=sorted({t for r,h,t in groups if r==near and h==n}&{t for r,h,t in groups if r==below and h==n})
            for step in times:
                left=groups[near,n,step];right=groups[below,n,step];fields=set(left[640101]['scalars'])
                if fields!=set(right[640101]['scalars']): raise AssertionError('Paired recipe coverage differs')
                differences={}
                for field in sorted(fields):
                    values=[]
                    for seed in range(640101,640105):
                        x=left[seed]['scalars'][field];y=right[seed]['scalars'][field]
                        values.append(x-y if x is not None and y is not None else None)
                    differences[field]=empirical_mean(values)
                contrasts.append(dict(first=near,second=below,heads=n,step=step,observations=differences))
    out=study/'analysis/predictions';out.mkdir(parents=True,exist_ok=False)
    bind_run(out,list(dict.fromkeys(inputs)),vars(a))
    write_json(out/'results.json',dict(status='complete',schema='scheduled-prediction-analysis-v1',states=records,
        conditions=summaries,paired_recipe_contrasts=contrasts,selected_states=320,task_and_generation_states=80,
        scope='All selected reference-recipe outcomes, proper-prefix comparisons, generation observations and task interventions are retained. Four near/below recipe contrasts are paired by complete initialization identity at every common saved horizon. The recipe names do not assign physical phases to the observed models.',
        uncertainty='The fixed context, task and sampling laws are conditioned upon. Each summary enumerates all 256 four-initialization resamples. These empirical percentile ranges have no guaranteed population coverage and are not adjusted significance claims. Undefined mean-normalized generation ratios retain null values and prevent a four-seed group mean rather than dropping seeds.',
        inference='Predictive entropy and stopping statistics do not define coherence. Task-margin retention and correct counterfactual premise response are separate observations. None of these statistics alone establishes a thermodynamic exponent or endogenous approach to a critical surface.'))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),states=320,conditions=80))
    print('Complete scheduled prediction panel:',len(records),'states and',len(contrasts),'paired recipe conditions',flush=True)


if __name__=='__main__':main()
