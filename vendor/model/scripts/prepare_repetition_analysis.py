#!/usr/bin/env python
"""Freeze paired data-law analysis and the inventory of claims needing decisions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    output=study/'protocols/repetition-analysis-selection.json'
    if output.exists():raise FileExistsError(output)
    native=study/'protocols/onepass-training-selection.json';natives=json.loads(native.read_text())
    observation=study/'protocols/onepass-observation-selection.json';observations=json.loads(observation.read_text())
    repeated_selection=study/'protocols/repeated-risk-comparison.json'
    repeated=json.loads(repeated_selection.read_text());const=root/'critical-scaling-20260906'
    inputs=[native,observation,repeated_selection,const/'verification/full-selected/raw.json',
        const/'verification/full-selected/statistics.json',study/'verification/repeated-risk-comparison.json',
        study/'repeated-observations/protocols/repeated-observation-selection.json',
        study/'qualification/repetition-analysis/verification.json']
    lookup={}
    for path in (const/'measurements').glob('cpu-*/manifest.json'):
        m=json.loads(path.read_text());c=m['case']
        if (c.get('multiplier')!=1 or c.get('shared_seed')!=640011 or c.get('stream_seed')!=640001 or
            c.get('heads') not in [4,8,14] or c.get('seed') not in range(640101,640105) or
            c['step'] not in [8192,32768,65536,98304,131072]):continue
        key=c['heads'],c['seed'],c['step']
        if key in lookup:raise AssertionError('Duplicate matched constant observation')
        lookup[key]=(path,m)
    pairs=[]
    for c in observations['cases']:
        if c['role'] not in ['data_constant','data_constant_short','data_schedule']:continue
        if c['role']=='data_schedule' and c['step'] not in [32768,65536]:continue
        fresh=dict(case=c,collective=str(study/'measurements'/c['name']/'manifest.json'),
            risk=str(study/'measurements'/('risk-'+c['name'].removeprefix('cpu-'))/'manifest.json'),
            verification=str(study/'verification/observations'/(c['name']+'.json')),
            native_verification=str(study/'verification/training'/(c['run_id']+'.json')))
        if c['role']=='data_schedule':
            old_case=next(q for q in repeated['cases'] if q['recipe']==c['recipe'] and q['step']==c['step'])
            path=study/'repeated-observations/measurements'/('cpu-'+old_case['name'])/'manifest.json'
            old=dict(case=old_case,collective=str(path),
                risk=str(study/'measurements/repeated-risk'/old_case['name']/'manifest.json'),
                risk_kind='full_position_reference',
                verification=str(study/'repeated-observations/verification/collectives'/('cpu-'+old_case['name']+'.json')),
                risk_verification=str(study/'verification/repeated-risk-comparison.json'),
                parent_manifest=old_case['sealed_manifest'])
            family='reference_schedule'; precisions=['float32','float64']
        else:
            path,meta=lookup[c['heads'],c['seed'],c['step']];old_case=meta['case']
            parent=Path(old_case['state']).parent/'manifest.json'
            old=dict(case=old_case,collective=str(path),parent_manifest=str(parent),
                verification=str(const/'verification/full-selected/raw.json'),risk_kind='last_target_training_law')
            risk=const/'measurements'/f'training-risk-h{c["heads"]}-t{c["step"]}-s{c["seed"]}'/'manifest.json'
            if risk.exists():
                old['risk']=str(risk);old['risk_verification']=str(const/'verification/training-risk.json')
                inputs.append(risk)
            inputs.extend([path,parent]);family='constant';precisions=[v['precision'] for v in meta['precision_records']]
        pairs.append(dict(name=f'{family}-{c["recipe"]}-h{c["heads"]}-s{c["seed"]}-t{c["step"]}',
            family=family,recipe=c['recipe'],heads=c['heads'],seed=c['seed'],step=c['step'],old=old,new=fresh,precisions=precisions))
    if len(pairs)!=40 or sum(q['family']=='constant' for q in pairs)!=36:
        raise AssertionError('The full data-law comparison must contain36 constant and4 reference pairs')
    groups={}
    for pair in pairs:
        key=f'{pair["family"]}-{pair["recipe"]}-h{pair["heads"]}-t{pair["step"]}'
        groups.setdefault(key,[]).append(pair['seed'])
    if sorted(map(len,groups.values())) != [1]*4+[2]*6+[4]*6:
        raise AssertionError('Matched initialization replication changed')
    claims=[
        dict(id='exact_rg_identities',kind='analytical',scope='Exact pushforward, composition, conditional-kernel and inference-risk statements under their stated hypotheses.',
             action='Retain after explicitly including sampler state, remaining data and external schedule coordinates. Training reruns are not tests of an algebraic identity.'),
        dict(id='released_checkpoint_inference',kind='fixed_checkpoint',scope='Two released models, proper-prefix and operator interventions, prompting and complete ARC results.',
             action='Retain the verified checkpoint/task scope. Controlled-corpus reruns do not recreate or invalidate the released pretraining histories.'),
        dict(id='collapse_clock_power',kind='data_dependent_training',scope='The repeated-law rate-clock power and threshold-crossing predictions.',
             affected_by=['row_energy','row_ratio','sampled_persistence'],
             action='Do not transfer the fitted power to single-pass pretraining without a separately frozen rate/width replication. It remains a finite resampled-law clock coefficient, not a critical exponent.'),
        dict(id='common_variance_scaling',kind='data_dependent_training',scope='Common-sector means, cross-head covariance and finite-width susceptibility amplitudes.',
             affected_by=['common_centroid','row_ratio','row_energy','logit_projection'],
             action='Use the matched single-pass width panel to re-estimate and test any asserted coefficient or scaling relation. Restrict unmatched widths, environments and fitted coefficients to their original data law.'),
        dict(id='late_loss_and_row_return',kind='data_dependent_training',scope='Late loss drift and row re-expansion on matched native paths.',
             affected_by=['nll','row_energy','row_ratio','sampled_persistence'],
             action='Replace transferred descriptions by actual paired single-pass outcomes, retain all selected initializations, and use focused continuations if additional replication is needed for a claimed general pattern.'),
        dict(id='optimizer_noise_and_time',kind='data_dependent_training',scope='Frozen native-step forcing amplitudes, Adam transport and rate/time fits.',
             affected_by=['common_centroid','nll','row_energy'],
             action='The exact decomposition remains valid. Existing measured coefficients describe their conditional repeated-data states; single-pass coefficients require new state-conditioned measurements before transfer.'),
        dict(id='recipe_regime_selection',kind='data_dependent_training',scope='Which row, operator and prediction regimes the source Near/Below recipes select.',
             affected_by=['row_ratio','row_energy','nll','sampled_persistence','prefix_risk','task_margins'],
             action='Use the completed compact single-pass family and matched schedule controls. Source labels cannot substitute for a measured phase or imply criticality.'),
        dict(id='critical_exponents_and_soc',kind='conditional_physics',scope='Thermodynamic singularities, critical exponents and endogenous attraction to a critical surface.',
             action='Require an identified joint limit, critical surface, appropriate scaling window, predictive visibility and an endogenous approach mechanism. Neither a passed data-law screen nor imposed annealing establishes these properties.')]
    names=['scripts/prepare_repetition_analysis.py','scripts/analyze_repetition_comparison.py',
        'scripts/verify_repetition_comparison.py','scripts/run_repetition_analysis.py','scripts/check_repetition_analysis.py',
        'src/model_rg/controlled.py','src/model_rg/provenance.py']
    spec=dict(schema='paired-data-law-analysis-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        pairs=pairs,groups=groups,claims=claims,inputs_sha256={str(path):sha256(path) for path in inputs},
        producer_sources={name:sha256(repo/name) for name in names},
        replication_rule=natives['repetition_assessment'],
        fields=['row_ratio','row_energy','total_energy','common_centroid','nll','entropy','logit_projection','context_kl'],
        precision_preference=['float32','float64'],primary_precision='float32',
        precision_scope='Every pair has matched native float32. Float64 comparisons are included only for whole initialization groups with existing float64 observations under both data laws; unavailable old precision states are not imputed.',
        row_log_floor=1e-30,collapse_thresholds=[.01,.001,.0001,.00001],
        threshold_scope='Collapse and subsequent return are evaluated only at the selected saved states, not inferred between checkpoints. Each threshold and every path are retained.',
        susceptibility_fields=['row_ratio','row_energy','common_centroid','nll','logit_projection'],
        uncertainty='Exact empirical whole-initialization resampling: all s**s resamples for s in{2,4}; single-initialization reference comparisons have no seed uncertainty interval. All paired identities are retained across data laws and times. Contexts and shared streams are conditioned quantities.',
        training_risk_scope='The old constant-risk cohort is sampled from the fixed repeated training law. The new cohort samples already consumed blocks. Their means describe different cohort laws and are not a matched-context memorization estimator. Held-out contexts are identical and provide the primary risk comparison. Old training-risk measurements are unavailable for sixteen early/N8 pairs; no value is imputed.',
        prior_information='All repeated-risk outcomes and native repeated-corpus logs are known. Single-pass paths have begun. No selected complete data-law endpoint comparison has yet been analyzed.',
        decision_scope='Every claim receives an explicit disposition after all selected pair groups are analyzed. A screen triggers focused replication if a transferred claim is needed, otherwise an explicit restriction to its supported data law. No-screen outcomes are not equivalence proofs.')
    write_json(output,spec);print('Frozen40 data-law state pairs and8 claim decisions',sha256(output),flush=True)


if __name__=='__main__':main()
