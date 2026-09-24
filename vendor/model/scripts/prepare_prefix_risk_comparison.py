#!/usr/bin/env python
"""Freeze complete-position causal risk comparisons before selected outcomes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/prefix-risk-comparison.json'
    if selection.exists():raise FileExistsError(selection)
    observations=study/'protocols/onepass-observation-selection.json'
    selected=json.loads(observations.read_text());retained=study/'protocols/repeated-risk-comparison.json'
    inputs=[observations,retained,study/'verification/gpu-qualification.json',
        study/'verification/repeated-checkpoint-sealing.json',
        root/'critical-scaling-20260906/verification/full-selected/raw.json',
        root/'controlled-study-20260905/data/short/tokens.npy',
        root/'controlled-study-20260905/data/short/offsets.npy']
    cases=[]
    for c in json.loads(retained.read_text())['cases']:
        v={k:value for k,value in c.items() if k!='sealed_manifest'}
        v.update(name='causal-'+c['name'],parent_manifest=c['sealed_manifest'],
                 data_law='repeated',family='reference_schedule',source_role='sealed_censored_checkpoint')
        cases.append(v);inputs.append(Path(c['sealed_manifest']))
    for c in selected['cases']:
        take=(c['role']=='data_schedule' and c['step'] in [32768,65536]) or (
            c['role']=='data_constant' and c['seed']==640101 and c['step'] in [32768,131072]) or (
            c['role']=='compact' and c['recipe'] in ['reference1','subcritical1'] and
            c['heads']==14 and c['seed']==640101 and c['step'] in [32768,40960])
        if take:
            v={k:value for k,value in c.items() if not k.startswith('training_cohort')}
            v.update(name='causal-'+c['name'].removeprefix('cpu-'),data_law='onepass',
                family=c['role'],source_role='verified_complete_trajectory',
                training_verification=str(study/'verification/training'/(c['run_id']+'.json')))
            cases.append(v)
    for n in [4,14]:
        for t in [32768,131072]:
            run=f'{"long" if t==32768 else "extended"}-h{n}-g1-s640101'
            parent=root/'critical-scaling-20260906/runs'/run/'manifest.json';m=json.loads(parent.read_text())
            if m['status']!='complete' or m['completed_step']!=t:raise AssertionError('Constant counterpart incomplete')
            state=parent.parent/'final-training-state.pt'
            cases.append(dict(name=f'causal-repeated-constant-h{n}-s640101-t{t}',run_id=run,
                heads=n,seed=640101,step=t,shared_seed=640011,stream_seed=640001,
                recipe='controlled',state=str(state),state_sha256=m['checkpoint_sha256'],
                parent_manifest=str(parent),data_law='repeated',family='constant',
                source_role='verified_complete_trajectory'))
            inputs.append(parent)
    if len(cases)!=20:raise AssertionError('The complete twenty-state causal comparison changed')
    qrun='qualification-reference1-h14-d768-continuous';qp=study/'runs'/qrun/'manifest.json'
    qm=json.loads(qp.read_text());inputs.append(qp)
    qualification=dict(name='causal-qualification-reference1-h14',run_id=qrun,heads=14,seed=650631,step=1024,
        shared_seed=640011,stream_seed=650641,recipe='reference1',state=str(qp.parent/'final-training-state.pt'),
        state_sha256=qm['checkpoint_sha256'],parent_manifest=str(qp),data_law='onepass',
        family='qualification',source_role='execution_qualification')
    names=['scripts/prepare_prefix_risk_comparison.py','scripts/measure_prefix_risk_comparison.py',
        'scripts/verify_prefix_risk_comparison.py','scripts/run_prefix_risk_comparison.py',
        'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/controlled.py',
        'src/model_rg/provenance.py']
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    inputs.extend(source/name for name in ['modeling_pldrllm.py','configuration_pldrllm.py','config.json'])
    write_json(selection,dict(schema='complete-position-prefix-risk-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        cases=cases,qualification=qualification,source=str(source),rows=list(range(512,544)),
        prefix_lengths=list(range(1,65)),batch_size=32,threads=4,
        inputs_sha256={str(path):sha256(path) for path in inputs},
        producer_sources={name:sha256(repo/name) for name in names},
        prior_information='Original constant outcomes, six early risk observations, retained schedule checkpoint logs, '
        'and released-model prefix/task results are known. New single-pass training has begun. This entire panel, '
        'including four compact-schedule endpoints and all proper positions, is fixed before any new causal-risk score.',
        scope='Twenty diagnostic states with one initialization per matched condition; no susceptibility or population '
        'accuracy estimate is inferred from this panel. All2048 target positions per state are evaluated, including '
        'all64 proper prefixes of each of32 fixed contexts. Full-window logits can depend on future tokens through '
        'the sequence-wide metric. Proper-prefix logits have access only to their stated prefix. Native float32 '
        'reductions and mathematical float64 log scores are retained separately. At prefix64 the two full projections '
        'must be bytewise identical. Zero targets are masked only in all-target aggregate risk.'))
    print('Frozen twenty complete-position comparisons and one execution qualification',sha256(selection),flush=True)


if __name__=='__main__':main()
