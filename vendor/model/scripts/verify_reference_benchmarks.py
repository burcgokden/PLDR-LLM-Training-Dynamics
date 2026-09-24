#!/usr/bin/env python
"""Independently reconstruct every retained full-ARC proper-prefix probability and score."""
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256,write_json


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=3e-10,atol=3e-13)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');p.add_argument('--case',required=True)
    p.add_argument('--selection',default='reference-benchmark-selection.json');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;folder=study/'measurements'/a.case
    output=study/'verification/reasoning'/(a.case+'.json')
    if output.exists():raise FileExistsError(output)
    checked={};cache={}
    def check(path,want=None):
        path=Path(path).resolve();st=path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=sha256(path)
        got=cache[key]
        if want is not None and want!=got:raise AssertionError('Changed reasoning evidence: '+str(path))
        checked[str(path)]=got;return got
    selection=study/'protocols'/a.selection;check(selection);spec=load_json_strict(selection.read_text())
    cases=[c for c in spec['cases'] if c['name']==a.case]
    if len(cases)!=1:raise AssertionError('Missing or ambiguous reasoning case')
    case=cases[0]
    for name,digest in spec['inputs_sha256'].items():check(name,digest)
    for name,digest in case.get('input_sha256',{}).items():check(name,digest)
    check(spec['cohort'],spec['cohort_sha256']);examples=load_json_strict(Path(spec['cohort']).read_text())['examples']
    meta=load_json_strict((folder/'manifest.json').read_text());check(folder/'manifest.json')
    if meta['status']!='complete' or meta['case']!=case:raise AssertionError('Incomplete or incorrect reasoning state')
    for name,key in [('binding.json','binding_sha256'),('measurements.npz','raw_sha256'),('results.json','results_sha256'),('prediction-index.json','prediction_index_sha256')]:check(folder/name,meta[key])
    for name,digest in meta['logit_sidecars'].items():check(folder/name,digest)
    binding=load_json_strict((folder/'binding.json').read_text())
    for name,digest in binding['inputs'].items():check(name,digest)
    for name,digest in binding['source_files'].items():check(folder/'source'/name,digest)
    for name,digest in spec['producer_sources'].items():check(folder/'source'/name,digest)
    result=load_json_strict((folder/'results.json').read_text());index=load_json_strict((folder/'prediction-index.json').read_text())
    if result['case']!=case or [r['mode'] for r in result['modes']]!=spec['modes']:raise AssertionError('Reasoning result selection changed')
    with np.load(folder/'measurements.npz') as z:raw={k:z[k] for k in z.files}
    probe=root/'controlled-study-20260905/data/short';tokens=np.load(probe/'tokens.npy',mmap_mode='r');offsets=np.load(probe/'offsets.npy')
    calibration=tokens[np.arange(64)[:,None],offsets[:64,None]+np.arange(64)]
    rows=np.arange(512,576);language=tokens[rows[:,None],offsets[rows,None]+np.arange(65)]
    for name,value in [('calibration_inputs',calibration),('language_rows',rows),('language_crops',language)]:np.testing.assert_array_equal(raw[name],value)
    full=raw['calibration_cache'];fixed=raw['fixed_cache'];permuted=raw['permuted_cache']
    expected=full.astype(float).mean(2,keepdims=True).astype(np.float32)
    np.testing.assert_array_equal(fixed,expected)
    if full.dtype!=np.float32 or full.shape[:3]!=(5,3,64) or full.shape[-2:]!=(64,64):raise AssertionError('Invalid calibration operator layout')
    np.testing.assert_array_equal(permuted[:,:2],fixed[:,:2])
    moved=np.roll(fixed[:,2],1,axis=2).astype(float)
    target_rms=np.sqrt(np.mean(fixed[:,2].astype(float)**2,axis=(-2,-1),keepdims=True))
    source_rms=np.sqrt(np.mean(moved*moved,axis=(-2,-1),keepdims=True))
    if np.any((source_rms==0)&(target_rms!=0)):raise AssertionError('Undefined operator norm matching')
    factor=np.divide(target_rms,source_rms,out=np.zeros_like(target_rms),where=source_rms>0)
    np.testing.assert_allclose(permuted[:,2],(moved*factor).astype(np.float32),rtol=3e-7,atol=1e-25)
    norm_error=float(np.max(np.abs(np.sqrt(np.mean(permuted[:,2].astype(float)**2,axis=(-2,-1),keepdims=True))-target_rms)/np.maximum(target_rms,1e-30)))
    if finite_greater(norm_error, 3e-7, 'scripts/verify_reference_benchmarks.py:64'):raise AssertionError('Permuted operator norms were not preserved')
    operator_change=np.sqrt(np.mean((permuted[:,2].astype(float)-fixed[:,2])**2,axis=(-2,-1)))/np.maximum(target_rms[...,0,0],1e-30)
    table=[];mapping={};jobs=[]
    def add(sequence,target,**fields):
        key=tuple(sequence)
        if key not in mapping:mapping[key]=len(table);table.append(list(sequence))
        jobs.append(dict(input_index=mapping[key],target=int(target),**fields))
    for i,item in enumerate(examples):
        for c,answer in enumerate(item['answer_tokens']):
            for k,target in enumerate(answer):add(item['prompt_tokens']+answer[:k],target,kind='task',example=i,choice=c,offset=k)
    for i,crop in enumerate(language):add(crop[:64].tolist(),crop[64],kind='language',index=i)
    if index!={'inputs':table,'jobs':jobs}:raise AssertionError('A reasoning prediction is not its exact proper prefix')
    if result['unique_prediction_prefixes']!=len(table) or result['answer_and_language_token_predictions']!=len(jobs):raise AssertionError('Incorrect task prediction count')
    inputs=np.array([j['input_index'] for j in jobs]);targets=np.array([j['target'] for j in jobs])
    np.testing.assert_array_equal(raw['job_inputs'],inputs);np.testing.assert_array_equal(raw['targets'],targets)
    if raw['token_log_probabilities'].shape!=(4,len(jobs)):raise AssertionError('Reasoning mode or token count changed')
    summaries=[];logit_elements=0;max_error=0.
    for mode_index,mode in enumerate(spec['modes']):
        logits=np.load(folder/('logits-'+mode+'.npy'),mmap_mode='r')
        if logits.dtype!=np.float32 or logits.shape[0]!=len(table):raise AssertionError('Invalid task logits')
        values=np.empty(len(jobs))
        for begin in range(0,len(table),32):
            zz=logits[begin:begin+32].astype(float)
            if not np.isfinite(zz).all():raise AssertionError('Nonfinite task logits')
            zz-=zz.max(-1,keepdims=True);lp=zz-np.log(np.exp(zz).sum(-1,keepdims=True))
            selected=np.flatnonzero((inputs>=begin)&(inputs<begin+len(lp)))
            values[selected]=lp[inputs[selected]-begin,targets[selected]]
        max_error=max(max_error,float(np.max(np.abs(values-raw['token_log_probabilities'][mode_index]))))
        close(raw['token_log_probabilities'][mode_index],values);logit_elements+=logits.size
        score=[np.zeros(len(e['choices'])) for e in examples];count=[np.zeros(len(e['choices']),dtype=int) for e in examples];nll=np.empty(64)
        for job,value in zip(jobs,values,strict=True):
            if job['kind']=='task':score[job['example']][job['choice']]+=value;count[job['example']][job['choice']]+=1
            else:nll[job['index']]=-value
        rr=result['modes'][mode_index]
        close(rr['external_target_nll'],nll.mean());close(rr['language_per_context_nll'],nll)
        if len(rr['examples'])!=len(examples):raise AssertionError('An answer score is missing')
        correctness={};mean_correctness={};pairs={}
        for item,scores,counts,got in zip(examples,score,count,rr['examples'],strict=True):
            averages=scores/counts;correct=item['correct'];pred=int(np.argmax(scores));pred_mean=int(np.argmax(averages))
            if any(got[k]!=item[k] for k in ['id','task','correct']) or got['prediction']!=pred or got['token_mean_prediction']!=pred_mean:
                raise AssertionError('Incorrect answer-scoring result')
            close(got['scores'],scores);close(got['token_mean_scores'],averages)
            close(got['margin'],scores[correct]-np.max(np.delete(scores,correct)))
            correctness.setdefault(item['task'],[]).append(pred==correct);mean_correctness.setdefault(item['task'],[]).append(pred_mean==correct)
            if 'pair' in item:pairs.setdefault(item['pair'],[]).append(pred==correct)
        summaries.append(dict(mode=mode,accuracy={k:float(np.mean(v)) for k,v in correctness.items()},
            token_mean_accuracy={k:float(np.mean(v)) for k,v in mean_correctness.items()},
            composition_both_correct=float(np.mean([all(v) for v in pairs.values()])) if pairs else None,external_target_nll=float(nll.mean())))
    write_json(output,dict(status='passed',case=case,selection_sha256=sha256(selection),
        logit_elements_independently_reduced=logit_elements,maximum_numpy_log_probability_error=max_error,
        proper_prediction_prefixes=len(table),answer_and_language_predictions=len(jobs),
        maximum_permuted_head_relative_norm_error=norm_error,permuted_head_relative_operator_difference=operator_change.tolist(),
        summaries=summaries,checked_sha256=checked,
        scope='Independent NumPy reduction of every used vocabulary logit and every proper-prefix candidate score, exact prediction-index reconstruction, calibrated-operator and RMS-matching reconstruction, and all task decisions. Native operator replay is qualified separately; task model forwards are not independently rerun here.'))
    print(case['name'],'reasoning verification passed',summaries,flush=True)


if __name__=='__main__':main()
