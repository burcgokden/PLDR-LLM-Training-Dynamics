#!/usr/bin/env python
"""Score proper-prefix task answers under reversible learned-operator interventions."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.inference_interventions import fixed_operators,operator_cache,projected_rows
from model_rg.native import NativeModel
from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');p.add_argument('--case',required=True)
    p.add_argument('--selection',default='reasoning-mechanism-selection.json');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;selection=study/'protocols'/a.selection
    spec=json.loads(selection.read_text());cases=[c for c in spec['cases'] if c['name']==a.case]
    if len(cases)!=1:raise AssertionError('A unique frozen task state is required')
    case=cases[0];cohort=Path(spec['cohort']);cm=Path(spec['cohort_manifest'])
    if sha256(cohort)!=spec['cohort_sha256']:raise AssertionError('The reasoning cohort changed')
    examples=json.loads(cohort.read_text())['examples'];source=Path(case['source'])
    probe=root/'controlled-study-20260905/data/short';inputs=[selection,cohort,cm,
        source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',probe/'tokens.npy',probe/'offsets.npy']
    for path,digest in spec['producer_sources'].items():
        if sha256(Path(__file__).resolve().parents[1]/path)!=digest:raise AssertionError('Frozen task producer changed: '+path)
    if case['kind']=='pretrained':inputs.extend([source/'config.json',source/'model.safetensors'])
    else:
        inputs.extend([Path(case['state']),Path(case['parent_manifest'])]);parent=json.loads(Path(case['parent_manifest']).read_text())
        if parent['status']!='complete':raise AssertionError('The task checkpoint parent is incomplete')
        signature=parent['saved_states'][str(case['step'])]['sha256']
        if sha256(case['state'])!=signature:raise AssertionError('The selected task state changed')
    for path,digest in case.get('input_sha256',{}).items():
        if sha256(path)!=digest:raise AssertionError('A frozen initial task input changed')
    out=study/'measurements'/case['name'];out.mkdir(parents=True,exist_ok=False)
    bind_run(out,inputs,vars(a));torch.set_num_threads(4);before=time.time()
    if case['kind']=='pretrained':model=NativeModel(source,'cpu')
    else:
        saved=torch.load(case['state'],map_location='cpu',mmap=True,weights_only=True)
        if saved['step']!=case['step']:raise AssertionError('The task checkpoint horizon changed')
        for key in ['heads','seed','recipe','shared_seed','stream_seed']:
            if key in case and saved['arguments'][key]!=case[key]:raise AssertionError('Task checkpoint condition changed')
        model=TrainingModel(source,saved['arguments']['heads'],saved['arguments']['seed'],'cpu')
        model.model.load_state_dict(saved['model']);del saved
    model.model.eval().requires_grad_(False)
    pt=np.load(probe/'tokens.npy',mmap_mode='r');po=np.load(probe/'offsets.npy')
    calibration=pt[np.arange(64)[:,None],po[:64,None]+np.arange(64)]
    language_rows=np.arange(512,576);language=pt[language_rows[:,None],po[language_rows,None]+np.arange(65)]
    with torch.no_grad():
        caches=[]
        for begin in range(0,64,8):
            output=model.forward(torch.tensor(calibration[begin:begin+8],dtype=torch.long),capture=True)
            caches.append(operator_cache(output).numpy())
        calibration_cache=np.concatenate(caches,axis=2)
        fixed=torch.from_numpy(calibration_cache.astype(float).mean(2,keepdims=True).astype(np.float32))
        permuted=fixed.clone();rotated=fixed[:,2].roll(1,dims=2)
        target_rms=fixed[:,2].double().square().mean((-2,-1),keepdim=True).sqrt()
        source_rms=rotated.double().square().mean((-2,-1),keepdim=True).sqrt()
        if ((source_rms==0)&(target_rms!=0)).any():raise AssertionError('A head-permutation norm match is undefined')
        factor=torch.where(source_rms>0,target_rms/source_rms.clamp_min(1e-300),torch.zeros_like(source_rms))
        permuted[:,2]=(rotated.double()*factor).float()

    table=[];lookup={};jobs=[]
    def add(sequence,target,**metadata):
        key=tuple(sequence)
        if key not in lookup:lookup[key]=len(table);table.append(list(sequence))
        jobs.append(dict(input_index=lookup[key],target=int(target),**metadata))
    for index,item in enumerate(examples):
        for choice,answer in enumerate(item['answer_tokens']):
            for offset,target in enumerate(answer):
                add(item['prompt_tokens']+answer[:offset],target,kind='task',example=index,choice=choice,offset=offset)
    for index,crop in enumerate(language):add(crop[:64].tolist(),crop[64],kind='language',index=index)
    if max(map(len,table))>64 or min(map(len,table))<1:raise AssertionError('Task prefix length is outside the selection')
    write_json(out/'prediction-index.json',dict(inputs=table,jobs=jobs))
    groups=defaultdict(list)
    for index,sequence in enumerate(table):groups[len(sequence)].append(index)
    job_inputs=np.array([j['input_index'] for j in jobs]);targets=np.array([j['target'] for j in jobs])
    token_logp=[];mode_records=[];sidecars={}
    for mode in spec['modes']:
        mode_start=time.time()
        path=out/('logits-'+mode+'.npy')
        stored=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(table),model.config.vocab_size))
        context=(fixed_operators(model,fixed) if mode=='calibration_G' else
                 fixed_operators(model,permuted) if mode=='permuted_G' else
                 projected_rows(model) if mode=='row_projection' else nullcontext())
        if mode not in ['native','calibration_G','permuted_G','row_projection']:raise ValueError('Unselected task intervention')
        with context,torch.no_grad():
            for length,indices in sorted(groups.items()):
                for begin in range(0,len(indices),32):
                    chosen=indices[begin:begin+32]
                    ids=torch.tensor([table[i] for i in chosen],dtype=torch.long)
                    logits=model.logits(ids)
                    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite proper task logits')
                    stored[chosen]=logits.numpy()
        stored.flush();values=np.empty(len(jobs),dtype=np.float64)
        with torch.no_grad():
            for begin in range(0,len(table),32):
                lp=torch.tensor(np.asarray(stored[begin:begin+32]),dtype=torch.float64).log_softmax(-1)
                selected=np.flatnonzero((job_inputs>=begin)&(job_inputs<begin+len(lp)))
                values[selected]=lp[torch.from_numpy(job_inputs[selected]-begin),torch.from_numpy(targets[selected])].numpy()
        del stored;token_logp.append(values);sidecars[path.name]=sha256(path)
        scores=np.full((len(examples),max(len(e['choices']) for e in examples)),np.nan,dtype=np.float64)
        counts=np.zeros_like(scores)
        for index,item in enumerate(examples):scores[index,:len(item['choices'])]=0
        language_nll=np.empty(64)
        for j,value in zip(jobs,values,strict=True):
            if j['kind']=='task':scores[j['example'],j['choice']]+=value;counts[j['example'],j['choice']]+=1
            else:language_nll[j['index']]=-value
        results=[]
        for index,item in enumerate(examples):
            values=scores[index,:len(item['choices'])];averages=values/counts[index,:len(values)]
            correct=item['correct'];competitors=np.delete(values,correct)
            results.append(dict(id=item['id'],task=item['task'],correct=correct,scores=values.tolist(),
                token_mean_scores=averages.tolist(),prediction=int(np.argmax(values)),token_mean_prediction=int(np.argmax(averages)),
                margin=float(values[correct]-np.max(competitors))))
        mode_records.append(dict(mode=mode,seconds=time.time()-mode_start,examples=results,
            external_target_nll=float(language_nll.mean()),language_per_context_nll=language_nll.tolist()))
        print(case['name'],mode,'seconds',round(time.time()-mode_start,1),'language NLL',float(language_nll.mean()),flush=True)
    np.savez_compressed(out/'measurements.npz',calibration_inputs=calibration,calibration_cache=calibration_cache,
        fixed_cache=fixed.numpy(),permuted_cache=permuted.numpy(),language_rows=language_rows,
        language_crops=language,token_log_probabilities=np.stack(token_logp),job_inputs=job_inputs,targets=targets)
    write_json(out/'results.json',dict(schema='proper-prefix-reasoning-mechanism-v1',status='complete',case=case,
        modes=mode_records,unique_prediction_prefixes=len(table),answer_and_language_token_predictions=len(jobs),
        scope='Fixed finite ARC/composition panel; proper autoregressive candidate scores with no answer token in its own input prefix. Model weights and optimizer states are unchanged. The native external-G branch fixes separately calibrated operators, the permuted control matches each G-head RMS, and row projection recomputes all downstream native operations. All used vocabulary logits are retained. These scores measure declared tasks and do not alone establish broad reasoning or criticality.'))
    write_json(out/'manifest.json',dict(status='complete',case=case,binding_sha256=sha256(out/'binding.json'),
        raw_sha256=sha256(out/'measurements.npz'),results_sha256=sha256(out/'results.json'),
        prediction_index_sha256=sha256(out/'prediction-index.json'),logit_sidecars=sidecars,seconds=time.time()-before))
    print(case['name'],'proper-prefix reasoning complete',round(time.time()-before,1),flush=True)


if __name__=='__main__':main()
