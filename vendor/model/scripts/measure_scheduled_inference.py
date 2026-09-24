#!/usr/bin/env python
"""Measure reference deductive-output stability under frozen stochastic generation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel


def nucleus(logits,uniforms,threshold=.8):
    probability=logits.double().softmax(-1)
    order=torch.argsort(probability,dim=-1,descending=True,stable=True)
    sorted_probability=probability.gather(1,order)
    cumulative=sorted_probability.cumsum(-1)
    keep=cumulative-sorted_probability<threshold
    retained=sorted_probability*keep
    retained=retained/retained.sum(-1,keepdim=True)
    cdf=retained.cumsum(-1);cdf[:,-1]=1.
    index=torch.searchsorted(cdf.contiguous(),torch.as_tensor(uniforms,dtype=torch.float64)[:,None].contiguous())
    if (index>=logits.shape[-1]).any():raise AssertionError('A sampling uniform exceeds the normalized nucleus')
    return order.gather(1,index).squeeze(1)


def compare(first,second,reference):
    x=first.astype(np.float64);y=second.astype(np.float64);r=reference.astype(np.float64)
    rmse=float(np.sqrt(np.mean((x-y)**2)));mean=float(r.mean());rms=float(np.sqrt(np.mean(r*r)))
    axes=tuple(range(1,x.ndim))
    return dict(rmse=rmse,reference_mean=mean,reference_rms=rms,
        mean_normalized_rmse=rmse/abs(mean) if mean else None,
        rms_normalized_rmse=rmse/rms if rms else None,
        mean_magnitude_to_rms=abs(mean)/rms if rms else None,
        per_prompt_rmse=np.sqrt(np.mean((x-y)**2,axis=axes)).tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');p.add_argument('--case',required=True)
    p.add_argument('--selection',default='regime-observation-selection.json')
    a=p.parse_args();root=Path(a.root).resolve();study=root/a.study
    selection=study/'protocols'/a.selection;spec=json.loads(selection.read_text())
    cases=[c for c in spec['cases'] if c['name']==a.case and c['inference_stability']]
    if len(cases)!=1:raise AssertionError('A unique inference-stability state must be selected')
    case=cases[0];parent=Path(case['parent_manifest']);meta=json.loads(parent.read_text());checkpoint=Path(case['state'])
    cohort=Path(spec['inference_cohort']);cm_path=Path(spec['inference_cohort_manifest']);cm=json.loads(cm_path.read_text())
    if meta['status']!='complete' or sha256(checkpoint)!=meta['saved_states'][str(case['step'])]['sha256']:
        raise AssertionError('The complete selected native state changed')
    if sha256(cohort)!=spec['inference_cohort_sha256'] or sha256(cohort)!=cm['cohort_sha256']:
        raise AssertionError('The selected inference cohort changed')
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1';probe=root/'controlled-study-20260905/data/short'
    out=study/'measurements'/('inference-'+case['name'].removeprefix('cpu-'));out.mkdir(parents=True,exist_ok=False)
    bind_run(out,[selection,cohort,cm_path,checkpoint,parent,source/'modeling_pldrllm.py',
                  source/'configuration_pldrllm.py',probe/'tokens.npy',probe/'offsets.npy'],vars(a))
    torch.set_num_threads(4);before=time.time()
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    for key in ['heads','seed','recipe','shared_seed','stream_seed']:
        if saved['arguments'][key]!=case[key]:raise AssertionError('Selected inference condition changed: '+key)
    if saved['step']!=case['step']:raise AssertionError('Selected inference horizon changed')
    model=TrainingModel(source,case['heads'],case['seed'],'cpu');model.model.load_state_dict(saved['model']);del saved
    model.model.eval().requires_grad_(False)
    if model.config.eos_token_id!=cm['eos_token_id'] or cm['eos_token_id']!=3:
        raise AssertionError('The tokenizer EOS convention changed')
    with np.load(cohort) as z:rows=z['rows'].copy();uniforms=z['uniforms'].copy()
    pt=np.load(probe/'tokens.npy',mmap_mode='r');po=np.load(probe/'offsets.npy')
    prompts=np.asarray(pt[rows[:,None],po[rows,None]+np.arange(64)])
    if prompts.shape!=(100,64) or uniforms.shape!=(2,100,32):raise AssertionError('The frozen generation program changed')
    generated=np.full((2,100,32),-1,dtype=np.int64);lengths=np.zeros((2,100),dtype=np.int64)
    generation_logits=np.zeros((2,100,32,model.config.vocab_size),dtype=np.float32)
    raw=dict(rows=rows,prompts=prompts,uniforms=uniforms)

    @torch.no_grad()
    def emit(sequence):
        output=model.forward(torch.as_tensor(sequence,dtype=torch.long)[None],capture=True)
        values={name:[] for name in ['A','A_LM','A_P','G_LM']}
        for layer in output.pldr_attentions:
            tensors=[layer[0],layer[1],torch.pow(layer[1],layer[2]),layer[5]]
            for name,tensor in zip(values,tensors,strict=True):
                if not torch.isfinite(tensor).all():raise FloatingPointError('Nonfinite terminal deductive tensor')
                values[name].append(tensor[0].numpy().copy())
        return {name:np.stack(v) for name,v in values.items()}

    with torch.no_grad():
        for run in range(2):
            for begin in range(0,100,4):
                active=np.arange(begin,begin+4)
                sequence=torch.as_tensor(prompts[active].copy(),dtype=torch.long)
                for step in range(32):
                    logits=model.logits(sequence)
                    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite generation logits')
                    generation_logits[run,active,step]=logits.numpy()
                    token=nucleus(logits,uniforms[run,active,step],cm['nucleus_probability'])
                    generated[run,active,step]=token.numpy();lengths[run,active]=step+1
                    sequence=torch.cat([sequence,token[:,None]],dim=1)
                    keep=token.ne(cm['eos_token_id']).numpy()
                    active=active[keep];sequence=sequence[keep]
                    if not len(active):break
            print(case['name'],'generation',run+1,'tokens',int(lengths[run].sum()),flush=True)
        for label,run in [('prompt',None),('run1',0),('run2',1)]:
            emitted={name:[] for name in ['A','A_LM','A_P','G_LM']}
            for index,prompt in enumerate(prompts):
                sequence=prompt if run is None else np.concatenate([prompt,generated[run,index,:lengths[run,index]]])
                for name,value in emit(sequence).items():emitted[name].append(value)
            for name,values in emitted.items():raw[label+'_'+name]=np.stack(values)
    raw.update(generated_tokens=generated,generated_lengths=lengths,generation_logits=generation_logits)
    comparisons={}
    for name in ['A','A_LM','A_P','G_LM']:
        comparisons[name]={label:compare(raw[left+'_'+name],raw[right+'_'+name],raw[reference+'_'+name])
            for label,left,right,reference in [('run1_run2','run1','run2','run1'),
                ('run1_prompt','run1','prompt','prompt'),('run2_prompt','run2','prompt','prompt')]}
    result=dict(schema='scheduled-inference-stability-v1',status='complete',case=case,comparisons=comparisons,
        generated_tokens=[int(x.sum()) for x in lengths],eos_stops=[int(np.sum(generated[r,np.arange(100),lengths[r]-1]==3)) for r in range(2)],
        sampling='Float64 softmax; stable descending probability order with token-index tie order; include the first nucleus boundary crossing; inverse CDF using the frozen scalar uniforms. All used logits and choices are retained.',
        arithmetic='Native CPU float32 model; generation batches of up to4 active prompts; all terminal/prompt tensor emissions use batch1; scalar reductions are float64.',
        scope='The paper mean-normalized RMSE statistic is reconstructed on the declared shorter RefinedWeb generation law. Prompt tensors supply the cached-reference values; no cached text-generation execution is claimed. Near-zero global means can destabilize the normalized ratio, so absolute and RMS-normalized errors are also retained. This input/generation stability statistic alone does not establish a critical surface, a thermodynamic exponent or self-organized criticality.')
    np.savez_compressed(out/'measurements.npz',**raw);write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(schema='scheduled-inference-observation-v1',status='complete',case=case,
        binding_sha256=sha256(out/'binding.json'),raw_sha256=sha256(out/'measurements.npz'),
        results_sha256=sha256(out/'results.json'),seconds=time.time()-before))
    print(case['name'],'inference stability complete',round(time.time()-before,1),flush=True)


if __name__=='__main__':main()
