#!/usr/bin/env python
"""Test finite return after an early shared-learner transplant into a late model."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import observations, optimizer_for, sample_batches, generator_parameter
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--case',required=True)
    parser.add_argument('--device',required=True);parser.add_argument('--protocol',default='collective-return.json');args=parser.parse_args()
    root=Path(args.root);study=root/'criticality-dynamics-20260906';protocol=study/'protocols'/args.protocol
    spec=json.loads(protocol.read_text());case=next(c for c in spec['cases'] if c['name']==args.case)
    measurement_protocol=study/'protocols/collective-return-emission.json'
    out=study/'runs'/case['name'];out.mkdir(parents=True,exist_ok=False)
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1';data=root/'data/refinedweb-4608';probe=root/'controlled-study-20260905/data/short'
    bind_run(out,[protocol,measurement_protocol,Path(case['parent']),Path(case['donor']),data/'tokens.npy',probe/'tokens.npy',probe/'offsets.npy',
                  source/'modeling_pldrllm.py',source/'configuration_pldrllm.py'],vars(args))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=torch.load(case['parent'],map_location='cpu',weights_only=True)
    donor=torch.load(case['donor'],map_location='cpu',weights_only=True)
    a=parent['arguments'];start=parent['step'];end=start+spec['update_count']
    for key in ['heads','seed','multiplier','shared_seed','stream_seed','normalization']:
        if a[key]!=donor['arguments'][key]:raise AssertionError('Donor trajectory changed')
    if start!=8192 or donor['step']!=2048:raise AssertionError('Checkpoint horizons changed')
    rows,offsets,batches=sample_batches(np.load(data/'tokens.npy'),a['stream_seed'],end)
    ptokens=np.load(probe/'tokens.npy');poffsets=np.load(probe/'offsets.npy');cohort=np.arange(*spec['rows'])
    crops=ptokens[cohort[:,None],poffsets[cohort,None]+np.arange(65)]
    raw=dict(cohort=cohort,rows=rows,offsets=offsets,probe_steps=np.arange(start,end+1,spec['probe_every']))
    records=[];started=time.time()
    for branch in spec['branches']:
        model=TrainingModel(source,a['heads'],a['seed'],args.device)
        model.model.load_state_dict(parent['model']);optimizer=optimizer_for(model,a['multiplier'])
        optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        named=list(model.model.named_parameters());names=[n for n,p in named]
        ordered=[n for n in names if generator_parameter(n)]+[n for n in names if not generator_parameter(n)]
        donor_ids=[i for group in donor['optimizer']['param_groups'] for i in group['params']]
        donor_states={n:donor['optimizer']['state'][i] for n,i in zip(ordered,donor_ids,strict=True)}
        replacement={n:donor['model'][n].to(args.device) for n in names if 'reslayerAs' in n}
        with torch.no_grad():
            if branch!='base':
                for name,parameter in named:
                    if name not in replacement:continue
                    parameter.copy_(replacement[name])
                    if branch=='early_shared_weights_moments':
                        for key in ['exp_avg','exp_avg_sq']:
                            optimizer.state[parameter][key].copy_(donor_states[name][key])
        fs=[];hs=[];losses=[];norms=[];status='complete';error=None
        def evaluate(step):
            model.model.eval();fields=[];heads=[];logits=[]
            with torch.no_grad():
                for i in range(0,len(cohort),32):
                    batch=torch.tensor(crops[i:i+32],dtype=torch.long,device=args.device)
                    f,h,z,_=observations(model,batch[:,:64],batch[:,64])
                    if not all(torch.isfinite(value).all() for value in [f,h,z]):raise FloatingPointError('Nonfinite observation')
                    fields.append(f.cpu().numpy());heads.append(h.cpu().numpy())
                    if step in [start,end]:logits.append(z.detach().cpu().numpy())
            fs.append(np.concatenate(fields));hs.append(np.concatenate(heads))
            if logits:raw[branch+('_initial_logits' if step==start else '_final_logits')]=np.concatenate(logits)
        try:
            evaluate(start)
            for step in range(start,end):
                model.model.train();optimizer.zero_grad(set_to_none=True)
                batch=torch.tensor(batches[step],dtype=torch.long,device=args.device)
                loss=torch.nn.functional.cross_entropy(model.logits(batch[:,:64]),batch[:,64])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                if branch=='frozen_early_shared_weights':
                    with torch.no_grad():
                        for name,parameter in named:
                            if name in replacement:parameter.copy_(replacement[name])
                losses.append(float(loss.detach()));norms.append(float(norm))
                if (step+1-start)%spec['probe_every']==0:
                    evaluate(step+1);print(case['name'],branch,step+1,'R',float(hs[-1][...,2].mean()),'seconds',round(time.time()-started,1),flush=True)
        except (FloatingPointError,RuntimeError) as exc:
            if 'nonfinite' not in str(exc).lower().replace('-',''):raise
            status='numerical_failure';error=str(exc)
        raw.update({branch+'_fields':np.array(fs),branch+'_heads':np.array(hs),branch+'_losses':np.array(losses),branch+'_gradient_norms':np.array(norms)})
        record=dict(branch=branch,status=status,error=error,updates=len(losses))
        if status=='complete':
            checkpoint=dict(model={n:p.detach().cpu().clone() for n,p in model.model.state_dict().items()},optimizer=copy.deepcopy(optimizer.state_dict()),
                            step=end,condition=a,intervention=branch,parent=case['parent'],donor=case['donor'])
            filename=branch+'-training-state.pt';torch.save(checkpoint,out/filename)
            record.update(checkpoint=filename,checkpoint_sha256=sha256(out/filename));del checkpoint
        records.append(record);del model,optimizer,replacement,donor_states,named;gc.collect();torch.cuda.empty_cache()
    for key,array in raw.items():
        if not np.isfinite(array).all():raise AssertionError('Nonfinite recorded array: '+key)
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(schema='learned-collective-return-v2',status='complete' if all(r['status']=='complete' for r in records) else 'complete_with_numerical_failures',
        case=case,condition=a,arguments=vars(args),start_step=start,end_step=end,branches=records,seconds=time.time()-started,
        raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json'),peak_cuda_gb=torch.cuda.max_memory_allocated(args.device)/2**30 if args.device.startswith('cuda') else None))


if __name__=='__main__':main()
