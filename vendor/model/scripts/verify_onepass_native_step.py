#!/usr/bin/env python
"""Reconstruct all conditional Adam steps and the complete risk-drift panel."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel


def tensor_record(value):
    a=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def observations(model,crops):
    pieces=[]
    for begin in range(0,len(crops),32):
        b=crops[begin:begin+32];model.model.eval()
        with torch.no_grad():
            output=model.forward(b[:,:64],capture=True);logits=output.logits[:,-1].double();lp=logits.log_softmax(-1)
            fields=[];centroids=[]
            for layer,entry in enumerate(output.pldr_attentions):
                matrix,operator,attention=entry[0].double(),entry[5].double(),entry[6][:,:,-1].double()
                center=matrix.mean(-2);centroids.append(center)
                row_energy=((matrix-center.unsqueeze(-2))**2).mean((-2,-1));total=(matrix**2).mean((-2,-1))
                fields.append(torch.stack([row_energy,total,row_energy/total.clamp_min(1e-30),
                    (center**2).mean(-1),(operator**2).mean((-2,-1)).sqrt(),
                    -(attention*attention.clamp_min(1e-300).log()).sum(-1),model.head_outputs[layer].double()],-1))
            value=dict(fields=torch.stack(fields,1),centroids=torch.stack(centroids,1),logits=logits,
                hidden_rms=torch.stack([(h[:,-1].double()**2).mean(-1).sqrt() for h in output.hidden_states],1),
                nll=-lp.gather(1,b[:,64,None]).squeeze(1),prediction_entropy=-(lp.exp()*lp).sum(-1))
            pieces.append({k:v.numpy().copy() for k,v in value.items()})
    return {k:np.concatenate([v[k] for v in pieces]) for k in pieces[0]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');p.add_argument('--protocol',required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    if Path(a.output).exists():raise FileExistsError(a.output)
    checked={}
    def check(path,digest=None):
        path=Path(path).resolve();name=str(path)
        if name not in checked:checked[name]=sha256(path)
        if digest is not None and checked[name]!=digest:raise AssertionError('Changed conditional-step evidence: '+name)
    protocol=study/'protocols'/a.protocol;check(protocol);spec=json.loads(protocol.read_text())
    case=next(c for c in spec['cases'] if c['name']==a.case)
    for name,digest in spec['inputs_sha256'].items():check(name,digest)
    for name,digest in spec['producer_sources'].items():check(repo/name,digest)
    folder=study/'measurements'/a.case;check(folder/'manifest.json');meta=json.loads((folder/'manifest.json').read_text())
    if meta['status']!='complete' or meta['case']!=case:raise AssertionError('The selected conditional panel is incomplete')
    for name,key in [('binding.json','binding_sha256'),('results.json','results_sha256'),
            ('measurements.npz','raw_sha256'),('first-step-digests.json','first_step_digests_sha256'),
            ('all-step-digests.json','all_step_digests_sha256')]:check(folder/name,meta[key])
    binding=json.loads((folder/'binding.json').read_text())
    for name,digest in binding['inputs'].items():check(name,digest)
    for name,digest in binding['source_files'].items():check(folder/'source'/name,digest)
    torch.set_num_threads(spec['threads']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=torch.load(case['parent'],map_location='cpu',mmap=True,weights_only=True);profile=parent['recipe']
    if (parent['step'],parent['arguments']['heads'],parent['arguments']['seed'])!=(case['step'],case['heads'],case['seed']):
        raise AssertionError('Wrong incoming state')
    model=TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1',case['heads'],case['seed'],'cpu')
    model.model.load_state_dict(parent['model']);named=dict(model.model.named_parameters())
    generator=[n for n in named if any(part in n for part in ['reslayerAs','plgatt_layer','layernormA'])]
    body=[n for n in named if n not in generator]
    optimizer=torch.optim.AdamW([dict(params=[named[n] for n in generator],lr=profile['generator_peak']),
        dict(params=[named[n] for n in body],lr=profile['body_peak'])],betas=tuple(profile['betas']),
        eps=profile['epsilon'],weight_decay=profile['weight_decay'],foreach=False)
    def rate(t):
        if profile['total_steps']==0:return 1.
        w=float(profile['warmup_steps']);h=float(profile['total_steps']);x=min(float(t),h);alpha=float(profile['floor_fraction'])
        return (1/w)*x if x<=w else (1-alpha)*.5*(1+math.cos(math.pi*((x-w)/(h-w))))+alpha
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    order=np.random.default_rng(parent['arguments']['stream_seed']+1000).permutation(4194304)
    available=order[32*case['step']:];rng=np.random.default_rng(spec['batch_seed'])
    blocks=np.stack([rng.choice(available,size=32,replace=False) for _ in range(spec['batch_replicas'])])
    rows,remainders=np.divmod(blocks,8);offsets=remainders*64
    tokens=np.load(root/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    batches=tokens[rows[...,None],offsets[...,None]+np.arange(65)]
    probe=root/'controlled-study-20260905/data/short';pt=np.load(probe/'tokens.npy');po=np.load(probe/'offsets.npy')
    cohort=np.arange(512,1024);crops=torch.tensor(pt[cohort[:,None],po[cohort,None]+np.arange(65)],dtype=torch.long)
    digests=json.loads((folder/'all-step-digests.json').read_text())
    if len(digests)!=len(batches) or digests[0]!=json.loads((folder/'first-step-digests.json').read_text()):
        raise AssertionError('A complete native step digest was omitted')
    elements=parameter_checks=slot_checks=0;maximum_risk_error=maximum_kl_error=0.;summaries={}
    with np.load(folder/'measurements.npz') as raw:
        for key,value in [('block_ids',blocks),('rows',rows),('offsets',offsets),('training_crops',batches),('cohort',cohort)]:
            if not np.array_equal(raw[key],value):raise AssertionError('Conditional source blocks or cohort changed')
        if any(len(set(map(int,b)))!=32 for b in blocks) or np.isin(blocks,order[:32*case['step']]).any():
            raise AssertionError('A conditional branch reuses a consumed source position')
        def emission(prefix,index=None):
            nonlocal elements
            actual=observations(model,crops)
            for name,value in actual.items():
                expected=raw[prefix+'_'+name]
                if index is not None:expected=expected[index]
                if value.shape!=expected.shape or value.dtype!=expected.dtype or value.tobytes()!=expected.tobytes():
                    raise AssertionError('Native emission replay differs: '+prefix+' '+name)
                elements+=value.size
        emission('base')
        for k,values in enumerate(batches):
            model.model.load_state_dict(parent['model']);optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
            scheduler.load_state_dict(copy.deepcopy(parent['scheduler']));model.model.train();optimizer.zero_grad(set_to_none=True)
            batch=torch.tensor(values,dtype=torch.long)
            if profile['objective']=='last_external_target':
                loss=torch.nn.functional.cross_entropy(model.logits(batch[:,:64]),batch[:,64]);count=32
            else:
                model.eta=None;model.capture=False;model.head_outputs={}
                logits=model.model(batch[:,:64],use_cache=False,logits_to_keep=0).logits
                mask=batch[:,1:]!=0
                losses=torch.nn.functional.cross_entropy(logits.transpose(1,2),batch[:,1:],reduction='none')
                loss=(losses*mask.to(losses.dtype)).sum()/mask.sum();count=int(mask.sum())
            loss.backward()
            norms=[torch.linalg.vector_norm(torch.stack([v.grad.norm() for v in g['params'] if v.grad is not None])) for g in optimizer.param_groups]
            norm=torch.linalg.vector_norm(torch.stack(norms))
            if float(loss.detach())!=float(raw['losses'][k]) or float(norm)!=float(raw['gradient_norms'][k]) or count!=raw['supervised_targets'][k]:
                raise AssertionError('Native objective, mask or gradient norm differs')
            if profile['clipping']=='value':torch.nn.utils.clip_grad_value_(model.model.parameters(),1.,foreach=False)
            else:torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();expected=digests[k]
            if set(named)!=set(expected['parameters']):raise AssertionError('Parameter coverage changed')
            for name,value in named.items():
                if tensor_record(value)!=expected['parameters'][name]:raise AssertionError('Native parameter step differs: '+name)
                parameter_checks+=1
                if set(optimizer.state[value])!=set(expected['optimizer_states'][name]):raise AssertionError('Adam slot coverage changed')
                for key,item in optimizer.state[value].items():
                    got=tensor_record(item) if isinstance(item,torch.Tensor) else item
                    if got!=expected['optimizer_states'][name][key]:raise AssertionError('Native Adam slot differs')
                    slot_checks+=1
            groups=[{key:value for key,value in g.items() if key!='params'} for g in optimizer.param_groups]
            if json.loads(json.dumps(groups))!=expected['optimizer_groups'] or scheduler.state_dict()!=expected['scheduler']:
                raise AssertionError('Native optimizer or scheduler clock differs')
            if k in {0,len(batches)-1}:
                emission('full',k);updated={n:named[n].detach().clone() for n in generator}
                with torch.no_grad():
                    for n in generator:named[n].copy_(parent['model'][n])
                emission('body',k)
                with torch.no_grad():
                    for n in body:named[n].copy_(parent['model'][n])
                    for n in generator:named[n].copy_(updated[n])
                emission('generator',k)
            print(case['name'],'independently reconstructed native step',k+1,flush=True)
        # Reconstruct all target losses and KLs, including the un-replayed
        # intermediate emissions, directly from retained native vocabulary logits.
        z=raw['base_logits'].astype(np.longdouble);z-=z.max(1,keepdims=True)
        lp=z-np.log(np.exp(z).sum(1,keepdims=True));prob=np.exp(lp);targets=crops[:,64].numpy()
        base_loss=-lp[np.arange(512),targets].astype(float)
        for branch in ['generator','body','full']:
            values=raw[branch+'_logits'];recorded_loss=raw[branch+'_nll'];recorded_kl=raw[branch+'_predictive_kl']
            risk_changes=[];kl_means=[]
            for k,current in enumerate(values):
                q=current.astype(np.longdouble);q-=q.max(1,keepdims=True);lq=q-np.log(np.exp(q).sum(1,keepdims=True))
                loss=-lq[np.arange(512),targets].astype(float);kl=np.sum(prob*(lp-lq),axis=1).astype(float)
                maximum_risk_error=max(maximum_risk_error,float(np.max(np.abs(loss-recorded_loss[k]))))
                maximum_kl_error=max(maximum_kl_error,float(np.max(np.abs(kl-recorded_kl[k]))))
                if not np.allclose(loss,recorded_loss[k],rtol=2e-12,atol=2e-12) or not np.allclose(kl,recorded_kl[k],rtol=2e-11,atol=2e-12):
                    raise AssertionError('Independent conditional risk or KL reconstruction differs')
                risk_changes.append(float(np.mean(loss-base_loss)));kl_means.append(float(np.mean(kl)))
            x=np.array(risk_changes);summaries[branch]=dict(risk_changes=risk_changes,mean_risk_change=float(x.mean()),
                conditional_standard_error=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None,
                mean_predictive_kl=float(np.mean(kl_means)),predictive_kl_means=kl_means)
        cross=np.array(summaries['full']['risk_changes'])-np.array(summaries['generator']['risk_changes'])-np.array(summaries['body']['risk_changes'])
        summaries['interaction']=dict(risk_changes=cross.tolist(),mean_risk_change=float(cross.mean()),
            conditional_standard_error=float(cross.std(ddof=1)/np.sqrt(len(cross))) if len(cross)>1 else None)
    reported=json.loads((folder/'results.json').read_text())
    if set(reported['risk_drift'])!=set(summaries):raise AssertionError('A conditional-risk corner was omitted')
    for branch,value in summaries.items():
        observed=reported['risk_drift'][branch]
        if set(observed)!=set(value):raise AssertionError('A conditional-risk statistic was omitted')
        for name,want in value.items():
            if want is None:
                if observed[name] is not None:raise AssertionError('An undefined conditional standard error changed')
            elif not np.allclose(observed[name],want,rtol=2e-10,atol=2e-12):
                raise AssertionError('An independently reconstructed conditional-risk mean or error differs')
    write_json(a.output,dict(status='passed',case=case,conditional_native_steps=len(batches),
        parameter_tensors_checked=parameter_checks,optimizer_slots_checked=slot_checks,
        native_emission_elements_replayed_bytewise=elements,full_emission_replay_draws=sorted({0,len(batches)-1}),
        contexts=512,remaining_blocks=len(available),summaries=summaries,
        maximum_extended_precision_risk_difference=maximum_risk_error,maximum_extended_precision_kl_difference=maximum_kl_error,
        checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='Every conditional native CPU AdamW parameter, moment and scheduler update is reconstructed bytewise. '
        'The baseline and all three corners of the first and last draws have complete bytewise emission replay. '
        'Every retained target loss and predictive KL has an independent extended-precision reconstruction. '
        'Conditional minibatch standard errors do not count as uncertainty over training initializations or a GPU-step equivalence claim.'))
    print('Complete single-pass conditional-step reconstruction passed',case['name'],flush=True)


if __name__=='__main__':main()
