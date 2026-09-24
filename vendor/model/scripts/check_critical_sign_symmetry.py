#!/usr/bin/env python3
"""Check the native head-sign symmetry in paired CPU training executions.

The query projection and the output PLGA coupling/bias change sign on the
same head. The query Gram, metric learner, attention and emissions are
invariant. First optimizer moments co-transform, while second moments do
not. These are qualification replays of short existing source prefixes,
not additional scientific trajectories or CPU/GPU equivalence claims.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.provenance import sha256,write_json
import run_critical_onepass as native

REPO=native.REPO
SELF='scripts/check_critical_sign_symmetry.py'


def sign_masks(model,pattern):
    masks={};head_signs=[]
    modules=[(name,module) for name,module in model.model.named_modules()
             if hasattr(module,'wq') and hasattr(module,'plgatt_layer')]
    if len(modules)!=5:raise ValueError('Expected five native multihead blocks')
    for layer,(name,module) in enumerate(modules):
        n=module.num_heads;sign=torch.ones(n,dtype=torch.float32,device=model.device)
        if pattern=='single_first':
            if layer==0:sign[0]=-1
        elif pattern=='checkerboard':
            for head in range(n):
                if (layer+head)%2==0:sign[head]=-1
        else:raise ValueError('Unregistered sign pattern')
        head_signs.append(sign.cpu().numpy())
        rows=sign.repeat_interleave(module.depth)
        masks[name+'.wq.weight']=rows[:,None]
        if module.wq.bias is not None:masks[name+'.wq.bias']=rows
        for suffix in ['alst','balst']:masks[name+'.plgatt_layer.'+suffix]=sign[:,None,None]
    params=dict(model.model.named_parameters())
    if not set(masks)<=set(params):raise ValueError('A native gauge coordinate is missing')
    return masks,np.stack(head_signs)


@torch.no_grad()
def transform(model,masks):
    for name,param in model.model.named_parameters():
        if name in masks:param.mul_(masks[name])


def tensor_error(a,b):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Symmetry changed shape or dtype')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Nonfinite symmetry comparison')
    error=float(((a.double()-b.double()).abs()/(1+a.double().abs())).max()) if a.numel() else 0.
    exact=a.detach().contiguous().numpy().tobytes()==b.detach().contiguous().numpy().tobytes()
    return error,exact


def compare_saved(base,changed,masks,optimizer_names):
    maximum=0.;exact=True;count=0
    for name,value in base['model'].items():
        expected=value*masks[name] if name in masks else value
        error,equal=tensor_error(expected,changed['model'][name]);maximum=max(maximum,error);exact &= equal;count+=1
    if base['optimizer']['param_groups']!=changed['optimizer']['param_groups']:raise ValueError('Optimizer groups changed')
    if base['optimizer']['state'].keys()!=changed['optimizer']['state'].keys():raise ValueError('Optimizer participation changed')
    for index,state in base['optimizer']['state'].items():
        other=changed['optimizer']['state'][index];name=optimizer_names[index]
        if state.keys()!=other.keys():raise ValueError('Optimizer state fields changed')
        for key,value in state.items():
            expected=value*masks[name] if key=='exp_avg' and name in masks else value
            error,equal=tensor_error(expected,other[key]);maximum=max(maximum,error);exact &= equal;count+=1
    if not torch.equal(base['rng'],changed['rng']):raise ValueError('Gauge transformation changed the random state')
    return dict(maximum_normalized_state_error=maximum,all_compared_state_tensor_bytes_equal=bool(exact),compared_state_tensors=count)


def execute(parent,output):
    if output.exists() or output==native.ROOT or not output.is_relative_to(native.ROOT):raise ValueError('A fresh authorized experiment directory is required')
    p=native.admit(parent);selected=np.load(parent/'selection.npz');steps=8
    cases=[dict(heads=n,seed=seed,pattern=pattern,control=1.,environment=0,steps=steps)
           for n in [2,4] for seed in [915001,915003,915005] for pattern in ['single_first','checkerboard']]
    for case in cases:
        if not any(all(j[key]==case[key] for key in ['heads','seed','control','environment']) for j in p['jobs']):raise ValueError('Missing recorded native parent')
    sources={**p['source_sha256'],SELF:sha256(__file__)}
    inputs={str(parent/'protocol.json'):sha256(parent/'protocol.json'),str(parent/'selection.npz'):sha256(parent/'selection.npz'),**p['input_sha256']}
    output.mkdir(parents=True)
    write_json(output/'protocol.json',dict(schema='critical-head-sign-check-v1',role='qualification',study=str(parent),cases=cases,
        source_sha256=sources,input_sha256=inputs,scientific_updates=0,replay_updates=2*steps*len(cases),
        device='cpu',dtype='float32',threads=2,source_blocks='The first eight recorded batches of distinct blocks, paired within each gauge comparison.',
        tolerance=2e-6,scope='Native CPU qualification of a mathematical sign action; no CPU/GPU equivalence or independent scientific replication is claimed.'))
    for name,digest in sources.items():
        target=output/'executed-source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((REPO/name).read_bytes())
        if sha256(target)!=digest:raise ValueError('Producer changed while freezing it')
    ph=sha256(output/'protocol.json');checked={str(output/'protocol.json'):ph};records=[]
    torch.set_num_threads(2)
    tokens=np.load(native.CORPUS/'tokens.npy',mmap_mode='r');blocks=selected['blocks_0'][:steps]
    if len(np.unique(blocks))!=blocks.size:raise ValueError('Source prefix repeats a block')
    for case in cases:
        started=time.monotonic();name=f'h{case["heads"]}-s{case["seed"]}-{case["pattern"]}'
        destination=output/'runs'/name;destination.mkdir(parents=True)
        models=[];optimizers=[]
        for _ in range(2):
            model=native.TrainingModel(native.NATIVE,case['heads'],case['seed'],'cpu')
            native.normalize_variance_initialization(model);native.fix_shared_generator(model,p['design']['shared_seed'])
            models.append(model);optimizers.append(native.optimizer_for(model,case['control']))
        masks,head_signs=sign_masks(models[1],case['pattern']);transform(models[1],masks)
        norms=[];losses=[];trace=[];max_gradient_error=0.;gradient_bytes=True
        for step,b in enumerate(blocks,1):
            ids=tokens[b[:,None]//8,64*(b[:,None]%8)+np.arange(65)]
            batch=torch.as_tensor(ids,dtype=torch.long);observed=[];step_rng=torch.get_rng_state()
            for model,opt in zip(models,optimizers):
                torch.set_rng_state(step_rng)
                model.model.train();opt.zero_grad(set_to_none=True)
                out=model.forward(batch[:,:64],capture=True);z=out.logits[:,-1]
                loss=torch.nn.functional.cross_entropy(z,batch[:,64]);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,foreach=False,error_if_nonfinite=True)
                observed.append(dict(logits=z.detach().clone(),loss=float(loss.detach()),norm=float(norm),
                    metric=[v[0].detach().clone() for v in out.pldr_attentions],operator=[v[5].detach().clone() for v in out.pldr_attentions],
                    attention=[v[6].detach().clone() for v in out.pldr_attentions],rng=torch.get_rng_state()))
                opt.step();del out,z,loss
            if not torch.equal(observed[0]['rng'],observed[1]['rng']):raise ValueError('Paired branches consumed different random draws')
            branch_rng=[x['rng'].clone() for x in observed]
            errors={'logits':tensor_error(observed[0]['logits'],observed[1]['logits'])[0]}
            for field in ['metric','operator','attention']:
                errors[field]=max(tensor_error(value*(torch.tensor(head_signs[layer])[None,:,None,None] if field=='operator' else 1),observed[1][field][layer])[0]
                    for layer,value in enumerate(observed[0][field]))
            left=dict(models[0].model.named_parameters());right=dict(models[1].model.named_parameters())
            for key,value in left.items():
                if value.grad is None or right[key].grad is None:raise ValueError('A native coordinate has no gradient')
                expected=value.grad*masks[key] if key in masks else value.grad
                error,equal=tensor_error(expected,right[key].grad);max_gradient_error=max(max_gradient_error,error);gradient_bytes &= equal
            norms.append([x['norm'] for x in observed]);losses.append([x['loss'] for x in observed])
            trace.append(dict(step=step,**errors));del observed
        names={id(param):name for name,param in models[0].model.named_parameters()}
        state_dict=optimizers[0].state_dict()
        opt_names={index:names[id(param)] for group,saved in zip(optimizers[0].param_groups,state_dict['param_groups']) for param,index in zip(group['params'],saved['params'])}
        for label,model,opt,rng in zip(['native','transformed'],models,optimizers,branch_rng):
            torch.save(dict(model=model.model.state_dict(),optimizer=opt.state_dict(),rng=rng),destination/(label+'.pt'))
        states=[torch.load(destination/(label+'.pt'),map_location='cpu',weights_only=False) for label in ['native','transformed']]
        comparison=compare_saved(*states,masks,opt_names)
        maximum=max([comparison['maximum_normalized_state_error'],max_gradient_error,*[value for row in trace for key,value in row.items() if key!='step']])
        passed=maximum<=2e-6 and np.max(np.abs(np.array(losses)[:,0]-np.array(losses)[:,1]))<=2e-6
        result=dict(case=case,status='passed' if passed else 'failed',role='qualification',scientific_updates=0,replay_updates=2*steps,
            protocol_sha256=ph,source_blocks=blocks.tolist(),trace=trace,losses=losses,gradient_norms=norms,
            maximum_normalized_gradient_error=max_gradient_error,all_compared_gradient_tensor_bytes_equal=bool(gradient_bytes),
            head_signs=head_signs.tolist(),optimizer_parameter_names={str(k):v for k,v in opt_names.items()},
            runtime_seconds=time.monotonic()-started,artifacts={label+'.pt':sha256(destination/(label+'.pt')) for label in ['native','transformed']},**comparison)
        write_json(destination/'manifest.json',result)
        checked[str(destination/'manifest.json')]=sha256(destination/'manifest.json')
        for key,digest in result['artifacts'].items():checked[str(destination/key)]=digest
        records.append(result);print(name,result['status'],'maximum',maximum,flush=True)
        del models,optimizers,states,left,right,state_dict,masks
    for name,digest in sources.items():
        if sha256(REPO/name)!=digest:raise ValueError('Executing producer changed')
        checked[str(REPO/name)]=digest
    for name,digest in inputs.items():
        if sha256(name)!=digest:raise ValueError('Qualification input changed')
        checked[name]=digest
    write_json(output/'analysis.json',dict(schema='critical-head-sign-analysis-v1',status='complete',study=str(output),parent_study=str(parent),
        role='qualification',scientific_updates=0,replay_updates=sum(r['replay_updates'] for r in records),reconstructed_paths=len(records),records=records,
        all_pairs_passed=all(r['status']=='passed' for r in records),checked_sha256=checked,source_sha256={SELF:sha256(__file__)},
        scope='Paired native CPU source-prefix executions test the head-sign action, including full gradients and saved model/Adam states. They add no scientific trajectory.'))
    print('Head-sign qualification complete',all(r['status']=='passed' for r in records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();execute(a.parent.resolve(),a.output.resolve())
