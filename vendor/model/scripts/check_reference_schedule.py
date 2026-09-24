#!/usr/bin/env python
"""Qualify the scalar drive against preserved public code and native CPU updates."""
import argparse
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.criticality import fix_shared_generator, parameter_digest, sample_batches
from model_rg.provenance import sha256, write_json
from model_rg.schedules import clip, loss, multiplier, optimizer_and_scheduler, recipe
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization


def same(left, right):
    if isinstance(left, torch.Tensor):
        return (left.dtype == right.dtype and left.shape == right.shape and
                left.detach().cpu().contiguous().numpy().tobytes() ==
                right.detach().cpu().contiguous().numpy().tobytes())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(x,y) for x,y in zip(left,right))
    return left == right


def public_functions(path):
    tree=ast.parse(path.read_text())
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and
               node.name in ['LinearWarmupCosineLRSchedule','masked_loss_function']]
    if len(functions)!=2:raise AssertionError('Public reference entrypoints changed')
    namespace=dict(torch=torch,nn=torch.nn,math=math,LambdaLR=torch.optim.lr_scheduler.LambdaLR)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.root);repo=Path(__file__).resolve().parents[1]
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    reference=repo/'internal/reference-schedule/pldr_run_model_v510.py'
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    inputs=[reference,repo/'internal/reference-schedule/manifest.json',
            source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',root/'data/refinedweb-4608/tokens.npy']
    bind_run(out,inputs,vars(a));torch.set_num_threads(4)
    public=public_functions(reference);raw={};checks={}
    for name in ['controlled','reference1','reference2']:
        profile=recipe(name,4);parameter=torch.nn.Parameter(torch.ones(1))
        optimizer=torch.optim.AdamW([parameter],lr=profile['generator_peak'])
        scheduler=public['LinearWarmupCosineLRSchedule'](optimizer,profile['total_steps'],
                    profile['warmup_steps'],profile['floor_fraction'])
        expected=np.array([scheduler.lr_lambdas[0](k) for k in range(250002)])
        actual=np.array([multiplier(k,profile['warmup_steps'],250000,.1) for k in range(250002)])
        if not same(torch.from_numpy(expected),torch.from_numpy(actual)):
            raise AssertionError('Scalar schedule differs from preserved public implementation')
        raw[name+'_curve']=actual
        checks[name]=dict(phase_values=len(actual),byte_equal=True,initial_rate=optimizer.param_groups[0]['lr'],
            applied_multiplier_sum=float(actual[:250000].sum()),
            applied_squared_multiplier_sum=float(np.square(actual[:250000]).sum()),
            warmup=profile['warmup_steps'],final_applied=float(actual[249999]),next_after_horizon=float(actual[250000]))
    # Masked loss and its gradient are checked against the extracted public function.
    generator=torch.Generator().manual_seed(650601)
    logits=torch.randn(32,64,127,generator=generator,requires_grad=True)
    crops=torch.randint(1,127,(32,65),generator=generator);crops[::3,1::7]=0
    adapted=SimpleNamespace(model=lambda *args,**kwargs:SimpleNamespace(logits=logits))
    got=loss(adapted,crops,recipe('reference1',4))
    expected=public['masked_loss_function'](crops[:,1:],logits)
    grad_got=torch.autograd.grad(got,logits,retain_graph=True)[0]
    grad_expected=torch.autograd.grad(expected,logits)[0]
    if not same(got,expected) or not same(grad_got,grad_expected):
        raise AssertionError('Public masked loss or its gradient differs')
    checks['masked_loss']=dict(value=float(got.detach()),byte_equal=True,gradient_elements=logits.numel(),
                               nonpadding_targets=int(crops[:,1:].ne(0).sum()))
    del logits,grad_got,grad_expected,got,expected
    # Native full-batch float32 updates: grouped bookkeeping versus one uniform
    # public-rate group, with a complete grouped checkpoint restart after step3.
    profile=recipe('reference1',4)
    left=TrainingModel(source,4,650611,'cpu');normalize_variance_initialization(left)
    fix_shared_generator(left,640011)
    right=TrainingModel(source,4,650611,'cpu');right.model.load_state_dict(left.model.state_dict())
    opt,scheduler=optimizer_and_scheduler(left,profile)
    reference_opt=torch.optim.AdamW(right.model.parameters(),lr=.0012,betas=(.9,.95),eps=1e-5,
                                   weight_decay=.1,foreach=False)
    reference_scheduler=public['LinearWarmupCosineLRSchedule'](reference_opt,250000,2000,.1)
    _,_,batches=sample_batches(np.load(root/'data/refinedweb-4608/tokens.npy'),650621,6)
    emissions=[];initial=parameter_digest(left)
    for step in range(6):
        if step==3:
            saved=dict(model=copy.deepcopy(left.model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),
                       scheduler=copy.deepcopy(scheduler.state_dict()),step=step)
            torch.save(saved,out/'native-resume.pt')
            del left,opt,scheduler,saved
            saved=torch.load(out/'native-resume.pt',weights_only=True)
            left=TrainingModel(source,4,650611,'cpu');left.model.load_state_dict(saved['model'])
            opt,scheduler=optimizer_and_scheduler(left,profile)
            opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
            del saved
        batch=torch.tensor(batches[step],dtype=torch.long)
        rates=[g['lr'] for g in opt.param_groups]
        expected_rate=reference_opt.param_groups[0]['lr']
        if rates != [expected_rate,expected_rate] or scheduler.last_epoch != step:
            raise AssertionError('Applied scheduler phase differs after resume')
        opt.zero_grad(set_to_none=True);reference_opt.zero_grad(set_to_none=True)
        left.model.train();right.model.train()
        value=loss(left,batch,profile);value.backward()
        right.eta=None;right.capture=False;right.head_outputs={}
        z=right.model(batch[:,:-1],use_cache=False,logits_to_keep=0).logits
        reference_value=public['masked_loss_function'](batch[:,1:],z);reference_value.backward()
        if not same(value,reference_value):raise AssertionError('Native objectives differ')
        clip(left,profile);torch.nn.utils.clip_grad_value_(right.model.parameters(),1.,foreach=False)
        opt.step();reference_opt.step();scheduler.step();reference_scheduler.step()
        left_named=dict(left.model.named_parameters());right_named=dict(right.model.named_parameters())
        for name,parameter in left_named.items():
            other=right_named[name]
            if not same(parameter,other) or not same(opt.state[parameter],reference_opt.state[other]):
                raise AssertionError('Native parameter or Adam state differs: '+name)
        emissions.append([step+1,rates[0],float(value.detach())])
        print('native reference qualification',step+1,'loss',float(value.detach()),flush=True)
        if step==0 and parameter_digest(left)!=initial:raise AssertionError('The initial zero-rate update moved weights')
        del value,reference_value,z
    raw['native_step_rate_loss']=np.array(emissions)
    checks['native_uniform_group_and_resume']=dict(native_updates_per_path=6,paths=2,heads=4,batch=32,
        inputs=64,objective='all_nonpadding_targets',arithmetic='CPU float32',
        parameter_tensors=len(left_named),parameters=sum(p.numel() for p in left_named.values()),
        parameter_and_adam_bytes_equal_at_every_step=True,resume_after=3,
        initial_zero_rate_preserves_weights=True,final_scheduler_phase=scheduler.last_epoch,
        scope='Qualification of the float32 controlled adaptation; not a BF16/FSDP pretraining replication.')
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'results.json',dict(status='complete',schema='reference-schedule-qualification-v1',checks=checks))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        raw_sha256=sha256(out/'measurements.npz'),results_sha256=sha256(out/'results.json'),
        checkpoint_sha256=sha256(out/'native-resume.pt')))
    print('Reference schedule, masked loss, native optimizer grouping and resume qualified',flush=True)


if __name__=='__main__':main()
