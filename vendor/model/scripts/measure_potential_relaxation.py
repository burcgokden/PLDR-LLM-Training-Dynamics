#!/usr/bin/env python3
"""Intervene on external gradients and optimizer memory at fixed native states."""
from companion_paths import configured_path
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.schedules import optimizer_and_scheduler
from model_rg.provenance import sha256,write_json,environment
from train_potential_avalanches import observe,FIELDS
from model_rg.deductive_activity import DeductiveActivity,TENSOR_NAMES,STAT_NAMES
ROOT=Path(configured_path('data:model'));STUDY=ROOT/'potential-avalanche-20260913'
REPO=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--device',required=True);a=p.parse_args()
    spec=json.loads((STUDY/'protocols/relaxation.json').read_text())
    matches=[c for c in spec['cases'] if c['name']==a.case]
    if len(matches)!=1:raise ValueError('Unique declared state required')
    case=matches[0]
    for f,h in spec['producer_sources'].items():
        if sha256(REPO/f)!=h:raise ValueError('Changed producer '+f)
    path=Path(case['checkpoint']);saved=torch.load(path,map_location='cpu',weights_only=True)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if case['kind']=='continuation_parent':job=saved['arguments']
    else:job=saved['job']
    out=STUDY/'relaxation'/case['name'];out.mkdir(parents=True,exist_ok=False)
    sel=np.load(STUDY/'selection.npz');probes=torch.as_tensor(sel['probes'][:2],dtype=torch.long,device=a.device)
    coords=torch.as_tensor(sel['coordinates'],dtype=torch.long,device=a.device)
    model=TrainingModel(ROOT/'assets/PLDR-LLM-v51-SOC-110M-1',job['heads'],job['seed'],a.device)
    power_ids={id(x) for n,x in model.model.named_parameters() if n.endswith('pwlst')}
    records={};t0=time.time()
    for branch in ['frozen','zero_gradient','zero_gradient_reset_moments']:
        model.model.load_state_dict(saved['model'])
        if case['kind']=='continuation_parent':opt,_=optimizer_and_scheduler(model,saved['recipe'])
        else:opt=torch.optim.AdamW(model.model.parameters(),lr=job['peak'],betas=(.9,.95),eps=1e-5,weight_decay=.1,foreach=False)
        opt.load_state_dict(copy.deepcopy(saved['optimizer']))
        if branch=='zero_gradient_reset_moments':
            for state in opt.state.values():state['exp_avg'].zero_();state['exp_avg_sq'].zero_()
        state=None;fields=[];logbase=[];power=[];nll=[];prediction_errors=[]
        deductive=DeductiveActivity(model,probes,coords)
        for k in range(spec['steps']+1):
            f,b,p,n,state=observe(model,probes,coords,state)
            fields.append(f);logbase.append(b);power.append(p);nll.append(n)
            deductive.snapshot()
            if k==spec['steps']:break
            if branch=='frozen':continue
            # No corpus block is evaluated for a gradient. Zero is explicit,
            # unlike grad=None, which causes AdamW to skip a parameter.
            expected=[]
            for group in opt.param_groups:
                b1,b2=group['betas'];rate=group['lr'];wd=group['weight_decay'];eps=group['eps']
                for parameter in group['params']:
                    parameter.grad=torch.zeros_like(parameter)
                    if id(parameter) in power_ids:
                        st=opt.state[parameter];step=int(st['step'])+1
                        mm=b1*st['exp_avg'].double();vv=b2*st['exp_avg_sq'].double()
                        prediction=(1-rate*wd)*parameter.detach().double()-rate*(mm/(1-b1**step))/((vv/(1-b2**step)).sqrt()+eps)
                        expected.append((parameter,prediction))
            opt.step()
            prediction_errors.append(max(float((param.double()-pred).abs().max()) for param,pred in expected))
        arrays=dict(fields=np.asarray(fields),logbase=np.asarray(logbase),power=np.asarray(power),probe_nll=np.asarray(nll),
                    predicted_exponent_step_max_error=np.asarray(prediction_errors),**deductive.arrays())
        if not all(np.isfinite(v).all() for v in arrays.values()):raise FloatingPointError('Nonfinite relaxation')
        np.savez_compressed(out/(branch+'.npz'),**arrays)
        records[branch]=dict(artifact_sha256=sha256(out/(branch+'.npz')),
            max_exponent_formula_error=max(prediction_errors) if prediction_errors else 0.,
            summed_logpotential_activity=float(np.sqrt(np.mean(arrays['fields'][1:,0,...,0]**2,axis=(1,2))).sum()))
    write_json(out/'manifest.json',dict(status='complete',case=case,checkpoint_sha256=sha256(path),
        protocol_sha256=sha256(STUDY/'protocols/relaxation.json'),environment=environment(),
        field_names=FIELDS,tensor_names=TENSOR_NAMES,tensor_stat_names=STAT_NAMES,steps_per_intervention=spec['steps'],branches=records,runtime_seconds=time.time()-t0,
        scope='Optimizer intervention with no training-data consumption. A fixed incoming learning rate and native weight decay are retained. This identifies passive optimizer motion, not native loss-driven relaxation or absence of all criticality.'))
    print(case['name'],records,flush=True)
if __name__=='__main__':main()
