#!/usr/bin/env python
"""Check a full-graph augmented Adam tangent against decreasing physical pulses."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.adam_tangent import AdamState,AdamDirection,adam_update,adam_update_tangent,squared_norm
from model_rg.controlled import bind_run
from model_rg.criticality import generator_parameter,predictive_kl,sample_batches,optimizer_for
from model_rg.precision import preserve_response_dtype,expand_smooth_response
from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--direction',default='gain-all')
    parser.add_argument('--amplitudes',default='0.01,0.001,0.0001,0.00001,0.000001')
    parser.add_argument('--contexts',type=int,default=4)
    args=parser.parse_args()
    root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(args.checkpoint);source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    training=root/'data/refinedweb-4608';probe=root/'controlled-study-20260905/data/short'
    bind_run(out,[checkpoint,source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',
                  training/'tokens.npy',probe/'tokens.npy',probe/'offsets.npy'],vars(args))
    torch.set_num_threads(4)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    condition=saved['arguments'];step=saved['step']
    model=TrainingModel(source,condition['heads'],condition['seed'],'cpu')
    model.model.load_state_dict(saved['model']);model.model.eval()
    ptokens=np.load(probe/'tokens.npy');offsets=np.load(probe/'offsets.npy');rows=np.arange(512,512+args.contexts)
    ids=torch.tensor(ptokens[rows[:,None],offsets[rows,None]+np.arange(64)],dtype=torch.long)
    _,_,batches=sample_batches(np.load(training/'tokens.npy'),condition['stream_seed'],step+1)
    batch=torch.tensor(batches[step],dtype=torch.long)
    with torch.no_grad():native_logits=model.logits(ids).double().numpy()
    model.model.double();preserve_response_dtype(model);expand_smooth_response(model)
    model.capture=False;model.eta=None
    named=list(model.model.named_parameters());names=[n for n,p in named]
    oldids=[i for g in saved['optimizer']['param_groups'] for i in g['params']]
    ordered=[n for n in names if generator_parameter(n)]+[n for n in names if not generator_parameter(n)]
    states={n:saved['optimizer']['state'][i] for n,i in zip(ordered,oldids,strict=True)}
    rates=tuple(3e-4*(condition['multiplier'] if generator_parameter(n) else 2/condition['heads']) for n in names)
    weights=tuple(p.detach().clone() for n,p in named)
    state=AdamState(weights,tuple(states[n]['exp_avg'].double() for n in names),
                    tuple(states[n]['exp_avg_sq'].double() for n in names),step)
    del saved,states;gc.collect()
    change=[]
    for name,p in zip(names,weights,strict=True):
        active='plgatt_layer' in name and name.rsplit('.',1)[-1] in ['alst','balst']
        if args.direction.startswith('gain-layer-'):
            active=active and f'dec_layers.{int(args.direction.rsplit("-",1)[-1])}.' in name
        elif args.direction=='shared-radial':active='reslayerAs' in name
        elif args.direction!='gain-all':raise ValueError('Unknown physical direction')
        change.append(p.clone() if active else torch.zeros_like(p))
    zeros=tuple(torch.zeros_like(p) for p in weights)
    direction=AdamDirection(tuple(change),zeros,zeros)
    def functional(values,input_ids,fields=False):
        result=torch.func.functional_call(model.model,dict(zip(names,values,strict=True)),(input_ids,),
                 dict(use_cache=False,logits_to_keep=1,output_pldr_attentions=fields,output_hidden_states=False))
        logits=result.logits[:,-1]
        if not fields:return logits
        observations=[]
        for info in result.pldr_attentions:
            metric=info[0];probability=info[-1][:,:,-1]
            entropy=-(probability*probability.clamp_min(1e-38).log()).sum(-1)/np.log(64)
            centered=metric-metric.mean(-2,keepdim=True)
            row=centered.square().mean((-1,-2))/metric.square().mean((-1,-2)).clamp_min(1e-30)
            observations.append(torch.stack([entropy.mean(1),row.mean(1)],-1))
        return logits,torch.stack(observations,1)
    def gradients(values,tangent=None):
        leaf=tuple(p.detach().requires_grad_() for p in values)
        loss=torch.nn.functional.cross_entropy(functional(leaf,batch[:,:64]),batch[:,64])
        grad=torch.autograd.grad(loss,leaf,create_graph=tangent is not None)
        hv=None
        if tangent is not None:
            dot=sum((g*d).sum() for g,d in zip(grad,tangent,strict=True))
            hv=tuple(v.detach() for v in torch.autograd.grad(dot,leaf))
        return tuple(g.detach() for g in grad),hv,float(loss.detach())
    started=time.time()
    grad,hv,loss=gradients(state.weights,direction.weights)
    print('Hessian-vector product',round(time.time()-started,2),flush=True)
    with torch.no_grad():
        updated,tangent,statistics=adam_update_tangent(state,direction,grad,hv,rates)
        reference,reference_statistics=adam_update(state,grad,rates)
        primal_error=max(float((a-b).abs().max()) for a,b in zip(updated.weights,reference.weights,strict=True))
        native_optimizer=optimizer_for(model,condition['multiplier'])
        for (name,parameter),weight,first,second,gradient in zip(named,state.weights,state.first,state.second,grad,strict=True):
            parameter.copy_(weight);parameter.grad=gradient.clone()
            native_optimizer.state[parameter]=dict(step=torch.tensor(float(step)),exp_avg=first.clone(),exp_avg_sq=second.clone())
        torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
        native_optimizer.step()
        native_update_error=max(float((parameter-value).abs().max()) for (name,parameter),value in zip(named,updated.weights,strict=True))
        native_moment_error=max(float((native_optimizer.state[parameter][key]-value).abs().max())
            for key,values in [('exp_avg',updated.first),('exp_avg_sq',updated.second)]
            for (name,parameter),value in zip(named,values,strict=True))
        for (name,parameter),weight in zip(named,state.weights,strict=True):
            parameter.copy_(weight);parameter.grad=None
        del native_optimizer,reference,grad,hv
        (initial_logits,initial_fields),(initial_jvp,initial_field_jvp)=torch.func.jvp(
            lambda parameters:functional(parameters,ids,True),(state.weights,),(direction.weights,))
        (logits,fields),(jvp,field_jvp)=torch.func.jvp(
            lambda parameters:functional(parameters,ids,True),(updated.weights,),(tangent.weights,))
    probability=logits.softmax(-1)
    centered=jvp-(probability*jvp).sum(-1,keepdim=True)
    norms=[squared_norm(getattr(tangent,k)) for k in ['weights','first','second']]
    raw=dict(cohort=rows,batch=batches[step],native_cpu_logits=native_logits,
             initial_logits=initial_logits.numpy(),initial_fields=initial_fields.numpy(),initial_jvp=initial_jvp.numpy(),initial_field_jvp=initial_field_jvp.numpy(),
             updated_logits=logits.numpy(),updated_fields=fields.numpy(),updated_jvp=jvp.numpy(),updated_field_jvp=field_jvp.numpy(),
             amplitudes=np.array([float(x) for x in args.amplitudes.split(',')]))
    records=[];positive=[];negative=[];positive_fields=[];negative_fields=[]
    for radius in raw['amplitudes']:
        branches=[]
        for sign in [1,-1]:
            branch=AdamState(tuple(p+sign*radius*d for p,d in zip(state.weights,direction.weights,strict=True)),state.first,state.second,step)
            g,_,_=gradients(branch.weights)
            with torch.no_grad():
                result,_=adam_update(branch,g,rates)
                emission,field=functional(result.weights,ids,True)
            branches.append((result,emission,field));del g,branch
        plus,zplus,fplus=branches[0];minus,zminus,fminus=branches[1]
        errors=[]
        for key,norm in zip(['weights','first','second'],norms,strict=True):
            diff=tuple((p-m)/(2*radius)-d for p,m,d in zip(getattr(plus,key),getattr(minus,key),getattr(tangent,key),strict=True))
            errors.append(float(torch.sqrt(squared_norm(diff)/norm)) if float(norm)>1e-30 else None)
        secant=(zplus-zminus)/(2*radius);secant-=(probability*secant).sum(-1,keepdim=True)
        absolute=(probability*(secant-centered).square()).sum(-1).sqrt()
        scale=(probability*centered.square()).sum(-1).sqrt()
        relative=absolute/scale.clamp_min(1e-14)
        records.append(dict(amplitude=float(radius),augmented_relative_errors=errors,
                            maximum_predictive_relative_error=float(relative.max()),maximum_predictive_absolute_error=float(absolute.max())))
        positive.append(zplus.numpy());negative.append(zminus.numpy());positive_fields.append(fplus.numpy());negative_fields.append(fminus.numpy())
        print(records[-1],round(time.time()-started,2),flush=True)
        del branches,plus,minus,zplus,zminus,fplus,fminus,diff,result
        gc.collect()
    raw.update(positive_logits=np.array(positive),negative_logits=np.array(negative),positive_fields=np.array(positive_fields),negative_fields=np.array(negative_fields))
    for key,value in raw.items():
        if not np.isfinite(value).all():raise AssertionError('Nonfinite tangent evidence: '+key)
    np.savez_compressed(out/'measurements.npz',**raw)
    results=dict(schema='native-adam-tangent-v1',arguments=vars(args),condition=condition,step=step,loss=loss,
        update_statistics=statistics,update_formula_max_parameter_error=primal_error,
        native_optimizer_max_parameter_error=native_update_error,native_optimizer_max_moment_error=native_moment_error,records=records,
        native_smooth_initial_kl=predictive_kl(torch.tensor(native_logits),initial_logits).tolist(),
        initial_source_curvature=(initial_logits.softmax(-1)*(initial_jvp-(initial_logits.softmax(-1)*initial_jvp).sum(-1,keepdim=True)).square()).sum(-1).tolist(),
        updated_source_curvature=(probability*centered.square()).sum(-1).tolist(),seconds=time.time()-started,
        interpretation='Full-batch-32 native graph and exact declared optimizer formulas evaluated in expanded float64 arithmetic. Full weight and moment tangent checked by independent centered physical-weight pulses with identical moments. This one-step control does not establish a stationary relaxation eigenvalue or critical attraction.')
    write_json(out/'results.json',results)
    write_json(out/'manifest.json',dict(status='complete',results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
