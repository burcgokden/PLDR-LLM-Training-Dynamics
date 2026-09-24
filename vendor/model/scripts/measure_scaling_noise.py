#!/usr/bin/env python
"""Conditional native Adam drift, batch noise and generator-only emission response."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.criticality import generator_parameter, optimizer_for, predictive_kl, sample_batches
from model_rg.precision import preserve_response_dtype, expand_smooth_response
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def snapshot(model, crops):
    model.model.eval()
    batch = torch.tensor(crops,device=model.device,dtype=torch.long)
    with torch.no_grad():
        output = model.forward(batch[:,:64],capture=True)
        logits = output.logits[:,-1].double()
        matrices = torch.stack([x[0].double() for x in output.pldr_attentions],1)
        operators = torch.stack([x[5].double() for x in output.pldr_attentions],1)
        mean = matrices.mean(-2,keepdim=True)
        energy = (matrices-mean).square().mean((-2,-1))
        total = matrices.square().mean((-2,-1))
        centered = matrices-matrices.mean(-1,keepdim=True)
        rows = centered/centered.square().sum(-1,keepdim=True).sqrt().clamp_min(1e-30)
        logp = logits.log_softmax(-1)
        entropy = -(logp.exp()*logp).sum(-1)
        nll = -logp.gather(1,batch[:,64,None]).squeeze(1)
        fields = torch.stack([energy,total,energy/total.clamp_min(1e-30),
                              mean.squeeze(-2).square().mean(-1),
                              operators.square().mean((-2,-1)).sqrt()],-1)
    return dict(fields=fields.cpu().numpy(),normalized_rows=rows.cpu().numpy(),
                prediction_entropy=entropy.cpu().numpy(),nll=nll.cpu().numpy(),logits=logits.cpu().numpy())


def analytical_emission(model, crops, base_parameters, directions, alpha):
    batch = torch.tensor(crops,device=model.device,dtype=torch.long)
    parameters = {name:value+alpha*directions[name] for name,value in base_parameters.items()}
    model.eta=None;model.capture=True;model.head_outputs={}
    output = torch.func.functional_call(model.model,parameters,(batch[:,:64],),
        dict(use_cache=False,logits_to_keep=1,output_pldr_attentions=True,output_hidden_states=True))
    logits = output.logits[:,-1]
    matrices = torch.stack([x[0] for x in output.pldr_attentions],1)
    operators = torch.stack([x[5] for x in output.pldr_attentions],1)
    mean = matrices.mean(-2,keepdim=True)
    energy = (matrices-mean).square().mean((-2,-1))
    total = matrices.square().mean((-2,-1))
    centered = matrices-matrices.mean(-1,keepdim=True)
    rows = centered/centered.square().sum(-1,keepdim=True).sqrt().clamp_min(1e-30)
    logp = logits.log_softmax(-1)
    fields = torch.stack([energy,total,energy/total.clamp_min(1e-30),
        mean.squeeze(-2).square().mean(-1),operators.square().mean((-2,-1)).sqrt()],-1)
    return rows,fields,logits,-logp.gather(1,batch[:,64,None]).squeeze(1),-(logp.exp()*logp).sum(-1)


def fit_common_tangent(base, tangent):
    """Fit one centered common forcing per layer on 8 contexts; validate on 8."""
    d = base.shape[-1]
    center = np.eye(d)-np.ones((d,d))/d
    records, vectors = [], []
    for layer in range(base.shape[1]):
        u = base[:8,layer].reshape(-1,d)
        v = tangent[:8,layer].reshape(-1,d)
        normal = len(u)*center-u.T@u
        rhs = (v-v.mean(-1,keepdims=True)-u*np.sum(u*v,axis=-1,keepdims=True)).sum(0)
        eigenvalue,eigenvector = np.linalg.eigh(normal)
        active = eigenvalue > max(eigenvalue.max()*1e-10,1e-30)
        vector = eigenvector[:,active]@((eigenvector[:,active].T@rhs)/eigenvalue[active])
        vectors.append(vector)
        item = dict(layer=layer,rank=int(active.sum()),condition=float(eigenvalue[active].max()/eigenvalue[active].min()))
        for name,selection in [('calibration',slice(0,8)),('validation',slice(8,16))]:
            x,y = base[selection,layer],tangent[selection,layer]
            prediction = vector-vector.mean()-x*np.sum(x*vector,axis=-1,keepdims=True)
            item[name+'_residual_squared'] = float(np.sum((prediction-y)**2))
            item[name+'_tangent_squared'] = float(np.sum(y*y))
        records.append(item)
    return np.array(vectors),records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906')
    p.add_argument('--protocol',default='conditional-noise.json')
    p.add_argument('--case',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    a = p.parse_args()
    root,study = Path(a.root),Path(a.root)/a.study
    protocol = study/'protocols'/a.protocol
    spec = json.loads(protocol.read_text())
    case = next(c for c in spec['cases'] if c['name']==a.case)
    out = study/'measurements'/case['name']
    out.mkdir(parents=True,exist_ok=False)
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    data,probe = root/'data/refinedweb-4608',root/'controlled-study-20260905/data/short'
    if sha256(case['parent'])!=case['parent_sha256']:
        raise AssertionError('Parent checkpoint changed')
    bind_run(out,[protocol,Path(case['parent']),Path(case['parent_manifest']),data/'tokens.npy',
                  probe/'tokens.npy',probe/'offsets.npy',source/'modeling_pldrllm.py',
                  source/'configuration_pldrllm.py'],vars(a))
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    parent = torch.load(case['parent'],map_location='cpu',weights_only=True)
    condition = parent['arguments']
    if (condition['heads'],condition['seed'],parent['step'])!=(case['heads'],case['seed'],case['step']):
        raise AssertionError('Parent identity changed')
    count,response_count = spec['batch_replicas'],spec['response_batches']
    rows,offsets,batches = sample_batches(np.load(data/'tokens.npy'),spec['batch_seed'],count)
    pt,po = np.load(probe/'tokens.npy'),np.load(probe/'offsets.npy')
    selected = np.arange(512,528)
    crops = pt[selected[:,None],po[selected,None]+np.arange(65)]
    model = TrainingModel(source,case['heads'],case['seed'],a.device)
    model.model.load_state_dict(parent['model'])
    optimizer = optimizer_for(model,condition['multiplier'])
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    response = TrainingModel(source,case['heads'],case['seed'],a.device)
    response.model.load_state_dict(parent['model'])
    response.model.double()
    preserve_response_dtype(response)
    native_base,dtype_base = snapshot(model,crops),snapshot(response,crops)
    if spec.get('analytical_response'):
        expand_smooth_response(response)
    response.model.requires_grad_(False)
    base = snapshot(response,crops)
    named = [(n,p) for n,p in model.model.named_parameters() if generator_parameter(n)]
    total = sum(p.numel() for _,p in named)
    velocity_path = out/'generator-velocities.npy'
    velocities = np.lib.format.open_memmap(velocity_path,mode='w+',dtype=np.float64,shape=(count,total))
    response_named = dict(response.model.named_parameters())
    raw = dict(rows=rows,offsets=offsets,cohort=selected,
               amplitudes=np.array(spec['amplitudes']),native_base_fields=native_base['fields'],
               base_fields=base['fields'],dtype_base_fields=dtype_base['fields'],dtype_base_logits=dtype_base['logits'],
               base_normalized_rows=base['normalized_rows'],
               base_nll=base['nll'],base_prediction_entropy=base['prediction_entropy'],base_logits=base['logits'])
    q_records,projection_records,noise_vectors,tangents = [],[],[],[]
    names,slices = [],[]
    first=0
    for n,pv in named:
        names.append(n);slices.append([first,first+pv.numel()]);first+=pv.numel()
    losses,norms,clip_factors = [],[],[]
    native_update = None
    started = time.time()
    for k in range(count):
        model.model.train();optimizer.zero_grad(set_to_none=True)
        batch = torch.tensor(batches[k],device=model.device,dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model.logits(batch[:,:64]),batch[:,64])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite conditional loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
        losses.append(float(loss.detach()));norms.append(float(norm));clip_factors.append(min(1.,1/(float(norm)+1e-6)))
        directions = []
        for (name,pv),(lo,hi) in zip(named,slices,strict=True):
            st = optimizer.state[pv]
            counter = int(st['step'])+1
            gradient = pv.grad.detach().double()
            m = .9*st['exp_avg'].double()+.1*gradient
            v = .95*st['exp_avg_sq'].double()+.05*gradient.square()
            velocity = -.01*pv.detach().double()-(m/(1-.9**counter))/(torch.sqrt(v/(1-.95**counter))+1e-8)
            velocities[k,lo:hi] = velocity.detach().cpu().reshape(-1).numpy()
            directions.append(velocity)
        if k==0:
            rate = optimizer.param_groups[0]['lr']
            predicted = [pv.detach().double()+rate*v for (_,pv),v in zip(named,directions,strict=True)]
            optimizer.step()
            numerator = sum(float((pv.detach().double()-v).square().sum()) for (_,pv),v in zip(named,predicted,strict=True))
            denominator = sum(float(pv.detach().double().square().sum()) for _,pv in named)
            native_update = dict(relative_parameter_l2=(numerator/denominator)**.5,
                                 maximum_absolute=max(float((pv.detach().double()-v).abs().max()) for (_,pv),v in zip(named,predicted,strict=True)),
                                 scope='Real-arithmetic Adam formula evaluated in float64 from clipped CPU float32 gradients, compared with the native float32 AdamW step.')
            model.model.load_state_dict(parent['model'])
            optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        if k<response_count:
            derivative=None;second_fields=None
            if spec.get('analytical_response'):
                bp = {name:parent['model'][name].to(device=a.device,dtype=torch.float64) for name,_ in named}
                direction = {name:3e-4*condition['multiplier']*velocity for (name,_),velocity in zip(named,directions,strict=True)}
                zero = torch.tensor(0.,device=a.device,dtype=torch.float64)
                one = torch.ones_like(zero)
                def emission(alpha):
                    return analytical_emission(response,crops,bp,direction,alpha)
                with torch.no_grad():
                    primal,derivative = torch.func.jvp(emission,(zero,),(one,))
                    def field_derivative(alpha):
                        return torch.func.jvp(lambda x:emission(x)[1],(alpha,),(one,))[1]
                    _,second_fields = torch.func.jvp(field_derivative,(zero,),(one,))
                for label,value in zip(['normalized_rows','fields','logits','nll','prediction_entropy'],derivative,strict=True):
                    raw[f'b{k}_jvp_'+label] = value.detach().cpu().numpy()
                raw[f'b{k}_second_fields'] = second_fields.detach().cpu().numpy()
                raw[f'b{k}_jvp_primal_fields'] = primal[1].detach().cpu().numpy()
                raw[f'b{k}_jvp_primal_logits'] = primal[2].detach().cpu().numpy()
            values = {}
            for amplitude in spec['amplitudes']:
                with torch.no_grad():
                    for (name,_),velocity in zip(named,directions,strict=True):
                        response_named[name].copy_(parent['model'][name].to(device=a.device,dtype=torch.float64)+
                                                  amplitude*3e-4*condition['multiplier']*velocity)
                measured = snapshot(response,crops)
                values[amplitude] = measured
                for field in ['fields','nll','prediction_entropy']:
                    raw[f'b{k}_a{amplitude}_{field}'] = measured[field]
                raw[f'b{k}_a{amplitude}_logit_delta'] = measured['logits']-base['logits']
                raw[f'b{k}_a{amplitude}_kl'] = predictive_kl(torch.from_numpy(base['logits']),torch.from_numpy(measured['logits'])).numpy()
            projection_amplitude = spec.get('projection_amplitude', .25)
            tangent = (raw[f'b{k}_jvp_normalized_rows'] if derivative is not None else
                (values[projection_amplitude]['normalized_rows']-values[-projection_amplitude]['normalized_rows'])/(2*projection_amplitude))
            vector,fit = fit_common_tangent(base['normalized_rows'],tangent)
            tangents.append(tangent);noise_vectors.append(vector)
            projection_records.append(dict(batch=k,layers=fit))
            for magnitude in sorted(x for x in spec['amplitudes'] if x>0):
                plus,minus = values[magnitude]['fields'],values[-magnitude]['fields']
                row_secant = (values[magnitude]['normalized_rows']-values[-magnitude]['normalized_rows'])/(2*magnitude)
                odd = (plus-minus)/(2*magnitude)
                even = (plus+minus-2*base['fields'])/(2*magnitude**2)
                errors = {}
                if derivative is not None:
                    first = raw[f'b{k}_jvp_fields'];second = raw[f'b{k}_second_fields']/2
                    errors = dict(odd_relative_error=float(np.linalg.norm(odd-first)/max(np.linalg.norm(first),1e-30)),
                                  even_relative_error=float(np.linalg.norm(even-second)/max(np.linalg.norm(second),1e-30)))
                q_records.append(dict(batch=k,amplitude=magnitude,**errors,
                    row_secant_relative_difference=float(np.linalg.norm(row_secant-tangent)/max(np.linalg.norm(tangent),1e-30)),
                    odd_mean=((plus-minus)/(2*magnitude)).mean((0,1,2)).tolist(),
                    even_mean=((plus+minus-2*base['fields'])/(2*magnitude**2)).mean((0,1,2)).tolist()))
        print(a.case,k+1,'of',count,'seconds',round(time.time()-started,1),flush=True)
    velocities.flush()
    raw.update(losses=np.array(losses),gradient_norms=np.array(norms),clip_factors=np.array(clip_factors),
               projected_common_forcing=np.array(noise_vectors),normalized_row_tangents=np.array(tangents))
    group_records = []
    for group in ['all_generator','shared_metric_network','plga_affine']:
        columns = [(lo,hi) for name,(lo,hi) in zip(names,slices,strict=True)
                   if group=='all_generator' or ('reslayerAs' in name)==(group=='shared_metric_network')]
        second,mean_squared,trace = 0.,0.,0.
        for lo,hi in columns:
            v = np.asarray(velocities[:,lo:hi])
            mean = v.mean(0)
            second += float(np.square(v).mean(0).sum())
            mean_squared += float(np.square(mean).sum())
            trace += float(np.square(v-mean).sum()/(count-1))
        group_records.append(dict(group=group,coordinates=sum(hi-lo for lo,hi in columns),
                                  second_raw_moment=second,squared_sample_mean=mean_squared,
                                  unbiased_noise_trace=trace,unbiased_squared_drift=mean_squared-trace/count))
    for key,value in raw.items():
        if not np.isfinite(value).all():
            raise AssertionError('Nonfinite output '+key)
    np.savez_compressed(out/'measurements.npz',**raw)
    result = dict(schema='conditional-native-adam-noise-v1',status='complete',case=case,arguments=vars(a),
                  groups=group_records,native_update_control=native_update,response_moments=q_records,
                  common_tangent_fits=projection_records,parameter_names=names,parameter_slices=slices,
                  field_names=['absolute_row_energy','total_metric_energy','normalized_row_energy','common_row_energy','operator_rms'],
                  velocity_units='Generator parameter displacement divided by its native learning rate; full clipped gradient and incoming Adam state retained.',
                  response_units='Float64 mathematical emission after a signed multiple of one generator-only real-arithmetic update; remaining parameters and all optimizer moments fixed.',
                  independence='32 fresh IID empirical-data batches conditional on one full trained state. First 8 batches have paired emissions. Neither batch replicas nor contexts count as independent training identities.',
                  projection_scope='One common centered forcing per decoder is fitted to the normalized-row tangent on contexts 512:520 and validated on 520:528. This diagnostic changes coordinates explicitly and does not replace fixed-unit susceptibilities.',
                  seconds=time.time()-started)
    write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(schema=result['schema'],status='complete',case=case,
        raw_sha256=sha256(out/'measurements.npz'),velocities_sha256=sha256(velocity_path),
        results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json')))
    del model,response,optimizer,parent,velocities
    gc.collect()


if __name__=='__main__':
    main()
