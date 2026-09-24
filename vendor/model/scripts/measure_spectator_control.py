#!/usr/bin/env python
"""Verify an exact, prediction-invisible, decay-only native parameter sector."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.criticality import optimizer_for, sample_batches
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def digest_remainder(model, optimizer, embedding_name, token):
    digest=hashlib.sha256()
    for name,parameter in model.model.named_parameters():
        value=parameter.detach().cpu()
        digest.update(name.encode())
        if name==embedding_name:
            digest.update(value[:token].contiguous().numpy().tobytes())
            digest.update(value[token+1:].contiguous().numpy().tobytes())
        else:
            digest.update(value.contiguous().numpy().tobytes())
        for key in ['step','exp_avg','exp_avg_sq']:
            digest.update(key.encode())
            digest.update(optimizer.state[parameter][key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906')
    p.add_argument('--protocol',default='spectator-control.json')
    a=p.parse_args()
    root,study=Path(a.root),Path(a.root)/a.study
    protocol=study/'protocols'/a.protocol
    spec=json.loads(protocol.read_text())
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    data,probe=root/'data/refinedweb-4608',root/'controlled-study-20260905/data/short'
    tokens=np.load(data/'tokens.npy')
    pt,po=np.load(probe/'tokens.npy'),np.load(probe/'offsets.npy')
    all_rows=np.arange(1024)
    inputs=pt[all_rows[:,None],po[all_rows,None]+np.arange(64)]
    training_types=np.unique(tokens[:3072,:512])
    probe_types=np.unique(inputs)
    absent=np.setdiff1d(np.arange(32000),np.union1d(training_types,probe_types))
    token=int(absent[absent>=1024][0])
    if token!=spec['token']:
        raise AssertionError('Declared data support changed')
    torch.set_num_threads(4)
    results=[]
    for case in spec['cases']:
        out=study/'measurements'/case['name'];out.mkdir(parents=True,exist_ok=False)
        if sha256(case['state'])!=case['state_sha256']:
            raise AssertionError('Parent changed')
        bind_run(out,[protocol,Path(case['state']),Path(case['parent_manifest']),data/'tokens.npy',
                      probe/'tokens.npy',probe/'offsets.npy',source/'modeling_pldrllm.py',
                      source/'configuration_pldrllm.py'],vars(a))
        parent=torch.load(case['state'],map_location='cpu',weights_only=True)
        condition=parent['arguments'];start=parent['step']
        if start!=16384 or condition['heads']!=case['heads']:
            raise AssertionError('Parent identity changed')
        rows,offsets,batches=sample_batches(tokens,condition['stream_seed'],start+spec['updates'])
        batches=batches[start:]
        if (batches[:,:,:64]==token).any():
            raise AssertionError('Spectator appears in an input')
        raw=dict(absent_input_types=absent,training_input_types=training_types,probe_input_types=probe_types,
                 batches=batches,rows=rows[start:],offsets=offsets[start:],steps=np.arange(spec['updates']+1))
        records=[];started=time.time()
        for name,amplitude in [('base',0.),('pulse',spec['amplitude'])]:
            model=TrainingModel(source,case['heads'],condition['seed'],'cpu')
            model.model.load_state_dict(parent['model'])
            optimizer=optimizer_for(model,condition['multiplier'])
            optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
            embedding=model.model.get_input_embeddings().weight
            output=model.model.get_output_embeddings().weight
            if embedding is output or embedding.data_ptr()==output.data_ptr() or model.config.tie_word_embeddings:
                raise AssertionError('This control requires untied embeddings')
            embedding_name=next(n for n,pv in model.model.named_parameters() if pv is embedding)
            st=optimizer.state[embedding]
            if st['exp_avg'][absent].abs().max()!=0 or st['exp_avg_sq'][absent].abs().max()!=0:
                raise AssertionError('An absent-input moment is nonzero')
            with torch.no_grad():
                if amplitude:
                    direction=torch.from_numpy(np.random.default_rng(spec['direction_seed']).normal(size=embedding.shape[1])).float()
                    direction/=direction.norm()
                    embedding[token].add_(amplitude*direction)
            rate=optimizer.param_groups[1]['lr'];factor=1-rate*.01
            expected=embedding[token].detach().clone()
            row_values=[expected.numpy().copy()];losses=[];norms=[];max_gradient=0.;bitwise=True
            model.model.eval()
            with torch.no_grad():
                raw[name+'_initial_logits']=model.logits(torch.tensor(inputs[512:528])).numpy()
            for k,batch_np in enumerate(batches):
                model.model.train();optimizer.zero_grad(set_to_none=True)
                batch=torch.tensor(batch_np,dtype=torch.long)
                loss=torch.nn.functional.cross_entropy(model.logits(batch[:,:64]),batch[:,64])
                loss.backward()
                max_gradient=max(max_gradient,float(embedding.grad[absent].abs().max()))
                norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():expected.mul_(factor)
                bitwise=bitwise and (expected.detach().numpy().tobytes()
                    == embedding[token].detach().numpy().tobytes())
                row_values.append(embedding[token].detach().numpy().copy())
                losses.append(float(loss.detach()));norms.append(float(norm))
            model.model.eval()
            with torch.no_grad():
                raw[name+'_final_logits']=model.logits(torch.tensor(inputs[512:528])).numpy()
            raw[name+'_embedding_rows']=np.array(row_values)
            raw[name+'_losses']=np.array(losses);raw[name+'_gradient_norms']=np.array(norms)
            records.append(dict(branch=name,amplitude=amplitude,maximum_absent_gradient=max_gradient,
                native_decay_bitwise=bitwise,remainder_state_sha256=digest_remainder(model,optimizer,embedding_name,token),
                embedding_parameter=embedding_name,rate=rate,real_decay_factor=factor,
                real_relaxation_updates=float(-1/np.log1p(-rate*.01))))
            del model,optimizer,embedding,output,st;gc.collect()
        same=(records[0]['remainder_state_sha256']==records[1]['remainder_state_sha256'])
        for label in ['initial_logits','final_logits','losses','gradient_norms']:
            same=same and (raw['base_'+label].dtype == raw['pulse_'+label].dtype
                and raw['base_'+label].shape == raw['pulse_'+label].shape
                and raw['base_'+label].tobytes() == raw['pulse_'+label].tobytes())
        complete=same and all(r['native_decay_bitwise'] and r['maximum_absent_gradient']==0 for r in records)
        np.savez_compressed(out/'measurements.npz',**raw)
        result=dict(schema='native-spectator-control-v1',status='complete' if complete else 'control_failure',
                    case=case,token=token,absent_input_types=len(absent),training_input_types=len(training_types),
                    probe_input_types=len(probe_types),branches=records,unchanged_observed_and_remainder_state=same,
                    seconds=time.time()-started,scope='Untied input-embedding rows absent from every input of the declared empirical training and 1024 probe contexts. The complete predictive vocabulary remains observed. No assertion of invisibility on arbitrary prompts is made.')
        write_json(out/'results.json',result)
        write_json(out/'manifest.json',dict(schema=result['schema'],status=result['status'],case=case,
            results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json')))
        results.append(dict(name=case['name'],status=result['status'],manifest_sha256=sha256(out/'manifest.json')))
        print(case['name'],result['status'],round(result['seconds'],1),flush=True)
        if not complete:
            raise AssertionError('The declared spectator control failed')
    write_json(study/'launcher-spectator-control.json',dict(status='complete',records=results,protocol_sha256=sha256(protocol)))


if __name__=='__main__':
    main()
