#!/usr/bin/env python
"""Compare a native CPU float32 Adam step with its bound smooth64 reference."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import optimizer_for, sample_batches, predictive_kl, observations
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--reference',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root);reference=Path(args.reference)
    result=json.loads((reference/'results.json').read_text())
    checkpoint=Path(result['arguments']['checkpoint']);condition=result['condition']
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1';probe=root/'controlled-study-20260905/data/short'
    bind_run(out,[checkpoint,reference/'results.json',reference/'measurements.npz',
                  source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',
                  root/'data/refinedweb-4608/tokens.npy',probe/'tokens.npy',probe/'offsets.npy'],vars(args))
    torch.set_num_threads(4);started=time.time()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=TrainingModel(source,condition['heads'],condition['seed'],'cpu')
    model.model.load_state_dict(saved['model']);optimizer=optimizer_for(model,condition['multiplier'])
    optimizer.load_state_dict(saved['optimizer']);step=saved['step'];del saved
    raw=np.load(reference/'measurements.npz');rows=raw['cohort']
    ptokens=np.load(probe/'tokens.npy');offsets=np.load(probe/'offsets.npy')
    crops=ptokens[rows[:,None],offsets[rows,None]+np.arange(65)]
    ids=torch.tensor(crops[:,:64],dtype=torch.long);targets=torch.tensor(crops[:,64],dtype=torch.long)
    _,_,batches=sample_batches(np.load(root/'data/refinedweb-4608/tokens.npy'),condition['stream_seed'],step+1)
    np.testing.assert_array_equal(batches[step],raw['batch'])
    batch=torch.tensor(batches[step],dtype=torch.long)
    with torch.no_grad():
        model.model.eval();z0=model.logits(ids).detach().double()
    np.testing.assert_array_equal(z0.numpy(),raw['native_cpu_logits'])
    model.model.train();optimizer.zero_grad(set_to_none=True)
    loss=torch.nn.functional.cross_entropy(model.logits(batch[:,:64]),batch[:,64]);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();optimizer.zero_grad(set_to_none=True);model.model.eval()
    with torch.no_grad():
        fields,heads,z1,_=observations(model,ids,targets);z1=z1.double()
    k0=predictive_kl(z0,torch.tensor(raw['initial_logits'])).numpy()
    k1=predictive_kl(z1,torch.tensor(raw['updated_logits'])).numpy()
    arrays=dict(cohort=rows,batch=batches[step],initial_native_logits=z0.numpy(),updated_native_logits=z1.numpy(),
                updated_native_fields=fields.numpy(),updated_native_heads=heads.numpy(),initial_arithmetic_kl=k0,updated_arithmetic_kl=k1)
    for key,value in arrays.items():
        if not np.isfinite(value).all():raise AssertionError('Nonfinite arithmetic control: '+key)
    np.savez_compressed(out/'measurements.npz',**arrays)
    write_json(out/'results.json',dict(schema='native-transport-arithmetic-v1',arguments=vars(args),condition=condition,step=step,
        initial_arithmetic_kl=k0.tolist(),updated_arithmetic_kl=k1.tolist(),native_loss=float(loss.detach()),
        smooth_loss=result['loss'],native_gradient_norm=float(norm),smooth_gradient_norm=result['update_statistics'],seconds=time.time()-started,
        interpretation='The reference alters rounding and expands smooth primitives. This paired native CPU float32 step measures the finite arithmetic difference independently of its derivative. A resolved derivative of the smooth64 program does not alone establish a native-arithmetic response window.'))
    write_json(out/'manifest.json',dict(status='complete',results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
