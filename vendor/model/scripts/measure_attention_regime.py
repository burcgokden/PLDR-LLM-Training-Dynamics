#!/usr/bin/env python
"""Resolve microscopic saturation in completed checkpoints on a fixed cohort."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--parent',required=True);args=parser.parse_args()
    root=Path(args.root);study=root/'criticality-study-20260905'
    parent=study/'runs'/args.parent;out=study/'attention-regimes'/args.parent
    out.mkdir(parents=True,exist_ok=False)
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1';probe=root/'controlled-study-20260905/data/short'
    bind_run(out,[parent/'manifest.json',parent/'final-training-state.pt',probe/'manifest.json',
                  probe/'tokens.npy',probe/'offsets.npy',source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',
                  study/'protocols/attention-regimes.json'],vars(args))
    meta=json.loads((parent/'manifest.json').read_text());condition=meta['arguments']
    if meta['status']!='complete':raise RuntimeError('A complete training checkpoint is required')
    torch.set_num_threads(4)
    state=torch.load(parent/'final-training-state.pt',map_location='cpu',weights_only=True,mmap=True)
    model=TrainingModel(source,condition['heads'],condition['seed'],'cpu')
    model.model.load_state_dict(state['model']);model.model.eval();del state
    tokens=np.load(probe/'tokens.npy');offsets=np.load(probe/'offsets.npy');rows=np.arange(512,576)
    crops=tokens[rows[:,None],offsets[rows,None]+np.arange(64)]
    arrays={name:[] for name in ['probability_max','jacobian_trace','jacobian_frobenius','top_key','operator_rms']}
    started=time.time()
    with torch.no_grad():
        for start in range(0,len(crops),16):
            output=model.forward(torch.tensor(crops[start:start+16],dtype=torch.long),capture=True)
            fields={key:[] for key in arrays}
            for values in output.pldr_attentions:
                g,weights=values[5:7]
                p=weights[:,:,-1].double();p=p/p.sum(-1,keepdim=True)
                p2=p.square().sum(-1)
                trace=1-p2
                frobenius=(p2-2*p.pow(3).sum(-1)+p2.square()).clamp_min(0).sqrt()
                fields['probability_max'].append(p.max(-1).values.numpy())
                fields['jacobian_trace'].append(trace.numpy())
                fields['jacobian_frobenius'].append(frobenius.numpy())
                fields['top_key'].append(p.argmax(-1).numpy())
                fields['operator_rms'].append(g.double().square().mean((-2,-1)).sqrt().numpy())
            for key in arrays:arrays[key].append(np.stack(fields[key],axis=1))
    arrays={key:np.concatenate(value) for key,value in arrays.items()}
    np.savez_compressed(out/'measurements.npz',**arrays)
    write_json(out/'manifest.json',dict(schema='attention-regime-v1',condition=condition,rows=rows.tolist(),
        seconds=time.time()-started,raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json'),
        precision='Native float32 attention probabilities renormalized in float64 for categorical Jacobian statistics',
        summary=dict(mean_pmax=float(arrays['probability_max'].mean()),
                     fraction_pmax_above_099=float((arrays['probability_max']>.99).mean()),
                     mean_jacobian_trace=float(arrays['jacobian_trace'].mean()),
                     mean_jacobian_frobenius=float(arrays['jacobian_frobenius'].mean()),
                     first_key_fraction=float((arrays['top_key']==0).mean()),
                     last_key_fraction=float((arrays['top_key']==63).mean()))))


if __name__=='__main__':main()
