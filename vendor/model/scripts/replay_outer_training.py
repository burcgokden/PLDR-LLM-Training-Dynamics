#!/usr/bin/env python
"""Replay a complete primary path from its seeds, without loading its weights."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from model_rg.criticality import fix_shared_generator,generator_parameter
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization
from measure_law_closure import digest_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--case',required=True);p.add_argument('--device',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();study=Path(a.study).resolve();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(out)
    begin=time.time();folder=study/'runs'/a.case;meta=json.loads((folder/'results.json').read_text());case=meta['case']
    raw=np.load(folder/'primary.npz');assert sha256(folder/'primary.npz')==meta['files']['primary.npz']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1',case['heads'],case['seed'],a.device)
    normalize_variance_initialization(model);fix_shared_generator(model,case['generator_seed'])
    digests={k:hashlib.sha256() for k in ['body','generator','metric']}
    for name,value in model.model.named_parameters():
        data=value.detach().cpu().numpy().tobytes();key='generator' if generator_parameter(name) else 'body'
        digests[key].update(name.encode());digests[key].update(data)
        if 'reslayerAs' in name:digests['metric'].update(name.encode());digests['metric'].update(data)
    assert {k:h.hexdigest() for k,h in digests.items()}==meta['initial_component_sha256']
    optimizer,scheduler=optimizer_and_scheduler(model,case['profile'])
    corpus=np.load(study/'data'/f"corpus-{case['corpus']}.npy",mmap_mode='r')
    with np.load(study/'data/panels.npz') as f:crops=f['evaluation']
    def emission(key):
        model.model.eval()
        with torch.no_grad():
            ids=torch.as_tensor(crops[:,:64],dtype=torch.long,device=a.device)
            logits=model.model(ids,use_cache=False,logits_to_keep=1).logits[:,-1].cpu().numpy()
        assert logits.dtype==raw[key].dtype and logits.shape==raw[key].shape and logits.tobytes()==raw[key].tobytes()
    emission('initial_logits')
    order=np.random.default_rng(case['stream_seed']).permutation(524288)[:65536].reshape(2048,32)
    assert np.array_equal(order,raw['blocks'])
    for step,blocks in enumerate(order):
        batch=corpus[(blocks//8)[:,None],(64*(blocks%8))[:,None]+np.arange(65)]
        x=torch.as_tensor(batch,dtype=torch.long,device=a.device)
        model.model.train();optimizer.zero_grad(set_to_none=True)
        logits=model.model(x[:,:64],use_cache=False,logits_to_keep=1).logits[:,-1]
        loss=torch.nn.functional.cross_entropy(logits,x[:,64]);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        assert float(loss.detach())==float(raw['losses'][step]),step
        if (step+1)%512==0:print(a.case,'replayed primary',step+1,flush=True)
    emission('incoming_logits')
    assert digest_state(model,optimizer,scheduler)==meta['incoming_digest']
    write_json(out,dict(status='passed',case=a.case,device=a.device,native_updates=2048,
        source='Complete replay from seeds and raw source blocks, with direct native loss; no trained weights loaded.',
        initial_component_digests_bitwise=True,losses_bitwise=True,logits_bitwise=True,
        incoming_complete_parameter_adam_digest_bitwise=True,logit_coordinates=2048000,
        inputs_sha256={str(folder/'primary.npz'):sha256(folder/'primary.npz'),str(folder/'results.json'):sha256(folder/'results.json'),
                       str(study/'protocol.json'):sha256(study/'protocol.json')},
        replayer_sha256=sha256(__file__),seconds=time.time()-begin,
        scope='Verification replay of an existing primary path, not another scientific training replication.'))


if __name__=='__main__':main()
