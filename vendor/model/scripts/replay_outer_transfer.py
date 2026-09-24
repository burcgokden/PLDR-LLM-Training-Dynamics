#!/usr/bin/env python
"""Independently replay one assessment continuation at a selected width."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.provenance import sha256,write_json
from model_rg.schedules import loss,clip
from model_rg.training import TrainingModel
from measure_law_closure import digest_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--case',required=True);p.add_argument('--device',required=True);p.add_argument('--branch',type=int,default=31)
    p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();study=Path(a.study).resolve();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(out)
    start=time.time();folder=study/'runs'/a.case;meta=json.loads((folder/'results.json').read_text());case=meta['case']
    assert 24<=a.branch<56;record=meta['records'][a.branch]
    assert sha256(folder/'incoming-state.pt')==meta['incoming_state_sha256'] and sha256(folder/record['raw'])==record['sha256']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(folder/'incoming-state.pt',map_location='cpu',mmap=True,weights_only=True)
    model=TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1',case['heads'],case['seed'],a.device)
    model.model.load_state_dict(saved['model']);optimizer,scheduler=optimizer_and_scheduler(model,case['profile'])
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer']));scheduler.load_state_dict(copy.deepcopy(saved['scheduler']))
    assert digest_state(model,optimizer,scheduler)==meta['incoming_digest'];del saved
    data=np.load(study/'data'/f"corpus-{case['corpus']}.npy",mmap_mode='r')
    with np.load(folder/'sampling.npz') as f:blocks=f['branch_blocks'][a.branch];crops=f['evaluation_crops']
    raw=np.load(folder/record['raw']);count=0
    def check_logits(index):
        nonlocal count
        model.model.eval()
        with torch.no_grad():
            x=torch.as_tensor(crops[:,:64],dtype=torch.long,device=a.device)
            value=model.model(x,use_cache=False,logits_to_keep=1).logits[:,-1].cpu().numpy()
        expected=raw['logits'][index]
        assert value.shape==expected.shape and value.dtype==expected.dtype and value.tobytes()==expected.tobytes()
        count+=value.size
    check_logits(0);index=1
    for step,ids in enumerate(blocks,1):
        array=data[(ids//8)[:,None],(64*(ids%8))[:,None]+np.arange(65)]
        b=torch.as_tensor(array,dtype=torch.long,device=a.device);model.model.train();optimizer.zero_grad(set_to_none=True)
        value=loss(model,b,case['profile']);value.backward();clip(model,case['profile']);optimizer.step();scheduler.step()
        assert float(value.detach())==float(raw['losses'][step-1])
        if step in [1,4,16,64]:check_logits(index);index+=1
    assert digest_state(model,optimizer,scheduler)==record['final_state_sha256']
    write_json(out,dict(status='passed',case=a.case,branch=a.branch,device=a.device,native_updates=64,
        logit_coordinates=count,logits_bitwise=True,losses_bitwise=True,complete_parameter_adam_digest_bitwise=True,
        input_sha256={str(folder/'incoming-state.pt'):meta['incoming_state_sha256'],str(folder/record['raw']):record['sha256'],
                      str(folder/'sampling.npz'):sha256(folder/'sampling.npz')},replayer_sha256=sha256(__file__),
        completed_at=datetime.now(timezone.utc).isoformat(),seconds=time.time()-start,
        scope='Replay of an existing assessment path, not a new independent observation.'))


if __name__=='__main__':main()
