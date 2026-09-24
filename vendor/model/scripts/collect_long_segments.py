#!/usr/bin/env python
"""Native model-wide segment fields and direct G variation on a fresh cohort."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.native import NativeModel
from model_rg.controlled import bind_run, device_name
from model_rg.provenance import sha256, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--checkpoint',type=int,choices=[1,2],required=True);ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args();a.device=device_name(a.device);root=Path(a.root);study=root/'controlled-study-20260905'
    out=study/'runs'/f'long-soc{a.checkpoint}';out.mkdir(parents=True,exist_ok=False)
    source=root/f'assets/PLDR-LLM-v51-SOC-110M-{a.checkpoint}';data=study/'data/long'
    bind_run(out,[source/'model.safetensors',source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',
                  data/'manifest.json',data/'tokens.npy',study/'protocol.json'],vars(a))
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    m=NativeModel(source,a.device);tokens=np.load(data/'tokens.npy');n=len(tokens)
    ids=tokens[:,:4096].reshape(-1,64);targets=tokens[:,np.arange(64,4097,64)].reshape(-1)
    pieces=[];start=time.time()
    with torch.no_grad():
        for i in range(0,len(ids),64):
            f,_=m.features(torch.tensor(ids[i:i+64],dtype=torch.long,device=a.device),
                           torch.tensor(targets[i:i+64],dtype=torch.long,device=a.device))
            if not torch.isfinite(f).all():raise RuntimeError('Nonfinite segment field')
            pieces.append(f.cpu().numpy())
            if i%8192==0:print(a.checkpoint,i,len(ids),round(time.time()-start,1),flush=True)
    np.savez_compressed(out/'segments.npz',features=np.concatenate(pieces).reshape(n,64,-1))
    write_json(out/'manifest.json',dict(schema='long-segments-v1',arguments=vars(a),documents=n,
        segments=64,segment_length=64,feature_names=m.feature_names(),raw_sha256=sha256(out/'segments.npz'),
        binding_sha256=sha256(out/'binding.json'),seconds=time.time()-start,
        peak_cuda_gb=torch.cuda.max_memory_allocated(a.device)/2**30))
    print('Completed',a.checkpoint,round(time.time()-start,1),flush=True)


if __name__=='__main__':main()
