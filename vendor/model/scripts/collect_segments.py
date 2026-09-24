#!/usr/bin/env python
"""Keep text-segment dependence while treating complete documents as sample units."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.native import NativeModel
from model_rg.provenance import environment, sha256, source_manifest, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    args=p.parse_args()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    m=NativeModel(args.model,args.device)
    tokens=np.load(Path(args.data)/"tokens.npy")
    # Eight unpadded calls. Each target is outside its own 64-token input.
    ids=tokens[:,:512].reshape(-1,64)
    targets=tokens[:,np.arange(64,513,64)].reshape(-1)
    parts=[]
    start=time.time()
    with torch.no_grad():
        for i in range(0,len(ids),args.batch_size):
            f,_=m.features(torch.tensor(ids[i:i+args.batch_size],dtype=torch.long,device=args.device),
                           torch.tensor(targets[i:i+args.batch_size],dtype=torch.long,device=args.device))
            if not torch.isfinite(f).all():raise FloatingPointError("Nonfinite features")
            parts.append(f.cpu().numpy())
            if i%(args.batch_size*32)==0:print(i,len(ids),time.time()-start,flush=True)
    features=np.concatenate(parts).reshape(len(tokens),8,-1)
    np.savez_compressed(out/"segments.npz",features=features)
    write_json(out/"manifest.json",{"schema":"model-rg-segments-v1","arguments":vars(args),
        "environment":environment(),"source_files":source_manifest(),"feature_names":m.feature_names(),
        "documents":len(tokens),"segments_per_document":8,"segment_tokens":64,
        "model_sha256":sha256(Path(args.model)/"model.safetensors"),
        "data_manifest_sha256":sha256(Path(args.data)/"manifest.json"),
        "segments_sha256":sha256(out/"segments.npz"),"seconds":time.time()-start,
        "max_gpu_memory_bytes":torch.cuda.max_memory_allocated(args.device)})


if __name__=="__main__":main()
