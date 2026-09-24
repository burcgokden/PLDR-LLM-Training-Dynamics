#!/usr/bin/env python
"""Reexecute all primary boundary emissions and independently reduce row fields."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from model_rg.criticality import fix_shared_generator
from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();study=Path(a.study).resolve();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(out)
    start=time.time();inputs={};results=[];coordinate_count=0;logit_count=0
    def bind(path):inputs[str(path)]=sha256(path)
    bind(study/'protocol.json');spec=json.loads((study/'protocol.json').read_text())
    bind(study/'data/panels.npz')
    with np.load(study/'data/panels.npz') as f:crops=f['evaluation']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for case in spec['cases']:
        folder=study/'runs'/case['name'];bind(folder/'primary.npz');bind(folder/'incoming-state.pt')
        raw=np.load(folder/'primary.npz')
        model=TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1',case['heads'],case['seed'],a.device)
        normalize_variance_initialization(model);fix_shared_generator(model,case['generator_seed'])
        x=torch.as_tensor(crops[:,:64],dtype=torch.long,device=a.device)
        for boundary in ['initial','incoming']:
            if boundary=='incoming':
                saved=torch.load(folder/'incoming-state.pt',map_location='cpu',mmap=True,weights_only=True)
                assert saved['case']==case and saved['protocol_sha256']==inputs[str(study/'protocol.json')]
                model.model.load_state_dict(saved['model']);del saved
            model.model.eval()
            with torch.no_grad():
                emission=model.model(x,use_cache=False,logits_to_keep=1,output_pldr_attentions=True)
            logits=emission.logits[:,-1].cpu().numpy();expected=raw[boundary+'_logits']
            assert logits.dtype==expected.dtype and logits.shape==expected.shape and logits.tobytes()==expected.tobytes()
            logit_count+=logits.size;row=[]
            for decoder in emission.pldr_attentions:
                matrix=decoder[0].cpu().numpy().astype(np.float64)
                centered=matrix-matrix.mean(-2,keepdims=True)
                denominator=np.maximum(np.sum(matrix*matrix,axis=(-2,-1)),1e-30*matrix.shape[-2]*matrix.shape[-1])
                row.append(float(np.mean(np.sum(centered*centered,axis=(-2,-1))/denominator)))
                coordinate_count+=matrix.size
            error=float(np.max(abs(np.array(row)-raw[boundary+'_rows'])))
            assert error<1e-12
            results.append(dict(case=case['name'],boundary=boundary,rows=row,maximum_row_error=error,logits_bitwise=True))
            del emission,matrix,centered
        raw.close();del model,x;gc.collect();torch.cuda.empty_cache()
        print('Reconstructed row fields and boundary logits:',case['name'],flush=True)
    assert logit_count==16384000 and coordinate_count==94371840
    write_json(out,dict(status='passed',states=8,boundaries=16,decoder_fields=80,
        logit_coordinates=logit_count,matrix_coordinates_reexecuted=coordinate_count,
        results=results,device=a.device,inputs_sha256=inputs,verifier_sha256=sha256(__file__),seconds=time.time()-start,
        scope='All initial and incoming native boundary logits compared bitwise; row fields reexecuted and reduced independently in NumPy. No training updates or independent scientific observations.'))


if __name__=='__main__':main()
