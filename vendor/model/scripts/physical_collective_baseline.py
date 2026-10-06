#!/usr/bin/env python
"""Matched incoming and random-initialization controls for physical readouts."""
from companion_paths import configured_path
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import sentencepiece as spm
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from physical_mechanisms import observe
from physical_collective_validation import analyze_arrays
from model_rg.training import TrainingModel
from model_rg.physical_native import spin_tokens
from model_rg.lattice import dyadic_block
from model_rg.provenance import sha256,write_json
ROOT=Path(configured_path('data:model'))
NAMES=['scripts/physical_collective_baseline.py','scripts/physical_collective_validation.py',
       'scripts/physical_mechanisms.py','scripts/analyze_physical_mechanisms.py',
       'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/physical_native.py',
       'src/model_rg/lattice.py','src/model_rg/provenance.py']


def prepare(base):
    base=Path(base).resolve()
    if not base.is_relative_to(ROOT):raise ValueError('Unauthorized destination')
    out=base/'collective-baseline';out.mkdir(exist_ok=False)
    reference=json.loads((base/'collective-validation/protocol.json').read_text())
    sources={n:sha256(REPO/n) for n in NAMES}
    for n in sources:
        p=out/'executed-source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p)
    write_json(out/'protocol.json',dict(status='frozen',schema='physical-readout-origins-v1',
        sources=sources,cases=reference['cases'],cells=reference['cells'],origins=['random','pretrained'],
        reference_protocol_sha256=sha256(base/'collective-validation/protocol.json'),
        role='Additional descriptive controls on the same already measured fresh physical cohort. Readout specification, calibration chains and held-out chains match the adapted endpoints; native weights are never fitted in this control. The comparison is not a new independent confirmation cohort.'))
    print(out,flush=True)


def worker(base,heads,device):
    base=Path(base);study=base/'collective-baseline';spec=json.loads((study/'protocol.json').read_text())
    if {n:sha256(REPO/n) for n in NAMES}!=spec['sources']:raise ValueError('Changed origin-control source')
    case=next(c for c in spec['cases'] if c['heads']==heads)
    source=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    torch.set_num_threads(4);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    processor=spm.SentencePieceProcessor(model_file=str(source/'tokenizer.model'));alphabet=spin_tokens(processor)
    for origin in spec['origins']:
        out=study/f'h{heads}-{origin}';out.mkdir(exist_ok=False)
        adapter=TrainingModel(source,heads,case['seed'],device)
        if origin=='pretrained':
            if sha256(case['state'])!=case['state_sha256']:raise ValueError('Incoming state changed')
            state=torch.load(case['state'],map_location='cpu',weights_only=False)
            adapter.model.load_state_dict(state['model']);del state;gc.collect()
        adapter.model.requires_grad_(False).eval();rows=[]
        for c in spec['cells']:
            if sha256(c['path'])!=c['sha256']:raise ValueError('Changed matched physical cohort')
            raw=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
            x=np.asarray(raw[:,::8]).reshape(-1,c['L'],c['L']);arrays={'chain':np.repeat(np.arange(16),8)}
            for depth in [0,1]:
                arrays[f'configurations-{depth}']=x
                arrays[f'fractions-{depth}']=np.stack([(x==a).mean((1,2)) for a in range(c['q'])],1)
                collected={}
                for start in range(0,len(x),32):
                    for k,v in observe(adapter,processor,alphabet,x[start:start+32],c['q'],1.,True).items():
                        collected.setdefault(k,[]).append(v)
                for k,values in collected.items():arrays[f'{k}-{depth}']=np.concatenate(values)
                if depth==0:x=dyadic_block(x,c['q'],1970000+c['id'])
            for r in analyze_arrays(arrays,c['q'],1980000+c['id']):rows.append(dict(heads=heads,origin=origin,q=c['q'],L=c['L'],**r))
            np.savez_compressed(out/f'q{c["q"]}-L{c["L"]}.npz',**arrays)
        write_json(out/'verification.json',dict(status='passed',case=case,origin=origin,rows=rows,
            files={p.name:sha256(p) for p in out.iterdir() if p.is_file()},
            protocol_sha256=sha256(study/'protocol.json'),producer_sha256=sha256(__file__),native_updates=0))
        print('complete origin control',heads,origin,flush=True)
        del adapter;gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--study',required=True)
    p=sub.add_parser('worker');p.add_argument('--study',required=True);p.add_argument('--heads',type=int,required=True);p.add_argument('--device',required=True)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.study)
    else:worker(a.study,a.heads,a.device)


if __name__=='__main__':main()
