#!/usr/bin/env python
"""Fresh-chain validation of symmetry-based native magnetic collectives."""
from companion_paths import legacy_path
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
from prepare_lattice_data import generate,compile_sampler
from physical_mechanisms import observe
from analyze_physical_mechanisms import coordinates,regression,apply
from model_rg.training import TrainingModel
from model_rg.physical_native import spin_tokens
from model_rg.lattice import dyadic_block
from model_rg.provenance import sha256,write_json
ROOT=Path(legacy_path('/pldr-data/model'))
NAMES=['scripts/physical_collective_validation.py','scripts/prepare_lattice_data.py',
       'scripts/physical_mechanisms.py','scripts/analyze_physical_mechanisms.py',
       'scripts/potts_mc.cpp','src/model_rg/training.py','src/model_rg/native.py',
       'src/model_rg/physical_native.py','src/model_rg/lattice.py','src/model_rg/provenance.py']


def simplex_projection(rows):
    rows=np.asarray(rows,dtype=float)
    sorted_values=np.sort(rows,axis=1)[:,::-1]
    thresholds=(np.cumsum(sorted_values,axis=1)-1)/np.arange(1,rows.shape[1]+1)
    active=np.sum(sorted_values>thresholds,axis=1)-1
    return np.maximum(rows-thresholds[np.arange(len(rows)),active,None],0)


def prepare(base):
    base=Path(base).resolve()
    if not base.is_relative_to(ROOT):raise ValueError('Unauthorized destination')
    out=base/'collective-validation';out.mkdir(exist_ok=False)
    prior=json.loads((base/'mechanisms/protocol.json').read_text())
    qualification=json.loads((base/'data-qualification/verification.json').read_text())
    if qualification['status']!='passed':raise ValueError('Sampler not qualified')
    binary=compile_sampler(base)
    if sha256(binary)!=qualification['sampler_binary_sha256']:raise ValueError('Sampler binary changed')
    cells=[]
    for q in [2,3]:
        for L in [8,16]:
            c=generate(binary,out/f'data-q{q}-L{L}',q,L,1.,1960000+len(cells),16,64)
            c['id']=len(cells);cells.append(c)
    sources={n:sha256(REPO/n) for n in NAMES}
    for n in sources:
        dest=out/'executed-source'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,dest)
    write_json(out/'protocol.json',dict(status='frozen',schema='physical-invariant-validation-v1',
        sources=sources,cells=cells,cases=prior['cases'],endpoints=prior['endpoints'],
        sampler_qualification_sha256=sha256(base/'data-qualification/verification.json'),
        method='Fresh independent Monte Carlo seeds. Eight evenly spaced samples per chain; first eight chains calibrate up to eight PCA coordinates and ridge coefficients, last eight assess them. Predict all color fractions, project onto the probability simplex, then compute q/(q-1) times squared distance from uniform. Compare against a direct linear readout of m^2 using identical coordinates. Repeat after one random-tie 2x2 majority block.',
        inference='Native final proper prefix at the physical metadata for each fine or coarse configuration. The last spin is excluded from the model input.',
        role='The invariant construction was motivated by a separate finite readout study; this fresh source law was generated before inspecting these validation outcomes.'))
    print(out,flush=True)


def analyze_arrays(arrays,q,seed):
    chain=arrays['chain'];cal=chain<8;test=~cal;rows=[]
    rng=np.random.default_rng(seed)
    for depth in [0,1]:
        fractions=arrays[f'fractions-{depth}']
        vectors=np.sqrt(q/(q-1))*(fractions-1/q)
        target=np.square(vectors).sum(1)
        for family in ['hidden','A_common','G_common']:
            x=arrays[f'{family}-{depth}']
            if family=='hidden':x=x[:,-1]
            code,rank=coordinates(x,cal)
            fit=regression(code,fractions,cal)
            predicted_fraction=simplex_projection(apply(fit,code))
            predicted_vector=np.sqrt(q/(q-1))*(predicted_fraction-1/q)
            invariant=np.square(predicted_vector).sum(1)
            direct=apply(regression(code,target[:,None],cal),code)[:,0]
            error=np.linalg.norm(vectors-predicted_vector,axis=1)
            if np.any(np.abs(invariant-target)>2*error+1e-12):raise ValueError('Invariant error bound failed')
            denominator=np.mean((target[test]-target[cal].mean())**2)
            err_invariant=(invariant[test]-target[test])**2
            err_direct=(direct[test]-target[test])**2
            differences=(err_invariant-err_direct).reshape(8,8).mean(1)
            boots=differences[rng.integers(8,size=(2000,8))].mean(1)
            rows.append(dict(depth=depth,family=family,rank=rank,
                invariant_skill=float(1-err_invariant.mean()/denominator),
                linear_skill=float(1-err_direct.mean()/denominator),
                vector_rms_error=float(np.sqrt(np.mean(error[test]**2))),
                mean_m2_relative_error=float(invariant[test].mean()/target[test].mean()-1),
                squared_error_difference=float(differences.mean()),
                difference_bootstrap95=np.quantile(boots,[.025,.975]).tolist()))
            arrays[f'{family}-predicted-fractions-{depth}']=predicted_fraction
            arrays[f'{family}-invariant-{depth}']=invariant
            arrays[f'{family}-linear-m2-{depth}']=direct
    return rows


def worker(base,name,device):
    base=Path(base);study=base/'collective-validation';spec=json.loads((study/'protocol.json').read_text())
    if {n:sha256(REPO/n) for n in NAMES}!=spec['sources']:raise ValueError('Changed validation source')
    case=next(c for c in spec['cases'] if c['name']==name);endpoint=spec['endpoints'][name]
    if sha256(endpoint['state']['path'])!=endpoint['state']['sha256']:raise ValueError('Changed endpoint')
    out=study/name;out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    adapter=TrainingModel(source,case['heads'],case['seed'],device)
    state=torch.load(endpoint['state']['path'],map_location='cpu',weights_only=False)
    adapter.model.load_state_dict(state['model']);del state;gc.collect()
    adapter.model.requires_grad_(False).eval()
    processor=spm.SentencePieceProcessor(model_file=str(source/'tokenizer.model'));alphabet=spin_tokens(processor)
    rows=[]
    for c in spec['cells']:
        if sha256(c['path'])!=c['sha256']:raise ValueError('Changed fresh reference')
        raw=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
        x=np.asarray(raw[:,::8]).reshape(-1,c['L'],c['L'])
        arrays={'chain':np.repeat(np.arange(16),8)}
        for depth in [0,1]:
            arrays[f'configurations-{depth}']=x
            arrays[f'fractions-{depth}']=np.stack([(x==color).mean((1,2)) for color in range(c['q'])],1)
            collected={}
            for start in range(0,len(x),32):
                for k,v in observe(adapter,processor,alphabet,x[start:start+32],c['q'],1.,True).items():
                    collected.setdefault(k,[]).append(v)
            for k,values in collected.items():arrays[f'{k}-{depth}']=np.concatenate(values)
            if depth==0:x=dyadic_block(x,c['q'],1970000+c['id'])
        for row in analyze_arrays(arrays,c['q'],1980000+c['id']):rows.append(dict(case=name,q=c['q'],L=c['L'],**row))
        np.savez_compressed(out/f'q{c["q"]}-L{c["L"]}.npz',**arrays)
    files={p.name:sha256(p) for p in out.iterdir() if p.is_file()}
    write_json(out/'verification.json',dict(status='passed',case=case,rows=rows,files=files,
        protocol_sha256=sha256(study/'protocol.json'),producer_sha256=sha256(__file__),native_updates=0,
        scope='Fresh-chain finite prediction of a physical quadratic invariant. The fit has no identified internal RG eigenvalue or thermodynamic native exponent.'))
    print('fresh collective validation complete',name,flush=True)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--study',required=True)
    p=sub.add_parser('worker');p.add_argument('--study',required=True);p.add_argument('--name',required=True);p.add_argument('--device',required=True)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.study)
    else:worker(a.study,a.name,a.device)


if __name__=='__main__':main()
