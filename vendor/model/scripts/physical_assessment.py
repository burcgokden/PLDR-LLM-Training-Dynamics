#!/usr/bin/env python
"""Frozen proper-prefix, generation and metric-intervention assessment."""
from companion_paths import configured_path
import argparse
from contextlib import nullcontext
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import sentencepiece as spm
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.training import TrainingModel
from model_rg.physical_native import context_tokens,spin_tokens,prefix_batch,selected_forward,generate
from model_rg.inference_interventions import projected_rows
from model_rg.lattice import observables,summarize
from model_rg.provenance import sha256,write_json

ROOT=Path(configured_path('data:model'))
NAMES=['scripts/physical_assessment.py','src/model_rg/physical_native.py','src/model_rg/training.py',
       'src/model_rg/native.py','src/model_rg/inference_interventions.py','src/model_rg/lattice.py',
       'src/model_rg/provenance.py']


def source_hashes():return {n:sha256(REPO/n) for n in NAMES}


def prepare(study):
    study=Path(study).resolve();out=study/'assessment'
    if not study.is_relative_to(ROOT):raise ValueError('Unauthorized assessment destination')
    if out.exists():raise FileExistsError(out)
    training=study/'confirmation/protocol.json';spec=json.loads(training.read_text())
    data_path=study/'assessment-data/manifest.json';data=json.loads(data_path.read_text())
    if data['status']!='complete':raise ValueError('Reference data incomplete')
    cells=[c for c in data['cells'] if c['L'] in [4,6,8,12,16]]
    out.mkdir();sources=source_hashes()
    for n in sources:
        p=out/'executed-source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/n,p)
    protocol=dict(schema='physical-assessment-v1',status='frozen',created_at=datetime.now(timezone.utc).isoformat(),
        training_protocol=str(training),training_protocol_sha256=sha256(training),
        data_manifest=str(data_path),data_manifest_sha256=sha256(data_path),sources=sources,
        cases=spec['cases'],cells=cells,batch_size=64,
        times=[512,2048,spec['steps']],terminal_step=spec['steps'],
        generation='512 configurations per critical cell at the terminal state, 128 per off-critical terminal cell, and 128 per critical cell at earlier checkpoints.',
        observation='64 configurations per cell: four evenly spaced samples from each of 16 independent reference chains. Full spin-alphabet logits, layerwise hidden states and common metric rows at the final proper prefix are saved.',
        interventions='For body seed 640101 at each width: native, row-projected A, and all-head G multipliers 0.9 and 1.1. Critical L=8 and L=16, both classes; 256 generated configurations per intervention with common random draws.',
        statistics='Reference errors use independent-chain blocks. Generated samples are independent conditional on a checkpoint. Model-initialization replication and conditional Monte Carlo error are reported separately. All predeclared cells are retained.',
        scope='Physical L scaling, learned configuration-law error and finite native interventions. Two widths alone do not identify a native thermodynamic exponent. Temperature metadata are tokenized discrete settings; finite-temperature contrasts are not analytic derivatives of the tokenizer.')
    write_json(out/'protocol.json',protocol);print(out,flush=True)


def selected_samples(c):
    raw=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
    offsets=[0,128,256,384]
    x=np.asarray(raw[:,offsets]).reshape(-1,c['L'],c['L'])
    ids=np.array([(chain,offset) for chain in range(c['chains']) for offset in offsets])
    return x,ids


@torch.no_grad()
def measure(adapter,processor,alphabet,c,folder,full_capture=True):
    adapter.model.eval();x,ids=selected_samples(c);L=c['L'];q=c['q']
    meta=context_tokens(processor,q,L,c['temperature_ratio']);letters=alphabet[:q]
    arrays={'sample_ids':ids,'configurations':x};nll=[]
    for fraction in [.25,.5,.875,1.]:
        site=min(L*L-1,int(fraction*L*L));capture=full_capture and fraction==1.
        inputs=prefix_batch(x,site,meta,letters,adapter.device)
        logits,out=selected_forward(adapter,inputs,letters,capture=capture)
        targets=torch.as_tensor(x.reshape(len(x),-1)[:,site].astype('int64'),device=adapter.device)
        loss=torch.nn.functional.cross_entropy(logits,targets,reduction='none')
        key=f'j{site}';arrays[key+'-logits']=logits.cpu().numpy();arrays[key+'-targets']=targets.cpu().numpy()
        arrays[key+'-nll']=loss.cpu().numpy();nll.append(float(loss.mean()))
        if capture:
            arrays['hidden']=torch.stack([v[:,-1] for v in out.hidden_states],1).cpu().numpy()
            common=[];grow=[];fractions=[];headnorm=[]
            for layer,att in enumerate(out.pldr_attentions):
                a=att[0].double();g=att[5].double();mean=a.mean(-2)
                common.append(mean);grow.append(g.mean(-2))
                fractions.append((a-mean.unsqueeze(-2)).square().sum((-2,-1))/a.square().sum((-2,-1)).clamp_min(1e-300))
                headnorm.append(adapter.head_outputs[layer])
            arrays['A_common']=torch.stack(common,1).cpu().numpy()
            arrays['G_common']=torch.stack(grow,1).cpu().numpy()
            arrays['row_fraction']=torch.stack(fractions,1).cpu().numpy()
            arrays['headnorm']=torch.stack(headnorm,1).cpu().numpy()
    adapter.capture=False;adapter.head_outputs={}
    if not all(np.isfinite(v).all() for v in arrays.values()):raise FloatingPointError('Nonfinite physical observation')
    p=folder/f'cell-{c["id"]:03d}-observation.npz';np.savez_compressed(p,**arrays)
    return dict(cell=c['id'],q=q,L=L,ratio=c['temperature_ratio'],nll_by_prefix=nll,
                file=p.name,sha256=sha256(p),sample_count=len(x))


def rich_summary(x,q):
    result=summarize(x,q)
    fractions=np.stack([(x==a).mean((1,2)) for a in range(q)],1)
    mean=fractions.mean(0)
    result['mean_color_fractions']=mean.tolist()
    result['connected_chi']=float(x.shape[-1]**2*q/(q-1)*np.square(fractions-mean).sum(1).mean())
    return result


def worker(study,name,device):
    study=Path(study);spec=json.loads((study/'protocol.json').read_text());base=study.parent
    if source_hashes()!=spec['sources']:raise ValueError('Assessment source changed')
    if sha256(spec['data_manifest'])!=spec['data_manifest_sha256']:raise ValueError('Assessment data changed')
    if sha256(spec['training_protocol'])!=spec['training_protocol_sha256']:raise ValueError('Training protocol changed')
    case=next(c for c in spec['cases'] if c['name']==name)
    trained=base/'confirmation'/name
    manifest=json.loads((trained/'manifest.json').read_text())
    if manifest['status']!='complete' or manifest['completed_updates']!=spec['terminal_step']:
        raise ValueError('Complete fine-tuning required')
    out=study/name;out.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    processor=spm.SentencePieceProcessor(model_file=str(source/'tokenizer.model'))
    alphabet=spin_tokens(processor);adapter=TrainingModel(source,case['heads'],case['seed'],device)
    adapter.model.requires_grad_(False).eval();torch.cuda.reset_peak_memory_stats(device)
    checkpoints=[];start=time.perf_counter()
    for step in spec['times']:
        checkpoint=next(c for c in manifest['checkpoints'] if c['step']==step)
        if sha256(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Changed checkpoint')
        state=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
        adapter.model.load_state_dict(state['model']);del state;gc.collect()
        folder=out/f'time-{step:05d}';folder.mkdir();records=[];observations=[]
        cells=spec['cells'] if step==spec['terminal_step'] else [c for c in spec['cells'] if c['temperature_ratio']==1.]
        for c in cells:
            if sha256(c['path'])!=c['sha256']:raise ValueError('Changed reference configurations')
            observation=measure(adapter,processor,alphabet,c,folder,full_capture=True)
            observations.append(observation)
            count=512 if step==spec['terminal_step'] and c['temperature_ratio']==1. else 128
            seed=1800000+case['seed']*100+c['id']
            x=generate(adapter,context_tokens(processor,c['q'],c['L'],c['temperature_ratio']),
                alphabet[:c['q']],c['L'],count,seed,batch_size=spec['batch_size'])
            p=folder/f'cell-{c["id"]:03d}-generated.npy';np.save(p,x)
            records.append(dict(cell=c['id'],q=c['q'],L=c['L'],ratio=c['temperature_ratio'],
                count=count,seed=seed,path=str(p),sha256=sha256(p),summary=rich_summary(x,c['q'])))
            print(name,'assessment',step,'q',c['q'],'L',c['L'],'r',c['temperature_ratio'],flush=True)
        write_json(folder/'results.json',dict(step=step,generated=records,observations=observations))
        checkpoints.append(dict(step=step,input_checkpoint=checkpoint,results=str(folder/'results.json'),sha256=sha256(folder/'results.json')))
    interventions=[]
    if case['seed']==640101 and case['arm']=='full':
        folder=out/'interventions';folder.mkdir()
        for c in spec['cells']:
            if c['L'] not in [8,16] or c['temperature_ratio']!=1.:continue
            configs,ids=selected_samples(c);meta=context_tokens(processor,c['q'],c['L'],1.);letters=alphabet[:c['q']]
            inputs=prefix_batch(configs,c['L']**2-1,meta,letters,device)
            baseline,_=selected_forward(adapter,inputs,letters)
            baseline=baseline.detach()
            for mode in ['native','row-projected','G0.9','G1.1']:
                eta=None if mode in ['native','row-projected'] else torch.full((5,case['heads']),-.1 if mode=='G0.9' else .1,device=device)
                context=projected_rows(adapter) if mode=='row-projected' else nullcontext()
                with context:
                    logits,_=selected_forward(adapter,inputs,letters,eta=eta)
                    lp=baseline.log_softmax(-1);lq=logits.log_softmax(-1)
                    kl=(lp.exp()*(lp-lq)).sum(-1)
                    x=generate(adapter,meta,letters,c['L'],256,1900000+c['id'],batch_size=spec['batch_size'],eta=eta)
                p=folder/f'cell-{c["id"]:03d}-{mode}.npy';np.save(p,x)
                interventions.append(dict(cell=c['id'],q=c['q'],L=c['L'],mode=mode,path=str(p),sha256=sha256(p),
                    max_prefix_logit_difference=float((logits-baseline).abs().max()),
                    mean_prefix_kl=float(kl.mean()),summary=rich_summary(x,c['q'])))
        write_json(folder/'results.json',interventions)
    write_json(out/'manifest.json',dict(status='complete',case=case,checkpoints=checkpoints,
        interventions=interventions,protocol_sha256=sha256(study/'protocol.json'),
        trained_manifest_sha256=sha256(trained/'manifest.json'),runtime_seconds=time.perf_counter()-start,
        peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
    print('assessment complete',name,flush=True)


def main():
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--study',required=True)
    p=sub.add_parser('worker');p.add_argument('--study',required=True);p.add_argument('--name',required=True);p.add_argument('--device',required=True)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.study)
    else:worker(a.study,a.name,a.device)


if __name__=='__main__':main()
