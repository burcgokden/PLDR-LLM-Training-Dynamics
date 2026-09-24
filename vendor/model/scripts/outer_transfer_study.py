#!/usr/bin/env python
"""Freeze and execute a bounded corpus/full-initialization transfer experiment.

Subcommands protect completed outputs. Scientific settings are recorded before
training. Forecasts use only the archive or the designated adaptation paths.
"""
from companion_paths import child_pythonpath
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from model_rg.criticality import fix_shared_generator, generator_parameter
from model_rg.inference_interventions import fixed_operators, operator_cache, projected_rows
from model_rg.moment_rg import forecast, moments
from model_rg.onepass_regimes import optimizer_and_scheduler, recipe
from model_rg.provenance import sha256, write_json
from model_rg.schedules import loss as native_loss, clip as native_clip
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization
from measure_law_closure import digest_state

HORIZONS=[0,1,4,16,64]


def now(): return datetime.now(timezone.utc).isoformat()


def freeze(root, study, repo):
    path=study/'protocol.json'
    if path.exists(): raise FileExistsError(path)
    data=json.loads((study/'data/manifest.json').read_text()); assert data['status']=='complete'
    inputs={str(study/'data'/name):digest for name,digest in data['files'].items()}
    inputs[str(study/'data/manifest.json')]=sha256(study/'data/manifest.json')
    old=root/'moment-transport-20260911/frozen-predictions.json'
    archive=json.loads(old.read_text()); inputs[str(old)]=sha256(old)
    fits={}
    for heads in [4,14]:
        states=[s for name,s in archive['fits'].items() if name.startswith(f'h{heads}-')]
        assert len(states)==4
        fits[str(heads)]={}
        for target in ['risk','risk_entropy']:
            covariances=[np.array(s['targets'][target]['covariance']) for s in states]
            correlations=[c/np.outer(np.sqrt(np.diag(c)),np.sqrt(np.diag(c))) for c in covariances]
            corr=np.mean(correlations,axis=0)
            mean=np.mean([s['targets'][target]['mean'] for s in states],axis=0)
            fits[str(heads)][target]=dict(correlation=corr.tolist(),mean=mean.tolist(),
                archive_covariance=np.mean(covariances,axis=0).tolist())
    write_json(study/'archive-forecast.json',dict(status='frozen',frozen_at=now(),fits=fits,archive_sha256=sha256(old)))
    inputs[str(study/'archive-forecast.json')]=sha256(study/'archive-forecast.json')
    sources=['scripts/outer_transfer_study.py','scripts/measure_law_closure.py',
        'src/model_rg/native.py','src/model_rg/training.py','src/model_rg/criticality.py',
        'src/model_rg/controlled.py','src/model_rg/onepass_regimes.py',
        'src/model_rg/scheduled_regimes.py','src/model_rg/schedules.py',
        'src/model_rg/variance_family.py','src/model_rg/moment_rg.py',
        'src/model_rg/inference_interventions.py','src/model_rg/provenance.py']
    native=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    for name in ['modeling_pldrllm.py','configuration_pldrllm.py']:
        inputs[str(native/name)]=sha256(native/name)
    cases=[]
    for heads in [4,14]:
        for corpus in range(2):
            for identity in range(2):
                cases.append(dict(name=f'h{heads}-c{corpus}-i{identity}',heads=heads,corpus=corpus,
                    identity=identity,seed=9110101+100*corpus+identity,
                    generator_seed=9120101+100*corpus+identity,
                    stream_seed=9130101+corpus,branch_seed=9140101+100*corpus+identity,
                    profile=recipe('controlled',heads,0)))
    write_json(path,dict(schema='outer-transfer-protocol-v1',status='frozen',frozen_at=now(),
        cases=cases,primary_steps=2048,horizons=HORIZONS,batch_size=32,
        branches=dict(adaptation=16,calibration=8,assessment=32),qualification_steps=256,
        normalization='Shape-aware variance; independent body and shared-metric seeds in each corpus/identity. Metric draw paired across widths. Both parameter families train.',
        resource='65536 documents per corpus; eight disjoint context blocks per crop; 524288 block indices. Each path uses a prefix of a uniform permutation, followed by a without-replacement sample of remaining indices. Counterfactual paths may overlap.',
        observation='32 fixed new evaluation documents, 64-token proper prefixes, external target risk and full-vocabulary entropy. Frozen-operator donors are 32 separate new documents.',
        forecasts='Zero-branch no-change and archived mean increment. Separately, 16 adaptation branches estimate mean and diagonal scale, comparing local shrinkage, diagonal, and archived correlation shape under equal adaptation budget. All covariances regularized at fine resolution then transported.',
        regularization=dict(shrinkage=.25,floor=.001,standard_deviation_floor=1e-6),
        calibration='Eight independent calibration paths set the maximum of whole-risk-path standardized errors about the adaptation mean, with scales from the transported local covariance. Assessment is never fitted or calibrated. Conditional rank coverage at least 8/9, averaged over calibration.',
        reporting='Both targets, three temporal maps, every state and candidate. Within-state paired Monte Carlo standard errors; no independent-head or branch-as-model inference. Two corpus draws and four nested full-initialization identities are a pilot.',
        primary='Four-risk-increment paired scores, transferred correlation minus diagonal and local shrinkage minus diagonal. Mean-risk path errors and calibrated whole-path coverage reported separately.',
        inference_prefixes=[1,4,16,64],
        sources={name:sha256(repo/name) for name in sources},inputs=inputs))
    print('Frozen 8 training paths and 448 conditional continuations; 45056 scientific updates.',flush=True)


def check(spec,repo):
    for name,digest in spec['sources'].items():
        assert sha256(repo/name)==digest, 'Changed frozen source: '+name
    for name,digest in spec['inputs'].items():
        assert sha256(name)==digest, 'Changed frozen input: '+name


def component_digests(model):
    digests={name:hashlib.sha256() for name in ['body','generator','metric']}
    for name,p in model.model.named_parameters():
        key='generator' if generator_parameter(name) else 'body'
        raw=p.detach().cpu().contiguous().numpy().tobytes()
        digests[key].update(name.encode());digests[key].update(raw)
        if 'reslayerAs' in name:
            digests['metric'].update(name.encode());digests['metric'].update(raw)
    return {key:h.hexdigest() for key,h in digests.items()}


def observe(model,crops):
    model.model.eval()
    with torch.no_grad():
        x=torch.as_tensor(crops,dtype=torch.long,device=model.device)
        out=model.forward(x[:,:64],capture=True)
        logits=out.logits[:,-1].double(); logp=logits.log_softmax(-1)
        nll=-logp.gather(1,x[:,64,None]).squeeze(1)
        entropy=-(logp.exp()*logp).sum(-1)
        rows=[]
        for layer in out.pldr_attentions:
            mat=layer[0].double(); center=mat.mean(-2,keepdim=True)
            rows.append(((mat-center).square().mean((-2,-1))/mat.square().mean((-2,-1)).clamp_min(1e-30)).mean())
        ans=dict(nll=nll.cpu().numpy(),entropy=entropy.cpu().numpy(),rows=torch.stack(rows).cpu().numpy(),
                 logits=logits.float().cpu().numpy())
    model.capture=False; model.head_outputs={}
    return ans


def fits_from_paths(paths,archived):
    fitted={}
    for count,target in [(1,'risk'),(2,'risk_entropy')]:
        x=np.diff(paths[:,:,:count],axis=1).transpose(0,2,1).reshape(len(paths),-1)
        f=forecast(x);scale=np.array(f['scale']);outer=np.outer(scale,scale)
        corr=np.array(archived[target]['correlation'])
        transferred=(.75*corr+.25*np.diag(np.diag(corr))+.001*np.eye(len(corr)))*outer
        fitted[target]=dict(mean=f['mean'],scale=f['scale'],
            covariances=dict(local=(np.array(f['regularized'])*outer).tolist(),
                             diagonal=(np.array(f['diagonal'])*outer).tolist(),
                             transferred=transferred.tolist()))
    return fitted


def worker(root,study,repo,spec,case,device,qualification=False):
    check(spec,repo)
    out=study/('qualification' if qualification else 'runs')/case['name']
    out.mkdir(parents=True,exist_ok=False);start=time.time()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model=TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1',case['heads'],case['seed'],device)
    normalize_variance_initialization(model);fix_shared_generator(model,case['generator_seed'])
    initial=component_digests(model)
    optimizer,scheduler=optimizer_and_scheduler(model,case['profile'])
    corpus=np.load(study/'data'/f"corpus-{case['corpus']}.npy",mmap_mode='r')
    with np.load(study/'data/panels.npz') as f: crops=f['evaluation'];donors=f['donors']
    order=np.random.default_rng(case['stream_seed']).permutation(len(corpus)*8)
    steps=spec['qualification_steps'] if qualification else spec['primary_steps']
    used=order[:32*steps].reshape(-1,32);remaining=order[32*steps:]
    def batch(blocks):
        arr=corpus[(blocks//8)[:,None],(64*(blocks%8))[:,None]+np.arange(65)]
        return torch.as_tensor(arr,dtype=torch.long,device=model.device)
    def update(blocks):
        model.model.train();optimizer.zero_grad(set_to_none=True)
        loss=native_loss(model,batch(blocks),case['profile'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();native_clip(model,case['profile']);optimizer.step();scheduler.step()
        return float(loss.detach())
    base=observe(model,crops)
    # An independent direct full-vocabulary CE checks the training adapter before
    # any updates, including all gradients. It is qualification work, not a sample.
    gradient_check=None
    if qualification:
        model.model.train();optimizer.zero_grad(set_to_none=True)
        b=batch(used[0]);l1=native_loss(model,b,case['profile']);l1.backward()
        first={n:p.grad.detach().cpu().clone() for n,p in model.model.named_parameters() if p.grad is not None}
        optimizer.zero_grad(set_to_none=True)
        logits=model.model(b[:,:64],use_cache=False,logits_to_keep=1).logits[:,-1]
        l2=torch.nn.functional.cross_entropy(logits,b[:,64]);l2.backward()
        assert l1.detach().cpu().numpy().tobytes()==l2.detach().cpu().numpy().tobytes()
        assert all(torch.equal(first[n],p.grad.detach().cpu()) for n,p in model.model.named_parameters() if p.grad is not None)
        gradient_check=dict(loss_bitwise=True,all_gradients_bitwise=True,parameters=len(first))
        del first,logits,l1,l2;optimizer.zero_grad(set_to_none=True)
    losses=[]
    for t,blocks in enumerate(used,1):
        losses.append(update(blocks))
        if t%256==0:print(case['name'],'primary',t,'elapsed',round(time.time()-start,1),flush=True)
    incoming=observe(model,crops)
    np.savez_compressed(out/'primary.npz',blocks=used,losses=losses,
        initial_nll=base['nll'],incoming_nll=incoming['nll'],initial_entropy=base['entropy'],
        incoming_entropy=incoming['entropy'],initial_rows=base['rows'],incoming_rows=incoming['rows'],
        initial_logits=base['logits'],incoming_logits=incoming['logits'])
    checkpoint=dict(model={n:p.detach().cpu().clone() for n,p in model.model.state_dict().items()},
        optimizer=copy.deepcopy(optimizer.state_dict()),scheduler=copy.deepcopy(scheduler.state_dict()),
        step=steps,case=case,protocol_sha256=sha256(study/'protocol.json'))
    # CPU copies also prevent optimizer state tensors from aliasing a live branch.
    for state in checkpoint['optimizer']['state'].values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor):state[key]=value.cpu().clone()
    torch.save(checkpoint,out/'incoming-state.pt')
    def restore():
        optimizer.zero_grad(set_to_none=True);model.capture=False;model.eta=None;model.head_outputs={}
        model.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        scheduler.load_state_dict(copy.deepcopy(checkpoint['scheduler']))
    incoming_digest=digest_state(model,optimizer,scheduler)
    horizons=[0,1,4] if qualification else spec['horizons']
    total=2 if qualification else sum(spec['branches'].values())
    rng=np.random.default_rng(case['branch_seed'])
    blocks=np.stack([rng.choice(remaining,32*max(horizons),replace=False).reshape(-1,32) for _ in range(total)])
    np.savez_compressed(out/'sampling.npz',primary_blocks=used,branch_blocks=blocks,evaluation_crops=crops)
    paths=[];records=[];first_final=None
    for b in range(total+1):
        replay=b==total;j=0 if replay else b
        restore();assert digest_state(model,optimizer,scheduler)==incoming_digest
        observations=[observe(model,crops)];trace=[]
        for t,ids in enumerate(blocks[j],1):
            trace.append(update(ids))
            if t in horizons:observations.append(observe(model,crops))
        arrays={key:np.stack([o[key] for o in observations]) for key in observations[0]}
        arrays.update(horizons=np.array(horizons),losses=np.array(trace))
        assert all(np.isfinite(x).all() for x in arrays.values())
        final=digest_state(model,optimizer,scheduler)
        if replay:
            assert final==first_final
            with np.load(out/'branch-00.npz') as f:
                assert all(arrays[k].dtype==f[k].dtype and arrays[k].shape==f[k].shape and arrays[k].tobytes()==f[k].tobytes() for k in arrays)
        else:
            if b==0:first_final=final
            paths.append(np.stack([arrays['nll'].mean(1),arrays['entropy'].mean(1)],axis=1))
            # Full vocabulary panels for every branch permit independent reductions.
            filename=out/f'branch-{b:02d}.npz';np.savez_compressed(filename,**arrays)
            role='qualification' if qualification else 'adaptation' if b<16 else 'calibration' if b<24 else 'assessment'
            records.append(dict(branch=b,role=role,raw=filename.name,sha256=sha256(filename),final_state_sha256=final))
            if not qualification and b==15:
                archived=json.loads((study/'archive-forecast.json').read_text())['fits'][str(case['heads'])]
                write_json(out/'adapted-forecast.json',dict(status='frozen',frozen_at=now(),
                    fits=fits_from_paths(np.array(paths),archived),inputs={r['raw']:r['sha256'] for r in records},
                    protocol_sha256=sha256(study/'protocol.json')))
            if not qualification and b==23:
                fit=json.loads((out/'adapted-forecast.json').read_text())['fits']['risk']
                mean=np.cumsum(fit['mean']);summation=np.tril(np.ones((4,4)))
                covariance=np.array(fit['covariances']['local'])
                scale=np.sqrt(np.maximum(np.diag(summation@covariance@summation.T),1e-12))
                values=np.array(paths)[16:24,1:,0]-np.array(paths)[16:24,0,None,0]
                scores=np.max(np.abs(values-mean)/scale,axis=1)
                write_json(out/'calibration.json',dict(status='frozen',frozen_at=now(),radius=float(scores.max()),
                    scale=scale.tolist(),mean=mean.tolist(),scores=scores.tolist(),
                    fit_sha256=sha256(out/'adapted-forecast.json'),inputs={r['raw']:r['sha256'] for r in records[16:24]}))
        print(case['name'],'replay' if replay else f'branch {b+1}/{total}',round(time.time()-start,1),'seconds',flush=True)
    restore();restored=observe(model,crops)
    assert all(restored[k].tobytes()==incoming[k].tobytes() for k in restored)
    inference={}
    if not qualification:
        model.model.eval()
        with torch.no_grad():
            donor=torch.as_tensor(donors,dtype=torch.long,device=model.device)
            donor_out=model.forward(donor[:,:64],capture=True)
            cache=operator_cache(donor_out).mean(2,keepdim=True);del donor_out
            np.save(out/'fixed-operators.npy',cache.cpu().numpy())
            x=torch.as_tensor(crops,dtype=torch.long,device=model.device)
            for length in spec['inference_prefixes']:
                native=model.logits(x[:,:length]).detach().clone()
                with fixed_operators(model,cache):frozen=model.logits(x[:,:length]).detach().clone()
                with projected_rows(model):projected=model.logits(x[:,:length]).detach().clone()
                replay=model.logits(x[:,:length]);assert torch.equal(native,replay)
                filename=out/f'inference-{length}.npz'
                np.savez_compressed(filename,native=native.cpu().numpy(),fixed=frozen.cpu().numpy(),
                    projected=projected.cpu().numpy(),targets=crops[:,length])
                inference[str(length)]=dict(raw=filename.name,sha256=sha256(filename))
    result=dict(status='complete',case=case,started_at=datetime.fromtimestamp(start,timezone.utc).isoformat(),
        completed_at=now(),seconds=time.time()-start,initial_component_sha256=initial,
        protocol_sha256=sha256(study/'protocol.json'),incoming_state_sha256=sha256(out/'incoming-state.pt'),
        incoming_digest=incoming_digest,restoration_bitwise=True,replay_bitwise=True,
        replay_updates=max(horizons),qualification_updates=steps+total*max(horizons) if qualification else 0,
        scientific_primary_updates=0 if qualification else steps,
        scientific_branch_updates=0 if qualification else total*max(horizons),
        records=records,inference=inference,gradient_qualification=gradient_check,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
        files={p.name:sha256(p) for p in out.iterdir() if p.is_file() and not p.name.startswith('branch-')})
    write_json(out/'results.json',result)


def launch(root,study,repo,spec,qualification):
    target=study/('qualification-launcher.json' if qualification else 'launcher.json')
    if target.exists():raise FileExistsError(target)
    if not qualification:assert json.loads((study/'qualification-launcher.json').read_text())['status']=='complete'
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    def lane(heads,device):
        selected=[c for c in spec['cases'] if c['heads']==heads]
        if qualification:selected=selected[:1]
        records=[]
        for c in selected:
            label=c['name']+('-qualification' if qualification else '')
            command=[sys.executable,str(repo/'scripts/outer_transfer_study.py'),'worker','--root',str(root),'--study',str(study),'--case',c['name'],'--device',device]
            if qualification:command+=['--qualification']
            started=now()
            with (study/(label+'.log')).open('x') as log:subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=study/('qualification' if qualification else 'runs')/c['name']/'results.json'
            records.append(dict(case=c['name'],started_at=started,command=command,result=str(result),sha256=sha256(result)))
            print('Completed',label,flush=True)
        return records
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,4,'cuda:0'),pool.submit(lane,14,'cuda:1')]
        records=[x for f in futures for x in f.result()]
    write_json(target,dict(status='complete',completed_at=now(),records=records,protocol_sha256=sha256(study/'protocol.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','qualify','run','worker'])
    p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--case');p.add_argument('--device');p.add_argument('--qualification',action='store_true');a=p.parse_args()
    root=Path(a.root).resolve();study=Path(a.study).resolve();repo=Path(__file__).resolve().parents[1]
    if a.phase=='freeze':freeze(root,study,repo);return
    spec=json.loads((study/'protocol.json').read_text());check(spec,repo)
    if a.phase in ['qualify','run']:launch(root,study,repo,spec,a.phase=='qualify')
    else:worker(root,study,repo,spec,next(c for c in spec['cases'] if c['name']==a.case),a.device,a.qualification)


if __name__=='__main__':main()
