#!/usr/bin/env python
"""Frozen paired native response on single-pass remaining-corpus continuations.

The experiment estimates parameter-direction secants of complete-state evolution.
It never interprets a finite pulse as a full augmented Jacobian or spectral gap.
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
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from model_rg.criticality import generator_parameter
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.provenance import sha256, write_json
from model_rg.schedules import loss as native_loss, clip as native_clip
from model_rg.training import TrainingModel
from model_rg.qualification import admit, freeze_contract, source_names, validate_qualification
from measure_law_closure import digest_state


def now():
    return datetime.now(timezone.utc).isoformat()


def arms():
    return [dict(name='zero', family=None, multiplier=0.)]+[
        dict(name=f'{family}-{label}', family=family, multiplier=multiplier)
        for family in ['generator', 'body']
        for label, multiplier in [('plus', 1.), ('minus', -1.), ('halfplus', .5), ('halfminus', -.5)]]


def prepare(root, study, repo, kind, amplitude):
    qualified = None if kind == 'qualification' else validate_qualification(study.parent/'qualification/protocol.json', repo, 'P3')
    study.mkdir(parents=True, exist_ok=False)
    old = root/'outer-transfer-20260911'
    prior = json.loads((old/'protocol.json').read_text())
    cases = prior['cases']
    horizon, sources = (8, 2) if kind == 'qualification' else ((128, 1) if kind == 'development' else (128, 4))
    if kind == 'qualification':
        cases = [c for c in cases if c['corpus'] == 0 and c['identity'] == 0]
    inputs = {str(old/p): sha256(old/p) for p in ['protocol.json', 'data/manifest.json',
        'data-verification.json', 'data/panels.npz', 'data/corpus-0.npy', 'data/corpus-1.npy']}
    if kind == 'confirmation':
        d = study.parent/'development/verification.json'
        assert json.loads(d.read_text())['status'] == 'passed', 'Independent development required'
        inputs[str(d)] = sha256(d)
        # A finite secant domain can be tested even when a derivative gate fails.
        # Its outcomes must remain visible, rather than changing the gate.
    (study/'sampling').mkdir()
    for case in cases:
        parent = old/'runs'/case['name']
        for name in ['incoming-state.pt', 'results.json']:
            inputs[str(parent/name)] = sha256(parent/name)
        assert inputs[str(parent/'incoming-state.pt')] == json.loads((parent/'results.json').read_text())['incoming_state_sha256']
        order = np.random.default_rng(case['stream_seed']).permutation(524288)
        used, remaining = order[:65536], order[65536:]
        seed = 121200000 + {'qualification': 0, 'development': 1000000, 'confirmation': 2000000}[kind] + case['heads']*1000 + case['corpus']*100 + case['identity']
        rng = np.random.default_rng(seed)
        blocks = np.stack([rng.choice(remaining, 32*horizon, replace=False).reshape(horizon, 32) for _ in range(sources)])
        file = study/'sampling'/(case['name']+'.npz')
        np.savez(file, primary=used, blocks=blocks, seed=seed)
        inputs[str(file)] = sha256(file)
    native = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    for name in ['modeling_pldrllm.py', 'configuration_pldrllm.py']:
        inputs[str(native/name)] = sha256(native/name)
    files = source_names(repo, 'P3', set(prior['sources']) | {'scripts/directional_study.py',
        'scripts/analyze_directional.py', 'scripts/verify_directional.py'})
    hashes = {name: sha256(repo/name) for name in files}
    for name in files:
        target = study/'executed-source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, target)
    write_json(study/'executed-source/manifest.json', dict(sources=hashes))
    times = sorted(set([0, 1, 2]+list(range(4, horizon+1, 4))))
    spec = dict(schema='onepass-directional-v1', status='frozen',
        frozen_at=now(), kind=kind, amplitude=amplitude, cases=cases, arms=arms(),
        horizon=horizon, times=times, source_replicates=sources, batch_size=32,
        direction='Negative external-target loss gradient on the fixed document-excluded donor panel, separately normalized within generator and body. Relative displacement h times incoming family parameter norm. Both moments, counters, scheduler and remaining corpus held fixed at pulse.',
        observation='Fixed 32 document-excluded 64-token proper prefixes, full vocabulary risk/entropy; log1p common and transverse metric RMS at all five layers. Unit coordinate scales.',
        primary='Paired full-versus-half amplitude response discrepancy for both directions at every state/source over the complete observation path; signal floor and 10% relative domain gate fixed before outcomes.',
        secondary='Finite-pulse temporal response; exact signed covariance and chronological endpoint/block identities; forcing-only diagonal control. These are conditional empirical identities, not population forecasts or identified gaps.',
        gate=dict(relative_tolerance=.1, absolute_floor=1e-7, signal_multiple=5.,
                  convention='Equal weight for every saved time in {0,1,2} union {4j: 1<=j<=horizon/4} and each of 12 coordinates, norm squared sum/(12*number_of_times); small-amplitude signed displacement must exceed five times the fixed absolute floor. The floor is an engineering resolution criterion; zero-arm bitwise replay checks reproducibility separately.'),
        resource='Each arm uses 65536 consumed prefix blocks and 32*horizon new distinct blocks. All nine arms in a source replicate share exactly the same without-replacement indices. Arms and nested states are not independent pretrained models.',
        predictor='No fitted response matrix or spectral model. All four source replicates assess a fixed secant approximation and its signed empirical covariance. Development uses separate source seeds.',
        scientific_updates=0 if kind=='qualification' else len(cases)*9*sources*horizon,
        qualification_updates=len(cases)*9*sources*horizon if kind=='qualification' else 0,
        replay_updates=len(cases)*horizon, sources=hashes, inputs=inputs,
        parent_study=str(old), root=str(root), wall_cap_hours=4)
    spec['execution_contract'] = freeze_contract(spec, repo, root, study)
    if qualified is not None:
        if qualified.contract != spec['execution_contract']:
            raise ValueError('Scientific settings differ from qualification')
        spec['qualification_binding'] = qualified.record()
    write_json(study/'protocol.json', spec)
    print('Frozen', kind, len(cases), 'states,', len(cases)*9*sources*horizon, 'native updates', flush=True)


def bind(spec, repo, study):
    admit(spec, repo, spec['root'], study)

def observe(model, crops):
    model.model.eval()
    with torch.no_grad():
        x = torch.as_tensor(crops, dtype=torch.long, device=model.device)
        out = model.forward(x[:, :64], capture=True)
        logits = out.logits[:, -1].double()
        logp = logits.log_softmax(-1)
        nll = -logp.gather(1, x[:, 64, None]).squeeze(1)
        entropy = -(logp.exp()*logp).sum(-1)
        common, transverse, total = [], [], []
        for layer in out.pldr_attentions:
            matrix = layer[0].double()
            center = matrix.mean(-2, keepdim=True)
            common.append(center.square().mean((-2, -1)))
            transverse.append((matrix-center).square().mean((-2, -1)))
            total.append(matrix.square().mean((-2, -1)))
        common = torch.stack(common); transverse = torch.stack(transverse); total = torch.stack(total)
        q = torch.cat((torch.stack([nll.mean(), entropy.mean()]),
            torch.log1p(common.mean((1, 2)).sqrt()), torch.log1p(transverse.mean((1, 2)).sqrt())))
        result = dict(q=q.cpu().numpy(), nll=nll.cpu().numpy(), entropy=entropy.cpu().numpy(),
            common_ms=common.cpu().numpy(), transverse_ms=transverse.cpu().numpy(), total_ms=total.cpu().numpy(),
            logits=logits.float().cpu().numpy())
    model.capture = False; model.head_outputs = {}
    return result


def worker(root, study, repo, spec, case, device):
    bind(spec, repo, study)
    out = study/'runs'/case['name']; out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(4); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    model = TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1', case['heads'], case['seed'], device)
    optimizer, scheduler = optimizer_and_scheduler(model, case['profile'])
    old = Path(spec['parent_study'])
    checkpoint = torch.load(old/'runs'/case['name']/'incoming-state.pt', map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 2048
    corpus = np.load(old/'data'/f"corpus-{case['corpus']}.npy", mmap_mode='r')
    with np.load(old/'data/panels.npz') as f:
        crops, donors = f['evaluation'], f['donors']
    with np.load(study/'sampling'/(case['name']+'.npz')) as f:
        sampling = f['blocks']
    timers = dict(restore=0., direction=0., pulse=0., update=0., observation=0., digest=0., serialization=0.)
    def timed(key, fn):
        tick = time.perf_counter(); result = fn(); timers[key] += time.perf_counter()-tick
        return result
    def restore():
        optimizer.zero_grad(set_to_none=True); model.capture=False; model.eta=None; model.head_outputs={}
        model.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer']))
        scheduler.load_state_dict(copy.deepcopy(checkpoint['scheduler']))
    def batch(ids):
        arr = corpus[(ids//8)[:, None], (64*(ids%8))[:, None]+np.arange(65)]
        return torch.as_tensor(arr, dtype=torch.long, device=model.device)
    def update(ids):
        model.model.train(); optimizer.zero_grad(set_to_none=True)
        loss = native_loss(model, batch(ids), case['profile'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite native loss')
        loss.backward(); native_clip(model, case['profile']); optimizer.step(); scheduler.step()
        return float(loss.detach())
    timed('restore', restore)
    incoming = timed('digest', lambda: digest_state(model, optimizer, scheduler))
    assert incoming == json.loads((old/'runs'/case['name']/'results.json').read_text())['incoming_digest']
    def directions():
        model.model.train(); optimizer.zero_grad(set_to_none=True)
        loss = native_loss(model, torch.as_tensor(donors, dtype=torch.long, device=model.device), case['profile'])
        loss.backward()
        result, info = {}, {}
        for family in ['generator', 'body']:
            params = [(n,p) for n,p in model.model.named_parameters() if generator_parameter(n)==(family=='generator')]
            norm = sum(p.detach().double().square().sum() for _,p in params).sqrt().item()
            gradnorm = sum(p.grad.detach().double().square().sum() for _,p in params).sqrt().item()
            assert norm > 0 and gradnorm > 0 and np.isfinite(norm+gradnorm)
            direction = {n: -p.grad.detach().clone()/gradnorm for n,p in params}
            h = hashlib.sha256()
            for n,d in direction.items():
                h.update(n.encode()); h.update(d.cpu().numpy().tobytes())
            result[family] = direction
            info[family] = dict(parameter_norm=norm, gradient_norm=gradnorm,
                direction_sha256=h.hexdigest(), parameters=sum(p.numel() for _,p in params), tensors=len(params))
        optimizer.zero_grad(set_to_none=True)
        return result, info, float(loss.detach())
    direction, info, direction_loss = timed('direction', directions)
    assert digest_state(model, optimizer, scheduler) == incoming
    write_json(out/'directions.json', dict(families=info, loss=direction_loss, incoming_digest=incoming))
    def pulse(arm):
        if arm['family'] is None:
            return dict(relative_norm=0., relative_displacement_error=0.)
        family = arm['family']; coeff = arm['multiplier']*spec['amplitude']*info[family]['parameter_norm']
        actual2, error2 = 0., 0.
        with torch.no_grad():
            for name,p in model.model.named_parameters():
                if name not in direction[family]:
                    continue
                previous = p.detach().clone(); delta = coeff*direction[family][name]
                p.add_(delta)
                actual = p.double()-previous.double()
                actual2 += actual.square().sum().item()
                error2 += (actual-delta.double()).square().sum().item()
        return dict(relative_norm=np.sqrt(actual2)/info[family]['parameter_norm'],
            relative_displacement_error=np.sqrt(error2)/abs(coeff))
    records = []; first_final = None; first_data = None
    for s in range(spec['source_replicates']):
        for arm in spec['arms']:
            before = timers.copy(); timed('restore', restore)
            assert timed('digest', lambda: digest_state(model, optimizer, scheduler)) == incoming
            displacement = timed('pulse', lambda: pulse(arm))
            observations = [timed('observation', lambda: observe(model, crops))]; losses=[]
            for t, ids in enumerate(sampling[s], 1):
                losses.append(timed('update', lambda ids=ids: update(ids)))
                if t in spec['times']:
                    observations.append(timed('observation', lambda: observe(model, crops)))
            data = {key: np.stack([r[key] for r in observations]) for key in observations[0]}
            data.update(losses=np.array(losses), blocks=sampling[s], times=np.array(spec['times']))
            path = out/f"source{s}-{arm['name']}.npz"
            timed('serialization', lambda: np.savez(path, **data))
            final = timed('digest', lambda: digest_state(model, optimizer, scheduler))
            if s == 0 and arm['name']=='zero':
                first_final=final; first_data=data
            records.append(dict(source=s, arm=arm['name'], file=path.name, sha256=sha256(path),
                displacement=displacement, final_digest=final,
                seconds={k:timers[k]-before[k] for k in timers}))
        print(case['name'], 'source', s+1, '/', spec['source_replicates'], 'elapsed', round(time.perf_counter()-start,1), flush=True)
        if time.perf_counter()-start > spec['wall_cap_hours']*3600:
            raise TimeoutError('Fixed worker wall cap exceeded; incomplete outputs retained')
    # Full zero trajectory replay, including every observation and all final Adam tensors.
    timed('restore', restore)
    observations=[timed('observation', lambda: observe(model,crops))]; losses=[]
    for t,ids in enumerate(sampling[0],1):
        losses.append(timed('update', lambda ids=ids: update(ids)))
        if t in spec['times']:
            observations.append(timed('observation', lambda: observe(model,crops)))
    replay={key:np.stack([r[key] for r in observations]) for key in observations[0]}
    replay.update(losses=np.array(losses),blocks=sampling[0],times=np.array(spec['times']))
    for key in first_data:
        assert np.array_equal(replay[key],first_data[key]), 'Zero replay discrepancy: '+key
    assert timed('digest',lambda:digest_state(model,optimizer,scheduler))==first_final
    timed('restore',restore)
    assert digest_state(model,optimizer,scheduler)==incoming
    write_json(out/'results.json',dict(status='complete',case=case,protocol_sha256=sha256(study/'protocol.json'),
        started_from=incoming,restoration_bitwise=True,zero_replay_bitwise=True,
        replay_updates=spec['horizon'],records=records,timers=timers,
        elapsed_seconds=time.perf_counter()-start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
    print('Complete',case['name'],round(time.perf_counter()-start,1),'seconds',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','worker'])
    p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--kind',choices=['qualification','development','confirmation'],default='confirmation')
    p.add_argument('--amplitude',type=float,default=3e-5);p.add_argument('--case');p.add_argument('--device')
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];root=Path(a.root).resolve();study=Path(a.study).resolve()
    if a.action=='prepare':
        assert 0 < a.amplitude < .01
        prepare(root,study,repo,a.kind,a.amplitude);return
    spec=json.loads((study/'protocol.json').read_text())
    if a.action=='worker':
        worker(root,study,repo,spec,next(c for c in spec['cases'] if c['name']==a.case),a.device);return
    if (study/'run-status.json').exists():
        raise FileExistsError(study/'run-status.json')
    bind(spec,repo,study);tick=time.perf_counter()
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    def queue(device,cases):
        for case in cases:
            log=study/(case['name']+'.log')
            with log.open('x') as stream:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'worker','--root',str(root),
                    '--study',str(study),'--case',case['name'],'--device',device],cwd=repo,env=env,
                    stdout=stream,stderr=subprocess.STDOUT,check=True)
    # One process per GPU; balance the widths by assigning alternate cases.
    cases=sorted(spec['cases'],key=lambda c:(c['corpus'],c['identity'],c['heads']))
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures=[executor.submit(queue,f'cuda:{i}',cases[i::2]) for i in range(2)]
        for f in futures:f.result()
    write_json(study/'run-status.json',dict(status='complete',elapsed_seconds=time.perf_counter()-tick,
        scientific_updates=spec['scientific_updates'],qualification_updates=spec['qualification_updates'],
        replay_updates=spec['replay_updates'],protocol_sha256=sha256(study/'protocol.json')))

if __name__=='__main__':
    main()
