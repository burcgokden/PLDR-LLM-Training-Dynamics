#!/usr/bin/env python
"""Qualify, freeze and run a two-GPU single-pass state-refresh experiment."""
from companion_paths import child_pythonpath
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.provenance import sha256, write_json
from model_rg.refresh import fit, calibrate
from model_rg.qualification import admit, freeze_contract, source_names, validate_qualification
from model_rg.schedules import loss as native_loss, clip as native_clip
from model_rg.training import TrainingModel
from measure_law_closure import digest_state
from outer_transfer_study import observe


def now():
    return datetime.now(timezone.utc).isoformat()


def bind(spec, repo, study):
    admit(spec, repo, spec['root'], study)


def prepare(root, study, repo, qualification):
    qualified = None if qualification else validate_qualification(study.parent/'qualification/protocol.json', repo, 'P1')
    study.mkdir(parents=True, exist_ok=True)
    if (study/'protocol.json').exists():
        raise FileExistsError(study/'protocol.json')
    old = root/'outer-transfer-20260911'
    prior = json.loads((old/'protocol.json').read_text())
    cases = prior['cases']
    if qualification:
        cases = [c for c in cases if c['corpus'] == 0 and c['identity'] == 0]
    horizon = 4 if qualification else 64
    assessment = 4 if qualification else 32
    total = 48+assessment
    inputs = {str(old/'protocol.json'): sha256(old/'protocol.json'),
              str(old/'data/manifest.json'): sha256(old/'data/manifest.json'),
              str(old/'data-verification.json'): sha256(old/'data-verification.json')}
    for name in ['panels.npz', 'corpus-0.npy', 'corpus-1.npy']:
        inputs[str(old/'data'/name)] = sha256(old/'data'/name)
    for name in ['modeling_pldrllm.py', 'configuration_pldrllm.py']:
        p = root/'assets/PLDR-LLM-v51-SOC-110M-1'/name
        inputs[str(p)] = sha256(p)
    shutil.copy2(old/'archive-forecast.json', study/'archive-forecast.json')
    inputs[str(study/'archive-forecast.json')] = sha256(study/'archive-forecast.json')
    (study/'sampling').mkdir()
    for case in cases:
        state = old/'runs'/case['name']/'incoming-state.pt'
        inputs[str(state)] = sha256(state)
        meta = old/'runs'/case['name']/'results.json'
        inputs[str(meta)] = sha256(meta)
        assert inputs[str(state)] == json.loads(meta.read_text())['incoming_state_sha256']
        order = np.random.default_rng(case['stream_seed']).permutation(524288)
        used = order[:65536]; remaining = order[65536:]
        # New seeds are unrelated to all archived branch generators.
        seed = 115110000 + case['heads']*1000 + case['corpus']*100 + case['identity']
        if qualification:
            seed += 10000000
        rng = np.random.default_rng(seed)
        successor = rng.choice(remaining, 32*horizon, replace=False).reshape(-1, 32)
        remaining_next = remaining[~np.isin(remaining, successor.ravel())]
        parent = np.stack([rng.choice(remaining, 32*horizon, replace=False).reshape(-1, 32) for _ in range(total)])
        later = np.stack([rng.choice(remaining_next, 32*horizon, replace=False).reshape(-1, 32) for _ in range(total)])
        path = study/'sampling'/(case['name']+'.npz')
        np.savez_compressed(path, primary=used, successor=successor, parent=parent, later=later)
        inputs[str(path)] = sha256(path)
    sources = source_names(repo, 'P1', set(prior['sources']) | {
        'scripts/refresh_study.py', 'scripts/analyze_refresh.py', 'scripts/verify_refresh.py',
        'src/model_rg/refresh.py'})
    snapshot = study/'executed-source'
    for name in sources:
        target = snapshot/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo/name, target)
    source_hashes = {name: sha256(repo/name) for name in sources}
    write_json(snapshot/'manifest.json', dict(sources=source_hashes))
    spec = dict(
        schema='onepass-refresh-protocol-v1', status='frozen', frozen_at=now(),
        qualification=qualification, cases=cases, primary_steps=2048,
        successor_updates=horizon, horizon=horizon,
        horizons=[0, 1, 2, 3, 4] if qualification else [0, 1, 4, 16, 64],
        fit_replicates=2, adaptation=16, calibration=8, assessment=assessment,
        budgets=[4, 8, 16], batch_size=32, bootstrap_samples=2000, bootstrap_seed=115119000,
        scientific_updates=0 if qualification else len(cases)*(horizon+2*total*horizon),
        primary='Four-risk-increment score: local-minus-diagonal and transferred-minus-diagonal, budget-minus-16 within candidate, and parent-reuse minus successor-refit. Every state, budget, and fit cohort retained.',
        secondary='Joint risk/entropy and paired/endpoint coarsenings; path mean errors, whole-path coverage, widths and cost.',
        gate=dict(mean_path_tolerance=.01, score_excess_tolerance=.05,
                  rule='A cheaper budget must have upper 95% paired path-bootstrap bounds below both tolerances and lower acquisition plus score-query cost in every state and fit cohort; crossing is unresolved. These are conditional empirical gates, not population or simultaneous confidence guarantees.'),
        controls='Zero-acquisition archive mean/covariance and no-change mean/archive covariance; full frozen parent forecasts and tubes reused at the successor; separately reacquired successor fits and calibration.',
        independence='Counterfactual paths start from identical complete states and independently sample without replacement from the remaining blocks. Paths may overlap each other. Parent/successor and fit replicates sharing assessment are dependent.',
        resource='Single pass over distinct source token blocks within every trajectory, including its 2048-update prefix and 64-update successor when applicable. Corpora and fixed evaluation panels inherited by immutable identity.',
        regularization=prior['regularization'], sources=source_hashes, inputs=inputs,
        parent_study=str(old), root=str(root))
    spec['execution_contract'] = freeze_contract(spec, repo, root, study)
    if qualified is not None:
        if qualified.contract != spec['execution_contract']:
            raise ValueError('Scientific settings differ from qualification')
        spec['qualification_binding'] = qualified.record()
    write_json(study/'protocol.json', spec)
    print('Frozen', len(cases), 'lineages;', 0 if qualification else len(cases)*(horizon+2*total*horizon), 'scientific updates', flush=True)


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
    original = checkpoint
    corpus = np.load(old/'data'/f"corpus-{case['corpus']}.npy", mmap_mode='r')
    with np.load(old/'data/panels.npz') as f:
        crops = f['evaluation']
    with np.load(study/'sampling'/(case['name']+'.npz')) as f:
        sampling = {k: f[k] for k in f.files}
    archive = json.loads((study/'archive-forecast.json').read_text())['fits'][str(case['heads'])]
    timers = dict(restore=0., update=0., observation=0., serialization=0., fit=0., calibration=0.)
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
    def save_state(path, step):
        state = dict(model={n: p.detach().cpu().clone() for n, p in model.model.state_dict().items()},
                     optimizer=copy.deepcopy(optimizer.state_dict()), scheduler=copy.deepcopy(scheduler.state_dict()),
                     step=step, case=case, protocol_sha256=sha256(study/'protocol.json'))
        for item in state['optimizer']['state'].values():
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.cpu().clone()
        torch.save(state, path)
        return state
    states = []
    for state_name, sample_key in [('parent', 'parent'), ('successor', 'later')]:
        folder = out/state_name; folder.mkdir()
        if state_name == 'successor':
            checkpoint = original; timed('restore', restore)
            transition_losses = [timed('update', lambda ids=ids: update(ids)) for ids in sampling['successor']]
            np.savez_compressed(out/'successor-transition.npz', losses=transition_losses, blocks=sampling['successor'])
            checkpoint = timed('serialization', lambda: save_state(out/'successor-state.pt', 2048+spec['successor_updates']))
        timed('restore', restore)
        incoming_digest = digest_state(model, optimizer, scheduler)
        if state_name == 'parent':
            assert incoming_digest == json.loads((old/'runs'/case['name']/'results.json').read_text())['incoming_digest']
        records=[]; values=[]; fitted={}; tubes={}; fit_seconds={}; calibration_seconds={}; first_final=None
        blocks = sampling[sample_key]
        for b in range(len(blocks)+1):
            replay = b == len(blocks); j = 0 if replay else b
            tick=time.perf_counter(); before=timers.copy(); branch_started=now()
            timed('restore', restore)
            assert digest_state(model, optimizer, scheduler) == incoming_digest
            obs = [timed('observation', lambda: observe(model, crops))]; losses=[]
            for t, ids in enumerate(blocks[j], 1):
                losses.append(timed('update', lambda ids=ids: update(ids)))
                if t in spec['horizons']:
                    obs.append(timed('observation', lambda: observe(model, crops)))
            arrays = {k: np.stack([o[k] for o in obs]) for k in obs[0]}
            arrays.update(horizons=np.array(spec['horizons']), losses=np.array(losses))
            assert all(np.isfinite(x).all() for x in arrays.values())
            final = digest_state(model, optimizer, scheduler)
            if replay:
                assert final == first_final
                with np.load(folder/'branch-000.npz') as f:
                    assert all(arrays[k].tobytes() == f[k].tobytes() for k in arrays)
                break
            if b == 0:
                first_final=final
            path=folder/f'branch-{b:03d}.npz'
            timed('serialization', lambda: np.savez_compressed(path, **arrays))
            values.append(np.stack([arrays['nll'].mean(1), arrays['entropy'].mean(1)], axis=1))
            cohort = b//24 if b<48 else None
            role = ('adaptation' if b%24<16 else 'calibration') if b<48 else 'assessment'
            records.append(dict(branch=b, role=role, cohort=cohort, raw=path.name, sha256=sha256(path),
                                started_at=branch_started, completed_at=now(),
                                final_digest=final, seconds=time.perf_counter()-tick,
                                components={k: timers[k]-before[k] for k in timers}))
            if b<48 and b%24 == 15:
                rep=b//24; fitted[str(rep)]={}; fit_seconds[str(rep)]={}
                for m in spec['budgets']:
                    tick=time.perf_counter()
                    fitted[str(rep)][str(m)] = timed('fit', lambda m=m: fit(np.asarray(values)[rep*24:rep*24+m], archive))
                    fit_seconds[str(rep)][str(m)] = time.perf_counter()-tick
                write_json(folder/f'fit-{rep}.json', dict(status='frozen', frozen_at=now(), fits=fitted[str(rep)],
                    inputs={r['raw']: r['sha256'] for r in records[rep*24:rep*24+16]}))
            if b<48 and b%24 == 23:
                rep=b//24; tubes[str(rep)]={}; calibration_seconds[str(rep)]={}
                for m in spec['budgets']:
                    tick=time.perf_counter()
                    tubes[str(rep)][str(m)] = timed('calibration', lambda m=m: calibrate(
                        np.asarray(values)[rep*24+16:rep*24+24], fitted[str(rep)][str(m)]))
                    calibration_seconds[str(rep)][str(m)] = time.perf_counter()-tick
                write_json(folder/f'calibration-{rep}.json', dict(status='frozen', frozen_at=now(), tubes=tubes[str(rep)],
                    fit_sha256=sha256(folder/f'fit-{rep}.json'),
                    inputs={r['raw']: r['sha256'] for r in records[rep*24+16:rep*24+24]}))
            if b%8 == 7:
                print(case['name'], state_name, b+1, '/', len(blocks), round(time.perf_counter()-start, 1), 's', flush=True)
        np.save(folder/'paths.npy', np.array(values))
        entry=dict(state=state_name, step=2048+(spec['successor_updates'] if state_name=='successor' else 0),
                   incoming_digest=incoming_digest, records=records, replay_bitwise=True,
                   fit_seconds=fit_seconds, calibration_seconds=calibration_seconds,
                   paths_sha256=sha256(folder/'paths.npy'))
        write_json(folder/'results.json', entry); states.append(entry)
    result=dict(status='complete', case=case, qualification=spec['qualification'], completed_at=now(),
                seconds=time.perf_counter()-start, components=timers, peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                scientific_updates=0 if spec['qualification'] else spec['successor_updates']+2*len(blocks)*spec['horizon'],
                qualification_updates=(spec['successor_updates']+2*len(blocks)*spec['horizon']) if spec['qualification'] else 0,
                replay_updates=2*spec['horizon'], protocol_sha256=sha256(study/'protocol.json'),
                states=[dict(state=s['state'], result_sha256=sha256(out/s['state']/'results.json')) for s in states],
                successor_state_sha256=sha256(out/'successor-state.pt'),
                transition_sha256=sha256(out/'successor-transition.npz'))
    write_json(out/'results.json', result)
    print('Completed', case['name'], round(result['seconds'], 1), 's', flush=True)


def launch(root, study, repo, spec):
    target=study/'launcher.json'
    if target.exists():
        raise FileExistsError(target)
    admit(spec, repo, root, study)
    # Pair a wide and a narrow case on each lane for a balanced finite budget.
    wide=[c for c in spec['cases'] if c['heads']==14]; narrow=[c for c in spec['cases'] if c['heads']==4]
    ordered=[c for pair in zip(wide,narrow) for c in pair]
    lanes=[[],[]]
    for i,c in enumerate(ordered):
        lanes[(i//2+i%2)%2].append(c)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    start=time.perf_counter()
    def lane(cases, gpu):
        records=[]
        for c in cases:
            command=[sys.executable, str(repo/'scripts/refresh_study.py'), 'worker', '--root', str(root),
                     '--study', str(study), '--case', c['name'], '--device', f'cuda:{gpu}']
            with (study/(c['name']+'.log')).open('x') as log:
                subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            p=study/'runs'/c['name']/'results.json'
            records.append(dict(case=c['name'],command=command,result_sha256=sha256(p)))
            print('Completed',c['name'],flush=True)
        return records
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,lane_cases,gpu) for gpu,lane_cases in enumerate(lanes)]
        records=[r for f in futures for r in f.result()]
    write_json(target,dict(status='complete',completed_at=now(),seconds=time.perf_counter()-start,
                          records=records,protocol_sha256=sha256(study/'protocol.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','worker'])
    p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--qualification',action='store_true');p.add_argument('--case');p.add_argument('--device')
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];root=Path(a.root).resolve();study=Path(a.study).resolve()
    if a.phase=='prepare':
        prepare(root,study,repo,a.qualification);return
    spec=json.loads((study/'protocol.json').read_text())
    if a.phase=='run':
        bind(spec,repo,study);launch(root,study,repo,spec)
    else:
        worker(root,study,repo,spec,next(c for c in spec['cases'] if c['name']==a.case),a.device)


if __name__=='__main__':
    main()
