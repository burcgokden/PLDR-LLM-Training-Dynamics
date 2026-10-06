#!/usr/bin/env python3
"""Qualified finite optimizer-state interventions on single-pass RefinedWeb.

All public preparation, execution, and worker entries validate the same bound
sources, inputs and runtime. Assessment additionally requires independently
verified short native qualification for both widths and both implementations.
"""
from companion_paths import configured_path
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
from model_rg.training import TrainingModel
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.schedules import loss as native_loss
from finite_response_study import observe, promote_control
from measure_law_closure import digest_state


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0), threads=4, tf32=False)


def check_files(items):
    for p, h in items.items():
        if not Path(p).is_file() or sha(p) != h: raise ValueError('Changed or missing bound file: '+p)


def common_contract(spec):
    return {k: spec[k] for k in ['schema', 'cases', 'arms', 'modes', 'batch_size',
            'prediction_tolerance_units', 'signal_floor', 'observation', 'intervention',
            'sources', 'inputs', 'runtime']}


def validate(study):
    study = Path(study)
    spec = json.loads((study/'protocol.json').read_text())
    if spec['schema'] != 'finite-optimizer-transport-v1': raise ValueError('Unsupported protocol')
    if runtime() != spec['runtime']: raise ValueError('Runtime differs from qualified protocol')
    check_files(spec['sources']); check_files(spec['inputs']); check_files(spec['sampling'])
    expected_stage = dict(qualification=(2, 1, [0, 1, 2]), assessment=(8, 2, [0, 1, 2, 4, 8]))
    if spec['stage'] not in expected_stage: raise ValueError('Invalid stage')
    if (spec['horizon'], spec['replicates'], spec['times']) != expected_stage[spec['stage']]:
        raise ValueError('Changed stage design')
    for c in spec['cases']:
        order = np.random.default_rng(c['stream_seed']).permutation(524288)
        rng = np.random.default_rng(spec['source_seed']+c['heads'])
        expected = np.stack([rng.choice(order[65536:], 32*spec['horizon'], replace=False).reshape(-1, 32)
                             for _ in range(spec['replicates'])])
        actual = np.load(study/(c['name']+'-blocks.npy'))
        if not np.array_equal(actual, expected): raise ValueError('Source sequence mismatch')
    if spec['stage'] == 'assessment':
        qualification = Path(spec['qualification']['path'])
        if sha(qualification) != spec['qualification']['sha256']: raise ValueError('Qualification changed')
        result = json.loads(qualification.read_text())
        if result['status'] != 'passed' or result['stage'] != 'qualification':
            raise ValueError('A passed native qualification is required')
        qspec = validate(qualification.parent)
        if result['protocol_sha256'] != sha(qualification.parent/'protocol.json'):
            raise ValueError('Qualification protocol mismatch')
        check_files(result['verified_files'])
        if common_contract(qspec) != common_contract(spec): raise ValueError('Qualification does not cover this design')
    return spec


def prepare(study, root, stage, qualification):
    root = Path(root).resolve(); study = Path(study).resolve()
    if not study.is_relative_to(root): raise ValueError('Scientific destination must be inside its data root')
    if stage == 'assessment':
        if qualification is None: raise ValueError('Assessment requires --qualification')
        qualification = Path(qualification).resolve()
        qspec = validate(qualification.parent)
        terminal = json.loads(qualification.read_text())
        if terminal['status'] != 'passed' or terminal['stage'] != 'qualification':
            raise ValueError('Unqualified native producer')
        if terminal['protocol_sha256'] != sha(qualification.parent/'protocol.json'):
            raise ValueError('Qualification protocol mismatch')
        check_files(terminal['verified_files'])
    parent = root/'outer-transfer-20260911'
    original = json.loads((parent/'protocol.json').read_text())
    cases = [c for c in original['cases'] if c['corpus'] == c['identity'] == 0]
    if sorted(c['heads'] for c in cases) != [4, 14]: raise ValueError('Unexpected incoming states')
    inputs = {str(p): sha(p) for p in [parent/'protocol.json', parent/'data/panels.npz', parent/'data/corpus-0.npy']}
    for c in cases:
        p = parent/'runs'/c['name']/'incoming-state.pt'; inputs[str(p)] = sha(p)
    assets = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    for p in assets.iterdir():
        if p.is_file() and p.suffix in ['.py', '.json', '.model']: inputs[str(p)] = sha(p)
    names = set(original['sources']) | {'scripts/finite_response_study.py', 'scripts/measure_law_closure.py',
            'scripts/optimizer_transport_study.py', 'scripts/analyze_optimizer_transport.py',
            'scripts/verify_optimizer_transport.py', 'src/model_rg/finite_optimizer.py'}
    # Bind all model modules; dynamic imports in the native adapter must not escape the inventory.
    names |= {str(p.relative_to(REPO)) for p in (REPO/'src/model_rg').glob('*.py')}
    sources = {str(REPO/n): sha(REPO/n) for n in sorted(names)}
    arms = [dict(name='zero', family=None, h=0.)]
    for family in ['first', 'logsecond']:
        for tag, h in [('plus', .1), ('minus', -.1), ('halfplus', .05), ('halfminus', -.05)]:
            arms.append(dict(name=family+'-'+tag, family=family, h=h))
    horizon, replicates, times = (2, 1, [0, 1, 2]) if stage == 'qualification' else (8, 2, [0, 1, 2, 4, 8])
    spec = dict(schema='finite-optimizer-transport-v1', stage=stage, created_at=datetime.now(timezone.utc).isoformat(),
        cases=cases, arms=arms, modes=['native32', 'arithmetic64'], horizon=horizon, replicates=replicates,
        times=times, batch_size=32, source_seed=914140000 if stage == 'qualification' else 914141000,
        prediction_tolerance_units=32., signal_floor=5e-7,
        observation='Twelve coordinates: external-target risk, entropy, five log1p common RMS and five log1p transverse RMS; equal saved-time and coordinate weights. Full path primary; no elapsed-time quadrature.',
        intervention='All first moments multiplied by 1+h OR all positive second moments multiplied by exp(h); zeros preserved. h=+-0.1,+-0.05; all other incoming state unchanged. Uniform radial moment directions, no equality of physical norms across families asserted.',
        primary='Fixed-clipped-gradient first-step state prediction; all model-coordinate errors divided by dtype epsilon times magnitude scale <=32. Incoming full logits equal baseline; measure later visibility with fixed signal floor. Every source/sign/mode retained. No derivative-domain success claim.',
        sources=sources, inputs=inputs, runtime=runtime(), data_root=str(root), sampling={})
    if stage == 'assessment':
        if common_contract(qspec) != common_contract(spec): raise ValueError('Source or design changed since qualification')
        spec['qualification'] = dict(path=str(qualification), sha256=sha(qualification))
    study.mkdir(parents=True, exist_ok=False)
    for c in cases:
        order = np.random.default_rng(c['stream_seed']).permutation(524288)
        rng = np.random.default_rng(spec['source_seed']+c['heads'])
        blocks = np.stack([rng.choice(order[65536:], 32*horizon, replace=False).reshape(horizon, 32) for _ in range(replicates)])
        dest = study/(c['name']+'-blocks.npy'); np.save(dest, blocks); spec['sampling'][str(dest)] = sha(dest)
    for p in sources:
        dest = study/'executed-source'/Path(p).relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(Path(p).read_bytes())
    write(study/'protocol.json', spec)
    validate(study)
    print('Prepared', stage, str(study), flush=True)


def sampled_state(model, optimizer):
    rows = []; names = []; offsets = [0]
    by_param = {id(p): group for group in optimizer.param_groups for p in group['params']}
    for n, p in model.model.named_parameters():
        state = optimizer.state[p]; group = by_param[id(p)]
        # Use float64 CPU index construction to avoid float32 endpoint overflow.
        ids = torch.as_tensor(np.linspace(0, p.numel()-1, min(32, p.numel()), dtype=np.int64), device=p.device)
        cols = [p.detach().reshape(-1)[ids], state['exp_avg'].reshape(-1)[ids], state['exp_avg_sq'].reshape(-1)[ids],
                (p.grad.reshape(-1)[ids] if p.grad is not None else torch.zeros_like(ids, dtype=p.dtype))]
        vals = torch.stack(cols, -1).double().cpu().numpy()
        constants = [group['betas'][0], group['betas'][1], float(state['step']), group['lr'], group['eps'], group['weight_decay']]
        rows.append(np.column_stack([ids.cpu().numpy(), vals, np.tile(constants, (len(ids), 1))]))
        names.append(n); offsets.append(offsets[-1]+len(ids))
    return np.concatenate(rows), np.array(names), np.array(offsets)


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for tensor in tensors: digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict_full(model, optimizer):
    """Compute before optimizer.step; retain predictions on CPU to bound GPU use."""
    predictions = []
    for group in optimizer.param_groups:
        b1, b2 = group['betas']; rate = group['lr']; eps = group['eps']; decay = group['weight_decay']
        for p in group['params']:
            if p.grad is None: raise ValueError('Unexpected unused parameter')
            state = optimizer.state[p]; k = int(state['step'])+1
            x, m, v, g = [t.detach().double() for t in [p, state['exp_avg'], state['exp_avg_sq'], p.grad]]
            mp = b1*m+(1-b1)*g; vp = b2*v+(1-b2)*g*g
            update = rate*mp/(1-b1**k)/(torch.sqrt(vp/(1-b2**k))+eps)
            pp = (1-rate*decay)*x-update
            # Magnitude scales retain summands, so exact cancellation is admissible.
            mscale = b1*m.abs()+(1-b1)*g.abs()
            source_scale = rate*mscale/(1-b1**k)/(torch.sqrt(vp/(1-b2**k))+eps)
            scales = [x.abs()+source_scale, mscale, b2*v.abs()+(1-b2)*g*g]
            predictions.append(([t.cpu() for t in [pp, mp, vp]], [t.cpu() for t in scales]))
    return predictions


def compare_full(optimizer, predictions):
    maximum = np.zeros(3); sumsq = np.zeros(3); count = 0
    idx = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            predicted, scales = predictions[idx]; idx += 1
            eps = torch.finfo(p.dtype).eps; tiny = torch.finfo(p.dtype).tiny
            state = optimizer.state[p]
            for j, actual in enumerate([p, state['exp_avg'], state['exp_avg_sq']]):
                error = (actual.detach().double().cpu()-predicted[j]).abs()
                normalized = error/(eps*scales[j]+tiny)
                maximum[j] = max(maximum[j], float(normalized.max()))
                sumsq[j] += float(error.square().sum())
            count += p.numel()
    return dict(max_roundoff_units=maximum.tolist(), error_l2=np.sqrt(sumsq).tolist(), coordinates=count)


def worker(study, heads, device):
    study = Path(study).resolve(); spec = validate(study)
    if device not in ['cuda:0', 'cuda:1']: raise ValueError('Select one physical GPU per worker')
    case = next(c for c in spec['cases'] if c['heads'] == heads)
    dest = study/case['name']; dest.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    root = Path(spec['data_root']); parent = root/'outer-transfer-20260911'
    ck = torch.load(parent/'runs'/case['name']/'incoming-state.pt', map_location='cpu', weights_only=False)
    if ck['step'] != 2048: raise ValueError('Incorrect complete incoming state')
    corpus = np.load(parent/'data/corpus-0.npy', mmap_mode='r')
    with np.load(parent/'data/panels.npz') as f: crops = f['evaluation']
    source_blocks = np.load(study/(case['name']+'-blocks.npy'))
    start = time.perf_counter(); records = []
    for mode in spec['modes']:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        model = TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1', heads, case['seed'], device)
        promotion = promote_control(model) if mode == 'arithmetic64' else None
        optimizer, scheduler = optimizer_and_scheduler(model, case['profile'])
        def restore():
            optimizer.zero_grad(set_to_none=True); model.capture = False; model.eta = None; model.head_outputs = {}
            model.model.load_state_dict(ck['model']); optimizer.load_state_dict(copy.deepcopy(ck['optimizer'])); scheduler.load_state_dict(copy.deepcopy(ck['scheduler']))
        restore(); initial = digest_state(model, optimizer, scheduler)
        modeout = dest/mode; modeout.mkdir(); baselines = {}
        jobs = [(s, arm) for s in range(spec['replicates']) for arm in spec['arms']]
        jobs.append((0, dict(name='zero-replay', family=None, h=0.)))
        for source, arm in jobs:
            if time.perf_counter()-start > 1800: raise TimeoutError('Thirty-minute worker budget exceeded')
            restore()
            if digest_state(model, optimizer, scheduler) != initial: raise ValueError('Complete reset failed')
            incoming, names, offsets = sampled_state(model, optimizer)
            with torch.no_grad():
                if arm['family']:
                    key = 'exp_avg' if arm['family'] == 'first' else 'exp_avg_sq'
                    factor = 1+arm['h'] if key == 'exp_avg' else math.exp(arm['h'])
                    for state in optimizer.state.values(): state[key].mul_(factor)
            pulsed, _, _ = sampled_state(model, optimizer)
            moment_values = [s['exp_avg_sq'] for s in optimizer.state.values()]
            zeros = sum(int((v == 0).sum()) for v in moment_values)
            positive_min = min(float(v[v > 0].min()) for v in moment_values if bool((v > 0).any()))
            del moment_values
            observations = [observe(model, crops)]; losses = []; grads = []; formula = None
            blocks = source_blocks[source]
            first_gradient_digest = None
            for t, ids in enumerate(blocks, 1):
                model.model.train(); optimizer.zero_grad(set_to_none=True)
                arr = corpus[(ids//8)[:, None], (64*(ids % 8))[:, None]+np.arange(65)]
                loss = native_loss(model, torch.as_tensor(arr, dtype=torch.long, device=device), case['profile'])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1., error_if_nonfinite=True)
                losses.append(float(loss.detach())); grads.append(float(gn))
                if t == 1:
                    first_gradient_digest = tensor_digest(p.grad for p in model.model.parameters())
                    before, _, _ = sampled_state(model, optimizer)
                    predictions = predict_full(model, optimizer)
                optimizer.step(); scheduler.step()
                if t == 1:
                    formula = compare_full(optimizer, predictions); del predictions
                    after, _, _ = sampled_state(model, optimizer)
                if t in spec['times']: observations.append(observe(model, crops))
            arrays = {k: np.stack([o[k] for o in observations]) for k in observations[0]}
            if not all(np.isfinite(x).all() for x in arrays.values()): raise FloatingPointError('Nonfinite observation')
            final = digest_state(model, optimizer, scheduler)
            if arm['name'] == 'zero':
                baselines[source] = dict(arrays=arrays, final=final, gradient=first_gradient_digest, grads=grads,
                                         zeros=zeros, incoming=incoming)
            base = baselines[source]
            if not np.array_equal(base['arrays']['logits'][0], arrays['logits'][0]): raise ValueError('Incoming emission changed')
            if first_gradient_digest != base['gradient']: raise ValueError('First gradient changed under moment-only pulse')
            if zeros != base['zeros']: raise ValueError('Moment zero support changed')
            if arm['name'] == 'zero-replay':
                if final != base['final'] or grads != base['grads'] or not all(np.array_equal(v, base['arrays'][k]) for k,v in arrays.items()):
                    raise ValueError('Bitwise replay failed')
            path = modeout/(f'source{source}-'+arm['name']+'.npz')
            np.savez(path, **arrays, times=spec['times'], blocks=blocks, losses=losses, gradnorms=grads,
                     incoming_sample=incoming, pulsed_sample=pulsed, first_before=before, first_after=after,
                     parameter_names=names, parameter_offsets=offsets)
            records.append(dict(file=str(path), sha256=sha(path), source=source, arm=arm['name'], mode=mode,
                                first_gradient_sha256=first_gradient_digest, final_digest=final,
                                full_prediction=formula, zero_second_moments=zeros, min_positive_second=positive_min))
            print(case['name'], mode, source, arm['name'], 'seconds', round(time.perf_counter()-start, 1),
                  'prediction units', formula['max_roundoff_units'], flush=True)
        write(modeout/'metadata.json', dict(peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
              promotion=promotion, initial_digest=initial, replay_bitwise=True))
        del model, optimizer, scheduler, baselines, observations, arrays, loss, gn
        torch.cuda.empty_cache()
    write(dest/'results.json', dict(status='complete', stage=spec['stage'], records=records,
          seconds=time.perf_counter()-start, protocol_sha256=sha(study/'protocol.json')))


def run(study):
    spec = validate(study); jobs = []
    for gpu, c in enumerate(sorted(spec['cases'], key=lambda c: c['heads'])):
        log = (Path(study)/(c['name']+'-worker.log')).open('x')
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--study', str(study),
                   '--heads', str(c['heads']), '--device', f'cuda:{gpu}']
        jobs.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    try:
        while any(p.poll() is None for p, _ in jobs):
            if any(p.poll() not in [None, 0] for p, _ in jobs):
                for p, _ in jobs:
                    if p.poll() is None: p.terminate()
                break
            time.sleep(.2)
        returns = [p.wait() for p, _ in jobs]
    finally:
        for _, log in jobs: log.close()
    if any(returns): raise RuntimeError('Native worker failed; inspect complete logs: '+str(returns))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['prepare', 'run', 'worker', 'validate'])
    ap.add_argument('--study', required=True); ap.add_argument('--stage', choices=['qualification', 'assessment'])
    ap.add_argument('--data-root', default=configured_path('data:model'))
    ap.add_argument('--qualification'); ap.add_argument('--heads', type=int); ap.add_argument('--device')
    args = ap.parse_args()
    if args.action == 'prepare':
        if args.stage is None: ap.error('--stage is required for preparation')
        prepare(args.study, args.data_root, args.stage, args.qualification)
    elif args.action == 'run': run(args.study)
    elif args.action == 'worker': worker(args.study, args.heads, args.device)
    else: validate(args.study); print('admitted')
