#!/usr/bin/env python3
"""Equal-physical-duration single-pass continuations with complete matrix observations.

No fitted successor is evaluated. All displacement diagnostics are retrospective.
The two retained initialization identities are paired across widths and controls.
"""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from model_rg.provenance import sha256, write_json
from run_matched_clock import (ROOT, REPO, CORPUS, NATIVE, SOURCES as BASE,
                               read, configure, initialize, update)

PARENT = ROOT / 'matched-clock-onepass-20260916'
SOURCES = ['scripts/run_equal_time_flux.py'] + BASE
JOBS = [dict(heads=n, control=g, seed=s, steps=128*n,
             run_id=f'h{n}-g{g:g}-s{s}')
        for n in [4, 8, 14, 24] for g in [1.5, 2.]
        for s in [9163402, 9163403]]
DESIGN = dict(schema='equal-time-flux-v1', jobs=JOBS, updates_per_head=8, physical_duration=1/16,
              contexts=list(range(16, 24)), physical_block_fractions=[1, 2, 4, 8],
              executed_updates=1600, scientific_updates=1472, replay_updates=128, native_forwards=3216,
              memory_ceiling_bytes=22*1024**3, seconds_per_path=900,
              arithmetic_tolerance=3e-12,
              primary='Complete finite row increment and aligned chronological composition.',
              secondary='Matrix directional derivative error, accumulated finite cross and quadratic terms, signed cross-time matrix energy, target NLL.',
              source_rule='8*N consecutive next-unused batches of 32 distinct master documents from the parent checkpoint. The first eight updates replay the fixed-count study; each path itself is single pass.',
              conditioning='Fixed parent corpus/order, shared generator, and eight reused assessment contexts; two complete initialization identities paired across width/control.',
              independence='Sixteen continuations of existing training paths; zero new independently pretrained models. No population confidence interval or prospective successor claim.')


def versions():
    return {name: importlib.metadata.version(name)
            for name in ['numpy', 'torch', 'transformers', 'safetensors']}


def dependencies():
    parent = read(PARENT / 'protocol.json')
    files = [PARENT / n for n in ['protocol.json', 'selection.npz', 'reservation.json']]
    files += [Path(n) for n in parent['input_sha256']]
    files += [NATIVE / 'config.json']
    for j in JOBS:
        files += [PARENT / 'runs' / j['run_id'] / n
                  for n in ['manifest.json', 'final-state.pt', f"age-{j['steps']}.npz"]]
    return sorted(set(files))


def prepare(study):
    if study.exists():
        raise FileExistsError(study)
    for j in JOBS:
        folder = PARENT / 'runs' / j['run_id']
        m = read(folder / 'manifest.json')
        if m['status'] != 'complete' or m['job'] != j:
            raise ValueError('Incomplete parent path')
        for name in ['final-state.pt', f"age-{j['steps']}.npz"]:
            if sha256(folder/name) != m['artifacts'][name]:
                raise ValueError('Changed parent artifact')
    with np.load(PARENT/'selection.npz') as z:
        order = z['order']
        if not np.array_equal(np.sort(order), np.arange(524288)):
            raise ValueError('Parent order is not a permutation')
        for j in JOBS:
            stop = 32*(j['steps']+8*j['heads'])
            if stop > 16384*j['heads'] or len(np.unique(order[:stop])) != stop:
                raise ValueError('Repeated or exhausted source')
    study.mkdir(parents=True)
    p = dict(DESIGN, status='frozen_before_acquisition', packages=versions(),
             source_sha256={n: sha256(REPO/n) for n in SOURCES},
             input_sha256={str(f): sha256(f) for f in dependencies()},
             panel_reuse=dict(parent=str(PARENT), independent_panel=False,
                             reservation_sha256=sha256(PARENT/'reservation.json')))
    write_json(study/'protocol.json', p)
    for n in SOURCES:
        dst = study/'executed-source'/n
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((REPO/n).read_bytes())
    print({'prepared': str(study), 'paths': len(JOBS), 'executed_updates': 1600}, flush=True)


def admit(study, index=None):
    p = read(study/'protocol.json')
    if any(p.get(k) != v for k, v in DESIGN.items()) or p['packages'] != versions():
        raise ValueError('Changed design or environment')
    if set(p['source_sha256']) != set(SOURCES):
        raise ValueError('Incomplete source inventory')
    for n, h in p['source_sha256'].items():
        if sha256(REPO/n) != h or sha256(study/'executed-source'/n) != h:
            raise ValueError('Changed executing source')
    if set(p['input_sha256']) != {str(f) for f in dependencies()}:
        raise ValueError('Incomplete input inventory')
    for n, h in p['input_sha256'].items():
        if Path(n).name != 'final-state.pt' and sha256(n) != h:
            raise ValueError('Changed external input')
    if index is not None:
        if type(index) is not int or not 0 <= index < len(JOBS):
            raise ValueError('Invalid worker')
        f = PARENT/'runs'/JOBS[index]['run_id']/'final-state.pt'
        if sha256(f) != p['input_sha256'][str(f)]:
            raise ValueError('Changed selected state')
    return p


@torch.no_grad()
def observe(model, probes):
    model.model.eval()
    y = model.forward(probes[:, :64], capture=True)
    matrices = np.stack([layer[0].cpu().numpy() for layer in y.pldr_attentions], axis=1)
    nll = torch.nn.functional.cross_entropy(y.logits[:, -1], probes[:, 64], reduction='none').cpu().numpy()
    if not np.isfinite(matrices).all() or not np.isfinite(nll).all():
        raise ValueError('Nonfinite observation')
    model.head_outputs = {}
    return matrices, nll


def worker(study, index, device):
    if device not in ['cuda:0', 'cuda:1']:
        raise ValueError('Invalid device')
    p = admit(study, index)
    j = JOBS[index]
    out = study/'runs'/j['run_id']
    out.mkdir(parents=True, exist_ok=False)
    m = dict(status='started', job=j, protocol_sha256=sha256(study/'protocol.json'),
             scientific_updates=0, replay_updates=0, executed_updates=0, native_forwards=0, device=device)
    write_json(out/'manifest.json', m)
    configure(device)
    start = time.monotonic()
    try:
        model, opt = initialize(j['heads'], j['control'], j['seed'], device, read(PARENT/'protocol.json'))
        snap = torch.load(PARENT/'runs'/j['run_id']/'final-state.pt', map_location='cpu', weights_only=False)
        model.model.load_state_dict(snap['model'])
        opt.load_state_dict(snap['optimizer'])
        if (snap['consumed_documents'], snap['remaining_order']) != (32*j['steps'], 32*j['steps']):
            raise ValueError('Wrong incoming source cursor')
        if any(int(s['step']) != j['steps'] for s in opt.state.values()):
            raise ValueError('Wrong incoming optimizer clock')
        torch.set_rng_state(snap['cpu_rng'])
        torch.cuda.set_rng_state(snap['cuda_rng'], device)
        del snap
        with np.load(PARENT/'selection.npz') as z:
            probes = torch.as_tensor(z['probes'][16:24], device=device, dtype=torch.long)
            rows = z['order'][32*j['steps']:32*(j['steps']+8*j['heads'])].reshape(8*j['heads'], 32).copy()
            if np.intersect1d(rows, z['order'][:32*j['steps']]).size or np.unique(rows).size != 256*j['heads']:
                raise ValueError('Repeated supervised block')
        a, nll = observe(model, probes)
        m['native_forwards'] += 1
        aa = a.astype(np.float64)
        row = np.square(aa-aa.mean(-2, keepdims=True)).sum((-2,-1))/np.square(aa).sum((-2,-1))
        with np.load(PARENT/'runs'/j['run_id']/f"age-{j['steps']}.npz") as z:
            replay = float(np.max(np.abs(row-z['heads'][:8, :, :, 0])))
        if replay > 1e-9:
            raise ValueError('Incoming row replay differs')
        del aa, row
        matrices = np.lib.format.open_memmap(out/'matrices.npy', mode='w+', dtype=np.float32, shape=(len(rows)+1,)+a.shape)
        matrices[0] = a
        risks, losses, norms = [nll], [], []
        tokens = np.load(CORPUS/'tokens.npy', mmap_mode='r')
        for k, batch in enumerate(rows):
            loss, norm = update(model, opt, tokens, batch, device)
            m['executed_updates'] += 1
            m['replay_updates' if k < 8 else 'scientific_updates'] += 1
            m['native_forwards'] += 1
            a, nll = observe(model, probes)
            m['native_forwards'] += 1
            matrices[k+1] = a; risks.append(nll); losses.append(loss); norms.append(norm)
            if torch.cuda.max_memory_allocated(device) >= p['memory_ceiling_bytes'] or time.monotonic()-start >= p['seconds_per_path']:
                raise ValueError('Resource ceiling exceeded')
        if any(int(s['step']) != j['steps']+8*j['heads'] for s in opt.state.values()):
            raise ValueError('Wrong outgoing optimizer clock')
        matrices.flush()
        del matrices
        np.savez_compressed(out/'observations.npz', target_nll=np.stack(risks),
                            training_loss=losses, gradient_norm=norms, document_rows=rows,
                            optimizer_steps=np.arange(j['steps'], j['steps']+8*j['heads']+1))
        m.update(status='complete', incoming_row_replay_error=replay,
                 outgoing_consumed_documents=32*(j['steps']+8*j['heads']),
                 artifacts={name: sha256(out/name) for name in ['observations.npz','matrices.npy']})
    except Exception as exc:
        m.update(status='failed', error=repr(exc))
        raise
    finally:
        m.update(elapsed_seconds=time.monotonic()-start,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        write_json(out/'manifest.json', m)
    print(m, flush=True)


def run(study):
    admit(study)
    logs = study/'logs'; logs.mkdir(exist_ok=True)
    def launch(index, device):
        folder = study/'runs'/JOBS[index]['run_id']
        if folder.exists():
            m = read(folder/'manifest.json')
            if m['status'] != 'complete' or m['protocol_sha256'] != sha256(study/'protocol.json') or any(sha256(folder/n) != h for n,h in m['artifacts'].items()):
                raise ValueError('Incomplete or changed retained path')
            return
        with (logs/(JOBS[index]['run_id']+'.log')).open('x') as log:
            dispatch_worker([sys.executable, '-B', __file__, 'worker', '--study', str(study),
                            '--index', str(index), '--device', device], cwd=REPO,
                           env=dict(os.environ, PYTHONPATH=child_pythonpath("model"), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2'),
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1100)
        print('COMPLETE', JOBS[index]['run_id'], flush=True)
    # The widest path qualifies this exact workload before the rest; counted once.
    launch(15, 'cuda:0')
    m = read(study/'runs'/JOBS[15]['run_id']/'manifest.json')
    write_json(study/'qualification.json', dict(status='passed', protocol_sha256=sha256(study/'protocol.json'),
               manifest_sha256=sha256(study/'runs'/JOBS[15]['run_id']/'manifest.json'),
               qualification_is_scientific_path=True, additional_updates=0,
               peak_allocated_bytes=m['peak_allocated_bytes'], elapsed_seconds=m['elapsed_seconds']))
    def queue(indices, device):
        for index in indices:
            launch(index, device)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, list(range(15))[d::2], f'cuda:{d}') for d in range(2)]
        for f in futures:
            f.result()


if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(); study = args.study.resolve()
    if study == ROOT or not study.is_relative_to(ROOT):
        raise ValueError('Authorized experiment directory required')
    if args.action == 'prepare': prepare(study)
    elif args.action == 'run': run(study)
    else: worker(study, args.index, args.device)
