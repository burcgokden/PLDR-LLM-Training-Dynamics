#!/usr/bin/env python3
"""Frozen, matched native width/schedule experiment with distinct corpus blocks.

Prepare binds actual input bytes and producer sources. Qualification runs the
widest model twice. Scientific workers cannot override the frozen role, job,
source, or horizon. One process per device, with two devices in the queue.
"""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
from companion_paths import configured_path
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time

import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization
from model_rg.criticality import parameter_digest
from model_rg.provenance import sha256, write_json, environment
from model_rg.replay_equality import require_replay, require_observations, contract

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(configured_path('data:model'))
DEFAULT = ROOT/'matched-onepass-20260914'
CORPUS = ROOT/'data/refinedweb-onepass-524288'
PROBES = ROOT/'controlled-study-20260905/data/short'
NATIVE = ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
SOURCES = ['scripts/run_matched_onepass.py', 'src/model_rg/training.py',
           'src/model_rg/native.py', 'src/model_rg/variance_family.py',
           'src/model_rg/criticality.py', 'src/model_rg/controlled.py',
           'src/model_rg/provenance.py', 'src/model_rg/replay_equality.py']


def rate(step, job):
    if step <= job['warmup']:
        return job['peak']*step/job['warmup']
    if job['schedule'] == 'plateau':
        return job['peak']
    return job['peak']*(.1+.45*(1+math.cos(math.pi*(step-job['warmup'])/
                                         (job['steps']-job['warmup']))))


def prepare(study):
    if study.exists():
        raise FileExistsError(study)
    tokens = np.load(CORPUS/'tokens.npy', mmap_mode='r')
    meta = json.loads((CORPUS/'manifest.json').read_text())
    if tokens.shape != (524288, 513) or tokens.dtype != np.int32:
        raise ValueError('Unexpected corpus layout')
    if sha256(CORPUS/'tokens.npy') != meta['tokens_sha256']:
        raise ValueError('Corpus hash mismatch')
    if sha256(CORPUS/'records.json') != meta['records_sha256']:
        raise ValueError('Corpus records hash mismatch')
    seeds = [914241, 914242, 914243]
    steps = 1024
    selection = {}
    for seed in seeds:
        blocks = np.random.default_rng(seed+700000).permutation(524288*8)[:steps*32]
        selection[f'blocks_{seed}'] = blocks.reshape(steps, 32)
    # New fixed evaluation cohort, outside the training corpus by document.
    pt = np.load(PROBES/'tokens.npy', mmap_mode='r')
    po = np.load(PROBES/'offsets.npy')
    rows = np.arange(768, 896)
    selection['probe_rows'] = rows
    selection['probes'] = pt[rows[:, None], po[rows, None]+np.arange(65)]
    selection['coordinates'] = np.sort(np.random.default_rng(914240).choice(4096, 32, replace=False))
    study.mkdir(parents=True)
    np.savez_compressed(study/'selection.npz', **selection)
    inputs = [CORPUS/'tokens.npy', CORPUS/'records.json', CORPUS/'manifest.json',
              PROBES/'tokens.npy', PROBES/'offsets.npy', PROBES/'manifest.json',
              NATIVE/'modeling_pldrllm.py', NATIVE/'configuration_pldrllm.py']
    jobs = [dict(run_id=f'h{h}-{schedule}-s{seed}', heads=h, seed=seed,
                 schedule=schedule, steps=steps, warmup=128, peak=8e-4)
            for h in [2, 4, 8, 14] for seed in seeds for schedule in ['plateau', 'cosine']]
    protocol = dict(schema='matched-onepass-v1', role='scientific', jobs=jobs,
        input_sha256={str(p): sha256(p) for p in inputs},
        producer_sources={p: sha256(REPO/p) for p in SOURCES},
        selection_sha256=sha256(study/'selection.npz'), environment=environment(),
        objective='One external target after each distinct 64-token prefix; batch 32.',
        initialization='Native five-layer PLDR, head dimension 64, generator width 170, variance normalization.',
        optimizer=dict(betas=[.9, .95], eps=1e-5, weight_decay=.1, clip_value=1., foreach=False),
        arithmetic='float32 native, TF32 disabled, two CPU threads, no mixed precision',
        observations=dict(cadence=32, full_vocabulary_probes=8, evaluation_probes=128,
                          selected_entries_per_head=32),
        memory_ceiling_bytes=20*1024**3,
        primary_plan=dict(
            endpoints=['final external-target NLL', 'final metric row residual ratio',
                       'log-potential motion across the final 256 updates',
                       'categorical metric bounds at aligned observation scales'],
            block_lengths_in_observations=[1, 2, 4, 8, 16, 32],
            metric_tolerance='2e-12*(1+abs(KL)); real-arithmetic bounds evaluated in float64',
            replication='Three initialization/order seeds on one realized corpus; paired schedules per seed.',
            uncertainty='Report each seed and paired range; no population or critical-exponent interval.',
            scope='Finite consuming transients; no stationary criticality or autonomous-reduction claim.'),
        qualification=dict(heads=14, steps=32, replay=True, role='qualification'))
    protocol['replay_contract'] = contract()
    write_json(study/'protocol.json', protocol)
    for name in SOURCES:
        target = study/'executed-source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/name, target)
    print(study/'protocol.json', flush=True)


def admit(study, scientific=False):
    protocol = json.loads((study/'protocol.json').read_text())
    if protocol['schema'] != 'matched-onepass-v1' or protocol['role'] != 'scientific':
        raise ValueError('Wrong protocol')
    for name, digest in protocol['producer_sources'].items():
        if sha256(REPO/name) != digest:
            raise ValueError('Producer changed: '+name)
    for name, digest in protocol['input_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Input changed: '+name)
    if sha256(study/'selection.npz') != protocol['selection_sha256']:
        raise ValueError('Selection changed')
    if scientific:
        q = json.loads((study/'qualification/verification.json').read_text())
        if q['status'] != 'passed' or q['protocol_sha256'] != sha256(study/'protocol.json'):
            raise ValueError('Missing source-bound qualification')
        for path, digest in q['checked_sha256'].items():
            if sha256(path) != digest:
                raise ValueError('Qualification changed')
    return protocol


@torch.no_grad()
def observation(model, probes, coordinates):
    model.model.eval()
    out = model.forward(probes[:8, :64], capture=True)
    metric, base, power = [], [], []
    for a, m, p, *_ in out.pldr_attentions:
        metric.append(a.double())
        base.append(m.double())
        power.append(p.detach().double())
    a = torch.stack(metric, 1)
    b = torch.stack(base, 1).log()
    p = torch.stack(power, 0)
    row = ((a-a.mean(-2, keepdim=True)).square().mean((-2, -1))/
           a.square().mean((-2, -1)).clamp_min(1e-30))
    selected_b = b.flatten(-2).index_select(-1, coordinates)
    selected_p = p.flatten(-2).index_select(-1, coordinates)
    result = dict(logits=out.logits[:, -1].detach().cpu().numpy(),
                  row_ratio=row.cpu().numpy(), logbase=selected_b.cpu().numpy(),
                  power=selected_p.cpu().numpy())
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite observation')
    return result


def train(study, job, device, role, destination):
    protocol = admit(study, scientific=role == 'scientific')
    if role == 'scientific' and job not in protocol['jobs']:
        raise ValueError('Unlisted scientific job')
    destination.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(device)
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    selected = np.load(study/'selection.npz', allow_pickle=False)
    blocks = selected[f'blocks_{job["seed"]}'][:job['steps']]
    if blocks.shape != (job['steps'], 32) or len(np.unique(blocks)) != blocks.size:
        raise ValueError('Invalid one-pass selection')
    token = np.load(CORPUS/'tokens.npy', mmap_mode='r')
    batches = token[blocks[:, :, None]//8, 64*(blocks[:, :, None]%8)+np.arange(65)]
    probes = torch.as_tensor(selected['probes'], dtype=torch.long, device=device)
    coordinates = torch.as_tensor(selected['coordinates'], dtype=torch.long, device=device)
    model = TrainingModel(NATIVE, job['heads'], job['seed'], device)
    normalize_variance_initialization(model)
    initial = parameter_digest(model)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=job['peak'],
        betas=(.9, .95), eps=1e-5, weight_decay=.1, foreach=False)
    parameters = list(model.model.parameters())
    binding = dict(job=job, role=role, protocol_sha256=sha256(study/'protocol.json'),
                   producer_sources=protocol['producer_sources'], device=device,
                   initial_parameter_sha256=initial)
    write_json(destination/'binding.json', binding)
    observations = [observation(model, probes, coordinates)]
    observed_steps = [0]
    evaluation = {}
    losses, rates = [], []

    @torch.no_grad()
    def evaluate(step):
        model.model.eval()
        values = []
        for i in range(0, len(probes), 8):
            z = model.forward(probes[i:i+8, :64]).logits[:, -1].double()
            values.extend(torch.nn.functional.cross_entropy(z, probes[i:i+8, 64],
                          reduction='none').cpu().tolist())
        evaluation[str(step)] = values

    evaluate(0)
    for index in range(job['steps']):
        step = index+1
        lr = rate(step, job)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = torch.as_tensor(batches[index], dtype=torch.long, device=device)
        z = model.forward(batch[:, :64]).logits[:, -1]
        loss = torch.nn.functional.cross_entropy(z, batch[:, 64])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_value_(parameters, 1., foreach=False)
        optimizer.step()
        losses.append(float(loss.detach()))
        rates.append(lr)
        if step % 32 == 0 or step == job['steps']:
            observations.append(observation(model, probes, coordinates))
            observed_steps.append(step)
            if torch.cuda.max_memory_allocated(device) > protocol['memory_ceiling_bytes']:
                raise MemoryError('Declared memory ceiling exceeded')
        if step in {job['warmup'], job['steps']//2, job['steps']}:
            evaluate(step)
        if step % 128 == 0:
            progress = dict(run_id=job['run_id'], step=step, seconds=time.monotonic()-start)
            write_json(destination/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    arrays = {k: np.stack([r[k] for r in observations]) for k in observations[0]}
    np.savez_compressed(destination/'observations.npz', **arrays, steps=observed_steps,
                        blocks=blocks, loss=losses, lr=rates)
    torch.save(dict(model=model.model.state_dict(), optimizer=optimizer.state_dict(),
                    step=job['steps'], job=job, consumed_blocks=blocks,
                    protocol_sha256=binding['protocol_sha256'],
                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device)),
               destination/'final-state.pt')
    manifest = dict(binding, status='complete', completed_steps=job['steps'],
        scientific_updates=job['steps'] if role == 'scientific' else 0,
        final_parameter_sha256=parameter_digest(model), evaluation_nll=evaluation,
        parameter_count=sum(p.numel() for p in parameters), runtime_seconds=time.monotonic()-start,
        max_cuda_memory_bytes=torch.cuda.max_memory_allocated(device),
        artifacts={name: sha256(destination/name) for name in ['observations.npz', 'final-state.pt']})
    write_json(destination/'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['status', 'runtime_seconds', 'max_cuda_memory_bytes']}), flush=True)


def qualify(study, device):
    spec = admit(study)
    job = dict(next(j for j in spec['jobs'] if j['heads'] == 14), steps=32, run_id='widest-profile')
    paths = [study/'qualification'/name for name in ['native', 'replay']]
    for path in paths:
        train(study, job, device, 'qualification', path)
    manifests = [json.loads((p/'manifest.json').read_text()) for p in paths]
    if manifests[0]['final_parameter_sha256'] != manifests[1]['final_parameter_sha256']:
        raise ValueError('Parameter replay mismatch')
    with np.load(paths[0]/'observations.npz') as a, np.load(paths[1]/'observations.npz') as b:
        require_observations(a, b)
    states = [torch.load(p/'final-state.pt', map_location='cpu', weights_only=False) for p in paths]
    require_replay(*states)
    checked = {str(p/name): sha256(p/name) for p in paths
               for name in ['manifest.json', 'observations.npz', 'final-state.pt']}
    write_json(study/'qualification/verification.json', dict(status='passed',
        protocol_sha256=sha256(study/'protocol.json'), checked_sha256=checked,
        replay_contract=contract(), native_updates=64, scientific_updates=0, bitwise_saved_state_replay=True))


def queue(study):
    spec = admit(study, scientific=True)
    # Each device owns one serial queue; subprocess isolation releases GPU state.
    def worker(device, jobs):
        for job in jobs:
            command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                       '--study', str(study), '--run-id', job['run_id'], '--device', device]
            logs = study/'logs'
            logs.mkdir(exist_ok=True)
            with (logs/(job['run_id']+'.log')).open('x') as log:
                dispatch_worker(command, env=dict(os.environ, PYTHONPATH=child_pythonpath("model"),
                    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'), stdout=log,
                    stderr=subprocess.STDOUT, check=True)
            print('COMPLETE '+job['run_id'], flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, f'cuda:{i}', spec['jobs'][i::2]) for i in range(2)]
        for future in futures:
            future.result()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ['prepare', 'qualify', 'run', 'worker']:
        command = actions.add_parser(action)
        command.add_argument('--study', type=Path, default=DEFAULT)
        if action in ['qualify', 'worker']:
            command.add_argument('--device', default='cuda:0')
        if action == 'worker':
            command.add_argument('--run-id', required=True)
    args = parser.parse_args(argv)
    study = args.study.resolve()
    if not study.is_relative_to(ROOT) or study == ROOT:
        raise ValueError('Use a fresh study inside the experiment workdir')
    if args.action == 'prepare':
        prepare(study)
    elif args.action == 'qualify':
        qualify(study, args.device)
    elif args.action == 'run':
        queue(study)
    else:
        spec = admit(study, scientific=True)
        matches = [j for j in spec['jobs'] if j['run_id'] == args.run_id]
        if len(matches) != 1:
            raise ValueError('Expected one frozen scientific job')
        train(study, matches[0], args.device, 'scientific', study/'runs'/args.run_id)


if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__ == '__main__':
    main()
