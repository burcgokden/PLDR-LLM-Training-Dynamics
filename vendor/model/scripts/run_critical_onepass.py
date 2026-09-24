#!/usr/bin/env python3
"""Frozen single-pass native criticality family; preparation is the only design input."""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
from companion_paths import legacy_path
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
from model_rg.criticality import fix_shared_generator, generator_parameter, optimizer_for, parameter_digest
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization
from model_rg.provenance import environment, sha256, write_json
from model_rg.replay_equality import logical_equal as same, require_replay, require_observations, finite_state, contract
from model_rg.run_outcome import start_record, record_failure, save_artifact, clip_finite_norm, inventory

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(legacy_path('/pldr-data/model'))
CORPUS = ROOT/'data/refinedweb-onepass-524288'
PROBES = ROOT/'controlled-study-20260905/data/short'
NATIVE = ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
SOURCES = ['scripts/run_critical_onepass.py', 'src/model_rg/criticality.py',
           'src/model_rg/training.py', 'src/model_rg/native.py',
           'src/model_rg/variance_family.py', 'src/model_rg/controlled.py',
           'src/model_rg/provenance.py', 'src/model_rg/replay_equality.py', 'src/model_rg/run_outcome.py']


def validate_design(d):
    required = {'name', 'heads', 'controls', 'seeds', 'environments', 'steps', 'shared_seed'}
    if set(d) != required:
        raise ValueError('The design must contain exactly '+str(sorted(required)))
    for name in ['heads', 'controls', 'seeds', 'environments']:
        if not d[name] or len(set(d[name])) != len(d[name]):
            raise ValueError('Empty or duplicated design axis: '+name)
    if any(type(n) is not int or n < 2 or n > 32 for n in d['heads']):
        raise ValueError('Unsupported head count')
    if any(not math.isfinite(g) or g < 0 or g > 16 for g in d['controls']):
        raise ValueError('Unsupported control')
    if any(type(s) is not int or s < 0 for s in d['seeds']):
        raise ValueError('Unsupported seed')
    if not set(d['environments']) <= {0, 1}:
        raise ValueError('The source partitions are 0 and 1')
    if type(d['steps']) is not int or d['steps'] < 256 or d['steps'] > 32768 or d['steps'] % 256:
        raise ValueError('Horizon must be 256..32768, in multiples of 256')
    if type(d['shared_seed']) is not int:
        raise ValueError('Shared seed must be an integer')
    if not isinstance(d['name'], str) or not d['name']:
        raise ValueError('The stage needs a name')


def prepare(study, design_path):
    d = json.loads(design_path.read_text())
    validate_design(d)
    if study.exists():
        raise FileExistsError(study)
    tokens = np.load(CORPUS/'tokens.npy', mmap_mode='r')
    if tokens.shape != (524288, 513) or tokens.dtype != np.int32:
        raise ValueError('Unexpected master corpus')
    records = json.loads((CORPUS/'records.json').read_text())
    heldout = json.loads((PROBES/'records.json').read_text())
    hashes = [r['content_sha256'] for r in records]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Master corpus repeats a document')
    probe_rows = np.arange(1024, 1152)
    if set(hashes) & {heldout[i]['content_sha256'] for i in probe_rows}:
        raise ValueError('Evaluation document occurs in the master corpus')
    selection = {'probe_rows': probe_rows}
    pt = np.load(PROBES/'tokens.npy', mmap_mode='r')
    po = np.load(PROBES/'offsets.npy')
    selection['probes'] = pt[probe_rows[:, None], po[probe_rows, None]+np.arange(65)]
    populations = np.random.default_rng(9152501).permutation(524288).reshape(2, -1)
    for e in d['environments']:
        population = populations[e]
        local = np.random.default_rng(9152502+e).permutation(len(population)*8)[:32*d['steps']]
        blocks = 8*population[local//8]+local % 8
        selection[f'population_{e}'] = population
        selection[f'blocks_{e}'] = blocks.reshape(-1, 32)
    inputs = [CORPUS/'tokens.npy', CORPUS/'records.json', CORPUS/'manifest.json',
              PROBES/'tokens.npy', PROBES/'offsets.npy', PROBES/'records.json', PROBES/'manifest.json',
              NATIVE/'modeling_pldrllm.py', NATIVE/'configuration_pldrllm.py']
    jobs = [dict(run_id=f'e{e}-h{h}-g{g:g}-s{s}', environment=e, heads=h,
                 control=g, seed=s, steps=d['steps'], shared_seed=d['shared_seed'],
                 save_optimizer=s in d['seeds'][:2])
            for e in d['environments'] for g in d['controls'] for h in d['heads'] for s in d['seeds']]
    study.mkdir(parents=True)
    np.savez_compressed(study/'selection.npz', **selection)
    protocol = dict(schema='critical-onepass-v1', role='scientific', design=d, jobs=jobs,
        source_sha256={name: sha256(REPO/name) for name in SOURCES},
        input_sha256={str(p): sha256(p) for p in inputs},
        selection_sha256=sha256(study/'selection.npz'), design_sha256=sha256(design_path),
        environment=environment(),
        architecture=dict(depth=5, head_dimension=64, generator_width=170, generator_residual_units=8),
        initialization='Native variance normalization, then fixed initial shared generator; remaining initialization varies by seed.',
        optimizer=dict(generator_rate='0.0003*g', other_rate='0.0003*2/N', betas=[.9, .95],
                       epsilon=1e-8, weight_decay=.01, global_norm_clip=1., foreach=False, schedule='constant'),
        source=dict(master_documents=524288, population_documents=262144, population_blocks=2097152,
                    batch=32, prefix_tokens=64, target_tokens=1, no_repeated_supervised_positions=True,
                    partition_seed=9152501, stream_seed='9152502+environment',
                    conditioning='Fixed master RefinedWeb corpus; disjoint document partitions are source robustness, not independent master-corpus draws.'),
        observations=dict(cadence=64, head_contexts=64, evaluation_contexts=128,
                          full_vocabulary_contexts=8, row_denominator_floor=1e-30,
                          milestone_steps=sorted({0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768} & set(range(d['steps']+1)))),
        arithmetic=dict(dtype='float32', tf32=False, observation_reductions='float64', cpu_threads=2),
        primary_plan=dict(field='context-conditioned all-entry row energy; attention entropy is a separate secondary field',
            replication='Independent wide initializations at fixed source order and initial shared generator; contexts and times are not seed replicates.',
            first_stage='Complete all declared controls, report every seed and the full size/time grid.',
            region='Seek an interior susceptibility maximum, a resolved change in the mean field, and reproducibility under fresh seeds before selecting a finer bracket.',
            scaling='Distinguish a finite-time crossover from a positive limiting critical control; compare g*T, horizon drift, regular-plus-shared variance and singular scaling with withheld sizes.',
            exponents='No universal exponent is accepted from a selected peak or a fitted slope alone. Require independent bracket replication, stable size/time fits, predictive checks and a justified limiting law.',
            failures='Retain every numerical failure and all selection decisions; never substitute successful runs for failed seeds.'),
        checkpoint_policy='Every final model and RNG state; full Adam states for the first two predeclared seeds per cell.',
        qualification=dict(steps=128, heads=max(d['heads']), control=2., full_state_replay=True,
                           role='qualification'), memory_ceiling_bytes=22*1024**3)
    protocol['replay_contract'] = contract()
    write_json(study/'protocol.json', protocol)
    for name in SOURCES:
        target = study/'executed-source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/name, target)
    print(json.dumps({'protocol': str(study/'protocol.json'), 'jobs': len(jobs)}), flush=True)


def admit(study, scientific=False):
    p = json.loads((study/'protocol.json').read_text())
    if p['schema'] != 'critical-onepass-v1' or p['role'] != 'scientific':
        raise ValueError('Wrong protocol or role')
    validate_design(p['design'])
    for name, digest in p['source_sha256'].items():
        if sha256(REPO/name) != digest:
            raise ValueError('Changed producer: '+name)
    for name, digest in p['input_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Changed input: '+name)
    if sha256(study/'selection.npz') != p['selection_sha256']:
        raise ValueError('Changed source selection')
    if scientific:
        q = json.loads((study/'qualification/verification.json').read_text())
        if q['status'] != 'passed' or q['protocol_sha256'] != sha256(study/'protocol.json'):
            raise ValueError('Missing qualified protocol')
        for name, digest in q['checked_sha256'].items():
            if sha256(name) != digest:
                raise ValueError('Changed qualification artifact')
    return p


@torch.no_grad()
def observe(model, probes, milestone):
    model.model.eval()
    heads, nll, logits = [], [], []
    for start in range(0, 128 if milestone else 64, 16):
        batch = probes[start:start+16]
        out = model.forward(batch[:, :64], capture=True)
        z = out.logits[:, -1].double()
        nll.extend(torch.nn.functional.cross_entropy(z, batch[:, 64], reduction='none').cpu().tolist())
        if start == 0 and milestone:
            logits = z[:8].float().cpu().numpy()
        if start >= 64:
            continue
        per_layer = []
        for layer, values in enumerate(out.pldr_attentions):
            a, _, _, _, _, operator, weights = values
            a = a.double()
            attention = weights[:, :, -1].double()
            entropy = -(attention*attention.clamp_min(1e-300).log()).sum(-1)/math.log(64)
            energy = a.square().mean((-2, -1))
            row = (a-a.mean(-2, keepdim=True)).square().mean((-2, -1))/energy.clamp_min(1e-30)
            op = operator.double().square().mean((-2, -1)).sqrt()
            per_layer.append(torch.stack([row, entropy, op, energy], -1))
        heads.append(torch.stack(per_layer, 1).cpu().numpy())
    result = dict(heads=np.concatenate(heads), nll=np.array(nll))
    if milestone:
        result['logits'] = logits
    if not all(np.isfinite(a).all() for a in result.values()):
        raise FloatingPointError('Nonfinite native observation')
    if np.any(result['heads'][..., :2] < -1e-12) or np.any(result['heads'][..., :2] > 1+1e-6):
        raise FloatingPointError('Bounded head field exceeds its declared units')
    return result


def train(study, job, device, role, destination, after_update=None):
    started = time.monotonic()
    protocol = admit(study, scientific=role == 'scientific')
    if role not in ['scientific', 'qualification']:
        raise ValueError('Invalid native role')
    if role == 'scientific' and job not in protocol['jobs']:
        raise ValueError('Unlisted job')
    destination.mkdir(parents=True, exist_ok=False)
    binding = dict(job=job, role=role, protocol_sha256=sha256(study/'protocol.json'),
                   producer_sha256=protocol['source_sha256'], device=device,
                   intended_steps=job['steps'])
    m = start_record(destination, binding)
    model = optimizer = blocks = None
    params = []
    times, fields, nll_path, losses, norms = [], [], [], [], []
    raw = {}
    native_start = time.monotonic()
    pending = None
    def snapshot(step):
        m['stage'] = 'observation'
        result = observe(model, probes, step in milestones)
        # Publish a snapshot only after all required fields are acquired.
        field, nll = result['heads'], result['nll'][:64]
        if step in milestones:
            logits, evaluation = result['logits'], result['nll']
            raw[f'logits_{step}'] = logits
            raw[f'evaluation_nll_{step}'] = evaluation
        times.append(step); fields.append(field); nll_path.append(nll)
    try:
        m['stage'] = 'cuda_initialization'
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
        m['stage'] = 'source_loading'
        with np.load(study/'selection.npz', allow_pickle=False) as selected:
            blocks = selected[f'blocks_{job["environment"]}'][:job['steps']]
            if blocks.shape != (job['steps'], 32) or len(np.unique(blocks)) != blocks.size:
                raise ValueError('Invalid one-pass stream')
            probes = torch.as_tensor(selected['probes'], dtype=torch.long, device=device)
        tokens = np.load(CORPUS/'tokens.npy', mmap_mode='r')
        m['stage'] = 'model_initialization'
        model = TrainingModel(NATIVE, job['heads'], job['seed'], device)
        normalize_variance_initialization(model)
        m['initial_shared_sha256'] = fix_shared_generator(model, job['shared_seed'])
        m['initial_parameter_sha256'] = parameter_digest(model)
        m['stage'] = 'optimizer_initialization'
        optimizer = optimizer_for(model, job['control'])
        params = list(model.model.parameters())
        write_json(destination/'binding.json', {k: m[k] for k in [*binding, 'initial_shared_sha256', 'initial_parameter_sha256']})
        milestones = set(protocol['observations']['milestone_steps']) | {0, job['steps']}
        native_start = time.monotonic()
        snapshot(0)
        for index in range(job['steps']):
            m['attempted_steps'] = index+1
            m['stage'] = 'forward_backward'
            b = blocks[index]
            ids = tokens[b[:, None]//8, 64*(b[:, None] % 8)+np.arange(65)]
            batch = torch.as_tensor(ids, dtype=torch.long, device=device)
            model.model.train()
            optimizer.zero_grad(set_to_none=True)
            z = model.forward(batch[:, :64]).logits[:, -1]
            loss = torch.nn.functional.cross_entropy(z, batch[:, 64])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            m['stage'] = 'gradient_clipping'
            norm = clip_finite_norm(params)
            m['stage'] = 'optimizer_step'
            m['optimizer_partial_mutation'] = True
            optimizer.step()
            m['completed_steps'] = index+1
            m['optimizer_partial_mutation'] = False
            losses.append(float(loss.detach())); norms.append(float(norm))
            m['stage'] = 'passive_observation'
            if after_update is not None:
                after_update(m['completed_steps'])
            if m['completed_steps'] % 64 == 0 or m['completed_steps'] == job['steps']:
                snapshot(m['completed_steps'])
            m['stage'] = 'resource_check'
            if torch.cuda.max_memory_allocated(device) > protocol['memory_ceiling_bytes']:
                raise MemoryError('Declared memory ceiling exceeded')
            if m['completed_steps'] % 256 == 0:
                write_json(destination/'progress.json', dict(run_id=job['run_id'], step=m['completed_steps'],
                    target_steps=job['steps'], elapsed_seconds=time.monotonic()-started,
                    mean_row=float(fields[-1][..., 0].mean()), mean_nll=float(np.mean(nll_path[-1]))))
        m['stage'] = 'finite_state_admission'
        if not finite_state(model.model.state_dict()) or not finite_state(optimizer.state_dict()):
            raise FloatingPointError('Nonfinite final model or optimizer state')
        m['status'] = 'finalizing'
    except Exception as exc:
        record_failure(m, exc)
        pending = exc if m['status'] == 'program_error' else None
    completed = m['completed_steps']
    m.update(scientific_updates=completed if role == 'scientific' else 0,
             qualification_updates=completed if role == 'qualification' else 0,
             available_snapshot_steps=list(times), acquired_observations=len(times),
             source_blocks=completed*32, runtime_seconds=time.monotonic()-started,
             native_seconds=time.monotonic()-native_start, parameter_count=sum(p.numel() for p in params))
    # Commit the failure record before any optional large serialization.
    write_json(destination/'manifest.json', m)
    m['stage'] = 'observation_save'
    def save_observations():
        raw.update(steps=np.asarray(times, dtype=np.int64),
            heads=np.stack(fields) if fields else np.empty((0, 64, 5, job['heads'], 4)),
            nll_path=np.stack(nll_path) if nll_path else np.empty((0, 64)),
            blocks=blocks[:completed] if blocks is not None else np.empty((0, 32), dtype=np.int64),
            training_loss=np.asarray(losses), gradient_norm=np.asarray(norms))
        np.savez_compressed(destination/'observations.npz', **raw)
    exc = save_artifact(destination, m, 'observations.npz', save_observations)
    if exc is not None and m['status'] == 'program_error' and pending is None: pending = exc
    if model is not None:
        m['stage'] = 'checkpoint_save'
        def save_checkpoint():
            checkpoint = dict(model={n: p.detach().cpu() for n, p in model.model.state_dict().items()},
                step=completed, job=job, protocol_sha256=binding['protocol_sha256'],
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device))
            if job['save_optimizer'] and optimizer is not None:
                checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, destination/'final-state.pt')
            m.update(final_parameter_sha256=parameter_digest(model), checkpoint_available=True,
                     optimizer_saved='optimizer' in checkpoint)
        exc = save_artifact(destination, m, 'final-state.pt', save_checkpoint)
        if exc is not None and m['status'] == 'program_error' and pending is None: pending = exc
    if m['original_failure'] is None:
        m.update(status='complete', stage='complete', checkpoint_resumable=bool(job['save_optimizer']))
    else:
        m['stage'] = m['original_failure']['stage']
    m['runtime_seconds'] = time.monotonic()-started
    try:
        m['maximum_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(device)
    except Exception:
        m['maximum_cuda_memory_bytes'] = None
    write_json(destination/'manifest.json', m)
    print(json.dumps({k:m[k] for k in ['status','completed_steps','runtime_seconds']}), flush=True)
    if pending is not None:
        raise pending.with_traceback(pending.__traceback__)
    return m


def qualify(study, device):
    p = admit(study)
    q = p['qualification']
    job = dict(p['jobs'][0], run_id='widest-qualification', heads=q['heads'], control=q['control'],
               steps=q['steps'], save_optimizer=True)
    paths = [study/'qualification'/name for name in ['native', 'replay']]
    results = [train(study, job, device, 'qualification', path) for path in paths]
    if any(r['status'] != 'complete' for r in results):
        raise ValueError('Native qualification did not finish')
    states = [torch.load(path/'final-state.pt', map_location='cpu', weights_only=False) for path in paths]
    require_replay(*states)
    with np.load(paths[0]/'observations.npz') as a, np.load(paths[1]/'observations.npz') as b:
        require_observations(a, b)
    checked = {str(path/name): sha256(path/name) for path in paths
               for name in ['manifest.json', 'final-state.pt', 'observations.npz']}
    write_json(study/'qualification/verification.json', dict(status='passed',
        protocol_sha256=sha256(study/'protocol.json'), checked_sha256=checked,
        replay_contract=contract(), full_state_bitwise_replay=True, scientific_updates=0, qualification_updates=2*q['steps']))
    print('Native full-state qualification passed', flush=True)


def run(study):
    p = admit(study, scientific=True)
    def worker(device, jobs):
        for job in jobs:
            path = study/'runs'/job['run_id']
            if path.exists():
                mpath = path/'manifest.json'
                if not mpath.exists(): raise ValueError('Unfinished collision: '+str(path))
                m = json.loads(mpath.read_text())
                if m['job'] != job or m['protocol_sha256'] != sha256(study/'protocol.json'):
                    raise ValueError('Foreign existing run')
                for name, digest in m['artifacts'].items():
                    if sha256(path/name) != digest: raise ValueError('Existing artifact changed')
                print('RETAIN '+job['run_id']+' '+m['status'], flush=True)
                continue
            logs = study/'logs'
            logs.mkdir(exist_ok=True)
            with (logs/(job['run_id']+'.log')).open('x') as log:
                dispatch_worker([sys.executable, str(REPO/'scripts/run_critical_onepass.py'), 'worker',
                    '--study', str(study), '--run-id', job['run_id'], '--device', device],
                    env=dict(os.environ, PYTHONPATH=child_pythonpath("model"), OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            print('COMPLETE '+job['run_id'], flush=True)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, f'cuda:{i}', p['jobs'][i::2]) for i in range(2)]
            for f in futures: f.result()
    finally:
        outcome = inventory(study, p)
    write_json(study/'execution-complete.json', outcome)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for action in ['prepare', 'qualify', 'run', 'worker']:
        c = actions.add_parser(action)
        c.add_argument('--study', type=Path, required=True)
        if action == 'prepare': c.add_argument('--design', type=Path, required=True)
        if action in ['qualify', 'worker']: c.add_argument('--device', default='cuda:0')
        if action == 'worker': c.add_argument('--run-id', required=True)
    a = parser.parse_args(argv)
    study = a.study.resolve()
    if not study.is_relative_to(ROOT) or study == ROOT:
        raise ValueError('Use a study directory inside the authorized experiment root')
    if a.action == 'prepare': prepare(study, a.design)
    elif a.action == 'qualify': qualify(study, a.device)
    elif a.action == 'run': run(study)
    else:
        p = admit(study, scientific=True)
        selected = [j for j in p['jobs'] if j['run_id'] == a.run_id]
        if len(selected) != 1: raise ValueError('Expected one frozen scientific job')
        train(study, selected[0], a.device, 'scientific', study/'runs'/a.run_id)


if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__ == '__main__':
    main()
