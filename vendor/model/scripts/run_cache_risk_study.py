#!/usr/bin/env python3
"""Frozen finite-panel cache risk and proper-prefix transfer at single-pass endpoints."""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel
from model_rg.inference_interventions import fixed_operators, operator_cache
from numerical_validation import load_json_strict, finite_array
from context_reservations import reserve
from context_execution_contract import attach, admit as shared_admit
from context_categorical_study import ROOT, NATIVE, PARENT, PROBES, CORPUS

REPO = Path(__file__).resolve().parents[1]
SOURCES = ['scripts/run_cache_risk_study.py', 'scripts/analyze_cache_risk_study.py', 'scripts/cache_risk_validation.py',
 'scripts/analyze_operator_cache.py', 'scripts/verify_cache_risk_study.py', 'scripts/context_categorical_study.py',
 'scripts/numerical_validation.py', 'src/model_rg/training.py', 'src/model_rg/native.py',
 'src/model_rg/inference_interventions.py', 'src/model_rg/provenance.py']


def arm_name(length, calibration_length, panel, size):
    return f'L{length}-C{calibration_length}-P{panel}-M{size}'


def arms():
    result = []
    for length in [32, 64, 128]:
        for cl in ([64] if length == 64 else [length, 64]):
            for panel in range(4):
                for size in ([4, 8, 16] if length == 64 else [16]):
                    result.append(dict(name=arm_name(length, cl, panel, size), length=length,
                        calibration_length=cl, panel=panel, size=size))
    return result


def prepare(study, start, reproduce=None):
    if study.exists():
        raise FileExistsError(study)
    receipt = reserve(study, 128, start, reproduce=reproduce)
    rows = np.array(receipt['context_rows']); hashes = receipt['document_hashes']
    exclusions = [Path(x) for x in receipt['imported_protocols']]
    tokens = np.load(PROBES/'tokens.npy', mmap_mode='r')
    blocks = np.array(tokens[rows, :129])
    if blocks.shape != (128, 129):
        raise ValueError('The source does not support all declared prefix lengths')
    parent = load_json_strict((PARENT/'protocol.json').read_text())
    jobs = [dict(j) for j in parent['jobs'] if j['heads'] in [8, 24] and j['control'] in [0, 1.5]]
    if len(jobs) != 24:
        raise ValueError('Incomplete endpoint grid')
    files = [PARENT/'protocol.json', PROBES/'records.json', PROBES/'tokens.npy',
             CORPUS/'records.json', NATIVE/'modeling_pldrllm.py',
             NATIVE/'configuration_pldrllm.py', *exclusions]
    bound = {str(f): sha256(f) for f in files}
    for job in jobs:
        path = PARENT/'runs'/job['run_id']/'manifest.json'
        m = load_json_strict(path.read_text())
        if m['status'] != 'complete' or m['job'] != job:
            raise ValueError('Invalid parent')
        job['checkpoint_sha256'] = m['artifacts']['final-state.pt']
        bound[str(path)] = sha256(path)
    study.mkdir(parents=True)
    np.savez_compressed(study/'inputs.npz', blocks=blocks, rows=rows)
    p = dict(schema='cache-risk-v1', status='frozen_before_acquisition', parent=str(PARENT),
        jobs=jobs, arms=arms(), contexts=64, calibration_panels=4, calibration_contexts=64,
        context_rows=rows.tolist(), document_hashes=hashes, batch_size=8,
        prefix_lengths=[32, 64, 128], vocabulary=32000, source_prefix_start=0,
        cache_rule='Float64 mean of native A, A_LM and G over first m prefixes within each disjoint 16-document panel; cast to float32.',
        target_exclusion='At length L only tokens 0 through L-1 enter either model; token L supplies external NLL only. Calibration documents and assessment documents are disjoint.',
        scientific_hypotheses=['Exact fixed-panel operator risk equals scatter plus squared cache displacement.',
          'Prediction error is state-, prefix-length- and calibration-conditioned; no monotone cache-size or training trend is assumed.',
          'Report centered RMS 0.25 and mean KL 0.03 nats operating targets for every arm; transferred and recalibrated caches are separate arms.'],
        primary_targets=dict(centered_rms=.25, mean_kl_nats=.03),
        selection='All four conditions and six complete seed identities; every arm and context retained.',
        independence='Six paired initialization identities per condition; four fixed calibration panels and 64 assessment documents are not additional training replicas. No iid sampling or population 1/m law is asserted.',
        arithmetic='float32 native, TF32 disabled; float64 operator means and reductions',
        training_updates=0, new_training_replicas=0,
        expected_calls=dict(calibration=576, native_assessment=576, cached_assessment=5376, qualification=336),
        worker_hour_cap=3, memory_ceiling_bytes=22*1024**3,
        input_sha256=bound, selection_sha256=sha256(study/'inputs.npz'),
        source_sha256={n: sha256(REPO/n) for n in SOURCES})
    write_json(study/'protocol.json', p)
    for name in SOURCES:
        dest = study/'executed-source'/name
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((REPO/name).read_bytes())
    attach(study, p, receipt)
    print(json.dumps(dict(study=str(study), protocol_sha256=sha256(study/'protocol.json'),
                         calls=p['expected_calls'])), flush=True)


def admit(study, index=None):
    return shared_admit(study, index)


@torch.no_grad()
def worker(study, index, device):
    if type(index) is not int or device not in ['cuda:0','cuda:1']: raise ValueError('Explicit worker index/device required')
    p = admit(study,index); job = p['jobs'][index]
    out = study/'runs'/job['run_id']; out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    counts = {k: 0 for k in p['expected_calls']}
    m = dict(status='started', job=job, protocol_sha256=sha256(study/'protocol.json'),
             device=device, training_updates=0)
    write_json(out/'manifest.json', m)
    artifacts = []
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        checkpoint = Path(p['parent'])/'runs'/job['run_id']/'final-state.pt'
        if sha256(checkpoint) != job['checkpoint_sha256']:
            raise ValueError('Changed checkpoint')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        model = TrainingModel(NATIVE, job['heads'], job['seed'], device)
        model.model.load_state_dict(state['model'], strict=True)
        model.model.eval().requires_grad_(False); del state
        with np.load(study/'inputs.npz') as a:
            blocks = a['blocks']
        def forward(x, role, capture=False):
            counts[role] += 1
            y = model.forward(x, capture=capture)
            if torch.cuda.max_memory_allocated(device) >= p['memory_ceiling_bytes']:
                raise RuntimeError('Declared memory ceiling reached')
            if time.monotonic()-started >= p['worker_hour_cap']*3600:
                raise RuntimeError('Declared worker cap reached')
            return y
        # Longest prefix, batch one, before any batch-eight acquisition.
        x = torch.as_tensor(blocks[:1, :128].copy(), dtype=torch.long, device=device)
        y = forward(x, 'qualification', True)
        own = operator_cache(y); z = y.logits[:, -1].clone(); del y
        with fixed_operators(model, own):
            check = forward(x, 'qualification').logits[:, -1]
        if not torch.equal(z, check):
            raise ValueError('Longest-prefix own-operator replay differs')
        del own, z, check
        caches = {}; calibration_means = {}
        for length in p['prefix_lengths']:
            tensors = torch.as_tensor(blocks[:, :length].copy(), dtype=torch.long, device=device)
            for panel in range(4):
                pieces = []
                for offset in [panel*16, panel*16+8]:
                    y = forward(tensors[offset:offset+8], 'calibration', True)
                    pieces.append(operator_cache(y).cpu()); del y
                ops = torch.cat(pieces, dim=2); del pieces
                for size in ([4, 8, 16] if length == 64 else [16]):
                    caches[length, panel, size] = ops[:, :, :size].double().mean(2, keepdim=True).float()
                del ops
        for length in p['prefix_lengths']:
            tensors = torch.as_tensor(blocks[:, :length].copy(), dtype=torch.long, device=device)
            # Instrumented exact replay/restoration at every tested length.
            calls = [0]
            def counted(*_): calls[0] += 1
            handles = [layer.register_forward_hook(counted) for dec in model.model.decoder.dec_layers for layer in dec.mha1.reslayerAs]
            y = forward(tensors[:8], 'qualification', True)
            own = operator_cache(y); z = y.logits[:, -1].clone(); del y
            if calls[0] != 40: raise ValueError('Native generator calls differ')
            calls[0] = 0
            with fixed_operators(model, own):
                check = forward(tensors[:8], 'qualification').logits[:, -1]
            if calls[0] or not torch.equal(z, check): raise ValueError('Own-operator replay differs')
            restored = forward(tensors[:8], 'qualification').logits[:, -1]
            if calls[0] != 40 or not torch.equal(z, restored): raise ValueError('Native restoration differs')
            calls[0] = 0
            with fixed_operators(model, caches[length, 0, 16].to(device)):
                check = forward(tensors[:8], 'qualification').logits[:, -1]
            if calls[0] or not torch.isfinite(check).all(): raise ValueError('Generator bypass failed')
            for handle in handles: handle.remove()
            del own, z, check, restored
            native = []; g_parts = []
            for offset in range(64, 128, 8):
                y = forward(tensors[offset:offset+8], 'native_assessment', True)
                native.append(y.logits[:, -1].cpu().numpy())
                g_parts.append(operator_cache(y)[:, 2].cpu()); del y
            name = f'native-L{length}.npy'; np.save(out/name, np.concatenate(native)); artifacts.append(name)
            g = torch.cat(g_parts, dim=1).double(); del g_parts, native
            # Coordinates use a per-layer normalized Frobenius norm, averaged over heads and entries.
            mean = g.mean(1); powers = (g*g).mean(dim=(2, 3, 4))
            scatter = ((g-mean[:, None])**2).mean(dim=(1, 2, 3, 4))
            statistics = dict(mean=mean.numpy(), powers=powers.numpy(), scatter=scatter.numpy())
            for arm in p['arms']:
                if arm['length'] != length: continue
                cache = caches[arm['calibration_length'], arm['panel'], arm['size']]
                cg = cache[:, 2, 0].double()
                statistics[arm['name']+'-risk'] = ((g-cg[:, None])**2).mean(dim=(1, 2, 3, 4)).numpy()
                statistics[arm['name']+'-bias'] = ((mean-cg)**2).mean(dim=(1, 2, 3)).numpy()
                cache_gpu = cache.to(device); predictions = []
                with fixed_operators(model, cache_gpu):
                    for offset in range(64, 128, 8):
                        y = forward(tensors[offset:offset+8], 'cached_assessment')
                        predictions.append(y.logits[:, -1].cpu().numpy()); del y
                name = arm['name']+'.npy'
                a = np.concatenate(predictions); finite_array(a, 'cached logits')
                np.save(out/name, a); artifacts.append(name)
                del predictions, a, cache_gpu
            name = f'operators-L{length}.npz'
            np.savez_compressed(out/name, **statistics); artifacts.append(name)
            del statistics, g, tensors
        name = 'caches.npz'
        np.savez_compressed(out/name, **{f'C{k[0]}-P{k[1]}-M{k[2]}': v.numpy() for k, v in caches.items()})
        artifacts.append(name)
        m.update(status='complete', qualification=dict(exact_own_operator_replay=True,
            exact_native_restoration=True, native_generator_calls=40, cached_generator_calls=0,
            prefix_lengths=p['prefix_lengths'], longest_batch_one=True),
            artifacts={name: sha256(out/name) for name in artifacts},
            checkpoint=str(checkpoint), checkpoint_sha256=job['checkpoint_sha256'])
    except Exception as error:
        m.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        m.update(calls=counts, elapsed_seconds=time.monotonic()-started,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        write_json(out/'manifest.json', m)
    print(json.dumps(dict(run=job['run_id'], seconds=m['elapsed_seconds'], calls=counts)), flush=True)


def run(study):
    p = admit(study)
    for name, digest in p['input_sha256'].items():
        if sha256(name) != digest: raise ValueError('Changed scientific input '+name)
    logs = study/'logs'; logs.mkdir(exist_ok=True)
    def queue(device, indices):
        for index in indices:
            folder = study/'runs'/p['jobs'][index]['run_id']
            if folder.exists():
                m = load_json_strict((folder/'manifest.json').read_text())
                if m['status'] != 'complete' or m['job'] != p['jobs'][index] or m['protocol_sha256'] != sha256(study/'protocol.json'):
                    raise ValueError('Incomplete or foreign retained run')
                if any(sha256(folder/n) != h for n, h in m['artifacts'].items()):
                    raise ValueError('Changed retained artifact')
                continue
            elapsed = sum(load_json_strict(f.read_text()).get('elapsed_seconds', 0)
                          for f in (study/'runs').glob('*/manifest.json'))
            # Leave a bounded allowance for the at-most-two active workers.
            remaining = p['worker_hour_cap']*3600-elapsed
            if remaining <= 0: raise RuntimeError('Declared summed worker budget exhausted')
            with (logs/(p['jobs'][index]['run_id']+'.log')).open('x') as log:
                dispatch_worker([sys.executable, __file__, 'worker', '--study', str(study),
                                '--index', str(index), '--device', device], cwd=REPO,
                    env=dict(os.environ, PYTHONPATH=child_pythonpath("model"), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2'),
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=max(1, remaining/2))
            print('COMPLETE '+p['jobs'][index]['run_id'], flush=True)
    widest = next(i for i, j in enumerate(p['jobs']) if j['heads'] == 24 and j['control'] == 1.5)
    queue('cuda:0', [widest])
    indices = [i for i in range(len(p['jobs'])) if i != widest]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(queue, f'cuda:{d}', indices[d::2]) for d in range(2)]
        for future in futures: future.result()


if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--start',type=int)
    parser.add_argument('--index', type=int); parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reproduce-panel',type=Path)
    a = parser.parse_args(); study = a.study.resolve()
    if study == ROOT or not study.is_relative_to(ROOT):
        raise ValueError('Use the authorized experiment root')
    if a.action == 'prepare': prepare(study, a.start, a.reproduce_panel)
    elif a.action == 'run': run(study)
    else: worker(study, a.index, a.device)
