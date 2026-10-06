#!/usr/bin/env python
"""Paired physical blocking and finite optimizer responses at adapted endpoints."""
from companion_paths import configured_path
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from model_rg.training import TrainingModel
from model_rg.physical_native import context_tokens, spin_tokens, prefix_batch, selected_forward
from model_rg.lattice import dyadic_block, observables
from model_rg.provenance import sha256, write_json

ROOT = Path(configured_path('data:model'))
NAMES = ['scripts/physical_mechanisms.py', 'src/model_rg/physical_native.py',
         'src/model_rg/training.py', 'src/model_rg/native.py', 'src/model_rg/lattice.py',
         'src/model_rg/provenance.py']


def prepare(base):
    base = Path(base).resolve()
    if not base.is_relative_to(ROOT):
        raise ValueError('Unauthorized study path')
    out = base / 'mechanisms'
    out.mkdir(exist_ok=False)
    protocol = json.loads((base / 'confirmation/protocol.json').read_text())
    ds = json.loads((base / 'training-data/manifest.json').read_text())
    old = np.load(base / 'confirmation/draws.npz')
    rng = np.random.default_rng(1910000)
    cell_ids = rng.integers(len(ds['cells']), size=128)
    sites = np.array([rng.integers(ds['cells'][i]['L']**2) for i in cell_ids])
    indices = np.zeros((128, 32), dtype='int64')
    for c in ds['cells']:
        used = old['index'][old['cell'] == c['id']].ravel()
        available = np.setdiff1d(np.arange(c['chains'] * c['samples_per_chain']), used)
        where = np.flatnonzero(cell_ids == c['id'])
        if len(available) < 32 * len(where):
            raise ValueError('No remaining distinct physical configurations')
        indices[where] = rng.permutation(available)[:32 * len(where)].reshape(-1, 32)
    np.savez(out / 'draws.npz', cell=cell_ids, site=sites, index=indices)
    sources = {n: sha256(REPO / n) for n in NAMES}
    for n in sources:
        p = out / 'executed-source' / n
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / n, p)
    cases = [c for c in protocol['cases'] if c['seed'] == 640101 and c['arm'] == 'full']
    endpoints = {}
    for c in cases:
        p = base / 'confirmation' / c['name'] / 'manifest.json'
        m = json.loads(p.read_text())
        if m['status'] != 'complete':
            raise ValueError('Adapted endpoint incomplete')
        endpoints[c['name']] = {'manifest': str(p), 'sha256': sha256(p), 'state': m['checkpoints'][-1]}
    spec = dict(status='frozen', schema='physical-mechanisms-v1', cases=cases,
        endpoints=endpoints, sources=sources, draws_sha256=sha256(out / 'draws.npz'),
        data_manifest_sha256=sha256(base / 'training-data/manifest.json'),
        reference_manifest_sha256=sha256(base / 'assessment-data/manifest.json'),
        blocking='At each critical q=2,3 and L=8,16, four fixed samples from each of 16 independent assessment chains are blocked twice by color-symmetric 2x2 majority with independent random ties. Each exact fine/coarse pair is observed at its last proper prefix. First eight chains are calibration; last eight are held out.',
        response='Scale all residual metric-network parameters by 1+a for a=0,+/-0.001,+/-0.0005, holding incoming Adam memory fixed. Run 128 paired fresh single-pass updates at the terminal learning rate, with identical remaining-configuration draws on every branch. This is a finite directional response, not a native critical-exponent fit.',
        amplitudes=[0., .001, -.001, .0005, -.0005], times=[0,1,2,4,8,16,32,64,128],
        learning_rate=protocol['learning_rate'] * .2,
        random_seed=1920000)
    write_json(out / 'protocol.json', spec)
    print(out, flush=True)


@torch.no_grad()
def observe(adapter, processor, alphabet, x, q, ratio, capture):
    L = x.shape[-1]
    inputs = prefix_batch(x, L*L-1, context_tokens(processor, q, L, ratio), alphabet[:q], adapter.device)
    logits, out = selected_forward(adapter, inputs, alphabet[:q], capture=capture)
    result = {'logits': logits.cpu().numpy()}
    if capture:
        result['hidden'] = torch.stack([h[:, -1] for h in out.hidden_states], 1).cpu().numpy()
        result['A_common'] = torch.stack([a[0].mean(-2) for a in out.pldr_attentions], 1).cpu().numpy()
        result['G_common'] = torch.stack([a[5].mean(-2) for a in out.pldr_attentions], 1).cpu().numpy()
    adapter.capture = False
    adapter.head_outputs = {}
    return result


def worker(base, name, device):
    base = Path(base)
    study = base / 'mechanisms'
    spec = json.loads((study / 'protocol.json').read_text())
    if {n: sha256(REPO / n) for n in NAMES} != spec['sources']:
        raise ValueError('Changed mechanism producer')
    if sha256(study / 'draws.npz') != spec['draws_sha256']:
        raise ValueError('Changed mechanism draws')
    for f, key in [('training-data', 'data_manifest_sha256'), ('assessment-data', 'reference_manifest_sha256')]:
        if sha256(base / f / 'manifest.json') != spec[key]:
            raise ValueError('Changed input manifest')
    case = next(c for c in spec['cases'] if c['name'] == name)
    endpoint = spec['endpoints'][name]
    if sha256(endpoint['manifest']) != endpoint['sha256'] or sha256(endpoint['state']['path']) != endpoint['state']['sha256']:
        raise ValueError('Changed adapted endpoint')
    out = study / name
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device)
    source = ROOT / 'assets/PLDR-LLM-v51-SOC-110M-1'
    adapter = TrainingModel(source, case['heads'], case['seed'], device)
    processor = spm.SentencePieceProcessor(model_file=str(source / 'tokenizer.model'))
    alphabet = spin_tokens(processor)
    state = torch.load(endpoint['state']['path'], map_location='cpu', weights_only=False)
    adapter.model.load_state_dict(state['model'])
    adapter.model.eval()
    reference = json.loads((base / 'assessment-data/manifest.json').read_text())
    cells = [c for c in reference['cells'] if c['L'] in [8,16] and c['temperature_ratio'] == 1.]
    rng = np.random.default_rng(spec['random_seed'])
    contexts = []
    started = time.perf_counter()
    for c in cells:
        if sha256(c['path']) != c['sha256']:
            raise ValueError('Changed blocking input')
        raw = np.memmap(c['path'], mode='r', dtype=np.uint8, shape=tuple(c['shape']))
        x = np.asarray(raw[:, [0,128,256,384]]).reshape(-1,c['L'],c['L'])
        contexts.append((c, x[::4]))
        arrays = {'chain': np.repeat(np.arange(16), 4)}
        for depth in range(3):
            arrays[f'configurations-{depth}'] = x
            for k, v in observe(adapter, processor, alphabet, x, c['q'], 1., True).items():
                arrays[f'{k}-{depth}'] = v
            o = observables(x,c['q'])
            arrays[f'targets-{depth}'] = np.stack([o['field'], o['m2'], o['energy']/x.shape[-1]**2],1)
            if depth < 2:
                x = dyadic_block(x, c['q'], rng)
        np.savez_compressed(out / f'blocking-q{c["q"]}-L{c["L"]}.npz', **arrays)
    ds = json.loads((base / 'training-data/manifest.json').read_text())
    raw = []
    for c in ds['cells']:
        if sha256(c['path']) != c['sha256']:
            raise ValueError('Changed training input')
        raw.append(np.memmap(c['path'], mode='r', dtype=np.uint8,
            shape=(c['chains']*c['samples_per_chain'], c['L'], c['L'])))
    draws = np.load(study / 'draws.npz')
    parameters = list(adapter.model.parameters())
    traces = []
    output = {}
    generator_names = [n for n, _ in adapter.model.named_parameters() if '.reslayerAs.' in n]
    if not generator_names:
        raise ValueError('Metric generator not identified')
    for branch, amplitude in enumerate(spec['amplitudes']):
        adapter.model.load_state_dict(state['model'])
        optimizer = torch.optim.AdamW(parameters, lr=spec['learning_rate'], betas=(.9,.95), eps=1e-8, weight_decay=.01)
        optimizer.load_state_dict(state['optimizer'])
        optimizer.param_groups[0]['lr'] = spec['learning_rate']
        with torch.no_grad():
            for n, p in adapter.model.named_parameters():
                if n in generator_names:
                    p.mul_(1+amplitude)
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'], device)
        for age in range(129):
            if age in spec['times']:
                adapter.model.eval()
                for c, x in contexts:
                    output[f'b{branch}-t{age}-q{c["q"]}-L{c["L"]}'] = observe(adapter,processor,alphabet,x,c['q'],1.,False)['logits']
            if age == 128:
                break
            c = ds['cells'][int(draws['cell'][age])]
            site = int(draws['site'][age])
            x = np.asarray(raw[c['id']][draws['index'][age]])
            inputs = prefix_batch(x, site, context_tokens(processor,c['q'],c['L'],c['temperature_ratio']), alphabet[:c['q']], device)
            target = torch.as_tensor(x.reshape(32,-1)[:,site].astype('int64'),device=device)
            adapter.model.train()
            optimizer.zero_grad(set_to_none=True)
            logits, _ = selected_forward(adapter, inputs, alphabet[:c['q']])
            loss = torch.nn.functional.cross_entropy(logits,target)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite finite-response trajectory')
            optimizer.step()
            traces.append([branch, age+1, float(loss.detach()), float(norm)])
        torch.save(dict(model={n:p.detach().cpu() for n,p in adapter.model.state_dict().items()},
            optimizer=optimizer.state_dict(), amplitude=amplitude, step=128,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)),out/f'endpoint-{branch}.pt')
        del optimizer
        gc.collect()
        print(name, 'response branch', branch, 'complete', flush=True)
    np.savez_compressed(out/'response-logits.npz',**output)
    np.save(out/'response-trace.npy',np.asarray(traces))
    files={str(p.relative_to(out)):sha256(p) for p in out.iterdir() if p.is_file()}
    write_json(out/'manifest.json',dict(status='complete',case=case,
        protocol_sha256=sha256(study/'protocol.json'),files=files,
        completed_updates=5*128,distinct_configurations_per_branch=32*128,
        runtime_seconds=time.perf_counter()-started,peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
    print(name, 'mechanisms complete', flush=True)


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--study',required=True)
    p=sub.add_parser('worker');p.add_argument('--study',required=True);p.add_argument('--name',required=True);p.add_argument('--device',required=True)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.study)
    else:worker(a.study,a.name,a.device)


if __name__=='__main__':main()
