#!/usr/bin/env python
"""Execute the declared width-time study with dense fields and intermediate Adam states."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run, collective_names, device_name
from model_rg.criticality import (generator_parameter, normalize_initialization, fix_shared_generator, optimizer_for,
                                 parameter_digest, sample_batches, observations, predictive_kl)
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--multiplier', type=float, required=True)
    parser.add_argument('--steps', type=int, default=2048)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume')
    parser.add_argument('--normalization', choices=['fan_in', 'variance'], default='fan_in')
    parser.add_argument('--stream-seed', type=int, default=640001)
    parser.add_argument('--shared-seed', type=int, default=640011)
    parser.add_argument('--protocol', default='pilot.json')
    parser.add_argument('--study', default='criticality-dynamics-20260906')
    parser.add_argument('--probe-every', type=int, default=32)
    args = parser.parse_args()
    args.device = device_name(args.device)
    if args.multiplier < 0 or args.steps < 1 or args.probe_every < 1:
        raise ValueError('Invalid multiplier or horizon')
    root = Path(args.root)
    study = root / args.study
    out = study / 'runs' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'assets/PLDR-LLM-v51-SOC-110M-1'
    data = root / 'data/refinedweb-4608'
    probe = root / 'controlled-study-20260905/data/short'
    inputs = [data/'manifest.json', data/'tokens.npy', probe/'manifest.json', probe/'tokens.npy',
              probe/'offsets.npy', source/'modeling_pldrllm.py', source/'configuration_pldrllm.py',
              study/'protocols'/args.protocol]
    if args.resume:
        inputs.append(Path(args.resume))
    bind_run(out, inputs, vars(args))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokens = np.load(data / 'tokens.npy')
    ptokens = np.load(probe / 'tokens.npy')
    poffsets = np.load(probe / 'offsets.npy')
    crops = ptokens[np.arange(len(ptokens))[:, None], poffsets[:, None] + np.arange(65)]
    rows, offsets, batches = sample_batches(tokens, args.stream_seed, args.steps)
    np.savez_compressed(out/'sampling.npz', rows=rows, offsets=offsets,
                        calibration_rows=np.arange(64), evaluation_rows=np.arange(512, 1024), dense_rows=np.arange(512,528))
    model = TrainingModel(source, args.heads, args.seed, args.device)
    if args.normalization == 'fan_in':
        normalize_initialization(model)
    else:
        normalize_variance_initialization(model)
    shared_digest = fix_shared_generator(model, args.shared_seed)
    initial_digest = parameter_digest(model)
    optimizer = optimizer_for(model, args.multiplier)
    counts = np.zeros(model.config.vocab_size, dtype=np.int64)
    start_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        for key in ['heads', 'seed', 'multiplier', 'stream_seed', 'shared_seed']:
            if saved['arguments'][key] != getattr(args, key):
                raise ValueError(f'Resume changes condition: {key}')
        if saved['arguments'].get('normalization', 'fan_in') != args.normalization:
            raise ValueError('Resume changes normalization')
        model.model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_step = saved['step']
        counts = saved['supervised_target_counts'].numpy().copy()
        if start_step >= args.steps:
            raise ValueError('The resumed horizon must increase')
        del saved
    model.config.to_json_file(str(out/'config.json'))
    milestones = sorted({t for t in [start_step, 128, 512, 1024, 2048, 4096, 8192, args.steps]
                         if start_step <= t <= args.steps})
    raw = {}
    path, path_steps, losses, norms, dense_fields, dense_heads = [], [], [], [], [], []
    saved_states = {}
    named_parameters = list(model.model.named_parameters())
    pool_counts = np.bincount(tokens[:3072].ravel(), minlength=model.config.vocab_size)
    pool_log = np.log((pool_counts + .5) / (pool_counts.sum() + .5 * len(pool_counts)))
    pairs = np.random.default_rng(640071).permutation(512).reshape(-1, 2)
    raw['context_pairs'] = pairs
    started = time.time()
    completed_step = start_step

    def evaluate(selected, matrices=False):
        model.model.eval()
        fields, heads, logits, operators = [], [], [], []
        with torch.no_grad():
            for start in range(0, len(selected), 32):
                batch = torch.tensor(crops[selected[start:start+32]], dtype=torch.long, device=args.device)
                f, h, z, gs = observations(model, batch[:, :64], batch[:, 64], matrices=matrices)
                if not all(torch.isfinite(x).all() for x in [f, h, z]):
                    raise FloatingPointError('Nonfinite observation')
                fields.append(f.cpu().numpy())
                heads.append(h.cpu().numpy())
                logits.append(z.detach().cpu())
                if matrices:
                    operators.append(torch.stack(gs, 1).cpu().numpy())
        return np.concatenate(fields), np.concatenate(heads), torch.cat(logits), (np.concatenate(operators) if matrices else None)

    def save_state(step):
        filename = 'final-training-state.pt' if step == args.steps else f'training-state-{step}.pt'
        checkpoint = dict(model={n:p.detach().cpu().clone() for n,p in model.model.state_dict().items()},
                          optimizer=copy.deepcopy(optimizer.state_dict()), step=step,
                          arguments=vars(args), supervised_target_counts=torch.tensor(counts),
                          initial_parameter_sha256=initial_digest)
        torch.save(checkpoint, out/filename)
        saved_states[str(step)] = dict(filename=filename, sha256=sha256(out/filename))
        del checkpoint

    @torch.no_grad()
    def parameter_statistics(step):
        measured = []
        for name, parameter in named_parameters:
            state = optimizer.state.get(parameter)
            values = [parameter.square().mean().sqrt()]
            if not state:
                zero=torch.zeros((),device=parameter.device)
                values += [zero,zero,zero,torch.ones((),device=parameter.device),zero]
            else:
                first, second = state['exp_avg'], state['exp_avg_sq']
                counter = int(state['step'])
                corrected = second.sqrt() / ((1-.95**counter)**.5)
                ratio = (first/(1-.9**counter)) / (corrected+1e-8)
                rate = 3e-4*(args.multiplier if generator_parameter(name) else 2/args.heads)
                # Algebraic post-update reconstruction; native rounding is not inverted exactly.
                before = (parameter+rate*ratio)/(1-rate*.01)
                delta = parameter-before
                values += [first.square().mean().sqrt(),second.square().mean().sqrt(),
                           ratio.square().mean().sqrt(),(corrected<1e-8).float().mean(),delta.square().mean().sqrt()]
            measured.append(torch.stack(values))
        return torch.stack(measured).cpu().numpy()

    def snapshot(step):
        f, h, _, _ = evaluate(np.arange(512,528))
        path.append(f.mean(0))
        path_steps.append(step)
        dense_fields.append(f)
        dense_heads.append(h)
        if step not in milestones:
            return
        _, _, _, operators = evaluate(np.arange(64), matrices=True)
        ef, eh, logits, _ = evaluate(np.arange(512, 1024))
        raw[f'fields_{step}'] = ef
        raw[f'heads_{step}'] = eh
        lp = logits.double().log_softmax(-1)
        left, right = pairs.T
        target = torch.tensor(crops[512:1024, 64], dtype=torch.long)
        with torch.no_grad():
            pair_kl = .5 * (predictive_kl(logits[left], logits[right]) + predictive_kl(logits[right], logits[left]))
        raw[f'context_kl_{step}'] = pair_kl.numpy()
        raw[f'context_loss_{step}'] = .5 * ((lp[left, target[left]] - lp[right, target[left]]) +
                                           (lp[right, target[right]] - lp[left, target[right]])).numpy()
        matched = np.log((counts + .5) / (counts.sum() + .5 * len(counts)))
        raw[f'matched_nll_{step}'] = -matched[target.numpy()]
        raw[f'pool_nll_{step}'] = -pool_log[target.numpy()]
        mean = operators.mean(0)
        raw[f'operator_dispersion_{step}'] = np.sqrt(np.mean((operators - mean[None])**2, axis=(0,3,4)) /
                                                   np.maximum(np.mean(mean**2, axis=(2,3)), 1e-30))
        print(args.run_id, step, 'nll', float(ef[:, 25].mean()), 'Ictx', float(pair_kl.mean()),
              'Gdisp', float(np.median(raw[f'operator_dispersion_{step}'])),
              'seconds', round(time.time()-started, 1), flush=True)
        raw[f'parameter_statistics_{step}'] = parameter_statistics(step)
        if step in [2048,4096,8192,args.steps] and step > start_step:
            save_state(step)

    status, error = 'complete', None
    try:
        snapshot(start_step)
        for step in range(start_step, args.steps):
            model.model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = torch.tensor(batches[step], dtype=torch.long, device=args.device)
            loss = torch.nn.functional.cross_entropy(model.logits(batch[:, :64]), batch[:, 64])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            group_norms = []
            for group in optimizer.param_groups:
                group_norms.append(torch.linalg.vector_norm(torch.stack([
                    p.grad.norm() for p in group['params'] if p.grad is not None])))
            total = torch.linalg.vector_norm(torch.stack(group_norms))
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite gradient')
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            completed_step = step + 1
            losses.append(float(loss.detach()))
            norms.append([float(total), *[float(x) for x in group_norms]])
            counts += np.bincount(batches[step, :, 64], minlength=len(counts))
            if completed_step % args.probe_every == 0 or completed_step in milestones:
                snapshot(completed_step)
    except FloatingPointError as exc:
        status, error = 'numerical_failure', str(exc)

    raw.update(steps=np.array(path_steps), mean_fields=np.array(path), dense_fields=np.array(dense_fields),
               dense_heads=np.array(dense_heads), losses=np.array(losses),
               gradient_norms=np.array(norms), supervised_target_counts=counts)
    checkpoint_hash = None
    if status == 'complete':
        if str(args.steps) not in saved_states:
            save_state(args.steps)
        checkpoint_hash = saved_states[str(args.steps)]['sha256']
        source_rows = np.arange(512, 576)
        source_batch = torch.tensor(crops[source_rows, :64], dtype=torch.long, device=args.device)
        with torch.no_grad():
            model.model.eval()
            z = model.logits(source_batch).detach().cpu().double()
            perturbations = {}
            for amplitude in [-.01, -.005, .005, .01]:
                eta = torch.full((5, args.heads), amplitude, device=args.device)
                perturbations[amplitude] = model.logits(source_batch, eta=eta).detach().cpu().double()
            probability = z.softmax(-1)
            v = (perturbations[.01]-perturbations[-.01])/.02
            half = (perturbations[.005]-perturbations[-.005])/.01
            v -= (probability*v).sum(-1, keepdim=True)
            half -= (probability*half).sum(-1, keepdim=True)
            raw['source_baseline_logits'] = z.float().numpy()
            raw['source_secant'] = v.float().numpy()
            raw['source_half_secant'] = half.float().numpy()
            raw['source_curvature'] = (probability*v.square()).sum(-1).numpy()
            raw['source_half_curvature'] = (probability*half.square()).sum(-1).numpy()
            raw['source_secant_discrepancy'] = (probability*(v-half).square()).sum(-1).sqrt().numpy()
            raw['source_kl'] = torch.stack([predictive_kl(z, perturbations[e]) for e in [-.01,-.005,.005,.01]]).numpy()
    for name, array in raw.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f'Nonfinite raw array outside declared failure: {name}')
    np.savez_compressed(out/'measurements.npz', **raw)
    manifest = dict(schema='criticality-dynamics-training-v1', status=status, error=error, arguments=vars(args),
                    start_step=start_step, completed_step=completed_step, initial_parameter_sha256=initial_digest,
                    shared_generator_sha256=shared_digest,
                    parameters=sum(p.numel() for p in model.model.parameters()),
                    generator_parameters=[n for n,p in model.model.named_parameters() if generator_parameter(n)],
                    feature_names=collective_names(), head_fields=['attention_entropy/log64','head_output_rms','normalized_row_energy','operator_rms'],
                    milestones=milestones, dense_rows=[512,528], probe_every=args.probe_every,
                    saved_states=saved_states, parameter_names=[n for n,p in named_parameters],
                    parameter_statistics=['weight_rms','first_moment_rms','second_moment_rms','adaptive_ratio_rms','fraction_sqrt_v_below_epsilon','reconstructed_update_rms'],
                    calibration_rows=[0,64], evaluation_rows=[512,1024], source_rows=[512,576],
                    pairing='512 evaluation documents permuted once and partitioned into 256 disjoint pairs; fixed finite-cohort estimates',
                    sampling_sha256=sha256(out/'sampling.npz'), raw_sha256=sha256(out/'measurements.npz'),
                    checkpoint_sha256=checkpoint_hash, binding_sha256=sha256(out/'binding.json'),
                    seconds=time.time()-started, peak_cuda_gb=torch.cuda.max_memory_allocated(args.device)/2**30)
    write_json(out/'manifest.json', manifest)
    print(args.run_id, status, round(manifest['seconds'],1), flush=True)


if __name__ == '__main__':
    main()
