#!/usr/bin/env python
"""Paired physical operator pulses, retaining the complete Adam training state."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run, device_name
from model_rg.criticality import observations, optimizer_for, predictive_kl, sample_batches
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def apply_operator_pulse(model, amplitude):
    """Multiply G by 1+amplitude at fixed layer input, in every head/layer."""
    names = []
    with torch.no_grad():
        for name, parameter in model.model.named_parameters():
            if 'plgatt_layer' in name and name.rsplit('.', 1)[-1] in ('alst', 'balst'):
                parameter.mul_(1 + amplitude)
                names.append(name)
    if len(names) != 10:
        raise AssertionError('The operator pulse must cover both affine coefficients in all five layers')
    return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--updates', type=int, default=512)
    parser.add_argument('--protocol', default='restoration.json')
    parser.add_argument('--mode', choices=['restoration','replay'], default='restoration')
    args = parser.parse_args(); args.device = device_name(args.device)
    root = Path(args.root); study = root/'criticality-study-20260905'
    parent = study/'runs'/args.parent
    out = study/('training-replays' if args.mode=='replay' else 'runs')/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    data = root/'data/refinedweb-4608'; probe = root/'controlled-study-20260905/data/short'
    bind_run(out, [parent/'manifest.json', parent/'final-training-state.pt',
                   data/'manifest.json', data/'tokens.npy', probe/'manifest.json', probe/'tokens.npy',
                   probe/'offsets.npy', source/'modeling_pldrllm.py', source/'configuration_pldrllm.py',
                   study/'protocols'/args.protocol], vars(args))
    saved = torch.load(parent/'final-training-state.pt', map_location='cpu', weights_only=True)
    condition = saved['arguments']; initial_step = saved['step']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokens = np.load(data/'tokens.npy')
    rows, offsets, batches = sample_batches(tokens, condition['stream_seed'], initial_step + args.updates)
    batches = batches[initial_step:]
    np.savez_compressed(out/'sampling.npz', rows=rows[initial_step:], offsets=offsets[initial_step:])
    ptokens = np.load(probe/'tokens.npy'); poffsets = np.load(probe/'offsets.npy')
    probe_rows = np.arange(512, 576)
    crops = ptokens[probe_rows[:, None], poffsets[probe_rows, None] + np.arange(65)]
    points = sorted({x for x in [0, 1, 4, 16, 64, 128, 256, 512, args.updates] if x <= args.updates})
    branches = [('base', 0., False), ('plus_small', .05, False), ('minus_small', -.05, False),
                ('plus', .1, False), ('minus', -.1, False), ('frozen_base', 0., True),
                ('frozen_plus', .1, True), ('frozen_minus', -.1, True)]
    if args.mode=='replay':
        branches = [('base',0.,False),('replay',0.,False)]
    raw = dict(steps=np.array(points), probe_rows=probe_rows)
    baseline = {}; pulse_validation = {}; started = time.time()
    statuses = []; pulse_names = []
    for name, amplitude, freeze in branches:
        model = TrainingModel(source, condition['heads'], condition['seed'], args.device)
        model.model.load_state_dict(saved['model'])
        optimizer = optimizer_for(model, condition['multiplier'])
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        if freeze:
            optimizer.param_groups[0]['lr'] = 0.
        if amplitude:
            pulse_names = apply_operator_pulse(model, amplitude)
        model.model.eval()
        fields, heads, kl, losses, norms = [], [], [], [], []
        reference_name = 'frozen_base' if freeze else 'base'
        finite = True

        def snapshot(step):
            fs, hs, zs = [], [], []
            with torch.no_grad():
                for start in range(0, len(crops), 32):
                    batch = torch.tensor(crops[start:start+32], dtype=torch.long, device=args.device)
                    f, h, z, _ = observations(model, batch[:, :64], batch[:, 64])
                    fs.append(f.cpu().numpy()); hs.append(h.cpu().numpy()); zs.append(z.cpu())
            z = torch.cat(zs)
            if not torch.isfinite(z).all():
                raise FloatingPointError('Nonfinite pulse response')
            fields.append(np.concatenate(fs)); heads.append(np.concatenate(hs))
            if amplitude == 0:
                baseline[(name, step)] = z
                raw[name+f'_logits_{step}'] = z.numpy()
            kl.append(predictive_kl(baseline[(reference_name, step)], z).numpy())
            if step == 0 and amplitude != 0 and not freeze:
                # Check that physical coefficient scaling realizes the recorded
                # inference source, including recomputation of downstream layers.
                reference_model = TrainingModel(source, condition['heads'], condition['seed'], args.device)
                reference_model.model.load_state_dict(saved['model'])
                reference_model.model.eval()
                zz = []
                with torch.no_grad():
                    eta = torch.full((5, condition['heads']), amplitude, device=args.device)
                    for start in range(0, len(crops), 32):
                        batch = torch.tensor(crops[start:start+32, :64], dtype=torch.long, device=args.device)
                        zz.append(reference_model.logits(batch, eta=eta).cpu())
                explicit = torch.cat(zz)
                pulse_validation[name] = dict(max_logit_difference=float((explicit-z).abs().max()),
                                               mean_kl=float(predictive_kl(explicit,z).mean()))
                del reference_model

        try:
            snapshot(0)
            for step in range(args.updates):
                model.model.train(); optimizer.zero_grad(set_to_none=True)
                batch = torch.tensor(batches[step], dtype=torch.long, device=args.device)
                loss = torch.nn.functional.cross_entropy(model.logits(batch[:, :64]), batch[:, 64])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss)); norms.append(float(norm))
                if step + 1 in points:
                    model.model.eval(); snapshot(step+1)
        except FloatingPointError:
            finite = False
        raw[name+'_fields'] = np.array(fields); raw[name+'_heads'] = np.array(heads)
        raw[name+'_kl'] = np.array(kl); raw[name+'_losses'] = np.array(losses); raw[name+'_norms'] = np.array(norms)
        statuses.append(dict(branch=name, amplitude=amplitude, freeze_generator=freeze,
                             status='complete' if finite else 'numerical_failure', completed_updates=len(losses)))
        print(args.run_id, name, statuses[-1]['status'], round(time.time()-started,1), flush=True)
        del optimizer, model
        torch.cuda.empty_cache()
        if not finite and amplitude == 0:
            break
    for name, values in raw.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f'Nonfinite saved restoration array: {name}')
    np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(schema='training-replay-v1' if args.mode=='replay' else 'training-restoration-v1', arguments=vars(args), condition=condition,
        status='complete' if len(statuses)==len(branches) and all(s['status']=='complete' for s in statuses) else 'numerical_failure',
        branches=statuses, pulse_parameters=pulse_names, pulse_equivalence=pulse_validation,
        initial_step=initial_step, observation_steps=points, seconds=time.time()-started,
        peak_cuda_gb=torch.cuda.max_memory_allocated(args.device)/2**30 if str(args.device).startswith('cuda') else 0.,
        binding_sha256=sha256(out/'binding.json'), sampling_sha256=sha256(out/'sampling.npz'),
        raw_sha256=sha256(out/'measurements.npz')))


if __name__ == '__main__':
    main()
