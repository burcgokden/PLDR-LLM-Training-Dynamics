#!/usr/bin/env python
"""Execute a selected single-pass training trajectory with complete native state."""
import argparse
import copy
import json
import os
import signal
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run, collective_names, device_name
from model_rg.criticality import (generator_parameter, normalize_initialization, fix_shared_generator,
                                 parameter_digest, predictive_kl)
from model_rg.scaling import observations
from model_rg.onepass_regimes import recipe, RECIPE_NAMES, sample_batches, multiplier, optimizer_and_scheduler
from model_rg.schedules import loss as recipe_loss, clip as recipe_clip
from model_rg.checkpointing import enable_decoder_checkpointing
from model_rg.update_records import SharedUpdateRecorder
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--heads', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--recipe', choices=RECIPE_NAMES, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--schedule-horizon', type=int, required=True)
    parser.add_argument('--warmup-override', type=int, default=-1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume')
    parser.add_argument('--normalization', choices=['fan_in', 'variance'], default='variance')
    parser.add_argument('--stream-seed', type=int, default=640001)
    parser.add_argument('--shared-seed', type=int, default=640011)
    parser.add_argument('--protocol', default='onepass-training-selection.json')
    parser.add_argument('--study', default='scheduled-training-feasible-20260908')
    parser.add_argument('--probe-every', type=int, default=64)
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument('--checkpoint-decoders', action='store_true')
    parser.add_argument('--save-steps', default='', help='Additional full-state horizons; final state is always saved')
    args = parser.parse_args()
    args.device = device_name(args.device)
    if args.steps < 1 or args.probe_every < 1 or args.microbatch != 32:
        raise ValueError('This adaptation uses a full native batch of32')
    profile=recipe(args.recipe,args.heads,args.schedule_horizon,args.warmup_override)
    root = Path(args.root)
    study = root / args.study
    out = study / 'runs' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    source = root / 'assets/PLDR-LLM-v51-SOC-110M-1'
    data = root / 'data/refinedweb-onepass-524288'
    probe = root / 'controlled-study-20260905/data/short'
    inputs = [data/'manifest.json', data/'tokens.npy', probe/'manifest.json', probe/'tokens.npy',
              probe/'offsets.npy', source/'modeling_pldrllm.py', source/'configuration_pldrllm.py',
              study/'protocols'/args.protocol]
    spec=json.loads((study/'protocols'/args.protocol).read_text())
    jobs=[j for j in spec['jobs'] if j['run_id']==args.run_id]
    if len(jobs)!=1:raise AssertionError('A unique frozen trajectory is required')
    job=jobs[0]
    repo=Path(__file__).resolve().parents[1]
    for name,signature in spec['producer_sources'].items():
        if sha256(repo/name)!=signature:raise AssertionError('The frozen training producer changed: '+name)
    for key in ['heads','seed','recipe','steps','shared_seed','stream_seed','probe_every','microbatch','save_steps','checkpoint_decoders','schedule_horizon','warmup_override']:
        if job[key]!=getattr(args,key):raise AssertionError('The selected condition changed: '+key)
    if job['profile']!=profile:
        raise AssertionError('The selected optimizer or schedule changed')
    if args.normalization!='variance':raise AssertionError('Use the declared variance initialization')
    for filename,signature in spec['reference_inputs'].items():
        if sha256(filename)!=signature:raise AssertionError('A reference or qualification changed')
        inputs.append(Path(filename))
    if args.resume:
        inputs.append(Path(args.resume))
        if not job.get('resume') or Path(job['resume']).resolve()!=Path(args.resume).resolve():
            raise AssertionError('The resumed parent was not selected')
        if sha256(args.resume)!=job['parent_sha256']:raise AssertionError('The selected parent changed')
    elif job.get('resume'):
        raise AssertionError('A selected continuation requires its full parent')
    update_protocol = study/'protocols/onepass-shared-update-observation.json'
    update_spec = None
    if update_protocol.exists():
        candidate = json.loads(update_protocol.read_text())
        if args.run_id in candidate['run_ids']:
            update_spec = candidate
            projection_path = Path(candidate['projection'])
            if sha256(projection_path) != candidate['projection_sha256']:
                raise AssertionError('The fixed shared projection changed')
            if sha256(candidate['qualification']) != candidate['qualification_sha256']:
                raise AssertionError('The passive observation qualification changed')
            inputs.extend([update_protocol, projection_path, Path(candidate['qualification'])])
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
    if args.checkpoint_decoders:
        enable_decoder_checkpointing(model)
    optimizer,scheduler = optimizer_and_scheduler(model,profile)
    counts = np.zeros(model.config.vocab_size, dtype=np.int64)
    start_step = 0
    normalization_alias = None
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        for key in ['heads', 'seed', 'recipe', 'stream_seed', 'shared_seed', 'schedule_horizon', 'warmup_override']:
            if saved['arguments'][key] != getattr(args, key):
                raise ValueError(f'Resume changes condition: {key}')
        parent_normalization = saved['arguments'].get('normalization', 'fan_in')
        if parent_normalization != args.normalization:
            if (args.heads == 2 and {parent_normalization, args.normalization} == {'fan_in', 'variance'}
                    and saved['initial_parameter_sha256'] == initial_digest):
                normalization_alias = dict(parent=parent_normalization, current=args.normalization,
                    justification='Both normalizations are the identity at N=2; the complete initial parameter digest agrees.')
            else:
                raise ValueError('Resume changes normalization')
        model.model.load_state_dict(saved['model'])
        if saved['recipe']!=profile:raise AssertionError('Resume changes the drive or optimizer')
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start_step = saved['step']
        if scheduler.last_epoch!=start_step:raise AssertionError('Resume loses the scheduler phase')
        expected=[profile[k]*multiplier(start_step,profile['warmup_steps'],profile['total_steps'],profile['floor_fraction'])
                  for k in ['generator_peak','body_peak']]
        if [g['lr'] for g in optimizer.param_groups]!=expected:
            raise AssertionError('Resume loses the next applied learning rates')
        counts = saved['supervised_target_counts'].numpy().copy()
        if start_step >= args.steps:
            raise ValueError('The resumed horizon must increase')
        del saved
    update_recorder = (SharedUpdateRecorder(model.model, optimizer, update_spec['projection'],
                        update_spec['parameter_names']) if update_spec else None)
    model.config.to_json_file(str(out/'config.json'))
    milestones = sorted({t for t in [start_step, profile['warmup_steps'], profile['total_steps'], 128, 512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144, args.steps]
                         if start_step <= t <= args.steps})
    save_steps={int(x) for x in args.save_steps.split(',') if x}
    if not save_steps.issubset(set(milestones)):
        raise AssertionError('Every selected saved state must be an observation milestone')
    raw = {}
    dense_centroids, dense_energies = [], []
    path, path_steps, losses, norms, dense_fields, dense_heads = [], [], [], [], [], []
    saved_states = {}
    learning_rates=[];supervised_counts=[]
    named_parameters = list(model.model.named_parameters())
    pool_targets = tokens[:,1:] if profile['objective']=='all_nonpadding_targets' else tokens[:,64::64]
    if profile['objective']=='all_nonpadding_targets': pool_targets=pool_targets[pool_targets!=0]
    pool_counts = np.bincount(pool_targets.ravel(), minlength=model.config.vocab_size)
    del pool_targets
    pool_log = np.log((pool_counts + .5) / (pool_counts.sum() + .5 * len(pool_counts)))
    pairs = np.random.default_rng(640071).permutation(512).reshape(-1, 2)
    raw['context_pairs'] = pairs
    started = time.time()
    completed_step = start_step

    def evaluate(selected, matrices=False):
        model.model.eval()
        fields, heads, logits, operators = [], [], [], []
        extras = {}
        with torch.no_grad():
            for start in range(0, len(selected), 32):
                batch = torch.tensor(crops[selected[start:start+32]], dtype=torch.long, device=args.device)
                f, h, z, gs, extra = observations(model, batch[:, :64], batch[:, 64], matrices=matrices)
                for name, value in extra.items():
                    extras.setdefault(name, []).append(value.detach().cpu().numpy())
                if not all(torch.isfinite(x).all() for x in [f, h, z]):
                    raise FloatingPointError('Nonfinite observation')
                fields.append(f.cpu().numpy())
                heads.append(h.cpu().numpy())
                logits.append(z.detach().cpu())
                if matrices:
                    operators.append(torch.stack(gs, 1).cpu().numpy())
        return (np.concatenate(fields), np.concatenate(heads), torch.cat(logits),
                np.concatenate(operators) if matrices else None,
                {name:np.concatenate(values) for name,values in extras.items()})

    def save_state(step):
        filename = 'final-training-state.pt' if step == args.steps else f'training-state-{step}.pt'
        checkpoint = dict(model={n:p.detach().cpu().clone() for n,p in model.model.state_dict().items()},
                          optimizer=copy.deepcopy(optimizer.state_dict()), scheduler=copy.deepcopy(scheduler.state_dict()),
                          recipe=profile, step=step,
                          arguments=vars(args), supervised_target_counts=torch.tensor(counts),
                          initial_parameter_sha256=initial_digest)
        temporary = out/(filename+'.tmp')
        torch.save(checkpoint, temporary)
        os.replace(temporary, out/filename)
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
                corrected = second.sqrt() / ((1-profile['betas'][1]**counter)**.5)
                ratio = (first/(1-profile['betas'][0]**counter)) / (corrected+profile['epsilon'])
                # Snapshot follows optimizer.step and scheduler.step. The applied
                # rate belonged to phase step-1; the stored next rate belongs to step.
                peak=profile['generator_peak' if generator_parameter(name) else 'body_peak']
                rate=peak*multiplier(max(0,step-1),profile['warmup_steps'],profile['total_steps'],profile['floor_fraction'])
                # Algebraic post-update reconstruction; native rounding is not inverted exactly.
                before = (parameter+rate*ratio)/(1-rate*profile['weight_decay'])
                delta = parameter-before
                values += [first.square().mean().sqrt(),second.square().mean().sqrt(),
                           ratio.square().mean().sqrt(),(corrected<profile['epsilon']).float().mean(),delta.square().mean().sqrt()]
            measured.append(torch.stack(values))
        return torch.stack(measured).cpu().numpy()

    def snapshot(step):
        f, h, _, _, extra = evaluate(np.arange(512,528))
        dense_centroids.append(extra['centroids'].mean(2))
        dense_energies.append(extra['energies'].mean(2))
        path.append(f.mean(0))
        path_steps.append(step)
        dense_fields.append(f)
        dense_heads.append(h)
        if step not in milestones:
            return
        _, _, _, operators, calibration = evaluate(np.arange(64), matrices=True)
        raw[f'mean_metric_{step}'] = calibration['mean_matrices']
        ef, eh, logits, _, extra = evaluate(np.arange(512, 1024))
        raw[f'centroids_{step}'] = extra['centroids']
        raw[f'energies_{step}'] = extra['energies']
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
        if step in (save_steps | {args.steps}) and step > start_step:
            save_state(step)

    status, error = 'complete', None
    stop_request = []
    def request_stop(number, frame):
        stop_request.append(number)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        snapshot(start_step)
        for step in range(start_step, args.steps):
            model.model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = torch.tensor(batches[step], dtype=torch.long, device=args.device)
            loss_value = torch.zeros((), device=args.device)
            for begin in range(0, 32, args.microbatch):
                piece = batch[begin:begin+args.microbatch]
                loss = recipe_loss(model,piece,profile)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss * (args.microbatch / 32)).backward()
                loss_value += loss.detach() * (args.microbatch / 32)
            del loss
            group_norms = []
            for group in optimizer.param_groups:
                group_norms.append(torch.linalg.vector_norm(torch.stack([
                    p.grad.norm() for p in group['params'] if p.grad is not None])))
            total = torch.linalg.vector_norm(torch.stack(group_norms))
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite gradient')
            recipe_clip(model,profile)
            if scheduler.last_epoch!=step:raise AssertionError('The scheduler phase differs from the native update index')
            applied=[g['lr'] for g in optimizer.param_groups]
            optimizer.step()
            scheduler.step()
            learning_rates.append(applied)
            completed_step = step + 1
            losses.append(float(loss_value))
            norms.append([float(total), *[float(x) for x in group_norms]])
            targets=(batches[step,:,1:].reshape(-1) if profile['objective']=='all_nonpadding_targets'
                     else batches[step,:,64])
            if profile['objective']=='all_nonpadding_targets':targets=targets[targets!=0]
            counts += np.bincount(targets,minlength=len(counts))
            supervised_counts.append(len(targets))
            if completed_step % args.probe_every == 0 or completed_step in milestones:
                snapshot(completed_step)
            if stop_request:
                status='administratively_censored'
                error='A signal requested a stop after a completed optimizer update'
                if completed_step not in path_steps: snapshot(completed_step)
                if str(completed_step) not in saved_states: save_state(completed_step)
                break
    except FloatingPointError as exc:
        status, error = 'numerical_failure', str(exc)

    raw.update(steps=np.array(path_steps), mean_fields=np.array(path), dense_fields=np.array(dense_fields),
               dense_heads=np.array(dense_heads), dense_centroids=np.array(dense_centroids),
               dense_energies=np.array(dense_energies), losses=np.array(losses),
               gradient_norms=np.array(norms), supervised_target_counts=counts,
               applied_learning_rates=np.asarray(learning_rates), supervised_targets_per_update=np.asarray(supervised_counts))
    if update_recorder:
        raw.update(update_recorder.arrays())
        update_recorder.close()
        if len(raw['shared_update_moments']) != completed_step-start_step:
            raise AssertionError('Update observation count differs from completed native steps')
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
    manifest = dict(schema='native-onepass-training-v1', status=status, error=error, arguments=vars(args),
                    start_step=start_step, completed_step=completed_step, initial_parameter_sha256=initial_digest,
                    recipe=profile, scheduler_phase=scheduler.last_epoch,
                    next_learning_rates=[g['lr'] for g in optimizer.param_groups],
                    schedule_indexing='Update k+1 applies s(k); optimizer.step precedes scheduler.step. Scheduled drives begin at zero and still update Adam moments; the constant-rate control has s(k)=1 from its first update.',
                    data_law=dict(corpus=str(data),manifest_sha256=sha256(data/'manifest.json'),
                        documents=524288,blocks=4194304,maximum_updates=131072,
                        sampling='One fixed permutation of nonoverlapping 64-target blocks, consumed without replacement; no supervised token position is reused.',
                        stream_seed=args.stream_seed,consumed_blocks=32*completed_step,
                        completed_input_tokens=2048*completed_step,
                        pool_baseline='Full objective-specific uniform block-population target histogram; nonpadding targets only for the all-target objective.'),
                    stop_signals=stop_request,
                    reference_adaptation=spec['adaptation'],
                    normalization_alias=normalization_alias,
                    shared_generator_sha256=shared_digest,
                    shared_update_observation=(dict(protocol=str(update_protocol), protocol_sha256=sha256(update_protocol),
                        projection_sha256=update_spec['projection_sha256'], coordinate_count=update_spec['coordinate_count'],
                        parameter_names=update_spec['parameter_names'], projection_columns=update_spec['projection_columns'],
                        steps=[start_step+1, completed_step],
                        moment_names=['mean_squared_update','mean_squared_pre_update_weight','mean_update_times_pre_update_weight'])
                        if update_spec else None),
                    parameters=sum(p.numel() for p in model.model.parameters()),
                    generator_parameters=[n for n,p in model.model.named_parameters() if generator_parameter(n)],
                    feature_names=collective_names(), head_fields=['attention_entropy/log64','head_output_rms','normalized_row_energy','operator_rms'],
                    milestones=milestones, dense_rows=[512,528], probe_every=args.probe_every,
                    saved_states=saved_states, parameter_names=[n for n,p in named_parameters],
                    parameter_statistics=['weight_rms','first_moment_rms','second_moment_rms','adaptive_ratio_rms','fraction_sqrt_v_below_epsilon','reconstructed_update_rms'],
                    calibration_rows=[0,64], evaluation_rows=[512,1024], source_rows=[512,576],
                    metric_units='E=||P A||_F^2/d^2, T=||A||_F^2/d^2; centroid coordinates use fixed native column units',
                    metric_accumulation='float64 reductions of native CUDA float32 matrices',
                    effective_batch_size=32, microbatch_size=args.microbatch,
                    pairing='512 evaluation documents permuted once and partitioned into 256 disjoint pairs; fixed finite-cohort estimates',
                    sampling_sha256=sha256(out/'sampling.npz'), raw_sha256=sha256(out/'measurements.npz'),
                    checkpoint_sha256=checkpoint_hash, binding_sha256=sha256(out/'binding.json'),
                    seconds=time.time()-started, peak_cuda_gb=torch.cuda.max_memory_allocated(args.device)/2**30 if args.device.startswith('cuda') else None)
    write_json(out/'manifest.json', manifest)
    print(args.run_id, status, round(manifest['seconds'],1), flush=True)


if __name__ == '__main__':
    main()
