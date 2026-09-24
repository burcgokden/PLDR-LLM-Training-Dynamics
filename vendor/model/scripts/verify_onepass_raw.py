#!/usr/bin/env python
"""Reconstruct single-pass sampling, drive, Adam state and observation identities.

This checker imports no scheduled producer. The reference scalar expression,
optimizer recipe, all-target counts, and schedule indexing are reconstructed
independently. Reused utilities are independent raw-data verification routines.
"""
import argparse
from datetime import datetime
import gc
import math
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256, write_json
from verify_scaling_raw import (BoundFiles,load_json,generator,close,equal,bitwise_equal,vector)


def independent_recipe(name,heads,total_steps,warmup_override):
    if name not in ['controlled','reference1','reference2','subcritical1','subcritical2']:raise AssertionError('Unknown selected recipe')
    if total_steps == 0:
        if name != 'controlled' or warmup_override != -1: raise AssertionError('Invalid constant-rate recipe')
        warmup=0;floor=1.
    elif total_steps == 768 and warmup_override == 128:
        warmup=128;floor=.1
    elif total_steps in [32768,250000] and warmup_override == -1:
        warmup=1000 if name=='reference2' else 6000 if name=='subcritical1' else 2000;floor=.1
    else: raise AssertionError('Unselected drive')
    reference=name!='controlled'
    peak={'controlled':.0003,'reference1':.0012,'reference2':.001,'subcritical1':.0006,'subcritical2':.0003}[name]
    return dict(name=name,heads=heads,total_steps=total_steps,warmup_steps=warmup,
        floor_fraction=floor,generator_peak=peak,body_peak=peak if reference else .0006/heads,
        betas=[.9,.95],epsilon=1e-5 if reference else 1e-8,weight_decay=.1 if reference else .01,
        clipping='value' if reference else 'norm',objective='all_nonpadding_targets' if reference else 'last_external_target',
        arithmetic='float32 full batch32; no TF32; per-tensor AdamW')


def reference_multiplier(step,profile):
    if profile['total_steps']==0: return 1.
    horizon=float(profile['total_steps'])
    t=float(min(step,horizon));w=float(profile['warmup_steps'])
    if t<=w:return (1/w)*t
    return .9*.5*(1+math.cos(math.pi*((t-w)/(horizon-w))))+.1


def sample_indices(seed,steps):
    if not 1 <= steps <= 131072: raise AssertionError('The nonrepeated data capacity is exceeded')
    population=np.arange(4194304,dtype=np.int64)
    np.random.default_rng(seed+1000).shuffle(population)
    selected=population[:32*steps].reshape(steps,32)
    return selected//8,64*(selected%8)


def count_targets(tokens,rows,offsets,objective,boundaries):
    histogram=np.zeros(32000,dtype=np.int64);counts={0:histogram.copy()}
    per_update=np.empty(len(rows),dtype=np.int64);begin=0
    for end in sorted(boundaries-{0}):
        while begin<end:
            stop=min(end,begin+1024)
            if objective=='last_external_target':
                target=tokens[rows[begin:stop],offsets[begin:stop]+64]
                per_update[begin:stop]=32
            else:
                target=tokens[rows[begin:stop,:,None],offsets[begin:stop,:,None]+np.arange(1,65)]
                per_update[begin:stop]=np.count_nonzero(target,axis=(1,2))
                target=target[target!=0]
            histogram+=np.bincount(target.ravel(),minlength=32000);begin=stop
        counts[end]=histogram.copy()
    return counts,per_update


def verify_training(root, study, item, files):
    protocol, spec, job = item
    folder = study/'runs'/job['run_id']
    meta = load_json(folder/'manifest.json')
    if meta['schema'] != 'native-onepass-training-v1' or meta['status'] != 'complete':
        raise AssertionError('A declared training outcome is not complete: ' + job['run_id'])
    args = meta['arguments']
    for name in ['heads', 'seed', 'recipe', 'steps', 'run_id','save_steps','microbatch','checkpoint_decoders','schedule_horizon','warmup_override','profile']:
        if (meta['recipe'] if name=='profile' else args[name]) != job[name]:
            raise AssertionError('Training condition changed: ' + name)
    for name in ['shared_seed', 'stream_seed', 'probe_every']:
        if (meta['recipe'] if name=='profile' else args[name]) != job[name]:
            raise AssertionError('Conditional environment changed: ' + name)
    if args['normalization'] != 'variance' or args['microbatch'] != 32 or args.get('checkpoint_decoders', False):
        raise AssertionError('Unqualified training arithmetic entered the main family')
    if meta['effective_batch_size'] != 32 or meta['microbatch_size'] != 32:
        raise AssertionError('Effective batch changed')
    binding = files.binding(folder, meta)
    files.check(protocol, binding['inputs'][str(protocol.resolve())])
    if datetime.fromisoformat(spec['frozen_at']) > datetime.fromisoformat(binding['environment']['utc']):
        raise AssertionError('Training began before its execution protocol was fixed')
    if args['resume'] != job.get('resume'):
        raise AssertionError('Resumed parent changed')
    profile=independent_recipe(args['recipe'],args['heads'],args['schedule_horizon'],args['warmup_override'])
    if meta['recipe']!=profile or job['profile']!=profile:
        raise AssertionError('The declared schedule recipe differs from the released reference or controlled adaptation')
    for name,signature in spec['producer_sources'].items():
        if binding['source_files'][name]!=signature:raise AssertionError('A frozen scheduled producer changed')
    origin=spec['source_selection'];files.check(origin['path'],origin['sha256'])
    original_jobs=load_json(origin['path'])['jobs']
    base=[j for j in original_jobs if j['run_id']==job['run_id']]
    if len(base)!=1 or any(job[k]!=v for k,v in base[0].items()):
        raise AssertionError('The execution protocol differs from the selected trajectory')
    start, stop = meta['start_step'], meta['completed_step']
    if not 0 <= start < stop == args['steps']:
        raise AssertionError('Training time is incomplete')
    parent = None
    if args['resume']:
        files.check(args['resume'], job['parent_sha256'])
        parent = torch.load(args['resume'], map_location='cpu', mmap=True, weights_only=True)
        if parent['step'] != start:
            raise AssertionError('Parent optimizer time changed')
        if parent['initial_parameter_sha256'] != meta['initial_parameter_sha256']:
            raise AssertionError('Continuation initialization identity changed')
        parent_meta = load_json(Path(args['resume']).parent/'manifest.json')
        if parent_meta['shared_generator_sha256'] != meta['shared_generator_sha256']:
            raise AssertionError('Shared initial state changed')
        if parent['arguments'].get('normalization', 'fan_in') != args['normalization']:
            if args['heads'] != 2 or not meta['normalization_alias']:
                raise AssertionError('Undeclared initialization alias')
    elif start != 0:
        raise AssertionError('A fresh trajectory cannot begin after zero')
    rows, offsets = sample_indices(args['stream_seed'], stop)
    tokens = np.load(root/'data/refinedweb-onepass-524288/tokens.npy', mmap_mode='r')
    if tokens.shape != (524288,513) or tokens.dtype != np.int32: raise AssertionError('Nonrepeated corpus changed')
    law=meta['data_law']
    if law['consumed_blocks']!=32*stop or law['completed_input_tokens']!=2048*stop:
        raise AssertionError('The completed unique-data exposure differs')
    if law['documents']!=524288 or law['blocks']!=4194304 or law['maximum_updates']!=131072:
        raise AssertionError('The declared data capacity changed')
    files.check(Path(law['corpus'])/'manifest.json',law['manifest_sha256'])
    if meta['stop_signals']: raise AssertionError('An administratively stopped path cannot pass as complete')
    expected_saves = {stop} | {int(t) for t in args['save_steps'].split(',') if t and start < int(t) <= stop}
    target_counts,per_update_counts=count_targets(tokens,rows,offsets,profile['objective'],expected_saves|{start})
    counts=target_counts[stop]
    if parent is not None:
        equal(parent['supervised_target_counts'],target_counts[start])
        if parent['recipe']!=profile or parent['scheduler']['last_epoch']!=start:
            raise AssertionError('The resumed recipe or phase changed')
    files.check(folder/'sampling.npz', meta['sampling_sha256'])
    with np.load(folder/'sampling.npz') as sampled:
        equal(sampled['rows'], rows)
        equal(sampled['offsets'], offsets)
        block_ids=(sampled['rows']*8+sampled['offsets']//64).ravel()
        if np.any(sampled['offsets']%64) or len(np.unique(block_ids)) != 32*stop:
            raise AssertionError('Repeated or overlapping target blocks entered training')
        equal(sampled['calibration_rows'], np.arange(64))
        equal(sampled['evaluation_rows'], np.arange(512, 1024))
        equal(sampled['dense_rows'], np.arange(512, 528))
    del rows, offsets
    states = meta['saved_states']
    expected_saves = {stop} | {int(t) for t in args['save_steps'].split(',') if t and start < int(t) <= stop}
    if set(map(int, states)) != expected_saves:
        raise AssertionError('A selected intermediate optimizer state is missing')
    final = None
    for time, record in states.items():
        t = int(time)
        path = folder/record['filename']
        files.check(path, record['sha256'])
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        if saved['step'] != t or saved['arguments'] != args:
            raise AssertionError('Saved state metadata changed')
        if saved['initial_parameter_sha256'] != meta['initial_parameter_sha256']:
            raise AssertionError('Saved initialization digest changed')
        if saved['recipe']!=profile:raise AssertionError('Saved recipe changed')
        expected_rates=[profile[k]*reference_multiplier(t,profile) for k in ['generator_peak','body_peak']]
        scheduler=saved['scheduler']
        if (scheduler['last_epoch']!=t or scheduler['_step_count']!=t+1 or
            scheduler['base_lrs']!=[profile[k] for k in ['generator_peak','body_peak']] or
            scheduler['_last_lr']!=expected_rates):
            raise AssertionError('The native scheduler phase or rates changed')
        groups = saved['optimizer']['param_groups']
        if len(groups) != 2:
            raise AssertionError('Optimizer partition changed')
        names = meta['parameter_names']
        ordered_names = [[n for n in names if generator(n)], [n for n in names if not generator(n)]]
        ids = []
        for group, group_names, lr in zip(groups, ordered_names, expected_rates, strict=True):
            if len(group['params']) != len(group_names):
                raise AssertionError('Optimizer parameter count changed')
            if tuple(group['betas']) != (.9, .95) or group['eps'] != profile['epsilon'] or group['weight_decay'] != profile['weight_decay']:
                raise AssertionError('Adam law changed')
            if group['lr']!=lr:raise AssertionError('Stored next learning rate differs from the reference phase')
            ids.extend(group['params'])
            for i, name in zip(group['params'], group_names, strict=True):
                state = saved['optimizer']['state'][i]
                if float(state['step']) != t:
                    raise AssertionError('An optimizer counter was reset or omitted')
                for key in ['exp_avg', 'exp_avg_sq']:
                    value = state[key]
                    if value.shape != saved['model'][name].shape or not torch.isfinite(value).all():
                        raise AssertionError('Invalid optimizer moment')
                if (state['exp_avg_sq'] < 0).any():
                    raise AssertionError('Negative Adam second moment')
        if len(set(ids)) != len(ids) or set(ids) != set(saved['optimizer']['state']):
            raise AssertionError('Unassigned or duplicate optimizer coordinates')
        if set(meta['generator_parameters']) != set(ordered_names[0]):
            raise AssertionError('Declared generator partition differs from optimizer')
        for name, value in saved['model'].items():
            if not torch.isfinite(value).all():
                raise AssertionError('Nonfinite trained parameter')
        if t == stop:
            equal(saved['supervised_target_counts'], counts)
            final = saved
        else:
            equal(saved['supervised_target_counts'],target_counts[t])
            del saved
    files.check(folder/'final-training-state.pt', meta['checkpoint_sha256'])
    files.check(folder/'measurements.npz', meta['raw_sha256'])
    passive = None
    batch_observation_differences = []
    with np.load(folder/'measurements.npz') as raw:
        for key in raw.files:
            if not np.isfinite(raw[key]).all():
                raise AssertionError('Nonfinite completed observation: ' + key)
        milestones = meta['milestones']
        expected_dense = sorted({start, stop, *milestones, *range(((start//args['probe_every'])+1)*args['probe_every'], stop+1, args['probe_every'])})
        equal(raw['steps'], expected_dense)
        equal(raw['supervised_target_counts'], counts)
        equal(raw['supervised_targets_per_update'],per_update_counts[start:stop])
        scalar=np.array([reference_multiplier(k,profile) for k in range(start,stop)])
        expected_rates=scalar[:,None]*np.array([profile[k] for k in ['generator_peak','body_peak']])[None,:]
        if not bitwise_equal(raw['applied_learning_rates'],expected_rates):
            raise AssertionError('Recorded applied rates differ from the exact reference ordering')
        if meta['scheduler_phase']!=stop or meta['next_learning_rates']!=[profile[k]*reference_multiplier(stop,profile) for k in ['generator_peak','body_peak']]:
            raise AssertionError('Final scheduler phase or next rate differs')
        if raw['losses'].shape != (stop-start,) or raw['gradient_norms'].shape != (stop-start, 3):
            raise AssertionError('Native update records omitted')
        close(raw['gradient_norms'][:, 0], np.linalg.norm(raw['gradient_norms'][:, 1:], axis=1), rtol=2e-7)
        n = args['heads']
        if raw['dense_heads'].shape != (len(expected_dense), 16, 5, n, 4):
            raise AssertionError('Dense observation cohort changed')
        for t in milestones:
            h, f = raw[f'heads_{t}'], raw[f'fields_{t}']
            c, e = raw[f'centroids_{t}'], raw[f'energies_{t}']
            if h.shape != (512, 5, n, 4) or f.shape != (512, 36) or c.shape != (512, 5, n, 64):
                raise AssertionError('Full evaluation cohort changed')
            if e.shape != (512, 5, n, 2) or np.any(e < 0) or np.any(e[..., 0] > e[..., 1]*(1+2e-12)):
                raise AssertionError('Invalid row energy')
            close(e[..., 0]+np.mean(c*c, axis=-1), e[..., 1], rtol=2e-12)
            # Raw R retains native float32 arithmetic; E/T is a distinct float64 reduction.
            close(h[..., 2], e[..., 0]/np.maximum(e[..., 1], 1e-30), rtol=2e-5, atol=2e-12)
            position = expected_dense.index(t)
            # GPU observation kernels may depend on batch shape (16 versus 32).
            # Keep both programs and quantify the paired discrepancy, never silently
            # replacing the frozen dense-cohort measurement with its endpoint copy.
            dense_difference = {}
            for field, left, right in [('heads', raw['dense_heads'][position], h[:16]),
                                       ('fields', raw['dense_fields'][position], f[:16])]:
                error = left.astype(float)-right.astype(float)
                dense_difference[field] = dict(maximum_absolute=float(np.max(np.abs(error))),
                    rms=float(np.sqrt(np.mean(error*error))), bitwise_equal=bitwise_equal(left,right))
            batch_observation_differences.append(dict(step=t, discrepancies=dense_difference))
        if parent is not None:
            old_path = Path(args['resume']).parent/'measurements.npz'
            with np.load(old_path) as old:
                for field in ['heads', 'fields', 'context_kl', 'operator_dispersion']:
                    equal(raw[f'{field}_{start}'], old[f'{field}_{start}'])
        rec = meta.get('shared_update_observation')
        if rec:
            sidecar = load_json(rec['protocol'])
            files.check(rec['protocol'], rec['protocol_sha256'])
            files.check(sidecar['projection'], sidecar['projection_sha256'])
            if args['run_id'] not in sidecar['run_ids'] or rec['steps'] != [start+1, stop]:
                raise AssertionError('Passive observation selection changed')
            moments = raw['shared_update_moments']
            delta = raw['shared_update_projection']
            if moments.shape != (stop-start, 3) or delta.shape != (stop-start, 16):
                raise AssertionError('Passive observations omitted an update')
            if raw['shared_clipped_gradient_projection'].shape != delta.shape:
                raise AssertionError('Clipped gradient observation omitted')
            close(moments[:-1, 1]+moments[:-1, 0]+2*moments[:-1, 2], moments[1:, 1], rtol=4e-13)
            shared_names = rec['parameter_names']
            finish = vector(final['model'], shared_names)
            if parent is None:
                # Shared initialization is exactly the unnormalized native N2 reference.
                from model_rg.training import TrainingModel
                reference = TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1', 2, args['shared_seed'], 'cpu')
                begin = vector(reference.model.state_dict(), shared_names)
                del reference
            else:
                begin = vector(parent['model'], shared_names)
            signs = np.load(sidecar['projection'], mmap_mode='r')
            endpoint = (finish-begin) @ signs / np.sqrt(len(begin))
            close(delta.sum(0), endpoint, rtol=2e-10, atol=2e-12)
            change = np.mean(finish*finish)-np.mean(begin*begin)
            recorded = np.sum(moments[:, 0]+2*moments[:, 2])
            close(recorded, change, rtol=2e-10, atol=2e-13)
            close(moments[0, 1], np.mean(begin*begin), rtol=4e-13)
            close(moments[-1, 1]+moments[-1, 0]+2*moments[-1, 2], np.mean(finish*finish), rtol=4e-13)
            if start==0 and profile['total_steps']>0:
                equal(delta[0],np.zeros_like(delta[0]))
                equal(moments[0,[0,2]],np.zeros(2))
            passive = dict(steps=stop-start, squared_norm_telescoping_error=float(abs(recorded-change)),
                           projection_telescoping_max_error=float(np.max(np.abs(delta.sum(0)-endpoint))))
    record = dict(run_id=args['run_id'], protocol=protocol.name, heads=args['heads'], seed=args['seed'],
                  recipe=args['recipe'],schedule_horizon=args['schedule_horizon'],data_blocks=32*stop,shared_seed=args['shared_seed'],stream_seed=args['stream_seed'],
                  start=start, stop=stop, updates=stop-start, saved_states=sorted(expected_saves),
                  scheduler_phase=stop,supervised_targets=int(counts.sum()),
                  passive_update_check=passive, observation_batch_comparisons=batch_observation_differences,
                  manifest_sha256=files.digest(folder/'manifest.json'))
    del final, parent
    gc.collect()
    return record


def verify_selection(root,study,kind,files):
    selection=study/'protocols'/f'onepass-{kind}-selection.json';spec=load_json(selection)
    records=[]
    for j in spec['jobs']:
        execution=study/'protocols/executions'/(j['run_id']+'.json');bound=load_json(execution)
        if bound['source_selection']!={'path':str(selection),'sha256':sha256(selection)}:
            raise AssertionError('A completed execution has a different selection')
        records.append(verify_training(root,study,(execution,bound,bound['jobs'][0]),files))
        print('Verified scheduled raw state',j['run_id'],flush=True)
    expected=(spec['selected_updates'] if kind=='training' else spec['native_updates'])
    if sum(r['updates'] for r in records)!=expected:raise AssertionError('Selected native updates were omitted')
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');p.add_argument('--kind',choices=['training','qualification'],default='training')
    p.add_argument('--output');a=p.parse_args();root=Path(a.root).resolve();study=root/a.study
    torch.set_num_threads(4);files=BoundFiles();records=verify_selection(root,study,a.kind,files)
    repo=Path(__file__).resolve().parents[1]
    output=Path(a.output) if a.output else study/'verification'/(a.kind+'-raw.json')
    if output.exists():raise FileExistsError(output)
    write_json(output,dict(schema='onepass-raw-reconstruction-v1',status='complete',kind=a.kind,
        trajectories=records,artifacts=len(records),native_updates=sum(r['updates'] for r in records),
        selection_sha256=sha256(study/'protocols'/f'onepass-{a.kind}-selection.json'),
        verified_files={key[0]:value for key,value in files.cache.items()},
        verifier_sources={str(p.relative_to(repo)):sha256(p) for p in [Path(__file__),repo/'scripts/verify_scaling_raw.py']},
        scope='Independent reconstruction of the complete nonrepeated block stream, bound sampling, optimizer/scheduler state, recorded applied rates, target counts, row-energy identities and passive-update telescoping. This does not replay every native training update.'))


if __name__=='__main__':main()
