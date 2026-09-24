#!/usr/bin/env python
"""Independently verify native trajectory lineage and frozen observation identities.

No training or statistical producer helpers are imported. Sampling, optimizer
partitions, energy identities, and passive-update telescoping are reconstructed
from the declared protocol and raw arrays. A partial snapshot is never a release
gate. All protocol-selected outcomes, including failures, remain in the inventory.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256, write_json


def load_json(path):
    return json.loads(Path(path).read_text())


def generator(name):
    return any(part in name for part in ('reslayerAs', 'plgatt_layer', 'layernormA'))


def close(a, b, *, rtol=2e-11, atol=1e-25):
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def equal(a, b):
    np.testing.assert_array_equal(a, b)


def bitwise_equal(a, b):
    """Compare shape, dtype and logical C-order bytes, including signed zero."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes(order='C') == b.tobytes(order='C')


class BoundFiles:
    def __init__(self):
        self.cache = {}

    def digest(self, path):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self.cache:
            self.cache[key] = sha256(path)
        return self.cache[key]

    def check(self, path, signature):
        if self.digest(path) != signature:
            raise AssertionError('Changed bound file: ' + str(path))

    def binding(self, folder, meta):
        self.check(folder/'binding.json', meta['binding_sha256'])
        binding = load_json(folder/'binding.json')
        for path, signature in binding['inputs'].items():
            self.check(path, signature)
        for name, signature in binding['source_files'].items():
            self.check(folder/'source'/name, signature)
        return binding


def selection(study, additional, partial):
    jobs = []
    protocols = ['main-size-time-execution.json', 'clock-holdout.json',
                 'diffusive-size-map.json', 'environment-factorial.json',
                 'zero-horizon.json', 'extended-horizon.json', *additional]
    pending = []
    for name in protocols:
        path = study/'protocols'/name
        if not path.exists():
            if not partial:
                raise FileNotFoundError(path)
            pending.append(name)
            continue
        spec = load_json(path)
        jobs.extend((path, spec, job) for job in spec['jobs'])
    original = study/'protocols/long-horizon.json'
    spec = load_json(original)
    preserved = [j for j in spec['jobs'] if j['run_id'] == 'long-h4-g1-s640101']
    if len(preserved) != 1:
        raise AssertionError('Preserved native child is not uniquely selected')
    jobs.append((original, spec, preserved[0]))
    if len({j['run_id'] for _, _, j in jobs}) != len(jobs):
        raise AssertionError('Training identities counted twice')
    if not pending and not additional and len(jobs) != 96:
        raise AssertionError('The complete selected design has 96 training artifacts')
    return jobs, pending


def sample_indices(seed, steps):
    rows = np.random.default_rng(seed + 1000).integers(0, 3072, size=(steps, 32))
    offsets = np.random.default_rng(seed + 1001000).integers(0, 449, size=(steps, 32))
    return rows, offsets


def vector(model, names):
    return np.concatenate([model[name].numpy().reshape(-1).astype(np.float64) for name in names])


def verify_training(root, study, item, files):
    protocol, spec, job = item
    folder = study/'runs'/job['run_id']
    meta = load_json(folder/'manifest.json')
    if meta['schema'] != 'native-size-time-training-v1' or meta['status'] != 'complete':
        raise AssertionError('A declared training outcome is not complete: ' + job['run_id'])
    args = meta['arguments']
    for name in ['heads', 'seed', 'multiplier', 'steps', 'run_id']:
        if args[name] != job[name]:
            raise AssertionError('Training condition changed: ' + name)
    for name in ['shared_seed', 'stream_seed', 'probe_every']:
        if args[name] != job.get(name, spec[name]):
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
    tokens = np.load(root/'data/refinedweb-4608/tokens.npy', mmap_mode='r')
    targets = tokens[rows, offsets+64]
    counts = np.bincount(targets.ravel(), minlength=32000)
    if parent is not None:
        equal(parent['supervised_target_counts'], np.bincount(targets[:start].ravel(), minlength=32000))
    files.check(folder/'sampling.npz', meta['sampling_sha256'])
    with np.load(folder/'sampling.npz') as sampled:
        equal(sampled['rows'], rows)
        equal(sampled['offsets'], offsets)
        equal(sampled['calibration_rows'], np.arange(64))
        equal(sampled['evaluation_rows'], np.arange(512, 1024))
        equal(sampled['dense_rows'], np.arange(512, 528))
    del rows, offsets, targets
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
        groups = saved['optimizer']['param_groups']
        if len(groups) != 2:
            raise AssertionError('Optimizer partition changed')
        names = meta['parameter_names']
        ordered_names = [[n for n in names if generator(n)], [n for n in names if not generator(n)]]
        ids = []
        for group, group_names, lr in zip(groups, ordered_names, [3e-4*args['multiplier'], 3e-4*2/args['heads']], strict=True):
            if len(group['params']) != len(group_names):
                raise AssertionError('Optimizer parameter count changed')
            if tuple(group['betas']) != (.9, .95) or group['eps'] != 1e-8 or group['weight_decay'] != .01:
                raise AssertionError('Adam law changed')
            close(group['lr'], lr)
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
            if args['multiplier'] == 0 and generator(name):
                if parent is None or not bitwise_equal(value.numpy(), parent['model'][name].numpy()):
                    raise AssertionError('A zero-rate generator weight changed')
        if t == stop:
            equal(saved['supervised_target_counts'], counts)
            final = saved
        else:
            rr, oo = sample_indices(args['stream_seed'], t)
            equal(saved['supervised_target_counts'], np.bincount(tokens[rr, oo+64].ravel(), minlength=32000))
            del rr, oo, saved
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
            if args['multiplier'] == 0:
                equal(delta, np.zeros_like(delta))
                equal(moments[:, [0, 2]], np.zeros_like(moments[:, [0, 2]]))
            passive = dict(steps=stop-start, squared_norm_telescoping_error=float(abs(recorded-change)),
                           projection_telescoping_max_error=float(np.max(np.abs(delta.sum(0)-endpoint))))
    record = dict(run_id=args['run_id'], protocol=protocol.name, heads=args['heads'], seed=args['seed'],
                  multiplier=args['multiplier'], shared_seed=args['shared_seed'], stream_seed=args['stream_seed'],
                  start=start, stop=stop, updates=stop-start, saved_states=sorted(expected_saves),
                  zero_generator_bitwise_unchanged=args['multiplier'] == 0,
                  passive_update_check=passive, observation_batch_comparisons=batch_observation_differences,
                  manifest_sha256=files.digest(folder/'manifest.json'))
    del final, parent
    gc.collect()
    return record


def verify_measurement(root, path, files):
    meta = load_json(path)
    folder = path.parent
    files.binding(folder, meta)
    files.check(folder/'measurements.npz', meta['raw_sha256'])
    condition, case = meta['condition'], meta['case']
    saved = torch.load(case['state'], map_location='cpu', mmap=True, weights_only=True)
    if saved['arguments'] != condition or saved['step'] != case['step']:
        raise AssertionError('Frozen measurement condition differs from saved state')
    if meta['device'] != 'cpu' or meta['batch_size'] != 32:
        raise AssertionError('The common observation platform changed')
    del saved
    n = condition['heads']
    probe = root/'controlled-study-20260905/data/short'
    tokens, offsets = np.load(probe/'tokens.npy', mmap_mode='r'), np.load(probe/'offsets.npy')
    with np.load(folder/'measurements.npz') as raw:
        rows = np.arange(512, 1024)
        equal(raw['rows'], rows)
        equal(raw['crops'], tokens[rows[:, None], offsets[rows, None]+np.arange(65)])
        equal(raw['calibration_rows'], np.arange(64))
        equal(raw['calibration_crops'], tokens[np.arange(64)[:, None], offsets[:64, None]+np.arange(65)])
        equal(raw['context_pairs'], np.random.default_rng(640071).permutation(512).reshape(-1, 2))
        for key in raw.files:
            if not np.isfinite(raw[key]).all():
                raise AssertionError('Nonfinite frozen measurement')
        for r in meta['precision_records']:
            prefix = r['precision']+'_'
            c, e, h = (raw[prefix+k] for k in ['centroids', 'energies', 'heads'])
            if c.shape != (512, 5, n, 64) or e.shape != (512, 5, n, 2) or h.shape != (512, 5, n, 4):
                raise AssertionError('Frozen field dimensions changed')
            if np.any(e < 0) or np.any(e[..., 0] > e[..., 1]*(1+2e-12)):
                raise AssertionError('Invalid frozen energy')
            close(e[..., 0]+np.mean(c*c, axis=-1), e[..., 1], rtol=2e-12)
            close(h[..., 2], e[..., 0]/np.maximum(e[..., 1], 1e-30), rtol=2e-5, atol=2e-12)
            close(raw[prefix+'context_kl'].mean(), r['mean_context_kl'])
            matrix = raw[prefix+'mean_metric']
            if matrix.shape != (64, 5, 64, 64):
                raise AssertionError('Calibration cohort changed')
            if meta.get('evaluation_matrix_contexts'):
                matrix = raw[prefix+'evaluation_mean_metric']
                if matrix.shape != (512, 5, 64, 64):
                    raise AssertionError('Evaluation mean-matrix cohort changed')
                close(matrix.mean(-2), c.mean(2), rtol=2e-11, atol=3e-15)
        if len(meta['precision_records']) == 2:
            if raw['paired_predictive_kl'].shape != (512,) or raw['paired_predictive_kl'].min() < -1e-12:
                raise AssertionError('Invalid paired prediction response')
    return dict(name=folder.name, case=case, precisions=[r['precision'] for r in meta['precision_records']],
                manifest_sha256=files.digest(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906')
    p.add_argument('--output', required=True)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--additional-protocols', nargs='*', default=[])
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4)
    root, repo = Path(a.root), Path(__file__).resolve().parents[1]
    study = root/a.study
    sources = {name:sha256(repo/name) for name in ['scripts/verify_scaling_raw.py',
               'src/model_rg/provenance.py', 'src/model_rg/training.py', 'src/model_rg/native.py']}
    files = BoundFiles()
    jobs, pending = selection(study, a.additional_protocols, a.allow_partial)
    training, missing = [], []
    for item in jobs:
        path = study/'runs'/item[2]['run_id']/'manifest.json'
        if not path.exists():
            missing.append(item[2]['run_id'])
            if a.allow_partial:
                continue
            raise FileNotFoundError(path)
        training.append(verify_training(root, study, item, files))
        print('Verified native trajectory', item[2]['run_id'], flush=True)
    measurements = []
    for path in sorted((study/'measurements').glob('cpu-*/manifest.json')):
        measurements.append(verify_measurement(root, path, files))
        print('Verified frozen observation', path.parent.name, flush=True)
    if not a.allow_partial:
        observation_protocols = [study/'protocols/baseline-collectives.json', study/'protocols/late-collectives.json']
        selected = load_json(study/'protocols/target-collective-selection.json')
        observation_protocols += [study/'protocols'/('observations-'+j['run_id']+'.json') for j in selected['jobs']]
        expected = {c['name'] for protocol in observation_protocols for c in load_json(protocol)['cases']}
        if expected != {r['name'] for r in measurements}:
            raise AssertionError('The complete CPU observation selection is not exactly represented')
        if not a.additional_protocols and len(expected) != 224:
            raise AssertionError('The paired CPU study requires 224 selected states')
    for name, signature in sources.items():
        files.check(repo/name, signature)
    write_json(a.output, dict(schema='native-scaling-raw-verification-v1',
        status='partial_snapshot_passed' if a.allow_partial else 'passed', recorded_at=datetime.now(timezone.utc).isoformat(),
        arguments=vars(a), training=training, observations=measurements,
        training_artifacts=len(training), additional_updates=sum(r['updates'] for r in training),
        frozen_observation_states=len(measurements), pending_protocols=pending, pending_training=missing,
        verifier_sources=sources, verifier_sha256=sha256(__file__), hashed_file_versions=len(files.cache),
        scope='Independent raw lineage, sample law, optimizer time, finite-state, energy and passive-update identities. This is not an independent retraining or a statistical theory confirmation.'))
    print('Raw reconstruction passed', len(training), 'trajectories and', len(measurements), 'frozen states', flush=True)


if __name__ == '__main__':
    main()
