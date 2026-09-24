#!/usr/bin/env python
"""Seal the retained repeated-data states and freeze their matched observations.

Only the immutable copies are complete. The interrupted source trajectories
remain administratively censored and contribute no completed training identity.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from verify_scaling_raw import generator, sample_indices
from verify_scheduled_raw import independent_recipe, reference_multiplier, count_targets


def validate_state(path, case, binding, target_counts):
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    a = saved['arguments']; t = case['step']
    if a != binding['arguments'] or saved['step'] != t or a['steps'] != 262144:
        raise AssertionError('Retained checkpoint identity or source horizon changed')
    for key in ['run_id', 'recipe', 'heads', 'seed', 'shared_seed', 'stream_seed']:
        if a[key] != case[key]: raise AssertionError('Retained checkpoint condition changed: '+key)
    if a['normalization'] != 'variance' or a['microbatch'] != 32 or a['checkpoint_decoders'] or a['resume']:
        raise AssertionError('An unselected native arithmetic or continuation was used')
    profile = independent_recipe(case['recipe'], 14)
    if saved['recipe'] != profile: raise AssertionError('The reference recipe changed')
    rates = [profile[key]*reference_multiplier(t, profile) for key in ['generator_peak', 'body_peak']]
    scheduler = saved['scheduler']
    if (scheduler['last_epoch'] != t or scheduler['_step_count'] != t+1 or
        scheduler['base_lrs'] != [profile[key] for key in ['generator_peak','body_peak']] or
        scheduler['_last_lr'] != rates): raise AssertionError('The saved drive phase changed')
    ordered = [[n for n in saved['model'] if generator(n)], [n for n in saved['model'] if not generator(n)]]
    groups = saved['optimizer']['param_groups']; identities = []
    if len(groups) != 2: raise AssertionError('The optimizer partition changed')
    for group, names, rate in zip(groups, ordered, rates, strict=True):
        if (len(group['params']) != len(names) or tuple(group['betas']) != (.9,.95) or
            group['eps'] != profile['epsilon'] or group['weight_decay'] != profile['weight_decay'] or
            group['lr'] != rate): raise AssertionError('The saved Adam law changed')
        identities.extend(group['params'])
        for index, name in zip(group['params'], names, strict=True):
            state = saved['optimizer']['state'][index]
            if float(state['step']) != t: raise AssertionError('An Adam counter was reset')
            for key in ['exp_avg','exp_avg_sq']:
                if state[key].shape != saved['model'][name].shape or not torch.isfinite(state[key]).all():
                    raise AssertionError('Invalid retained Adam moment')
            if (state['exp_avg_sq'] < 0).any(): raise AssertionError('Negative second moment')
    if len(set(identities)) != len(identities) or set(identities) != set(saved['optimizer']['state']):
        raise AssertionError('Unassigned optimizer coordinates')
    if any(not torch.isfinite(value).all() for value in saved['model'].values()):
        raise AssertionError('Nonfinite retained model coordinate')
    np.testing.assert_array_equal(saved['supervised_target_counts'], target_counts)
    return dict(arguments=a, initial_parameter_sha256=saved['initial_parameter_sha256'],
                parameters=len(identities), optimizer_counters=t, next_learning_rates=rates,
                supervised_targets=int(np.sum(target_counts)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908'); a = p.parse_args()
    root = Path(a.root).resolve(); study = root/a.study
    old = root/'scheduled-training-20260908'; repo = Path(__file__).resolve().parents[1]
    selection = study/'protocols/repeated-risk-comparison.json'
    if selection.exists(): raise FileExistsError(selection)
    torch.set_num_threads(4)
    policy = old/'development/feasibility-redesign/no-repeat-data-policy.json'
    if json.loads(policy.read_text())['schema'] != 'data-policy-interruption-v1':
        raise AssertionError('The interrupted data-law record is missing')
    observations_path = old/'protocols/regime-observation-selection.json'
    observations = json.loads(observations_path.read_text())
    training_path = old/'protocols/regime-training-selection.json'
    training = json.loads(training_path.read_text())
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    inputs = [policy, observations_path, training_path,
        study/'protocols/onepass-training-selection.json',
        Path(observations['training_cohort']), Path(observations['training_cohort_manifest']),
        root/'data/refinedweb-4608/tokens.npy', root/'data/refinedweb-4608/manifest.json',
        root/'controlled-study-20260905/data/short/tokens.npy',
        root/'controlled-study-20260905/data/short/offsets.npy',
        source/'modeling_pldrllm.py', source/'configuration_pldrllm.py', source/'config.json']
    checked = {}
    def check(path, digest):
        path = str(Path(path).resolve())
        if path not in checked: checked[path] = sha256(path)
        if checked[path] != digest: raise AssertionError('Changed source binding: '+path)
    rows, offsets = sample_indices(640001, 65536)
    tokens = np.load(root/'data/refinedweb-4608/tokens.npy', mmap_mode='r')
    counts, _ = count_targets(tokens, rows, offsets, 'all_nonpadding_targets', {0,32768,65536})
    cases = []; seals = []
    for recipe in ['reference1','subcritical1']:
        run_id = f'{recipe}-h14-s640101'; folder = old/'runs'/run_id
        binding_path = folder/'binding.json'; binding = json.loads(binding_path.read_text())
        for path, digest in binding['inputs'].items(): check(path, digest)
        for name, digest in training['producer_sources'].items():
            if binding['source_files'][name] != digest: raise AssertionError('The bound native producer changed')
            check(folder/'source'/name, digest)
        for t in [32768,65536]:
            name = f'repeated-{run_id}-t{t}'; original = folder/f'training-state-{t}.pt'
            case = dict(name=name, run_id=run_id, recipe=recipe, heads=14, seed=640101,
                        shared_seed=640011, stream_seed=640001, step=t)
            signature = sha256(original)
            validation = validate_state(original, case, binding, counts[t])
            if t == 32768:
                early = old/'early-checkpoints'/f'early-prefix-{run_id}-t{t}'/'checkpoint/manifest.json'
                if json.loads(early.read_text())['source_checkpoint_sha256'] != signature:
                    raise AssertionError('The retained state differs from its earlier sealed copy')
                inputs.append(early)
            parent = study/'repeated-checkpoints'/name/'checkpoint'
            parent.mkdir(parents=True, exist_ok=False)
            state = parent/'state.pt'; temporary = parent/'state.pt.copying'
            shutil.copyfile(original, temporary)
            if sha256(temporary) != signature or sha256(original) != signature:
                raise AssertionError('Retained checkpoint changed while sealing')
            temporary.rename(state)
            bind_run(parent, [original,binding_path,policy,training_path], vars(a))
            meta_path = parent/'manifest.json'
            write_json(meta_path, dict(schema='sealed-native-checkpoint-observation-parent-v1',status='complete',
                role='Immutable observation copy of a saved state from an administratively censored trajectory.',
                captured_at=datetime.now(timezone.utc).isoformat(), completed_step=t,
                saved_states={str(t):dict(filename='state.pt',sha256=signature)},
                binding_sha256=sha256(parent/'binding.json'),source_checkpoint=str(original),
                source_checkpoint_sha256=signature,source_binding=str(binding_path),
                source_binding_sha256=sha256(binding_path),source_trajectory_manifest=str(folder/'manifest.json'),
                source_trajectory_status_at_capture='administratively_censored',final_selected_horizon=262144,
                additional_independent_training_identities=0,**validation))
            cases.append(dict(**case,state=str(state),state_sha256=signature,sealed_manifest=str(meta_path)))
            inputs.extend([meta_path,parent/'binding.json',binding_path])
            seals.append(dict(name=name,manifest_sha256=sha256(meta_path),validation=validation))
            print('Sealed and validated', name, flush=True)
    names = ['scripts/prepare_repeated_comparison.py','scripts/measure_repeated_comparison_risk.py',
        'scripts/verify_repeated_comparison_risk.py','scripts/run_repeated_comparison_risk.py',
        'scripts/verify_scheduled_raw.py','scripts/verify_scaling_raw.py',
        'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/controlled.py','src/model_rg/provenance.py']
    write_json(selection,dict(schema='retained-repeated-risk-selection-v1',frozen_at=datetime.now(timezone.utc).isoformat(),
        cases=cases,training_cohort=observations['training_cohort'],training_contexts=1024,heldout_contexts=512,
        training_indices=list(range(1024)),heldout_rows=list(range(512,1024)),batch_size=32,threads=4,
        source=str(source),inputs_sha256={str(path):sha256(path) for path in inputs},
        producer_sources={name:sha256(repo/name) for name in names},
        prior_information='All six earlier 128-context training/held-out risks and native diagnostic logs through65536 '
        'are known. The two matched single-pass trajectories have begun; no selected endpoint comparison is available. '
        'This complete four-state panel measures the retained32768 and65536 states with the same512 held-out contexts '
        'and the whole1024-context frozen old training-law cohort.',
        interpretation='The two original repeated-data trajectories are administratively censored. Their complete saved '
        'checkpoint copies are eligible for fixed-state observation only. Full-position native float32 logits, CE and '
        'entropy are retained for every target; last_nll uses the full vocabulary projection. Single-pass risk uses a '
        'separate last-only projection for its last-target field, so all-target CE is the primary schedule-risk comparison. '
        'No native trajectory is completed or missing dense online history reconstructed by these measurements.'))
    write_json(study/'verification/repeated-checkpoint-sealing.json',dict(status='passed',states=4,records=seals,
        selection_sha256=sha256(selection),checked_original_inputs_sha256=checked,
        verifier_sha256=sha256(__file__),additional_independent_training_identities=0))
    print('Frozen all four retained-state observations',sha256(selection),flush=True)


if __name__ == '__main__': main()
