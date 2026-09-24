#!/usr/bin/env python
"""Reconstruct the single-pass prediction summaries without importing their producer."""
import argparse
from collections import defaultdict
import itertools
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import math
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


KEYS = ('schedule_horizon', 'recipe', 'heads', 'step', 'shared_seed', 'stream_seed')


def mean(values):
    values = list(values)
    return math.fsum(values)/len(values)


def summary(values):
    s = len(values)
    if s not in (1, 2, 4): raise AssertionError('Unexpected selected initialization count')
    report = dict(seed_values=values, selected_seeds=s, defined_seeds=sum(v is not None for v in values))
    if report['defined_seeds'] != s:
        return dict(**report, mean=None, empirical_percentiles=None, leave_one_out=None)
    if s == 1:
        return dict(**report, mean=values[0], empirical_percentiles=None, leave_one_out=None)
    boot = [mean(values[i] for i in draw) for draw in itertools.product(range(s), repeat=s)]
    return dict(**report, mean=mean(values), empirical_percentiles=np.quantile(boot, [.025, .975]).tolist(),
        leave_one_out=[mean(values[j] for j in range(s) if j != i) for i in range(s)])


def equal(actual, expected, name='root'):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError('Changed fields at ' + name)
        for key in expected:
            equal(actual[key], expected[key], name + '.' + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError('Changed coverage at ' + name)
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            equal(a, e, name + '[' + str(i) + ']')
    elif isinstance(expected, (str, bool, int)) or expected is None:
        if actual != expected:
            raise AssertionError('Changed identity, count or undefined value at ' + name)
    else:
        if actual is None or not math.isfinite(actual) or not math.isfinite(expected):
            raise AssertionError('Nonfinite statistic at ' + name)
        if not math.isclose(actual, expected, rel_tol=3e-11, abs_tol=2e-13):
            raise AssertionError(f'Statistic differs at {name}: {actual!r} != {expected!r}')


def tasks(modes, definitions):
    native = {e['id']: e for e in modes['native']['examples']}
    if len(native) != 320 or set(native) != set(definitions):
        raise AssertionError('The full fixed 320-task cohort is required')
    output = []
    for mode, data in modes.items():
        examples = {e['id']: e for e in data['examples']}
        if len(data['examples']) != 320 or set(examples) != set(native):
            raise AssertionError('An intervention omits a selected task')
        accuracy = defaultdict(list)
        normalized = defaultdict(list)
        pairs = defaultdict(dict)
        errors = []
        flips = certified = correct = 0
        for identifier, item in examples.items():
            reference = native[identifier]
            definition = definitions[identifier]
            if item['correct'] != definition['correct'] or item['correct'] != reference['correct']:
                raise AssertionError('The fixed task answer changed')
            scores = item['scores']
            baseline = reference['scores']
            error = max(abs(scores[i] - baseline[i]) for i in range(len(baseline)))
            if len(scores) != len(baseline):
                raise AssertionError('Candidate coverage changed')
            margin = baseline[item['correct']] - max(s for i, s in enumerate(baseline) if i != item['correct'])
            equal(reference['margin'], margin, identifier + '.native_margin')
            prediction = max(range(len(scores)), key=scores.__getitem__)
            normalized_prediction = max(range(len(scores)), key=item['token_mean_scores'].__getitem__)
            if (prediction, normalized_prediction) != (item['prediction'], item['token_mean_prediction']):
                raise AssertionError('A candidate decision changed')
            accuracy[item['task']].append(prediction == item['correct'])
            normalized[item['task']].append(normalized_prediction == item['correct'])
            flips += prediction != reference['prediction']
            correct += margin > 0
            certified += finite_greater(margin, 2 * error, 'scripts/verify_onepass_predictions.py:92')
            if finite_greater(margin, 2 * error, 'scripts/verify_onepass_predictions.py:92') and prediction != item['correct']:
                raise AssertionError('A strictly certified margin was lost')
            errors.append(error)
            if 'pair' in definition:
                pairs[definition['pair']][definition['variant']] = (item, definition)
        by_depth = defaultdict(list)
        both = defaultdict(list)
        if len(pairs) != 96:
            raise AssertionError('All 96 counterfactual pairs are required')
        for pair in pairs.values():
            if set(pair) != {0, 1}:
                raise AssertionError('An incomplete counterfactual pair was used')
            left, left_definition = pair[0]
            right, right_definition = pair[1]
            depth = left_definition['depth']
            if (left['correct'], right['correct']) != (0, 1) or right_definition['depth'] != depth:
                raise AssertionError('Counterfactual orientation or depth changed')
            by_depth[depth].append((right['scores'][1] - right['scores'][0]) -
                                  (left['scores'][1] - left['scores'][0]))
            both[depth].append(left['prediction'] == 0 and right['prediction'] == 1)
        output.append(dict(mode=mode, accuracy={k: mean(v) for k, v in sorted(accuracy.items())},
            token_mean_accuracy={k: mean(v) for k, v in sorted(normalized.items())},
            external_target_nll=data['external_target_nll'],
            paired_external_nll_change=data['external_target_nll'] - modes['native']['external_target_nll'],
            answer_decision_changes=flips, strictly_correct_native=correct, certified_preserved_correct=certified,
            maximum_candidate_score_error=max(errors), mean_maximum_candidate_score_error=mean(errors),
            premise_responses={str(depth): dict(pairs=len(values), mean_signed_response=mean(values),
                positive_response_fraction=mean(v > 0 for v in values), both_correct_fraction=mean(both[depth]))
                for depth, values in sorted(by_depth.items())}))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908')
    args = p.parse_args()
    study = Path(args.root).resolve() / args.study
    folder = study / 'analysis/onepass-predictions'
    output = study / 'verification/onepass-prediction-statistics.json'
    if output.exists():
        raise FileExistsError(output)
    checked = {}
    digest_cache = {}
    verified = {}

    def check(path, expected=None):
        path = Path(path).resolve()
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in digest_cache:
            digest_cache[key] = sha256(path)
        digest = digest_cache[key]
        if expected is not None and digest != expected:
            raise AssertionError('Changed scheduled prediction evidence: ' + str(path))
        checked[str(path)] = digest
        return digest

    def load(path):
        check(path)
        return load_json_strict(Path(path).read_text())

    def proof(path, case):
        record = load(path)
        if record['status'] != 'passed' or record['case'] != case:
            raise AssertionError('The selected underlying observation was not verified')
        for name, digest in record['checked_sha256'].items():
            if name in verified and digest != verified[name]:
                raise AssertionError('Conflicting underlying verification bindings')
            verified[name] = digest

    def result(parent):
        parent = Path(parent)
        meta = load(parent / 'manifest.json')
        if meta['status'] != 'complete':
            raise AssertionError('An incomplete state entered the analysis')
        for name in ['manifest.json', 'results.json']:
            path = parent / name
            check(path, verified[str(path)])
        check(parent / 'results.json', meta['results_sha256'])
        return load(parent / 'results.json')

    meta = load(folder / 'manifest.json')
    if meta['status'] != 'complete':
        raise AssertionError('The full scheduled prediction analysis is required')
    for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256')]:
        check(folder / name, meta[key])
    binding = load(folder / 'binding.json')
    for name, digest in binding['inputs'].items():
        check(name, digest)
    for name, digest in binding['source_files'].items():
        check(folder / 'source' / name, digest)
    selection = load(study / 'protocols/onepass-observation-selection.json')
    prefix_selection = load(study / 'protocols/scheduled-prefix-selection.json')
    task_selection = load(study / 'protocols/reasoning-mechanism-selection.json')
    definition_path = Path(task_selection['cohort'])
    definitions = {e['id']: e for e in load(definition_path)['examples']}
    identity = lambda c: (c['run_id'], c['step'])
    prefixes = {identity(c): c for c in prefix_selection['cases']}
    task_cases = {identity(c): c for c in task_selection['cases']}
    if (len(selection['cases']), len(prefixes), len(task_cases)) != (284, 284, 94):
        raise AssertionError('The full selected observation panel is required')
    expected = []
    task_count = 0
    for case in selection['cases']:
        prefix_case = prefixes[identity(case)]
        proof(study / 'verification/observations' / (case['name'] + '.json'), case)
        proof(study / 'verification/prefix' / (prefix_case['name'] + '.json'), prefix_case)
        risk = result(study / 'measurements' / ('risk-' + case['name'].removeprefix('cpu-')))
        prefix = result(study / 'measurements' / prefix_case['name'])
        scalars = {}
        for population in ['training', 'heldout']:
            for field in ['last_nll', 'last_entropy', 'all_token_weighted_nll', 'all_token_weighted_entropy']:
                scalars[population + '_' + field] = risk['cohorts'][population][field]
        for field in ['last_nll', 'all_token_weighted_nll']:
            scalars['generalization_gap_' + field] = risk['cohorts']['heldout'][field] - risk['cohorts']['training'][field]
        scalars['preceding_online_loss'] = risk['online_window']['preupdate_mean_loss']
        proper = [r for r in prefix['conditions'] if r['variant'] == 'prefix' and r['length'] < 64]
        if [r['length'] for r in proper] != [16, 32, 48]:
            raise AssertionError('Proper-prefix coverage changed')
        scalars['parallel_nll_prefix_panel'] = mean(r['full_window_nll'] for r in proper)
        scalars['proper_nll_prefix_panel'] = mean(r['nll'] for r in proper)
        for variant in ['prefix', 'target_swap', 'suffix_swap']:
            selected = [r for r in prefix['conditions'] if r['variant'] == variant and r['length'] < 64]
            if [r['length'] for r in selected] != [16, 32, 48]:
                raise AssertionError('A selected suffix intervention was omitted')
            for field in ['nll_change', 'mean_predictive_kl', 'mean_relative_G_difference', 'mean_row_ratio']:
                scalars[variant + '_' + field] = mean(r[field] for r in selected)
            scalars[variant + '_maximum_logit_difference'] = max(r['maximum_logit_difference'] for r in selected)
        record = dict(case=case, scalars=scalars, prefix_conditions=prefix['conditions'], risk=risk['cohorts'])
        if case['inference_stability']:
            generation = result(study / 'measurements' / ('inference-' + case['name'].removeprefix('cpu-')))
            for tensor, comparisons in generation['comparisons'].items():
                for comparison, values in comparisons.items():
                    for field in ['rmse', 'mean_normalized_rmse', 'rms_normalized_rmse', 'mean_magnitude_to_rms']:
                        scalars[f'generation_{tensor}_{comparison}_{field}'] = values[field]
            if len(generation['generated_tokens']) != 2 or len(generation['eos_stops']) != 2:
                raise AssertionError('The full 100-prompt paired-generation panel is required')
            scalars['mean_generated_length'] = sum(generation['generated_tokens'])/200
            scalars['eos_stop_fraction'] = sum(generation['eos_stops'])/200
            task_case = task_cases[identity(case)]
            proof(study / 'verification/reasoning' / (task_case['name'] + '.json'), task_case)
            task = result(study / 'measurements' / task_case['name'])
            modes = {m['mode']: m for m in task['modes']}
            if set(modes) != {'native', 'calibration_G', 'permuted_G', 'row_projection'}:
                raise AssertionError('A selected operator intervention was omitted')
            record['task_interventions'] = tasks(modes, definitions)
            task_count += 1
            for mode in record['task_interventions']:
                for metric in ['accuracy', 'token_mean_accuracy']:
                    for task_name, value in mode[metric].items():
                        scalars[f'task_{mode["mode"]}_{metric}_{task_name}'] = value
                for field in ['external_target_nll', 'paired_external_nll_change', 'answer_decision_changes',
                              'strictly_correct_native', 'certified_preserved_correct',
                              'maximum_candidate_score_error', 'mean_maximum_candidate_score_error']:
                    scalars[f'task_{mode["mode"]}_{field}'] = mode[field]
                for depth, values in mode['premise_responses'].items():
                    for field in ['mean_signed_response', 'positive_response_fraction', 'both_correct_fraction']:
                        scalars[f'task_{mode["mode"]}_depth{depth}_{field}'] = values[field]
        expected.append(record)
    check(definition_path, verified[str(definition_path)])
    reported = load(folder / 'results.json')
    equal(reported['states'], expected, 'states')
    if task_count != 94:
        raise AssertionError('Endpoint task coverage changed')
    spec = load(study/'protocols/onepass-analysis-selection.json')
    groups = defaultdict(dict)
    for record in expected:
        c = record['case']; key = tuple(c[k] for k in KEYS)
        if c['seed'] in groups[key]: raise AssertionError('An initialization was counted twice')
        groups[key][c['seed']] = record['scalars']
    selected_groups = {tuple(r[k] for k in KEYS): r['seeds'] for r in spec['groups']}
    if {k: sorted(v) for k, v in groups.items()} != selected_groups:
        raise AssertionError('Selected group coverage changed')
    conditions = []
    for key, seeds in sorted(groups.items()):
        fields = sorted({name for row in seeds.values() for name in row})
        observations = {}
        for name in fields:
            included = sorted(s for s, row in seeds.items() if name in row)
            excluded = sorted(set(seeds)-set(included))
            observations[name] = dict(**summary([seeds[s][name] for s in included]),
                seed_ids=included, unselected_seed_ids=excluded)
            if excluded and not (name.startswith('generation_') or name.startswith('task_') or name in ['mean_generated_length', 'eos_stop_fraction']):
                raise AssertionError('A mandatory non-endpoint field was omitted')
        conditions.append(dict(**dict(zip(KEYS, key, strict=True)), seeds=sorted(seeds), observations=observations))
    equal(reported['conditions'], conditions, 'conditions')
    contrasts = []
    for definition in spec['recipe_contrasts']:
        left = groups[tuple(definition['first_key'])]; right = groups[tuple(definition['second_key'])]
        identities = sorted(left)
        if identities != sorted(right) or identities != definition['seeds']:
            raise AssertionError('A paired recipe omits an initialization')
        left_fields = {name for row in left.values() for name in row}
        right_fields = {name for row in right.values() for name in row}
        observations = {}
        for field in sorted(left_fields & right_fields):
            selected = [s for s in identities if field in left[s] and field in right[s]]
            values = [None if left[s][field] is None or right[s][field] is None else left[s][field]-right[s][field]
                      for s in selected]
            observations[field] = dict(**summary(values), seed_ids=selected,
                unselected_seed_ids=sorted(set(identities)-set(selected)))
        contrasts.append(dict(**definition, observations=observations))
    equal(reported['paired_recipe_contrasts'], contrasts, 'contrasts')
    if len(conditions) != 80 or reported['selected_states'] != 284 or reported['task_and_generation_states'] != 94:
        raise AssertionError('Complete-panel counts changed')
    write_json(output, dict(status='passed', selected_states=284, conditions=80, paired_recipe_conditions=len(contrasts),
        task_and_generation_states=94, candidate_comparisons=94*4*320, counterfactual_mode_pairs=94*4*96,
        scalar_summaries=sum(len(c['observations']) for c in conditions+contrasts),
        checked_sha256=checked, verifier_sha256=sha256(__file__),
        scope='Independent reconstruction of every retained prediction, task-margin, counterfactual, generation and paired-recipe summary. Complete ordered empirical resampling uses the declared one, two or four identities. Single-seed uncertainty is unavailable. The two selected endpoint identities at constant N4/N14 update32768 are distinguished from the four measured core identities. Undefined selected ratios are never removed to obtain a finite mean.'))
    print('Verified all single-pass predictions and', len(contrasts), 'paired recipe conditions', flush=True)


if __name__ == '__main__': main()
