#!/usr/bin/env python
"""Independent finite-panel counts, margins, prompt repeats and replay checks."""
import argparse
from collections import defaultdict
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import math
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def panel(items, reference, definitions):
    margins = []; errors = []; original_margins = []; predictions = []
    original_predictions = []; correct = []; pairs = defaultdict(dict)
    for item, definition in zip(items, definitions, strict=True):
        old = reference[item['id']]; answer = definition['correct']; scores = item['scores']
        prediction = 0 if scores[0] >= scores[1] else 1
        previous = 0 if old['scores'][0] >= old['scores'][1] else 1
        margin = scores[answer]-scores[1-answer]
        old_margin = old['scores'][answer]-old['scores'][1-answer]
        error = max(abs(x-y) for x, y in zip(scores, old['scores'], strict=True))
        if finite_greater(abs(margin-old_margin), 2*error+2e-12, 'scripts/verify_prompt_conditioning.py:24'):
            raise AssertionError('The score-margin inequality failed')
        if finite_greater(old_margin, 2*error, 'scripts/verify_prompt_conditioning.py:26') and prediction != answer:
            raise AssertionError('A certified correct answer changed')
        predictions.append(prediction); original_predictions.append(previous); correct.append(answer)
        margins.append(margin); original_margins.append(old_margin); errors.append(error)
        pairs[definition['pair']][definition['variant']] = (prediction, margin, scores, answer)
    responses = []; both = []; strict = []
    for pair in pairs.values():
        if set(pair) != {0, 1}: raise AssertionError('A premise counterfactual was omitted')
        left, right = pair[0], pair[1]
        if (left[3], right[3]) != (0, 1): raise AssertionError('Premise orientation changed')
        responses.append((right[2][1]-right[2][0])-(left[2][1]-left[2][0]))
        both.append(left[0] == 0 and right[0] == 1); strict.append(left[1] > 0 and right[1] > 0)
    count = len(items); successes = sum(x == y for x, y in zip(predictions, correct, strict=True))
    return dict(examples=count, pairs=len(pairs), correct=successes, accuracy=successes/count,
        strictly_correct=sum(x > 0 for x in margins), mean_correct_margin=math.fsum(margins)/count,
        both_correct=sum(both), both_correct_fraction=sum(both)/len(pairs), strictly_both_correct=sum(strict),
        mean_signed_premise_response=math.fsum(responses)/len(pairs), positive_premise_responses=sum(x > 0 for x in responses),
        answer_changes_from_native=sum(x != y for x, y in zip(predictions, original_predictions, strict=True)),
        strictly_correct_native=sum(x > 0 for x in original_margins),
        certified_preserved_correct=sum(m > 2*e for m, e in zip(original_margins, errors, strict=True)),
        maximum_candidate_score_error=max(errors), mean_candidate_score_error=math.fsum(errors)/count)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--study', default='scheduled-training-20260908'); args = parser.parse_args()
    study = Path(args.root).resolve()/args.study; output = study/'verification/prompt-conditioning-summary.json'
    if output.exists(): raise FileExistsError(output)
    checked = {}; cache = {}

    def check(path, expected=None):
        path = Path(path).resolve(); stat = path.stat(); key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in cache: cache[key] = sha256(path)
        digest = cache[key]
        if expected is not None and digest != expected: raise AssertionError('Changed prompt evidence: '+str(path))
        checked[str(path)] = digest; return digest

    def load(path):
        check(path); return load_json_strict(Path(path).read_text())

    def verified(path):
        proof = load(path)
        if proof['status'] != 'passed': raise AssertionError('Underlying proper-prefix verification is required')
        for name, digest in proof['checked_sha256'].items(): check(name, digest)
        return proof

    def compare(left, right):
        if isinstance(right, dict):
            if set(left) != set(right): raise AssertionError('A finite-panel diagnostic was omitted or added')
            for key, value in right.items(): compare(left[key], value)
        elif isinstance(right, (int, bool, str)):
            if left != right: raise AssertionError('A finite-panel count or identity changed')
        else: np.testing.assert_allclose(left, right, rtol=3e-11, atol=2e-12)

    selection = load(study/'protocols/prompt-conditioning-selection.json')
    cohort = load(selection['cohort'])['examples']; definitions = {e['id']: e for e in cohort}
    cohort_proof = load(study/'verification/prompt-conditioning-cohort.json')
    if cohort_proof['status'] != 'passed': raise AssertionError('The selected demonstration answers are unverified')
    for key in ['checked_sha256', 'checked_files']:
        for name, digest in cohort_proof.get(key, {}).items(): check(name, digest)
    folder = study/'analysis/prompt-conditioning'; meta = load(folder/'manifest.json')
    if meta['status'] != 'complete': raise AssertionError('The complete prompting analysis is required')
    for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256'), ('measurements.npz', 'raw_sha256')]:
        check(folder/name, meta[key])
    for name, digest in meta['figures'].items(): check(folder/name, digest)
    binding = load(folder/'binding.json')
    for name, digest in binding['inputs'].items(): check(name, digest)
    for name, digest in binding['source_files'].items(): check(folder/'source'/name, digest)
    report = load(folder/'results.json')
    if report['status'] != 'complete' or len(report['states']) != 5 or len(cohort) != 576:
        raise AssertionError('A selected checkpoint or task was omitted')
    comparisons = group_checks = contrast_checks = 0; qualifications = []; repeats = []
    with np.load(folder/'measurements.npz') as stored:
        if stored['candidate_scores'].shape != (5, 4, 576, 2) or stored['original_zero_demonstration_scores'].shape != (5, 4, 192, 2):
            raise AssertionError('The complete score-array inventory changed')
        for state_index, (case, state) in enumerate(zip(selection['cases'], report['states'], strict=True)):
            if state['case'] != case: raise AssertionError('Checkpoint identity changed')
            current_proof = verified(study/'verification/reasoning'/(case['name']+'.json'))
            verified(study/'verification/reasoning'/(case['base_reasoning_case']+'.json'))
            parent = study/'measurements'/case['name']; original = study/'measurements'/case['base_reasoning_case']
            current = load(parent/'results.json'); old = load(original/'results.json'); index = load(parent/'prediction-index.json')
            if current_proof['case'] != case or current['case'] != case: raise AssertionError('The proof belongs to another checkpoint')
            longest = sorted(range(len(index['inputs'])), key=lambda i: (-len(index['inputs'][i]), i))[:2]
            lengths = [len(index['inputs'][i]) for i in longest]
            with np.load(parent/'measurements.npz') as raw:
                np.testing.assert_array_equal(raw['qualification_indices'], longest)
                native = raw['qualification_native_logits']; replay = raw['qualification_replayed_logits']; restored = raw['qualification_restored_logits']
                if native.shape != (2, 32000) or native.dtype != np.float32 or not np.isfinite(native).all():
                    raise AssertionError('Long-prefix qualification output is invalid')
                if native.tobytes() != replay.tobytes() or native.tobytes() != restored.tobytes():
                    raise AssertionError('Long-prefix own-G replay or restoration is not bytewise equal')
                if raw['qualification_caches'].shape != (2, 5, 3, 1, 14, 64, 64) or not np.isfinite(raw['qualification_caches']).all():
                    raise AssertionError('Long-prefix operator capture is incomplete')
            expected = dict(indices=longest, lengths=lengths, own_G_replay_byte_equal=True, restoration_byte_equal=True)
            if state['long_context_qualification'] != expected or current['long_context_qualification'] != expected:
                raise AssertionError('The retained qualification result changed')
            if state['maximum_forward_context'] != max(map(len, index['inputs'])):
                raise AssertionError('The observed prefix length changed')
            qualifications.append(dict(case=case['name'], prefixes=2, vocabulary_logits=64000, **expected))
            modes = {m['mode']: {e['id']: e for e in m['examples']} for m in current['modes']}
            old_modes = {m['mode']: {e['id']: e for e in m['examples']} for m in old['modes']}
            if [m['mode'] for m in state['modes']] != selection['modes']: raise AssertionError('An operator mode was omitted')
            for mode_index, summary in enumerate(state['modes']):
                mode = summary['mode']; values = modes[mode]; original_values = old_modes[mode]
                actual_scores = np.array([values[e['id']]['scores'] for e in cohort])
                if actual_scores.tobytes() != stored['candidate_scores'][state_index, mode_index].tobytes():
                    raise AssertionError('The saved candidate scores changed')
                groups = {(g['demonstrations'], g['depth']): g for g in summary['groups']}
                contrasts = {(g['demonstrations'], g['depth']): g for g in summary['demonstration_contrasts']}
                if set(groups) != {(s, d) for s in [0, 2, 4] for d in [0, 1, 2, 3]} or set(contrasts) != {(s, d) for s in [2, 4] for d in [0, 1, 2, 3]}:
                    raise AssertionError('A demonstration level or composition depth was omitted')
                zero = {e['base_id']: e for e in cohort if e['demonstrations'] == 0}
                for shots, depth in sorted(groups):
                    selected = [e for e in cohort if e['demonstrations'] == shots and (depth == 0 or e['depth'] == depth)]
                    items = [values[e['id']] for e in selected]
                    expected = panel(items, modes['native'], selected)
                    compare(groups[shots, depth], dict(demonstrations=shots, depth=depth, **expected)); group_checks += 1
                    if shots:
                        earlier = [zero[e['base_id']] for e in selected]; before = [values[e['id']] for e in earlier]
                        old_summary = panel(before, modes['native'], earlier)
                        new_predictions = [max(range(2), key=x['scores'].__getitem__) for x in items]
                        old_predictions = [max(range(2), key=x['scores'].__getitem__) for x in before]
                        new_ok = [p == e['correct'] for p, e in zip(new_predictions, selected, strict=True)]
                        old_ok = [p == e['correct'] for p, e in zip(old_predictions, selected, strict=True)]
                        delta = dict(demonstrations=shots, depth=depth,
                            newly_correct=sum(a and not b for a, b in zip(new_ok, old_ok, strict=True)),
                            newly_incorrect=sum(not a and b for a, b in zip(new_ok, old_ok, strict=True)),
                            answer_changes=sum(a != b for a, b in zip(new_predictions, old_predictions, strict=True)),
                            mean_margin_change=expected['mean_correct_margin']-old_summary['mean_correct_margin'],
                            both_correct_change=expected['both_correct']-old_summary['both_correct'],
                            mean_premise_response_change=expected['mean_signed_premise_response']-old_summary['mean_signed_premise_response'])
                        compare(contrasts[shots, depth], delta); contrast_checks += 1
                zero_examples = [e for e in cohort if e['demonstrations'] == 0]
                old_scores = np.array([original_values[e['base_id']]['scores'] for e in zero_examples])
                if old_scores.tobytes() != stored['original_zero_demonstration_scores'][state_index, mode_index].tobytes():
                    raise AssertionError('The original zero-demonstration scores changed')
                differences = []; changes = 0
                for e in zero_examples:
                    new_score = values[e['id']]['scores']; old_score = original_values[e['base_id']]['scores']
                    differences += [x-y for x, y in zip(new_score, old_score, strict=True)]
                    changes += max(range(2), key=new_score.__getitem__) != max(range(2), key=old_score.__getitem__)
                repeat = dict(examples=192, maximum_score_difference=max(map(abs, differences)),
                    rms_score_difference=math.sqrt(math.fsum(x*x for x in differences)/len(differences)), answer_changes=changes)
                compare(summary['zero_demonstration_repeat'], repeat); repeats.append(dict(case=case['name'], mode=mode, **repeat))
                compare(summary['external_target_nll'], next(m for m in current['modes'] if m['mode'] == mode)['external_target_nll'])
                comparisons += 576
            print('Verified complete prompt panel', case['name'], flush=True)
    write_json(output, dict(status='passed', selected_states=5, modes=4, examples_per_state=576,
        candidate_margin_comparisons=comparisons, group_summaries=group_checks, demonstration_contrasts=contrast_checks,
        long_context_qualifications=qualifications, zero_demonstration_repeats=repeats,
        checked_sha256=checked, verifier_sha256=sha256(__file__),
        scope='Independent scalar reductions of all selected prompt/depth/mode summaries and paired contrasts, '
        'plus bytewise score-array correspondence and retained long-context native replay/restoration. '
        'Underlying independent vocabulary-logit reductions are required and rehashed. No new native forward is '
        'executed by this summary checker; no population reasoning or criticality claim is certified.'))
    print('Verified', comparisons, 'candidate-margin comparisons,', group_checks, 'groups and', contrast_checks, 'contrasts', flush=True)


if __name__ == '__main__': main()
