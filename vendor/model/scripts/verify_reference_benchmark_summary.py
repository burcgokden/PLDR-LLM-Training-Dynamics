#!/usr/bin/env python
"""Reconstruct complete ARC normalization, intervention counts and qualifications."""
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import math
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908'); args = p.parse_args()
    study = Path(args.root).resolve()/args.study; output = study/'verification/reference-benchmark-summary.json'
    if output.exists(): raise FileExistsError(output)
    cache = {}; checked = {}

    def check(path, expected=None):
        path = Path(path).resolve(); st = path.stat(); key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in cache: cache[key] = sha256(path)
        digest = cache[key]
        if expected is not None and digest != expected: raise AssertionError('Changed ARC evidence: '+str(path))
        checked[str(path)] = digest; return digest

    def load(path):
        check(path); return load_json_strict(Path(path).read_text())

    def proof(path):
        value = load(path)
        if value['status'] != 'passed': raise AssertionError('Every selected ARC observation must be independently reconstructed')
        for name, digest in value['checked_sha256'].items(): check(name, digest)
        return value

    def compare(observed, expected):
        if set(observed) != set(expected): raise AssertionError('An ARC summary diagnostic was omitted')
        for name, value in expected.items():
            if isinstance(value, (int, str)):
                if observed[name] != value: raise AssertionError('An ARC count or identity changed: '+name)
            else: np.testing.assert_allclose(observed[name], value, rtol=3e-11, atol=2e-12)

    selection = load(study/'protocols/reference-benchmark-selection.json')
    cohort = load(selection['cohort'])['examples']; definitions = {e['id']: e for e in cohort}
    proof(study/'verification/reference-benchmark-cohort.json')
    folder = study/'analysis/reference-benchmarks'; meta = load(folder/'manifest.json')
    if meta['status'] != 'complete': raise AssertionError('The complete ARC synthesis is required')
    for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256'), ('measurements.npz', 'raw_sha256')]: check(folder/name, meta[key])
    binding = load(folder/'binding.json')
    for name, digest in binding['inputs'].items(): check(name, digest)
    for name, digest in binding['source_files'].items(): check(folder/'source'/name, digest)
    report = load(folder/'results.json')
    if report['status'] != 'complete' or len(report['states']) != 2 or len(cohort) != 3548: raise AssertionError('The complete reference benchmark inventory is required')
    group_count = comparisons = 0; qualifications = []
    with np.load(folder/'measurements.npz') as stored:
        sizes = [len(e['choices']) for e in cohort]
        np.testing.assert_array_equal(stored['candidate_counts'], sizes)
        if stored['candidate_scores'].shape != (2, 4, 3548, max(sizes)): raise AssertionError('The complete candidate-score inventory changed')
        for state_index, (case, state) in enumerate(zip(selection['cases'], report['states'], strict=True)):
            if state['case'] != case: raise AssertionError('A released checkpoint identity changed')
            raw_proof = proof(study/'verification/reasoning'/(case['name']+'.json'))
            if raw_proof['case'] != case: raise AssertionError('A benchmark proof has the wrong model')
            parent = study/'measurements'/case['name']; result = load(parent/'results.json'); index = load(parent/'prediction-index.json')
            maximum = max(map(len, index['inputs'])); longest = sorted(range(len(index['inputs'])), key=lambda i: (-len(index['inputs'][i]), i))[:2]
            lengths = [len(index['inputs'][i]) for i in longest]
            with np.load(parent/'measurements.npz') as raw:
                np.testing.assert_array_equal(raw['qualification_indices'], longest)
                native = raw['qualification_native_logits']; replay = raw['qualification_replayed_logits']; restored = raw['qualification_restored_logits']
                if native.shape != (2, 32000) or native.dtype != np.float32 or not np.isfinite(native).all(): raise AssertionError('Invalid native benchmark qualification')
                if native.tobytes() != replay.tobytes() or native.tobytes() != restored.tobytes(): raise AssertionError('Benchmark native replay or restoration failed')
                if raw['qualification_caches'].shape != (2, 5, 3, 1, 14, 64, 64) or not np.isfinite(raw['qualification_caches']).all():
                    raise AssertionError('Long-prefix operator capture is incomplete')
            qualification = dict(indices=longest, lengths=lengths, own_G_replay_byte_equal=True, restoration_byte_equal=True)
            if state['maximum_forward_context'] != maximum or state['long_context_qualification'] != qualification or result['long_context_qualification'] != qualification:
                raise AssertionError('A retained benchmark qualification changed')
            qualifications.append(dict(case=case['name'], prefixes=2, vocabulary_logits=64000, **qualification))
            modes = {m['mode']: m for m in result['modes']}; groups = {(g['mode'], g['normalization'], g['subset']): g for g in state['groups']}
            expected_groups = {(mode, norm, subset) for mode in selection['modes'] for norm in selection['normalizations'] for subset in ['ARC-Easy', 'ARC-Challenge']}
            if set(groups) != expected_groups or len(state['groups']) != 32: raise AssertionError('A normalization, subset or operator mode was omitted')
            predictions = {}
            for mode_index, mode in enumerate(selection['modes']):
                entries = modes[mode]['examples']
                if [e['id'] for e in entries] != [e['id'] for e in cohort]: raise AssertionError('A benchmark question was omitted or reordered')
                for i, item in enumerate(entries):
                    scores = stored['candidate_scores'][state_index, mode_index, i]
                    if scores[:sizes[i]].tobytes() != np.asarray(item['scores'], dtype=float).tobytes() or not np.isnan(scores[sizes[i]:]).all():
                        raise AssertionError('A candidate score changed or an unused candidate was populated')
                for norm in selection['normalizations']:
                    all_values = []
                    for i, definition in enumerate(cohort):
                        old = modes['native']['examples'][i]['scores']; now = entries[i]['scores']
                        if norm == 'sum': denominators = [1]*sizes[i]
                        elif norm == 'answer_tokens': denominators = [len(tokens) for tokens in definition['answer_tokens']]
                        elif norm == 'answer_characters': denominators = [len(answer) for answer in definition['choices']]
                        elif norm == 'answer_utf8_bytes': denominators = [len(answer.encode('utf-8')) for answer in definition['choices']]
                        else: raise AssertionError('An unselected score convention entered the summary')
                        score = [a/b for a, b in zip(now, denominators, strict=True)]; previous = [a/b for a, b in zip(old, denominators, strict=True)]
                        answer = definition['correct']; pred = max(range(sizes[i]), key=score.__getitem__); old_pred = max(range(sizes[i]), key=previous.__getitem__)
                        margin = score[answer]-max(x for j, x in enumerate(score) if j != answer)
                        baseline = previous[answer]-max(x for j, x in enumerate(previous) if j != answer)
                        error = max(abs(a-b) for a, b in zip(score, previous, strict=True)); certified = finite_greater(baseline, 2*error, 'scripts/verify_reference_benchmark_summary.py:101')
                        if finite_greater(abs(margin-baseline), 2*error+2e-12, 'scripts/verify_reference_benchmark_summary.py:102') or (certified and pred != answer): raise AssertionError('A benchmark score-margin certificate failed')
                        all_values.append(dict(prediction=pred, correct=pred == answer, margin=margin, baseline=baseline,
                            error=error, changed=pred != old_pred, certified=certified)); comparisons += 1
                    predictions[mode, norm] = [v['prediction'] for v in all_values]
                    for subset in ['ARC-Easy', 'ARC-Challenge']:
                        chosen = [v for v, d in zip(all_values, cohort, strict=True) if d['task'] == subset]; count = len(chosen)
                        expected = dict(mode=mode, normalization=norm, subset=subset, examples=count,
                            correct=sum(v['correct'] for v in chosen), accuracy=sum(v['correct'] for v in chosen)/count,
                            strictly_correct=sum(v['margin'] > 0 for v in chosen), mean_correct_margin=math.fsum(v['margin'] for v in chosen)/count,
                            answer_changes_from_native=sum(v['changed'] for v in chosen), strictly_correct_native=sum(v['baseline'] > 0 for v in chosen),
                            certified_preserved_correct=sum(v['certified'] for v in chosen), maximum_candidate_score_error=max(v['error'] for v in chosen),
                            mean_candidate_score_error=math.fsum(v['error'] for v in chosen)/count)
                        compare(groups[mode, norm, subset], expected); group_count += 1
                np.testing.assert_allclose(state['external_target_nll'][mode], modes[mode]['external_target_nll'], rtol=0, atol=0)
            expected_contrasts = []
            for mode in selection['modes']:
                for subset in ['ARC-Easy', 'ARC-Challenge']:
                    for norm in ['sum', 'answer_tokens', 'answer_utf8_bytes']:
                        changed = sum(predictions[mode, norm][i] != predictions[mode, 'answer_characters'][i] for i, e in enumerate(cohort) if e['task'] == subset)
                        expected_contrasts.append(dict(mode=mode, subset=subset, reference='answer_characters', alternative=norm, answer_changes=changed))
            if state['normalization_comparisons'] != expected_contrasts: raise AssertionError('A normalization sensitivity outcome changed')
            print('Verified complete normalized ARC outcomes', case['name'], flush=True)
    if (comparisons, group_count) != (113536, 64): raise AssertionError('A complete benchmark comparison was omitted')
    write_json(output, dict(status='passed', selected_states=2, questions_per_state=3548, candidate_margin_comparisons=comparisons,
        normalized_subset_groups=group_count, normalization_contrasts=48, long_context_qualifications=qualifications,
        checked_sha256=checked, verifier_sha256=sha256(__file__),
        scope='Independent scalar reconstruction of every normalization, question decision, score-margin certificate and '
        'paired operator/normalization contrast. Retained longest-prefix own-G replay and restoration agree bytewise. '
        'Every underlying complete proper-prefix vocabulary-logit reconstruction is required and rehashed; native '
        'forwards are not rerun by this checker. No population reasoning or thermodynamic claim is certified.'))
    print('Verified all', comparisons, 'normalized candidate-margin comparisons', flush=True)


if __name__ == '__main__': main()
