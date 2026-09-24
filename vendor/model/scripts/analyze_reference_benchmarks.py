#!/usr/bin/env python
"""Summarize every completed ARC question under all selected score conventions."""
import argparse
import json
from pathlib import Path

import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def normalize(item, scores, convention):
    denominator = (np.ones(len(scores)) if convention == 'sum' else
        np.array([len(a) for a in item['answer_tokens']]) if convention == 'answer_tokens' else np.array(item[convention]))
    if np.any(denominator <= 0): raise AssertionError('An answer normalization is undefined')
    return np.asarray(scores, dtype=float)/denominator


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908'); args = p.parse_args()
    study = Path(args.root).resolve()/args.study; protocol = study/'protocols/reference-benchmark-selection.json'
    spec = json.loads(protocol.read_text()); cohort = json.loads(Path(spec['cohort']).read_text())['examples']
    if len(cohort) != 3548 or len(spec['cases']) != 2: raise AssertionError('Both complete benchmark panels are required')
    inputs = [protocol, Path(spec['cohort']), study/'verification/reference-benchmark-cohort.json']
    observations = []
    for case in spec['cases']:
        parent = study/'measurements'/case['name']; proof_path = study/'verification/reasoning'/(case['name']+'.json')
        proof = json.loads(proof_path.read_text())
        if proof['status'] != 'passed' or proof['case'] != case: raise AssertionError('A selected benchmark lacks raw reconstruction')
        for filename in ['manifest.json', 'results.json']:
            path = parent/filename
            if sha256(path) != proof['checked_sha256'][str(path)]: raise AssertionError('A verified benchmark changed')
            inputs.append(path)
        inputs.append(proof_path); observations.append(json.loads((parent/'results.json').read_text()))
    output = study/'analysis/reference-benchmarks'; output.mkdir(parents=True, exist_ok=False)
    bind_run(output, inputs, vars(args)); states = []; scores_saved = []
    for case, observation in zip(spec['cases'], observations, strict=True):
        modes = {m['mode']: m for m in observation['modes']}
        if list(modes) != spec['modes']: raise AssertionError('An operator intervention was omitted')
        raw_scores = np.full((4, 3548, max(len(e['choices']) for e in cohort)), np.nan)
        groups = []; decisions = {}
        for k, mode in enumerate(spec['modes']):
            entries = modes[mode]['examples']
            if [e['id'] for e in entries] != [e['id'] for e in cohort]: raise AssertionError('The complete question order changed')
            for i, item in enumerate(entries): raw_scores[k, i, :len(item['scores'])] = item['scores']
            for convention in spec['normalizations']:
                predictions = []; margins = []; errors = []; native_margins = []; native_predictions = []
                for i, definition in enumerate(cohort):
                    score = normalize(definition, entries[i]['scores'], convention)
                    native = normalize(definition, modes['native']['examples'][i]['scores'], convention)
                    answer = definition['correct']; predictions.append(int(np.argmax(score))); native_predictions.append(int(np.argmax(native)))
                    margins.append(float(score[answer]-np.max(np.delete(score, answer))))
                    native_margins.append(float(native[answer]-np.max(np.delete(native, answer))))
                    errors.append(float(np.max(np.abs(score-native))))
                decisions[mode, convention] = predictions
                for subset in ['ARC-Easy', 'ARC-Challenge']:
                    indices = [i for i, item in enumerate(cohort) if item['task'] == subset]
                    correct = np.array([predictions[i] == cohort[i]['correct'] for i in indices])
                    margin = np.array(margins)[indices]; baseline = np.array(native_margins)[indices]; error = np.array(errors)[indices]
                    certified = baseline > 2*error
                    if np.any(certified & ~correct): raise AssertionError('An operator score certificate failed')
                    groups.append(dict(mode=mode, normalization=convention, subset=subset, examples=len(indices),
                        correct=int(correct.sum()), accuracy=float(correct.mean()), strictly_correct=int(np.sum(margin > 0)),
                        mean_correct_margin=float(margin.mean()), answer_changes_from_native=sum(predictions[i] != native_predictions[i] for i in indices),
                        strictly_correct_native=int(np.sum(baseline > 0)), certified_preserved_correct=int(certified.sum()),
                        maximum_candidate_score_error=float(error.max()), mean_candidate_score_error=float(error.mean())))
        normalization_comparisons = []
        for mode in spec['modes']:
            for subset in ['ARC-Easy', 'ARC-Challenge']:
                selected = [i for i, e in enumerate(cohort) if e['task'] == subset]
                for convention in ['sum', 'answer_tokens', 'answer_utf8_bytes']:
                    normalization_comparisons.append(dict(mode=mode, subset=subset, reference='answer_characters', alternative=convention,
                        answer_changes=sum(decisions[mode, 'answer_characters'][i] != decisions[mode, convention][i] for i in selected)))
        states.append(dict(case=case, groups=groups, normalization_comparisons=normalization_comparisons,
            maximum_forward_context=observation['maximum_forward_context'], long_context_qualification=observation['long_context_qualification'],
            external_target_nll={name: modes[name]['external_target_nll'] for name in spec['modes']}))
        scores_saved.append(raw_scores)
    np.savez_compressed(output/'measurements.npz', candidate_scores=np.stack(scores_saved),
        candidate_counts=np.array([len(e['choices']) for e in cohort]))
    write_json(output/'results.json', dict(status='complete', schema='complete-reference-arc-summary-v1', states=states,
        scope='All 3548 pinned ARC test questions, both released checkpoints, four native operator modes and four explicit '
        'scoring conventions are retained. Scores use proper prefixes without BOS/EOS and native CPU float32. '
        'Character normalization corresponds to the pinned public harness definition; byte and token normalizations '
        'are distinct sensitivity observations. Counts describe this finite benchmark, not independent pretraining '
        'replications, a contamination-controlled generalization guarantee or a proof of thermodynamic criticality.'))
    write_json(output/'manifest.json', dict(status='complete', binding_sha256=sha256(output/'binding.json'),
        results_sha256=sha256(output/'results.json'), raw_sha256=sha256(output/'measurements.npz')))
    print('Completed full ARC summary for both released checkpoints and all score conventions', flush=True)


if __name__ == '__main__': main()
