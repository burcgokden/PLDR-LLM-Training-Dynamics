#!/usr/bin/env python
"""Retain all outcomes of the completed, paired demonstration-count panel."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def summarize(scores, native, examples):
    correct = np.array([e['correct'] for e in examples])
    prediction = scores.argmax(1); original = native.argmax(1)
    margin = scores[np.arange(len(scores)), correct]-scores[np.arange(len(scores)), 1-correct]
    native_margin = native[np.arange(len(scores)), correct]-native[np.arange(len(scores)), 1-correct]
    error = np.max(np.abs(scores-native), axis=1)
    certified = native_margin > 2*error
    if np.any(certified & (prediction != correct)):
        raise AssertionError('A certified answer margin was lost')
    pairs = defaultdict(dict)
    for i, item in enumerate(examples): pairs[item['pair']][item['variant']] = i
    responses = []; both = []; strict = []
    for pair in pairs.values():
        if set(pair) != {0, 1}: raise AssertionError('An incomplete premise pair entered the summary')
        i, j = pair[0], pair[1]
        if (correct[i], correct[j]) != (0, 1): raise AssertionError('Counterfactual orientation changed')
        responses.append(margin[i]+margin[j])
        both.append(prediction[i] == 0 and prediction[j] == 1)
        strict.append(margin[i] > 0 and margin[j] > 0)
    return dict(examples=len(scores), pairs=len(pairs), correct=int(np.sum(prediction == correct)),
        accuracy=float(np.mean(prediction == correct)), strictly_correct=int(np.sum(margin > 0)),
        mean_correct_margin=float(margin.mean()), both_correct=int(np.sum(both)),
        both_correct_fraction=float(np.mean(both)), strictly_both_correct=int(np.sum(strict)),
        mean_signed_premise_response=float(np.mean(responses)),
        positive_premise_responses=int(np.sum(np.array(responses) > 0)),
        answer_changes_from_native=int(np.sum(prediction != original)),
        strictly_correct_native=int(np.sum(native_margin > 0)),
        certified_preserved_correct=int(np.sum(certified)),
        maximum_candidate_score_error=float(error.max()), mean_candidate_score_error=float(error.mean()))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--study', default='scheduled-training-20260908'); args = parser.parse_args()
    study = Path(args.root).resolve()/args.study; output = study/'analysis/prompt-conditioning'
    selection_path = study/'protocols/prompt-conditioning-selection.json'
    selection = json.loads(selection_path.read_text())
    cohort_path = Path(selection['cohort']); cohort = json.loads(cohort_path.read_text())['examples']
    if len(cohort) != 576 or len(selection['cases']) != 5 or selection['demonstration_levels'] != [0, 2, 4]:
        raise AssertionError('The complete selected prompt panel is required')
    inputs = [selection_path, cohort_path, study/'verification/prompt-conditioning-cohort.json']
    cohort_proof = json.loads(inputs[-1].read_text())
    if cohort_proof['status'] != 'passed': raise AssertionError('The demonstration cohort is unverified')
    observations = []; originals = []
    for case in selection['cases']:
        pair = []
        for name in [case['name'], case['base_reasoning_case']]:
            folder = study/'measurements'/name; proof_path = study/'verification/reasoning'/(name+'.json')
            proof = json.loads(proof_path.read_text())
            if proof['status'] != 'passed' or proof['case']['name'] != name:
                raise AssertionError('A selected proper-prefix observation is unverified')
            for filename in ['manifest.json', 'results.json']:
                path = folder/filename
                if sha256(path) != proof['checked_sha256'][str(path)]:
                    raise AssertionError('A verified task result changed')
                inputs.append(path)
            inputs.append(proof_path); pair.append(json.loads((folder/'results.json').read_text()))
        observations.append(pair[0]); originals.append(pair[1])
    output.mkdir(parents=True, exist_ok=False); bind_run(output, inputs, vars(args))
    states = []; raw_scores = []; raw_original = []
    for case, observation, original in zip(selection['cases'], observations, originals, strict=True):
        scores = np.array([[item['scores'] for item in mode['examples']] for mode in observation['modes']])
        if scores.shape != (4, 576, 2): raise AssertionError('The complete score tensor changed')
        if [m['mode'] for m in observation['modes']] != selection['modes']:
            raise AssertionError('An operator intervention was omitted')
        old_modes = {m['mode']: {e['id']: e for e in m['examples']} for m in original['modes']}
        zero_indices = [i for i, e in enumerate(cohort) if e['demonstrations'] == 0]
        old = np.array([[old_modes[mode][cohort[i]['base_id']]['scores'] for i in zero_indices]
                        for mode in selection['modes']])
        modes = []
        for k, mode in enumerate(selection['modes']):
            groups = []; contrasts = []
            for shots in [0, 2, 4]:
                for depth in [0, 1, 2, 3]:
                    indices = [i for i, e in enumerate(cohort) if e['demonstrations'] == shots and (depth == 0 or e['depth'] == depth)]
                    definitions = [cohort[i] for i in indices]
                    summary = summarize(scores[k, indices], scores[0, indices], definitions)
                    groups.append(dict(demonstrations=shots, depth=depth, **summary))
                    if shots:
                        lookup = {e['base_id']: i for i, e in enumerate(cohort) if e['demonstrations'] == 0}
                        before = [lookup[e['base_id']] for e in definitions]
                        correct = np.array([e['correct'] for e in definitions])
                        old_correct = scores[k, before].argmax(1) == correct
                        new_correct = scores[k, indices].argmax(1) == correct
                        old_summary = summarize(scores[k, before], scores[0, before], [cohort[i] for i in before])
                        contrasts.append(dict(demonstrations=shots, depth=depth,
                            newly_correct=int(np.sum(new_correct & ~old_correct)),
                            newly_incorrect=int(np.sum(~new_correct & old_correct)),
                            answer_changes=int(np.sum(scores[k, indices].argmax(1) != scores[k, before].argmax(1))),
                            mean_margin_change=summary['mean_correct_margin']-old_summary['mean_correct_margin'],
                            both_correct_change=summary['both_correct']-old_summary['both_correct'],
                            mean_premise_response_change=summary['mean_signed_premise_response']-old_summary['mean_signed_premise_response']))
            difference = scores[k, zero_indices]-old[k]
            modes.append(dict(mode=mode, groups=groups, demonstration_contrasts=contrasts,
                zero_demonstration_repeat=dict(examples=192,
                    maximum_score_difference=float(np.max(np.abs(difference))),
                    rms_score_difference=float(np.sqrt(np.mean(difference*difference))),
                    answer_changes=int(np.sum(scores[k, zero_indices].argmax(1) != old[k].argmax(1)))),
                external_target_nll=observation['modes'][k]['external_target_nll']))
        states.append(dict(case=case, modes=modes,
            long_context_qualification=observation['long_context_qualification'],
            maximum_forward_context=observation['maximum_forward_context']))
        raw_scores.append(scores); raw_original.append(old)
    np.savez_compressed(output/'measurements.npz', candidate_scores=np.stack(raw_scores),
        original_zero_demonstration_scores=np.stack(raw_original))
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), layout='constrained', sharey=True)
    labels = ['Released 1', 'Released 2', 'Constant g=1, 32,768', 'Constant g=0, 32,768', 'Constant g=1, 131,072']
    for depth, ax in enumerate(axes, 1):
        for state, label in zip(states, labels, strict=True):
            values = [g['both_correct_fraction'] for g in state['modes'][0]['groups'] if g['depth'] == depth]
            ax.plot([0, 2, 4], values, marker='o', markersize=4, label=label)
        ax.set_xticks([0, 2, 4]); ax.set_xlabel('Worked examples'); ax.set_title(f'Composition depth {depth}')
        ax.set_ylim(-.015, .26)
    axes[0].set_ylabel('Both premise variants answered correctly')
    axes[1].legend(fontsize=6.5, frameon=False, loc='upper center')
    fig.savefig(output/'prompt-conditioning.pdf'); fig.savefig(output/'prompt-conditioning.png', dpi=180); plt.close(fig)
    write_json(output/'results.json', dict(schema='complete-prompt-conditioning-summary-v1', status='complete',
        states=states, scope='All five fixed checkpoints, all 576 questions, all zero/two/four demonstration levels, '
        'all composition depths and all four operator modes are retained. Depth zero denotes pooling the three depths, '
        'not an additional task. Outcomes and paired changes describe this finite task law without population coverage '
        'or a claim of general reasoning. The original zero-demonstration prefixes are repeated at CPU batch size8 '
        'instead of32; score and decision differences are measured explicitly. These observations add no training identities.'))
    write_json(output/'manifest.json', dict(status='complete', binding_sha256=sha256(output/'binding.json'),
        results_sha256=sha256(output/'results.json'), raw_sha256=sha256(output/'measurements.npz'),
        figures={name: sha256(output/name) for name in ['prompt-conditioning.pdf', 'prompt-conditioning.png']}))
    print('Completed all five prompt-conditioning summaries', flush=True)


if __name__ == '__main__': main()
