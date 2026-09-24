#!/usr/bin/env python
"""Check the five-state synthesis against independently verified observations."""
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import math
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--study', default='scheduled-training-20260908')
    args = parser.parse_args()
    study = Path(args.root).resolve() / args.study
    output = study / 'verification/initial-inference-summary.json'
    if output.exists():
        raise FileExistsError(output)
    checked = {}

    def check(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise AssertionError('Changed inference summary evidence: ' + str(path))
        checked[str(path)] = digest
        return digest

    def load(path):
        check(path)
        return load_json_strict(Path(path).read_text())

    def verified(path):
        data = load(path)
        if data['status'] != 'passed':
            raise AssertionError('An underlying verification did not pass')
        for name, digest in data['checked_sha256'].items():
            check(name, digest)
        return data

    def close(left, right):
        np.testing.assert_allclose(left, right, rtol=3e-11, atol=2e-12)

    folder = study / 'analysis/initial-inference-mechanism'
    manifest = load(folder / 'manifest.json')
    if manifest['status'] != 'complete':
        raise AssertionError('Incomplete mechanism synthesis')
    for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256')]:
        check(folder / name, manifest[key])
    for name, digest in manifest['figures'].items():
        check(folder / name, digest)
    binding = load(folder / 'binding.json')
    for name, digest in binding['inputs'].items():
        check(name, digest)
    for name, digest in binding['source_files'].items():
        check(folder / 'source' / name, digest)
    verified(study / 'verification/prefix-mechanism-initial.json')
    transport = verified(study / 'verification/row-input-transport.json')
    transports = {x['name']: x for x in transport['summaries']}
    selected = load(study / 'protocols/prefix-mechanism-initial-selection.json')['cases']
    result = load(folder / 'results.json')
    if result['status'] != 'complete' or len(result['states']) != len(selected) or len(selected) != 5:
        raise AssertionError('The selected synthesis must contain all five states')
    items_checked = 0
    for case, state in zip(selected, result['states'], strict=True):
        name = case['name'].removeprefix('prefix-')
        if state['name'] != name:
            raise AssertionError('A selected state changed order or identity')
        verified(study / 'verification/reasoning' / ('reasoning-' + name + '.json'))
        prefix = load(study / 'measurements' / case['name'] / 'results.json')
        reasoning = load(study / 'measurements' / ('reasoning-' + name) / 'results.json')
        row = transports['row-transport-' + name]
        close(state['row_transport_factors'], row['row_overall_factors'])
        close(state['suffix_transport_factors'], row['suffix_overall_factors'])
        for variant, summary in zip(['prefix', 'target_swap', 'suffix_swap'], state['prefix'], strict=True):
            if summary['variant'] != variant:
                raise AssertionError('Missing intervention in synthesis')
            records = [r for r in prefix['conditions'] if r['variant'] == variant and r['length'] < 64]
            if [r['length'] for r in records] != [16, 32, 48]:
                raise AssertionError('The equal-weight prefix panel changed')
            for target, source in [('mean_predictive_kl', 'mean_predictive_kl'),
                                   ('mean_relative_G_difference', 'mean_relative_G_difference'),
                                   ('mean_nll_change', 'nll_change')]:
                close(summary[target], math.fsum(r[source] for r in records) / 3)
        modes = {m['mode']: m for m in reasoning['modes']}
        native = {x['id']: x for x in modes['native']['examples']}
        if len(native) != 320 or [m['mode'] for m in state['modes']] != ['native', 'calibration_G', 'permuted_G', 'row_projection']:
            raise AssertionError('The selected task or intervention inventory changed')
        for summary in state['modes']:
            changed = modes[summary['mode']]
            errors = []
            flips = certified = positive = 0
            for example in changed['examples']:
                reference = native[example['id']]
                scores = example['scores']; old_scores = reference['scores']; correct = reference['correct']
                error = max(abs(x-y) for x, y in zip(scores, old_scores, strict=True))
                old_margin = old_scores[correct] - max(v for i, v in enumerate(old_scores) if i != correct)
                new_margin = scores[correct] - max(v for i, v in enumerate(scores) if i != correct)
                if finite_greater(abs(new_margin-old_margin), 2*error + 2e-12, 'scripts/verify_initial_inference_summary.py:102'):
                    raise AssertionError('Candidate-score margin bound violated')
                old_decision = max(range(len(old_scores)), key=old_scores.__getitem__)
                decision = max(range(len(scores)), key=scores.__getitem__)
                flips += decision != old_decision
                positive += old_margin > 0
                certified += finite_greater(old_margin, 2*error, 'scripts/verify_initial_inference_summary.py:109')
                if finite_greater(old_margin, 2*error, 'scripts/verify_initial_inference_summary.py:109') and decision != correct:
                    raise AssertionError('A certified answer was lost')
                errors.append(error); items_checked += 1
            for key, value in [('answer_decision_changes', flips), ('strictly_correct_native', positive),
                               ('certified_preserved_correct', certified)]:
                if summary[key] != value:
                    raise AssertionError('Incorrect task-margin count: ' + key)
            close(summary['maximum_candidate_score_error'], max(errors))
            close(summary['mean_maximum_candidate_score_error'], math.fsum(errors)/len(errors))
            close(summary['external_target_nll'], changed['external_target_nll'])
            close(summary['paired_external_nll_change'], changed['external_target_nll']-modes['native']['external_target_nll'])
            # Every task outcome is carried through, including weak and adverse outcomes.
            underlying = load_json_strict((study/'verification/reasoning'/('reasoning-'+name+'.json')).read_text())
            expected = next(m for m in underlying['summaries'] if m['mode'] == summary['mode'])
            for key in ['accuracy', 'token_mean_accuracy', 'composition_both_correct']:
                if summary[key] != expected[key]:
                    raise AssertionError('A retained task outcome was altered')
    write_json(output, dict(status='passed', selected_states=5, candidate_margin_comparisons=items_checked,
        checked_sha256=checked, verifier_sha256=sha256(__file__),
        scope='Independent synthesis arithmetic and complete outcome correspondence. Underlying retained-logit, stage-energy and native-replay verifications are required and their bound evidence is rehashed. This does not certify unobserved Lipschitz constants or infer task competence from operator stability.'))
    print('Passed five-state mechanism synthesis with', items_checked, 'candidate margin comparisons', flush=True)


if __name__ == '__main__':
    main()
