#!/usr/bin/env python
"""Count actual source-position reuse in completed, matched training prefixes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def position_counts(rows, offsets, documents, objective):
    """Integer interval endpoints; a source position, not a token type, is the unit."""
    if objective == 'external':
        return np.bincount(rows * 513 + offsets + 64,
                           minlength=documents * 513).reshape(documents, 513)
    start = np.bincount(rows * 514 + offsets + 1, minlength=documents * 514)
    stop = np.bincount(rows * 514 + offsets + 65, minlength=documents * 514)
    return np.cumsum((start - stop).reshape(documents, 514), axis=1)[:, :513]


def summarize(counts, crop_ids, documents):
    events = int(counts.sum())
    unique = int(np.count_nonzero(counts))
    return dict(target_events=events, unique_target_positions=unique,
                repeated_target_events=events - unique,
                repeated_target_fraction=1 - unique / events,
                maximum_target_exposure=int(counts.max()),
                unique_crops=int(np.unique(crop_ids).size),
                crop_events=int(crop_ids.size), unique_documents=int(documents))


def prepare(root, study, repo, output):
    source = study / 'protocols/repetition-analysis-selection.json'
    spec = json.loads(source.read_text())
    pairs = [p for p in spec['pairs'] if p['heads'] == 14 and p['seed'] == 640101]
    if len(pairs) != 9:
        raise AssertionError('All five constant and four source-recipe prefix pairs are required')
    inputs = {str(source): sha256(source)}
    cases = []
    for pair in pairs:
        case = dict(name=pair['name'], family=pair['family'], recipe=pair['recipe'],
                    step=pair['step'], objective='external' if pair['family'] == 'constant' else 'all_nonpadding')
        for side in ['old', 'new']:
            entry = pair[side]
            state = Path(entry['case']['state'])
            parent = Path(entry['parent_manifest'] if side == 'old' else entry['case']['parent_manifest'])
            meta = json.loads(parent.read_text())
            if meta['status'] != 'complete' or meta.get('completed_step', pair['step']) < pair['step']:
                raise AssertionError('Only completed retained checkpoints may enter this analysis')
            args = meta['arguments']
            sampling = Path(args['root']) / args['study'] / 'runs' / args['run_id'] / 'sampling.npz' if 'study' in args else parent.parent / 'sampling.npz'
            token_path = root / 'data' / ('refinedweb-4608' if side == 'old' else 'refinedweb-onepass-524288') / 'tokens.npy'
            proof = Path(entry['verification'] if side == 'old' else entry['native_verification'])
            proof_data = json.loads(proof.read_text())
            if proof_data['status'] not in ['passed', 'complete']:
                raise AssertionError('The retained state requires its completed upstream check')
            case[side] = dict(state=str(state), parent=str(parent), sampling=str(sampling),
                              tokens=str(token_path), verification=str(proof))
            for path in [state, parent, sampling, token_path, proof]:
                if str(path) not in inputs:
                    inputs[str(path)] = sha256(path)
            expected_state = meta.get('saved_states', {}).get(str(pair['step']), {}).get('sha256', meta.get('checkpoint_sha256'))
            if expected_state != inputs[str(state)]:
                raise AssertionError('The retained parent manifest does not bind this state')
            if side == 'old' and entry['case']['state_sha256'] != inputs[str(state)]:
                raise AssertionError('The paired checkpoint selection changed')
        cases.append(case)
    names = ['scripts/analyze_source_reuse.py', 'scripts/verify_source_reuse.py',
             'src/model_rg/controlled.py', 'src/model_rg/provenance.py']
    write_json(output, dict(schema='completed-source-reuse-selection-v1',
        selected_at=datetime.now(timezone.utc).isoformat(), cases=cases, inputs_sha256=inputs,
        producer_sources={n: sha256(repo / n) for n in names}, batch_size=32,
        stream_seed=640001, repeated_documents=3072, onepass_documents=524288,
        interpretation='Nine already completed matched prefix pairs, seven objective/horizon combinations, '
        'one shared realized stream per law. Source recipes share the same exposure counts. '
        'No additional native training or independent stream replication. Source position means '
        'document identity plus token offset; natural token types may repeat.'))
    print('Selected all nine completed prefix pairs for source-position counting', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--study', default='scheduled-training-feasible-20260908')
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve(); study = root / args.study
    repo = Path(__file__).resolve().parents[1]
    selection = study / 'protocols/source-reuse-selection.json'
    if args.prepare:
        if selection.exists():
            raise FileExistsError(selection)
        prepare(root, study, repo, selection)
        return
    spec = json.loads(selection.read_text())
    for name, digest in spec['inputs_sha256'].items():
        if sha256(name) != digest:
            raise AssertionError('A selected source-reuse input changed')
    for name, digest in spec['producer_sources'].items():
        if sha256(repo / name) != digest:
            raise AssertionError('A selected implementation changed')
    out = study / 'analysis/source-reuse'
    out.mkdir(parents=True, exist_ok=False)
    bind_run(out, [selection, *map(Path, spec['inputs_sha256'])], vars(args))
    torch.set_num_threads(1)
    raw = {}; groups = {}; records = []; cache = {}
    for case in spec['cases']:
        objective, step = case['objective'], case['step']
        key = f'{objective}-t{step}'
        if key not in groups:
            group = dict(name=key, objective=objective, step=step, cases=[])
            for side in ['old', 'new']:
                entry = case[side]
                if entry['sampling'] not in cache:
                    with np.load(entry['sampling']) as archive:
                        cache[entry['sampling']] = (archive['rows'], archive['offsets'])
                rows, offsets = [v[:step].reshape(-1) for v in cache[entry['sampling']]]
                if rows.size != 32 * step:
                    raise AssertionError('Incomplete recorded stream')
                tokens = np.load(entry['tokens'], mmap_mode='r')
                histogram = np.zeros(32000, dtype=np.int64)
                if side == 'old':
                    if rows.min() < 0 or rows.max() >= 3072 or offsets.min() < 0 or offsets.max() > 448:
                        raise AssertionError('The recorded repeated crop domain changed')
                    counts = position_counts(rows, offsets, 3072, objective)
                    if objective == 'all_nonpadding':
                        counts[tokens[:3072] == 0] = 0
                    raw[key + '-old-position-counts'] = counts
                    np.add.at(histogram, tokens[:3072].reshape(-1), counts.reshape(-1))
                    result = summarize(counts, 449 * rows + offsets, np.unique(rows).size)
                    if objective == 'external':
                        probabilities = np.full((3072, 449), 1 / (3072 * 449))
                    else:
                        positions = np.arange(1, 513)
                        covering = np.minimum(448, positions - 1) - np.maximum(0, positions - 64) + 1
                        probabilities = np.broadcast_to(covering / (3072 * 449), (3072, 512)).copy()
                        probabilities[tokens[:3072, 1:] == 0] = 0
                    result['expected_unique_target_positions'] = float((-np.expm1(rows.size * np.log1p(-probabilities))).sum())
                    result['available_target_positions'] = int(np.count_nonzero(probabilities))
                else:
                    if rows.min() < 0 or rows.max() >= 524288 or np.any(offsets % 64) or offsets.min() < 0 or offsets.max() > 448:
                        raise AssertionError('The recorded single-pass block domain changed')
                    block_ids = 8 * rows + offsets // 64
                    occupancy = np.bincount(block_ids, minlength=524288 * 8)
                    if occupancy.max() != 1:
                        raise AssertionError('A single-pass source block was reused')
                    raw[f't{step}-new-block-occupancy'] = occupancy.astype(np.uint8)
                    for start in range(0, rows.size, 16384):
                        rr, oo = rows[start:start + 16384], offsets[start:start + 16384]
                        values = tokens[rr, oo + 64] if objective == 'external' else tokens[rr[:, None], oo[:, None] + np.arange(1, 65)]
                        values = values.reshape(-1)
                        if objective == 'all_nonpadding':
                            values = values[values != 0]
                        token_ids, multiplicities = np.unique(values, return_counts=True)
                        histogram[token_ids] += multiplicities
                    events = int(histogram.sum())
                    result = dict(target_events=events, unique_target_positions=events,
                        repeated_target_events=0, repeated_target_fraction=0., maximum_target_exposure=1,
                        unique_crops=int(rows.size), crop_events=int(rows.size),
                        unique_documents=int(np.unique(rows).size))
                raw[key + '-' + side + '-target-histogram'] = histogram
                group[side] = result
            groups[key] = group
        group = groups[key]; group['cases'].append(case['name'])
        for side in ['old', 'new']:
            saved = torch.load(case[side]['state'], map_location='cpu', mmap=True, weights_only=True)
            if saved['step'] != step:
                raise AssertionError('The selected checkpoint time changed')
            np.testing.assert_array_equal(saved['supervised_target_counts'].numpy(), raw[key + '-' + side + '-target-histogram'])
        records.append(dict(case=case['name'], group=key, checkpoint_target_histograms='exact'))
        print('Counted', case['name'], flush=True)
    np.savez_compressed(out / 'measurements.npz', **raw)
    write_json(out / 'results.json', dict(status='complete', groups=list(groups.values()), records=records,
        scope=spec['interpretation'], formulas='For K independent sampled crops and source-position inclusion probability p_j, E[unique]=sum_j(1-(1-p_j)^K). Positions within a crop need not be independent.'))
    write_json(out / 'manifest.json', dict(status='complete', schema='completed-source-reuse-v1',
        binding_sha256=sha256(out / 'binding.json'), raw_sha256=sha256(out / 'measurements.npz'),
        results_sha256=sha256(out / 'results.json'), selection_sha256=sha256(selection),
        matched_pairs=9, checkpoint_histograms=18, exposure_groups=7, additional_training_updates=0))


if __name__ == '__main__':
    main()
