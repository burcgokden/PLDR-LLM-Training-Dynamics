#!/usr/bin/env python
"""Freeze complete ARC splits, explicit normalization and two released states."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908'); args = p.parse_args()
    root = Path(args.root).resolve(); study = root/args.study; repo = Path(__file__).resolve().parents[1]
    source = study/'data/arc-source'; tokenizer_path = root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'
    harness = repo/'internal/reference-evaluation'; harness_manifest = json.loads((harness/'manifest.json').read_text())
    inputs = [source/'manifest.json', tokenizer_path, harness/'manifest.json']
    source_manifest = json.loads((source/'manifest.json').read_text())
    for record in source_manifest['files']:
        path = source/record['path']
        if sha256(path) != record['sha256']: raise AssertionError('The pinned ARC source changed')
        inputs.append(path)
    for name, digest in harness_manifest['files'].items():
        if sha256(harness/name) != digest: raise AssertionError('The pinned benchmark implementation changed')
        inputs.append(harness/name)
    original_path = study/'protocols/reasoning-mechanism-selection.json'
    original = json.loads(original_path.read_text()); inputs.append(original_path)
    cases = []
    for case in original['cases']:
        if case['kind'] != 'pretrained': continue
        name = 'benchmark-'+case['name'].removeprefix('reasoning-')
        if (study/'measurements'/name).exists(): raise AssertionError('Benchmark selection must precede the new scores')
        cases.append(dict(case, name=name, base_reasoning_case=case['name']))
        inputs += [study/'measurements'/case['name']/'results.json', study/'verification/reasoning'/(case['name']+'.json')]
    if len(cases) != 2: raise AssertionError('Both released references are required')
    # The prompting outcome is prior information, not an independent holdout.
    inputs += [study/'analysis/prompt-conditioning/results.json', study/'verification/prompt-conditioning-summary.json']
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path)); examples = []; counts = {}
    for subset, expected in [('ARC-Easy', 2376), ('ARC-Challenge', 1172)]:
        rows = pq.read_table(source/subset/'test-00000-of-00001.parquet').to_pylist()
        if len(rows) != expected: raise AssertionError('A complete ARC test split is missing')
        for row in rows:
            prompt = 'Question: '+row['question']+'\nAnswer:'; prefix = tokenizer.encode(prompt)
            choices = row['choices']['text']; full = [tokenizer.encode(prompt+' '+answer) for answer in choices]
            if not prefix or any(t[:len(prefix)] != prefix or len(t) <= len(prefix) or len(t)-1 > 1024 for t in full):
                raise AssertionError('Every selected question needs its complete, untruncated proper prefixes')
            labels = row['choices']['label']; correct = labels.index(row['answerKey'])
            characters = [len(answer) for answer in choices]; byte_counts = [len(answer.encode('utf-8')) for answer in choices]
            if min(characters) <= 0: raise AssertionError('A selected answer has undefined length normalization')
            examples.append(dict(id=subset+'/'+row['id'], source_id=row['id'], task=subset, prompt=prompt,
                choices=choices, correct=correct, prompt_tokens=prefix,
                answer_tokens=[t[len(prefix):] for t in full], answer_characters=characters, answer_utf8_bytes=byte_counts))
        counts[subset] = expected
    if len({e['id'] for e in examples}) != 3548: raise AssertionError('Question identities are duplicated')
    out = study/'data/reference-benchmark-cohort'; out.mkdir(parents=True, exist_ok=False)
    bind_run(out, inputs, vars(args))
    write_json(out/'cohort.json', dict(examples=examples, source=str(source), source_manifest_sha256=sha256(source/'manifest.json')))
    maximum = max(len(e['prompt_tokens'])+max(map(len, e['answer_tokens']))-1 for e in examples)
    write_json(out/'manifest.json', dict(status='complete', schema='complete-reference-arc-cohort-v1',
        binding_sha256=sha256(out/'binding.json'), cohort_sha256=sha256(out/'cohort.json'),
        frozen_at=datetime.now(timezone.utc).isoformat(), examples=3548, subsets=counts,
        actual_maximum_forward_context=maximum, excluded_questions=0,
        tokenization='Pinned SentencePiece processor; public harness question template and a single leading space before each choice. No added BOS or EOS. Every answer is a prefix-preserving continuation and is scored without truncation.',
        normalization='Retain raw answer log-probability sums, per-answer-token means, per-Unicode-character means matching the pinned public acc_norm definition, and separate UTF-8-byte means. Character and byte counts exclude the separating leading space.',
        scope='All questions in both pinned ARC test splits, with no performance-based filtering. This is not a claim that the released pretraining corpus excludes these published questions.'))
    producers = {k: v for k, v in original['producer_sources'].items() if k != 'scripts/measure_reasoning_mechanism.py'}
    producers['scripts/measure_reference_benchmarks.py'] = sha256(repo/'scripts/measure_reference_benchmarks.py')
    inputs += [out/'cohort.json', out/'manifest.json']
    protocol = study/'protocols/reference-benchmark-selection.json'
    if protocol.exists(): raise FileExistsError(protocol)
    write_json(protocol, dict(schema='complete-reference-arc-selection-v1', frozen_at=datetime.now(timezone.utc).isoformat(),
        cases=cases, cohort=str(out/'cohort.json'), cohort_manifest=str(out/'manifest.json'), cohort_sha256=sha256(out/'cohort.json'),
        modes=original['modes'], examples=3548, subsets=counts, maximum_forward_context=1024, forward_batch_size=8,
        normalizations=['sum', 'answer_tokens', 'answer_characters', 'answer_utf8_bytes'],
        prior_information='The initial short ARC subset and all composition/prompting outcomes were known. The two models are unchanged. This complete-split extension is selected before new full-benchmark scores and is not an independent holdout of the already observed questions.',
        comparison='Use the pinned public ARC template, candidate order, no-BOS/no-EOS tokenization and character-normalized score definition. The new observations deliberately use proper prefix probabilities on the released CPU float32 checkpoints; exact reproduction of published scores or the original evaluation arithmetic is not asserted.',
        qualification='Before task scores, retain bytewise own-G replay and restoration on the two longest proper prefixes. Calibration and RMS-matched head permutation use the unchanged 64-context RefinedWeb panel.',
        retention='All questions, both subsets, both released states, four operator modes and four normalization conventions remain in the result. These task measurements add no training identities.',
        inputs_sha256={str(path): sha256(path) for path in inputs}, producer_sources=producers))
    print('Frozen all', len(examples), 'ARC questions; maximum proper prefix', maximum, flush=True)


if __name__ == '__main__': main()
