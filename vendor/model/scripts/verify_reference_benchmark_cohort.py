#!/usr/bin/env python
"""Reconstruct every selected ARC question, answer and exact candidate prefix."""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import sentencepiece as spm

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908'); args = p.parse_args()
    root = Path(args.root).resolve(); study = root/args.study; output = study/'verification/reference-benchmark-cohort.json'
    if output.exists(): raise FileExistsError(output)
    checked = {}

    def check(path, expected=None):
        path = Path(path).resolve(); digest = sha256(path)
        if expected is not None and digest != expected: raise AssertionError('Changed benchmark cohort evidence: '+str(path))
        checked[str(path)] = digest; return digest

    def load(path):
        check(path); return json.loads(Path(path).read_text())

    selection = load(study/'protocols/reference-benchmark-selection.json')
    cohort_path = Path(selection['cohort']); folder = cohort_path.parent
    meta = load(folder/'manifest.json'); check(cohort_path, selection['cohort_sha256']); check(cohort_path, meta['cohort_sha256'])
    if meta['status'] != 'complete': raise AssertionError('The complete ARC cohort is required')
    check(folder/'binding.json', meta['binding_sha256']); binding = load(folder/'binding.json')
    for name, digest in binding['inputs'].items(): check(name, digest)
    for name, digest in binding['source_files'].items(): check(folder/'source'/name, digest)
    for name, digest in selection['inputs_sha256'].items(): check(name, digest)
    cohort = load(cohort_path); examples = cohort['examples']; source = Path(cohort['source'])
    source_meta = load(source/'manifest.json'); check(source/'manifest.json', cohort['source_manifest_sha256'])
    for record in source_meta['files']: check(source/record['path'], record['sha256'])
    tokenizer_path = root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'; check(tokenizer_path)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path)); cursor = 0; answers = maximum = nonascii = 0
    for subset, count in [('ARC-Easy', 2376), ('ARC-Challenge', 1172)]:
        rows = pq.read_table(source/subset/'test-00000-of-00001.parquet').to_pylist()
        if len(rows) != count: raise AssertionError('The complete pinned test split changed')
        for row in rows:
            item = examples[cursor]; cursor += 1
            prompt = f'Question: {row["question"]}\nAnswer:'
            correct = next(i for i, label in enumerate(row['choices']['label']) if label == row['answerKey'])
            if (item['id'], item['source_id'], item['task'], item['prompt'], item['choices'], item['correct']) != (
                    subset+'/'+row['id'], row['id'], subset, prompt, row['choices']['text'], correct):
                raise AssertionError('A source question, candidate order or answer was changed')
            prefix = tokenizer.encode(prompt, out_type=int)
            if prefix != item['prompt_tokens'] or not prefix: raise AssertionError('Question tokenization changed')
            for i, answer in enumerate(row['choices']['text']):
                tokens = tokenizer.encode(prompt+' '+answer, out_type=int)
                if tokens[:len(prefix)] != prefix or tokens[len(prefix):] != item['answer_tokens'][i] or len(tokens) == len(prefix):
                    raise AssertionError('A candidate lost its exact proper-prefix decomposition')
                if item['answer_characters'][i] != len(answer) or item['answer_utf8_bytes'][i] != len(answer.encode('utf-8')):
                    raise AssertionError('Character or UTF-8-byte normalization changed')
                answers += 1; maximum = max(maximum, len(tokens)-1); nonascii += len(answer) != len(answer.encode('utf-8'))
    if cursor != len(examples) or cursor != 3548 or meta['actual_maximum_forward_context'] != maximum or maximum > 1024:
        raise AssertionError('A complete test question was omitted or truncated')
    if selection['normalizations'] != ['sum', 'answer_tokens', 'answer_characters', 'answer_utf8_bytes']:
        raise AssertionError('A prespecified scoring convention is missing')
    write_json(output, dict(status='passed', examples=cursor, candidates=answers, maximum_proper_prefix=maximum,
        choices_with_distinct_character_and_byte_count=nonascii, excluded_questions=0,
        cohort_sha256=check(cohort_path), selection_sha256=check(study/'protocols/reference-benchmark-selection.json'),
        checked_sha256=checked, verifier_sha256=sha256(__file__),
        scope='All pinned ARC test rows, supplied answer keys, candidate order, explicit length conventions and no-BOS/no-EOS proper prefixes are reconstructed. No model score has been selected or evaluated by this checker.'))
    print('Verified all', cursor, 'ARC questions and', answers, 'candidates', flush=True)


if __name__ == '__main__': main()
