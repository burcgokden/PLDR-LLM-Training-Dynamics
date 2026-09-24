#!/usr/bin/env python
"""Verify demonstration semantics, pairing and every prompt-conditioned token prefix."""
import argparse
import json
from pathlib import Path
import re

import sentencepiece as spm

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908')
    args = p.parse_args()
    root = Path(args.root).resolve()
    study = root/args.study
    output = study/'verification/prompt-conditioning-cohort.json'
    if output.exists():
        raise FileExistsError(output)
    checked = {}

    def check(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise AssertionError('Changed prompting cohort evidence: ' + str(path))
        checked[str(path)] = digest

    def load(path):
        check(path)
        return json.loads(Path(path).read_text())

    protocol = study/'protocols/prompt-conditioning-selection.json'
    spec = load(protocol)
    for path, digest in spec['inputs_sha256'].items():
        check(path, digest)
    parent = Path(spec['cohort']).parent
    meta = load(parent/'manifest.json')
    if meta['status'] != 'complete' or meta['examples'] != 576:
        raise AssertionError('The full prompting cohort is required')
    check(parent/'binding.json', meta['binding_sha256'])
    binding = load(parent/'binding.json')
    for name, digest in binding['source_files'].items():
        check(parent/'source'/name, digest)
    data = load(spec['cohort'])
    check(spec['cohort'], spec['cohort_sha256'])
    original = {e['id']: e for e in load(data['base_cohort'])['examples'] if 'pair' in e}
    check(data['base_cohort'], data['base_cohort_sha256'])
    demos = {d['id']: d for d in data['demonstrations']}
    if len(demos) != 12 or len(original) != 192 or data['demonstration_levels'] != [0, 2, 4]:
        raise AssertionError('The declared demonstration or target panel changed')
    original_people = {'john', 'mary', 'daniel', 'sandra', 'david', 'sarah', 'james', 'emma'}
    original_objects = {'apple', 'book', 'ball', 'key', 'coin', 'ring', 'toy', 'marble', 'box'}
    for demo in demos.values():
        if set(n.lower() for n in demo['people']) & original_people or set(demo['objects']) & original_objects:
            raise AssertionError('Demonstrations reuse the original target entity vocabulary')
        edges = {}
        for fact in demo['facts']:
            sentence = fact.lower()
            located = re.fullmatch(r'(?:the )?([a-z]+) is in the ([a-z]+)\.', sentence)
            carried = re.fullmatch(r'([a-z]+) carries the ([a-z]+)\.', sentence)
            if located:
                left, right = located.groups()
            elif carried:
                right, left = carried.groups()
            else:
                raise AssertionError('A demonstration contains an unrecognized relation')
            if left in edges:
                raise AssertionError('A demonstration has ambiguous containment')
            edges[left] = right
        current = demo['query'].lower().removeprefix('the ')
        visited = set()
        depth = 0
        while current in edges:
            if current in visited:
                raise AssertionError('A demonstration relation cycle was introduced')
            visited.add(current)
            current = edges[current]
            depth += 1
        if current != demo['answer'] or depth != demo['depth'] or demo['choices'][demo['correct']] != current:
            raise AssertionError('A demonstration answer does not follow its stated premises')
        order = demo['sentence_order']
        if sorted(order) != list(range(len(demo['facts']))):
            raise AssertionError('A demonstration premise was omitted')
        prompt = ' '.join(demo['facts'][i] for i in order)+f'\nQuestion: Where is {demo["query"]}?\nAnswer:'
        if prompt != demo['prompt']:
            raise AssertionError('Demonstration text differs from its checked relation graph')
    for depth in [1, 2, 3]:
        for shots in [2, 4]:
            selected = [d for d in demos.values() if d['depth'] == depth and d['index'] < shots]
            if len(selected) != shots or sum(d['correct'] == 0 for d in selected) != shots//2:
                raise AssertionError('The demonstration answer roles are not balanced')
    tokenizer_path = root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'
    check(tokenizer_path)
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    seen = set()
    pairs = {}
    maximum = 0
    for item in data['examples']:
        base = original[item['base_id']]
        shots = item['demonstrations']
        identity = (base['id'], shots)
        if identity in seen or shots not in [0, 2, 4]:
            raise AssertionError('A base question or shot condition was repeated or omitted')
        seen.add(identity)
        selected = sorted((d for d in demos.values() if d['depth'] == base['depth'] and d['index'] < shots), key=lambda d: d['index'])
        prefix = '\n\n'.join(f'Example {i+1}:\n{d["prompt"]} {d["answer"]}' for i, d in enumerate(selected))
        prompt = prefix+'\n\n'+base['prompt'] if shots else base['prompt']
        if item['prompt'] != prompt or item['demonstration_ids'] != [d['id'] for d in selected]:
            raise AssertionError('The selected demonstrations changed')
        for field in ['choices', 'correct', 'facts', 'sentence_order', 'variant', 'depth']:
            if item[field] != base[field]:
                raise AssertionError('A target premise or counterfactual answer was changed')
        if item['id'] != base['id']+f'-shots{shots}' or item['pair'] != base['pair']+f'-shots{shots}' or item['task'] != base['task']+f'-shots{shots}':
            raise AssertionError('A prompting identity was relabeled')
        tokens = [2]+tokenizer.encode(prompt)
        if tokens != item['prompt_tokens']:
            raise AssertionError('Prompt tokenization changed')
        for choice, answer in zip(item['choices'], item['answer_tokens'], strict=True):
            full = [2]+tokenizer.encode(prompt+' '+choice)
            if full != tokens+answer or not answer or len(full) > 513:
                raise AssertionError('A candidate is not scored from its proper bounded prefix')
            maximum = max(maximum, len(full)-1)
        if shots == 0 and (item['prompt_tokens'] != base['prompt_tokens'] or item['answer_tokens'] != base['answer_tokens']):
            raise AssertionError('The zero-demonstration control differs from the original question')
        pairs.setdefault(item['pair'], []).append(item)
    if seen != {(name, shots) for name in original for shots in [0, 2, 4]} or len(pairs) != 288:
        raise AssertionError('The complete base-question by demonstration-count product is required')
    for pair in pairs.values():
        if len(pair) != 2 or {e['variant'] for e in pair} != {0, 1} or pair[0]['demonstration_ids'] != pair[1]['demonstration_ids']:
            raise AssertionError('A premise counterfactual does not share the same demonstrations')
    if maximum != meta['actual_maximum_forward_context']:
        raise AssertionError('The reported context bound changed')
    write_json(output, dict(status='passed', examples=576, counterfactual_pairs=288, base_counterfactual_pairs=96,
        demonstrations=12, shot_counts=[0, 2, 4], actual_maximum_forward_context=maximum,
        cohort_sha256=sha256(spec['cohort']), checked_sha256=checked, verifier_sha256=sha256(__file__),
        scope='Independent containment/carrying graph answers, balanced demonstration roles, disjoint demonstration entity vocabulary, complete repeated-question pairing and exact SentencePiece proper-prefix reconstruction. All zero-demonstration targets match the original verified short questions. This validates a finite prompting design, not a new independent question sample or training run.'))
    print('Verified all 576 prompt-conditioned questions and 12 demonstrations', flush=True)


if __name__ == '__main__':
    main()
