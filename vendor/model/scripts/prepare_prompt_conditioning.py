#!/usr/bin/env python
"""Freeze a balanced demonstration panel around the existing composition questions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import sentencepiece as spm

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908')
    args = p.parse_args()
    root = Path(args.root).resolve()
    study = root/args.study
    repo = Path(__file__).resolve().parents[1]
    parent_path = study/'data/reasoning-cohort/cohort.json'
    parent_proof = study/'verification/reasoning-cohort.json'
    original_selection = study/'protocols/reasoning-mechanism-selection.json'
    original = json.loads(original_selection.read_text())
    proof = json.loads(parent_proof.read_text())
    if proof['status'] != 'passed' or proof['cohort_sha256'] != sha256(parent_path):
        raise AssertionError('The existing finite composition questions must be verified and unchanged')
    base = [e for e in json.loads(parent_path.read_text())['examples'] if 'pair' in e]
    if len(base) != 192:
        raise AssertionError('All 96 existing counterfactual pairs are required')
    cases = []
    prior = [parent_path, parent_proof, original_selection]
    for case in original['cases']:
        if case.get('role') == 'scheduled_endpoint':
            continue
        name = 'prompt-'+case['name'].removeprefix('reasoning-')
        if (study/'measurements'/name).exists():
            raise AssertionError('Prompting selection must precede every selected new observation')
        cases.append(dict(case, name=name, base_reasoning_case=case['name']))
        prior.extend([study/'measurements'/case['name']/'results.json',
                      study/'verification/reasoning'/(case['name']+'.json')])
    if len(cases) != 5:
        raise AssertionError('The same five released/control checkpoints are required')
    tokenizer_path = root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    if tokenizer.bos_id() != 2 or tokenizer.eos_id() != 3:
        raise AssertionError('Tokenizer special tokens changed')
    people = ['Alice', 'Bob', 'Carol', 'Frank', 'Helen', 'Kevin', 'Linda', 'Peter']
    objects = ['cup', 'pencil', 'phone', 'bottle', 'watch', 'plate', 'hat', 'towel']
    places = ['kitchen', 'garden', 'bedroom', 'bathroom', 'office', 'hallway']
    demonstrations = []
    for depth in [1, 2, 3]:
        for index in range(4):
            actor, other = people[2*index:2*index+2]
            item, distractor = objects[2*index:2*index+2]
            locations = [places[(2*depth+index)%6], places[(2*depth+index+3)%6]]
            correct = index % 2
            carrier = [actor, other][correct]
            facts = [f'{actor} is in the {locations[0]}.', f'{other} is in the {locations[1]}.']
            if depth == 1:
                query = carrier
            elif depth == 2:
                facts += [f'{actor} carries the {item}.', f'{other} carries the {distractor}.']
                query = 'the '+[item, distractor][correct]
            else:
                facts += [f'The {item} is in the bag.', f'{carrier} carries the bag.']
                query = 'the '+item
            order = list(range(len(facts)))
            if index >= 2:
                order.reverse()
            prompt = ' '.join(facts[i] for i in order)+f'\nQuestion: Where is {query}?\nAnswer:'
            demonstrations.append(dict(id=f'demo-d{depth}-i{index}', depth=depth, index=index, people=[actor, other],
                objects=[item, distractor], choices=locations, correct=correct, answer=locations[correct],
                facts=facts, sentence_order=order, query=query, prompt=prompt))
    examples = []
    for shots in [0, 2, 4]:
        for item in base:
            selected = [d for d in demonstrations if d['depth'] == item['depth'] and d['index'] < shots]
            prefix = '\n\n'.join(f'Example {i+1}:\n{d["prompt"]} {d["answer"]}' for i, d in enumerate(selected))
            prompt = prefix+'\n\n'+item['prompt'] if shots else item['prompt']
            tokens = [2]+tokenizer.encode(prompt)
            full = [[2]+tokenizer.encode(prompt+' '+choice) for choice in item['choices']]
            if any(t[:len(tokens)] != tokens or len(t) <= len(tokens) for t in full) or max(map(len, full)) > 513:
                raise AssertionError('A declared prompt or proper candidate prefix exceeds the fixed context limit')
            examples.append(dict(item, id=item['id']+f'-shots{shots}', base_id=item['id'],
                pair=item['pair']+f'-shots{shots}', task=item['task']+f'-shots{shots}', demonstrations=shots,
                demonstration_ids=[d['id'] for d in selected], prompt=prompt, prompt_tokens=tokens,
                answer_tokens=[t[len(tokens):] for t in full]))
    output = study/'data/prompt-conditioning-cohort'
    output.mkdir(parents=True, exist_ok=False)
    prior.append(tokenizer_path)
    bind_run(output, prior, vars(args))
    write_json(output/'cohort.json', dict(examples=examples, demonstrations=demonstrations,
        base_cohort=str(parent_path), base_cohort_sha256=sha256(parent_path), demonstration_levels=[0, 2, 4]))
    write_json(output/'manifest.json', dict(status='complete', schema='prompt-conditioning-cohort-v1',
        binding_sha256=sha256(output/'binding.json'), cohort_sha256=sha256(output/'cohort.json'),
        recorded_at=datetime.now(timezone.utc).isoformat(), examples=576, counterfactual_pairs=288,
        base_counterfactual_pairs=96, demonstration_levels=[0, 2, 4], maximum_forward_context=512,
        actual_maximum_forward_context=max(len(e['prompt_tokens'])+max(map(len,e['answer_tokens']))-1 for e in examples),
        semantics='The original 192 premise-counterfactual items are unchanged apart from preceding demonstrations. Demonstration people and objects are disjoint from the original entity vocabulary. The first two and all four demonstrations balance which person supplies the answer; their answers are checked by containment/carrying graph traversal.',
        scope='Developmental task-format diagnostic specified after observing the original zero-shot panel. Every shot count, depth, counterfactual variant, original checkpoint and operator mode is retained. Repeated questions across shot counts are not independent data or additional training.'))
    producers = {k: v for k, v in original['producer_sources'].items() if k != 'scripts/measure_reasoning_mechanism.py'}
    producers['scripts/measure_prompt_conditioning.py'] = sha256(repo/'scripts/measure_prompt_conditioning.py')
    protocol = study/'protocols/prompt-conditioning-selection.json'
    if protocol.exists():
        raise FileExistsError(protocol)
    inputs = [*prior, output/'cohort.json', output/'manifest.json']
    write_json(protocol, dict(schema='prompt-conditioning-selection-v1', frozen_at=datetime.now(timezone.utc).isoformat(),
        cases=cases, cohort=str(output/'cohort.json'), cohort_manifest=str(output/'manifest.json'),
        cohort_sha256=sha256(output/'cohort.json'), modes=original['modes'], maximum_forward_context=512,
        forward_batch_size=8, demonstration_levels=[0, 2, 4], examples=576, base_counterfactual_pairs=96,
        prior_information='All five original short zero-shot task panels and their operator interventions were known; their exact results and verification records are bound. These prompting choices are developmental and are frozen before any new prompt-conditioned model score.',
        conditioning='Use the same five immutable checkpoints and the same 64 calibration crops, 64 external language targets, and four operator modes as the original task study. Demonstration counts share the complete original counterfactual questions. The zero-demonstration panel repeats the same proper prefixes at the new declared CPU batch size8; any arithmetic or decision change is measured.',
        qualification='Before any candidate scores, replay each state with its own native G cache on its two longest selected prefixes and restore native mode. Both full vocabulary outputs must agree bytewise. The longest selected contexts are retained in the output.',
        inference='Demonstration-dependent task accuracy measures elicited competence on this fixed finite question law. It does not establish a critical surface, broad reasoning or generalization of the controlled short-crop training law to all longer contexts.',
        inputs_sha256={str(path): sha256(path) for path in inputs}, producer_sources=producers))
    print('Frozen', len(examples), 'prompt-conditioned questions at five existing checkpoints; maximum prefix',
          max(len(e['prompt_tokens'])+max(map(len,e['answer_tokens']))-1 for e in examples), flush=True)


if __name__ == '__main__':
    main()
