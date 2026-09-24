#!/usr/bin/env python
"""Check source sampling, native tokenization and exact composition answers."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm

from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;folder=study/'data/reasoning-cohort'
    output=study/'verification/reasoning-cohort.json'
    if output.exists():raise FileExistsError(output)
    meta=json.loads((folder/'manifest.json').read_text());cohort=folder/'cohort.json'
    if meta['status']!='complete' or sha256(cohort)!=meta['cohort_sha256']:raise AssertionError('Reasoning cohort changed')
    binding=json.loads((folder/'binding.json').read_text());checked={str(cohort):sha256(cohort)}
    if sha256(folder/'binding.json')!=meta['binding_sha256']:raise AssertionError('Reasoning binding changed')
    for name,digest in binding['inputs'].items():
        if sha256(name)!=digest:raise AssertionError('Reasoning source changed')
        checked[name]=digest
    for name,digest in binding['source_files'].items():
        if sha256(folder/'source'/name)!=digest:raise AssertionError('Reasoning producer snapshot changed')
    spec=json.loads(cohort.read_text());examples=spec['examples'];tokenizer=spm.SentencePieceProcessor(model_file=str(root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'))
    if len(examples)!=320 or len({x['id'] for x in examples})!=320:raise AssertionError('Reasoning item count or identities changed')
    for item in examples:
        prefix=[2]+tokenizer.encode(item['prompt'])
        if prefix!=item['prompt_tokens']:raise AssertionError('Reasoning prefix tokens changed')
        if len(item['choices'])!=len(item['answer_tokens']) or not 0<=item['correct']<len(item['choices']):raise AssertionError('Invalid answer index')
        for answer,tokens in zip(item['choices'],item['answer_tokens'],strict=True):
            full=[2]+tokenizer.encode(item['prompt']+' '+answer)
            if full!=prefix+tokens or not tokens or len(full)>65:raise AssertionError('Answer tokens are not the exact short continuation')
    arc_source=study/'data/arc-source'
    for panel in spec['arc_selection']:
        original=pq.read_table(arc_source/panel['subset']/'test-00000-of-00001.parquet').to_pylist();eligible=[]
        for row in original:
            prompt='Question: '+row['question']+'\nAnswer:';prefix=[2]+tokenizer.encode(prompt)
            full=[[2]+tokenizer.encode(prompt+' '+s) for s in row['choices']['text']]
            if any(z[:len(prefix)]!=prefix or len(z)<=len(prefix) for z in full) or max(map(len,full))>65:continue
            eligible.append(row)
        indices=np.random.default_rng(panel['seed']).choice(len(eligible),64,replace=False)
        np.testing.assert_array_equal(indices,panel['selected_indices'])
        if len(eligible)!=panel['eligible_rows'] or len(original)!=panel['source_test_rows']:raise AssertionError('ARC eligibility count changed')
        selected=[eligible[int(i)] for i in indices];actual=[x for x in examples if x['task']==panel['subset']]
        if [r['id'] for r in selected]!=panel['ids'] or [r['id'] for r in actual]!=panel['ids']:raise AssertionError('ARC selected identities changed')
        for left,right in zip(selected,actual,strict=True):
            if right['prompt']!='Question: '+left['question']+'\nAnswer:' or right['choices']!=left['choices']['text']:
                raise AssertionError('ARC source text changed')
            if right['correct']!=left['choices']['label'].index(left['answerKey']):raise AssertionError('ARC answer key changed')
    pairs={};depths={1:0,2:0,3:0}
    def entity(text):return text.lower().removeprefix('the ')
    for item in examples:
        if not item['task'].startswith('composition-'):continue
        graph={}
        for fact in item['facts']:
            place=re.fullmatch(r'(.+) is in the (.+)\.',fact);carry=re.fullmatch(r'(.+) carries the (.+)\.',fact)
            if place:child,parent=place.groups()
            elif carry:parent,child=carry.groups()
            else:raise AssertionError('Unrecognized composition fact')
            child,parent=entity(child),entity(parent)
            if child in graph:raise AssertionError('Ambiguous composition fact')
            graph[child]=parent
        question=re.search(r'\nQuestion: Where is (.+)\?\nAnswer:$',item['prompt'])
        if not question:raise AssertionError('Unknown composition question')
        current=entity(question.group(1));visited=[]
        while current in graph:
            if current in visited:raise AssertionError('Cyclic composition facts')
            visited.append(current);current=graph[current]
        if len(visited)!=item['depth'] or current!=entity(item['choices'][item['correct']]):raise AssertionError('Composition answer is not entailed at the declared depth')
        story=' '.join(item['facts'][i] for i in item['sentence_order'])
        if not item['prompt'].startswith(story+'\nQuestion:'):raise AssertionError('Composition sentence order changed')
        pairs.setdefault(item['pair'],[]).append(item);depths[item['depth']]+=1
    if len(pairs)!=96 or depths!={1:64,2:64,3:64}:raise AssertionError('Composition balance changed')
    for pair in pairs.values():
        if len(pair)!=2:raise AssertionError('Missing counterfactual member')
        x,y=sorted(pair,key=lambda q:q['variant'])
        if [x['correct'],y['correct']]!=[0,1] or x['choices']!=y['choices'] or x['sentence_order']!=y['sentence_order']:
            raise AssertionError('Counterfactual answer/order balance changed')
        first,second=x['choices'];swap=x['prompt'].replace('the '+first+'.','the __FIRST__.').replace('the '+second+'.','the '+first+'.').replace('the __FIRST__.','the '+second+'.')
        if swap!=y['prompt']:raise AssertionError('A counterfactual changed more than the two locations')
    write_json(output,dict(status='passed',examples=320,arc_items=128,composition_items=192,counterfactual_pairs=96,
        cohort_sha256=sha256(cohort),checked_sha256=checked,
        scope='Independent reconstruction of ARC eligibility/sampling and answer keys; explicit SentencePiece continuation tokenization; graph traversal of every composition answer and exact paired location substitution. No model score is used.'))
    print('Reasoning cohort independently verified:320 items,96 counterfactual pairs',flush=True)


if __name__=='__main__':main()
