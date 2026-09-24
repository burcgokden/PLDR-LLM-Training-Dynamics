#!/usr/bin/env python
"""Freeze short ARC questions and counterfactual fact-composition probes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import sentencepiece as spm

from model_rg.controlled import bind_run
from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;source=study/'data/arc-source'
    tokenizer_path=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    out=study/'data/reasoning-cohort';out.mkdir(parents=True,exist_ok=False)
    source_meta=json.loads((source/'manifest.json').read_text())
    inputs=[source/'manifest.json',*[source/f['path'] for f in source_meta['files']],
        *[tokenizer_path/n for n in ['tokenizer_config.json','tokenizer.model','tokenizer.json','special_tokens_map.json']]]
    for f in source_meta['files']:
        if sha256(source/f['path'])!=f['sha256']:raise AssertionError('Pinned ARC source changed')
    bind_run(out,inputs,vars(a))
    tokenizer=spm.SentencePieceProcessor(model_file=str(tokenizer_path/'tokenizer.model'))
    if tokenizer.bos_id()!=2 or tokenizer.eos_id()!=3:raise AssertionError('Tokenizer special tokens changed')

    def encode(prompt,choices):
        prefix=[2]+tokenizer.encode(prompt)
        full=[[2]+tokenizer.encode(prompt+' '+answer) for answer in choices]
        if any(tokens[:len(prefix)]!=prefix or len(tokens)<=len(prefix) for tokens in full):return None
        if max(map(len,full))>65:return None
        return dict(prompt_tokens=prefix,answer_tokens=[tokens[len(prefix):] for tokens in full])

    examples=[];arc_selection=[]
    for i,subset in enumerate(['ARC-Easy','ARC-Challenge']):
        all_rows=pq.read_table(source/subset/'test-00000-of-00001.parquet').to_pylist()
        eligible=[]
        for row in all_rows:
            prompt='Question: '+row['question']+'\nAnswer:';choices=row['choices']['text']
            encoding=encode(prompt,choices)
            if encoding is None:continue
            labels=row['choices']['label']
            if row['answerKey'] not in labels:raise AssertionError('ARC answer key has no candidate')
            eligible.append(dict(id=row['id'],task=subset,prompt=prompt,choices=choices,
                correct=labels.index(row['answerKey']),**encoding))
        chosen=np.random.default_rng(650811+i).choice(len(eligible),64,replace=False)
        for index in chosen:examples.append(eligible[int(index)])
        arc_selection.append(dict(subset=subset,source_test_rows=len(all_rows),eligible_rows=len(eligible),
            seed=650811+i,selected_indices=chosen.tolist(),ids=[eligible[int(j)]['id'] for j in chosen]))

    rng=np.random.default_rng(650821)
    people=['John','Mary','Daniel','Sandra','David','Sarah','James','Emma']
    places=['kitchen','garden','bedroom','bathroom','office','hallway']
    objects=['apple','book','ball','key','coin','ring','toy','marble']
    for depth in [1,2,3]:
        for unit in range(32):
            actor,other=rng.choice(people,2,replace=False).tolist()
            locations=rng.choice(places,2,replace=False).tolist()
            item,distractor=rng.choice(objects,2,replace=False).tolist()
            order=rng.permutation(2 if depth==1 else 4).tolist()
            for variant in [0,1]:
                first,second=locations[variant],locations[1-variant]
                facts=[f'{actor} is in the {first}.',f'{other} is in the {second}.']
                if depth==1:query=actor
                elif depth==2:
                    facts+=[f'{actor} carries the {item}.',f'{other} carries the {distractor}.'];query='the '+item
                else:
                    facts+=[f'The {item} is in the box.',f'{actor} carries the box.'];query='the '+item
                story=' '.join(facts[j] for j in order)
                prompt=story+f'\nQuestion: Where is {query}?\nAnswer:'
                encoding=encode(prompt,locations)
                if encoding is None:raise AssertionError('A prespecified composition task exceeds the short context law')
                examples.append(dict(id=f'composition-d{depth}-u{unit}-v{variant}',task=f'composition-{depth}',
                    pair=f'composition-d{depth}-u{unit}',variant=variant,depth=depth,prompt=prompt,
                    choices=locations,correct=variant,facts=facts,sentence_order=order,**encoding))
    if len(examples)!=320:raise AssertionError('Reasoning item count changed')
    write_json(out/'cohort.json',dict(examples=examples,arc_selection=arc_selection))
    write_json(out/'manifest.json',dict(schema='short-reasoning-cohort-v1',status='complete',
        frozen_at=datetime.now(timezone.utc).isoformat(),binding_sha256=sha256(out/'binding.json'),
        cohort_sha256=sha256(out/'cohort.json'),examples=len(examples),arc_items=128,
        composition_items=192,composition_counterfactual_pairs=96,maximum_forward_context=64,
        eligibility='ARC test questions whose separately tokenized question prefix is retained under every answer concatenation, with at most65 total tokens including BOS2. Select64 without replacement per subset using fixed650811/650812 seeds before any model scores. This is a length-restricted subset, not the full ARC benchmark.',
        tokenization='Pinned SentencePiece processor, identical to the corpus preparation pipeline; explicit BOS2; no EOS in answer scores; candidates start with a single space. Every target is scored from its proper prefix of at most64 tokens.',
        synthetic='Three depths of static location composition;32 counterfactual pairs per depth; swap the two location assignments while retaining candidate order, entity names and sentence order. Answers follow the declared containment/carrying semantics. Inspired by fact-composition proxy tasks; these are newly generated probes and not the bAbI dataset.',
        scores='Primary: sum of proper conditional answer-token log probabilities. Sensitivity: mean per answer token. Retain each candidate score and each token log probability. Composition also reports both members correct and signed counterfactual margin change.',
        scope='Task performance on a frozen finite panel. No claim of broad reasoning from fluency, no finetuning on these examples, and no inference of thermodynamic criticality from task accuracy.'))
    print('Frozen',len(examples),'reasoning items',sha256(out/'cohort.json'),arc_selection,flush=True)


if __name__=='__main__':main()
