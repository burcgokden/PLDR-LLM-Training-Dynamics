#!/usr/bin/env python
"""Construct outcome-independent lexical strata of the unused RefinedWeb resource."""
from companion_paths import legacy_path
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import sentencepiece as spm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from model_rg.provenance import sha256, write_json

TECHNICAL = ('algorithm algorithms database software server network networks computer computers '
             'programming function functions variable variables equation equations theorem proof '
             'quantum physics chemical chemistry protein proteins molecular molecule experiment '
             'experiments scientific research mathematics mathematical statistical statistics '
             'data code system systems parameter parameters').split()
NARRATIVE = ('i me my mine we our us she he her him his mother father friend friends family '
             'love loved felt feeling remember remembered told said story stories baby children '
             'child married husband wife home heart').split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=legacy_path('/pldr-data/model'))
    ap.add_argument('--study', required=True)
    args = ap.parse_args()
    root = Path(args.root).resolve(); study = Path(args.study).resolve()
    if not study.is_relative_to(root): raise ValueError('Destination outside authorized experiment root')
    out = study/'data'; out.mkdir(parents=True, exist_ok=False)
    corpus = root/'data/refinedweb-onepass-524288'
    tokenizer = root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'
    spec = dict(schema='pldr-finetuning-data-v1', created_at=datetime.now(timezone.utc).isoformat(),
                input_manifest=str(corpus/'manifest.json'), tokenizer=str(tokenizer),
                incoming_step=40960, stream_seed=640001, blocks=4194304, consumed=1310720,
                technical_words=TECHNICAL, narrative_words=NARRATIVE,
                classifier='Count whole ASCII words after lowercasing the first 128 stored tokens of each document. Technical if count_T >= 2 and count_T-count_N >= 2; narrative if count_N >= 2 and count_N-count_T >= 2; otherwise unassigned. Labels describe this lexical rule, not human annotations.',
                selection='Training uses only unconsumed blocks and excludes every reserved evaluation document. Four hundred eighty fully unconsumed documents are reserved, 160 per stratum technical/narrative/unassigned. Each contributes its first 65-token block. Evaluation is excluded from every fine-tuning arm.',
                seeds=dict(evaluation=915001, sources=915002), source_sha256=sha256(__file__))
    write_json(out/'selection.json', spec)
    start = time.perf_counter()
    tokens = np.load(corpus/'tokens.npy', mmap_mode='r')
    if tokens.shape != (524288, 513) or tokens.dtype != np.int32: raise ValueError('Unexpected corpus')
    expected = json.loads((corpus/'manifest.json').read_text())
    if sha256(corpus/'tokens.npy') != expected['tokens_sha256']: raise ValueError('Corpus hash mismatch')
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    tech, narrative = set(TECHNICAL), set(NARRATIVE)
    scores = np.zeros((len(tokens), 2), np.int16)
    for begin in range(0, len(tokens), 4096):
        strings = processor.decode(tokens[begin:begin+4096, :128].tolist())
        for j, s in enumerate(strings):
            words = re.findall(r'[a-z]+', s.lower())
            scores[begin+j] = [sum(w in tech for w in words), sum(w in narrative for w in words)]
        if begin % 65536 == 0: print('classified', begin, 'seconds', round(time.perf_counter()-start, 1), flush=True)
    labels = np.full(len(tokens), -1, np.int8)
    labels[(scores[:, 0] >= 2) & (scores[:, 0]-scores[:, 1] >= 2)] = 1
    labels[(scores[:, 1] >= 2) & (scores[:, 1]-scores[:, 0] >= 2)] = 0
    order = np.random.default_rng(641001).permutation(4194304)
    consumed = order[:1310720]; remaining = order[1310720:]
    untouched = np.ones(len(tokens), bool); untouched[consumed//8] = False
    rng = np.random.default_rng(spec['seeds']['evaluation'])
    evdocs = []; evlabels = []
    for label in [1, 0, -1]:
        candidates = np.flatnonzero(untouched & (labels == label))
        if len(candidates) < 160: raise ValueError('Insufficient unseen evaluation documents')
        selected = rng.choice(candidates, 160, replace=False)
        evdocs.extend(selected.tolist()); evlabels.extend([label]*160)
    evdocs = np.asarray(evdocs); evlabels = np.asarray(evlabels)
    reserved = np.zeros(len(tokens), bool); reserved[evdocs] = True
    remaining = remaining[~reserved[remaining//8]]
    pools = dict(technical=remaining[labels[remaining//8] == 1],
                 narrative=remaining[labels[remaining//8] == 0], general=remaining)
    for name, pool in pools.items():
        if len(pool) < 65536: raise ValueError('Insufficient training resource in '+name)
        np.save(out/(name+'-blocks.npy'), pool)
    np.savez_compressed(out/'lexical-labels.npz', labels=labels, scores=scores)
    np.savez_compressed(out/'evaluation.npz', crops=np.asarray(tokens[evdocs, :65]),
                        document_ids=evdocs, labels=evlabels,
                        split=np.tile(np.r_[np.zeros(80, dtype=np.int8), np.ones(80, dtype=np.int8)], 3))
    records = json.loads((corpus/'records.json').read_text())
    write_json(out/'evaluation-records.json', [dict(document_id=int(i), label=int(l), **records[i])
                                             for i, l in zip(evdocs, evlabels)])
    # Freeze source choices as part of data preparation, before a study exists.
    stream_rng = np.random.default_rng(915002)
    streams = {name:stream_rng.permutation(pools[name])[:32768]
               for name in ['technical', 'narrative', 'general']}
    np.savez_compressed(out/'streams.npz', **streams, uniforms=stream_rng.uniform(size=(1024, 32)))
    files = {p.name:sha256(p) for p in out.iterdir() if p.is_file()}
    report = dict(schema=spec['schema'], status='complete', seconds=time.perf_counter()-start,
                  pools={k:len(v) for k,v in pools.items()},
                  label_counts={str(k):int((labels == k).sum()) for k in [-1, 0, 1]},
                  untouched_documents=int(untouched.sum()), reserved_documents=len(evdocs),
                  input_files={str(corpus/n):sha256(corpus/n) for n in ['tokens.npy','manifest.json','records.json']},
                  tokenizer_sha256=sha256(tokenizer), files=files)
    write_json(out/'manifest.json', report); print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
