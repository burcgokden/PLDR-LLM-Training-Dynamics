#!/usr/bin/env python
"""Draw two finite RefinedWeb corpora and document-excluded observation panels.

All writes are inside --study. The supplied RefinedWeb Arrow shards are read-only.
Each corpus is an independent seeded row permutation per declared shard stratum,
with length eligibility and exact-content deduplication within that draw.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import sentencepiece as spm

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', required=True); a=p.parse_args()
    root=Path(a.root).resolve(); study=Path(a.study).resolve()
    out=study/'data'; out.mkdir(parents=True,exist_ok=False)
    start=time.time()
    source=root/'data/refinedweb-4608'
    old=json.loads((source/'manifest.json').read_text())
    old_records=json.loads((source/'records.json').read_text())
    previous=json.loads((root/'data/refinedweb-onepass-524288/records.json').read_text())
    excluded={r['content_sha256'] for r in old_records}
    known={r['content_sha256'] for r in previous}|excluded
    tokenizer=root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'
    tok=spm.SentencePieceProcessor(model_file=str(tokenizer))
    spec=dict(schema='outer-transfer-data-law-v1', frozen_at=datetime.now(timezone.utc).isoformat(),
        documents_per_corpus=65536, documents_per_stratum=4096, corpus_seeds=[9101101,9101201],
        evaluation_seed=9101001, evaluation_documents=32, operator_donors=32,
        crop_length=513, context_length=64, blocks_per_document=8, batch_size=32,
        strata=old['shards'], tokenizer_sha256=sha256(tokenizer),
        selection='Uniform seeded row permutations in each of sixteen fixed shard strata. Retain eligible documents with at least 513 tokens, deduplicate exact content within each corpus, and exclude the old 4608 documents plus both new observation panels. Crop offsets are independently uniform. The two corpus draws may overlap; they are not forced disjoint. Evaluation and operator-donor documents are also excluded from the existing 524288-document corpus. No BOS/EOS is added.',
        sources={str(Path(__file__).resolve()):sha256(__file__)},
        inputs={str(source/'manifest.json'):sha256(source/'manifest.json'),
                str(source/'records.json'):sha256(source/'records.json'),
                str(root/'data/refinedweb-onepass-524288/records.json'):sha256(root/'data/refinedweb-onepass-524288/records.json')})
    write_json(out/'protocol.json', spec)
    corpora=[[],[]]; records=[[],[]]; panel=[]; panel_records=[]
    seen=[set(excluded),set(excluded)]; seen_eval=set(known)
    def select(table, s, seed, count, forbidden):
        order=np.random.default_rng(seed+s).permutation(table.num_rows)
        rng=np.random.default_rng(seed+100000+s)
        xs=[]; rs=[]
        for start in range(0,len(order),1024):
            ids=order[start:start+1024]
            text=table.column('content').take(pa.array(ids)).to_pylist()
            enc=tok.encode(text,num_threads=8)
            for row,content,tokens in zip(ids,text,enc,strict=True):
                digest=hashlib.sha256(content.encode()).hexdigest()
                if digest in forbidden or len(tokens)<513: continue
                offset=int(rng.integers(len(tokens)-512))
                xs.append(tokens[offset:offset+513])
                rs.append(dict(stratum=s,shard=Path(spec['strata'][s]['path']).name,row=int(row),
                    content_sha256=digest,token_offset=offset,document_tokens=len(tokens)))
                forbidden.add(digest)
                if len(xs)==count: return xs,rs
        raise RuntimeError('Insufficient eligible source documents')
    for phase in ['panels','corpora']:
        if phase=='corpora':
            for c in range(2):seen[c].update(r['content_sha256'] for r in panel_records)
        for s,entry in enumerate(spec['strata']):
            path=Path(entry['path']); assert sha256(path)==entry['sha256']
            with pa.memory_map(str(path),'r') as mm: table=pa.ipc.open_stream(mm).read_all()
            if phase=='panels':
                xs,rs=select(table,s,spec['evaluation_seed'],4,seen_eval)
                panel.extend(xs); panel_records.extend(rs)
            else:
                for c in range(2):
                    xs,rs=select(table,s,spec['corpus_seeds'][c],4096,seen[c])
                    corpora[c].extend(xs); records[c].extend(rs)
            print(phase,'stratum',s,'elapsed',round(time.time()-start,1),flush=True)
    panel_hashes={r['content_sha256'] for r in panel_records}
    for rows in records:
        assert not panel_hashes.intersection(r['content_sha256'] for r in rows)
    panel=np.array(panel,dtype=np.int32)
    evaluation=np.array([4*s+j for s in range(16) for j in [0,1]])
    donors=np.array([4*s+j for s in range(16) for j in [2,3]])
    np.savez_compressed(out/'panels.npz',evaluation=panel[evaluation,:65],donors=panel[donors,:65])
    write_json(out/'panel-records.json',dict(evaluation=[panel_records[i] for i in evaluation],donors=[panel_records[i] for i in donors]))
    for c in range(2):
        order=np.random.default_rng(spec['corpus_seeds'][c]+200000).permutation(65536)
        np.save(out/f'corpus-{c}.npy',np.array(corpora[c],dtype=np.int32)[order])
        write_json(out/f'records-{c}.json',[records[c][i] for i in order])
    sets=[{r['content_sha256'] for r in rows} for rows in records]
    write_json(out/'manifest.json',dict(status='complete',protocol_sha256=sha256(out/'protocol.json'),
        files={p.name:sha256(p) for p in out.iterdir() if p.is_file()},
        documents_per_corpus=65536, cross_corpus_overlap=len(sets[0]&sets[1]),
        overlap_with_existing=[len(x&known) for x in sets], evaluation_document_overlap=0,
        seconds=time.time()-start))
    print('complete',round(time.time()-start,1),'seconds',flush=True)


if __name__=='__main__': main()
