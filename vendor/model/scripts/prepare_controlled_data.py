#!/usr/bin/env python
"""Fresh document cohorts, read-only corpus, explicit exclusion and crop laws."""
from companion_paths import configured_path
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import sentencepiece as spm
from model_rg.provenance import sha256, source_manifest, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--corpus', default=configured_path('assets:refinedweb'))
    ap.add_argument('--kind', choices=['short', 'long'], required=True)
    a = ap.parse_args(); root = Path(a.root)
    out = root/'controlled-study-20260905'/'data'/a.kind
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((root/'data/refinedweb-4608/records.json').read_text())
    excluded = {r['content_sha256'] for r in original}
    prior_shards = {r['shard'] for r in original}
    other = out.parent/('short' if a.kind == 'long' else 'long')/'records.json'
    if other.exists():
        excluded.update(r['content_sha256'] for r in json.loads(other.read_text()))
    count, length, seed = (2048, 513, 630051) if a.kind == 'short' else (1024, 4097, 630052)
    rng = np.random.default_rng(seed)
    tokpath = root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model'
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokpath))
    paths = sorted(Path(a.corpus).rglob('falcon-refinedweb-train-*-of-05518.arrow'))
    if len(paths) != 5518: raise ValueError('Unexpected corpus shard inventory')
    selected = [rng.choice([p for p in stratum if p.name not in prior_shards])
                for stratum in np.array_split(np.array(paths, dtype=object), 16)]
    rows, records, shards = [], [], []; start = time.time()
    for path in selected:
        with pa.memory_map(str(path), 'r') as f:
            table = pa.ipc.open_stream(f).read_all()
        taken = 0
        for i in rng.permutation(table.num_rows):
            content = table.column('content')[int(i)].as_py()
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in excluded: continue
            ids = tokenizer.encode(content)
            if len(ids) < length: continue
            offset = int(rng.integers(len(ids)-length+1))
            rows.append(ids[offset:offset+length]); excluded.add(digest)
            records.append(dict(shard=path.name, row=int(i), content_sha256=digest,
                                token_offset=offset, document_tokens=len(ids)))
            taken += 1
            if taken == count//16: break
        if taken != count//16: raise RuntimeError('Insufficient qualifying documents')
        shards.append(dict(path=str(path), sha256=sha256(path), selected=taken))
        print(a.kind, len(rows), 'seconds', round(time.time()-start, 1), flush=True)
    # Stratify all analysis partitions by shard, avoiding a shard/split confound.
    order = np.concatenate([rng.permutation(np.arange(j*count//16, (j+1)*count//16))[:,None]
                            for j in range(16)], axis=1).ravel()
    np.save(out/'tokens.npy', np.asarray(rows, dtype=np.int32)[order])
    records = [records[i] for i in order]
    write_json(out/'records.json', records)
    offsets = np.random.default_rng(seed+1).integers(0, 449, size=count) if a.kind == 'short' else np.zeros(count, dtype=int)
    np.save(out/'offsets.npy', offsets)
    write_json(out/'manifest.json', dict(schema='controlled-cohort-v1', kind=a.kind,
        count=count, length=length, seed=seed, shards=shards,
        selection='One fresh shard per sixteen index strata; uniform shuffled qualifying documents; content-hash exclusions; uniform crop; no BOS/EOS',
        split_rows={'calibration':[0,512], 'evaluation':[512,2048]} if a.kind=='short' else
                   {'projection':[0,256], 'fit':[256,512], 'evaluation':[512,1024]},
        crop='short calls use saved uniform offset 0..448 inside a uniform 513-token crop; long calls use 64 adjacent 64-token prefixes',
        source_files=source_manifest(), inputs={str(root/'data/refinedweb-4608/records.json'):sha256(root/'data/refinedweb-4608/records.json')},
        tokens_sha256=sha256(out/'tokens.npy'), offsets_sha256=sha256(out/'offsets.npy'),
        records_sha256=sha256(out/'records.json'), tokenizer_sha256=sha256(tokpath), seconds=time.time()-start))


if __name__ == '__main__': main()
