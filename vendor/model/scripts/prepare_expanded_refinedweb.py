#!/usr/bin/env python
"""Freeze and construct a larger, disjoint finite RefinedWeb training law."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908')
    p.add_argument('--prepare', action='store_true')
    a = p.parse_args()
    root = Path(a.root).resolve()
    study = root/a.study
    protocol = study/'protocols/expanded-data-selection.json'
    repo = Path(__file__).resolve().parents[1]
    if a.prepare:
        if protocol.exists():
            raise FileExistsError(protocol)
        old = root/'data/refinedweb-4608'
        previous = json.loads((old/'manifest.json').read_text())
        paths = [old/'manifest.json', old/'records.json',
                 root/'assets/PLDR-LLM-v51-SOC-110M-1/tokenizer.model',
                 root/'scheduled-training-20260908/development/early-training-risk/verification.json',
                 root/'scheduled-training-20260908/development/feasibility-redesign/decision.json']
        spec = dict(schema='expanded-refinedweb-selection-v1',
            frozen_at=datetime.now(timezone.utc).isoformat(),
            output=str(root/'data/refinedweb-expanded-307200'),
            training_documents=307200, documents_per_stratum=19200,
            crop_length=513, sampling_offsets=449, minibatch=32,
            shard_strata=previous['shards'], row_seed=650821,
            crop_seed=1650821, permutation_seed=2650821,
            tokenizer=str(paths[2]), exclude_records=str(old/'records.json'),
            source_files={str(repo/'scripts/prepare_expanded_refinedweb.py'):
                          sha256(repo/'scripts/prepare_expanded_refinedweb.py'),
                          str(repo/'src/model_rg/provenance.py'):sha256(repo/'src/model_rg/provenance.py')},
            inputs={str(x):sha256(x) for x in paths},
            tokenization_threads=8, tokenization_batch=1024,
            distribution='Use the sixteen already selected RefinedWeb shard strata. Within each shard, visit an independently seeded uniform row permutation and retain the first 19200 distinct documents having at least 513 SentencePiece tokens, excluding every content hash in the 4608-document corpus. Draw an independent uniform 513-token crop per accepted document, then uniformly permute all accepted crops. No BOS or EOS is inserted. Training samples a uniform row and one of its 449 length-65 crops with replacement.',
            conditioning='This is a larger finite training law conditioned on the same shard set and tokenizer. The common evaluation documents remain disjoint. Document deduplication uses exact content hashes, not semantic similarity. Finite-corpus size and schedule duration are separately controlled; this does not recreate the reference eight-billion-token pretraining exposure.',
            known_information='The six verified early risk diagnostics and all completed constant-rate outcomes were available when this larger-data control was chosen. No outcome of this new training law has been observed.')
        protocol.parent.mkdir(parents=True, exist_ok=True)
        write_json(protocol, spec)
        print(json.dumps(dict(protocol=str(protocol), sha256=sha256(protocol))), flush=True)
        return
    spec = json.loads(protocol.read_text())
    for path, signature in dict(spec['inputs'], **spec['source_files']).items():
        if sha256(path) != signature:
            raise AssertionError('A selected data input or producer changed: '+path)
    out = Path(spec['output'])
    out.mkdir(parents=True, exist_ok=False)
    seen = {r['content_sha256'] for r in json.loads(Path(spec['exclude_records']).read_text())}
    excluded_count = len(seen)
    tok = spm.SentencePieceProcessor(model_file=spec['tokenizer'])
    sequences = np.empty((spec['training_documents'], 513), dtype=np.int32)
    records, shard_records = [], []
    started = time.time()
    for s, entry in enumerate(spec['shard_strata']):
        path = Path(entry['path'])
        if sha256(path) != entry['sha256']:
            raise AssertionError('A read-only source shard changed')
        with pa.memory_map(str(path), 'r') as mapped:
            table = pa.ipc.open_stream(mapped).read_all()
        order = np.random.default_rng(spec['row_seed']+s).permutation(table.num_rows)
        crop_rng = np.random.default_rng(spec['crop_seed']+s)
        taken, visited = 0, 0
        for start in range(0, len(order), spec['tokenization_batch']):
            indices = order[start:start+spec['tokenization_batch']]
            contents = table.column('content').take(pa.array(indices)).to_pylist()
            encoded = tok.encode(contents, num_threads=spec['tokenization_threads'])
            for row, content, ids in zip(indices, contents, encoded, strict=True):
                visited += 1
                digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
                if digest in seen or len(ids) < 513:
                    continue
                offset = int(crop_rng.integers(0, len(ids)-512))
                index = len(records)
                sequences[index] = ids[offset:offset+513]
                records.append(dict(stratum=s, shard=path.name, row=int(row),
                                    content_sha256=digest, document_tokens=len(ids),
                                    token_offset=offset, visit_position=visited-1))
                seen.add(digest)
                taken += 1
                if taken == spec['documents_per_stratum']:
                    break
            if taken == spec['documents_per_stratum']:
                break
        if taken != spec['documents_per_stratum']:
            raise AssertionError('A selected shard has insufficient eligible unique documents')
        shard_records.append(dict(stratum=s, selected=taken, visited=visited,
                                  source_sha256=entry['sha256']))
        write_json(out/'progress.json', dict(status='constructing', shards=shard_records,
                   documents=len(records), seconds=time.time()-started))
        print(path.name, len(records), 'documents', round(time.time()-started, 1), 'seconds', flush=True)
        del table, contents, encoded
    order = np.random.default_rng(spec['permutation_seed']).permutation(len(records))
    np.save(out/'tokens.npy', sequences[order])
    write_json(out/'records.json', [records[i] for i in order])
    write_json(out/'manifest.json', dict(schema='expanded-refinedweb-corpus-v1', status='complete',
        protocol=str(protocol), protocol_sha256=sha256(protocol),
        training_documents=len(records), excluded_document_hashes=excluded_count,
        shape=[len(records),513], dtype='int32', shards=shard_records,
        tokenizer_sha256=sha256(spec['tokenizer']), tokens_sha256=sha256(out/'tokens.npy'),
        records_sha256=sha256(out/'records.json'), source_files=spec['source_files'],
        seconds=time.time()-started, scope=spec['conditioning']))
    print('complete', len(records), 'documents', flush=True)


if __name__ == '__main__':
    main()
