#!/usr/bin/env python
"""Independently replay every selected document, rejection, crop and token."""
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
    a = p.parse_args()
    root = Path(a.root).resolve()
    study = root/a.study
    protocol = study/'protocols/expanded-data-selection.json'
    spec = json.loads(protocol.read_text())
    out = Path(spec['output'])
    destination = study/'verification/expanded-data.json'
    if destination.exists():
        raise FileExistsError(destination)
    checked = {}

    def check(path, signature=None):
        path = Path(path)
        actual = sha256(path)
        if signature is not None and signature != actual:
            raise AssertionError('Changed bound input: '+str(path))
        checked[str(path)] = actual

    meta = json.loads((out/'manifest.json').read_text())
    if meta['status'] != 'complete' or meta['training_documents'] != 307200:
        raise AssertionError('The selected larger corpus is incomplete')
    check(protocol, meta['protocol_sha256'])
    check(out/'manifest.json')
    check(out/'tokens.npy', meta['tokens_sha256'])
    check(out/'records.json', meta['records_sha256'])
    for path, signature in dict(spec['inputs'], **spec['source_files']).items():
        check(path, signature)
    expected = dict(training_documents=307200, documents_per_stratum=19200,
                    crop_length=513, sampling_offsets=449, minibatch=32,
                    row_seed=650821, crop_seed=1650821, permutation_seed=2650821)
    if any(spec[k] != v for k, v in expected.items()) or len(spec['shard_strata']) != 16:
        raise AssertionError('The frozen finite-data design changed')
    tokens = np.load(out/'tokens.npy', mmap_mode='r')
    records = json.loads((out/'records.json').read_text())
    if tokens.shape != (307200, 513) or tokens.dtype != np.int32 or len(records) != 307200:
        raise AssertionError('The data shape or arithmetic changed')
    occupied = {x['content_sha256'] for x in json.loads(Path(spec['exclude_records']).read_text())}
    initial_exclusions = len(occupied)
    positions = np.empty(307200, dtype=np.int64)
    positions[np.random.default_rng(2650821).permutation(307200)] = np.arange(307200)
    tokenizer = spm.SentencePieceProcessor(model_file=spec['tokenizer'])
    total, visited_total, rejected_short, rejected_duplicate = 0, 0, 0, 0
    started = time.time()
    strata = []
    for s, source in enumerate(spec['shard_strata']):
        path = Path(source['path'])
        check(path, source['sha256'])
        with pa.memory_map(str(path), 'r') as stream:
            table = pa.ipc.open_stream(stream).read_all()
        row_order = np.random.default_rng(650821+s).permutation(table.num_rows)
        crop_rng = np.random.default_rng(1650821+s)
        accepted, visited = 0, 0
        # A different tokenization batch boundary must reproduce the same law.
        for start in range(0, table.num_rows, 768):
            indices = row_order[start:start+768]
            contents = table.column('content').take(pa.array(indices)).to_pylist()
            encoded = tokenizer.encode(contents, num_threads=8)
            for row, content, ids in zip(indices, contents, encoded, strict=True):
                visited += 1
                digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
                if digest in occupied:
                    rejected_duplicate += 1
                    continue
                if len(ids) < 513:
                    rejected_short += 1
                    continue
                offset = int(crop_rng.integers(0, len(ids)-512))
                location = int(positions[total])
                record = dict(stratum=s, shard=path.name, row=int(row), content_sha256=digest,
                              document_tokens=len(ids), token_offset=offset, visit_position=visited-1)
                if records[location] != record:
                    raise AssertionError('Selected document, ordering, rejection or crop differs')
                expected_tokens = np.asarray(ids[offset:offset+513], dtype=np.int32)
                if tokens[location].tobytes() != expected_tokens.tobytes():
                    raise AssertionError('Stored tokens differ from the selected source document')
                occupied.add(digest)
                total += 1
                accepted += 1
                if accepted == 19200:
                    break
            if accepted == 19200:
                break
        if accepted != 19200:
            raise AssertionError('An incomplete stratum was retained')
        item = dict(stratum=s, selected=accepted, visited=visited, source_sha256=source['sha256'])
        if item != meta['shards'][s]:
            raise AssertionError('Recorded source traversal differs')
        strata.append(item)
        visited_total += visited
        print('verified', total, 'documents', round(time.time()-started, 1), 'seconds', flush=True)
        del table, contents, encoded
    if len(occupied) != initial_exclusions+307200:
        raise AssertionError('Document disjointness failed')
    write_json(destination, dict(schema='expanded-refinedweb-verification-v1', status='passed',
        completed_at=datetime.now(timezone.utc).isoformat(), checked_sha256=checked,
        documents=total, native_token_values_replayed_bytewise=307200*513,
        source_rows_visited=visited_total, excluded_document_hashes=initial_exclusions,
        rejected_short_documents=rejected_short, rejected_duplicate_documents=rejected_duplicate,
        strata=strata, verifier_sha256=sha256(__file__), seconds=time.time()-started,
        scope='Every visited source document and every accepted token crop was independently replayed from the read-only Arrow shards. The row permutations, exact-content exclusions, eligibility, uniform crop offsets and final permutation all agree. This verifies the declared finite selection; it does not establish semantic deduplication or absence of benchmark contamination in released pretrained models.'))
    print('passed', total, 'documents', flush=True)


if __name__ == '__main__':
    main()
