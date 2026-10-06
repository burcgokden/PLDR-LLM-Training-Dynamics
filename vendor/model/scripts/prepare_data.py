#!/usr/bin/env python
"""Stratified, document-disjoint RefinedWeb contexts from read-only Arrow files."""
from companion_paths import configured_path
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import sentencepiece as spm
from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default=configured_path('assets:refinedweb'))
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--documents", type=int, default=4608)
    p.add_argument("--seed", type=int, default=20260905)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)
    paths = sorted(Path(args.dataset_root).rglob("falcon-refinedweb-train-*-of-05518.arrow"))
    # Evenly spread shard strata; random choices within each stratum and record batch.
    strata = np.array_split(np.arange(len(paths)), 16)
    selected = [paths[int(rng.choice(s))] for s in strata]
    sequences, records, shards = [], [], []
    seen = set()
    start = time.time()
    per = (args.documents + 15) // 16
    for path in selected:
        with pa.memory_map(str(path), "r") as f:
            table = pa.ipc.open_stream(f).read_all()
        order = rng.permutation(table.num_rows)
        taken = 0
        for i in order:
            content = table.column("content")[int(i)].as_py()
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in seen:
                continue
            ids = tokenizer.encode(content)
            if len(ids) < 513:
                continue
            start_token = int(rng.integers(0, len(ids) - 512))
            sequences.append(ids[start_token:start_token + 513])
            records.append({"shard": path.name, "row": int(i), "content_sha256": digest,
                            "token_offset": start_token, "document_tokens": len(ids)})
            seen.add(digest)
            taken += 1
            if taken >= per or len(sequences) >= args.documents:
                break
        shards.append({"path": str(path), "sha256": sha256(path), "rows": table.num_rows, "selected": taken})
        print(f"{path.name}: {taken} documents; total {len(sequences)}", flush=True)
    if len(sequences) != args.documents:
        raise RuntimeError("Not enough qualifying documents")
    perm = rng.permutation(len(sequences))
    array = np.asarray(sequences, dtype=np.int32)[perm]
    records = [records[i] for i in perm]
    np.save(out / "tokens.npy", array)
    write_json(out / "records.json", records)
    write_json(out / "manifest.json", {"schema": "model-rg-data-v1", "seed": args.seed,
        "selection": "16 evenly spread shard strata; one random shard per stratum; uniformly shuffled rows; unique documents with >=513 SentencePiece tokens; uniform 513-token crop; no BOS/EOS",
        "calibration_documents": 512, "confirmation_documents": args.documents - 512,
        "tokenizer_sha256": sha256(args.tokenizer), "tokens_sha256": sha256(out / "tokens.npy"),
        "records_sha256": sha256(out / "records.json"), "shards": shards, "seconds": time.time()-start})


if __name__ == "__main__":
    main()
