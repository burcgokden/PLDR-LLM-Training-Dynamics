"""Pre-tokenize a fixed contiguous token stream from tiiuae/falcon-refinedweb.

Matches the packing semantics of the reference PLDR-LLM data pipeline
(PLDR-LLM-Self-Organized-Criticality/src/pldr_data_prep.py): each document is
tokenized with the study's SentencePiece unigram tokenizer with add_eos=True
(no BOS), all documents are concatenated in dataset order, and training blocks
are contiguous MAX_LENGTH chunks of the stream ('pack' padding type).

Output: a single uint16 .npy memmap of token ids plus a JSON manifest.
"""

import argparse
import json
import os
import time

import numpy as np
import sentencepiece as spm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/refinedweb_tokens.npy")
    ap.add_argument("--tok_model", default="data/tokenizer.model")
    ap.add_argument("--n_tokens", type=int, default=220_000_000)
    ap.add_argument("--doc_batch", type=int, default=2048)
    args = ap.parse_args()

    import datasets

    sp = spm.SentencePieceProcessor(model_file=args.tok_model)
    eos = sp.eos_id()
    print(f"tokenizer: vocab={sp.vocab_size()} eos_id={eos} pad/unk_id={sp.unk_id()}")
    assert sp.vocab_size() < 65536

    ds = datasets.load_dataset(
        "tiiuae/falcon-refinedweb", split="train", streaming=True
    )

    buf = np.memmap(args.out, dtype=np.uint16, mode="w+", shape=(args.n_tokens,))
    pos = 0
    docs = 0
    t0 = time.time()
    batch = []
    for ex in ds:
        batch.append(ex["content"])
        if len(batch) >= args.doc_batch:
            ids_list = sp.encode(batch)
            batch = []
            for ids in ids_list:
                ids.append(eos)
                n = len(ids)
                if pos + n > args.n_tokens:
                    n = args.n_tokens - pos
                    ids = ids[:n]
                buf[pos : pos + n] = np.asarray(ids, dtype=np.uint16)
                pos += n
                docs += 1
                if pos >= args.n_tokens:
                    break
            el = time.time() - t0
            print(
                f"{pos/1e6:.1f}M tokens, {docs} docs, {el:.0f}s "
                f"({pos/max(el,1)/1e6:.2f} M tok/s)",
                flush=True,
            )
            if pos >= args.n_tokens:
                break

    buf.flush()
    manifest = {
        "dataset": "tiiuae/falcon-refinedweb",
        "split": "train (streaming, dataset order, from index 0)",
        "tokenizer": os.path.basename(args.tok_model),
        "encode": "add_bos=False, add_eos=True, docs concatenated in order",
        "n_tokens": int(pos),
        "n_docs": int(docs),
        "dtype": "uint16",
    }
    with open(args.out + ".manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("DONE", manifest)


if __name__ == "__main__":
    main()
