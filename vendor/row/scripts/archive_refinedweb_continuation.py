#!/usr/bin/env python3
"""Build the prefix-locked RefinedWeb archive for the 65,536-step study.

The predecessor archive ended after 220,000,000 tokens.  Its final 5,120
chunks were reserved for measurements.  This generator keeps the predecessor
training region byte-for-byte at the front, fills the enlarged training region
from the exact continuation of the same packed document stream, and relocates
the predecessor reserve intact to the new tail.  Thus no training chunk is
reused, the old training permutations retain their meaning, and the registered
measurement contexts remain unchanged.
"""

from __future__ import annotations
from companion_paths import configured_path

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import sentencepiece as spm


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
DEFAULT_ARROW_ROOT = Path(
    configured_path('assets:refinedweb/datasets/huggingface_datasets/tiiuae___falcon-refinedweb/default/0.0.0/c735840575b629292b41da8dde11dcd523d4f91c')
)
DEFAULT_PREDECESSOR = (
    PROJECTS / "experiment-data" / "shared" / "datasets"
    / "refinedweb-100m-tokens" / "refinedweb_tokens.npy"
)
DEFAULT_TOKENIZER = ROOT / "experiments" / "data" / "tokenizer.model"
DEFAULT_OUTPUT = (
    PROJECTS / "experiment-data" / "shared" / "datasets"
    / "refinedweb-538m-prefix-locked" / "refinedweb_tokens.npy"
)

CONTEXT_LENGTH = 256
BATCH_SIZE = 32
TERMINAL_UPDATE = 65_536
RESERVE_CHUNKS = 5_120
PREDECESSOR_TOKENS = 220_000_000
PREDECESSOR_CHUNKS = PREDECESSOR_TOKENS // CONTEXT_LENGTH
PREDECESSOR_TRAINING_CHUNKS = PREDECESSOR_CHUNKS - RESERVE_CHUNKS
PREDECESSOR_TRAINING_TOKENS = PREDECESSOR_TRAINING_CHUNKS * CONTEXT_LENGTH
TRAINING_CHUNKS = TERMINAL_UPDATE * BATCH_SIZE
OUTPUT_CHUNKS = TRAINING_CHUNKS + RESERVE_CHUNKS
OUTPUT_TOKENS = OUTPUT_CHUNKS * CONTEXT_LENGTH
CONTINUATION_TOKENS = (
    TRAINING_CHUNKS - PREDECESSOR_TRAINING_CHUNKS
) * CONTEXT_LENGTH
SOURCE_STOP_TOKEN = PREDECESSOR_TOKENS + CONTINUATION_TOKENS


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def copy_tokens(
    source: np.memmap,
    destination: np.memmap,
    *,
    source_start: int,
    destination_start: int,
    count: int,
    block_tokens: int = 8 * 1024 * 1024,
) -> None:
    for offset in range(0, count, block_tokens):
        width = min(block_tokens, count - offset)
        destination[
            destination_start + offset:destination_start + offset + width
        ] = source[source_start + offset:source_start + offset + width]


def equal_tokens(
    left: np.memmap,
    right: np.memmap,
    *,
    left_start: int,
    right_start: int,
    count: int,
    block_tokens: int = 8 * 1024 * 1024,
) -> bool:
    for offset in range(0, count, block_tokens):
        width = min(block_tokens, count - offset)
        if not np.array_equal(
            left[left_start + offset:left_start + offset + width],
            right[right_start + offset:right_start + offset + width],
        ):
            return False
    return True


def source_shards(arrow_root: Path) -> list[Path]:
    paths = sorted(arrow_root.glob("falcon-refinedweb-train-*-of-05518.arrow"))
    if len(paths) != 5_518:
        raise ValueError(
            f"expected 5,518 RefinedWeb train shards, found {len(paths)}"
        )
    return paths


def build(
    output: Path,
    predecessor: Path,
    tokenizer: Path,
    arrow_root: Path,
    *,
    document_batch: int = 2_048,
) -> dict[str, Any]:
    """Create the archive once and return its complete provenance manifest."""

    from datasets import Dataset

    for path in (predecessor, tokenizer, arrow_root / "dataset_info.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if predecessor.stat().st_size != PREDECESSOR_TOKENS * 2:
        raise ValueError("predecessor archive is not the registered 220M tokens")
    if document_batch <= 0:
        raise ValueError("document batch must be positive")
    manifest_path = Path(str(output) + ".manifest.json")
    partial = Path(str(output) + ".partial")
    manifest_partial = Path(str(manifest_path) + ".partial")
    for path in (output, manifest_path, partial, manifest_partial):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite archive artifact: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)

    predecessor_data = np.memmap(predecessor, dtype=np.uint16, mode="r")
    destination = np.memmap(
        partial, dtype=np.uint16, mode="w+", shape=(OUTPUT_TOKENS,)
    )
    copy_tokens(
        predecessor_data,
        destination,
        source_start=0,
        destination_start=0,
        count=PREDECESSOR_TRAINING_TOKENS,
    )
    reserve_destination = TRAINING_CHUNKS * CONTEXT_LENGTH
    copy_tokens(
        predecessor_data,
        destination,
        source_start=PREDECESSOR_TRAINING_TOKENS,
        destination_start=reserve_destination,
        count=RESERVE_CHUNKS * CONTEXT_LENGTH,
    )

    processor = spm.SentencePieceProcessor(model_file=str(tokenizer))
    if processor.vocab_size() >= np.iinfo(np.uint16).max + 1:
        raise ValueError("tokenizer vocabulary does not fit uint16")
    eos = processor.eos_id()
    if eos < 0:
        raise ValueError("tokenizer has no EOS token")

    source_position = 0
    document_count = 0
    used: list[dict[str, Any]] = []
    next_report = 25_000_000
    started = time.monotonic()
    for shard_index, shard in enumerate(source_shards(arrow_root)):
        dataset = Dataset.from_file(str(shard), in_memory=False).select_columns(
            ["content"]
        )
        rows_used = 0
        for start in range(0, len(dataset), document_batch):
            contents = dataset[start:min(start + document_batch, len(dataset))][
                "content"
            ]
            for ids in processor.encode(contents):
                values = np.asarray([*ids, eos], dtype=np.uint16)
                end = source_position + len(values)
                if source_position < PREDECESSOR_TOKENS:
                    check_end = min(end, PREDECESSOR_TOKENS)
                    width = check_end - source_position
                    if not np.array_equal(
                        values[:width],
                        predecessor_data[source_position:check_end],
                    ):
                        raise ValueError(
                            "local RefinedWeb stream diverges from predecessor "
                            f"at or before token {source_position}"
                        )
                continuation_left = max(source_position, PREDECESSOR_TOKENS)
                continuation_right = min(end, SOURCE_STOP_TOKEN)
                if continuation_left < continuation_right:
                    value_start = continuation_left - source_position
                    value_end = continuation_right - source_position
                    destination_start = (
                        PREDECESSOR_TRAINING_TOKENS
                        + continuation_left - PREDECESSOR_TOKENS
                    )
                    destination[
                        destination_start:
                        destination_start + value_end - value_start
                    ] = values[value_start:value_end]
                source_position = end
                document_count += 1
                rows_used += 1
                if source_position >= next_report:
                    elapsed = max(time.monotonic() - started, 1.0e-9)
                    print(
                        f"{min(source_position, SOURCE_STOP_TOKEN) / 1e6:.1f}M "
                        f"source tokens, {document_count} documents, "
                        f"{min(source_position, SOURCE_STOP_TOKEN) / elapsed / 1e6:.2f} "
                        "M token/s",
                        flush=True,
                    )
                    next_report += 25_000_000
                if source_position >= SOURCE_STOP_TOKEN:
                    break
            if source_position >= SOURCE_STOP_TOKEN:
                break
        used.append({
            "index": shard_index,
            "name": shard.name,
            "rows_used": rows_used,
            "file_size_bytes": shard.stat().st_size,
            "sha256": sha256_path(shard),
        })
        if source_position >= SOURCE_STOP_TOKEN:
            break
    if source_position < SOURCE_STOP_TOKEN:
        raise RuntimeError("local shards ended before the required source token")
    destination.flush()
    del destination
    os.replace(partial, output)

    manifest = {
        "schema_version": "pldr-refinedweb-prefix-locked-archive-v1",
        "dataset": "tiiuae/falcon-refinedweb",
        "split": "train",
        "packing": "document order; add_bos=False; add_eos=True; concatenate",
        "dtype": "uint16-raw-memmap",
        "context_length": CONTEXT_LENGTH,
        "batch_size": BATCH_SIZE,
        "terminal_update": TERMINAL_UPDATE,
        "output_tokens": OUTPUT_TOKENS,
        "output_chunks": OUTPUT_CHUNKS,
        "training_chunks": TRAINING_CHUNKS,
        "reserve_chunks": RESERVE_CHUNKS,
        "source_stop_token": SOURCE_STOP_TOKEN,
        "document_count_through_source_stop": document_count,
        "predecessor": {
            "path_at_build": str(predecessor.resolve()),
            "tokens": PREDECESSOR_TOKENS,
            "training_chunks": PREDECESSOR_TRAINING_CHUNKS,
            "sha256": sha256_path(predecessor),
        },
        "layout": [
            {
                "output_token_start": 0,
                "output_token_stop": PREDECESSOR_TRAINING_TOKENS,
                "source_token_start": 0,
                "source_token_stop": PREDECESSOR_TRAINING_TOKENS,
                "role": "prefix-locked-training",
            },
            {
                "output_token_start": PREDECESSOR_TRAINING_TOKENS,
                "output_token_stop": TRAINING_CHUNKS * CONTEXT_LENGTH,
                "source_token_start": PREDECESSOR_TOKENS,
                "source_token_stop": SOURCE_STOP_TOKEN,
                "role": "disjoint-training-continuation",
            },
            {
                "output_token_start": TRAINING_CHUNKS * CONTEXT_LENGTH,
                "output_token_stop": OUTPUT_TOKENS,
                "source_token_start": PREDECESSOR_TRAINING_TOKENS,
                "source_token_stop": PREDECESSOR_TOKENS,
                "role": "relocated-predecessor-reserve",
            },
        ],
        "tokenizer_sha256": sha256_path(tokenizer),
        "cache_dataset_info_sha256": sha256_path(
            arrow_root / "dataset_info.json"
        ),
        "cache_root_at_build": str(arrow_root.resolve()),
        "source_shards": used,
        "output_sha256": sha256_path(output),
        "globally_unique_training_chunk_indices": True,
        "predecessor_probe_rows_preserved": True,
    }
    manifest_partial.write_bytes(canonical(manifest))
    os.replace(manifest_partial, manifest_path)
    return manifest


def check(output: Path, predecessor: Path) -> dict[str, Any]:
    manifest_path = Path(str(output) + ".manifest.json")
    if not output.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("archive or manifest is absent")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "pldr-refinedweb-prefix-locked-archive-v1"
        or manifest.get("output_tokens") != OUTPUT_TOKENS
        or output.stat().st_size != OUTPUT_TOKENS * 2
        or manifest.get("output_sha256") != sha256_path(output)
        or manifest.get("predecessor", {}).get("sha256")
        != sha256_path(predecessor)
    ):
        raise ValueError("archive manifest or digest does not replay")
    archive = np.memmap(output, dtype=np.uint16, mode="r")
    old = np.memmap(predecessor, dtype=np.uint16, mode="r")
    if not equal_tokens(
        archive,
        old,
        left_start=0,
        right_start=0,
        count=PREDECESSOR_TRAINING_TOKENS,
    ) or not equal_tokens(
        archive,
        old,
        left_start=TRAINING_CHUNKS * CONTEXT_LENGTH,
        right_start=PREDECESSOR_TRAINING_TOKENS,
        count=RESERVE_CHUNKS * CONTEXT_LENGTH,
    ):
        raise ValueError("prefix or relocated reserve differs from predecessor")
    print(
        "RefinedWeb archive: verified "
        f"({TRAINING_CHUNKS:,} unique training chunks, "
        f"{RESERVE_CHUNKS:,} preserved reserve chunks)"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predecessor", type=Path, default=DEFAULT_PREDECESSOR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--arrow-root", type=Path, default=DEFAULT_ARROW_ROOT)
    parser.add_argument("--document-batch", type=int, default=2_048)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        check(arguments.output.resolve(), arguments.predecessor.resolve())
    else:
        result = build(
            arguments.output.resolve(),
            arguments.predecessor.resolve(),
            arguments.tokenizer.resolve(),
            arguments.arrow_root.resolve(),
            document_batch=arguments.document_batch,
        )
        print(json.dumps({
            "output": str(arguments.output.resolve()),
            "output_sha256": result["output_sha256"],
            "source_shards": len(result["source_shards"]),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
