#!/usr/bin/env python3
"""Merge a segmented confirmation chain into one run directory.

Produces runs/<name>-merged/ containing:
  log.jsonl        config record (steps = chain total), every per-step
                   record reindexed to GLOBAL step (seam offset + local
                   step), the last segment's final record
  chain_index.json seam positions and segment directories
  ckpt_<step>.pt   symlinks to each segment's with-optimizer
                   ckpt_final.pt (the branch/replay lattice)

The merged directory is consumable by the FROZEN analysis stack
(gates.load_run, gnorm_series, cell_events, ...) exactly like a
monolithic run.  Segment-local extra probes (the harness's step==1
probe and step==50 sharpness records) are kept: they are additional
measurements at their global steps, and every downstream estimator
selects records by key presence, never by step modulus.

Usage: collect.py <chain-name> [...]  (re-runs are idempotent)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RUNS = os.path.join(EXP, "runs")


def merge_chain(name, runs=RUNS):
    segs = []
    k = 1
    while os.path.isdir(os.path.join(runs, f"{name}-seg{k}")):
        segs.append(os.path.join(runs, f"{name}-seg{k}"))
        k += 1
    if not segs:
        raise SystemExit(f"{name}: no segments found under {runs}")
    outdir = os.path.join(runs, f"{name}-merged")
    os.makedirs(outdir, exist_ok=True)

    config = None
    final = None
    seams = []
    offset = 0
    out = open(os.path.join(outdir, "log.jsonl"), "w")
    try:
        for i, seg in enumerate(segs):
            seg_len = None
            with open(os.path.join(seg, "log.jsonl")) as f:
                for line in f:
                    d = json.loads(line)
                    if d.get("event") == "config":
                        seg_len = d["steps"]
                        if i == 0:
                            config = dict(d)
                        continue
                    if d.get("event") == "final":
                        final = d
                        continue
                    if "step" in d:
                        d["step"] = offset + d["step"]
                        out.write(json.dumps(d) + "\n")
            assert seg_len is not None, f"no config record in {seg}"
            offset += seg_len
            seams.append(offset)
            ck = os.path.join(seg, "ckpt_final.pt")
            if os.path.isfile(ck):
                dst = os.path.join(outdir, f"ckpt_{offset}.pt")
                if not os.path.islink(dst) and not os.path.isfile(dst):
                    os.symlink(os.path.relpath(ck, outdir), dst)
        config["steps"] = offset
        config["name"] = f"{name}-merged"
        config["chain_segments"] = len(segs)
        # rewrite with config first: reopen and prepend
    finally:
        out.close()
    # prepend config line (single rewrite; logs are tens of MB, fine)
    body = open(os.path.join(outdir, "log.jsonl")).read()
    with open(os.path.join(outdir, "log.jsonl"), "w") as f:
        f.write(json.dumps(config) + "\n")
        f.write(body)
        if final is not None:
            f.write(json.dumps(final) + "\n")
    with open(os.path.join(outdir, "chain_index.json"), "w") as f:
        json.dump({"chain": name, "segments": [os.path.basename(s)
                                               for s in segs],
                   "seams": seams, "total_steps": offset}, f, indent=1)
    print(f"{name}: merged {len(segs)} segments, {offset} steps "
          f"-> {outdir}")
    return outdir


if __name__ == "__main__":
    for nm in sys.argv[1:]:
        merge_chain(nm)
