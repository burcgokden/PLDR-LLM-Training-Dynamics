#!/usr/bin/env python3
"""Segmented-chain launcher for confirmation-program cells.

Runs a training cell as a chain of equal-length segments through the
FROZEN harness (train_run.py, untouched): segment 1 starts fresh,
every later segment continues via --init_from <prev>/ckpt_final.pt
--data_offset -1, so each seam leaves a with-optimizer checkpoint
(the branch/replay lattice the downstream experiments consume).  The
chain is exact: the model has no stochastic layers and the data order
is sequential, so a chained trajectory equals the monolithic one up
to floating-point kernel scheduling.

Segments are named <name>-seg<k> under experiments/runs/; the merged
view (global step = seam offset + local step) is produced by
confirm/collect.py, never by editing run logs.

Usage:
  run_chain.py --name c-e0-boundary --device cuda:0 --lr 7.5e-4 \
      --warmup 250 --steps 24000 --seg 2000 --seed 1234 \
      --probe_every 5 --sharp_block_every 5 [--extend]

--extend: continue an existing chain by appending segments up to
--steps (used by the predeclared 4k-block extension rules).
Schedule flags beyond --const_lr are passed through verbatim via
--extra (quoted string), for cells whose schedule the harness
expresses directly.
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)


def seg_dirs(runs, name):
    out = []
    k = 1
    while os.path.isdir(os.path.join(runs, f"{name}-seg{k}")):
        out.append(os.path.join(runs, f"{name}-seg{k}"))
        k += 1
    return out


def seg_done(d):
    return os.path.isfile(os.path.join(d, "ckpt_final.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--warmup", type=int, required=True,
                    help="warmup steps applied in segment 1 only")
    ap.add_argument("--steps", type=int, required=True,
                    help="total chain length in global steps")
    ap.add_argument("--seg", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--probe_every", type=int, default=5)
    ap.add_argument("--sharp_block_every", type=int, default=5)
    ap.add_argument("--sharp_every", type=int, default=250)
    ap.add_argument("--sharp_pre_every", type=int, default=1000)
    ap.add_argument("--extra", default="",
                    help="extra train_run.py flags, quoted")
    ap.add_argument("--extend", action="store_true")
    ap.add_argument("--outdir", default=os.path.join(EXP, "runs"))
    args = ap.parse_args()

    runs = args.outdir
    assert args.steps % args.seg == 0, "steps must be a multiple of seg"
    nseg = args.steps // args.seg

    existing = seg_dirs(runs, args.name)
    for d in existing[:-1]:
        assert seg_done(d), f"incomplete non-final segment {d}"
    start = len(existing)
    if start and not seg_done(existing[-1]):
        # resume policy: an interrupted final segment is re-run whole
        # (its directory is renamed aside, never overwritten silently)
        bad = existing[-1]
        os.rename(bad, bad + ".interrupted")
        start -= 1
    if start >= nseg and not args.extend:
        print(f"{args.name}: chain complete ({start} segments)")
        return

    for k in range(start + 1, nseg + 1):
        seg_name = f"{args.name}-seg{k}"
        cmd = [sys.executable, os.path.join(EXP, "train_run.py"),
               "--name", seg_name,
               "--lr", str(args.lr),
               "--steps", str(args.seg),
               "--const_lr",
               "--seed", str(args.seed),
               "--device", args.device,
               "--probe_every", str(args.probe_every),
               "--sharp_block_every", str(args.sharp_block_every),
               "--sharp_every", str(args.sharp_every),
               "--sharp_pre_every", str(args.sharp_pre_every),
               "--ckpt_steps", "",
               "--outdir", runs]
        if k == 1:
            cmd += ["--warmup", str(args.warmup)]
        else:
            prev = os.path.join(runs, f"{args.name}-seg{k-1}",
                                "ckpt_final.pt")
            cmd += ["--warmup", "1",
                    "--init_from", prev,
                    "--data_offset", "-1"]
        if args.extra:
            cmd += args.extra.split()
        meta = {"chain": args.name, "segment": k, "seg_len": args.seg,
                "global_step_offset": (k - 1) * args.seg,
                "cmd": cmd}
        print(f"[chain {args.name}] segment {k}/{nseg}", flush=True)
        r = subprocess.run(cmd, cwd=EXP)
        if r.returncode != 0:
            print(f"[chain {args.name}] segment {k} FAILED "
                  f"(rc {r.returncode})", flush=True)
            sys.exit(r.returncode)
        with open(os.path.join(runs, seg_name, "chain_meta.json"),
                  "w") as f:
            json.dump(meta, f, indent=1)
    print(f"[chain {args.name}] complete: {nseg} segments x "
          f"{args.seg} steps", flush=True)


if __name__ == "__main__":
    main()
