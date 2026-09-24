"""Post-hoc measurement of the separation-to-radius constant of the
jet reduction.  The concentration assumption posits anchor rows
rbar_a and a radius rho with r_a(x) = rbar_a + rho xi_a(x),
||xi_a|| <= 1, for the rows of LN(D_Q) indexed by a = (head, row)
within each layer; the glued directional estimate of the reduction
lemma additionally needs the anchors separated, sep > 2 rho + 2m, and
declares sep/rho a measurable validity constant.  This script measures
it on archived checkpoints:

  per layer: rbar_a = probe-input mean of the LN(D_Q) row at anchor a;
  rho_max   = max over anchors and probe inputs of ||r_a(x) - rbar_a||
              (the literal uniform bound of the assumption);
  rho_q95   = the 95th percentile of the same deviations (robust
              variant, reported alongside);
  sep       = min over distinct anchors of ||rbar_a - rbar_b||;
  ratio     = sep / rho_max  (and sep / rho_q95);
  sep_gt_2rho: whether sep > 2 rho_max (the m = 0 form of the
              separation hypothesis).

Probe inputs are drawn from the reserved probe chunks of the token
stream (the same reservation the training-time probes use), so the
measurement is on-distribution and disjoint from training batches.

Usage: python3 measure_separation.py [--device cuda:0]
           [--n_seq 128] [--batch 16] <run_name> [...]
Writes runs/<name>/separation.json (one entry per checkpoint).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_run import create_masks, PROBE_RESERVE_CHUNKS  # noqa: E402
from measure_tilt import build_model  # noqa: E402


@torch.no_grad()
def capture_ln_rows(model, x, mask):
    """LN(D_Q) per decoder layer for one batch: {layer: [B, h, dk, dk]}."""
    captured = {}
    hooks = []
    for li, dec in enumerate(model.decoder.dec_layers):
        def make_hook(li):
            def hook(module, inp, out):
                captured[li] = out.detach()
            return hook
        hooks.append(
            dec.mha1.layernorm1.register_forward_hook(make_hook(li)))
    model.eval()
    model([x, mask])
    for h in hooks:
        h.remove()
    return captured


@torch.no_grad()
def measure_ckpt(model, chunks, device, n_seq, batch):
    """Accumulate anchor rows over n_seq probe sequences and reduce to
    the per-layer separation/radius statistics."""
    rows = {}
    for lo in range(0, n_seq, batch):
        xb = torch.from_numpy(
            chunks[lo:lo + batch].astype(np.int64)).to(device)
        x = xb[:, :-1]
        mask = create_masks(x, device)
        cap = capture_ln_rows(model, x, mask)
        for li, A in cap.items():
            r = A.reshape(A.shape[0], -1, A.shape[-1]).float().cpu()
            rows.setdefault(li, []).append(r)
    out = {"layers": []}
    ratios, ratios_q, ok = [], [], []
    for li in sorted(rows):
        R = torch.cat(rows[li], dim=0)          # [B, A, dk]
        if not torch.isfinite(R).all():
            out["layers"].append({"layer": li, "finite": False})
            continue
        rbar = R.mean(dim=0)                    # [A, dk]
        dev = (R - rbar.unsqueeze(0)).norm(dim=-1)   # [B, A]
        rho_max = float(dev.max())
        rho_q95 = float(torch.quantile(dev.reshape(-1), 0.95))
        D = torch.cdist(rbar, rbar)             # [A, A]
        D.fill_diagonal_(float("inf"))
        sep = float(D.min())
        nn_med = float(D.min(dim=1).values.median())
        lay = {"layer": li, "finite": True,
               "n_anchors": int(rbar.shape[0]),
               "sep_min": sep, "nn_median": nn_med,
               "rho_max": rho_max, "rho_q95": rho_q95,
               "ratio": sep / rho_max if rho_max > 0 else None,
               "ratio_q95": sep / rho_q95 if rho_q95 > 0 else None,
               "sep_gt_2rho": bool(sep > 2 * rho_max)}
        out["layers"].append(lay)
        if lay["ratio"] is not None:
            ratios.append(lay["ratio"])
            ratios_q.append(lay["ratio_q95"])
            ok.append(lay["sep_gt_2rho"])
    out["ratio_min"] = min(ratios) if ratios else None
    out["ratio_min_q95"] = min(ratios_q) if ratios_q else None
    out["sep_gt_2rho"] = bool(ok and all(ok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tokens", default="data/refinedweb_tokens.npy")
    ap.add_argument("--n_seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = torch.device(args.device)
    toks = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    for name in args.runs:
        cfg = json.loads(open(f"runs/{name}/log.jsonl").readline())
        ctx = cfg["ctx"]
        chunks = toks[: len(toks) // ctx * ctx].reshape(-1, ctx)
        pb = chunks.shape[0] - PROBE_RESERVE_CHUNKS
        probe = chunks[pb:pb + args.n_seq]
        out = {"ckpts": {}, "n_seq": args.n_seq}
        for ckpt in sorted(glob.glob(f"runs/{name}/ckpt_*.pt")):
            model = build_model(cfg, device)
            ck = torch.load(ckpt, map_location=device)
            model.load_state_dict(ck["model"])
            out["ckpts"][os.path.basename(ckpt)] = measure_ckpt(
                model, probe, device, args.n_seq, args.batch)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        with open(f"runs/{name}/separation.json", "w") as f:
            json.dump(out, f, indent=1)
        print(name, "->", f"runs/{name}/separation.json")
        for k, v in out["ckpts"].items():
            print(f"  {k}: ratio_min={v['ratio_min']} "
                  f"ratio_min_q95={v['ratio_min_q95']} "
                  f"sep>2rho={v['sep_gt_2rho']}")


if __name__ == "__main__":
    main()
