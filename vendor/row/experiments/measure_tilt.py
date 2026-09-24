"""Post-hoc measurement of the tilt proxy (exploitation-covariance proxy).

The paper's exploitation tensor is C_a = Cov(g_a, xi_a) with g_a the
per-sample loss adjoint into the generator rows A_a evaluated AT THE
COLLAPSED POINT (frozen batch-mean generator) and xi_a the input fluctuation
of the corresponding LN(D_Q) row.  This script measures two objects:

- mode "current"  : the CURRENT-STATE tilt proxy of the original protocol
                    (adjoints at the current model), a low-rank proxy;
- mode "collapsed": the collapsed-point variant: every layer's generator A
                    is replaced by its batch mean (a leaf tensor), the
                    forward is re-run with the frozen generators, and the
                    per-sample adjoints are taken w.r.t. the frozen
                    generator rows, per the definition.

Both are measured on `--n_batches` independent batches drawn from the
globally reserved probe region at the tail of the token stream (untouched
by training in every wave), and reported per batch with mean/std summaries.
Each empirical per-batch covariance has rank at most batch_size-1 in the
sample direction; replication across batches is the uncertainty handle.

Scale conventions reported: rho_rms (global RMS of the row fluctuation, the
normalization used by xi) and rho_max (largest row-fluctuation norm, the
radius of the ||xi|| <= 1 convention).

Usage: python3 measure_tilt.py [--n_batches 8] [--mode both]
                               [--device cuda:0] <run_name> [...]
Writes runs/<name>/tilt_v2.json (the original tilt.json is left as is).
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pldr_model_v510 import PLDR_Model  # noqa: E402
from train_run import (masked_loss_function, create_masks,  # noqa: E402
                       PROBE_RESERVE_CHUNKS)
import instrument  # noqa: E402


def build_model(cfg, device):
    return PLDR_Model(
        num_layers=cfg["layers"], d_model=cfg["d_model"],
        num_heads=cfg["heads"], dff=cfg["dff"], input_vocab_size=32000,
        A_dff=cfg["adff"], num_reslayerA=8, num_denseA=2,
        max_seq_len=4096, device=device)


def _capture_rows(model):
    """Register hooks capturing each layer's LN(D_Q) output; returns
    (captured dict, remove callback)."""
    captured, hooks = {}, []
    for li, dec in enumerate(model.decoder.dec_layers):
        def mk(li):
            def hook(module, inp, out):
                captured[li] = out
            return hook
        hooks.append(dec.mha1.layernorm1.register_forward_hook(mk(li)))
    return captured, (lambda: [h.remove() for h in hooks])


def _tilt_stats(model, grads, captured):
    """Per-layer tilt statistics from per-sample adjoints and captured
    LN rows."""
    out = []
    for li in range(len(model.decoder.dec_layers)):
        r = captured[li]              # [B, h, dk, dk] rows of LN(D_Q)
        g = grads[li]                 # same shape: dL/dA rows
        B = r.shape[0]
        rbar = r.mean(dim=0, keepdim=True)
        dr = r - rbar
        rho_rms = dr.pow(2).mean().sqrt().clamp_min(1e-30)
        rho_max = dr.norm(dim=-1).max().clamp_min(1e-30)
        xi = dr / rho_rms
        gbar = g.mean(dim=0, keepdim=True)
        gflux = g - gbar
        # C_a = E_b [gflux_row (x) xi_row]: per (h, i) a dk x dk matrix
        C = torch.einsum("bhid,bhie->hide", gflux, xi) / B
        phi = instrument.make_phi(model.decoder.dec_layers[li].mha1)
        rows = rbar[0].reshape(-1, r.shape[-1])  # [h*dk, dk]
        J = torch.vmap(torch.func.jacrev(phi))(rows).reshape(C.shape)
        cn, jn = C.flatten(), J.flatten()
        denom = (cn.norm() * jn.norm()).clamp_min(1e-30)
        out.append({
            "layer": li,
            "C_fro": C.norm().item(),
            "gbar_norm": gbar.norm().item(),
            "g_rms": g.pow(2).mean().sqrt().item(),
            "rho_rms": rho_rms.item(),
            "rho_max": rho_max.item(),
            "cos_C_J": (cn @ jn / denom).item(),
            "J_fro": J.norm().item(),
        })
    return out


def measure_current(model, x, mask, y):
    """Tilt proxy at the current state (adjoints at the current model)."""
    captured, remove = _capture_rows(model)
    model.eval()
    preds, _, _, kvcachelst = model([x, mask])
    remove()
    loss = masked_loss_function(y, preds)
    A_list = [kv[2] for kv in kvcachelst]  # [B, h, dk, dk], in-graph
    grads = torch.autograd.grad(loss, A_list)
    stats = _tilt_stats(model, [g.detach() for g in grads],
                        {k: v.detach() for k, v in captured.items()})
    model.train()
    return stats, loss.item()


def measure_collapsed(model, x, mask, y):
    """Collapsed-point tilt: freeze each layer's generator at its batch
    mean (leaf tensors), re-run the forward with the frozen generators,
    and take per-sample adjoints w.r.t. the frozen rows."""
    # pass 1: batch-mean generators at the current state
    model.eval()
    with torch.no_grad():
        _, _, _, kvcachelst = model([x, mask])
        Abar = [kv[2].mean(dim=0, keepdim=True) for kv in kvcachelst]
    B = x.shape[0]
    leaves = [a.expand(B, -1, -1, -1).clone().requires_grad_(True)
              for a in Abar]

    # pass 2: forward with every plga layer's A input replaced by the leaf
    # (instance-level forward shadowing; removed again in `finally`)
    layers = [dec.mha1.plgatt_layer for dec in model.decoder.dec_layers]

    def make_patched(orig, leaf):
        def patched(inputs, Gcache=None, **kw):
            q, k, v, _, m = inputs
            return orig([q, k, v, leaf, m], Gcache=Gcache, **kw)
        return patched

    for ly, leaf in zip(layers, leaves):
        ly.forward = make_patched(ly.forward, leaf)
    captured, remove = _capture_rows(model)
    try:
        preds, _, _, _ = model([x, mask])
    finally:
        remove()
        for ly in layers:
            del ly.forward  # drop the instance shadow; class method resumes
    loss = masked_loss_function(y, preds)
    grads = torch.autograd.grad(loss, leaves)
    stats = _tilt_stats(model, [g.detach() for g in grads],
                        {k: v.detach() for k, v in captured.items()})
    model.train()
    return stats, loss.item()


def summarize(batch_stats):
    """mean/std across batches, per layer and key."""
    n_layers = len(batch_stats[0])
    keys = [k for k in batch_stats[0][0] if k != "layer"]
    summ = []
    for li in range(n_layers):
        row = {"layer": li}
        for k in keys:
            vals = np.array([bs[li][k] for bs in batch_stats])
            row[f"{k}_mean"] = float(vals.mean())
            row[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summ.append(row)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--mode", choices=["both", "current", "collapsed"],
                    default="both")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tokens", default="data/refinedweb_tokens.npy")
    args = ap.parse_args()
    device = torch.device(args.device)

    toks = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    for name in args.runs:
        cfgline = open(f"runs/{name}/log.jsonl").readline()
        cfg = json.loads(cfgline)
        ctx, batch = cfg["ctx"], cfg["batch"]
        chunks = toks[: len(toks) // ctx * ctx].reshape(-1, ctx)
        n_chunks = chunks.shape[0]
        pb = n_chunks - PROBE_RESERVE_CHUNKS
        batches = []
        for k in range(args.n_batches):
            xb = torch.from_numpy(
                chunks[pb + 2560 + 8 * k : pb + 2560 + 8 * (k + 1)]
                .astype(np.int64)).to(device)
            x, y = xb[:, :-1], xb[:, 1:]
            batches.append((x, create_masks(x, device), y))
        results = {"n_batches": args.n_batches, "mode": args.mode,
                   "batch_region_start_chunk": int(pb + 2560),
                   "ckpts": {}}
        for ckpt in sorted(glob.glob(f"runs/{name}/ckpt_*.pt")):
            model = build_model(cfg, device)
            ck = torch.load(ckpt, map_location=device)
            model.load_state_dict(ck["model"])
            entry = {}
            for mode, fn in [("current", measure_current),
                             ("collapsed", measure_collapsed)]:
                if args.mode not in ("both", mode):
                    continue
                per_batch, losses = [], []
                for (x, m, y) in batches:
                    stats, lval = fn(model, x, m, y)
                    per_batch.append(stats)
                    losses.append(lval)
                entry[mode] = {"loss_mean": float(np.mean(losses)),
                               "batches": per_batch,
                               "summary": summarize(per_batch)}
            results["ckpts"][os.path.basename(ckpt)] = entry
            del model
            torch.cuda.empty_cache()
        with open(f"runs/{name}/tilt_v2.json", "w") as f:
            json.dump(results, f, indent=1)
        print(name, "->", f"runs/{name}/tilt_v2.json")
        for k, v in results["ckpts"].items():
            for mode in v:
                s0 = v[mode]["summary"][0]
                print(f"  {k} [{mode}] L0: C_fro="
                      f"{s0['C_fro_mean']:.3e}+-{s0['C_fro_std']:.1e} "
                      f"cos(C,J)={s0['cos_C_J_mean']:+.2f} "
                      f"rho_rms={s0['rho_rms_mean']:.3e}")


if __name__ == "__main__":
    main()
