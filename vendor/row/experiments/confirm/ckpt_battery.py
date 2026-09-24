#!/usr/bin/env python3
"""E0 checkpoint battery: ledger defect and fiber constants.

Measured along a run's checkpoints (seam checkpoints carry optimizer
state and additionally yield the update-projection statistic):

  radial_deriv  <theta_S, grad_S lambda> along the coupling-path
                scaling rays, p = 2 (S = {W, a}) and p = 3
                (S = {W, a, phi block}; the exponent tensor P is not
                a path factor), by central difference on the ray
                theta_S -> (1+s) theta_S of the physical
                block-restricted Gauss--Newton curvature. The same
                direction is evaluated under the full Hessian as a
                residual certificate. The driver assembles the defect
                h = radial_deriv - (2 p kappa u^2 + m_b b)
                with the E0-measured (kappa, u, b) at the checkpoint's
                step; h_max is the
                maximum |h| over checkpoints.
  kappa_phi     fiber squared-norm curvature: the fiber is the
                compensation curve of Lemma lem:compensation realized
                EXACTLY in parameter space: s |-> (b_W + s*delta,
                b_a + a @ (Ap(0) - Ap(s))) with
                Ap(s) = (iswiglu(W Abar + b_W + s*delta) + eps)^P and
                Abar the probe-batch anchor-mean metric input (varying
                Abar is realized exactly by shifting b_W).  G_LM is
                invariant along the curve by the lemma's algebra.
                kappa_phi = (d^2/dphi^2) ||theta||^2 / 2 in arc length.
  C_phi         per-step disturbance projection on the unit fiber
                tangent: |<u, t_phi>| with u = m_hat/(sqrt(v_hat)+eps)
                the checkpoint's Adam step direction (eta-free), and
                the non-loss variant |<u - P^{-1} g, t_phi>| with g
                the probe-batch gradient (momentum/preconditioner
                disturbance).  Requires a with-optimizer checkpoint.

Usage: ckpt_battery.py <run> [--device cuda:0] [--max_ckpts 12]
Writes confirm/out/e0/ckpt_battery_<run>.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, os.path.join(EXP, "analysis"))

from train_run import (masked_loss_function, create_masks,   # noqa
                       PROBE_RESERVE_CHUNKS)
from measure_tilt import build_model                          # noqa
from power_law_attention_layer_v510 import iSwiGLU            # noqa
import instrument                                             # noqa

from e0_prime_adapters import global_checkpoint_step          # noqa

EPS_ADJ = 1e-9
N_FIBER_DIRS = 4


def probe_batch(cfg, device, n=8):
    toks = np.memmap(os.path.join(EXP, cfg["tokens"]),
                     dtype=np.uint16, mode="r")
    ctx = cfg["ctx"]
    chunks = toks[: len(toks) // ctx * ctx].reshape(-1, ctx)
    pb = chunks.shape[0] - PROBE_RESERVE_CHUNKS
    rows = np.array(chunks[pb:pb + n], dtype=np.int64)
    x = torch.from_numpy(rows).to(device)
    return x[:, :-1], x[:, 1:]


def capture_A(model, x, mask):
    """Anchor-mean metric input Abar per layer (the A argument of the
    PLGA head map), captured by forward hook."""
    caps = {}
    hooks = []
    for li, dec in enumerate(model.decoder.dec_layers):
        def mk(li):
            def hook(module, inp, out):
                A = inp[0][3]
                caps[li] = A.detach().mean(dim=0)  # [head, dk, dk]
            return hook
        hooks.append(dec.mha1.plgatt_layer.register_forward_hook(
            mk(li)))
    with torch.no_grad():
        model([x, mask])
    for h in hooks:
        h.remove()
    return caps


def checkpoint_optimizer(model, ck, cfg):
    """Reconstruct the exact AdamW metric carried by a checkpoint."""
    if "opt" not in ck or cfg.get("optimizer", "adamw") != "adamw":
        return None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
        eps=1e-5, weight_decay=cfg.get("wd", 0.1))
    try:
        optimizer.load_state_dict(ck["opt"])
    except Exception:
        return None
    return optimizer


def tracked_curvature(model, loss_fn, x, mask, y, params, optimizer,
                      seed=0):
    """Physical preconditioned GN quotient and full-Hessian residual."""
    if optimizer is None:
        return None
    stats, _ = instrument.mode_track(
        model, loss_fn, x, mask, y, optimizer, params,
        n_iter=20, tol=1e-3, seed=seed, variant="pre_live",
        operator="gauss_newton")
    return stats


def radial_derivative(model, loss_fn, x, mask, y, optimizer, s=2e-3):
    """Central differences of the plga-block tracked curvature along
    the coupling-path scaling rays: p = 2 (W, a; the multiplicative
    path factors of the head map; the exponent tensor P is NOT a
    path factor) and p = 3 (W, a, and the metric-learner phi block).
    Biases fixed. Returns the two derivatives and primary GN metadata."""
    if optimizer is None:
        return None
    plga_all = []
    for dec in model.decoder.dec_layers:
        plga_all += list(dec.mha1.plgatt_layer.parameters())
    paths = {}
    p2 = []
    for dec in model.decoder.dec_layers:
        pl = dec.mha1.plgatt_layer
        p2 += [pl.Wlst, pl.alst]
    paths[2] = p2
    p3 = list(p2)
    for dec in model.decoder.dec_layers:
        p3 += [p for p in dec.mha1.reslayerAs.parameters()]
    paths[3] = p3
    out = {}
    for pk, path in paths.items():
        with torch.no_grad():
            backup = [p.detach().clone() for p in path]
        vals = {}
        for sign in (+1, -1):
            with torch.no_grad():
                for p, b in zip(path, backup):
                    p.copy_(b * (1.0 + sign * s))
            probe = tracked_curvature(
                model, loss_fn, x, mask, y, plga_all, optimizer)
            if probe is None:
                return None
            vals[sign] = float(probe["mode_lam"])
        with torch.no_grad():
            for p, b in zip(path, backup):
                p.copy_(b)
        out[pk] = (vals[+1] - vals[-1]) / (2.0 * s)
    primary = tracked_curvature(
        model, loss_fn, x, mask, y, plga_all, optimizer)
    return out[2], out[3], primary


def fiber_layer(model, li, Abar, opt_dirs, grad_dirs, device,
                h=1e-2, n_dirs=N_FIBER_DIRS):
    """kappa_phi and the C_phi projections for one layer."""
    pl = model.decoder.dec_layers[li].mha1.plgatt_layer
    W, bW, P, a, ba = (pl.Wlst.detach(), pl.blst.detach(),
                       pl.pwlst.detach(), pl.alst.detach(),
                       pl.balst.detach())
    z0 = torch.matmul(W, Abar) + bW

    def Ap(z):
        return torch.pow(iSwiGLU(z) + EPS_ADJ, P)

    Ap0 = Ap(z0)
    out = []
    g = torch.Generator(device="cpu")
    for k in range(n_dirs):
        g.manual_seed(k)
        delta = torch.randn(bW.shape, generator=g).to(device)
        delta = delta / delta.norm()

        def theta(s):
            bW_s = bW + s * delta
            ba_s = ba + torch.matmul(a, Ap0 - Ap(z0 + s * delta))
            return bW_s, ba_s

        n_of = {}
        tan_ba = None
        for s in (+h, 0.0, -h):
            bW_s, ba_s = theta(s)
            n_of[s] = float((bW_s ** 2).sum() + (ba_s ** 2).sum())
        bWp, bap = theta(+h)
        bWm, bam = theta(-h)
        tan_ba = (bap - bam) / (2 * h)
        t_norm2 = float((delta ** 2).sum() + (tan_ba ** 2).sum())
        d2n = (n_of[+h] - 2 * n_of[0.0] + n_of[-h]) / (h * h)
        kappa_phi = d2n / (2.0 * t_norm2)
        rec = dict(dir=k, kappa_phi=float(kappa_phi))
        if opt_dirs is not None:
            u_bW, u_ba = opt_dirs[li]
            tn = np.sqrt(t_norm2)
            proj = float((u_bW * delta).sum()
                         + (u_ba * tan_ba).sum()) / tn
            rec["C_phi_update"] = abs(proj)
            if grad_dirs is not None:
                g_bW, g_ba = grad_dirs[li]
                proj_nl = float(((u_bW - g_bW) * delta).sum()
                                + ((u_ba - g_ba) * tan_ba).sum()) / tn
                rec["C_phi_nonloss"] = abs(proj_nl)
        out.append(rec)
    return out


def opt_directions(model, ck, cfg, x, mask, y, device):
    """Adam step direction u = m_hat/(sqrt(v_hat)+eps) and the
    preconditioned gradient P^{-1} g for the (b_W, b_a) coordinates
    of every layer, from the checkpoint's optimizer state."""
    if "opt" not in ck:
        return None, None
    params = list(model.parameters())
    st = ck["opt"]["state"]
    beta1, beta2 = 0.9, 0.95
    eps = 1e-5
    # gradient on the probe batch
    model.zero_grad(set_to_none=True)
    preds, _, _, _ = model([x, mask])
    loss = masked_loss_function(y, preds)
    loss.backward()
    by_id = {}
    for i, p in enumerate(params):
        if i not in st:
            continue
        s = st[i]
        step = s["step"]
        step = step.item() if torch.is_tensor(step) else step
        mh = s["exp_avg"] / (1 - beta1 ** step)
        vh = s["exp_avg_sq"] / (1 - beta2 ** step)
        den = vh.sqrt() + eps
        u = mh / den
        gp = (p.grad / den) if p.grad is not None else torch.zeros_like(u)
        by_id[id(p)] = (u, gp)
    opt_dirs, grad_dirs = {}, {}
    for li, dec in enumerate(model.decoder.dec_layers):
        pl = dec.mha1.plgatt_layer
        if id(pl.blst) in by_id and id(pl.balst) in by_id:
            opt_dirs[li] = (by_id[id(pl.blst)][0],
                            by_id[id(pl.balst)][0])
            grad_dirs[li] = (by_id[id(pl.blst)][1],
                             by_id[id(pl.balst)][1])
    model.zero_grad(set_to_none=True)
    return (opt_dirs or None), (grad_dirs or None)


def analyze_ckpt(path, cfg, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    chain_meta = None
    chain_meta_path = os.path.join(os.path.dirname(path), "chain_meta.json")
    if os.path.isfile(chain_meta_path):
        with open(chain_meta_path, encoding="utf-8") as handle:
            chain_meta = json.load(handle)
    clock = global_checkpoint_step(path, ck, chain_meta)
    model = build_model(cfg, device)
    model.load_state_dict(ck["model"])
    x, y = probe_batch(cfg, device)
    mask = create_masks(x, device)

    def loss_fn(yt, yp):
        return masked_loss_function(yt, yp)

    optimizer = checkpoint_optimizer(model, ck, cfg)
    radial = radial_derivative(
        model, loss_fn, x, mask, y, optimizer)
    if radial is not None:
        rad2, rad3, primary = radial
    else:
        rad2 = rad3 = None
        primary = {}
    gn_value = primary.get("mode_lam")
    residual_value = primary.get("mode_gn_residual")
    residual_relative = (
        abs(residual_value) / max(abs(gn_value), 1e-300)
        if residual_value is not None and gn_value is not None else None)
    Abar = capture_A(model, x, mask)
    rd = instrument.rowmap_diagnostics(model, x, mask)
    cn = instrument.coupling_norms(model)
    opt_dirs, grad_dirs = opt_directions(model, ck, cfg, x, mask, y,
                                         device)
    fibers = {}
    for li in range(len(model.decoder.dec_layers)):
        fibers[li] = fiber_layer(model, li, Abar[li], opt_dirs,
                                 grad_dirs, device)
    out = dict(
        ckpt=os.path.basename(path), step=clock["global_step"],
        segment_local_step=clock["segment_local_step"],
        step_source=clock["source"],
        primary_defined=bool(radial is not None),
        gn_preconditioned=gn_value,
        full_hessian_on_gn=primary.get("mode_full_hessian_on_gn"),
        gn_full_residual=residual_value,
        gn_full_residual_relative=residual_relative,
        gn_lanczos_residual=primary.get("mode_resid"),
        radial_deriv_p2=rad2, radial_deriv_p3=rad3,
        rowmap_sigma_samples=[[float(x) for x in layer]
                              for layer in rd["sigma_samples"]],
        rowmap_sigma_med=[float(v) for v in rd["sigma_med"]],
        a_norm_layers=[float(v) for v in cn["a_norm_layers"]],
        has_opt=bool("opt" in ck),
        fibers=fibers)
    del model, ck
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_ckpts", type=int, default=12)
    args = ap.parse_args()
    rundir = os.path.join(EXP, "runs", args.run)
    cfg = json.loads(open(os.path.join(rundir,
                                       "log.jsonl")).readline())
    def step_of(p):
        tag = os.path.basename(p).split("_")[1].split(".")[0]
        return int(tag) if tag.isdigit() else 10 ** 9
    cks = sorted(glob.glob(os.path.join(rundir, "ckpt_*.pt")),
                 key=step_of)
    if len(cks) > args.max_ckpts:
        idx = np.linspace(0, len(cks) - 1, args.max_ckpts).astype(int)
        cks = [cks[i] for i in idx]
    recs = [analyze_ckpt(p, cfg, args.device) for p in cks]
    outdir = os.path.join(HERE, "out", "e0")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,
                           f"ckpt_battery_{args.run}.json"), "w") as f:
        json.dump(dict(run=args.run, ckpts=recs), f, indent=1,
                  allow_nan=False)
    for r in recs:
        kmed = np.median([d["kappa_phi"]
                          for li in r["fibers"]
                          for d in r["fibers"][li]])
        print(r["ckpt"], "GN", r["gn_preconditioned"],
              "GN residual", r["gn_full_residual_relative"],
              "radial p2/p3", r["radial_deriv_p2"], r["radial_deriv_p3"],
              "kappa_phi(med)", round(float(kmed), 4),
              "opt", r["has_opt"])


if __name__ == "__main__":
    main()
