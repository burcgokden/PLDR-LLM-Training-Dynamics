"""Post-hoc measurement of the restricted jet-metric fidelity constants
(the manuscript's Assumption on the realizable tangent): the estimator
the wave-6 preregistration declares, replacing the eight-random-direction
Gram proxy that measured neither the stated operator nor the driven
directions.

Observable (the sampled jet evaluation, extended with the value jet):
Psi(theta) = concat over layers of [ phi(rbar_l) ;  {J_phi(r_i) v_l}_i ]
with rbar_l the batch-mean anchor row, r_i the sampled visited rows, and
v_l a fixed unit direction per layer (the same sampling scheme and seed
as the training-time proxy, so the two are comparable).  Parameters:
the phi (metric-learner) block.

Estimator, per checkpoint:
- adaptive randomized range-finder on DPsi: unit random parameter
  directions are added one at a time (images via the double-VJP trick);
  the declared stopping rule is sigma_k / sigma_1 < 1e-2 or k = 64;
  the actual Adam update direction (final checkpoint, where optimizer
  state exists) and the gradient direction are then ADDED to the basis;
- reported constants:
  * eps_T: residual-to-range error of the driven jet directions: the
    exploitation image d_C (per layer, alpha part zero, J part
    Cbar_l v_l tiled over the sampled rows, with Cbar_l the
    head/row-aggregated exploitation matrix from the adjoint
    measurement) and its transpose partner d_C2;
  * capture_grad / capture_upd: fraction of the gradient/update image
    captured by the random-only range (diagnostic that random probing
    finds the driven subspace);
  * kappa_T: (sigma_1 / sigma_r)^2 over the singular values above the
    1e-2 relative cutoff (the restricted Gram condition number on the
    estimated range);
  * eps_off: alpha-J mixing ratio of the orthonormal range basis,
    sum_k min(|q_k^alpha|^2, |q_k^J|^2) / sum_k max(...), in [0, 1]
    (0 = block-aligned, 1 = fully mixed);
  * holdout residuals of 4 fresh random probes (range convergence
    diagnostic).

Usage: python3 measure_fidelity.py [--device cuda:0] <run_name> [...]
Writes runs/<name>/fidelity.json.
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
from train_run import (masked_loss_function, create_masks,  # noqa: E402
                       PROBE_RESERVE_CHUNKS)
from measure_tilt import build_model, _capture_rows  # noqa: E402
import instrument  # noqa: E402

TOL_REL = 1e-2
K_MAX = 64
N_HOLDOUT = 4


def observable(model, probe_x, probe_mask, n_rows=8, seed=0):
    """The extended jet evaluation with its alpha/J block index masks.
    Returns (fvec with graph, phi_params, alpha_mask boolean np array)."""
    blocks = instrument.param_blocks(model)
    phi_params = blocks["phi"]
    g = torch.Generator().manual_seed(seed)
    rows = instrument.visited_rows(model, probe_x, probe_mask,
                                   max_rows_per_layer=n_rows, generator=g)
    model.eval()
    pieces, is_alpha, vs = [], [], []
    for li, dec in enumerate(model.decoder.dec_layers):
        phi = instrument.make_phi(dec.mha1)
        r = rows[li]
        v = torch.randn(r.shape[-1], generator=g).to(r.device, r.dtype)
        v = v / v.norm()
        vs.append(v)
        rbar = r.mean(dim=0, keepdim=True).detach().requires_grad_(True)
        a = phi(rbar).reshape(-1)
        pieces.append(a)
        is_alpha.append(np.ones(a.numel(), bool))
        jv = instrument._rowmap_jvp(phi, r, v.expand_as(r)).reshape(-1)
        pieces.append(jv)
        is_alpha.append(np.zeros(jv.numel(), bool))
    fvec = torch.cat(pieces)
    return fvec, phi_params, np.concatenate(is_alpha), rows, vs


def make_jvp(fvec, phi_params):
    def theta_jvp(w):
        u_dummy = torch.zeros_like(fvec, requires_grad=True)
        s = torch.autograd.grad(fvec, phi_params, grad_outputs=u_dummy,
                                create_graph=True, retain_graph=True,
                                allow_unused=True)
        inner = sum((si * wi).sum() for si, wi in zip(s, w)
                    if si is not None)
        jvp, = torch.autograd.grad(inner, u_dummy, retain_graph=True)
        return jvp.detach()
    return theta_jvp


def unit_dirs(phi_params, n, gen):
    out = []
    for _ in range(n):
        w = [torch.randn(p.shape, generator=gen).to(p.device, p.dtype)
             for p in phi_params]
        nw = math.sqrt(sum((wi * wi).sum().item() for wi in w))
        out.append([wi / nw for wi in w])
    return out


def exploitation_dirs(model, x, mask, y, rows, vs):
    """Driven jet-space directions from the adjoint measurement: per
    layer the aggregated exploitation matrix Cbar_l applied to v_l,
    tiled over the sampled rows (alpha part zero), plus the transpose
    partner."""
    captured, remove = _capture_rows(model)
    model.eval()
    preds, _, _, kvcachelst = model([x, mask])
    remove()
    loss = masked_loss_function(y, preds)
    A_list = [kv[2] for kv in kvcachelst]
    grads = torch.autograd.grad(loss, A_list)
    d_C, d_C2 = [], []
    for li in range(len(model.decoder.dec_layers)):
        r = captured[li].detach()
        g = grads[li].detach()
        B = r.shape[0]
        dr = r - r.mean(dim=0, keepdim=True)
        rho = dr.pow(2).mean().sqrt().clamp_min(1e-30)
        xi = dr / rho
        gf = g - g.mean(dim=0, keepdim=True)
        C = torch.einsum("bhid,bhie->de", gf, xi) / (B * r.shape[1]
                                                     * r.shape[2])
        v = vs[li]
        n_rows = rows[li].shape[0]
        a0 = torch.zeros(v.numel(), device=v.device, dtype=v.dtype)
        d_C.append(torch.cat([a0, (C @ v).repeat(n_rows)]))
        d_C2.append(torch.cat([a0, (C.T @ v).repeat(n_rows)]))
    model.zero_grad(set_to_none=True)
    return torch.cat(d_C).cpu(), torch.cat(d_C2).cpu()


def grad_dir(model, x, mask, y, phi_params):
    model.eval()
    preds, _, _, _ = model([x, mask])
    loss = masked_loss_function(y, preds)
    gs = torch.autograd.grad(loss, phi_params, allow_unused=True)
    gs = [torch.zeros_like(p) if g is None else g
          for g, p in zip(gs, phi_params)]
    n = math.sqrt(sum((g * g).sum().item() for g in gs))
    model.zero_grad(set_to_none=True)
    return [g / n for g in gs] if n > 0 else None


def upd_dir(model, ck, phi_params, cfg):
    """Adam update direction from a checkpoint carrying optimizer
    state (the final checkpoint); None otherwise."""
    if "opt" not in ck or cfg.get("optimizer", "adamw") != "adamw":
        return None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            betas=(0.9, 0.95), eps=1e-5,
                            weight_decay=cfg.get("wd", 0.1))
    try:
        opt.load_state_dict(ck["opt"])
    except Exception:
        return None
    u = []
    for p in phi_params:
        st = opt.state.get(p)
        if st is None or "exp_avg" not in st:
            return None
        t = st["step"]
        t = t.item() if torch.is_tensor(t) else t
        b1, b2 = 0.9, 0.95
        mhat = st["exp_avg"] / (1 - b1 ** t)
        vhat = st["exp_avg_sq"] / (1 - b2 ** t)
        u.append(mhat / (vhat.sqrt() + 1e-5))
    n = math.sqrt(sum((ui * ui).sum().item() for ui in u))
    return [ui / n for ui in u] if n > 0 else None


def analyze_ckpt(model, ck, cfg, x, mask, y, device):
    fvec, phi_params, alpha_mask, rows, vs = observable(model, x, mask)
    jvp = make_jvp(fvec, phi_params)
    gen = torch.Generator().manual_seed(1)

    images = []
    sigmas = []
    k_conv = None
    while len(images) < K_MAX:
        w = unit_dirs(phi_params, 1, gen)[0]
        images.append(jvp(w).cpu().double())
        Y = torch.stack(images)
        s = torch.linalg.svdvals(Y)
        sigmas = [float(v) for v in s]
        if len(images) >= 8 and sigmas[-1] / max(sigmas[0], 1e-300) \
                < TOL_REL:
            k_conv = len(images)
            break
    if k_conv is None:
        k_conv = len(images)

    # holdout residuals against the random-only range
    Y = torch.stack(images)
    Q, _ = torch.linalg.qr(Y.T)  # [m, k] orthonormal columns
    def resid(dvec):
        d = dvec.cpu().double()
        nd = d.norm().item()
        if nd <= 0:
            return None
        r = d - Q @ (Q.T @ d)
        return float(r.norm().item() / nd)
    holdout = [resid(jvp(w))
               for w in unit_dirs(phi_params, N_HOLDOUT, gen)]

    dC, dC2 = exploitation_dirs(model, x, mask, y, rows, vs)
    gdir = grad_dir(model, x, mask, y, phi_params)
    udir = upd_dir(model, ck, phi_params, cfg)
    cap_grad = None
    cap_upd = None
    if gdir is not None:
        gi = jvp(gdir).cpu().double()
        if gi.norm() > 0:
            cap_grad = float((Q.T @ gi).norm().item()
                             / gi.norm().item())
    if udir is not None:
        ui = jvp(udir).cpu().double()
        if ui.norm() > 0:
            cap_upd = float((Q.T @ ui).norm().item()
                            / ui.norm().item())

    # final basis includes the driven parameter directions
    extra = [jvp(d).cpu().double() for d in (gdir, udir)
             if d is not None]
    Yfull = torch.stack(images + [e for e in extra if e.norm() > 0])
    sf = torch.linalg.svdvals(Yfull)
    cutoff = TOL_REL * float(sf[0])
    live = [float(v) for v in sf if float(v) >= cutoff]
    kappa_T = (live[0] / live[-1]) ** 2 if len(live) >= 2 else None
    Qf, _ = torch.linalg.qr(Yfull.T)

    def resid_full(d):
        nd = d.norm().item()
        if nd <= 0:
            return None
        return float((d - Qf @ (Qf.T @ d)).norm().item() / nd)

    eps_T = {"exploitation": resid_full(dC.double()),
             "exploitation_partner": resid_full(dC2.double())}

    am = torch.from_numpy(alpha_mask)
    num = den = 0.0
    for k in range(Qf.shape[1]):
        q = Qf[:, k]
        a2 = float((q[am] ** 2).sum())
        j2 = float((q[~am] ** 2).sum())
        num += min(a2, j2)
        den += max(a2, j2)
    eps_off = num / den if den > 0 else None

    return {"k_random": len(images), "k_converged": k_conv,
            "singular_values": sigmas,
            "holdout_residuals": holdout,
            "eps_T": eps_T, "kappa_T": kappa_T, "eps_off": eps_off,
            "capture_grad_random_range": cap_grad,
            "capture_upd_random_range": cap_upd,
            "has_update_direction": udir is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tokens", default="data/refinedweb_tokens.npy")
    args = ap.parse_args()
    device = torch.device(args.device)
    toks = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    for name in args.runs:
        cfg = json.loads(open(f"runs/{name}/log.jsonl").readline())
        ctx = cfg["ctx"]
        chunks = toks[: len(toks) // ctx * ctx].reshape(-1, ctx)
        pb = chunks.shape[0] - PROBE_RESERVE_CHUNKS
        xb = torch.from_numpy(chunks[pb:pb + 8].astype(np.int64)).to(device)
        x, y = xb[:, :-1], xb[:, 1:]
        mask = create_masks(x, device)
        out = {"ckpts": {}, "tol_rel": TOL_REL, "k_max": K_MAX}
        for ckpt in sorted(glob.glob(f"runs/{name}/ckpt_*.pt")):
            model = build_model(cfg, device)
            ck = torch.load(ckpt, map_location=device)
            model.load_state_dict(ck["model"])
            try:
                out["ckpts"][os.path.basename(ckpt)] = analyze_ckpt(
                    model, ck, cfg, x, mask, y, device)
            except RuntimeError as e:
                out["ckpts"][os.path.basename(ckpt)] = {"error": str(e)}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        with open(f"runs/{name}/fidelity.json", "w") as f:
            json.dump(out, f, indent=1)
        print(name, "->", f"runs/{name}/fidelity.json")
        for k, v in out["ckpts"].items():
            if "eps_T" in v:
                print(f"  {k}: k={v['k_random']} kappa_T="
                      f"{v['kappa_T']:.3g} eps_off={v['eps_off']:.3f} "
                      f"eps_T(C)={v['eps_T']['exploitation']:.3f} "
                      f"cap_upd={v['capture_upd_random_range']}")


if __name__ == "__main__":
    main()
