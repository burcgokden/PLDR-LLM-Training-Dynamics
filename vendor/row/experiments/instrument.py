"""Along-training diagnostics for the PLDR-LLM row-map collapse study.

All quantities are defined to mirror the foundations paper
(power_law_graph_attention.tex) and the SOC paper:

- order parameter proxies: normalized RMSE between deductive outputs computed
  on two fixed, disjoint probe batches (Definition of the order parameter with
  the stabilized RMS normalization; cross-input version);
- row-map diagnostics: largest singular values of the Jacobian of the full
  composed row map phi at rows visited by the probe inputs, and pairwise
  contraction ratios ||phi(r)-phi(r')||/||r-r'|| (Appendix "Numerical audit"
  of the foundations paper, extended from one checkpoint to trajectories);
- curvature: Lanczos with full reorthogonalization on exact operator-vector
  products.  Four distinct families of quantities are logged and must not
  be conflated:
    * `lam_*`            raw-metric extremal eigenvalue estimates
                         lambda_max(H) (and lambda_min where noted), for the
                         full parameter vector and for parameter blocks
                         (metric-learner/"phi" block = ResLayerA parameters;
                         "plga" block = W, b_W, P, a, b_a; "nonemb" = all
                         parameters except the embedding and the LM head);
    * `lam_pre_*`        preconditioned extremal eigenvalue estimates
                         lambda_{max,min}(D^{-1/2} H D^{-1/2}) with
                         D = diag(sqrt(v_hat) + eps) from the Adam state,
                         with Ritz-residual bounds and multiple starts;
    * `gn_pre_*`         physical preconditioned generalized Gauss--Newton
                         eigenvalue estimates.  This is the primary stress
                         of the row-map-collapse theory;
    * `rayq_*`           SIGNED Rayleigh quotients along specific directions
                         (the Adam update direction; the gradient of the
                         transverse collapse statistic).  These are not
                         eigenvalues of any operator and can be negative.
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# deductive-output collection


@torch.no_grad()
def collect_deductive(model, x, mask):
    """Forward a probe batch; return per-layer A and G_LM (and A_LM)."""
    model.eval()
    _, _, att_weights, kvcachelst = model([x, mask])
    A_list = [kv[2].detach() for kv in kvcachelst]  # [B, h, dk, dk]
    ALM_list = [aw[0].detach() for aw in att_weights]
    GLM_list = [aw[4].detach() for aw in att_weights]
    model.train()
    return A_list, ALM_list, GLM_list


def _rms(t):
    return t.pow(2).mean().sqrt().item()


def norm_rmse(t1, t2):
    """Stabilized order-parameter statistic: RMSE / mean RMS magnitude."""
    num = (t1 - t2).pow(2).mean().sqrt().item()
    den = 0.5 * (_rms(t1) + _rms(t2))
    if num == 0.0:
        return 0.0
    return num / den


def order_params(model, x1, m1, x2, m2):
    """Cross-input order-parameter proxies between two probe batches.

    Returns dict with per-tensor-type model-level values (all layers pooled)
    and per-layer values for A and G_LM.
    """
    A1, ALM1, G1 = collect_deductive(model, x1, m1)
    A2, ALM2, G2 = collect_deductive(model, x2, m2)
    out = {}
    for name, l1, l2 in [("A", A1, A2), ("ALM", ALM1, ALM2), ("GLM", G1, G2)]:
        out[f"m_{name}"] = norm_rmse(torch.stack(l1), torch.stack(l2))
        out[f"m_{name}_layers"] = [norm_rmse(a, b) for a, b in zip(l1, l2)]
    # within-batch (intra-input) spread of A on probe 1: rms distance between
    # batch elements' tensors, normalized
    A = torch.stack(A1, dim=0)  # [L, B, h, dk, dk]
    Amean = A.mean(dim=1, keepdim=True)
    intra = (A - Amean).pow(2).mean().sqrt().item()
    out["m_A_intra"] = 0.0 if intra == 0 else intra / max(_rms(A), 1e-30)
    return out


# ---------------------------------------------------------------------------
# row-map diagnostics


def make_phi(mha):
    """The shared row map phi of one decoder layer: composition of ResLayerA
    units, acting on a single row r in R^dk (Proposition `rowfact`)."""

    def phi(r):
        a = r
        for unit in mha.reslayerAs:
            a = unit.ResUnit(a)
        return a

    return phi


@torch.no_grad()
def visited_rows(model, x, mask, max_rows_per_layer=64, generator=None):
    """Capture the rows of LN(D_Q) (input to phi) for each decoder layer."""
    captured = {}
    hooks = []
    for li, dec in enumerate(model.decoder.dec_layers):
        def make_hook(li):
            def hook(module, inp, out):
                captured[li] = out.detach()
            return hook
        hooks.append(dec.mha1.layernorm1.register_forward_hook(make_hook(li)))
    model.eval()
    model([x, mask])
    model.train()
    for h in hooks:
        h.remove()
    rows = {}
    for li, A in captured.items():
        r = A.reshape(-1, A.shape[-1])  # [(B*h*dk), dk]
        idx = torch.randperm(r.shape[0], generator=generator)[:max_rows_per_layer]
        rows[li] = r[idx].clone()
    return rows


def rowmap_diagnostics(model, x, mask, max_rows=32, n_pairs=64, seed=0):
    """Visited-row Jacobian singular maxima and contraction ratios.

    ``sigma_samples[layer][row]`` retains the raw within-layer axis
    from which all future endpoint summaries must be derived.
    """
    g = torch.Generator().manual_seed(seed)
    rows = visited_rows(model, x, mask, max_rows_per_layer=max_rows, generator=g)
    out = {"sigma_samples": [], "sigma_max": [], "sigma_med": [],
           "contract_max": [], "contract_med": [], "phi_out_diam": []}
    for li, dec in enumerate(model.decoder.dec_layers):
        phi = make_phi(dec.mha1)
        r = rows[li]
        J = torch.vmap(torch.func.jacrev(phi))(r)  # [n, dk, dk]
        if not torch.isfinite(J).all():
            # diverged run: record NaNs instead of crashing the SVD
            out["sigma_samples"].append(
                [float("nan")] * int(r.shape[0]))
            for k in out:
                if k != "sigma_samples":
                    out[k].append(float("nan"))
            continue
        sv = torch.linalg.svdvals(J)
        smax = sv[:, 0]
        out["sigma_samples"].append(
            [float(v) for v in smax.detach().cpu().tolist()])
        out["sigma_max"].append(smax.max().item())
        out["sigma_med"].append(smax.median().item())
        with torch.no_grad():
            fr = phi(r)
            i = torch.randint(0, r.shape[0], (n_pairs,), generator=g)
            j = torch.randint(0, r.shape[0], (n_pairs,), generator=g)
            keep = i != j
            i, j = i[keep], j[keep]
            dr = (r[i] - r[j]).norm(dim=-1)
            df = (fr[i] - fr[j]).norm(dim=-1)
            ratio = df / dr.clamp_min(1e-12)
            out["contract_max"].append(ratio.max().item())
            out["contract_med"].append(ratio.median().item())
            c = fr - fr.mean(0, keepdim=True)
            out["phi_out_diam"].append(2.0 * c.norm(dim=-1).max().item())
    return out


# ---------------------------------------------------------------------------
# parameter blocks


def param_blocks(model):
    """Partition trainable parameters into phi / plga / rest blocks."""
    phi_params, plga_params, rest_params = [], [], []
    phi_ids = set()
    plga_ids = set()
    for dec in model.decoder.dec_layers:
        for p in dec.mha1.reslayerAs.parameters():
            phi_ids.add(id(p))
        for p in dec.mha1.plgatt_layer.parameters():
            plga_ids.add(id(p))
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if id(p) in phi_ids:
            phi_params.append(p)
        elif id(p) in plga_ids:
            plga_params.append(p)
        else:
            rest_params.append(p)
    return {"phi": phi_params, "plga": plga_params, "rest": rest_params}


def nonemb_params(model):
    """All trainable parameters except the embedding and the LM head, the
    vocabulary-scale blocks whose near-dead directions dominate
    preconditioned spectra at small scale."""
    excl = set()
    for p in model.decoder.embedding.parameters():
        excl.add(id(p))
    for p in model.final_layer.parameters():
        excl.add(id(p))
    return [p for p in model.parameters()
            if p.requires_grad and id(p) not in excl]


def _flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


# ---------------------------------------------------------------------------
# Lanczos core (testable in isolation)


def _ritz(alphas, betas, beta_next):
    """Extremal Ritz values of the Lanczos tridiagonal and their residual
    bounds ||A y - theta y|| = beta_next * |last component of s|."""
    k = len(alphas)
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    for j, b in enumerate(betas[: k - 1]):
        T[j, j + 1] = b
        T[j + 1, j] = b
    evals, evecs = torch.linalg.eigh(T)
    rmax = abs(beta_next * evecs[-1, -1].item())
    rmin = abs(beta_next * evecs[-1, 0].item())
    return evals[-1].item(), evals[0].item(), rmax, rmin


_NAN_EXTREMAL = {"lam_max": float("nan"), "lam_min": float("nan"),
                 "resid_max": float("nan"), "resid_min": float("nan"),
                 "iters": 0}


def lanczos_extremal(matvec, template, n_iter=20, tol=0.0, seed=0):
    """Extremal eigenvalues of a symmetric operator by Lanczos with full
    reorthogonalization; robust to near-symmetric spectra
    (lam_max ~ -lam_min), where plain power iteration oscillates.

    matvec: maps a list of tensors shaped like `template` to the operator
    applied to it.  tol > 0 enables early stopping once both extremal
    Ritz-residual bounds fall below tol * spectral scale.

    Returns dict: lam_max, lam_min, resid_max, resid_min (absolute Ritz
    residual bounds), iters.  On a non-finite operator state (a diverged
    run) returns NaN entries instead of raising, so instrumentation of a
    deliberately destabilized cell records the divergence rather than
    crashing the run.
    """
    device = template[0].device
    g = torch.Generator(device=device).manual_seed(seed)
    v = [torch.randn(p.shape, generator=g, device=device,
                     dtype=p.dtype) for p in template]
    nv = _flat(v).norm()
    v = [vi / nv for vi in v]
    V = [v]
    alphas, betas = [], []
    beta_next = 0.0
    for j in range(n_iter):
        w = matvec(V[j])
        if not all(torch.isfinite(wi).all() for wi in w):
            return dict(_NAN_EXTREMAL)
        alpha = sum((wi * vi).sum() for wi, vi in zip(w, V[j])).item()
        alphas.append(alpha)
        w = [wi - alpha * vi for wi, vi in zip(w, V[j])]
        if j > 0:
            w = [wi - betas[-1] * ui for wi, ui in zip(w, V[j - 1])]
        # full reorthogonalization
        for u in V:
            proj = sum((wi * ui).sum() for wi, ui in zip(w, u))
            w = [wi - proj * ui for wi, ui in zip(w, u)]
        beta_next = _flat(w).norm().item()
        if beta_next < 1e-10:
            beta_next = 0.0
            break
        if tol > 0 and j >= 2:
            lmax, lmin, rmax, rmin = _ritz(alphas, betas, beta_next)
            scale = max(abs(lmax), abs(lmin), 1e-30)
            if max(rmax, rmin) < tol * scale:
                break
        if j < n_iter - 1:
            betas.append(beta_next)
            V.append([wi / beta_next for wi in w])
    if not (all(math.isfinite(a) for a in alphas)
            and all(math.isfinite(b) for b in betas)
            and math.isfinite(beta_next)):
        return dict(_NAN_EXTREMAL)
    try:
        lmax, lmin, rmax, rmin = _ritz(alphas, betas, beta_next)
    except RuntimeError:  # linalg.eigh failure on a degenerate state
        return dict(_NAN_EXTREMAL)
    return {"lam_max": lmax, "lam_min": lmin,
            "resid_max": rmax, "resid_min": rmin, "iters": len(alphas)}


def hessian_matvec(grads, params, dinv=None):
    """Hessian-vector product operator from create_graph gradients; with
    dinv (entries 1/sqrt(D_ii)) it is v -> D^{-1/2} H D^{-1/2} v."""

    def mv(v):
        vin = ([vi * di for vi, di in zip(v, dinv)]
               if dinv is not None else v)
        Hv = torch.autograd.grad(grads, params, grad_outputs=vin,
                                 retain_graph=True)
        if dinv is not None:
            Hv = [h * di for h, di in zip(Hv, dinv)]
        return [h.detach() for h in Hv]

    return mv


def _loss_graph(model, loss_fn, x, mask, y):
    """Return the scalar loss and model outputs on one retained graph."""
    model.eval()
    preds, _, _, _ = model([x, mask])
    loss = loss_fn(y, preds)
    return loss, preds


def _loss_grads(model, loss_fn, x, mask, y, params):
    loss, _preds = _loss_graph(model, loss_fn, x, mask, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return loss, grads


def gauss_newton_matvec(loss, outputs, params, dinv=None):
    """Generalized Gauss--Newton operator, optionally preconditioned.

    The returned map is ``D^-1/2 J^T H_loss J D^-1/2``.  It differentiates
    the loss only in output space and therefore excludes the indefinite
    model-curvature residual present in a full loss-Hessian product.
    """
    output_grad, = torch.autograd.grad(
        loss, outputs, create_graph=True, retain_graph=True)
    cotangent = torch.zeros_like(outputs, requires_grad=True)
    jt_cotangent = torch.autograd.grad(
        outputs, params, grad_outputs=cotangent, create_graph=True,
        retain_graph=True, allow_unused=True)
    jt_cotangent = tuple(
        torch.zeros_like(param) if value is None else value
        for param, value in zip(params, jt_cotangent))

    def mv(v):
        vin = ([vi * di for vi, di in zip(v, dinv)]
               if dinv is not None else v)
        jv, = torch.autograd.grad(
            jt_cotangent, cotangent, grad_outputs=vin, retain_graph=True)
        hout_jv, = torch.autograd.grad(
            output_grad, outputs, grad_outputs=jv, retain_graph=True)
        result = torch.autograd.grad(
            outputs, params, grad_outputs=hout_jv.detach(),
            retain_graph=True, allow_unused=True)
        result = [torch.zeros_like(param) if value is None else value
                  for param, value in zip(params, result)]
        if dinv is not None:
            result = [value * di for value, di in zip(result, dinv)]
        return [value.detach() for value in result]

    return mv


def sharpness_ext(model, loss_fn, x, mask, y, params, n_iter=20, tol=1e-3,
                  precond=None, n_starts=1, seed=0):
    """Extremal minibatch-loss Hessian eigenvalues w.r.t. `params` with
    Ritz residual bounds.  precond: list of D_ii tensors (sqrt(v_hat)+eps);
    if given, the operator is D^{-1/2} H D^{-1/2}.  Over n_starts random
    starts, reports the max of lam_max, the min of lam_min, and the worst
    (largest) residual bound among the reported extremes.
    """
    _, grads = _loss_grads(model, loss_fn, x, mask, y, params)
    dinv = [1.0 / d.sqrt() for d in precond] if precond is not None else None
    mv = hessian_matvec(grads, params, dinv)
    out = None
    for s in range(n_starts):
        r = lanczos_extremal(mv, params, n_iter=n_iter, tol=tol,
                             seed=seed + 101 * s)
        if out is None:
            out = dict(r)
        else:
            if r["lam_max"] > out["lam_max"]:
                out["lam_max"], out["resid_max"] = r["lam_max"], r["resid_max"]
            if r["lam_min"] < out["lam_min"]:
                out["lam_min"], out["resid_min"] = r["lam_min"], r["resid_min"]
            out["iters"] = max(out["iters"], r["iters"])
    model.train()
    model.zero_grad(set_to_none=True)
    return out


def sharpness(model, loss_fn, x, mask, y, params, n_iter=25, tol=1e-4,
              precond=None, seed=0):
    """lambda_max of the minibatch-loss Hessian w.r.t. `params`
    (backward-compatible wrapper around sharpness_ext)."""
    return sharpness_ext(model, loss_fn, x, mask, y, params, n_iter=n_iter,
                         tol=tol, precond=precond, seed=seed)["lam_max"]


def adam_precond(optimizer, params):
    """diag entries sqrt(v_hat) + eps of the Adam preconditioner for params.
    Returns None if any param has no second-moment state (e.g. SGD, or
    before the first optimizer step)."""
    d = []
    for p in params:
        st = optimizer.state.get(p, None)
        if st is None or "exp_avg_sq" not in st:
            return None
        step = st["step"]
        step = step.item() if torch.is_tensor(step) else step
        beta2 = optimizer.param_groups[0]["betas"][1]
        eps = optimizer.param_groups[0]["eps"]
        vhat = st["exp_avg_sq"] / (1 - beta2 ** step)
        d.append(vhat.sqrt() + eps)  # D_ii
    return d


def adam_inverse_precond(optimizer, params):
    """Return Q=P^{-1} entries for exact arithmetic window averages."""
    diagonal = adam_precond(optimizer, params)
    if diagonal is None:
        return None
    return [entry.reciprocal() for entry in diagonal]


def update_curvature(model, loss_fn, x, mask, y, optimizer, params):
    """SIGNED Rayleigh quotients along the actual Adam update direction
    u = m_hat/(sqrt(v_hat)+eps): raw (u'Hu/|u|^2) and D-weighted
    (u'Hu/(u'Du), D = sqrt(v_hat)+eps).  These are directional curvatures,
    NOT eigenvalues of H or of D^{-1/2} H D^{-1/2}, and can be negative;
    for the preconditioned eigenvalue see precond_sharpness."""
    u, d = [], []
    for p in params:
        st = optimizer.state.get(p, None)
        if st is None or "exp_avg" not in st:
            return None
        step = st["step"]
        step = step.item() if torch.is_tensor(step) else step
        b1, b2 = optimizer.param_groups[0]["betas"]
        eps = optimizer.param_groups[0]["eps"]
        mhat = st["exp_avg"] / (1 - b1 ** step)
        vhat = st["exp_avg_sq"] / (1 - b2 ** step)
        di = vhat.sqrt() + eps
        u.append(mhat / di)
        d.append(di)
    model.eval()
    preds, _, _, _ = model([x, mask])
    loss = loss_fn(y, preds)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    Hu = torch.autograd.grad(grads, params, grad_outputs=u)
    uHu = sum((h * ui).sum() for h, ui in zip(Hu, u)).item()
    uu = sum((ui * ui).sum() for ui in u).item()
    uDu = sum((ui * ui * di).sum() for ui, di in zip(u, d)).item()
    model.train()
    model.zero_grad(set_to_none=True)
    return {"rayq_upd": uHu / max(uu, 1e-30),
            "rayq_upd_pre": uHu / max(uDu, 1e-30)}


def all_sharpness(model, loss_fn, x, mask, y, optimizer=None, n_iter=20,
                  tol=1e-3, full=True):
    """Raw-metric Lanczos extremal curvature for the full parameter vector
    and the phi/plga blocks, plus the update-direction Rayleigh quotients.
    full=False skips the full-parameter-vector Lanczos and the
    update-direction quotients over all parameters (the full
    reorthogonalization basis costs n_iter parameter copies, prohibitive
    at large model scale); block statistics are unaffected."""
    blocks = param_blocks(model)
    allp = blocks["phi"] + blocks["plga"] + blocks["rest"]
    out = {}
    if full:
        fr = sharpness_ext(model, loss_fn, x, mask, y, allp,
                           n_iter=n_iter, tol=tol)
        out["lam_full"] = fr["lam_max"]
        out["lam_full_min"] = fr["lam_min"]
        out["lam_full_resid"] = fr["resid_max"]
    for name in ["phi", "plga"]:
        if blocks[name]:
            out[f"lam_{name}"] = sharpness_ext(
                model, loss_fn, x, mask, y, blocks[name], n_iter=n_iter,
                tol=tol)["lam_max"]
    if optimizer is not None:
        if full:
            uc = update_curvature(model, loss_fn, x, mask, y, optimizer,
                                  allp)
            if uc is not None:
                out.update(uc)
        if blocks["phi"]:
            uc_phi = update_curvature(model, loss_fn, x, mask, y, optimizer,
                                      blocks["phi"])
            if uc_phi is not None:
                out.update({k + "_phi": v for k, v in uc_phi.items()})
    return out


def precond_sharpness(model, loss_fn, batches, optimizer, n_iter=30,
                      n_starts=2, tol=1e-3, seed=0):
    """True preconditioned extremal curvature
    lambda_{max,min}(D^{-1/2} H D^{-1/2}) with D from the Adam state, for
    the full parameter vector and the non-embedding restriction, on each of
    the given (x, mask, y) batches; also the raw lam_max per batch for
    batch-replication checks.  Values are lists over batches."""
    blocks = param_blocks(model)
    allp = blocks["phi"] + blocks["plga"] + blocks["rest"]
    groups = {"full": allp, "nonemb": nonemb_params(model)}
    out = {}
    for gname, params in groups.items():
        d = adam_precond(optimizer, params)
        if d is None:
            return {}
        for bi, (x, mask, y) in enumerate(batches):
            r = sharpness_ext(model, loss_fn, x, mask, y, params,
                              n_iter=n_iter, tol=tol, precond=d,
                              n_starts=n_starts, seed=seed + 7 * bi)
            out.setdefault(f"lam_pre_max_{gname}", []).append(r["lam_max"])
            out.setdefault(f"lam_pre_min_{gname}", []).append(r["lam_min"])
            out.setdefault(f"lam_pre_resid_{gname}", []).append(
                max(r["resid_max"], r["resid_min"]))
    for bi, (x, mask, y) in enumerate(batches):
        r = sharpness_ext(model, loss_fn, x, mask, y, allp, n_iter=20,
                          tol=tol, seed=seed + 13 * bi)
        out.setdefault("lam_full_batches", []).append(r["lam_max"])
    return out


# ---------------------------------------------------------------------------
# curvature along the collapse coordinate


def _rowmap_jvp(phi, r, v):
    """J_phi(r_i) v_i for a batch of rows, differentiable w.r.t. the row-map
    parameters (double-backward trick; torch.func.jvp would detach them)."""
    r_ = r.detach().requires_grad_(True)
    y = phi(r_)
    u_dummy = torch.zeros_like(y, requires_grad=True)
    gr, = torch.autograd.grad(y, r_, grad_outputs=u_dummy, create_graph=True)
    jv, = torch.autograd.grad(gr, u_dummy, grad_outputs=v, create_graph=True)
    return jv


def _collapse_direction(model, probe_x, probe_mask, n_rows=8, seed=0):
    """The transverse statistic u_stat (mean over layers and sampled visited
    rows of ||J_phi(r) v|| for a fixed random unit v) and its parameter-space
    gradient d = grad_theta u_stat restricted to the phi block.  Returns
    (u_stat tensor with graph, d list of detached tensors, phi_params);
    (None, None, []) when the phi block is frozen.  Shared by the Rayleigh
    quotient and the finite-difference curvature probes so both use the
    identical direction."""
    blocks = param_blocks(model)
    phi_params = blocks["phi"]
    if not phi_params:
        return None, None, []
    g = torch.Generator().manual_seed(seed)
    rows = visited_rows(model, probe_x, probe_mask,
                        max_rows_per_layer=n_rows, generator=g)
    model.eval()
    layers = model.decoder.dec_layers
    u_stat = None
    for li, dec in enumerate(layers):
        phi = make_phi(dec.mha1)
        r = rows[li]
        v = torch.randn(r.shape[-1], generator=g).to(r.device, r.dtype)
        v = (v / v.norm()).expand_as(r)
        jv = _rowmap_jvp(phi, r, v)
        term = jv.pow(2).sum(-1).clamp_min(1e-30).sqrt().mean()
        u_stat = term if u_stat is None else u_stat + term
    u_stat = u_stat / len(layers)
    d = torch.autograd.grad(u_stat, phi_params, allow_unused=True)
    d = [torch.zeros_like(p) if di is None else di.detach()
         for di, p in zip(d, phi_params)]
    return u_stat, d, phi_params


def collapse_dir_curvature(model, loss_fn, x, mask, y, optimizer,
                           probe_x, probe_mask, n_rows=8, seed=0):
    """SIGNED Rayleigh quotient of the minibatch-loss Hessian along the
    parameter-space gradient direction d = grad_theta u_stat of a smooth
    transverse statistic u_stat (mean over layers and sampled visited rows
    of ||J_phi(r) v|| for a fixed random unit v), restricted to the phi
    block: the directional curvature of the training loss along the
    coordinate that the collapse statistic actually measures."""
    u_stat, d, phi_params = _collapse_direction(model, probe_x, probe_mask,
                                                n_rows=n_rows, seed=seed)
    if u_stat is None:
        return {}
    dn2 = sum((di * di).sum() for di in d).item()
    out = {"u_stat": u_stat.item()}
    if dn2 <= 0:
        model.train()
        model.zero_grad(set_to_none=True)
        return out
    _, grads = _loss_grads(model, loss_fn, x, mask, y, phi_params)
    Hd = torch.autograd.grad(grads, phi_params, grad_outputs=d)
    dHd = sum((h * di).sum() for h, di in zip(Hd, d)).item()
    out["rayq_collapse_raw"] = dHd / dn2
    if optimizer is not None:
        dpre = adam_precond(optimizer, phi_params)
        if dpre is not None:
            dDd = sum((di * di * dd).sum()
                      for di, dd in zip(d, dpre)).item()
            out["rayq_collapse_pre"] = dHd / max(dDd, 1e-30)
    model.train()
    model.zero_grad(set_to_none=True)
    return out


def fd_collapse_curvature(model, loss_fn, batches, probe_x, probe_mask,
                          n_rows=8, seed=0, hs=(1e-2, 2e-2)):
    """Finite-difference directional curvature of the minibatch loss along
    the FIXED normalized collapse direction d_hat = d/||d||,
    d = grad_theta u_stat (phi block, same direction as
    collapse_dir_curvature): [L(th + h d_hat) + L(th - h d_hat)
    - 2 L(th)] / h^2, per batch and per step size.  A directional second
    difference (not an eigenvalue and not an HVP), robust to Hessian-vector
    noise; replicated over the given (x, mask, y) batches.  Keys:
    fdcurv_collapse (list over batches at hs[0]), fdcurv_collapse_h2 (list
    over batches at hs[1]), fdcurv_dnorm (norm of the unnormalized
    direction; 0 disables the probe)."""
    u_stat, d, phi_params = _collapse_direction(model, probe_x, probe_mask,
                                                n_rows=n_rows, seed=seed)
    if u_stat is None:
        return {}
    model.zero_grad(set_to_none=True)
    dn = math.sqrt(sum((di * di).sum().item() for di in d))
    out = {"fdcurv_dnorm": dn}
    if dn <= 0:
        model.train()
        return out
    dhat = [di / dn for di in d]

    def eval_loss(x, mask, y):
        with torch.no_grad():
            preds, _, _, _ = model([x, mask])
            return loss_fn(y, preds).item()

    with torch.no_grad():
        base = [p.detach().clone() for p in phi_params]
        model.eval()
        l0 = [eval_loss(*b) for b in batches]
        for ki, h in enumerate(hs):
            key = "fdcurv_collapse" if ki == 0 else f"fdcurv_collapse_h{ki + 1}"
            for p, b0, dh in zip(phi_params, base, dhat):
                p.copy_(b0 + h * dh)
            lp = [eval_loss(*b) for b in batches]
            for p, b0, dh in zip(phi_params, base, dhat):
                p.copy_(b0 - h * dh)
            lm = [eval_loss(*b) for b in batches]
            for p, b0 in zip(phi_params, base):
                p.copy_(b0)
            out[key] = [(a + c - 2.0 * b) / (h * h)
                        for a, b, c in zip(lp, l0, lm)]
    model.train()
    return out


def gram_restricted(model, probe_x, probe_mask, optimizer=None, n_dirs=8,
                    n_rows=8, seed=0):
    """Restricted Gram geometry of the parameter-to-jet map on a declared
    subspace (the H2' measurement).  The jet evaluation used by the
    diagnostics is Psi(theta) = stacked {J_phi(r_i) v_l} over layers and
    sampled visited rows (fixed rows, fixed unit v_l per layer).  This
    computes DPsi[w_k] for n_dirs random unit phi-block directions plus,
    when Adam state exists, the current update direction restricted to the
    phi block, and reports the spectrum of the (normalized-direction) Gram
    matrix G_kl = <DPsi[w_k], DPsi[w_l]>.  Keys: gram_eigs (ascending),
    gram_cond (lam_max / max(lam_min, 0)), gram_upd_out (norm of DPsi along
    the normalized update direction; absent without Adam state)."""
    blocks = param_blocks(model)
    phi_params = blocks["phi"]
    if not phi_params:
        return {}
    g = torch.Generator().manual_seed(seed)
    rows = visited_rows(model, probe_x, probe_mask,
                        max_rows_per_layer=n_rows, generator=g)
    model.eval()
    pieces = []
    for li, dec in enumerate(model.decoder.dec_layers):
        phi = make_phi(dec.mha1)
        r = rows[li]
        v = torch.randn(r.shape[-1], generator=g).to(r.device, r.dtype)
        v = (v / v.norm()).expand_as(r)
        pieces.append(_rowmap_jvp(phi, r, v).reshape(-1))
    fvec = torch.cat(pieces)

    def theta_jvp(w):
        """(dPsi/dtheta) w via double VJP; w is a list over phi_params."""
        u_dummy = torch.zeros_like(fvec, requires_grad=True)
        s = torch.autograd.grad(fvec, phi_params, grad_outputs=u_dummy,
                                create_graph=True, retain_graph=True,
                                allow_unused=True)
        inner = sum((si * wi).sum() for si, wi in zip(s, w)
                    if si is not None)
        jvp, = torch.autograd.grad(inner, u_dummy, retain_graph=True)
        return jvp.detach()

    dirs = []
    for k in range(n_dirs):
        w = [torch.randn(p.shape, generator=g).to(p.device, p.dtype)
             for p in phi_params]
        nw = math.sqrt(sum((wi * wi).sum().item() for wi in w))
        dirs.append([wi / nw for wi in w])
    out = {}
    upd = None
    if optimizer is not None:
        ok = True
        u = []
        for p in phi_params:
            st = optimizer.state.get(p, None)
            if st is None or "exp_avg" not in st:
                ok = False
                break
            step = st["step"]
            step = step.item() if torch.is_tensor(step) else step
            b1, b2 = optimizer.param_groups[0]["betas"]
            eps = optimizer.param_groups[0]["eps"]
            mhat = st["exp_avg"] / (1 - b1 ** step)
            vhat = st["exp_avg_sq"] / (1 - b2 ** step)
            u.append(mhat / (vhat.sqrt() + eps))
        if ok:
            nu = math.sqrt(sum((ui * ui).sum().item() for ui in u))
            if nu > 0:
                upd = [ui / nu for ui in u]
    images = [theta_jvp(w) for w in dirs]
    if upd is not None:
        out["gram_upd_out"] = theta_jvp(upd).norm().item()
    M = torch.stack(images)  # [n_dirs, m]
    if not torch.isfinite(M).all():
        model.train()
        model.zero_grad(set_to_none=True)
        return {}
    G = (M @ M.T).double()
    try:
        eigs = torch.linalg.eigvalsh(G)
    except RuntimeError:
        model.train()
        model.zero_grad(set_to_none=True)
        return {}
    out["gram_eigs"] = [e.item() for e in eigs]
    lmin = max(eigs[0].item(), 0.0)
    out["gram_cond"] = (eigs[-1].item() / lmin) if lmin > 0 else float("inf")
    model.train()
    model.zero_grad(set_to_none=True)
    return out


# ---------------------------------------------------------------------------
# wave-6 additions: fine parameter blocks, preconditioned per-block extremal
# curvature, tracked signed modes, and clipping saturation


def param_blocks_fine(model):
    """Refined partition of the per-layer trainable parameters for the
    carrier search: plga (coupling block W, b_W, P, a, b_a), phi
    (metric-learner ResLayerA stack), attn (the q/k/v/output projections
    of every decoder layer), ffn (the pointwise feed-forward).  LayerNorms,
    the embedding, and the LM head are not in any fine block.  Frozen
    parameters are excluded (matching param_blocks semantics for probes
    over trainable blocks)."""
    groups = {"plga": [], "phi": [], "attn": [], "ffn": []}
    for dec in model.decoder.dec_layers:
        for p in dec.mha1.plgatt_layer.parameters():
            if p.requires_grad:
                groups["plga"].append(p)
        for p in dec.mha1.reslayerAs.parameters():
            if p.requires_grad:
                groups["phi"].append(p)
        for mod in (dec.mha1.wq, dec.mha1.wk, dec.mha1.wv, dec.mha1.dense):
            for p in mod.parameters():
                if p.requires_grad:
                    groups["attn"].append(p)
        for p in dec.ffn.parameters():
            if p.requires_grad:
                groups["ffn"].append(p)
    return groups


def adam_precond_live(optimizer, params, quantile=0.5):
    """Adam preconditioner entries for a block together with a 0/1 LIVE
    mask selecting the coordinates whose bias-corrected second moment is
    at or above the given within-block quantile.  Motivation: the
    unrestricted top preconditioned eigenvector of a block is dominated
    by near-dead coordinates (v_hat orders of magnitude below typical,
    so D^{-1/2} amplifies them enormously) that carry a negligible share
    of the realized update power; the live restriction is the block-level
    analogue of the non-embedding restriction used at model level.
    Returns (d, mask) lists or None without Adam state."""
    d = adam_precond(optimizer, params)
    if d is None:
        return None
    vhat = []
    for p in params:
        st = optimizer.state[p]
        step = st["step"]
        step = step.item() if torch.is_tensor(step) else step
        beta2 = optimizer.param_groups[0]["betas"][1]
        vhat.append(st["exp_avg_sq"] / (1 - beta2 ** step))
    flat = _flat(vhat)
    if flat.numel() <= (1 << 24):
        thr = torch.quantile(flat.float(), quantile).item()
    else:
        # torch.quantile rejects very large inputs (the 110M-scale
        # blocks); kthvalue gives the same order statistic without the
        # interpolation, which is immaterial for a mask threshold
        k = max(1, int(quantile * flat.numel()))
        thr = flat.float().kthvalue(k).values.item()
    mask = [(v >= thr).to(v.dtype) for v in vhat]
    return d, mask


def clip_fractions(model, clip):
    """Per-fine-block fraction of gradient coordinates at or beyond the
    value-clipping level, measured BEFORE clip_grad_value_ is applied.
    Reported next to any boundary-adjacent statistic because entrywise
    clipping breaks every linear-recurrence threshold assumption.
    Returns {} when clipping is off."""
    if clip <= 0:
        return {}
    out = {}
    for name, plist in param_blocks_fine(model).items():
        tot, sat = 0, 0
        for p in plist:
            if p.grad is None:
                continue
            g = p.grad.detach()
            tot += g.numel()
            sat += (g.abs() >= clip).sum().item()
        out[f"clip_frac_{name}"] = (sat / tot) if tot else 0.0
    return out


def lanczos_top_vector(matvec, template, n_iter=20, tol=1e-3, seed=0,
                       v0=None):
    """Top eigenpair of a symmetric operator by Lanczos with full
    reorthogonalization and a stored basis: returns
    (lam_max, top vector as a list shaped like template, residual bound,
    iters), or (nan, None, nan, 0) on a non-finite operator state.
    v0: optional warm-start direction (list of tensors, need not be
    normalized); falls back to a seeded random start."""
    device = template[0].device
    if v0 is not None:
        v = [vi.detach().clone() for vi in v0]
    else:
        g = torch.Generator(device=device).manual_seed(seed)
        v = [torch.randn(p.shape, generator=g, device=device, dtype=p.dtype)
             for p in template]
    nv = _flat(v).norm()
    if not torch.isfinite(nv) or nv.item() <= 0:
        return float("nan"), None, float("nan"), 0
    v = [vi / nv for vi in v]
    V = [v]
    alphas, betas = [], []
    beta_next = 0.0
    for j in range(n_iter):
        w = matvec(V[j])
        if not all(torch.isfinite(wi).all() for wi in w):
            return float("nan"), None, float("nan"), 0
        alpha = sum((wi * vi).sum() for wi, vi in zip(w, V[j])).item()
        alphas.append(alpha)
        w = [wi - alpha * vi for wi, vi in zip(w, V[j])]
        if j > 0:
            w = [wi - betas[-1] * ui for wi, ui in zip(w, V[j - 1])]
        for u in V:
            proj = sum((wi * ui).sum() for wi, ui in zip(w, u))
            w = [wi - proj * ui for wi, ui in zip(w, u)]
        beta_next = _flat(w).norm().item()
        if beta_next < 1e-10:
            beta_next = 0.0
            break
        if tol > 0 and j >= 2:
            lmax, lmin, rmax, rmin = _ritz(alphas, betas, beta_next)
            scale = max(abs(lmax), abs(lmin), 1e-30)
            if rmax < tol * scale:
                break
        if j < n_iter - 1:
            betas.append(beta_next)
            V.append([wi / beta_next for wi in w])
    if not (all(math.isfinite(a) for a in alphas)
            and all(math.isfinite(b) for b in betas)
            and math.isfinite(beta_next)):
        return float("nan"), None, float("nan"), 0
    k = len(alphas)
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    for j, b in enumerate(betas[: k - 1]):
        T[j, j + 1] = b
        T[j + 1, j] = b
    try:
        evals, evecs = torch.linalg.eigh(T)
    except RuntimeError:
        return float("nan"), None, float("nan"), 0
    s = evecs[:, -1]
    resid = abs(beta_next * s[-1].item())
    top = None
    for j in range(min(k, len(V))):
        c = s[j].item()
        if top is None:
            top = [c * vi for vi in V[j]]
        else:
            top = [t + c * vi for t, vi in zip(top, V[j])]
    nt = _flat(top).norm()
    if not torch.isfinite(nt) or nt.item() <= 0:
        return float("nan"), None, float("nan"), 0
    top = [t / nt for t in top]
    return evals[-1].item(), top, resid, k


def block_sharpness(model, loss_fn, x, mask, y, optimizer=None,
                    blocks=None, n_iter=15, tol=1e-3, seed=0):
    """Raw, Adam-preconditioned, and Gauss--Newton curvature per
    fine block, one sharpness batch, one start (the dense-cadence
    carrier-search probe; the sparse multi-start replicated measurement
    remains precond_sharpness).  Keys per block <n>: lam_blk_<n> (raw
    lambda_max), lam_pre_blk_<n> / lam_pre_blk_min_<n> /
    lam_pre_blk_resid_<n> (preconditioned, present only with Adam
    second-moment state)."""
    if blocks is None:
        blocks = param_blocks_fine(model)
    out = {}
    for name, params in blocks.items():
        if not params:
            continue
        loss, preds = _loss_graph(model, loss_fn, x, mask, y)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        r = lanczos_extremal(hessian_matvec(grads, params), params,
                             n_iter=n_iter, tol=tol, seed=seed)
        out[f"lam_blk_{name}"] = r["lam_max"]
        if optimizer is not None:
            dm = adam_precond_live(optimizer, params)
            if dm is not None:
                d, live_mask = dm
                dinv = [1.0 / di.sqrt() for di in d]
                rp = lanczos_extremal(
                    hessian_matvec(grads, params, dinv), params,
                    n_iter=n_iter, tol=tol, seed=seed)
                out[f"lam_pre_blk_{name}"] = rp["lam_max"]
                out[f"lam_pre_blk_min_{name}"] = rp["lam_min"]
                out[f"lam_pre_blk_resid_{name}"] = max(rp["resid_max"],
                                                       rp["resid_min"])
                # live restriction: preconditioned operator confined to
                # the coordinates with above-median second moment
                dinv_live = [iv * m for iv, m in zip(dinv, live_mask)]
                rl = lanczos_extremal(
                    hessian_matvec(grads, params, dinv_live), params,
                    n_iter=n_iter, tol=tol, seed=seed)
                out[f"lam_pre_live_{name}"] = rl["lam_max"]
                out[f"lam_pre_live_min_{name}"] = rl["lam_min"]
                out[f"lam_pre_live_resid_{name}"] = max(rl["resid_max"],
                                                        rl["resid_min"])
                # Primary physical stress: identical Adam metric and live
                # restriction, with the generalized Gauss--Newton operator.
                rg = lanczos_extremal(
                    gauss_newton_matvec(loss, preds, params, dinv_live),
                    params, n_iter=n_iter, tol=tol, seed=seed)
                out[f"gn_pre_live_{name}"] = rg["lam_max"]
                out[f"gn_pre_live_min_{name}"] = rg["lam_min"]
                out[f"gn_pre_live_resid_{name}"] = max(
                    rg["resid_max"], rg["resid_min"])
        del grads
        model.train()
        model.zero_grad(set_to_none=True)
    return out


def mode_track(model, loss_fn, x, mask, y, optimizer, params,
               prev_vec=None, n_iter=20, tol=1e-3, seed=0,
               variant="pre_live", operator="hessian",
               inverse_preconditioner_avg=None):
    """Tracked coherent signed mode of a parameter block: the top
    eigenvector of the block operator selected by `variant`, mapped to a
    PARAMETER-SPACE oscillation direction and normalized, overlap-matched
    against the previously tracked vector.  Variants: "raw" (H_blk;
    the mode is the eigenvector itself), "pre_live" (the live-restricted
    preconditioned operator M D^{-1/2} H_blk D^{-1/2} M with M the
    above-median second-moment mask; the parameter-space direction is
    x = M D^{-1/2} y, the direction the frozen-preconditioner recurrence
    actually moves; falls back to raw without Adam state).  The
    unrestricted preconditioned top vector is deliberately NOT a tracked
    family: it is dominated by near-dead coordinates carrying a
    negligible share of the realized update power.  Returns (stats dict,
    new parameter-space unit vector list or None).  Stats: mode_lam
    (eigenvalue in the probed metric), mode_resid, mode_overlap
    (|<x_new, x_prev>|; absent on the first probe), mode_upd_proj
    (|<u_hat, x_new>| with u the current Adam update direction; absent
    without Adam state).  The sign of the returned vector is aligned
    with prev_vec so the per-step signed projection series is continuous
    across refreshes.  With ``operator="gauss_newton"`` the tracked
    operator is generalized Gauss--Newton and the function also reports
    the full-Hessian minus Gauss--Newton directional residual in exactly
    the same metric and direction."""
    if operator not in ("hessian", "gauss_newton"):
        raise ValueError("operator must be 'hessian' or 'gauss_newton'")
    loss, preds = _loss_graph(model, loss_fn, x, mask, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dinv = None
    if variant == "pre_live" and optimizer is not None:
        dm = adam_precond_live(optimizer, params)
        if dm is not None:
            d, msk = dm
            if inverse_preconditioner_avg is None:
                dinv = [m / di.sqrt() for di, m in zip(d, msk)]
            else:
                if len(inverse_preconditioner_avg) != len(params):
                    raise ValueError("inverse-preconditioner average shape")
                if any(q.shape != p.shape for q, p in
                       zip(inverse_preconditioner_avg, params)):
                    raise ValueError("inverse-preconditioner tensor shape")
                dinv = [m * q.clamp_min(0).sqrt() for q, m in
                        zip(inverse_preconditioner_avg, msk)]
    v0 = None
    if prev_vec is not None:
        # warm start in operator coordinates: y = D^{1/2} x on the live
        # set (the dead components of prev_vec are dropped; a random
        # start takes over if nothing survives)
        if dinv is not None:
            v0 = [torch.where(iv > 0, xi / iv.clamp_min(1e-30),
                              torch.zeros_like(xi))
                  for xi, iv in zip(prev_vec, dinv)]
            if _flat(v0).norm().item() <= 0:
                v0 = None
        else:
            v0 = prev_vec
    full_mv = hessian_matvec(grads, params, dinv)
    primary_mv = (gauss_newton_matvec(loss, preds, params, dinv)
                  if operator == "gauss_newton" else full_mv)
    lam, y_top, resid, _ = lanczos_top_vector(
        primary_mv, params, n_iter=n_iter,
        tol=tol, seed=seed, v0=v0)
    if y_top is None:
        del grads
        model.train()
        model.zero_grad(set_to_none=True)
        return {"mode_lam": lam, "mode_resid": resid}, None
    gn_residual = None
    full_on_mode = None
    if operator == "gauss_newton":
        hy = full_mv(y_top)
        gy = primary_mv(y_top)
        full_on_mode = sum((a * b).sum()
                           for a, b in zip(y_top, hy)).item()
        gn_residual = sum((a * (b - c)).sum()
                          for a, b, c in zip(y_top, hy, gy)).item()
    del grads
    model.train()
    model.zero_grad(set_to_none=True)
    if dinv is not None:
        xvec = [yi * di for yi, di in zip(y_top, dinv)]
        nx = _flat(xvec).norm()
        if not torch.isfinite(nx) or nx.item() <= 0:
            return {"mode_lam": lam, "mode_resid": resid}, None
        xvec = [xi / nx for xi in xvec]
    else:
        xvec = y_top
    out = {"mode_lam": lam, "mode_resid": resid}
    if operator == "gauss_newton":
        out["mode_full_hessian_on_gn"] = full_on_mode
        out["mode_gn_residual"] = gn_residual
    if prev_vec is not None:
        ov = sum((a * b).sum() for a, b in zip(xvec, prev_vec)).item()
        if ov < 0:  # fix the sign so the projection series is continuous
            xvec = [-xi for xi in xvec]
            ov = -ov
        out["mode_overlap"] = ov
    if optimizer is not None:
        u, ok = [], True
        for p in params:
            st = optimizer.state.get(p, None)
            if st is None or "exp_avg" not in st:
                ok = False
                break
            step = st["step"]
            step = step.item() if torch.is_tensor(step) else step
            b1, b2 = optimizer.param_groups[0]["betas"]
            eps = optimizer.param_groups[0]["eps"]
            mhat = st["exp_avg"] / (1 - b1 ** step)
            vhat = st["exp_avg_sq"] / (1 - b2 ** step)
            u.append(mhat / (vhat.sqrt() + eps))
        if ok:
            nu = _flat(u).norm().item()
            if nu > 0:
                out["mode_upd_proj"] = abs(
                    sum((ui * xi).sum()
                        for ui, xi in zip(u, xvec)).item()) / nu
    return out, [xi.detach() for xi in xvec]


class DynMode:
    """Streaming tracker of the dominant coherent direction of a block's
    realized per-step updates: sign-corrected Oja/EMA iteration
    v <- normalize((1-alpha) v + alpha sgn(<dtheta, v>) dtheta/|dtheta|).
    For updates dtheta_t = c_t u + noise with c_t of alternating sign
    (a carried oscillation), the sign correction makes every increment
    push v toward the same +-u, so v converges to the oscillation
    direction; for pure drift it tracks the drift direction.  Unlike
    top-Hessian-eigenvector tracking (measured to be incoherent across
    refreshes and to carry ~1e-5 of the update power at this scale),
    this estimand cannot miss a dominant coherent update direction: if
    the captured power fraction <dtheta, v>^2/|dtheta|^2 stays at the
    1/dim floor, no such direction exists, which is itself the decisive
    carrier-search outcome.  alpha = 0.05 (about a 20-step memory) is
    the declared constant."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.v = None

    @torch.no_grad()
    def update(self, cur, prev):
        """Consume one realized update (cur - prev).  Returns the signed
        projection of the update onto the direction held BEFORE this
        call (None on the initializing call), then refreshes the
        direction."""
        d = [p - q for p, q in zip(cur, prev)]
        nd = _flat(d).norm().item()
        if not math.isfinite(nd) or nd <= 0:
            return None
        dh = [di / nd for di in d]
        if self.v is None:
            self.v = dh
            return None
        c = sum((di * vi).sum() for di, vi in zip(d, self.v)).item()
        s = 1.0 if c >= 0 else -1.0
        a = self.alpha
        vnew = [(1 - a) * vi + a * s * dhi
                for vi, dhi in zip(self.v, dh)]
        nv = _flat(vnew).norm().item()
        if math.isfinite(nv) and nv > 0:
            self.v = [vi / nv for vi in vnew]
        return c


@torch.no_grad()
def signed_projection(cur, prev, vec):
    """Signed projection <cur - prev, vec> of a per-step parameter change
    onto a tracked unit mode vector (all lists over the same block)."""
    return sum(((p - q) * v).sum()
               for p, q, v in zip(cur, prev, vec)).item()


# ---------------------------------------------------------------------------
# comparative diagnostics (attention entropy, representation rank, coupling)


@torch.no_grad()
def attention_entropy(model, x, mask):
    """Mean softmax attention entropy per layer on a probe batch, in nats.
    The attention probabilities are recomputed from the logged pre-mask
    scores Ep (att_weights index 5) with the same mask and softmax as the
    forward pass.  Comparative diagnostic for the attention-entropy-collapse
    literature; not part of the collapse criterion."""
    model.eval()
    _, _, att_weights, _ = model([x, mask])
    ents = []
    for aw in att_weights:
        ep = aw[5]
        logits = ep + mask * -1e9
        p = F.softmax(logits, dim=-1)
        ent = -(p * (p.clamp_min(1e-12)).log()).sum(-1)
        ents.append(ent.mean().item())
    model.train()
    return ents


@torch.no_grad()
def repr_effrank(model, x, mask):
    """Effective rank (exponential of the singular-value entropy) of the
    centered final decoder representation on a probe batch.  Comparative
    diagnostic for the token/representation rank-collapse literature."""
    model.eval()
    _, dec_out, _, _ = model([x, mask])
    X = dec_out.reshape(-1, dec_out.shape[-1])
    X = X - X.mean(0, keepdim=True)
    s = torch.linalg.svdvals(X)
    p = (s * s) / (s * s).sum().clamp_min(1e-30)
    model.train()
    return math.exp(-(p * p.clamp_min(1e-12).log()).sum().item())


@torch.no_grad()
def coupling_norms(model):
    """Per-layer mean Frobenius norm over heads of the PLGA coupling tensor
    a (and of W): the coupling-amplitude coordinate of the second route to
    operator invariance."""
    a_norms, w_norms = [], []
    for dec in model.decoder.dec_layers:
        pl = dec.mha1.plgatt_layer
        a_norms.append(pl.alst.flatten(1).norm(dim=-1).mean().item())
        w_norms.append(pl.Wlst.flatten(1).norm(dim=-1).mean().item())
    return {"a_norm_layers": a_norms, "w_norm_layers": w_norms}
