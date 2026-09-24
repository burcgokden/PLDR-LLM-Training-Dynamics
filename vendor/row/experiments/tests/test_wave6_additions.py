"""Tests of the wave-6 instrumentation: fine parameter blocks, the
Lanczos top-vector routine, per-block raw/preconditioned curvature, the
tracked signed mode with overlap matching and sign continuity, the
mode-amplitude intervention algebra, clipping saturation, exact gradient
accumulation, and the generalized finite-difference step grid."""

import math

import pytest
import torch
from torch import nn

import instrument
from pldr_model_v510 import PLDR_Model
from train_run import masked_loss_function, masked_loss_function64, \
    create_masks

LAYERS, HEADS, DK, VOCAB, SEQ, BATCH = 1, 2, 8, 64, 12, 2
D_MODEL = HEADS * DK


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = PLDR_Model(num_layers=LAYERS, d_model=D_MODEL, num_heads=HEADS,
                       dff=int(math.ceil(D_MODEL * 4 * 2 / 3)),
                       input_vocab_size=VOCAB, A_dff=8, num_reslayerA=2,
                       num_denseA=2, max_seq_len=64, device=device)
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    y = torch.randint(1, VOCAB, (BATCH, SEQ))
    mask = create_masks(x, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            eps=1e-5, weight_decay=0.1)
    preds, _, _, _ = model([x, mask])
    loss = masked_loss_function(y, preds)
    opt.zero_grad()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return model, opt, x, mask, y, device


def test_fine_blocks_partition(setup):
    """The fine blocks are disjoint, and together with the LayerNorms,
    the embedding, and the LM head they cover every trainable
    parameter."""
    model = setup[0]
    fine = instrument.param_blocks_fine(model)
    ids = [id(p) for plist in fine.values() for p in plist]
    assert len(ids) == len(set(ids)), "fine blocks overlap"
    covered = set(ids)
    ln_emb_head = set()
    for dec in model.decoder.dec_layers:
        for p in dec.mha1.layernorm1.parameters():
            ln_emb_head.add(id(p))
        for p in dec.layernorm1.parameters():
            ln_emb_head.add(id(p))
        for p in dec.layernorm2.parameters():
            ln_emb_head.add(id(p))
    for p in model.decoder.embedding.parameters():
        ln_emb_head.add(id(p))
    for p in model.decoder.layernorm1.parameters():
        ln_emb_head.add(id(p))
    for p in model.final_layer.parameters():
        ln_emb_head.add(id(p))
    allp = {id(p) for p in model.parameters() if p.requires_grad}
    assert covered | ln_emb_head == allp
    assert covered & ln_emb_head == set()


def test_lanczos_top_vector_dense():
    """Recovers the top eigenpair of a known symmetric matrix, and a
    warm start converges at least as fast."""
    torch.manual_seed(1)
    n = 24
    Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
    evals = torch.linspace(-3.0, 5.0, n, dtype=torch.float64)
    A = (Q * evals) @ Q.T
    vtrue = Q[:, -1]

    def mv(v):
        return [(A @ v[0].double()).to(v[0].dtype)]

    template = [torch.zeros(n, dtype=torch.float64)]
    lam, vec, resid, iters = instrument.lanczos_top_vector(
        mv, template, n_iter=n, tol=1e-9, seed=0)
    assert abs(lam - 5.0) < 1e-6
    assert abs(vec[0].double() @ vtrue) > 1 - 1e-6
    lam2, vec2, _, iters2 = instrument.lanczos_top_vector(
        mv, template, n_iter=n, tol=1e-9, v0=[vtrue.clone()])
    assert abs(lam2 - 5.0) < 1e-6
    assert iters2 <= iters


def test_block_sharpness_keys(setup):
    model, opt, x, mask, y, _ = setup
    out = instrument.block_sharpness(model, masked_loss_function, x, mask,
                                     y, optimizer=opt, n_iter=8)
    for name in ("plga", "phi", "attn", "ffn"):
        assert math.isfinite(out[f"lam_blk_{name}"]), name
        assert math.isfinite(out[f"lam_pre_blk_{name}"]), name
        assert (out[f"lam_pre_blk_{name}"]
                >= out[f"lam_pre_blk_min_{name}"]), name
        assert math.isfinite(out[f"lam_pre_live_{name}"]), name
    # without optimizer state, only raw keys appear
    out2 = instrument.block_sharpness(model, masked_loss_function, x, mask,
                                      y, optimizer=None, n_iter=6)
    assert "lam_blk_plga" in out2
    assert "lam_pre_blk_plga" not in out2


def test_live_restriction_interlaces_dense():
    """On a dense operator run to full Krylov dimension, the top
    eigenvalue of the masked compression M A M never exceeds the
    unrestricted top eigenvalue (Rayleigh quotient over a subspace)."""
    torch.manual_seed(4)
    n = 16
    B = torch.randn(n, n, dtype=torch.float64)
    A = B @ B.T
    m = (torch.arange(n) % 2 == 0).to(torch.float64)

    def mv_full(v):
        return [A @ v[0]]

    def mv_live(v):
        return [m * (A @ (m * v[0]))]

    tpl = [torch.zeros(n, dtype=torch.float64)]
    rf = instrument.lanczos_extremal(mv_full, tpl, n_iter=n, tol=0.0)
    rl = instrument.lanczos_extremal(mv_live, tpl, n_iter=n, tol=0.0)
    exact_full = torch.linalg.eigvalsh(A)[-1].item()
    Alive = (m[:, None] * A * m[None, :])
    exact_live = torch.linalg.eigvalsh(Alive)[-1].item()
    assert abs(rf["lam_max"] - exact_full) < 1e-8 * max(1, exact_full)
    assert abs(rl["lam_max"] - exact_live) < 1e-8 * max(1, exact_full)
    assert exact_live <= exact_full + 1e-12


def test_mode_track_overlap_and_sign(setup):
    model, opt, x, mask, y, _ = setup
    fine = instrument.param_blocks_fine(model)
    st1, v1 = instrument.mode_track(model, masked_loss_function, x, mask,
                                    y, opt, fine["plga"], n_iter=10)
    assert v1 is not None
    n1 = math.sqrt(sum((vi * vi).sum().item() for vi in v1))
    assert abs(n1 - 1.0) < 1e-4
    assert math.isfinite(st1["mode_lam"])
    assert "mode_upd_proj" in st1 and 0.0 <= st1["mode_upd_proj"] <= 1.0 + 1e-6
    # second call from the flipped previous vector: overlap reported,
    # and the returned vector is sign-aligned with what was passed in
    flipped = [-vi for vi in v1]
    st2, v2 = instrument.mode_track(model, masked_loss_function, x, mask,
                                    y, opt, fine["plga"],
                                    prev_vec=flipped, n_iter=10)
    assert v2 is not None
    assert st2["mode_overlap"] > 0.9
    ip = sum((a * b).sum().item() for a, b in zip(v2, flipped))
    assert ip > 0
    # the live mode is supported on the live coordinate set only
    dm = instrument.adam_precond_live(opt, fine["plga"])
    assert dm is not None
    _, msk = dm
    dead_mass = sum((vi * (1 - m)).pow(2).sum().item()
                    for vi, m in zip(v1, msk))
    assert dead_mass < 1e-10
    # the raw family works and is a unit parameter-space vector
    st3, v3 = instrument.mode_track(model, masked_loss_function, x, mask,
                                    y, opt, fine["plga"], n_iter=10,
                                    variant="raw")
    assert v3 is not None
    n3 = math.sqrt(sum((vi * vi).sum().item() for vi in v3))
    assert abs(n3 - 1.0) < 1e-4
    assert st3["mode_lam"] <= out_raw_bound(model, opt, x, mask, y)


def out_raw_bound(model, opt, x, mask, y):
    """Loose sanity bound: the raw plga top eigenvalue plus tolerance."""
    fine = instrument.param_blocks_fine(model)
    r = instrument.block_sharpness(model, masked_loss_function, x, mask,
                                   y, optimizer=None,
                                   blocks={"plga": fine["plga"]}, n_iter=10)
    return r["lam_blk_plga"] * 1.1 + 1e-6


def test_dyn_mode_tracker_finds_planted_oscillation():
    """The sign-corrected streaming tracker converges onto a planted
    alternating direction and reports its signed amplitude."""
    torch.manual_seed(5)
    n = 40
    u = torch.randn(n)
    u = u / u.norm()
    trk = instrument.DynMode(alpha=0.1)
    prev = [torch.zeros(n)]
    amp = 0.5
    cs = []
    for t in range(120):
        d = ((-1.0) ** t) * amp * u + 0.02 * torch.randn(n)
        cur = [prev[0] + d]
        c = trk.update(cur, prev)
        if c is not None:
            cs.append(c)
        prev = cur
    align = abs((trk.v[0] * u).sum().item())
    assert align > 0.95
    tail = cs[-40:]
    flips = sum(1 for i in range(1, len(tail))
                if (tail[i] > 0) != (tail[i - 1] > 0)) / (len(tail) - 1)
    assert flips > 0.9
    assert abs(sum(abs(c) for c in tail) / len(tail) - amp) < 0.1
    # pure drift: tracks the drift direction with constant sign
    trk2 = instrument.DynMode(alpha=0.1)
    prev = [torch.zeros(n)]
    cs2 = []
    for t in range(80):
        cur = [prev[0] + amp * u + 0.02 * torch.randn(n)]
        c = trk2.update(cur, prev)
        if c is not None:
            cs2.append(c)
        prev = cur
    assert abs((trk2.v[0] * u).sum().item()) > 0.95
    assert all(c > 0 for c in cs2[-40:])


def test_signed_projection_and_intervention_algebra():
    torch.manual_seed(2)
    prev = [torch.randn(5), torch.randn(3)]
    vec = [torch.randn(5), torch.randn(3)]
    nv = math.sqrt(sum((v * v).sum().item() for v in vec))
    vec = [v / nv for v in vec]
    delta = [torch.randn(5), torch.randn(3)]
    cur = [p + d for p, d in zip(prev, delta)]
    c = instrument.signed_projection(cur, prev, vec)
    expect = sum((d * v).sum().item() for d, v in zip(delta, vec))
    assert abs(c - expect) < 1e-6
    # applying theta += (gamma - 1) c v rescales the along-mode component
    gamma = 0.5
    cur2 = [p + (gamma - 1.0) * c * v for p, v in zip(cur, vec)]
    c2 = instrument.signed_projection(cur2, prev, vec)
    assert abs(c2 - gamma * c) < 1e-6


def test_clip_fractions(setup):
    model = setup[0]
    fine = instrument.param_blocks_fine(model)
    for plist in fine.values():
        for p in plist:
            p.grad = torch.zeros_like(p)
    # saturate exactly the plga block
    for p in fine["plga"]:
        p.grad.fill_(2.0)
    out = instrument.clip_fractions(model, clip=1.0)
    assert out["clip_frac_plga"] == 1.0
    assert out["clip_frac_attn"] == 0.0
    assert instrument.clip_fractions(model, clip=0.0) == {}
    for plist in fine.values():
        for p in plist:
            p.grad = None


def test_accum_identity(setup):
    """Backpropagating per-micro masked SUMS and dividing by the total
    mask count reproduces the exact large-batch gradient of the masked
    mean (the E5 identity control)."""
    model, _, _, _, _, device = setup
    torch.manual_seed(3)
    xb = torch.randint(1, VOCAB, (8, SEQ))
    yb = torch.randint(1, VOCAB, (8, SEQ))
    mb = create_masks(xb, device)
    params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad(set_to_none=True)
    preds, _, _, _ = model([xb, mb])
    masked_loss_function(yb, preds).backward()
    ref = [p.grad.detach().clone() for p in params]

    model.zero_grad(set_to_none=True)
    tot_cnt = 0.0
    for a in range(4):
        xm, ym = xb[2 * a:2 * a + 2], yb[2 * a:2 * a + 2]
        mm = create_masks(xm, device)
        preds, _, _, _ = model([xm, mm])
        msk = torch.ne(ym, 0)
        yp = torch.permute(preds, (0, 2, 1))
        l_ = nn.CrossEntropyLoss(reduction="none")(yp, ym)
        (l_ * msk.to(l_.dtype)).sum().backward()
        tot_cnt += msk.sum().item()
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.grad.div_(tot_cnt)
    for p, r in zip(params, ref):
        got = p.grad if p.grad is not None else torch.zeros_like(r)
        assert torch.allclose(got, r, atol=2e-6, rtol=2e-5)
    model.zero_grad(set_to_none=True)


def test_fd_hs_grid_and_fp64(setup):
    model, _, x, mask, y, _ = setup
    out = instrument.fd_collapse_curvature(
        model, masked_loss_function64, [(x, mask, y)], x, mask,
        hs=(5e-3, 1e-2, 2e-2))
    assert "fdcurv_collapse" in out
    assert "fdcurv_collapse_h2" in out
    assert "fdcurv_collapse_h3" in out
    assert out["fdcurv_dnorm"] > 0
    for k in ("fdcurv_collapse", "fdcurv_collapse_h2",
              "fdcurv_collapse_h3"):
        assert len(out[k]) == 1 and math.isfinite(out[k][0])
