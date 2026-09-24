"""Tests for the wave-5 additions: the random-single-position loss, the
genuinely fixed AdamW preconditioner, the pulse/validation plumbing, and the
new instrumentation probes (finite-difference collapse curvature, restricted
jet Gram, attention entropy, representation effective rank, coupling norms).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import instrument  # noqa: E402
from train_run import adamw_fixed_step, create_masks, rand1_targets  # noqa: E402
from pldr_model_v510 import PLDR_Model  # noqa: E402

EXP_DIR = os.path.join(os.path.dirname(__file__), "..")


# --------------------------------------------------------------------------
# rand1 loss semantics


def test_rand1_targets_one_nonfinal_position():
    y = torch.arange(1, 1 + 4 * 9).reshape(4, 9)
    yz = rand1_targets(y, seed=7)
    nz = (yz != 0)
    assert nz.sum(dim=1).tolist() == [1, 1, 1, 1]  # one position/sequence
    assert not nz[:, -1].any()  # never the final position
    rows, cols = nz.nonzero(as_tuple=True)
    assert torch.equal(yz[rows, cols], y[rows, cols])  # label unchanged


def test_rand1_targets_deterministic_and_seed_dependent():
    y = torch.randint(1, 100, (8, 12))
    a = rand1_targets(y, seed=3)
    b = rand1_targets(y, seed=3)
    c = rand1_targets(y, seed=4)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)  # (8 positions from 11 choices: collision
    # of the whole vector across seeds is astronomically unlikely)


# --------------------------------------------------------------------------
# fixed-denominator AdamW step


def test_adamw_fixed_step_matches_reference_and_freezes_v():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(5))
    opt = torch.optim.AdamW([p], lr=1e-2, betas=(0.9, 0.95), eps=1e-5,
                            weight_decay=0.1)
    # three normal steps to populate state
    for _ in range(3):
        opt.zero_grad()
        (p ** 2).sum().backward()
        opt.step()
    st = opt.state[p]
    t0 = st["step"].item() if torch.is_tensor(st["step"]) else st["step"]
    denom = (st["exp_avg_sq"] / (1 - 0.95 ** t0)).sqrt().add_(1e-5)
    frozen = {id(p): denom.clone()}
    v_before = st["exp_avg_sq"].clone()

    opt.zero_grad()
    (p ** 2).sum().backward()
    g = p.grad.clone()
    m_prev = st["exp_avg"].clone()
    p_prev = p.detach().clone()
    with torch.no_grad():
        adamw_fixed_step(opt, frozen)

    # second moment untouched; step counter advanced
    assert torch.equal(st["exp_avg_sq"], v_before)
    t1 = st["step"].item() if torch.is_tensor(st["step"]) else st["step"]
    assert t1 == t0 + 1
    # reference update: m <- b1 m + (1-b1) g; p <- p(1-lr wd) - lr m_hat/denom
    m_ref = 0.9 * m_prev + 0.1 * g
    p_ref = p_prev * (1 - 1e-2 * 0.1) - 1e-2 * (m_ref / (1 - 0.9 ** t1)) / denom
    assert torch.allclose(st["exp_avg"], m_ref, atol=1e-7)
    assert torch.allclose(p.detach(), p_ref, atol=1e-7)


def test_adamw_fixed_step_denominator_constant_across_steps():
    torch.manual_seed(1)
    p = torch.nn.Parameter(torch.randn(4))
    opt = torch.optim.AdamW([p], lr=1e-2, betas=(0.9, 0.95), eps=1e-5,
                            weight_decay=0.0)
    opt.zero_grad()
    (p ** 2).sum().backward()
    opt.step()
    st = opt.state[p]
    frozen = {id(p): (st["exp_avg_sq"] / (1 - 0.95)).sqrt().add_(1e-5)}
    d0 = frozen[id(p)].clone()
    v0 = st["exp_avg_sq"].clone()
    for _ in range(4):
        opt.zero_grad()
        (torch.sin(p) ** 2).sum().backward()
        with torch.no_grad():
            adamw_fixed_step(opt, frozen)
    assert torch.equal(frozen[id(p)], d0)
    assert torch.equal(st["exp_avg_sq"], v0)


# --------------------------------------------------------------------------
# new instrumentation probes on the tiny CPU model

LAYERS, HEADS, DK, VOCAB, SEQ, BATCH = 1, 2, 8, 64, 12, 2


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = PLDR_Model(num_layers=LAYERS, d_model=HEADS * DK,
                       num_heads=HEADS, dff=40, input_vocab_size=VOCAB,
                       A_dff=12, num_reslayerA=2, num_denseA=2,
                       max_seq_len=64, device=torch.device("cpu"))
    x = torch.randint(1, VOCAB, (BATCH, SEQ))
    inp, y = x[:, :-1], x[:, 1:]
    mask = create_masks(inp, torch.device("cpu"))
    return model, inp, mask, y


def _mll(y_true, y_pred):
    mask = torch.ne(y_true, 0)
    y_pred = torch.permute(y_pred, (0, 2, 1))
    loss_ = torch.nn.CrossEntropyLoss(reduction="none")(y_pred, y_true)
    mask = mask.to(loss_.dtype)
    return torch.sum(loss_ * mask) / torch.sum(mask)


def test_fd_collapse_curvature_keys_and_restore(tiny):
    model, inp, mask, y = tiny
    before = [p.detach().clone() for p in model.parameters()]
    out = instrument.fd_collapse_curvature(
        model, _mll, [(inp, mask, y), (inp, mask, y)], inp, mask)
    assert out["fdcurv_dnorm"] > 0
    assert len(out["fdcurv_collapse"]) == 2
    assert len(out["fdcurv_collapse_h2"]) == 2
    assert all(np.isfinite(v) for v in out["fdcurv_collapse"])
    # identical batches must give identical values
    assert out["fdcurv_collapse"][0] == pytest.approx(
        out["fdcurv_collapse"][1])
    # parameters restored exactly
    for p, b in zip(model.parameters(), before):
        assert torch.equal(p.detach(), b)


def test_gram_restricted_spectrum(tiny):
    model, inp, mask, y = tiny
    out = instrument.gram_restricted(model, inp, mask, n_dirs=4)
    assert len(out["gram_eigs"]) == 4
    assert all(np.isfinite(e) for e in out["gram_eigs"])
    assert out["gram_eigs"] == sorted(out["gram_eigs"])
    assert out["gram_eigs"][0] >= -1e-8
    assert out["gram_cond"] >= 1.0


def test_comparative_diagnostics(tiny):
    model, inp, mask, y = tiny
    ents = instrument.attention_entropy(model, inp, mask)
    assert len(ents) == LAYERS
    assert all(0.0 <= e <= float(np.log(SEQ)) + 1e-6 for e in ents)
    er = instrument.repr_effrank(model, inp, mask)
    assert 1.0 <= er <= HEADS * DK + 1e-6
    cn = instrument.coupling_norms(model)
    assert len(cn["a_norm_layers"]) == LAYERS
    assert all(v > 0 for v in cn["a_norm_layers"])


# --------------------------------------------------------------------------
# end-to-end CPU micro-run exercising rand1 + pulse + freeze_plga


def test_micro_run_rand1_pulse_freeze_plga(tmp_path):
    tokens = tmp_path / "toks.npy"
    ctx = 32
    n_chunks = 5320  # > PROBE_RESERVE_CHUNKS + train budget
    rng = np.random.default_rng(0)
    arr = rng.integers(1, 32000, size=n_chunks * ctx, dtype=np.uint16)
    arr.tofile(tokens)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, "train_run.py", "--name", "microw5",
           "--lr", "1e-3", "--warmup", "2", "--steps", "6",
           "--batch", "2", "--ctx", str(ctx), "--layers", "1",
           "--heads", "2", "--dk", "8", "--adff", "12",
           "--device", "cpu", "--tokens", str(tokens),
           "--outdir", str(tmp_path / "runs"),
           "--probe_every", "2", "--sharp_every", "3",
           "--sharp_pre_every", "5", "--ckpt_steps", "",
           "--val_batches", "1", "--loss_mode", "rand1",
           "--pulse_at", "3", "--pulse_len", "2", "--pulse_lr", "0.5",
           "--freeze_plga_at", "1",
           "--tok_model", "data/tokenizer.model"]
    r = subprocess.run(cmd, cwd=EXP_DIR, env=env, capture_output=True,
                       text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    recs = [json.loads(l) for l in
            open(tmp_path / "runs" / "microw5" / "log.jsonl")]
    steps = {rec["step"]: rec for rec in recs if "step" in rec}
    assert set(steps) == set(range(1, 7))
    # pulse: steps 3 and 4 log the pulse lr and flag
    assert steps[3]["lr"] == 0.5 and steps[3].get("pulse") is True
    assert steps[4]["lr"] == 0.5 and steps[4].get("pulse") is True
    assert "pulse" not in steps[5] and steps[5]["lr"] != 0.5
    # freeze_plga fires at step 2: trainable plga block empty from then on
    assert steps[1]["gnorm_plga"] > 0
    assert steps[2]["gnorm_plga"] == 0.0
    # per-step velocity keys present; plga frozen => zero velocity
    assert steps[5]["dtheta_plga"] == 0.0
    assert steps[5]["dtheta_phi"] > 0
    # probe records carry the new diagnostics
    assert "att_entropy_layers" in steps[2]
    assert "repr_effrank" in steps[2]
    assert "a_norm_layers" in steps[2]
    # sharp cadence carries the finite-difference curvature
    assert "fdcurv_collapse" in steps[3]
    # sharp_pre cadence carries the restricted Gram
    assert "gram_eigs" in steps[5]
    # rand1 validation metric absent before step 6's val block? (val every
    # 1000 steps: not reached) -- final record exists instead
    final = next(rec for rec in recs if rec.get("event") == "final")
    assert final["step"] == 6


def test_scheduler_validation_rejects_bad_warmup(tmp_path):
    tokens = tmp_path / "t.npy"
    np.zeros(64, dtype=np.uint16).tofile(tokens)
    cmd = [sys.executable, "train_run.py", "--name", "bad",
           "--lr", "1e-3", "--warmup", "6", "--steps", "6",
           "--device", "cpu", "--tokens", str(tokens),
           "--outdir", str(tmp_path / "runs")]
    r = subprocess.run(cmd, cwd=EXP_DIR, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode != 0
    assert "0 < warmup < steps" in r.stderr
