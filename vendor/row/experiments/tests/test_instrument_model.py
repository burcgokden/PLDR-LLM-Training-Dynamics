"""Integration tests of the instrumentation and controls on a tiny
PLDR model (CPU): curvature probes, preconditioned Lanczos, the
collapse-coordinate Rayleigh quotient, the SDPA Gcache control, and the
collapsed-forward patch of the tilt measurement."""

import math

import pytest
import torch

import instrument
from pldr_model_v510 import PLDR_Model
from train_run import masked_loss_function, create_masks
import measure_tilt


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
    # one step so the Adam state exists
    preds, _, _, _ = model([x, mask])
    loss = masked_loss_function(y, preds)
    opt.zero_grad()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return model, opt, x, mask, y, device


def test_all_sharpness_keys_and_finiteness(setup):
    model, opt, x, mask, y, _ = setup
    out = instrument.all_sharpness(model, masked_loss_function, x, mask, y,
                                   optimizer=opt, n_iter=10)
    for k in ["lam_full", "lam_full_min", "lam_full_resid", "lam_phi",
              "lam_plga", "rayq_upd", "rayq_upd_pre", "rayq_upd_phi",
              "rayq_upd_pre_phi"]:
        assert k in out, k
        assert math.isfinite(out[k]), k
    assert out["lam_full"] >= out["lam_full_min"]


def test_precond_sharpness_lists_and_residuals(setup):
    model, opt, x, mask, y, device = setup
    x2 = torch.randint(1, VOCAB, (BATCH, SEQ))
    y2 = torch.randint(1, VOCAB, (BATCH, SEQ))
    batches = [(x, mask, y), (x2, create_masks(x2, device), y2)]
    out = instrument.precond_sharpness(model, masked_loss_function, batches,
                                       opt, n_iter=10, n_starts=2)
    for k in ["lam_pre_max_full", "lam_pre_min_full", "lam_pre_resid_full",
              "lam_pre_max_nonemb", "lam_pre_min_nonemb",
              "lam_pre_resid_nonemb", "lam_full_batches"]:
        assert k in out, k
        assert len(out[k]) == 2
        assert all(math.isfinite(v) for v in out[k])
    for a, b in zip(out["lam_pre_min_full"], out["lam_pre_max_full"]):
        assert a <= b


def test_collapse_dir_curvature(setup):
    model, opt, x, mask, y, _ = setup
    out = instrument.collapse_dir_curvature(model, masked_loss_function,
                                            x, mask, y, opt, x, mask,
                                            n_rows=4)
    assert out["u_stat"] > 0
    assert math.isfinite(out["rayq_collapse_raw"])
    assert math.isfinite(out["rayq_collapse_pre"])


def test_nonemb_excludes_vocab_blocks(setup):
    model = setup[0]
    non = instrument.nonemb_params(model)
    total = [p for p in model.parameters() if p.requires_grad]
    n_excl = sum(p.numel() for p in model.decoder.embedding.parameters())
    n_excl += sum(p.numel() for p in model.final_layer.parameters())
    assert sum(p.numel() for p in non) == \
        sum(p.numel() for p in total) - n_excl


def test_sdpa_gcache_control(setup):
    model, _, x, mask, _, device = setup
    eye = torch.eye(DK, device=device)[None, None]
    gc = [[eye, eye] for _ in range(LAYERS)]
    logits_sdpa, _, att, kv = model([x, mask], Gcachelst=gc)
    # the deductive operator is exactly the identity
    assert torch.equal(att[0][4], eye)
    assert torch.isfinite(logits_sdpa).all()
    # and differs from the PLGA forward
    logits_plga, _, att2, _ = model([x, mask])
    assert not torch.equal(att2[0][4], eye)
    assert not torch.allclose(logits_sdpa, logits_plga)


def test_collapsed_forward_patch_and_restore(setup):
    model, _, x, mask, y, _ = setup
    stats, lval = measure_tilt.measure_collapsed(model, x, mask, y)
    # instance-level forward shadows removed: class method resolves again
    for dec in model.decoder.dec_layers:
        assert "forward" not in dec.mha1.plgatt_layer.__dict__
    assert math.isfinite(lval)
    for s in stats:
        for k in ["C_fro", "gbar_norm", "rho_rms", "rho_max", "J_fro"]:
            assert math.isfinite(s[k]) and s[k] >= 0
    # current-state variant also runs and both produce per-layer entries
    stats_cur, _ = measure_tilt.measure_current(model, x, mask, y)
    assert len(stats_cur) == len(stats) == LAYERS
