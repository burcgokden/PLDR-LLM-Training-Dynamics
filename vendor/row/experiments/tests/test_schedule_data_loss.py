"""Schedule shape (ramp then immediate cosine decay, no plateau), loss
modes, masks, and the data-offset / probe-region arithmetic."""

import math

import torch

from train_run import (LinearWarmupCosineLRSchedule, masked_loss_function,
                       masked_accuracy, last_only_targets, create_masks,
                       PROBE_RESERVE_CHUNKS)


def lr_factors(total, warmup, alpha=0.1):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = LinearWarmupCosineLRSchedule(opt, total_steps=total,
                                         warmup_steps=warmup, alpha=alpha)
    factors = []
    for _ in range(total):
        factors.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return factors


def test_schedule_linear_ramp_then_immediate_cosine_decay():
    f = lr_factors(1000, 100)
    # linear ramp
    assert abs(f[50] - 50 / 100) < 1e-9
    assert abs(f[100] - 1.0) < 1e-9
    # no plateau: strictly decreasing right after the ramp
    assert f[101] < f[100]
    assert f[150] < f[110] < f[101]
    # cosine endpoint at alpha
    assert abs(f[-1] - (0.9 * 0.5 * (1 + math.cos(math.pi * (899 / 900)))
                        + 0.1)) < 1e-9


def test_masked_loss_and_last_only_targets():
    torch.manual_seed(0)
    B, T, V = 2, 5, 11
    y = torch.randint(1, V, (B, T))
    logits = torch.randn(B, T, V)
    full = masked_loss_function(y, logits)
    yl = last_only_targets(y)
    assert (yl[:, :-1] == 0).all() and (yl[:, -1] == y[:, -1]).all()
    last = masked_loss_function(yl, logits)
    # last-only equals the mean CE of the final positions
    ce = torch.nn.CrossEntropyLoss()(logits[:, -1], y[:, -1])
    assert torch.allclose(last, ce, atol=1e-6)
    assert full.isfinite() and last.isfinite()
    acc = masked_accuracy(y, logits)
    assert 0.0 <= acc.item() <= 1.0


def test_create_masks_causal_and_padding():
    x = torch.tensor([[5, 6, 0]])
    m = create_masks(x, torch.device("cpu"))  # [1,1,3,3] in {0,1}
    m = m[0, 0]
    assert m[0, 1] == 1 and m[0, 2] == 1      # future masked
    assert m[1, 0] == 0                        # past visible
    assert m[1, 2] == 1 and m[2, 2] == 1      # pad column masked


def test_data_offset_and_probe_region_disjointness():
    ctx, batch = 256, 32
    n_tokens = 220_000_000
    n_chunks = n_tokens // ctx
    pb = n_chunks - PROBE_RESERVE_CHUNKS

    def train_range(offset, steps):
        return offset, offset + steps * batch

    # 12k and 24k fresh runs fit below the reserved region
    for steps in (12000, 24000):
        lo, hi = train_range(0, steps)
        assert hi <= pb
    # continuation resumes where the source ended: no replay
    src_lo, src_hi = train_range(0, 12000)
    cont_lo, cont_hi = train_range(src_hi, 6000)
    assert cont_lo == src_hi                      # stream continues
    assert cont_hi <= pb                          # still below probes
    # probe subregions used by train_run and measure_tilt are inside the
    # reserve and mutually disjoint
    probe_spans = [(pb, pb + 8), (pb + 64, pb + 72),
                   (pb + 128, pb + 136), (pb + 160, pb + 168),
                   (pb + 192, pb + 200),
                   (pb + 256, pb + 256 + 8 * batch),      # val
                   (pb + 2048, pb + 2048 + 37 * 7 + 1),   # prompts
                   (pb + 2560, pb + 2560 + 64)]           # tilt batches
    for lo, hi in probe_spans:
        assert pb <= lo < hi <= n_chunks
    for i in range(len(probe_spans)):
        for j in range(i + 1, len(probe_spans)):
            a, b = probe_spans[i], probe_spans[j]
            assert b[0] >= a[1] or a[0] >= b[1], (a, b)
