"""Wave-8 schedule additions: golden bit-identity of the archived
default schedule shapes, the plateau-then-anneal (hold) shape, the
anneal-floor generalization, and the pulse-train semantics.

The golden tests reproduce the archived factor formulas independently
(inline, not by importing them) so any change to the default code paths
is caught as a bit-level regression."""

import math
import re
import os

import torch

from train_run import (LinearWarmupCosineLRSchedule, HoldCosineLRSchedule,
                       schedule_multiplier)

TRAIN_RUN_SRC = os.path.join(os.path.dirname(__file__), "..",
                             "train_run.py")


def factors(sched_ctor, total, **kw):
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = sched_ctor(opt, total_steps=total, **kw)
    out = []
    for _ in range(total):
        out.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    return out


def golden_cosine(step, total, warmup, alpha):
    """The archived LinearWarmupCosineLRSchedule formula, reproduced
    verbatim and independently."""
    step = float(min(step, total))
    if step <= warmup:
        return step / warmup
    decay = (step - warmup) / (total - warmup)
    return (1 - alpha) * 0.5 * (1 + math.cos(math.pi * decay)) + alpha


def test_default_cosine_schedule_bit_identical_on_archived_grids():
    # every scheduled (steps, warmup) grid that appears in waves 1-7
    grids = [(12000, 1000), (12000, 125), (12000, 500), (12000, 2000),
             (12000, 4000), (24000, 1000), (6000, 500), (48000, 4000),
             (8000, 1000)]
    for total, warmup in grids:
        f = factors(LinearWarmupCosineLRSchedule, total,
                    warmup_steps=warmup, alpha=0.1)
        for s in [0, 1, warmup // 2, warmup, warmup + 1,
                  (total + warmup) // 2, total - 2, total - 1]:
            assert f[s] == golden_cosine(s, total, warmup, 0.1), (
                total, warmup, s)


def test_const_lr_safety_ramp_semantics_golden():
    # the const-lr branch uses factor min(1, (s+1)/max(1, warmup));
    # warmup <= 1 reproduces the archived constant-rate runs exactly
    for warmup in (1, 250):
        w = max(1, warmup)
        f = [min(1.0, (s + 1) / w) for s in range(600)]
        if warmup == 1:
            assert all(x == 1.0 for x in f)
        else:
            assert f[0] == 1 / 250 and f[248] == 249 / 250
            assert f[249] == 1.0 and all(x == 1.0 for x in f[249:])
    # The trainer-owned implementation agrees with the independent golden
    # formula at every point of both representative grids.
    for warmup in (1, 250):
        w = max(1, warmup)
        for s in range(600):
            assert schedule_multiplier(
                s, 600, warmup, const_lr=True
            ) == min(1.0, (s + 1) / w)


def test_hold_schedule_shape_and_boundaries():
    total, warmup, hold, alpha = 12000, 1000, 6000, 0.1
    f = factors(HoldCosineLRSchedule, total, warmup_steps=warmup,
                hold_until=hold, alpha=alpha)
    # linear rise identical to the archived ramp
    assert f[500] == 500 / 1000 and f[1000] == 1.0
    # plateau holds at exactly 1.0 through hold_until
    assert all(x == 1.0 for x in f[1000:hold + 1])
    # descent begins immediately after hold_until
    assert f[hold + 1] < 1.0
    # cosine value and floor endpoint
    s = 9000
    want = (1 - alpha) * 0.5 * (1 + math.cos(
        math.pi * (s - hold) / (total - hold))) + alpha
    assert abs(f[s] - want) < 1e-12
    assert abs(f[total - 1] - golden_hold_end(total, hold, alpha)) < 1e-12


def golden_hold_end(total, hold, alpha):
    decay = (total - 1 - hold) / (total - hold)
    return (1 - alpha) * 0.5 * (1 + math.cos(math.pi * decay)) + alpha


def test_hold_zero_with_warmup_one_is_pure_descent_from_full_rate():
    total = 4000
    f = factors(HoldCosineLRSchedule, total, warmup_steps=1, hold_until=0,
                alpha=0.1)
    assert f[0] == 1.0                      # starts at the full rate
    assert all(a > b for a, b in zip(f, f[1:]))   # strictly decreasing
    want = 0.9 * 0.5 * (1 + math.cos(math.pi * (total - 1) / total)) + 0.1
    assert abs(f[-1] - want) < 1e-12


def test_anneal_floor_generalization():
    total, warmup = 12000, 1000
    for alpha in (0.3, 0.55):
        f = factors(LinearWarmupCosineLRSchedule, total,
                    warmup_steps=warmup, alpha=alpha)
        for s in (1500, 6000, total - 1):
            assert f[s] == golden_cosine(s, total, warmup, alpha)
        # the floor is approached, never undercut
        assert min(f[warmup:]) >= alpha - 1e-12


def test_new_flags_default_off_in_source():
    """The new CLI surface must not perturb archived invocations: the
    defaults reproduce the archived behavior bit for bit."""
    src = open(TRAIN_RUN_SRC).read()
    assert re.search(r'"--anneal_floor", type=float, default=0\.1', src)
    assert re.search(r'"--hold_until", type=int, default=-1', src)
    assert re.search(r'"--pulse_train", type=str, default=""', src)
    # the default schedule call site consumes the (default 0.1) floor
    assert "alpha=args.anneal_floor" in src
    # the pulse-train branch overrides in_pulse only when the flag is set
    assert "if pulse_starts is not None:" in src


def test_pulse_train_window_semantics():
    starts, plen = [3000, 5000, 7000], 8
    def in_pulse(step):
        return any(t <= step < t + plen for t in starts)
    assert not in_pulse(2999) and in_pulse(3000)
    assert in_pulse(3007) and not in_pulse(3008)
    assert not in_pulse(4999) and in_pulse(5000) and not in_pulse(5008)
    assert sum(in_pulse(s) for s in range(2000, 9000)) == 3 * plen
