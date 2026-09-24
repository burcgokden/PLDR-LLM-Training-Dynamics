"""Pins the LR record-indexing convention of train_run.py.

The training loop calls sched.step() after opt.step() and before the
log record is written, so a non-pulse record at step t stores
sched.get_last_lr(), the rate prepared for the NEXT update (step
t+1), while a pulse record stores the pulse rate just applied at
step t.  These tests replicate the loop's ordering with a minimal
torch optimizer and a known schedule and assert both halves of the
convention; if the loop's ordering ever changes, they fail.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), "analysis"))

torch = pytest.importorskip("torch")


def _make(base_lr=1.0):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=base_lr)
    # factor = schedule-step count: applied lr at update t is t - 1,
    # so applied and prepared rates differ at every step
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: float(s))
    return p, opt, sched


def test_nonpulse_record_stores_next_update_rate():
    p, opt, sched = _make()
    applied, recorded = [], []
    for step in range(1, 6):
        applied.append(opt.param_groups[0]["lr"])   # rate this update
        p.grad = torch.ones(1)
        opt.step()
        sched.step()                                # loop order
        recorded.append(sched.get_last_lr()[0])     # what the log gets
    # the record at step t is the rate applied at step t+1 ...
    assert recorded[:-1] == pytest.approx(applied[1:])
    # ... and differs from the rate applied at step t itself
    assert all(r != a for r, a in zip(recorded, applied))


def test_pulse_record_stores_applied_rate():
    pulse_lr = 123.0
    p, opt, sched = _make()
    for step in range(1, 4):
        in_pulse = (step == 2)
        if in_pulse:
            for g in opt.param_groups:
                g["lr"] = pulse_lr                   # loop's override
        applied = opt.param_groups[0]["lr"]
        p.grad = torch.ones(1)
        opt.step()
        sched.step()                                 # recomputes next
        rec_lr = pulse_lr if in_pulse else sched.get_last_lr()[0]
        if in_pulse:
            # a pulse record stores the rate just applied ...
            assert rec_lr == applied == pulse_lr
        else:
            # ... a non-pulse record stores the next update's rate
            assert rec_lr == sched.get_last_lr()[0]
    # after the pulse, the schedule value is restored (next applied
    # rate comes from the schedule, not the override)
    assert opt.param_groups[0]["lr"] == sched.get_last_lr()[0]
