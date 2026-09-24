"""Exact global-update clock for the applied warm-up/cosine schedule."""

from __future__ import annotations

import math


def applied_factor(update, *, total_updates, warmup_updates, floor):
    """Return the factor prepared before a one-indexed optimizer update.

    The live ``LambdaLR`` convention uses scheduler index ``k = update - 1``.
    Consequently update one uses zero, update ``W + 1`` uses the peak, and
    the Adam first-moment bias correction after update ``s`` uses
    ``1 - beta1**s``.
    """

    update = int(update)
    total_updates = int(total_updates)
    warmup_updates = int(warmup_updates)
    floor = float(floor)
    if not 1 <= update <= total_updates:
        raise ValueError("global optimizer update lies outside the schedule")
    if not 0 < warmup_updates < total_updates:
        raise ValueError("warm-up must lie strictly inside the schedule")
    if not 0 <= floor <= 1:
        raise ValueError("cosine floor must lie in [0, 1]")
    index = update - 1
    if index <= warmup_updates:
        return index / warmup_updates
    decay_index = index - warmup_updates
    decay_length = total_updates - warmup_updates
    if decay_length <= 0:
        return floor
    return floor + 0.5 * (1 - floor) * (
        1 + math.cos(math.pi * decay_index / decay_length))


def boundary_table(*, total_updates, warmup_updates, floor, beta1):
    updates = {
        "first_update": 1,
        "last_subpeak_warmup_update": warmup_updates,
        "peak_update": warmup_updates + 1,
        "first_decay_update": warmup_updates + 2,
        "terminal_update": total_updates,
    }
    result = []
    for name, update in updates.items():
        if update > total_updates:
            continue
        result.append({
            "name": name,
            "global_optimizer_update": update,
            "scheduler_index_applied": update - 1,
            "applied_factor": applied_factor(
                update,
                total_updates=total_updates,
                warmup_updates=warmup_updates,
                floor=floor,
            ),
            "adam_first_moment_bias_denominator": 1 - float(beta1) ** update,
            "adam_bias_exponent": update,
        })
    return result


def verify_training_schedule(schedule, *, total_updates, warmup_updates, floor):
    expected = [
        applied_factor(
            update,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            floor=floor,
        )
        for update in range(1, total_updates + 1)
    ]
    if len(schedule) != len(expected):
        raise ValueError("realized applied-rate schedule has the wrong horizon")
    errors = [abs(float(actual) - target)
              for actual, target in zip(schedule, expected)]
    return {
        "maximum_absolute_error": max(errors, default=0.0),
        "matches": all(error <= 2e-15 for error in errors),
        "boundary_table": boundary_table(
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            floor=floor,
            beta1=0.9,
        ),
    }
