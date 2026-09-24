"""Canonical sign-summary semantics for direct-work analyses."""

from __future__ import annotations

from typing import Any


def summarize_sign_counts(
    *,
    planned: int,
    evaluable: int,
    positive: int,
    negative: int,
    neutral: int,
) -> dict[str, Any]:
    """Classify signed counts without assigning a sign to ties or neutrals."""

    counts = (planned, evaluable, positive, negative, neutral)
    if any(not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("sign-summary counts must be nonnegative integers")
    if evaluable > planned or positive + negative + neutral != evaluable:
        raise ValueError("sign-summary counts are inconsistent")

    if evaluable == 0:
        status = "unavailable"
        majority_sign = None
    elif positive == negative == 0:
        status = "neutral"
        majority_sign = None
    elif positive == negative:
        status = "tied"
        majority_sign = None
    elif positive > negative:
        status = "positive"
        majority_sign = "positive"
    else:
        status = "negative"
        majority_sign = "negative"

    majority_fraction = (
        max(positive, negative) / evaluable if evaluable else 0.0
    )
    return {
        "planned": planned,
        "evaluable": evaluable,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "sign_status": status,
        "majority_sign": majority_sign,
        "majority_fraction": float(majority_fraction),
    }
