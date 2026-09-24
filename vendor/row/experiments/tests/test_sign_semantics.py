"""Regression tests for canonical direct-work sign semantics."""

import pytest

from analysis.sign_semantics import summarize_sign_counts


@pytest.mark.parametrize(
    ("counts", "status", "majority", "fraction"),
    (
        ((0, 0, 0, 0), "unavailable", None, 0.0),
        ((0, 0, 5, 0), "neutral", None, 0.0),
        ((3, 3, 2, 2), "tied", None, 0.375),
        ((7, 2, 1, 0), "positive", "positive", 0.7),
        ((1, 6, 3, 0), "negative", "negative", 0.6),
    ),
)
def test_sign_truth_table(counts, status, majority, fraction):
    positive, negative, neutral, unevaluable = counts
    evaluable = positive + negative + neutral
    result = summarize_sign_counts(
        planned=evaluable + unevaluable,
        evaluable=evaluable,
        positive=positive,
        negative=negative,
        neutral=neutral,
    )
    assert result["sign_status"] == status
    assert result["majority_sign"] == majority
    assert result["majority_fraction"] == pytest.approx(fraction)


def test_sign_counts_fail_closed_on_inconsistent_partition():
    with pytest.raises(ValueError, match="inconsistent"):
        summarize_sign_counts(
            planned=4, evaluable=4, positive=1, negative=1, neutral=1,
        )
