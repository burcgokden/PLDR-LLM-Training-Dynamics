import math
from pathlib import Path
import sys

import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

from clipped_route import (  # noqa: E402
    clipped_affine_route_step,
    cumulative_log_contraction,
    cumulative_route_dominance,
    first_route_floor_crossing,
    realized_route_ratios,
)


def test_saturated_step_uses_realized_not_affine_ratio():
    step = clipped_affine_route_step(
        0.9, eta=0.1, theta=0.0, rebuild=2.0, upper=1.0
    )
    assert step.proposal == pytest.approx(1.1)
    assert step.after == pytest.approx(1.0)
    assert step.upper_clipped
    assert step.ratio == pytest.approx(1.0 / 0.9)
    assert step.log_contraction == pytest.approx(-math.log(1.0 / 0.9))


def test_time_varying_cumulative_rate_telescopes():
    path = (1.0, 0.8, 0.88, 0.44, 0.5, 0.1)
    ratios = realized_route_ratios(path)
    cumulative = cumulative_log_contraction(path)
    assert ratios == pytest.approx((0.8, 1.1, 0.5, 0.5 / 0.44, 0.2))
    assert cumulative[-1] == pytest.approx(math.log(10.0))
    assert cumulative[-1] == pytest.approx(
        sum(-math.log(ratio) for ratio in ratios)
    )


def test_route_comparison_is_cumulative_and_floor_based():
    jet = (1.0, 0.5, 0.6, 0.2)
    coupling = (1.0, 0.8, 0.4, 0.3)
    dominance = cumulative_route_dominance(jet, coupling)
    assert dominance[-1] == pytest.approx(math.log(0.3 / 0.2))
    assert first_route_floor_crossing(jet, 0.25) == 3
    assert first_route_floor_crossing(coupling, 0.25) is None
