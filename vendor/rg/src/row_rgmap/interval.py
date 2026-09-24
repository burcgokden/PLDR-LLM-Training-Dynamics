"""Nonnegative interval arithmetic for observer-resolved affine RG edges.

The module deliberately implements only the monotone operations needed by
the row-energy cocycle. It is not a general interval-arithmetic package.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _round_down_nonnegative(value: float) -> float:
    if value == 0.0:
        return 0.0
    return math.nextafter(value, -math.inf)


def _round_up(value: float) -> float:
    if value == math.inf:
        raise OverflowError("finite interval arithmetic overflowed")
    return math.nextafter(value, math.inf)


@dataclass(frozen=True)
class NonnegativeInterval:
    """A finite closed interval contained in the nonnegative real line."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
            or self.lower < 0.0
            or self.lower > self.upper
        ):
            raise ValueError("invalid finite nonnegative interval")

    @classmethod
    def singleton(cls, value: float) -> "NonnegativeInterval":
        return cls(float(value), float(value))

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def __mul__(self, other: "NonnegativeInterval") -> "NonnegativeInterval":
        upper = self.upper * other.upper
        return NonnegativeInterval(
            _round_down_nonnegative(self.lower * other.lower),
            0.0 if self.upper == 0.0 or other.upper == 0.0 else _round_up(upper),
        )

    def __add__(self, other: "NonnegativeInterval") -> "NonnegativeInterval":
        upper = self.upper + other.upper
        return NonnegativeInterval(
            _round_down_nonnegative(self.lower + other.lower),
            0.0 if self.upper == 0.0 and other.upper == 0.0 else _round_up(upper),
        )


@dataclass(frozen=True)
class IntervalAffineEdge:
    """A rectangular enclosure of nonnegative affine gain and source."""

    gain: NonnegativeInterval
    source: NonnegativeInterval

    @classmethod
    def singleton(cls, gain: float, source: float) -> "IntervalAffineEdge":
        return cls(
            NonnegativeInterval.singleton(gain),
            NonnegativeInterval.singleton(source),
        )

    def compose(self, earlier: "IntervalAffineEdge") -> "IntervalAffineEdge":
        """Chronologically compose ``earlier`` followed by ``self``."""

        return IntervalAffineEdge(
            gain=self.gain * earlier.gain,
            source=self.gain * earlier.source + self.source,
        )

    def act(self, energy: NonnegativeInterval) -> NonnegativeInterval:
        return self.gain * energy + self.source

    def contains(self, gain: float, source: float) -> bool:
        return self.gain.contains(gain) and self.source.contains(source)


def certified_positive_gain_interval(
    source_energy: NonnegativeInterval,
    target_energy: NonnegativeInterval,
) -> IntervalAffineEdge:
    """Enclose a canonical gain edge when the source is certified positive.

    A zero lower endpoint is rejected because energy enclosures alone then
    give no finite upper bound on the positive-branch ratio.
    """

    if source_energy.lower <= 0.0:
        raise ValueError("gain ratios require a strictly positive lower endpoint")
    return IntervalAffineEdge(
        gain=NonnegativeInterval(
            _round_down_nonnegative(
                target_energy.lower / source_energy.upper
            ),
            _round_up(target_energy.upper / source_energy.lower),
        ),
        source=NonnegativeInterval.singleton(0.0),
    )


def certified_face_edge(
    source_energy: NonnegativeInterval,
    target_energy: NonnegativeInterval,
) -> IntervalAffineEdge:
    """Enclose a canonical restart edge when the source is certified zero."""

    if source_energy.upper != 0.0:
        raise ValueError("a face edge requires a zero upper endpoint")
    return IntervalAffineEdge(
        gain=NonnegativeInterval.singleton(0.0),
        source=target_energy,
    )


def block_interval_edges(edges: list[IntervalAffineEdge]) -> IntervalAffineEdge:
    """Block a chronological list, with the first list element acting first."""

    result = IntervalAffineEdge.singleton(1.0, 0.0)
    for edge in edges:
        result = edge.compose(result)
    return result
