"""Exact positive-affine RG algebra.

An edge ``(q, zeta)`` acts on energy as ``E -> q E + zeta``.  The
first argument of :func:`compose` is the later edge.  This convention keeps
chronological order explicit for noncommuting source transport.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True, slots=True)
class Edge:
    """One nonnegative affine energy edge."""

    gain: float
    source: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "gain", _nonnegative(self.gain, "gain"))
        object.__setattr__(self, "source", _nonnegative(self.source, "source"))

    def act(self, energy: float) -> float:
        return self.gain * _nonnegative(energy, "energy") + self.source

    def as_dict(self) -> dict[str, float]:
        return {"gain": self.gain, "source": self.source}


IDENTITY = Edge(1.0, 0.0)


def compose(later: Edge, earlier: Edge) -> Edge:
    """Compose two chronologically ordered edges exactly."""

    return Edge(
        later.gain * earlier.gain,
        later.gain * earlier.source + later.source,
    )


def block_edges(edges: Iterable[Edge]) -> Edge:
    """Block a chronological edge sequence into one affine edge."""

    total = IDENTITY
    for edge in edges:
        total = compose(edge, total)
    return total


def normalize(
    edge: Edge, source_gauge: float, target_gauge: float
) -> Edge:
    """Change positive energy units at the two endpoints."""

    source_gauge = _finite(source_gauge, "source_gauge")
    target_gauge = _finite(target_gauge, "target_gauge")
    if source_gauge <= 0.0 or target_gauge <= 0.0:
        raise ValueError("endpoint gauges must be strictly positive")
    return Edge(
        edge.gain * source_gauge / target_gauge,
        edge.source / target_gauge,
    )


def canonical_edge(source_energy: float, target_energy: float) -> Edge:
    """Return the branch-safe canonical edge between two energies."""

    source_energy = _nonnegative(source_energy, "source_energy")
    target_energy = _nonnegative(target_energy, "target_energy")
    if source_energy > 0.0:
        return Edge(target_energy / source_energy, 0.0)
    return Edge(0.0, target_energy)


def canonical_edges(energies: Sequence[float]) -> list[Edge]:
    """Construct all canonical edges of a nonempty energy path."""

    values = [_nonnegative(value, "energy") for value in energies]
    if not values:
        raise ValueError("energies must be nonempty")
    return [
        canonical_edge(source, target)
        for source, target in zip(values, values[1:])
    ]


def replay(initial_energy: float, edges: Iterable[Edge]) -> list[float]:
    """Replay a chronological edge path, including the initial energy."""

    energy = _nonnegative(initial_energy, "initial_energy")
    values = [energy]
    for edge in edges:
        energy = edge.act(energy)
        values.append(energy)
    return values


def homogeneous_block(gain: float, source: float, steps: int) -> Edge:
    """Block ``steps`` identical affine edges without a singular formula."""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    edge = Edge(gain, source)
    return block_edges([edge] * steps)


def block_levels(edges: Sequence[Edge], factor: int = 2) -> list[dict]:
    """Return all aligned exact RG levels of a power-sized edge path."""

    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 2:
        raise ValueError("factor must be an integer at least two")
    if not edges:
        raise ValueError("edges must be nonempty")
    remainder = len(edges)
    while remainder > 1 and remainder % factor == 0:
        remainder //= factor
    if remainder != 1:
        raise ValueError("edge count must be an exact power of factor")

    levels: list[dict] = []
    width = 1
    while width <= len(edges):
        blocks = [
            block_edges(edges[start : start + width])
            for start in range(0, len(edges), width)
        ]
        levels.append(
            {
                "level": len(levels),
                "block_size": width,
                "blocks": [edge.as_dict() for edge in blocks],
            }
        )
        width *= factor
    return levels


def classify_gain(gain: float, *, tolerance: float = 0.0) -> str:
    """Classify an exact or tolerance-resolved block gain."""

    gain = _nonnegative(gain, "gain")
    tolerance = _nonnegative(tolerance, "tolerance")
    if gain < 1.0 - tolerance:
        return "contracting"
    if gain > 1.0 + tolerance:
        return "expanding"
    return "marginal"


def clean_flow(gain: float, block_size: float) -> float:
    """Continuous clean blocking ``q(b) = q**b`` for positive ``b``."""

    gain = _nonnegative(gain, "gain")
    block_size = _finite(block_size, "block_size")
    if block_size <= 0.0:
        raise ValueError("block_size must be strictly positive")
    if gain == 0.0:
        return 0.0
    return math.exp(block_size * math.log(gain))


def beta(gain: float) -> float:
    """Autonomous beta function for logarithmic block scale."""

    gain = _nonnegative(gain, "gain")
    if gain == 0.0:
        return 0.0
    return gain * math.log(gain)


def clean_correlation_scale(gain: float) -> float:
    """The e-folding scale ``1 / abs(log q)`` away from ``q = 1``."""

    gain = _nonnegative(gain, "gain")
    if gain in (0.0, 1.0):
        return math.inf if gain == 1.0 else 0.0
    return 1.0 / abs(math.log(gain))
