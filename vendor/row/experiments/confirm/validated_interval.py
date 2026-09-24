"""Exact rational interval primitives for confirmation certificates.

The certification path in this module never treats agreement between two
floating computations as an error enclosure.  Finite inputs are converted to
exact rational numbers, arithmetic endpoints are exact, and transcendental
functions use rational series remainders.  Floating conversion is provided
only for display and diagnostic comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt
from typing import Iterable


BACKEND = {
    "name": "fraction-taylor-interval",
    "version": 1,
    "endpoint_type": "exact_rational",
    "rounding": "exact arithmetic with proved rational remainder",
}


def as_fraction(value) -> Fraction:
    """Convert a finite scalar without a decimal round trip.

    Floats are interpreted as their exact IEEE binary value.  Certificate
    inputs should normally use integers, ``Fraction`` objects, or strings.
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("Boolean values are not certificate scalars")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction.from_float(value)
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, dict):
        if set(value) != {"numerator", "denominator"}:
            raise ValueError("a rational endpoint needs numerator and denominator")
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    raise TypeError(f"unsupported rational scalar {type(value).__name__}")


def rational_object(value: Fraction) -> dict:
    value = as_fraction(value)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _sqrt_floor(value: Fraction, scale: int) -> Fraction:
    """Return a provable lower rational bound for a nonnegative square root."""

    if value < 0:
        raise ValueError("square root domain is nonnegative")
    if scale < 1:
        raise ValueError("square-root scale must be positive")
    # sqrt(n / d) = sqrt(n d) / d.  Integer square root rounds downward.
    numerator = isqrt(value.numerator * value.denominator * scale * scale)
    return Fraction(numerator, value.denominator * scale)


def sqrt_enclosure(value, *, digits: int = 40) -> tuple[Fraction, Fraction]:
    """Enclose ``sqrt(value)`` by rationals with a decimal mesh."""

    value = as_fraction(value)
    scale = 10 ** int(digits)
    lower = _sqrt_floor(value, scale)
    if lower * lower == value:
        return lower, lower
    upper = lower + Fraction(1, scale)
    if upper * upper < value:  # Defensive check for the floor construction.
        raise ArithmeticError("internal square-root enclosure failure")
    return lower, upper


def exp_enclosure(value, *, tolerance=Fraction(1, 10**45),
                  max_terms: int = 20000) -> tuple[Fraction, Fraction]:
    """Enclose the exponential of a rational using a positive Taylor tail.

    For ``x >= 0`` the partial sum is a lower bound.  Once the ratio of all
    later terms is at most ``r < 1``, the remaining tail is bounded by the
    next term divided by ``1-r``.  Negative arguments are handled by the
    exact reciprocal identity.
    """

    x = as_fraction(value)
    tolerance = as_fraction(tolerance)
    if tolerance <= 0:
        raise ValueError("exponential tolerance must be positive")
    if x < 0:
        lo, hi = exp_enclosure(-x, tolerance=tolerance, max_terms=max_terms)
        return Fraction(1, hi), Fraction(1, lo)
    if x == 0:
        return Fraction(1), Fraction(1)

    total = Fraction(1)
    term = Fraction(1)
    for n in range(0, max_terms):
        term *= x / Fraction(n + 1)
        total += term
        next_term = term * x / Fraction(n + 2)
        ratio = x / Fraction(n + 3)
        if ratio < 1:
            tail = next_term / (1 - ratio)
            if tail <= tolerance:
                return total, total + tail
    raise ArithmeticError("exponential series did not reach its rational tolerance")


@dataclass(frozen=True)
class RationalInterval:
    """A closed interval with exact rational endpoints."""

    lower: Fraction
    upper: Fraction

    def __post_init__(self):
        lower = as_fraction(self.lower)
        upper = as_fraction(self.upper)
        if lower > upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def point(cls, value) -> "RationalInterval":
        value = as_fraction(value)
        return cls(value, value)

    @classmethod
    def from_object(cls, value) -> "RationalInterval":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict) or set(value) != {"lower", "upper"}:
            return cls.point(value)
        return cls(as_fraction(value["lower"]), as_fraction(value["upper"]))

    def to_object(self) -> dict:
        return {
            "lower": rational_object(self.lower),
            "upper": rational_object(self.upper),
        }

    @property
    def width(self) -> Fraction:
        return self.upper - self.lower

    @property
    def midpoint(self) -> Fraction:
        return (self.lower + self.upper) / 2

    def contains(self, value) -> bool:
        value = as_fraction(value)
        return self.lower <= value <= self.upper

    def contains_interval(self, other) -> bool:
        other = RationalInterval.from_object(other)
        return self.lower <= other.lower and other.upper <= self.upper

    def maximum_absolute(self) -> Fraction:
        return max(abs(self.lower), abs(self.upper))

    def minimum_absolute(self) -> Fraction:
        if self.lower <= 0 <= self.upper:
            return Fraction(0)
        return min(abs(self.lower), abs(self.upper))

    def __neg__(self):
        return RationalInterval(-self.upper, -self.lower)

    def __add__(self, other):
        other = RationalInterval.from_object(other)
        return RationalInterval(self.lower + other.lower,
                                self.upper + other.upper)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-RationalInterval.from_object(other))

    def __rsub__(self, other):
        return RationalInterval.from_object(other) - self

    def __mul__(self, other):
        other = RationalInterval.from_object(other)
        products = (
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        )
        return RationalInterval(min(products), max(products))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lower <= 0 <= self.upper:
            raise ZeroDivisionError("interval reciprocal crosses zero")
        values = (Fraction(1, self.lower), Fraction(1, self.upper))
        return RationalInterval(min(values), max(values))

    def __truediv__(self, other):
        return self * RationalInterval.from_object(other).reciprocal()

    def __rtruediv__(self, other):
        return RationalInterval.from_object(other) / self

    def square(self):
        if self.lower <= 0 <= self.upper:
            return RationalInterval(0, self.maximum_absolute() ** 2)
        squares = (self.lower ** 2, self.upper ** 2)
        return RationalInterval(min(squares), max(squares))

    def sqrt(self, *, digits: int = 40):
        if self.lower < 0:
            raise ValueError("interval square root crosses the negative axis")
        lower = sqrt_enclosure(self.lower, digits=digits)[0]
        upper = sqrt_enclosure(self.upper, digits=digits)[1]
        return RationalInterval(lower, upper)

    def exp(self, *, tolerance=Fraction(1, 10**45)):
        lower = exp_enclosure(self.lower, tolerance=tolerance)[0]
        upper = exp_enclosure(self.upper, tolerance=tolerance)[1]
        return RationalInterval(lower, upper)

    def sigmoid(self):
        return RationalInterval.point(1) / (
            RationalInterval.point(1) + (-self).exp())

    def hull(self, other):
        other = RationalInterval.from_object(other)
        return RationalInterval(min(self.lower, other.lower),
                                max(self.upper, other.upper))

    def as_float_pair(self) -> tuple[float, float]:
        return float(self.lower), float(self.upper)


def interval_sum(values: Iterable[RationalInterval]) -> RationalInterval:
    total = RationalInterval.point(0)
    for value in values:
        total += value
    return total


def interval_dot(left, right) -> RationalInterval:
    left = tuple(left)
    right = tuple(right)
    if len(left) != len(right):
        raise ValueError("interval dot-product dimensions disagree")
    return interval_sum(a * b for a, b in zip(left, right))


def nonnegative_sqrt_upper(value, *, digits: int = 40) -> Fraction:
    """Return only the proved upper endpoint for a nonnegative rational."""

    return sqrt_enclosure(value, digits=digits)[1]
