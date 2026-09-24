"""Validated observer calculations for epsilon-regularized LayerNorm.

The functions in this module treat every supplied binary floating-point input
as the exact rational number it represents.  They then enclose the irrational
square roots by dyadic rationals and propagate closed intervals through the
LayerNorm quotient energy.  This distinguishes a certified zero from a value
that merely rounded to zero in a native floating-point evaluation.
"""

from __future__ import annotations

from fractions import Fraction
from math import isqrt
from typing import Iterable

import numpy as np


RationalInterval = tuple[Fraction, Fraction]


def dyadic_sqrt_enclosure(value: Fraction, precision_bits: int = 192) -> RationalInterval:
    """Return dyadic lower and upper bounds for ``sqrt(value)``."""

    if value <= 0:
        raise ValueError("the square-root argument must be positive")
    if precision_bits < 32:
        raise ValueError("precision_bits must be at least 32")
    scaled_numerator = value.numerator << (2 * precision_bits)
    quotient = scaled_numerator // value.denominator
    integer_lower = isqrt(quotient)
    denominator = 1 << precision_bits
    lower = Fraction(integer_lower, denominator)
    if integer_lower * integer_lower * value.denominator == scaled_numerator:
        return lower, lower
    return lower, Fraction(integer_lower + 1, denominator)


def _multiply_scalar(interval: RationalInterval, scalar: Fraction) -> RationalInterval:
    lower, upper = interval
    if scalar >= 0:
        return scalar * lower, scalar * upper
    return scalar * upper, scalar * lower


def _dyadic_outward(interval: RationalInterval, precision_bits: int) -> RationalInterval:
    """Round both endpoints outward to one common dyadic denominator."""

    lower, upper = interval
    denominator = 1 << precision_bits
    scaled_lower = lower.numerator << precision_bits
    scaled_upper = upper.numerator << precision_bits
    lower_integer = scaled_lower // lower.denominator
    upper_integer = -((-scaled_upper) // upper.denominator)
    return Fraction(lower_integer, denominator), Fraction(upper_integer, denominator)


def _square_interval(interval: RationalInterval) -> RationalInterval:
    lower, upper = interval
    if lower <= 0 <= upper:
        return Fraction(0), max(lower * lower, upper * upper)
    return min(lower * lower, upper * upper), max(lower * lower, upper * upper)


def _exact_fraction_rows(values: np.ndarray) -> list[list[Fraction]]:
    return [
        [Fraction.from_float(float(value)) for value in row]
        for row in values.tolist()
    ]


def layernorm_row_energy_enclosure(
    inputs: np.ndarray | Iterable[Iterable[float]],
    gain: np.ndarray | Iterable[float],
    epsilon: float,
    *,
    precision_bits: int = 192,
) -> RationalInterval:
    """Enclose row-centered affine LayerNorm energy for exact float inputs.

    ``inputs`` is a row-by-feature matrix. LayerNorm acts on each row along
    the feature axis. The returned energy centers the affine output along the
    row axis. The affine bias cancels exactly and therefore is not an input.
    """

    matrix = np.asarray(inputs)
    scale = np.asarray(gain)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("inputs must be a nonempty matrix")
    if scale.shape != (matrix.shape[1],):
        raise ValueError("gain width differs from the input feature width")
    if not np.issubdtype(matrix.dtype, np.floating) or not np.issubdtype(
        scale.dtype, np.floating
    ):
        raise TypeError("inputs and gain must use binary floating-point dtypes")
    if not np.isfinite(matrix).all() or not np.isfinite(scale).all():
        raise ValueError("inputs and gain must be finite")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if np.all(matrix == matrix[0:1]) or np.all(scale == 0):
        return Fraction(0), Fraction(0)

    rows = _exact_fraction_rows(matrix)
    gains = [Fraction.from_float(float(value)) for value in scale.tolist()]
    eps = Fraction.from_float(float(epsilon))
    row_count = len(rows)
    width = len(rows[0])
    normalized: list[list[RationalInterval]] = []
    for row in rows:
        mean = sum(row, Fraction(0)) / width
        centered = [value - mean for value in row]
        variance = sum((value * value for value in centered), Fraction(0)) / width
        root_lower, root_upper = dyadic_sqrt_enclosure(
            variance + eps, precision_bits
        )
        output_row: list[RationalInterval] = []
        for value, gamma in zip(centered, gains):
            if value >= 0:
                quotient = value / root_upper, value / root_lower
            else:
                quotient = value / root_lower, value / root_upper
            # Division by a distinct irrational enclosure endpoint in every
            # row otherwise creates mutually coprime rational denominators.
            # Outward dyadic rounding here keeps subsequent row means and
            # squares exact over a shared denominator without weakening the
            # enclosure guarantee.
            quotient = _dyadic_outward(quotient, precision_bits)
            output_row.append(
                _dyadic_outward(
                    _multiply_scalar(quotient, gamma), precision_bits
                )
            )
        normalized.append(output_row)

    energy_lower = Fraction(0)
    energy_upper = Fraction(0)
    for feature in range(width):
        mean_lower = sum(
            (normalized[row][feature][0] for row in range(row_count)),
            Fraction(0),
        ) / row_count
        mean_upper = sum(
            (normalized[row][feature][1] for row in range(row_count)),
            Fraction(0),
        ) / row_count
        for row in range(row_count):
            value_lower, value_upper = normalized[row][feature]
            square_lower, square_upper = _square_interval(
                (value_lower - mean_upper, value_upper - mean_lower)
            )
            energy_lower += square_lower
            energy_upper += square_upper
    return energy_lower, energy_upper


def represented_input_class(
    interval: RationalInterval, *, input_rows_identical: bool
) -> str:
    """Classify what a represented-zero LayerNorm output establishes.

    If its represented input rows are identical, the zero is the exact image
    of an already coalesced input and is not evidence that this LayerNorm
    reached a face from a positive real input.  Otherwise a positive lower
    bound certifies a positive real output despite native rounding.
    """

    lower, upper = interval
    if lower < 0 or upper < lower:
        raise ValueError("invalid nonnegative energy enclosure")
    if input_rows_identical:
        if upper != 0:
            raise ValueError("identical input rows must have zero energy")
        return "REPRESENTED_INPUT_ZERO_IMAGE"
    if lower > 0:
        return "REPRESENTED_INPUT_POSITIVE"
    return "REPRESENTED_INPUT_UNRESOLVED"


def enclosure_class(interval: RationalInterval) -> str:
    """Classify an energy enclosure into exact face, positive, or unresolved."""

    lower, upper = interval
    if lower < 0 or upper < lower:
        raise ValueError("invalid nonnegative energy enclosure")
    if upper == 0:
        return "CERTIFIED_EXACT_FACE"
    if lower > 0:
        return "CERTIFIED_POSITIVE"
    return "OBSERVER_NEAR_FACE"
