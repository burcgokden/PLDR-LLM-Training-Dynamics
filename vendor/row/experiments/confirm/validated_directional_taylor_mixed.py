"""Exact mixed-jet certificate for a live row-Jacobian Taylor edge.

Only the derivatives needed by the integral remainder are propagated:
the first two derivatives in the path coordinate, every physical-row
probe derivative, and their first and second path derivatives.  This is
algebraically identical to extracting ``d_s^2 d_u`` from a full third-order
jet, without constructing unused third-order components.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from validated_interval import (
    BACKEND,
    RationalInterval,
    as_fraction,
    nonnegative_sqrt_upper,
    rational_object,
)


ZERO = RationalInterval.point(0)
ONE = RationalInterval.point(1)
TWO = RationalInterval.point(2)
THREE = RationalInterval.point(3)
FOUR = RationalInterval.point(4)
SIX = RationalInterval.point(6)
EIGHT = RationalInterval.point(8)
OUTWARD_MESH = 10 ** 18


def _outward(value):
    """Round outward to a fixed rational mesh without losing enclosure."""

    interval = RationalInterval.from_object(value)
    lower_steps = (
        interval.lower.numerator * OUTWARD_MESH
    ) // interval.lower.denominator
    upper_steps = -(
        (-interval.upper.numerator * OUTWARD_MESH)
        // interval.upper.denominator
    )
    return RationalInterval(
        Fraction(lower_steps, OUTWARD_MESH),
        Fraction(upper_steps, OUTWARD_MESH),
    )


@dataclass(frozen=True)
class MixedJet:
    """Interval jet retaining exactly value, s, ss, u, su, and ssu."""

    value: RationalInterval
    path: RationalInterval
    path2: RationalInterval
    probe: tuple[RationalInterval, ...]
    path_probe: tuple[RationalInterval, ...]
    path2_probe: tuple[RationalInterval, ...]

    def __post_init__(self):
        for name in ("value", "path", "path2"):
            object.__setattr__(self, name, _outward(getattr(self, name)))
        for name in ("probe", "path_probe", "path2_probe"):
            object.__setattr__(
                self,
                name,
                tuple(_outward(entry) for entry in getattr(self, name)),
            )

    @property
    def dimension(self):
        return len(self.probe)

    @classmethod
    def constant(cls, value, dimension):
        zeros = tuple(ZERO for _ in range(dimension))
        return cls(
            RationalInterval.from_object(value), ZERO, ZERO,
            zeros, zeros, zeros,
        )

    @classmethod
    def path_variable(cls, interval, dimension):
        value = cls.constant(interval, dimension)
        return cls(
            value.value, ONE, ZERO, value.probe,
            value.path_probe, value.path2_probe,
        )

    @classmethod
    def probe_variable(cls, index, dimension):
        value = cls.constant(0, dimension)
        probe = tuple(
            ONE if coordinate == index else ZERO
            for coordinate in range(dimension)
        )
        return cls(
            value.value, ZERO, ZERO, probe,
            value.path_probe, value.path2_probe,
        )

    def _coerce(self, other):
        return other if isinstance(other, MixedJet) else MixedJet.constant(
            other, self.dimension)

    def __neg__(self):
        return MixedJet(
            -self.value, -self.path, -self.path2,
            tuple(-entry for entry in self.probe),
            tuple(-entry for entry in self.path_probe),
            tuple(-entry for entry in self.path2_probe),
        )

    def __add__(self, other):
        other = self._coerce(other)
        return MixedJet(
            self.value + other.value,
            self.path + other.path,
            self.path2 + other.path2,
            tuple(
                left + right for left, right
                in zip(self.probe, other.probe)
            ),
            tuple(
                left + right for left, right
                in zip(self.path_probe, other.path_probe)
            ),
            tuple(
                left + right for left, right
                in zip(self.path2_probe, other.path2_probe)
            ),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        probe = tuple(
            self.probe[i] * other.value
            + self.value * other.probe[i]
            for i in range(self.dimension)
        )
        path_probe = tuple(
            self.path_probe[i] * other.value
            + self.probe[i] * other.path
            + self.path * other.probe[i]
            + self.value * other.path_probe[i]
            for i in range(self.dimension)
        )
        path2_probe = tuple(
            self.path2_probe[i] * other.value
            + TWO * self.path_probe[i] * other.path
            + self.probe[i] * other.path2
            + self.path2 * other.probe[i]
            + TWO * self.path * other.path_probe[i]
            + self.value * other.path2_probe[i]
            for i in range(self.dimension)
        )
        return MixedJet(
            self.value * other.value,
            self.path * other.value + self.value * other.path,
            self.path2 * other.value
            + TWO * self.path * other.path
            + self.value * other.path2,
            probe, path_probe, path2_probe,
        )

    __rmul__ = __mul__

    def square(self):
        value = self * self
        return MixedJet(
            self.value.square(), value.path, value.path2,
            value.probe, value.path_probe, value.path2_probe,
        )

    def compose(self, value, first, second, third):
        value = _outward(value)
        first = _outward(first)
        second = _outward(second)
        third = _outward(third)
        probe = tuple(
            first * self.probe[i] for i in range(self.dimension))
        path_probe = tuple(
            second * self.path * self.probe[i]
            + first * self.path_probe[i]
            for i in range(self.dimension)
        )
        path2_probe = tuple(
            third * self.path.square() * self.probe[i]
            + second * (
                self.path2 * self.probe[i]
                + TWO * self.path * self.path_probe[i]
            )
            + first * self.path2_probe[i]
            for i in range(self.dimension)
        )
        return MixedJet(
            value,
            first * self.path,
            second * self.path.square() + first * self.path2,
            probe, path_probe, path2_probe,
        )

    def reciprocal(self):
        inverse = _outward(self.value.reciprocal())
        return self.compose(
            inverse,
            -(inverse.square()),
            TWO * inverse * inverse * inverse,
            -SIX * inverse * inverse * inverse * inverse,
        )

    def __truediv__(self, other):
        return self * self._coerce(other).reciprocal()

    def sqrt(self):
        root = _outward(self.value.sqrt(digits=20))
        first = ONE / (TWO * root)
        second = -ONE / (FOUR * self.value * root)
        third = THREE / (
            EIGHT * self.value * self.value * root)
        return self.compose(root, first, second, third)

    def exp(self):
        value = _outward(self.value.exp(tolerance=Fraction(1, 10**22)))
        return self.compose(value, value, value, value)

    def sigmoid(self):
        exponential = _outward((-self.value).exp(
            tolerance=Fraction(1, 10**22)))
        value = _outward(ONE / (ONE + exponential))
        first = value * (ONE - value)
        second = first * (ONE - TWO * value)
        third = first * (ONE - SIX * value + SIX * value.square())
        return self.compose(value, first, second, third)

    def silu(self):
        return self * self.sigmoid()


def _jet(value, dimension):
    return value if isinstance(value, MixedJet) else MixedJet.constant(
        value, dimension)


def _segment(left, right, parameter):
    dimension = parameter.dimension
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("directional parameter mappings disagree")
        return {
            name: _segment(left[name], right[name], parameter)
            for name in left
        }
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise ValueError("directional parameter arrays disagree")
        return [
            _segment(before, after, parameter)
            for before, after in zip(left, right)
        ]
    before = as_fraction(left)
    return MixedJet.constant(before, dimension) + (
        as_fraction(right) - before
    ) * parameter


def _affine(matrix, vector, bias):
    dimension = vector[0].dimension
    matrix = tuple(
        tuple(_jet(entry, dimension) for entry in row) for row in matrix)
    bias = tuple(_jet(entry, dimension) for entry in bias)
    return tuple(
        sum(
            (coefficient * entry for coefficient, entry in zip(row, vector)),
            bias[index],
        )
        for index, row in enumerate(matrix)
    )


def _glu(block, vector):
    left = _affine(block["w1"], vector, block["b1"])
    right = _affine(block["w2"], vector, block["b2"])
    hidden = tuple(
        left_value.silu() * right_value
        for left_value, right_value in zip(left, right)
    )
    return _affine(block["w3"], hidden, block["b3"])


def _intersect_value(jet, enclosure):
    enclosure = RationalInterval.from_object(enclosure)
    lower = max(jet.value.lower, enclosure.lower)
    upper = min(jet.value.upper, enclosure.upper)
    if lower > upper:
        raise ArithmeticError("directional LayerNorm enclosures are disjoint")
    return MixedJet(
        RationalInterval(lower, upper), jet.path, jet.path2,
        jet.probe, jet.path_probe, jet.path2_probe,
    )


def _layer_norm(vector, gamma, beta, epsilon):
    feature_dimension = len(vector)
    dimension = vector[0].dimension
    gamma = tuple(_jet(entry, dimension) for entry in gamma)
    beta = tuple(_jet(entry, dimension) for entry in beta)
    epsilon = _jet(epsilon, dimension)
    mean = sum(vector, MixedJet.constant(0, dimension)) / feature_dimension
    centered = tuple(entry - mean for entry in vector)
    variance = sum(
        (entry.square() for entry in centered),
        MixedJet.constant(0, dimension),
    ) / feature_dimension
    denominator = (variance + epsilon).sqrt()
    dimension_root = nonnegative_sqrt_upper(Fraction(feature_dimension))
    normalized = RationalInterval(-dimension_root, dimension_root)
    output = tuple(
        _intersect_value(
            gamma[index] * centered[index] / denominator + beta[index],
            beta[index].value + gamma[index].value * normalized,
        )
        for index in range(feature_dimension)
    )
    return output, denominator.value.lower


def _row_map(units, row):
    vector = row
    floors = []
    for unit in units:
        transformed = vector
        for block in unit["glu_blocks"]:
            transformed = _glu(block, transformed)
        residual = tuple(
            source + change for source, change in zip(vector, transformed))
        vector, floor = _layer_norm(
            residual, unit["gamma"], unit["beta"], unit["epsilon"])
        floors.append(floor)
    return vector, floors


def certify_directional_row_jacobian_remainder(
        specification_before, specification_after, source_row, target_row):
    """Bound one live edge's row-Jacobian remainder with exact intervals."""

    width = int(specification_before["width"])
    if (
        int(specification_after["width"]) != width
        or len(source_row) != width
        or len(target_row) != width
    ):
        raise ValueError("directional row-Jacobian inputs disagree in width")
    source = tuple(as_fraction(value) for value in source_row)
    target = tuple(as_fraction(value) for value in target_row)
    path = MixedJet.path_variable(
        RationalInterval(Fraction(0), Fraction(1)), width)
    units = _segment(
        specification_before["residual_units"],
        specification_after["residual_units"],
        path,
    )
    row = tuple(
        MixedJet.constant(source[index], width)
        + (target[index] - source[index]) * path
        + MixedJet.probe_variable(index, width)
        for index in range(width)
    )
    output, floors = _row_map(units, row)
    squared = Fraction(0)
    component_intervals = []
    for output_index, component in enumerate(output):
        for input_index, interval in enumerate(component.path2_probe):
            squared += interval.maximum_absolute() ** 2
            component_intervals.append({
                "output": output_index,
                "input": input_index,
                "interval": interval.to_object(),
            })
    upper = nonnegative_sqrt_upper(squared) / 2
    return {
        "schema_version": "pldr-directional-row-jacobian-taylor-v3",
        "backend": {
            **BACKEND,
            "derivative_endpoint_mesh": f"1/{OUTWARD_MESH}",
            "derivative_rounding": (
                "exact rational arithmetic with outward mesh coarsening"
            ),
        },
        "jet": "mixed_path2_physical_probe",
        "width": width,
        "mixed_component_count": len(component_intervals),
        "layernorm_denominator_lower": rational_object(min(floors)),
        "remainder_upper": rational_object(upper),
        "mixed_second_derivative_intervals": component_intervals,
        "derivation": (
            "exact_mixed_interval_jet_for_one_live_parameter_and_"
            "physical_row_segment"
        ),
    }
