"""Analytic interval derivatives for a rational PLDR residual row map.

The module evaluates the implemented SiLU-gated affine composition and
stabilized row LayerNorm with second-order interval jets.  The Jacobian
intervals bound exact real derivatives on a complete rational row box.  The
Hessian tensor supplies a conservative derivative-Lipschitz modulus for the
finite-cover theorem.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path

from validated_interval import (
    BACKEND,
    RationalInterval,
    as_fraction,
    interval_sum,
    nonnegative_sqrt_upper,
    rational_object,
    sqrt_enclosure,
)


ZERO = RationalInterval.point(0)
ONE = RationalInterval.point(1)


@dataclass(frozen=True)
class Jet2:
    value: RationalInterval
    gradient: tuple[RationalInterval, ...]
    hessian: tuple[tuple[RationalInterval, ...], ...]

    @property
    def dimension(self):
        return len(self.gradient)

    @classmethod
    def constant(cls, value, dimension):
        return cls(
            RationalInterval.from_object(value),
            tuple(ZERO for _ in range(dimension)),
            tuple(tuple(ZERO for _ in range(dimension))
                  for _ in range(dimension)),
        )

    @classmethod
    def variable(cls, interval, index, dimension):
        gradient = tuple(ONE if j == index else ZERO
                         for j in range(dimension))
        return cls(
            RationalInterval.from_object(interval), gradient,
            tuple(tuple(ZERO for _ in range(dimension))
                  for _ in range(dimension)),
        )

    def _coerce(self, other):
        return other if isinstance(other, Jet2) else Jet2.constant(
            other, self.dimension)

    def __neg__(self):
        return Jet2(
            -self.value,
            tuple(-entry for entry in self.gradient),
            tuple(tuple(-entry for entry in row) for row in self.hessian),
        )

    def __add__(self, other):
        other = self._coerce(other)
        return Jet2(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.gradient, other.gradient)),
            tuple(tuple(a + b for a, b in zip(arow, brow))
                  for arow, brow in zip(self.hessian, other.hessian)),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        dimension = self.dimension
        return Jet2(
            self.value * other.value,
            tuple(
                self.gradient[i] * other.value
                + self.value * other.gradient[i]
                for i in range(dimension)
            ),
            tuple(tuple(
                self.hessian[i][j] * other.value
                + self.gradient[i] * other.gradient[j]
                + self.gradient[j] * other.gradient[i]
                + self.value * other.hessian[i][j]
                for j in range(dimension)
            ) for i in range(dimension)),
        )

    __rmul__ = __mul__

    def square(self):
        result = self * self
        return Jet2(self.value.square(), result.gradient, result.hessian)

    def compose(self, value, first, second):
        value = RationalInterval.from_object(value)
        first = RationalInterval.from_object(first)
        second = RationalInterval.from_object(second)
        dimension = self.dimension
        return Jet2(
            value,
            tuple(first * self.gradient[i] for i in range(dimension)),
            tuple(tuple(
                second * self.gradient[i] * self.gradient[j]
                + first * self.hessian[i][j]
                for j in range(dimension)
            ) for i in range(dimension)),
        )

    def reciprocal(self):
        inverse = self.value.reciprocal()
        return self.compose(
            inverse,
            -(inverse.square()),
            RationalInterval.point(2) * inverse * inverse * inverse,
        )

    def __truediv__(self, other):
        other = self._coerce(other)
        return self * other.reciprocal()

    def sqrt(self):
        root = self.value.sqrt()
        first = ONE / (RationalInterval.point(2) * root)
        second = -ONE / (
            RationalInterval.point(4) * self.value * root)
        return self.compose(root, first, second)

    def exp(self):
        value = self.value.exp()
        return self.compose(value, value, value)

    def sigmoid(self):
        value = ONE / (ONE + (-self.value).exp())
        first = value * (ONE - value)
        second = first * (ONE - RationalInterval.point(2) * value)
        return self.compose(value, first, second)

    def silu(self):
        return self * self.sigmoid()


def _jet_scalar(value, dimension):
    return value if isinstance(value, Jet2) else Jet2.constant(value, dimension)


def _vector(values, dimension):
    return tuple(_jet_scalar(value, dimension) for value in values)


def affine(matrix, vector, bias):
    if not matrix or any(len(row) != len(vector) for row in matrix):
        raise ValueError("affine matrix dimensions disagree")
    if len(bias) != len(matrix):
        raise ValueError("affine bias dimensions disagree")
    dimension = vector[0].dimension
    matrix = tuple(tuple(_jet_scalar(entry, dimension) for entry in row)
                   for row in matrix)
    bias = tuple(_jet_scalar(entry, dimension) for entry in bias)
    return tuple(
        sum((coefficient * entry for coefficient, entry
             in zip(row, vector)), bias[index])
        for index, row in enumerate(matrix)
    )


def glu(block, vector):
    left = affine(block["w1"], vector, block["b1"])
    right = affine(block["w2"], vector, block["b2"])
    hidden = tuple(a.silu() * b for a, b in zip(left, right))
    return affine(block["w3"], hidden, block["b3"])


def _intersect_value(jet, enclosure):
    enclosure = RationalInterval.from_object(enclosure)
    lower = max(jet.value.lower, enclosure.lower)
    upper = min(jet.value.upper, enclosure.upper)
    if lower > upper:
        raise ArithmeticError("independent interval enclosures are disjoint")
    return Jet2(RationalInterval(lower, upper), jet.gradient, jet.hessian)


def layer_norm(vector, gamma, beta, epsilon):
    feature_dimension = len(vector)
    if feature_dimension < 2:
        raise ValueError("row LayerNorm needs width at least two")
    if len(gamma) != feature_dimension or len(beta) != feature_dimension:
        raise ValueError("LayerNorm parameter dimensions disagree")
    jet_dimension = vector[0].dimension
    gamma = _vector(gamma, jet_dimension)
    beta = _vector(beta, jet_dimension)
    mean = (
        sum(vector, Jet2.constant(0, jet_dimension)) / feature_dimension
    )
    centered = tuple(entry - mean for entry in vector)
    variance = sum(
        (entry.square() for entry in centered),
        Jet2.constant(0, jet_dimension),
    ) / feature_dimension
    denominator = (variance + _jet_scalar(epsilon, jet_dimension)).sqrt()
    raw = tuple(
        gamma[index] * centered[index] / denominator + beta[index]
        for index in range(feature_dimension)
    )
    normalized_upper = nonnegative_sqrt_upper(Fraction(feature_dimension))
    normalized = RationalInterval(-normalized_upper, normalized_upper)
    return tuple(
        _intersect_value(
            entry,
            beta[index].value + gamma[index].value * normalized,
        )
        for index, entry in enumerate(raw)
    )


def residual_unit(unit, vector):
    transformed = vector
    for block in unit["glu_blocks"]:
        transformed = glu(block, transformed)
    if len(transformed) != len(vector):
        raise ValueError("residual unit output width differs from its input")
    residual = tuple(a + b for a, b in zip(vector, transformed))
    return layer_norm(
        residual, unit["gamma"], unit["beta"], unit["epsilon"])


def interval_row_map(specification, row_box):
    dimension = int(specification["width"])
    if dimension != len(row_box):
        raise ValueError("row box does not match the declared width")
    vector = tuple(
        Jet2.variable(RationalInterval.from_object(interval), index, dimension)
        for index, interval in enumerate(row_box)
    )
    for unit in specification["residual_units"]:
        vector = residual_unit(unit, vector)
    return vector


def _segment_value(left, right, segment, dimension):
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("parameter-segment specification keys disagree")
        return {
            name: _segment_value(left[name], right[name], segment, dimension)
            for name in left
        }
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise ValueError("parameter-segment arrays disagree")
        return [
            _segment_value(a, b, segment, dimension)
            for a, b in zip(left, right)
        ]
    a = as_fraction(left)
    b = as_fraction(right)
    return Jet2.constant(a, dimension) + (b - a) * segment


def interval_joint_row_map(specification_before, specification_after,
                           row_box):
    width = int(specification_before["width"])
    if int(specification_after["width"]) != width or len(row_box) != width:
        raise ValueError("joint row-map specifications disagree in width")
    dimension = width + 1
    segment = Jet2.variable(
        RationalInterval(Fraction(0), Fraction(1)), 0, dimension)
    before_units = specification_before["residual_units"]
    after_units = specification_after["residual_units"]
    units = _segment_value(before_units, after_units, segment, dimension)
    vector = tuple(
        Jet2.variable(
            RationalInterval.from_object(interval), index + 1, dimension)
        for index, interval in enumerate(row_box)
    )
    for unit in units:
        vector = residual_unit(unit, vector)
    return vector


def _operator_upper(jacobian):
    absolute = tuple(tuple(entry.maximum_absolute() for entry in row)
                     for row in jacobian)
    one_norm = max(
        sum((absolute[i][j] for i in range(len(absolute))), Fraction(0))
        for j in range(len(absolute[0]))
    )
    infinity_norm = max(sum(row, Fraction(0)) for row in absolute)
    return nonnegative_sqrt_upper(one_norm * infinity_norm)


def model_row_map_specification(model, layer_index):
    layers = model.decoder.dec_layers
    if not 0 <= layer_index < len(layers):
        raise IndexError("layer_index is outside the decoder")
    units = []
    for unit in layers[layer_index].mha1.reslayerAs:
        blocks = []
        for dense in unit.denseAs:
            activation = getattr(dense.activation, "__name__", "")
            if activation not in {"silu", "SiLU"}:
                raise ValueError("validated row map requires SiLU")
            blocks.append({
                "w1": dense.gluw1.weight.detach().cpu().tolist(),
                "b1": dense.gluw1.bias.detach().cpu().tolist(),
                "w2": dense.gluw2.weight.detach().cpu().tolist(),
                "b2": dense.gluw2.bias.detach().cpu().tolist(),
                "w3": dense.gluw3.weight.detach().cpu().tolist(),
                "b3": dense.gluw3.bias.detach().cpu().tolist(),
            })
        layernorm = unit.layernormA
        if not layernorm.elementwise_affine:
            raise ValueError("validated row map requires affine LayerNorm")
        units.append({
            "glu_blocks": blocks,
            "gamma": layernorm.weight.detach().cpu().tolist(),
            "beta": layernorm.bias.detach().cpu().tolist(),
            "epsilon": float(layernorm.eps),
        })
    return {"width": len(units[0]["gamma"]), "residual_units": units}


def certify_joint_row_map(specification_before, specification_after, row_box,
                          *, include_intervals=False):
    output = interval_joint_row_map(
        specification_before, specification_after, row_box)
    width = len(output)
    row_jacobian = tuple(
        tuple(component.gradient[index + 1] for index in range(width))
        for component in output
    )
    row_hessian = tuple(
        tuple(tuple(
            component.hessian[i + 1][j + 1] for j in range(width)
        ) for i in range(width))
        for component in output
    )
    mixed = tuple(
        tuple(component.hessian[0][index + 1] for index in range(width))
        for component in output
    )
    hessian_squared = sum((
        entry.maximum_absolute() ** 2
        for matrix in row_hessian for row in matrix for entry in row
    ), Fraction(0))
    mixed_squared = sum((
        entry.maximum_absolute() ** 2
        for row in mixed for entry in row
    ), Fraction(0))
    result = {
        "schema_version": "pldr-validated-joint-row-map-v1",
        "backend": BACKEND,
        "width": width,
        "parameter_segment": {
            "lower": rational_object(Fraction(0)),
            "upper": rational_object(Fraction(1)),
        },
        "row_box": [RationalInterval.from_object(entry).to_object()
                    for entry in row_box],
        "jacobian_operator_upper": rational_object(
            _operator_upper(row_jacobian)),
        "derivative_lipschitz_upper": rational_object(
            nonnegative_sqrt_upper(hessian_squared)),
        "parameter_row_mixed_upper": rational_object(
            nonnegative_sqrt_upper(mixed_squared)),
        "derivation": (
            "second_order_exact_rational_interval_jet_on_affine_"
            "checkpoint_parameter_segment_and_convex_row_box"
        ),
    }
    if include_intervals:
        result.update({
            "output_intervals": [entry.value.to_object() for entry in output],
            "row_jacobian_intervals": [
                [entry.to_object() for entry in row] for row in row_jacobian
            ],
            "row_hessian_intervals": [
                [[entry.to_object() for entry in row] for row in matrix]
                for matrix in row_hessian
            ],
            "parameter_row_mixed_intervals": [
                [entry.to_object() for entry in row] for row in mixed
            ],
        })
    return result




NORM_BOUND_DECIMAL_DIGITS = 80


def _outward_decimal_upper(value, digits=NORM_BOUND_DECIMAL_DIGITS):
    value = as_fraction(value)
    if value < 0:
        raise ValueError("outward upper rounding requires a nonnegative value")
    scale = 10 ** int(digits)
    quotient, remainder = divmod(value.numerator * scale, value.denominator)
    if remainder:
        quotient += 1
    return Fraction(quotient, scale)


@dataclass(frozen=True)
class NormJet3:
    """Outward-rounded exact rational joint derivative upper bounds."""

    value: Fraction
    first: Fraction
    second: Fraction
    third: Fraction

    def __post_init__(self):
        for name in ("value", "first", "second", "third"):
            value = as_fraction(getattr(self, name))
            if value < 0:
                raise ValueError("norm-jet bounds must be nonnegative")
            object.__setattr__(self, name, _outward_decimal_upper(value))

    def add(self, other):
        return NormJet3(
            self.value + other.value,
            self.first + other.first,
            self.second + other.second,
            self.third + other.third,
        )


def _flatten_scalars(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_scalars(item)
    else:
        yield as_fraction(value)


def _euclidean_upper(value):
    squared = sum(
        (entry * entry for entry in _flatten_scalars(value)), Fraction(0)
    )
    return nonnegative_sqrt_upper(squared)


def _difference(left, right):
    if isinstance(left, (list, tuple)):
        if not isinstance(right, (list, tuple)) or len(left) != len(right):
            raise ValueError("parameter segment arrays disagree")
        return [_difference(a, b) for a, b in zip(left, right)]
    return as_fraction(right) - as_fraction(left)


def _maximum_endpoint_norm(left, right):
    return max(_euclidean_upper(left), _euclidean_upper(right))


def _affine_norm(before, after, vector):
    weight = _maximum_endpoint_norm(before["weight"], after["weight"])
    bias = _maximum_endpoint_norm(before["bias"], after["bias"])
    delta_weight = _euclidean_upper(
        _difference(before["weight"], after["weight"]))
    delta_bias = _euclidean_upper(
        _difference(before["bias"], after["bias"]))
    return NormJet3(
        weight * vector.value + bias,
        weight * vector.first + delta_weight * vector.value + delta_bias,
        weight * vector.second + 2 * delta_weight * vector.first,
        weight * vector.third + 3 * delta_weight * vector.second,
    )


def _silu_norm(vector):
    first = Fraction(3, 2)
    second = Fraction(1)
    third = Fraction(5, 4)
    return NormJet3(
        vector.value,
        first * vector.first,
        second * vector.first ** 2 + first * vector.second,
        third * vector.first ** 3
        + 3 * second * vector.first * vector.second
        + first * vector.third,
    )


def _product_norm(left, right):
    return NormJet3(
        left.value * right.value,
        left.first * right.value + left.value * right.first,
        left.second * right.value
        + 2 * left.first * right.first
        + left.value * right.second,
        left.third * right.value
        + 3 * left.second * right.first
        + 3 * left.first * right.second
        + left.value * right.third,
    )


def _glu_norm(before, after, vector):
    left = _affine_norm(
        {"weight": before["w1"], "bias": before["b1"]},
        {"weight": after["w1"], "bias": after["b1"]},
        vector,
    )
    right = _affine_norm(
        {"weight": before["w2"], "bias": before["b2"]},
        {"weight": after["w2"], "bias": after["b2"]},
        vector,
    )
    hidden = _product_norm(_silu_norm(left), right)
    return _affine_norm(
        {"weight": before["w3"], "bias": before["b3"]},
        {"weight": after["w3"], "bias": after["b3"]},
        hidden,
    )


def _centered_variance_lower(vector):
    """Pairwise-identity lower bound for variance on one interval tube."""

    feature_dimension = len(vector)
    total = Fraction(0)
    for left in range(feature_dimension):
        for right in range(left + 1, feature_dimension):
            difference = vector[left].value - vector[right].value
            if difference.lower > 0:
                separation = difference.lower
            elif difference.upper < 0:
                separation = -difference.upper
            else:
                separation = Fraction(0)
            total += separation ** 2
    return total / Fraction(feature_dimension ** 2)


def _joint_layernorm_denominator_lowers_jet_reference(
        specification_before, specification_after, row_box):
    """Certify every LayerNorm denominator on the full joint tube."""

    width = int(specification_before["width"])
    dimension = width + 1
    segment = Jet2.variable(
        RationalInterval(Fraction(0), Fraction(1)), 0, dimension)
    units = _segment_value(
        specification_before["residual_units"],
        specification_after["residual_units"],
        segment,
        dimension,
    )
    vector = tuple(
        Jet2.variable(
            RationalInterval.from_object(interval), index + 1, dimension)
        for index, interval in enumerate(row_box)
    )
    lowers = []
    for unit in units:
        transformed = vector
        for block in unit["glu_blocks"]:
            transformed = glu(block, transformed)
        residual = tuple(
            source + change for source, change in zip(vector, transformed))
        epsilon = unit["epsilon"]
        epsilon_lower = (
            epsilon.value.lower if isinstance(epsilon, Jet2)
            else RationalInterval.from_object(epsilon).lower
        )
        denominator_lower, _ = sqrt_enclosure(
            _centered_variance_lower(residual) + epsilon_lower)
        if denominator_lower <= 0:
            raise ValueError("LayerNorm tube denominator is not positive")
        lowers.append(denominator_lower)
        vector = layer_norm(
            residual, unit["gamma"], unit["beta"], epsilon)
    return tuple(lowers)


VALUE_BOUND_DECIMAL_DIGITS = 18
VALUE_BOUND_SCALE = 10 ** VALUE_BOUND_DECIMAL_DIGITS
VALUE_EXP_TOLERANCE = Fraction(1, 10 ** 22)


def _outward_value_interval(value):
    """Enlarge an interval to a fixed decimal mesh using exact arithmetic."""

    interval = RationalInterval.from_object(value)
    lower_steps = (
        interval.lower.numerator * VALUE_BOUND_SCALE
    ) // interval.lower.denominator
    upper_steps = -(
        (-interval.upper.numerator * VALUE_BOUND_SCALE)
        // interval.upper.denominator
    )
    return RationalInterval(
        Fraction(lower_steps, VALUE_BOUND_SCALE),
        Fraction(upper_steps, VALUE_BOUND_SCALE),
    )


def _value_parameter_box(before, after):
    if isinstance(before, dict):
        if not isinstance(after, dict) or set(before) != set(after):
            raise ValueError("parameter value boxes disagree")
        return {
            name: _value_parameter_box(before[name], after[name])
            for name in before
        }
    if isinstance(before, (list, tuple)):
        if not isinstance(after, (list, tuple)) or len(before) != len(after):
            raise ValueError("parameter value arrays disagree")
        return [
            _value_parameter_box(left, right)
            for left, right in zip(before, after)
        ]
    return _outward_value_interval(
        RationalInterval.point(before).hull(
            RationalInterval.point(after)))


def _value_affine(matrix, vector, bias):
    return tuple(
        _outward_value_interval(
            interval_sum(
                coefficient * entry
                for coefficient, entry in zip(row, vector)
            ) + bias[index])
        for index, row in enumerate(matrix)
    )


def _value_sigmoid(value):
    exponential = _outward_value_interval(
        (-value).exp(tolerance=VALUE_EXP_TOLERANCE))
    return _outward_value_interval(ONE / (ONE + exponential))


def _value_glu(block, vector):
    left = _value_affine(block["w1"], vector, block["b1"])
    right = _value_affine(block["w2"], vector, block["b2"])
    hidden = tuple(
        _outward_value_interval(
            left_value * _value_sigmoid(left_value) * right_value)
        for left_value, right_value in zip(left, right)
    )
    return _value_affine(block["w3"], hidden, block["b3"])


def _value_variance_lower(vector):
    feature_dimension = len(vector)
    total = Fraction(0)
    for left in range(feature_dimension):
        for right in range(left + 1, feature_dimension):
            separation = (vector[left] - vector[right]).minimum_absolute()
            total += separation ** 2
    return _outward_value_interval(RationalInterval.point(
        total / Fraction(feature_dimension ** 2))).lower


def _value_layernorm(vector, gamma, beta, epsilon):
    feature_dimension = len(vector)
    mean = _outward_value_interval(
        interval_sum(vector) / feature_dimension)
    centered = tuple(
        _outward_value_interval(entry - mean) for entry in vector)
    variance = _outward_value_interval(
        interval_sum(
            _outward_value_interval(entry.square())
            for entry in centered
        ) / feature_dimension)
    epsilon = _outward_value_interval(epsilon)
    lower_squared = _outward_value_interval(RationalInterval.point(
        _value_variance_lower(vector) + epsilon.lower)).lower
    lower, _ = sqrt_enclosure(
        lower_squared, digits=VALUE_BOUND_DECIMAL_DIGITS)
    _, upper = sqrt_enclosure(
        variance.upper + epsilon.upper,
        digits=VALUE_BOUND_DECIMAL_DIGITS,
    )
    lower = _outward_value_interval(RationalInterval.point(lower)).lower
    upper = _outward_value_interval(RationalInterval.point(upper)).upper
    denominator = RationalInterval(lower, upper)
    dimension_root = nonnegative_sqrt_upper(
        Fraction(feature_dimension), digits=VALUE_BOUND_DECIMAL_DIGITS)
    normalized = _outward_value_interval(
        RationalInterval(-dimension_root, dimension_root))
    output = []
    for index, entry in enumerate(centered):
        raw = _outward_value_interval(
            gamma[index] * entry / denominator + beta[index])
        structural = _outward_value_interval(
            beta[index] + gamma[index] * normalized)
        intersection = RationalInterval(
            max(raw.lower, structural.lower),
            min(raw.upper, structural.upper),
        )
        output.append(_outward_value_interval(intersection))
    return tuple(output), lower


def _joint_layernorm_denominator_lowers(
        specification_before, specification_after, row_box):
    """Value-only exact interval pass for whole-tube denominators."""

    before_units = specification_before["residual_units"]
    after_units = specification_after["residual_units"]
    units = _value_parameter_box(before_units, after_units)
    vector = tuple(
        _outward_value_interval(interval) for interval in row_box)
    lowers = []
    for unit in units:
        transformed = vector
        for block in unit["glu_blocks"]:
            transformed = _value_glu(block, transformed)
        residual = tuple(
            source + change for source, change in zip(vector, transformed))
        vector, denominator_lower = _value_layernorm(
            residual, unit["gamma"], unit["beta"], unit["epsilon"])
        if denominator_lower <= 0:
            raise ValueError("LayerNorm tube denominator is not positive")
        lowers.append(denominator_lower)
    return tuple(lowers)






def _layernorm_norm(
        before, after, vector, feature_dimension, denominator_lower):
    denominator_lower = as_fraction(denominator_lower)
    if denominator_lower <= 0:
        raise ValueError("LayerNorm denominator lower bound must be positive")
    dimension_root = nonnegative_sqrt_upper(Fraction(feature_dimension))
    gamma = max(
        max(abs(entry) for entry in _flatten_scalars(before["gamma"])),
        max(abs(entry) for entry in _flatten_scalars(after["gamma"])),
    )
    delta_gamma = max(
        (abs(entry) for entry in _flatten_scalars(
            _difference(before["gamma"], after["gamma"]))),
        default=Fraction(0),
    )
    beta = _maximum_endpoint_norm(before["beta"], after["beta"])
    delta_beta = _euclidean_upper(
        _difference(before["beta"], after["beta"]))
    local_first = Fraction(1, 1) / denominator_lower
    local_second = Fraction(6, 1) / (
        dimension_root * denominator_lower ** 2)
    local_third = Fraction(36, 1) / (
        Fraction(feature_dimension) * denominator_lower ** 3)
    normalized_first = local_first * vector.first
    normalized_second = (
        local_second * vector.first ** 2 + local_first * vector.second
    )
    normalized_third = (
        local_third * vector.first ** 3
        + 3 * local_second * vector.first * vector.second
        + local_first * vector.third
    )
    return NormJet3(
        gamma * dimension_root + beta,
        gamma * normalized_first + delta_gamma * dimension_root + delta_beta,
        gamma * normalized_second + 2 * delta_gamma * normalized_first,
        gamma * normalized_third + 3 * delta_gamma * normalized_second,
    )


def certify_joint_row_map_norms(specification_before, specification_after,
                                row_box):
    """Certify joint (segment,row) derivatives with exact norm algebra.

    Weight matrices use their exact-rational Frobenius norms, hence bound
    spectral norms. The checkpoint interpolation is affine in one segment
    coordinate. Product and composition rules then propagate Euclidean
    operator bounds through third order without materializing derivative
    tensors.
    """

    width = int(specification_before["width"])
    if int(specification_after["width"]) != width or len(row_box) != width:
        raise ValueError("joint row-map specifications disagree in width")
    before_units = specification_before["residual_units"]
    after_units = specification_after["residual_units"]
    if len(before_units) != len(after_units):
        raise ValueError("joint row-map residual-unit counts disagree")
    denominator_lowers = _joint_layernorm_denominator_lowers(
        specification_before, specification_after, row_box)
    if len(denominator_lowers) != len(before_units):
        raise ArithmeticError("LayerNorm denominator ledger is incomplete")
    radius_squared = sum((
        RationalInterval.from_object(entry).maximum_absolute() ** 2
        for entry in row_box
    ), Fraction(0))
    vector = NormJet3(
        nonnegative_sqrt_upper(radius_squared), Fraction(1),
        Fraction(0), Fraction(0),
    )
    unit_bounds = []
    for index, (before, after, denominator_lower) in enumerate(zip(
        before_units, after_units, denominator_lowers,
    )):
        before_blocks = before["glu_blocks"]
        after_blocks = after["glu_blocks"]
        if len(before_blocks) != len(after_blocks):
            raise ValueError("joint row-map GLU block counts disagree")
        transformed = vector
        for block_before, block_after in zip(before_blocks, after_blocks):
            transformed = _glu_norm(block_before, block_after, transformed)
        residual = vector.add(transformed)
        vector = _layernorm_norm(
            before, after, residual, width, denominator_lower)
        unit_bounds.append({
            "index": index,
            "layernorm_denominator_lower": rational_object(
                denominator_lower),
            "value_upper": rational_object(vector.value),
            "joint_first_derivative_upper": rational_object(vector.first),
            "joint_second_derivative_upper": rational_object(vector.second),
            "joint_third_derivative_upper": rational_object(vector.third),
        })
    return {
        "schema_version": "pldr-validated-joint-row-map-norm-v1",
        "backend": BACKEND,
        "width": width,
        "row_box": [RationalInterval.from_object(entry).to_object()
                    for entry in row_box],
        "value_upper": rational_object(vector.value),
        "jacobian_operator_upper": rational_object(vector.first),
        "derivative_lipschitz_upper": rational_object(vector.second),
        "parameter_row_mixed_upper": rational_object(vector.second),
        "joint_third_derivative_upper": rational_object(vector.third),
        "residual_unit_bounds": unit_bounds,
        "derivation": (
            "fixed_mesh_outward_value_pass_then_exact_rational_frobenius_"
            "and_composition_norm_bounds_on_affine_checkpoint_segment_"
            "times_convex_row_box"
        ),
    }


def certify_row_map(specification, row_box):
    output = interval_row_map(specification, row_box)
    jacobian = tuple(tuple(entry for entry in component.gradient)
                     for component in output)
    hessian = tuple(component.hessian for component in output)
    jacobian_upper = _operator_upper(jacobian)
    hessian_frobenius_squared = sum((
        entry.maximum_absolute() ** 2
        for output_hessian in hessian
        for row in output_hessian
        for entry in row
    ), Fraction(0))
    derivative_modulus = nonnegative_sqrt_upper(hessian_frobenius_squared)
    return {
        "schema_version": "pldr-validated-row-map-v1",
        "backend": BACKEND,
        "width": len(output),
        "row_box": [RationalInterval.from_object(entry).to_object()
                    for entry in row_box],
        "output_intervals": [entry.value.to_object() for entry in output],
        "jacobian_intervals": [
            [entry.to_object() for entry in row] for row in jacobian
        ],
        "hessian_intervals": [
            [[entry.to_object() for entry in row] for row in matrix]
            for matrix in hessian
        ],
        "jacobian_operator_upper": rational_object(jacobian_upper),
        "derivative_lipschitz_upper": rational_object(derivative_modulus),
        "derivation": (
            "second_order_interval_jet_through_affine_silu_glu_"
            "residual_stabilized_layernorm"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specification", required=True)
    parser.add_argument("--row-box", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    specification = json.loads(
        Path(arguments.specification).read_text(encoding="utf-8"))
    row_box = json.loads(Path(arguments.row_box).read_text(encoding="utf-8"))
    result = certify_row_map(specification, row_box)
    Path(arguments.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
