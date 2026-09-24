"""Exact interval successor for the clipped-AdamW program state.

This module evaluates the implemented scalar operation order coordinate by
coordinate: clipping, moments, bias correction, adaptive denominator,
decoupled decay, intervention, optimizer clock, and scheduler state.  It also
propagates interval derivatives, so center errors and radius maps are outputs
of the constructor rather than caller-supplied certificate fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

from validated_interval import RationalInterval, as_fraction, rational_object


SCHEMA_VERSION = "pldr-program-state-cell-v1"


def _interval(value, label):
    try:
        return RationalInterval.from_object(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid interval for {label}") from error


def _vector(value, label, *, length=None):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty interval vector")
    result = tuple(_interval(entry, label) for entry in value)
    if length is not None and len(result) != length:
        raise ValueError(f"{label} has the wrong length")
    return result


def _point_vector(values):
    return tuple(RationalInterval.point(value) for value in values)


def _radius(interval):
    return interval.width / 2


def _center(interval):
    return interval.midpoint


def _maximum_distance(interval, center):
    center = as_fraction(center)
    return max(abs(interval.lower - center), abs(interval.upper - center))


@dataclass(frozen=True)
class IntervalJet:
    value: RationalInterval
    derivative: tuple[RationalInterval, ...]

    @classmethod
    def constant(cls, value, dimension):
        return cls(
            _interval(value, "jet constant"),
            tuple(RationalInterval.point(0) for _ in range(dimension)),
        )

    @classmethod
    def variable(cls, value, dimension, index):
        if not 0 <= index < dimension:
            raise IndexError("interval-jet variable index is out of range")
        derivative = [RationalInterval.point(0) for _ in range(dimension)]
        derivative[index] = RationalInterval.point(1)
        return cls(_interval(value, "jet variable"), tuple(derivative))

    def _coerce(self, other):
        if isinstance(other, IntervalJet):
            if len(other.derivative) != len(self.derivative):
                raise ValueError("interval-jet dimensions disagree")
            return other
        return IntervalJet.constant(other, len(self.derivative))

    def __neg__(self):
        return IntervalJet(-self.value, tuple(-entry for entry in self.derivative))

    def __add__(self, other):
        other = self._coerce(other)
        return IntervalJet(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.derivative, other.derivative)),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        other = self._coerce(other)
        return IntervalJet(
            self.value * other.value,
            tuple(
                a * other.value + self.value * b
                for a, b in zip(self.derivative, other.derivative)
            ),
        )

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = self.value.reciprocal()
        return IntervalJet(
            inverse,
            tuple(-entry * inverse.square() for entry in self.derivative),
        )

    def __truediv__(self, other):
        return self * self._coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self._coerce(other) / self

    def square(self):
        return self * self

    def sqrt(self):
        if self.value.lower <= 0:
            raise ValueError(
                "the adaptive second-moment cell reaches the square-root branch point")
        root = self.value.sqrt()
        return IntervalJet(
            root,
            tuple(entry / (2 * root) for entry in self.derivative),
        )


def _clip_jet(value, threshold):
    threshold = as_fraction(threshold)
    if threshold <= 0:
        raise ValueError("gradient clipping threshold must be positive")
    lower_boundary = -threshold
    upper_boundary = threshold
    interval = value.value
    if interval.upper < lower_boundary:
        return IntervalJet.constant(lower_boundary, len(value.derivative)), "LOW_CLIPPED"
    if interval.lower > upper_boundary:
        return IntervalJet.constant(upper_boundary, len(value.derivative)), "HIGH_CLIPPED"
    if lower_boundary < interval.lower and interval.upper < upper_boundary:
        return value, "UNCLIPPED"
    raise ValueError(
        "gradient interval touches or crosses a clipping boundary; split the cell")


def _adamw_coordinate(theta, first, second, gradient, intervention,
                      learning_rate, *, beta1, beta2, epsilon,
                      weight_decay, clip_value, bias1, bias2):
    clipped, branch = _clip_jet(gradient, clip_value)
    first_after = beta1 * first + (1 - beta1) * clipped
    second_after = beta2 * second + (1 - beta2) * clipped.square()
    first_hat = first_after / bias1
    second_hat = second_after / bias2
    denominator = second_hat.sqrt() + epsilon
    if denominator.value.lower <= 0:
        raise ValueError("Adam denominator interval contains zero")
    theta_after = (
        (1 - learning_rate * weight_decay) * theta
        - learning_rate * first_hat / denominator
        + intervention
    )
    return theta_after, first_after, second_after, branch


def _cell(value):
    required = {
        "schema_version", "theta", "first_moment", "second_moment",
        "optimizer_step", "scheduler_phase", "learning_rate", "cover_state",
        "intervention_state",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("program-state cell has missing or unknown fields")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unknown program-state schema")
    theta = _vector(value["theta"], "theta")
    first = _vector(value["first_moment"], "first_moment", length=len(theta))
    second = _vector(value["second_moment"], "second_moment", length=len(theta))
    if any(entry.lower < 0 for entry in second):
        raise ValueError("second-moment intervals must be nonnegative")
    step = value["optimizer_step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("optimizer_step must be a nonnegative integer")
    if not isinstance(value["scheduler_phase"], str) or not value["scheduler_phase"]:
        raise ValueError("scheduler_phase must be a nonempty string")
    learning_rate = _interval(value["learning_rate"], "learning_rate")
    if learning_rate.lower < 0:
        raise ValueError("learning-rate intervals must be nonnegative")
    for name in ("cover_state", "intervention_state"):
        if not isinstance(value[name], dict):
            raise ValueError(f"{name} must be a mapping")
    return {
        "theta": theta,
        "first_moment": first,
        "second_moment": second,
        "optimizer_step": step,
        "scheduler_phase": value["scheduler_phase"],
        "learning_rate": learning_rate,
        "cover_state": value["cover_state"],
        "intervention_state": value["intervention_state"],
    }


def _constant(value, name, *, lower=None, upper=None):
    result = as_fraction(value)
    if lower is not None and result < as_fraction(lower):
        raise ValueError(f"{name} is below its allowed range")
    if upper is not None and result > as_fraction(upper):
        raise ValueError(f"{name} is above its allowed range")
    return result


def interval_adamw_successor(
        *, source_cell, gradient_cell, beta1, beta2, epsilon,
        weight_decay, clip_value, intervention_cell, next_learning_rate,
        next_scheduler_phase, next_cover_state, next_intervention_state):
    """Evaluate one actual clipped-AdamW state-cell image and its derivatives."""

    source = _cell(source_cell)
    width = len(source["theta"])
    gradient = _vector(gradient_cell, "gradient_cell", length=width)
    intervention = _vector(
        intervention_cell, "intervention_cell", length=width)
    beta1 = _constant(beta1, "beta1", lower=0, upper=1)
    beta2 = _constant(beta2, "beta2", lower=0, upper=1)
    if beta1 >= 1 or beta2 >= 1:
        raise ValueError("Adam moment coefficients must be strictly below one")
    epsilon = _constant(epsilon, "epsilon", lower=0)
    if epsilon <= 0:
        raise ValueError("Adam epsilon must be strictly positive")
    weight_decay = _constant(weight_decay, "weight_decay", lower=0)
    clip_value = _constant(clip_value, "clip_value", lower=0)
    if clip_value <= 0:
        raise ValueError("clip_value must be strictly positive")
    next_learning_rate = _interval(next_learning_rate, "next_learning_rate")
    if next_learning_rate.lower < 0:
        raise ValueError("next learning rate must be nonnegative")
    if not isinstance(next_scheduler_phase, str) or not next_scheduler_phase:
        raise ValueError("next_scheduler_phase must be a nonempty string")
    if not isinstance(next_cover_state, dict) or not isinstance(
        next_intervention_state, dict,
    ):
        raise ValueError("successor cover and intervention states must be mappings")

    # State variables, exogenous gradient variables, intervention variables,
    # and the applied learning rate are all differentiated.  This makes the
    # source part of the positive map explicit.
    state_dimension = 3 * width
    total_dimension = 5 * width + 1
    theta = [IntervalJet.variable(value, total_dimension, index)
             for index, value in enumerate(source["theta"])]
    first = [IntervalJet.variable(value, total_dimension, width + index)
             for index, value in enumerate(source["first_moment"])]
    second = [IntervalJet.variable(value, total_dimension, 2 * width + index)
              for index, value in enumerate(source["second_moment"])]
    gradients = [IntervalJet.variable(
        value, total_dimension, state_dimension + index)
        for index, value in enumerate(gradient)]
    interventions = [IntervalJet.variable(
        value, total_dimension, state_dimension + width + index)
        for index, value in enumerate(intervention)]
    learning_rate = IntervalJet.variable(
        source["learning_rate"], total_dimension, total_dimension - 1)

    step_after = source["optimizer_step"] + 1
    bias1 = Fraction(1) - beta1 ** step_after
    bias2 = Fraction(1) - beta2 ** step_after
    if bias1 <= 0 or bias2 <= 0:
        raise ValueError("Adam bias-correction denominator is not positive")
    branches = []
    first_after = []
    second_after = []
    theta_after = []
    for theta_i, first_i, second_i, gradient_i, intervention_i in zip(
        theta, first, second, gradients, interventions,
    ):
        theta_i_after, first_i_after, second_i_after, branch = (
            _adamw_coordinate(
                theta_i, first_i, second_i, gradient_i, intervention_i,
                learning_rate, beta1=beta1, beta2=beta2,
                epsilon=epsilon, weight_decay=weight_decay,
                clip_value=clip_value, bias1=bias1, bias2=bias2,
            )
        )
        branches.append(branch)
        first_after.append(first_i_after)
        second_after.append(second_i_after)
        theta_after.append(theta_i_after)

    outputs = theta_after + first_after + second_after
    state_jacobian = []
    source_jacobian = []
    for output in outputs:
        state_jacobian.append([
            derivative.maximum_absolute()
            for derivative in output.derivative[:state_dimension]
        ])
        source_jacobian.append([
            derivative.maximum_absolute()
            for derivative in output.derivative[state_dimension:]
        ])
    state_radii = tuple(
        _radius(value) for value in (
            source["theta"] + source["first_moment"] + source["second_moment"]
        )
    )
    source_radii = tuple(
        _radius(value) for value in (
            gradient + intervention + (source["learning_rate"],)
        )
    )
    state_centers = tuple(
        _center(value) for value in (
            source["theta"] + source["first_moment"] + source["second_moment"]
        )
    )
    source_centers = tuple(
        _center(value) for value in (
            gradient + intervention + (source["learning_rate"],)
        )
    )

    # Exact center replay uses point intervals and the same operation graph.
    center_source = {
        "schema_version": SCHEMA_VERSION,
        "theta": [value.to_object() for value in _point_vector(
            state_centers[:width])],
        "first_moment": [value.to_object() for value in _point_vector(
            state_centers[width:2 * width])],
        "second_moment": [value.to_object() for value in _point_vector(
            state_centers[2 * width:])],
        "optimizer_step": source["optimizer_step"],
        "scheduler_phase": source["scheduler_phase"],
        "learning_rate": RationalInterval.point(source_centers[-1]).to_object(),
        "cover_state": source["cover_state"],
        "intervention_state": source["intervention_state"],
    }
    center_theta = [IntervalJet.constant(value, total_dimension)
                    for value in state_centers[:width]]
    center_first = [IntervalJet.constant(value, total_dimension)
                    for value in state_centers[width:2 * width]]
    center_second = [IntervalJet.constant(value, total_dimension)
                     for value in state_centers[2 * width:]]
    center_gradient = [IntervalJet.constant(value, total_dimension)
                       for value in source_centers[:width]]
    center_intervention = [IntervalJet.constant(value, total_dimension)
                           for value in source_centers[width:2 * width]]
    center_eta = IntervalJet.constant(source_centers[-1], total_dimension)
    center_theta_after = []
    center_first_after = []
    center_second_after = []
    for values in zip(
        center_theta, center_first, center_second,
        center_gradient, center_intervention,
    ):
        center_result = _adamw_coordinate(
            *values, center_eta, beta1=beta1, beta2=beta2,
            epsilon=epsilon, weight_decay=weight_decay,
            clip_value=clip_value, bias1=bias1, bias2=bias2,
        )
        center_theta_after.append(center_result[0].value.midpoint)
        center_first_after.append(center_result[1].value.midpoint)
        center_second_after.append(center_result[2].value.midpoint)
    center_images = tuple(
        center_theta_after + center_first_after + center_second_after)
    image_radii = tuple(
        _maximum_distance(output.value, center)
        for output, center in zip(outputs, center_images)
    )
    linear_state = tuple(sum(
        coefficient * radius
        for coefficient, radius in zip(row, state_radii)
    ) for row in state_jacobian)
    linear_source = tuple(sum(
        coefficient * radius
        for coefficient, radius in zip(row, source_radii)
    ) for row in source_jacobian)
    interval_remainders = tuple(max(
        Fraction(0), radius - state_part - source_part,
    ) for radius, state_part, source_part in zip(
        image_radii, linear_state, linear_source))

    image = {
        "schema_version": SCHEMA_VERSION,
        "theta": [value.value.to_object() for value in theta_after],
        "first_moment": [value.value.to_object() for value in first_after],
        "second_moment": [value.value.to_object() for value in second_after],
        "optimizer_step": step_after,
        "scheduler_phase": next_scheduler_phase,
        "learning_rate": next_learning_rate.to_object(),
        "cover_state": next_cover_state,
        "intervention_state": next_intervention_state,
    }
    return {
        "schema_version": "pldr-interval-adamw-successor-v1",
        "source_dimension": width,
        "source_optimizer_step": source["optimizer_step"],
        "target_optimizer_step": step_after,
        "clip_branches": branches,
        "image": image,
        "state_coordinate_order": [
            *(f"theta[{index}]" for index in range(width)),
            *(f"first_moment[{index}]" for index in range(width)),
            *(f"second_moment[{index}]" for index in range(width)),
        ],
        "source_coordinate_order": [
            *(f"gradient[{index}]" for index in range(width)),
            *(f"intervention[{index}]" for index in range(width)),
            "learning_rate",
        ],
        "state_jacobian_absolute_upper": [
            [rational_object(value) for value in row]
            for row in state_jacobian
        ],
        "source_jacobian_absolute_upper": [
            [rational_object(value) for value in row]
            for row in source_jacobian
        ],
        "state_radii": [rational_object(value) for value in state_radii],
        "source_radii": [rational_object(value) for value in source_radii],
        "image_centers": [rational_object(value) for value in center_images],
        "image_radii": [rational_object(value) for value in image_radii],
        "interval_remainder": [
            rational_object(value) for value in interval_remainders],
        "center_source": center_source,
        "derivation": (
            "exact_rational_interval_differentiation_of_clipped_adamw_"
            "with_branch_resolved_clipping"
        ),
    }


def check_program_state_inclusion(successor, target_cell):
    """Verify that an interval successor image lies in a declared target cell."""

    if successor.get("schema_version") != "pldr-interval-adamw-successor-v1":
        raise ValueError("unknown interval successor record")
    image = _cell(successor["image"])
    target = _cell(target_cell)
    if image["optimizer_step"] != target["optimizer_step"]:
        raise ValueError("program-state target uses the wrong optimizer clock")
    if image["scheduler_phase"] != target["scheduler_phase"]:
        raise ValueError("program-state target uses the wrong scheduler phase")
    for name in ("theta", "first_moment", "second_moment"):
        if any(
            not target_interval.contains_interval(image_interval)
            for image_interval, target_interval in zip(image[name], target[name])
        ):
            raise ValueError(f"program-state image escapes target {name} cell")
    if not target["learning_rate"].contains_interval(image["learning_rate"]):
        raise ValueError("program-state image escapes target learning-rate cell")
    if image["cover_state"] != target["cover_state"]:
        raise ValueError("program-state cover successor is not the declared target")
    if image["intervention_state"] != target["intervention_state"]:
        raise ValueError(
            "program-state intervention successor is not the declared target")
    return True


def derived_positive_map(successor, target_cell):
    """Build ``x' <= P x + d`` from the successor derivative record.

    The target center is read from the actual target cell.  The forcing vector
    consists of the evaluated center displacement, every exogenous-input
    derivative contribution, and the interval evaluation remainder.
    """

    check_program_state_inclusion(successor, target_cell)
    target = _cell(target_cell)
    target_intervals = (
        target["theta"] + target["first_moment"] + target["second_moment"])
    target_centers = tuple(_center(value) for value in target_intervals)
    image_centers = tuple(as_fraction(value) for value in successor["image_centers"])
    state_matrix = tuple(tuple(as_fraction(value) for value in row)
                         for row in successor["state_jacobian_absolute_upper"])
    source_matrix = tuple(tuple(as_fraction(value) for value in row)
                          for row in successor["source_jacobian_absolute_upper"])
    source_radii = tuple(as_fraction(value) for value in successor["source_radii"])
    remainder = tuple(as_fraction(value) for value in successor["interval_remainder"])
    forcing = []
    for center, target_center, source_row, residual in zip(
        image_centers, target_centers, source_matrix, remainder,
    ):
        forcing.append(
            abs(center - target_center)
            + sum((coefficient * radius for coefficient, radius in zip(
                source_row, source_radii)), Fraction(0))
            + residual
        )
    return {
        "schema_version": "pldr-derived-positive-radius-map-v1",
        "matrix": [[rational_object(value) for value in row]
                   for row in state_matrix],
        "forcing": [rational_object(value) for value in forcing],
        "state_coordinate_order": successor["state_coordinate_order"],
        "source_coordinate_order": successor["source_coordinate_order"],
        "provenance": (
            "actual_program_center_image_interval_derivatives_and_input_cell"
        ),
    }
