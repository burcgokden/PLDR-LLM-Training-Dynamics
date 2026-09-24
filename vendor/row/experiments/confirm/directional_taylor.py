"""Exact mesh-plus-modulus certificates for directional Taylor segments."""

from __future__ import annotations

from validated_interval import as_fraction, rational_object


SCHEMA_VERSION = "pldr-directional-taylor-mesh-v1"


def _nonnegative(value, label):
    value = as_fraction(value)
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def certify_directional_segment(
        *, point_second_directional_uppers, third_directional_modulus,
        subdivisions, observed_remainder=None):
    """Certify a complete unit segment from mesh values and a modulus.

    Point values bound the directional second derivative at the equally
    spaced mesh points. The third-directional modulus bounds motion between
    points. A sampled maximum alone is therefore never accepted.
    """

    if (
        not isinstance(subdivisions, int)
        or isinstance(subdivisions, bool)
        or subdivisions < 1
    ):
        raise ValueError("subdivisions must be a positive integer")
    if (
        not isinstance(point_second_directional_uppers, (list, tuple))
        or len(point_second_directional_uppers) != subdivisions + 1
    ):
        raise ValueError(
            "the directional mesh must contain subdivisions + 1 point bounds")
    points = tuple(
        _nonnegative(value, "point second-directional upper")
        for value in point_second_directional_uppers
    )
    third = _nonnegative(
        third_directional_modulus, "third-directional modulus")
    mesh_maximum = max(points)
    mesh_charge = third / (2 * subdivisions)
    segment_second_upper = mesh_maximum + mesh_charge
    remainder_upper = segment_second_upper / 2
    if observed_remainder is None:
        observed = None
        enclosed = None
    else:
        observed = _nonnegative(observed_remainder, "observed remainder")
        enclosed = observed <= remainder_upper
    return {
        "schema_version": SCHEMA_VERSION,
        "subdivisions": subdivisions,
        "point_second_directional_uppers": [
            rational_object(value) for value in points],
        "third_directional_modulus": rational_object(third),
        "mesh_maximum": rational_object(mesh_maximum),
        "mesh_modulus_charge": rational_object(mesh_charge),
        "segment_second_directional_upper": rational_object(
            segment_second_upper),
        "taylor_remainder_upper": rational_object(remainder_upper),
        "observed_remainder": (
            None if observed is None else rational_object(observed)),
        "observed_remainder_enclosed": enclosed,
        "derivation": (
            "nearest_mesh_point_distance_1_over_2N_plus_"
            "third_directional_modulus_and_integral_taylor_half"),
    }


def certify_operator_segment(
        *, point_second_operator_uppers, third_operator_modulus,
        displacement_norm_upper, subdivisions, observed_remainder=None):
    """Convert operator bounds to directional bounds and certify the segment."""

    displacement = _nonnegative(
        displacement_norm_upper, "displacement norm upper")
    second = [
        _nonnegative(value, "point second-operator upper")
        * displacement ** 2
        for value in point_second_operator_uppers
    ]
    third = (
        _nonnegative(third_operator_modulus, "third-operator modulus")
        * displacement ** 3
    )
    result = certify_directional_segment(
        point_second_directional_uppers=second,
        third_directional_modulus=third,
        subdivisions=subdivisions,
        observed_remainder=observed_remainder,
    )
    result["displacement_norm_upper"] = rational_object(displacement)
    result["point_second_operator_uppers"] = [
        rational_object(_nonnegative(value, "point second-operator upper"))
        for value in point_second_operator_uppers
    ]
    result["third_operator_modulus"] = rational_object(
        _nonnegative(third_operator_modulus, "third-operator modulus"))
    return result
