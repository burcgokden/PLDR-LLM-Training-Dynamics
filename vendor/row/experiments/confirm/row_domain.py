"""Convex row domains, deterministic tilings, and segment witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product

from validated_interval import (
    RationalInterval,
    as_fraction,
    nonnegative_sqrt_upper,
    rational_object,
)


@dataclass(frozen=True)
class IntervalHull:
    coordinates: tuple[RationalInterval, ...]

    @property
    def dimension(self):
        return len(self.coordinates)

    def contains(self, row):
        return len(row) == self.dimension and all(
            interval.contains(value)
            for interval, value in zip(self.coordinates, row)
        )

    def contains_segment(self, left, right):
        """A box contains the complete segment iff it contains both endpoints."""

        return self.contains(left) and self.contains(right)

    def point_on_segment(self, left, right, weight):
        weight = as_fraction(weight)
        if not 0 <= weight <= 1:
            raise ValueError("segment weight must lie in [0, 1]")
        if not self.contains_segment(left, right):
            raise ValueError("segment endpoints do not lie in the interval hull")
        point = tuple(
            (1 - weight) * as_fraction(a) + weight * as_fraction(b)
            for a, b in zip(left, right)
        )
        if not self.contains(point):
            raise ArithmeticError("convex interval-hull witness failed")
        return point

    def to_object(self):
        return {
            "kind": "convex_interval_hull",
            "dimension": self.dimension,
            "coordinates": [entry.to_object() for entry in self.coordinates],
        }


def interval_hull(rows, padding=0):
    rows = tuple(tuple(as_fraction(entry) for entry in row) for row in rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("registered rows must form a nonempty rectangular array")
    if isinstance(padding, (list, tuple)):
        padding = tuple(as_fraction(entry) for entry in padding)
    else:
        padding = tuple(as_fraction(padding) for _ in rows[0])
    if len(padding) != len(rows[0]) or any(entry < 0 for entry in padding):
        raise ValueError("row-domain padding must be nonnegative and dimensioned")
    coordinates = []
    for index, margin in enumerate(padding):
        values = [row[index] for row in rows]
        coordinates.append(RationalInterval(
            min(values) - margin, max(values) + margin))
    return IntervalHull(tuple(coordinates))


def tile_interval_hull(hull, subdivisions, *, maximum_boxes=1_000_000):
    """Tile a convex hull and return exact h-net proof data.

    Centers are box midpoints.  Every point in a tile is at Euclidean
    distance at most half of the tile diagonal from its center.
    """

    if not isinstance(hull, IntervalHull):
        raise TypeError("tiling requires an IntervalHull")
    if isinstance(subdivisions, int):
        subdivisions = (subdivisions,) * hull.dimension
    subdivisions = tuple(int(value) for value in subdivisions)
    if len(subdivisions) != hull.dimension or any(value < 1 for value in subdivisions):
        raise ValueError("subdivision counts must be positive in every dimension")
    box_count = 1
    for value in subdivisions:
        box_count *= value
    if box_count > maximum_boxes:
        return {
            "status": "INFEASIBLE",
            "reason_code": "ROW_COVER_BOX_LIMIT",
            "box_count": box_count,
            "maximum_boxes": int(maximum_boxes),
            "dimension": hull.dimension,
        }
    axis_intervals = []
    for interval, count in zip(hull.coordinates, subdivisions):
        width = interval.width / count
        axis_intervals.append(tuple(
            RationalInterval(
                interval.lower + index * width,
                interval.lower + (index + 1) * width,
            ) for index in range(count)
        ))
    boxes = []
    centers = []
    maximum_radius_squared = Fraction(0)
    for coordinate_box in product(*axis_intervals):
        center = tuple(entry.midpoint for entry in coordinate_box)
        radius_squared = sum(
            ((entry.width / 2) ** 2 for entry in coordinate_box),
            Fraction(0),
        )
        maximum_radius_squared = max(maximum_radius_squared, radius_squared)
        boxes.append([entry.to_object() for entry in coordinate_box])
        centers.append([rational_object(entry) for entry in center])
    h_upper = nonnegative_sqrt_upper(maximum_radius_squared)
    return {
        "status": "CERTIFIED",
        "domain": hull.to_object(),
        "subdivisions": list(subdivisions),
        "box_count": box_count,
        "boxes": boxes,
        "centers": centers,
        "h_upper": rational_object(h_upper),
        "proof": "midpoint_half_diagonal_h_net_of_axis_aligned_partition",
    }


def cover_segment_hull(rows, subdivisions_per_segment,
                       *, maximum_boxes=1_000_000):
    """Cover every registered chord without filling its ambient box.

    Every cell is the exact coordinate enclosure of a subsegment. Its center
    lies on that subsegment and its cover radius is half the subsegment's
    Euclidean length. The union therefore contains the segment between every
    pair of registered rows while retaining their primitive correlations.
    """

    rows = tuple(tuple(as_fraction(entry) for entry in row) for row in rows)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("registered rows must form a nonempty rectangular array")
    if (
        not isinstance(subdivisions_per_segment, int)
        or isinstance(subdivisions_per_segment, bool)
        or subdivisions_per_segment < 1
    ):
        raise ValueError("segment subdivisions must be a positive integer")
    pairs = list(combinations(range(len(rows)), 2))
    if not pairs:
        pairs = [(0, 0)]
    box_count = len(pairs) * subdivisions_per_segment
    if box_count > maximum_boxes:
        return {
            "status": "INFEASIBLE",
            "reason_code": "ROW_COVER_BOX_LIMIT",
            "box_count": box_count,
            "maximum_boxes": int(maximum_boxes),
            "dimension": len(rows[0]),
        }
    boxes = []
    centers = []
    cells = []
    maximum_radius_squared = Fraction(0)
    subdivision_count = Fraction(subdivisions_per_segment)
    for left_index, right_index in pairs:
        left = rows[left_index]
        right = rows[right_index]
        displacement = tuple(b - a for a, b in zip(left, right))
        length_squared = sum(
            (entry * entry for entry in displacement), Fraction(0))
        for index in range(subdivisions_per_segment):
            lower_s = Fraction(index, subdivisions_per_segment)
            upper_s = Fraction(index + 1, subdivisions_per_segment)
            midpoint_s = (lower_s + upper_s) / 2
            lower_point = tuple(
                a + lower_s * delta for a, delta in zip(left, displacement))
            upper_point = tuple(
                a + upper_s * delta for a, delta in zip(left, displacement))
            center = tuple(
                a + midpoint_s * delta for a, delta in zip(left, displacement))
            coordinate_box = tuple(
                RationalInterval(min(a, b), max(a, b))
                for a, b in zip(lower_point, upper_point)
            )
            radius_squared = (
                length_squared
                / (4 * subdivision_count * subdivision_count)
            )
            maximum_radius_squared = max(
                maximum_radius_squared, radius_squared)
            boxes.append([entry.to_object() for entry in coordinate_box])
            centers.append([rational_object(entry) for entry in center])
            cells.append({
                "left_row": left_index,
                "right_row": right_index,
                "segment_parameter": {
                    "lower": rational_object(lower_s),
                    "upper": rational_object(upper_s),
                },
                "radius_squared": rational_object(radius_squared),
            })
    h_upper = nonnegative_sqrt_upper(maximum_radius_squared)
    return {
        "status": "CERTIFIED",
        "domain": {
            "type": "registered_pairwise_segment_hull",
            "rows": [
                [rational_object(entry) for entry in row] for row in rows
            ],
        },
        "subdivisions_per_segment": subdivisions_per_segment,
        "box_count": box_count,
        "boxes": boxes,
        "centers": centers,
        "cells": cells,
        "h_upper": rational_object(h_upper),
        "proof": "pairwise_chord_subdivision_with_half_segment_radius",
    }


def validate_construction_validation_split(construction_ids, validation_ids):
    """Require a frozen, disjoint row-domain construction split."""

    construction = tuple(construction_ids)
    validation = tuple(validation_ids)
    if not construction or not validation:
        raise ValueError("row-domain partitions must both be nonempty")
    if len(set(construction)) != len(construction):
        raise ValueError("construction row identifiers are duplicated")
    if len(set(validation)) != len(validation):
        raise ValueError("validation row identifiers are duplicated")
    overlap = set(construction) & set(validation)
    if overlap:
        raise ValueError(
            "validation rows were used to construct the physical row domain")
    return True


def points_in_rational_boxes(points, boxes):
    """Return exact membership witnesses for a union of rational boxes."""

    points = tuple(tuple(as_fraction(value) for value in row) for row in points)
    boxes = tuple(tuple(RationalInterval.from_object(value) for value in box)
                  for box in boxes)
    if not points or not boxes:
        raise ValueError("box membership needs nonempty points and boxes")
    dimension = len(points[0])
    if (
        not dimension
        or any(len(point) != dimension for point in points)
        or any(len(box) != dimension for box in boxes)
    ):
        raise ValueError("box membership dimensions disagree")
    result = []
    for point in points:
        containing = [
            index for index, box in enumerate(boxes)
            if all(interval.contains(value)
                   for interval, value in zip(box, point))
        ]
        result.append({
            "member": bool(containing),
            "containing_box_indices": containing,
        })
    return result


def require_points_in_rational_boxes(points, boxes):
    """Require exact membership and return the corresponding witnesses."""

    witnesses = points_in_rational_boxes(points, boxes)
    missing = [index for index, row in enumerate(witnesses)
               if not row["member"]]
    if missing:
        raise ValueError(
            "validation rows lie outside the frozen physical row domain: "
            + ", ".join(str(index) for index in missing))
    return witnesses

def disconnected_ball_counterexample():
    """Return endpoints in separate balls whose joining segment is omitted."""

    left, right = (Fraction(-2),), (Fraction(2),)
    radius = Fraction(1, 4)
    midpoint = (Fraction(0),)
    endpoint_membership = (
        abs(left[0] - left[0]) <= radius
        and abs(right[0] - right[0]) <= radius
    )
    midpoint_membership = (
        abs(midpoint[0] - left[0]) <= radius
        or abs(midpoint[0] - right[0]) <= radius
    )
    return {
        "endpoints_in_union": endpoint_membership,
        "joining_midpoint_in_union": midpoint_membership,
        "global_segment_valid": endpoint_membership and midpoint_membership,
    }
