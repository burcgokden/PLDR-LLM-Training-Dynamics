"""Positive-affine renormalization tools for row-map energy."""

from .core import (
    Edge,
    beta,
    block_edges,
    block_levels,
    canonical_edge,
    canonical_edges,
    classify_gain,
    clean_correlation_scale,
    clean_flow,
    compose,
    homogeneous_block,
    normalize,
    replay,
)
from .confirmation import analyze_confirmation
from .confirmation_v3 import analyze_confirmation_v3
from .interval import (
    IntervalAffineEdge,
    NonnegativeInterval,
    block_interval_edges,
    certified_face_edge,
    certified_positive_gain_interval,
)
from .observer import (
    dyadic_sqrt_enclosure,
    enclosure_class,
    layernorm_row_energy_enclosure,
)
from .recorder import RowEnergyRecorder, row_centered_energy

__all__ = [
    "Edge",
    "analyze_confirmation",
    "analyze_confirmation_v3",
    "beta",
    "block_edges",
    "block_levels",
    "canonical_edge",
    "canonical_edges",
    "classify_gain",
    "clean_correlation_scale",
    "clean_flow",
    "compose",
    "homogeneous_block",
    "IntervalAffineEdge",
    "NonnegativeInterval",
    "normalize",
    "replay",
    "block_interval_edges",
    "certified_face_edge",
    "certified_positive_gain_interval",
    "dyadic_sqrt_enclosure",
    "enclosure_class",
    "layernorm_row_energy_enclosure",
    "RowEnergyRecorder",
    "row_centered_energy",
]
