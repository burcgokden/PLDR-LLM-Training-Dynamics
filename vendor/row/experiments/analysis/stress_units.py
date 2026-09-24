"""Typed curvature units used by the PLDR row-map collapse theory.

The primary state is physical preconditioned Gauss--Newton curvature.
Normalized sharpness is derived from it and raw/full-Hessian quantities are
separate observables.  Keeping the metadata with the values prevents a key
name or an implicit rescaling from changing the scientific estimand.
"""

from dataclasses import dataclass
from enum import Enum
import math


class CurvatureKind(str, Enum):
    GAUSS_NEWTON = "gauss_newton"
    FULL_HESSIAN = "full_hessian"


class MetricKind(str, Enum):
    PRECONDITIONED = "preconditioned"
    RAW = "raw"


class UnitKind(str, Enum):
    PHYSICAL = "physical"
    NORMALIZED = "normalized"


@dataclass(frozen=True)
class StressMetadata:
    curvature: CurvatureKind
    metric: MetricKind
    unit: UnitKind

    def as_dict(self):
        return {
            "curvature": self.curvature.value,
            "metric": self.metric.value,
            "unit": self.unit.value,
        }


PRIMARY_STRESS = StressMetadata(
    CurvatureKind.GAUSS_NEWTON,
    MetricKind.PRECONDITIONED,
    UnitKind.PHYSICAL,
)


def _finite(*values):
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("stress values must be finite")


def require_primary(metadata):
    """Reject any implicit curvature, metric, or unit substitution."""
    if metadata != PRIMARY_STRESS:
        raise ValueError(
            "primary stress must be physical preconditioned Gauss-Newton "
            f"curvature, got {metadata!r}")


def normalized_stress(x, one_minus_beta1, eta):
    """Return sigma = (1-beta1) eta x from physical curvature x."""
    _finite(x, one_minus_beta1, eta)
    if one_minus_beta1 <= 0 or eta <= 0:
        raise ValueError("one_minus_beta1 and eta must be positive")
    return one_minus_beta1 * eta * x


def normalized_stress_increment(x_t, x_next, one_minus_beta1,
                                eta_t, eta_next):
    """Exact moving-schedule increment of normalized sharpness."""
    _finite(x_t, x_next, one_minus_beta1, eta_t, eta_next)
    direct = (normalized_stress(x_next, one_minus_beta1, eta_next)
              - normalized_stress(x_t, one_minus_beta1, eta_t))
    curvature = one_minus_beta1 * eta_next * (x_next - x_t)
    schedule = one_minus_beta1 * x_t * (eta_next - eta_t)
    return {
        "total": direct,
        "curvature": curvature,
        "schedule": schedule,
    }


def physical_upper_threshold(edge_upper, one_minus_beta1, eta):
    """Convert an upper stability edge to the physical guard threshold."""
    _finite(edge_upper, one_minus_beta1, eta)
    if one_minus_beta1 <= 0 or eta <= 0:
        raise ValueError("one_minus_beta1 and eta must be positive")
    return edge_upper / (one_minus_beta1 * eta)


def physical_upper_guard(x, edge_upper, one_minus_beta1, eta):
    """Strict upper crossing in physical units."""
    return x > physical_upper_threshold(
        edge_upper, one_minus_beta1, eta)


def upper_headroom_physical(x, edge_upper, one_minus_beta1, eta):
    """Oriented physical dose needed to reach the upper edge."""
    return (physical_upper_threshold(edge_upper, one_minus_beta1, eta)
            - x)


def upper_headroom_normalized(x, edge_upper, one_minus_beta1, eta):
    """The same oriented headroom in normalized stability units."""
    return edge_upper - normalized_stress(x, one_minus_beta1, eta)


def gn_interval_from_full_hessian(h, residual_bound):
    """Transfer h = x + r and |r| <= residual_bound to an x interval."""
    _finite(h, residual_bound)
    if residual_bound < 0:
        raise ValueError("residual_bound must be nonnegative")
    return h - residual_bound, h + residual_bound
