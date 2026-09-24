"""One total, schema-versioned E0 construct-validity decision.

No upstream estimator may authorize a downstream experiment.  It may only
populate an :class:`E0Record`.  ``decide_e0`` is the single authorization
function and consumes every registered prerequisite and all three
inferential families.
"""

from dataclasses import asdict, dataclass
from enum import Enum
import math
import re


SCHEMA_VERSION = "pldr-e0-v2"
PRIMARY_CURVATURE = "gauss_newton"
PRIMARY_METRIC = "preconditioned"
PRIMARY_UNIT = "physical"
INFERENTIAL_FAMILIES = (
    "loading_law",
    "averaged_stability_regularity",
    "upper_crossing_signature",
)
REQUIRED_PREREQUISITES = (
    "artifact_hashes_match",
    "all_values_present",
    "all_values_finite",
    "window_band_containment",
    "window_box_containment",
    "averaging_amplitude",
    "averaging_centering",
    "averaging_preconditioner",
    "schedule_variation",
    "clipping",
    "chart_fidelity",
    "direction_separation",
    "factory_residual",
    "decay_ledger_residual",
    "source_edge_intervals",
    "gate_slope",
    "fiber_constants",
    "checkpoint_battery",
    "pilot_complete",
)


class E0Outcome(str, Enum):
    PASS = "PASS"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    INDETERMINATE = "INDETERMINATE"
    INCOMPLETE = "INCOMPLETE"
    MODEL_REJECTED = "MODEL_REJECTED"


@dataclass(frozen=True)
class E0Record:
    schema_version: str
    protocol_hash: str
    schema_hash: str
    source_hash: str
    calibration_hash: str
    curvature_kind: str
    metric_kind: str
    unit_kind: str
    residual_certificate_pass: bool | None
    n_upper_crossings: int | None
    pilot_region_status: str | None
    model_consistent: bool | None
    uncertainty_resolved: bool | None
    prerequisites: dict
    family_p_values: dict
    family_directional_pass: dict

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, value):
        return cls(**{name: value.get(name)
                      for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class E0Decision:
    outcome: E0Outcome
    reasons: tuple[str, ...]
    requirements: dict

    def to_dict(self):
        return {
            "outcome": self.outcome.value,
            "reasons": list(self.reasons),
            "requirements": self.requirements,
        }


def holm_all_pass(p_values, alpha=0.05):
    """Return the exact three-family Holm conjunction and thresholds."""
    if set(p_values) != set(INFERENTIAL_FAMILIES):
        raise ValueError("Holm input must contain exactly three E0 families")
    ordered = sorted((float(value), name)
                     for name, value in p_values.items())
    if not all(math.isfinite(value) and 0 <= value <= 1
               for value, _ in ordered):
        raise ValueError("family p-values must be finite probabilities")
    rows = []
    passed = True
    count = len(ordered)
    for index, (value, name) in enumerate(ordered):
        threshold = alpha / (count - index)
        local = value <= threshold
        passed = passed and local
        rows.append({
            "family": name,
            "p_value": value,
            "threshold": threshold,
            "pass": local,
        })
    return passed, rows


def _hash_present(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def decide_e0(record):
    """Compute the exhaustive E0 outcome; only PASS authorizes launch."""
    if isinstance(record, dict):
        record = E0Record.from_mapping(record)
    reasons = []
    requirements = {}

    hashes = {
        name: getattr(record, name)
        for name in ("protocol_hash", "schema_hash", "source_hash",
                     "calibration_hash")
    }
    complete = record.schema_version == SCHEMA_VERSION
    if not complete:
        reasons.append("schema version is missing or unsupported")
    for name, value in hashes.items():
        if not _hash_present(value):
            complete = False
            reasons.append(f"{name} is missing or malformed")

    prereq = record.prerequisites if isinstance(record.prerequisites, dict) else {}
    missing_prereq = [name for name in REQUIRED_PREREQUISITES
                      if name not in prereq or prereq[name] is None]
    extra_prereq = sorted(set(prereq) - set(REQUIRED_PREREQUISITES))
    if missing_prereq:
        complete = False
        reasons.append("missing prerequisites: " + ", ".join(missing_prereq))
    if extra_prereq:
        complete = False
        reasons.append("unknown prerequisites: " + ", ".join(extra_prereq))

    p_values = (record.family_p_values
                if isinstance(record.family_p_values, dict) else {})
    directions = (record.family_directional_pass
                  if isinstance(record.family_directional_pass, dict) else {})
    if set(p_values) != set(INFERENTIAL_FAMILIES):
        complete = False
        reasons.append("the three inferential-family p-values are incomplete")
    if set(directions) != set(INFERENTIAL_FAMILIES) or any(
            value is None for value in directions.values()):
        complete = False
        reasons.append("the three inferential-family directions are incomplete")
    if record.n_upper_crossings is None or record.n_upper_crossings < 0:
        complete = False
        reasons.append("upper-crossing count is missing")
    if record.pilot_region_status not in {"nonempty", "unresolved", "empty"}:
        complete = False
        reasons.append("pilot-region status is missing or unknown")
    if record.model_consistent is None or record.uncertainty_resolved is None:
        complete = False
        reasons.append("model-consistency or uncertainty status is missing")
    if record.residual_certificate_pass is None:
        complete = False
        reasons.append("Gauss-Newton residual certificate is missing")
    if any(value is None for value in (
            record.curvature_kind, record.metric_kind, record.unit_kind)):
        complete = False
        reasons.append("primary curvature metadata is missing")

    if not complete:
        return E0Decision(E0Outcome.INCOMPLETE, tuple(reasons), requirements)
    if prereq["all_values_present"] is False:
        reasons.append("one or more required E0 artifacts are absent")
        return E0Decision(E0Outcome.INCOMPLETE, tuple(reasons), requirements)

    primary_tuple = (
        record.curvature_kind == PRIMARY_CURVATURE
        and record.metric_kind == PRIMARY_METRIC
        and record.unit_kind == PRIMARY_UNIT
    )
    requirements["primary_metric_tuple"] = primary_tuple
    if not primary_tuple:
        reasons.append("primary state is not physical preconditioned Gauss-Newton curvature")
        return E0Decision(E0Outcome.MODEL_REJECTED, tuple(reasons), requirements)
    if record.residual_certificate_pass is False:
        reasons.append("Gauss-Newton residual certificate failed")
        return E0Decision(E0Outcome.MODEL_REJECTED, tuple(reasons), requirements)
    if record.pilot_region_status == "empty" or not record.model_consistent:
        reasons.append("finite-grid model inversion is empty or inconsistent")
        return E0Decision(E0Outcome.MODEL_REJECTED, tuple(reasons), requirements)

    requirements.update({name: bool(prereq[name])
                         for name in REQUIRED_PREREQUISITES})
    requirements["minimum_40_upper_crossings"] = record.n_upper_crossings >= 40
    try:
        holm_pass, holm_rows = holm_all_pass(p_values)
    except (TypeError, ValueError) as exc:
        return E0Decision(E0Outcome.INCOMPLETE, (str(exc),), requirements)
    requirements["holm_three_families"] = holm_pass
    requirements["holm_rows"] = holm_rows
    for family in INFERENTIAL_FAMILIES:
        requirements[f"direction_{family}"] = bool(directions[family])

    failed = [name for name, value in requirements.items()
              if name != "holm_rows" and value is False]
    if failed:
        reasons.append("failed requirements: " + ", ".join(failed))
        return E0Decision(E0Outcome.NOT_CONFIRMED, tuple(reasons), requirements)
    if not record.uncertainty_resolved or record.pilot_region_status == "unresolved":
        reasons.append("complete record lies in a declared uncertainty region")
        return E0Decision(E0Outcome.INDETERMINATE, tuple(reasons), requirements)
    return E0Decision(E0Outcome.PASS, (), requirements)
