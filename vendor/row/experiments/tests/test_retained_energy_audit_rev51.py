import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from analysis.analyze_retained_energy_audit import (  # noqa: E402
    REPORT_PATH,
    independent_work_charge,
)
from confirm.direct_work import center_rows, work_charge  # noqa: E402


def test_independent_work_charge_matches_canonical_random_maps():
    generator = np.random.default_rng(51051)
    source = generator.normal(size=(7, 5, 11))
    endpoint = source + 0.03 * generator.normal(size=source.shape)
    independent = independent_work_charge(source, endpoint)
    canonical = work_charge(center_rows(source), center_rows(endpoint))
    for name in (
        "increment",
        "source_energy",
        "endpoint_energy",
        "energy_change",
        "signed_work",
        "finite_step_charge",
        "identity_residual",
        "reopening_charge",
    ):
        np.testing.assert_allclose(independent[name], canonical[name], atol=2e-12)


def test_retained_audit_reports_exact_scope_and_missing_arrays():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    dense = report["dense_scalar_energy_archive"]
    endpoint = report["successive_physical_map_archive"]
    assert report["schema_version"] == "pldr-retained-energy-audit-v1"
    assert report["release_id"] == "rev55"
    assert report["analysis_role"] == "cpu-only-retained-record-audit"
    assert dense["captured_update_edges"] == 1056
    assert dense["map_increments"] == 304128
    assert dense["work_charge_status"] == "not-evaluable-from-retained-records"
    assert dense["gate_shape_status"] == "not-evaluable-from-retained-records"
    assert endpoint["successive_horizon_pairs"] == 9
    assert endpoint["map_increments"] == 864
    assert endpoint["records_without_successive_physical_maps"] == 46
    assert endpoint["gate_shape_status"] == "not-evaluable-from-retained-records"
    assert endpoint["all_increments_within_fixed_float64_tolerance"] is True
    assert report["conclusions"]["missing_arrays_are_not_reconstructed"] is True
    assert report["conclusions"]["asymptotic_collapse_is_not_inferred"] is True


def test_retained_audit_reproduces_cautious_concentration_counts():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    concentration = report["cancellation_concentration"]
    assert concentration["cancellation_dominant_total"] == 6
    assert concentration["largest_trajectory_count"] == 5
    assert concentration["layer_two_count"] == 4
    assert concentration["interpretation"] == "descriptive-finite-concentration-only"
