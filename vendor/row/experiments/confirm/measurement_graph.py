"""Build replayable primary certificate graphs from owned measurement artifacts."""

from __future__ import annotations

from pathlib import Path

from certificate_graph import (
    CertificateGraph,
    CertificateNode,
    digest_object,
)
import estimator_manifest as estimator_registry
import campaign_record


DERIVATION_KIND = {
    "E0": "rational_interval_evaluation",
    "E1": "exact_recurrence",
    "E2": "rational_interval_evaluation",
    "E3": "induced_norm_inequality",
    "E4": "induced_norm_inequality",
    "E5": "induced_norm_inequality",
    "E6": "induced_norm_inequality",
    "E7": "empirical_estimator",
    "E8": "empirical_estimator",
}


MEASUREMENT_DERIVATION_KIND = {
    ("E3", "structured_family_q"): "exact_ldl_witness",
    ("E3", "structured_family_slack"): "exact_ldl_witness",
    ("E3", "maximum_corner_h_gain"): "exact_ldl_witness",
    ("E3", "multiaffine_reconstruction_residual"): "exact_recurrence",
    ("E3", "primitive_box_enclosure_ratio"):
        "rational_interval_evaluation",
    ("E3", "path_window_gain"): "positive_comparison",
    ("E4", "positive_comparison_witness_ratio"): "positive_comparison",
    ("E5", "path_window_gain"): "exact_ldl_witness",
    ("E5", "block_lifted_budget_slack"): "positive_comparison",
    ("E5", "prefix_budget_slack_min"): "positive_comparison",
    ("E6", "predicted_entry_step"): "finite_horizon",
    ("E6", "observed_entry_step"): "finite_horizon",
    ("E6", "entry_error"): "finite_horizon",
    ("E6", "sustained_entry_indicator"): "finite_horizon",
    ("E6", "path_window_gain"): "exact_ldl_witness",
    ("E6", "block_product_convolution_upper"): "positive_comparison",
    ("E6", "schedule_product_convolution_upper"): "positive_comparison",
    ("E6", "schedule_predicted_entry_step"): "finite_horizon",
}


def _derivation_kind(protocol_id, measurement_name):
    return MEASUREMENT_DERIVATION_KIND.get(
        (protocol_id, measurement_name), DERIVATION_KIND[protocol_id])


def build_measurement_graph(protocol_id, measurement, manifest, root=None):
    """Bind every primary scalar to its registered live owner.

    The complete live artifact is content-addressed by the root node. Each
    scalar output names the exact owner and estimator identifier that the
    assembly boundary independently replays.
    """
    if protocol_id not in DERIVATION_KIND:
        raise ValueError("protocol_id must be E0 through E8")
    if not isinstance(measurement, dict):
        raise ValueError("measurement artifact must be an object")
    rows = measurement.get("measurements")
    if not isinstance(rows, list) or not rows:
        raise ValueError("measurement artifact has no measurement array")
    by_name = {row.get("name"): row for row in rows}
    if len(by_name) != len(rows) or None in by_name:
        raise ValueError("measurement names are missing or duplicated")
    required = campaign_record.MEASUREMENT_NAMES[protocol_id]
    if set(by_name) != set(required):
        raise ValueError("live measurement set disagrees with the manifest")

    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    constructor_path = root / "experiments/confirm/measurement_graph.py"
    constructor_digest = estimator_registry.sha256_file(constructor_path)
    graph = CertificateGraph()
    graph.add(CertificateNode(
        name="live_measurement_artifact",
        derivation_kind="root_artifact",
        value={
            "protocol_id": protocol_id,
            "artifact_sha256": digest_object(measurement),
            "schema_version": measurement.get("schema_version", "unspecified"),
        },
        unit="content_addressed_live_artifact",
        domain="registered_live_unit",
        constructor=(
            "experiments/confirm/measurement_graph.py:"
            "build_measurement_graph"
        ),
        constructor_sha256=constructor_digest,
        parents=(),
        metadata={"protocol_id": protocol_id},
    ))
    outputs = []
    bindings = {}
    for name in required:
        row = by_name[name]
        if row.get("status") != "OBSERVED" or row.get("reason_code") is not None:
            raise ValueError("graph construction requires an observed scalar")
        identifier = f"{protocol_id}.{name}"
        owner = manifest["estimators"][identifier]
        node_name = f"primary_{name}"
        graph.add(CertificateNode(
            name=node_name,
            derivation_kind=_derivation_kind(protocol_id, name),
            value=row["value"],
            unit=row["unit"],
            domain="registered_live_unit",
            constructor=owner["executable"] + ":" + owner["function"],
            constructor_sha256=owner["executable_sha256"],
            parents=("live_measurement_artifact",),
            metadata={"estimator_id": identifier},
        ))
        outputs.append(node_name)
        bindings[name] = node_name
    return graph.to_object(outputs=outputs), bindings
