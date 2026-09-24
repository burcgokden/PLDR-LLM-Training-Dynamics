#!/usr/bin/env python3
"""Assemble one hash-bound, exhaustive E0 decision record.

Estimator programs only write measurements.  This module is the sole E0
decision point.  It binds the protocol, schema, calibration, source bundle,
and input artifacts, evaluates every registered prerequisite, applies Holm
to exactly three inferential families through :mod:`e0_decision`, and writes
both the immutable record and its total outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from e0_decision import (
    E0Record,
    INFERENTIAL_FAMILIES,
    REQUIRED_PREREQUISITES,
    SCHEMA_VERSION,
    decide_e0,
)
import factory_residual as FR
import gn_residual_certificate as GRC


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
OUT = HERE / "out" / "e0"
RUNS = EXP / "runs"
CALIBRATION_PATH = EXP / "calibration" / "calibration.json"
PROTOCOL_PATH = EXP / "protocols" / "e0_construct_validity.md"

CELLS = ("c-e0-boundary-merged", "c-e0-subcrit-merged")
SOURCE_FILES = (
    EXP / "analysis" / "oriented_headroom.py",
    EXP / "analysis" / "stress_units.py",
    EXP / "confirm" / "bridge.py",
    EXP / "confirm" / "ckpt_battery.py",
    EXP / "confirm" / "e0_decision.py",
    EXP / "confirm" / "e0_measure.py",
    EXP / "confirm" / "factory_residual.py",
    EXP / "confirm" / "gn_residual_certificate.py",
    EXP / "confirm" / "pilot_fit.py",
    EXP / "confirm" / "run_e0_analysis.py",
    EXP / "confirm" / "source_edge_certificate.py",
    EXP / "instrument.py",
    EXP / "optimizer_ledger.py",
    EXP / "train_run.py",
)


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(value, path):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_bundle_hash(paths=SOURCE_FILES):
    digest = hashlib.sha256()
    for path in sorted(map(Path, paths), key=lambda item: str(item)):
        rel = path.relative_to(ROOT).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big"))
        digest.update(rel)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _finite(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return False


def _required_estimates_present(summary):
    if not isinstance(summary, dict):
        return False
    p_values = summary.get("family_p_values")
    directions = summary.get("family_directional_pass")
    return (
        set(p_values or {}) == set(INFERENTIAL_FAMILIES)
        and set(directions or {}) == set(INFERENTIAL_FAMILIES)
        and all((p_values or {}).get(name) is not None
                for name in INFERENTIAL_FAMILIES)
        and summary.get("n_upper_crossings") is not None
        and summary.get("stress_metadata") is not None
    )


def _fidelity_pass(reports, limits):
    rows = []
    for report in reports.values():
        if not isinstance(report, dict) or not report.get("ckpts"):
            return False
        rows.extend(report["ckpts"].values())
    for row in rows:
        if "error" in row or not isinstance(row.get("eps_T"), dict):
            return False
        eps = row["eps_T"]
        if not all(math.isfinite(float(eps.get(key, math.inf)))
                   and float(eps[key]) <= limits["fidelity_eps_T_max"]
                   for key in ("exploitation", "exploitation_partner")):
            return False
        if not (math.isfinite(float(row.get("kappa_T", math.inf)))
                and float(row["kappa_T"])
                <= limits["fidelity_kappa_T_max"]):
            return False
        if not (math.isfinite(float(row.get("eps_off", math.inf)))
                and float(row["eps_off"])
                <= limits["fidelity_eps_off_max"]):
            return False
    return bool(rows)


def _separation_pass(reports):
    rows = []
    for report in reports.values():
        if not isinstance(report, dict) or not report.get("ckpts"):
            return False
        rows.extend(report["ckpts"].values())
    return bool(rows) and all(row.get("sep_gt_2rho") is True for row in rows)


def _fiber_constants_pass(batteries):
    seen = False
    for battery in batteries.values():
        if not isinstance(battery, dict) or not battery.get("ckpts"):
            return False
        for checkpoint in battery["ckpts"]:
            for directions in checkpoint.get("fibers", {}).values():
                for row in directions:
                    seen = True
                    kappa = row.get("kappa_phi")
                    cup = row.get("C_phi_update")
                    if (kappa is None or not math.isfinite(float(kappa))
                            or float(kappa) <= 0):
                        return False
                    if cup is not None and not math.isfinite(float(cup)):
                        return False
    return seen


def _hashes_match_declared(artifacts, expected):
    """Reject a declared stale hash; absence is handled by completeness."""
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        metadata = artifact.get("artifact_metadata", {})
        for key, value in expected.items():
            declared = metadata.get(key)
            if declared is not None and declared != value:
                return False
    return True


def _window_prerequisites(summary, limits):
    ws = ((summary or {}).get("window_summary") or {})
    kept = int(ws.get("n_kept") or 0)
    fraction_min = float(limits["regular_window_fraction_min"])

    def fraction(key):
        return kept > 0 and int(ws.get(key) or 0) >= fraction_min * kept

    return {
        "window_band_containment": kept > 0 and ws.get("in_band") == kept,
        "window_box_containment": kept > 0 and ws.get("in_box") == kept,
        "averaging_amplitude": fraction("sc_A"),
        "averaging_centering": kept > 0 and ws.get("sc_e") == kept,
        "averaging_preconditioner": (
            (summary or {}).get("averaging_preconditioner")
            == {"operator": "inverse", "window": "trailing"}),
        "schedule_variation": fraction("sc_drift"),
        "clipping": kept > 0 and ws.get("cwin") == kept,
    }


def assemble():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = load_json(CALIBRATION_PATH)
    limits = (calibration or {}).get("e0_acceptance", {})

    paths = {
        "e0_results": OUT / "e0_results.json",
        "pilot_fit": OUT / "pilot_fit.json",
    }
    paths.update({f"checkpoint_{cell}": OUT / f"ckpt_battery_{cell}.json"
                  for cell in CELLS})
    paths.update({f"fidelity_{cell}": RUNS / cell / "fidelity.json"
                  for cell in CELLS})
    paths.update({f"separation_{cell}": RUNS / cell / "separation.json"
                  for cell in CELLS})
    artifacts = {name: load_json(path) for name, path in paths.items()}
    gaps = [name for name, value in artifacts.items() if value is None]
    artifact_hashes = {
        name: sha256_file(paths[name])
        for name, value in artifacts.items() if value is not None
    }

    protocol_hash = sha256_file(PROTOCOL_PATH)
    schema_hash = sha256_file(HERE / "e0_decision.py")
    calibration_hash = sha256_file(CALIBRATION_PATH)
    source_hash = source_bundle_hash()
    expected_hashes = {
        "protocol_hash": protocol_hash,
        "schema_hash": schema_hash,
        "calibration_hash": calibration_hash,
        "source_hash": source_hash,
    }

    results = artifacts["e0_results"] or {}
    summary = results.get("estimator_summary") or {}
    boundary = results.get("boundary") or {}
    pilot = artifacts["pilot_fit"] or {}
    batteries = {cell: artifacts[f"checkpoint_{cell}"] for cell in CELLS}
    fidelity = {cell: artifacts[f"fidelity_{cell}"] for cell in CELLS}
    separation = {cell: artifacts[f"separation_{cell}"] for cell in CELLS}

    residual = boundary.get("gn_full_residual_relative_max")
    residual_certificate = GRC.certificate(
        boundary, batteries,
        limits["gn_full_residual_relative_max"])
    residual_pass = residual_certificate["pass"]
    factory_certificate = FR.certificate(
        boundary, batteries.get(CELLS[0]),
        limits["factory_residual_relative_max"])
    ledger_residual = boundary.get("decay_ledger_residual_max")
    ledger_pass = bool(
        ledger_residual is not None
        and math.isfinite(float(ledger_residual))
        and float(ledger_residual)
        <= limits["optimizer_ledger_relative_max"])
    source_edge = boundary.get("source_edge_certificate") or {}
    rates = boundary.get("rates") or {}

    prerequisites = {name: False for name in REQUIRED_PREREQUISITES}
    prerequisites.update(_window_prerequisites(summary, limits))
    prerequisites.update({
        "artifact_hashes_match": _hashes_match_declared(
            artifacts.values(), expected_hashes),
        "all_values_present": (
            not gaps and _required_estimates_present(summary)
            and residual_certificate.get("certificate_ready")
            and factory_certificate.get("certificate_ready")
            and ledger_residual is not None and bool(limits)
            and pilot.get("region_status") is not None),
        "all_values_finite": _finite({
            "family_p_values": summary.get("family_p_values"),
            "slope": summary.get("slope"),
            "regular_fraction": summary.get("regular_fraction"),
            "signature_fraction": summary.get("signature_frac"),
            "gn_full_residual_relative_max": residual,
            "factory_residual_relative_max":
                factory_certificate.get("relative_residual_max"),
            "decay_ledger_residual_max": ledger_residual,
        }),
        "chart_fidelity": _fidelity_pass(fidelity, limits)
        if all(fidelity.values()) and limits else False,
        "direction_separation": _separation_pass(separation)
        if all(separation.values()) else False,
        "factory_residual": factory_certificate.get("pass") is True,
        "decay_ledger_residual": ledger_pass,
        "source_edge_intervals": bool(
            source_edge.get("certificate_ready", False)
            and source_edge.get("classification") in
            {"capture", "release", "boundary"}),
        "gate_slope": bool(
            math.isfinite(float((boundary.get("gate_regression") or {})
                                .get("delta_J0", math.nan)))
            and float(boundary["gate_regression"]["delta_J0"]) > 0),
        "fiber_constants": _fiber_constants_pass(batteries)
        if all(batteries.values()) else False,
        "checkpoint_battery": all(
            isinstance(value, dict) and any(
                row.get("primary_defined") for row in value.get("ckpts", []))
            for value in batteries.values()),
        "pilot_complete": bool(
            pilot.get("region_status") is not None
            and pilot.get("selection") is not None),
    })

    decay_finite = all(
        isinstance(rates.get(name), dict)
        and math.isfinite(float(rates[name].get("th_loss", math.nan)))
        and math.isfinite(float(rates[name].get("decay_share", math.nan)))
        for name in ("a", "J", "w"))
    prerequisites["decay_ledger_residual"] = bool(
        prerequisites["decay_ledger_residual"] and decay_finite)

    metadata = summary.get("stress_metadata") or {}
    p_values = summary.get("family_p_values") or {
        name: None for name in INFERENTIAL_FAMILIES
    }
    directions = summary.get("family_directional_pass") or {
        name: None for name in INFERENTIAL_FAMILIES
    }
    record = E0Record(
        schema_version=SCHEMA_VERSION,
        protocol_hash=protocol_hash,
        schema_hash=schema_hash,
        source_hash=source_hash,
        calibration_hash=calibration_hash,
        curvature_kind=metadata.get("curvature"),
        metric_kind=metadata.get("metric"),
        unit_kind=metadata.get("unit"),
        residual_certificate_pass=residual_pass,
        n_upper_crossings=summary.get("n_upper_crossings"),
        pilot_region_status=pilot.get("region_status"),
        model_consistent=pilot.get("model_consistent"),
        uncertainty_resolved=pilot.get("uncertainty_resolved"),
        prerequisites=prerequisites,
        family_p_values=p_values,
        family_directional_pass=directions,
    )
    decision = decide_e0(record)

    record_dict = record.to_dict()
    decision_dict = decision.to_dict()
    record_bytes = json.dumps(
        record_dict, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    decision_dict["record_hash"] = hashlib.sha256(record_bytes).hexdigest()

    package = {
        "schema_version": SCHEMA_VERSION,
        "gaps": gaps,
        "artifact_hashes": artifact_hashes,
        "binding_hashes": expected_hashes,
        "acceptance_limits": limits,
        "record": record_dict,
        "decision": decision_dict,
        "certificates": {
            "gn_full_residual": residual_certificate,
            "factory_residual": factory_certificate,
            "optimizer_decay_ledger": {
                "residual_max": ledger_residual, "pass": ledger_pass},
        },
        "measurements": artifacts,
    }
    dump_json(record_dict, OUT / "E0_RECORD.json")
    dump_json(decision_dict, OUT / "E0_DECISION.json")
    dump_json(package, OUT / "E0_DATA_PACKAGE.json")

    lines = [
        "# E0 construct-validity decision",
        "",
        f"- outcome: {decision.outcome.value}",
        f"- strict upper crossings: {record.n_upper_crossings}",
        f"- inferential families: {len(INFERENTIAL_FAMILIES)}",
        f"- record hash: {decision_dict['record_hash']}",
    ]
    if decision.reasons:
        lines.extend(["", "## Reasons", ""])
        lines.extend(f"- {reason}" for reason in decision.reasons)
    if gaps:
        lines.extend(["", "## Missing artifacts", ""])
        lines.extend(f"- {name}: {paths[name]}" for name in gaps)
    (OUT / "E0_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return record, decision, package


def main():
    _, decision, _ = assemble()
    print(json.dumps(decision.to_dict(), indent=1))


if __name__ == "__main__":
    main()
