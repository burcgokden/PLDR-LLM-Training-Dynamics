#!/usr/bin/env python3
"""Replay retained physical-energy increments without reconstructing missing arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.direct_work import center_rows, work_charge  # noqa: E402
from confirm.direct_work_specs import NUMERICAL_POLICY  # noqa: E402


SCHEMA = "pldr-retained-energy-audit-v1"
DENSE_CAMPAIGN = "pldr-mixed-row-map-collapse-v1"
ENDPOINT_CAMPAIGN = "pldr-finite-increment-observable-balance-v1"
DENSE_RELATIVE = Path(
    "experiment-data/manuscript-revisions/rev47/"
    "mixed-row-map-collapse-confirmation"
)
ENDPOINT_RELATIVE = Path(
    "experiment-data/manuscript-revisions/rev50/"
    "finite-increment-observable-balance"
)
RELEASE_ID = 'rev55'  # retained acquisition identity
OUTPUT = ROOT / "docs" / "figures"
REPORT_PATH = OUTPUT / "retained_energy_audit.json"
MACRO_PATH = OUTPUT / "retained_energy_macros.tex"
TABLE_PATH = OUTPUT / "retained_energy_results.tex"
STEP_PATTERN = re.compile(r'"step"\s*:\s*(\d+)')


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError(f"unsafe manifest path: {value}")
    return Path(*candidate.parts)


def _bound_path(binding_root: Path, relative: Path) -> Path:
    root = binding_root.resolve(strict=True)
    candidate = (root / relative).resolve(strict=True)
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"audit source leaves binding root: {candidate}")
    if not candidate.is_dir():
        raise ValueError(f"audit source is not a directory: {candidate}")
    return candidate


def _manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"malformed manifest line in {path.name}")
        normalized = _safe_relative(relative).as_posix()
        if normalized in entries:
            raise ValueError(f"duplicate manifest member: {normalized}")
        entries[normalized] = digest
    return entries


def _verify_member(bundle: Path, entries: dict[str, str], relative: str) -> str:
    if relative not in entries:
        raise ValueError(f"manifest omits retained audit input: {relative}")
    path = bundle / _safe_relative(relative)
    actual = sha256_path(path)
    if actual != entries[relative]:
        raise ValueError(f"retained audit input drifted: {relative}")
    return actual


def _summary(value: Any) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("audit summary needs nonempty finite data")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def _within_tolerance(left: Any, right: Any) -> np.ndarray:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("audit comparison arrays disagree")
    absolute = float(NUMERICAL_POLICY["identity_absolute_tolerance"])
    relative = float(NUMERICAL_POLICY["identity_relative_tolerance"])
    return np.abs(a - b) <= absolute + relative * np.maximum.reduce(
        (np.abs(a), np.abs(b), np.ones_like(a))
    )


def independent_work_charge(source: Any, endpoint: Any) -> dict[str, np.ndarray]:
    """Second implementation of the centered quadratic-energy polarization."""

    x0 = np.asarray(source, dtype=np.float64)
    x1 = np.asarray(endpoint, dtype=np.float64)
    if x0.shape != x1.shape or x0.ndim < 2 or x0.shape[-2] < 2:
        raise ValueError("successive physical map arrays disagree")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(x1)):
        raise ValueError("successive physical map arrays must be finite")
    z0 = x0 - np.einsum("...rw->...w", x0)[..., None, :] / x0.shape[-2]
    z1 = x1 - np.einsum("...rw->...w", x1)[..., None, :] / x1.shape[-2]
    increment = z1 - z0
    energy0 = np.einsum("...rw,...rw->...", z0, z0)
    energy1 = np.einsum("...rw,...rw->...", z1, z1)
    signed_work = 2.0 * np.einsum("...rw,...rw->...", z0, increment)
    charge = np.einsum("...rw,...rw->...", increment, increment)
    change = energy1 - energy0
    return {
        "source_centered": z0,
        "endpoint_centered": z1,
        "increment": increment,
        "source_energy": energy0,
        "endpoint_energy": energy1,
        "energy_change": change,
        "signed_work": signed_work,
        "finite_step_charge": charge,
        "identity_residual": change - signed_work - charge,
        "reopening_charge": np.maximum(change, 0.0),
    }


def _read_dense_points(
    log_path: Path,
    expected_steps: set[int],
    map_count: int,
) -> tuple[dict[int, np.ndarray], dict[str, bool]]:
    points: dict[int, np.ndarray] = {}
    availability = {
        "physical_row_maps": False,
        "final_gate": True,
        "centered_normalized_shape": False,
        "normalized_shape_coordinate_energy": True,
    }
    dense_token = '"mixed_collapse_dense_timepoint"'
    full_token = '"mixed_collapse_full_timepoint"'
    with log_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            is_dense = dense_token in line
            is_full = full_token in line
            if not is_dense and not is_full:
                continue
            if not is_dense:
                match = STEP_PATTERN.search(line)
                if match is None or int(match.group(1)) not in expected_steps:
                    continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed dense audit JSON at {log_path.name}:{line_number}"
                ) from error
            point = row.get("mixed_collapse_dense_timepoint")
            if point is None:
                point = row.get("mixed_collapse_full_timepoint")
            if not isinstance(point, dict):
                continue
            step = int(point.get("step", -1))
            if step not in expected_steps:
                continue
            if point.get("campaign_id") != DENSE_CAMPAIGN:
                raise ValueError("dense audit point has the wrong campaign")
            if step in points:
                raise ValueError(f"duplicate dense audit point at update {step}")
            energy = np.asarray(point.get("energy"), dtype=np.float64)
            if (
                energy.shape != (map_count,)
                or not np.all(np.isfinite(energy))
                or np.any(energy < 0.0)
            ):
                raise ValueError(f"invalid retained energy vector at update {step}")
            required = {
                "layer_final_gate",
                "normalized_shape_coordinate_energy",
                "gate_shape_coordinate_energy",
            }
            if not required.issubset(point):
                raise ValueError("dense audit point omits registered factor energies")
            if any(
                name in point
                for name in (
                    "physical_row_maps",
                    "centered_physical_row_maps",
                    "centered_normalized_shape",
                    "normalized_shape",
                )
            ):
                raise ValueError("dense archive availability contract changed")
            points[step] = energy
    if set(points) != expected_steps:
        missing = sorted(expected_steps - set(points))
        extra = sorted(set(points) - expected_steps)
        raise ValueError(
            f"dense audit schedule mismatch: missing={missing[:4]}, extra={extra[:4]}"
        )
    return points, availability


def _compare_dense_window(
    computed: dict[str, Any], recorded: dict[str, Any]
) -> None:
    if (
        int(computed["left"]) != int(recorded["left"])
        or int(computed["center"]) != int(recorded["center"])
        or int(computed["right"]) != int(recorded["right"])
        or int(computed["edge_count"]) != int(recorded["edge_count"])
        or int(computed["maps_with_reopening"])
        != int(recorded["maps_with_reopening"])
    ):
        raise ValueError("independent dense-window counts do not replay")
    for name in (
        "exact_excursion",
        "exact_positive_variation",
        "exact_excursion_over_anchor",
        "exact_positive_variation_over_anchor",
    ):
        for statistic in ("minimum", "median", "p90", "maximum"):
            if not bool(
                _within_tolerance(
                    computed[name][statistic], recorded[name][statistic]
                ).item()
            ):
                raise ValueError(
                    f"independent dense replay disagrees for {name}.{statistic}"
                )


def audit_dense_archive(bundle: Path) -> dict[str, Any]:
    manifest_path = bundle / "MANIFEST.sha256"
    entries = _manifest(manifest_path)
    specification_relative = "protocol/training_campaign_spec.json"
    analysis_relative = "reports/final-analysis.json"
    specification_digest = _verify_member(
        bundle, entries, specification_relative
    )
    analysis_digest = _verify_member(bundle, entries, analysis_relative)
    specification = json.loads(
        (bundle / specification_relative).read_text(encoding="utf-8")
    )
    recorded_analysis = json.loads(
        (bundle / analysis_relative).read_text(encoding="utf-8")
    )
    if (
        specification.get("campaign_id") != DENSE_CAMPAIGN
        or recorded_analysis.get("campaign_id") != DENSE_CAMPAIGN
    ):
        raise ValueError("unknown retained dense archive identity")
    registry = specification["registry"]
    if (
        registry.get("map_count") != 288
        or registry.get("map_order")
        != "context-major-layer-major-head-major-v1"
        or registry.get("context_count") != 24
        or registry.get("layers") != [0, 1, 2]
        or registry.get("heads") != [0, 1, 2, 3]
    ):
        raise ValueError("retained dense map registry changed")
    expected_steps = {int(value) for value in specification["dense_snapshot_updates"]}
    centers = tuple(int(value) for value in specification["dense_centers"])
    half_width = int(specification["dense_half_width"])
    terminal = int(specification["terminal_update"])
    recorded_rows = {
        str(row["trajectory"]): row
        for row in recorded_analysis["trajectories"].values()
    }
    trajectory_rows = []
    all_changes = []
    all_reopenings = []
    layer_changes: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    log_digests: dict[str, str] = {}
    common_availability: dict[str, bool] | None = None
    for trajectory in specification["trajectories"]:
        name = str(trajectory["name"])
        label = str(trajectory["label"])
        relative = f"runs/{name}/log.jsonl"
        log_digests[label] = _verify_member(bundle, entries, relative)
        points, availability = _read_dense_points(
            bundle / relative, expected_steps, int(registry["map_count"])
        )
        if common_availability is None:
            common_availability = availability
        elif availability != common_availability:
            raise ValueError("dense archive array availability changes by trajectory")
        recorded = recorded_rows[name]
        recorded_windows = {
            int(row["center"]): row for row in recorded["dense_windows"]
        }
        local_changes = []
        local_reopenings = []
        maps_with_any = np.zeros(int(registry["map_count"]), dtype=bool)
        window_rows = []
        for center in centers:
            left = max(0, center - half_width)
            right = min(terminal, center + half_width)
            block = np.stack([points[step] for step in range(left, right + 1)])
            change = np.diff(block, axis=0)
            reopening = np.maximum(change, 0.0)
            excursion = np.maximum(np.max(block, axis=0) - block[0], 0.0)
            variation = np.sum(reopening, axis=0)
            scale = np.maximum(block[0], np.finfo(np.float64).tiny)
            computed = {
                "left": left,
                "center": center,
                "right": right,
                "edge_count": right - left,
                "exact_excursion": _summary(excursion),
                "exact_positive_variation": _summary(variation),
                "exact_excursion_over_anchor": _summary(excursion / scale),
                "exact_positive_variation_over_anchor": _summary(
                    variation / scale
                ),
                "maps_with_reopening": int(np.count_nonzero(variation > 0.0)),
            }
            _compare_dense_window(computed, recorded_windows[center])
            window_rows.append(computed)
            local_changes.append(change)
            local_reopenings.append(reopening)
            maps_with_any |= variation > 0.0
            reshaped = change.reshape(
                change.shape[0],
                int(registry["context_count"]),
                len(registry["layers"]),
                len(registry["heads"]),
            )
            for layer in registry["layers"]:
                layer_changes[int(layer)].append(reshaped[:, :, int(layer), :])
        changes = np.concatenate(local_changes, axis=0)
        reopenings = np.concatenate(local_reopenings, axis=0)
        all_changes.append(changes)
        all_reopenings.append(reopenings)
        positive = int(np.count_nonzero(changes > 0.0))
        trajectory_rows.append(
            {
                "label": label,
                "trajectory": name,
                "captured_update_edges": int(changes.shape[0]),
                "map_increments": int(changes.size),
                "positive_energy_increments": positive,
                "nonpositive_energy_increments": int(changes.size - positive),
                "positive_fraction": float(positive / changes.size),
                "maps_with_any_reopening": int(np.count_nonzero(maps_with_any)),
                "positive_variation": _summary(np.sum(reopenings, axis=0)),
                "maximum_one_step_reopening": float(np.max(reopenings)),
                "windows": window_rows,
            }
        )
    dense_changes = np.concatenate(all_changes, axis=0)
    dense_reopenings = np.concatenate(all_reopenings, axis=0)
    layer_rows = []
    for layer, blocks in layer_changes.items():
        values = np.concatenate([block.reshape(-1) for block in blocks])
        positive = int(np.count_nonzero(values > 0.0))
        layer_rows.append(
            {
                "layer": layer,
                "map_increments": int(values.size),
                "positive_energy_increments": positive,
                "positive_fraction": float(positive / values.size),
            }
        )
    positive = int(np.count_nonzero(dense_changes > 0.0))
    assert common_availability is not None
    return {
        "campaign_id": DENSE_CAMPAIGN,
        "manifest_sha256": sha256_path(manifest_path),
        "specification_sha256": specification_digest,
        "recorded_analysis_sha256": analysis_digest,
        "log_sha256": log_digests,
        "trajectory_count": len(trajectory_rows),
        "captured_update_edges": int(dense_changes.shape[0]),
        "map_increments": int(dense_changes.size),
        "positive_energy_increments": positive,
        "nonpositive_energy_increments": int(dense_changes.size - positive),
        "positive_fraction": float(positive / dense_changes.size),
        "positive_reopening_charge": _summary(dense_reopenings),
        "maximum_one_step_reopening": float(np.max(dense_reopenings)),
        "trajectory_rows": trajectory_rows,
        "layer_rows": layer_rows,
        "retained_array_availability": common_availability,
        "energy_and_reopening_status": "evaluable-from-retained-scalar-energy",
        "work_charge_status": "not-evaluable-from-retained-records",
        "work_charge_reason": "successive-physical-row-map-arrays-are-absent",
        "gate_shape_status": "not-evaluable-from-retained-records",
        "gate_shape_reason": (
            "final-gates-and-squared-coordinate-shape-energies-do-not-"
            "determine-signed-centered-normalized-shape-secants"
        ),
        "independent_replay": {
            "windows_checked": len(centers) * len(trajectory_rows),
            "recorded_summary_statistics_checked_per_window": 16,
            "all_within_fixed_float64_tolerance": True,
        },
    }


def _endpoint_inventory(records: Path) -> dict[str, int]:
    families = {
        "construction": len(list(records.glob("construction-*.npz"))),
        "prediction": len(list(records.glob("prediction-*.npz"))),
        "validation": len(list(records.glob("validation-*.npz"))),
        "qualification": len(list(records.glob("qualification-*.npz"))),
        "stratum_control": len(list(records.glob("stratum-control-*.npz"))),
    }
    if sum(families.values()) != 55 or families != {
        "construction": 9,
        "prediction": 18,
        "validation": 18,
        "qualification": 1,
        "stratum_control": 9,
    }:
        raise ValueError("retained endpoint record inventory changed")
    return families


def audit_endpoint_archive(bundle: Path) -> dict[str, Any]:
    from analysis.summarize_observable_balance_manuscript import verify_bundle

    provenance = verify_bundle(bundle)
    records = bundle / "records"
    inventory = _endpoint_inventory(records)
    changes = []
    reopenings = []
    rows = []
    maximum_identity_residual = 0.0
    maximum_independent_disagreement = 0.0
    maximum_stored_centering_residual = 0.0
    compared_values = 0
    for path in sorted(records.glob("construction-*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["campaign_id"].item()) != ENDPOINT_CAMPAIGN:
                raise ValueError(f"unknown endpoint record identity: {path.name}")
            horizons = [int(value) for value in archive["horizons"].tolist()]
            endpoints = np.asarray(archive["base_endpoints"], dtype=np.float64)
            trajectory = str(archive["trajectory"].item())
            layer = int(archive["layer"].item())
            source_step = int(archive["source_step"].item())
        if endpoints.shape != (len(horizons), 24, 4, 64, 64):
            raise ValueError(f"endpoint map population changed: {path.name}")
        pairs = [
            (index, index + 1)
            for index in range(len(horizons) - 1)
            if horizons[index + 1] == horizons[index] + 1
        ]
        if pairs != [(0, 1)]:
            raise ValueError(f"unexpected consecutive endpoint pairs: {path.name}")
        local_changes = []
        local_reopenings = []
        for left, right in pairs:
            source = endpoints[left]
            endpoint = endpoints[right]
            independent = independent_work_charge(source, endpoint)
            canonical = work_charge(center_rows(source), center_rows(endpoint))
            key_pairs = {
                "increment": "increment",
                "source_energy": "source_energy",
                "endpoint_energy": "endpoint_energy",
                "energy_change": "energy_change",
                "signed_work": "signed_work",
                "finite_step_charge": "finite_step_charge",
                "identity_residual": "identity_residual",
                "reopening_charge": "reopening_charge",
            }
            for independent_key, canonical_key in key_pairs.items():
                first = independent[independent_key]
                second = canonical[canonical_key]
                if not np.all(_within_tolerance(first, second)):
                    raise ValueError(
                        f"independent endpoint ledger disagrees: {path.name}"
                    )
                maximum_independent_disagreement = max(
                    maximum_independent_disagreement,
                    float(np.max(np.abs(first - second))),
                )
                compared_values += int(first.size)
            residual = independent["identity_residual"]
            scale = np.maximum.reduce(
                (
                    np.abs(independent["energy_change"]),
                    np.abs(independent["signed_work"])
                    + independent["finite_step_charge"],
                    np.ones_like(independent["energy_change"]),
                )
            )
            tolerance = float(NUMERICAL_POLICY["identity_absolute_tolerance"]) + (
                float(NUMERICAL_POLICY["identity_relative_tolerance"]) * scale
            )
            if not np.all(np.abs(residual) <= tolerance):
                raise ValueError(f"endpoint work-charge identity failed: {path.name}")
            maximum_identity_residual = max(
                maximum_identity_residual, float(np.max(np.abs(residual)))
            )
            maximum_stored_centering_residual = max(
                maximum_stored_centering_residual,
                float(np.max(np.abs(np.mean(source, axis=-2)))),
                float(np.max(np.abs(np.mean(endpoint, axis=-2)))),
            )
            local_changes.append(independent["energy_change"].reshape(-1))
            local_reopenings.append(independent["reopening_charge"].reshape(-1))
        unit_changes = np.concatenate(local_changes)
        unit_reopenings = np.concatenate(local_reopenings)
        changes.append(unit_changes)
        reopenings.append(unit_reopenings)
        positive = int(np.count_nonzero(unit_changes > 0.0))
        rows.append(
            {
                "record": path.name,
                "trajectory": trajectory,
                "layer": layer,
                "source_step": source_step,
                "source_horizon": horizons[0],
                "endpoint_horizon": horizons[1],
                "map_increments": int(unit_changes.size),
                "positive_energy_increments": positive,
                "nonpositive_energy_increments": int(
                    unit_changes.size - positive
                ),
                "positive_fraction": float(positive / unit_changes.size),
                "maximum_reopening": float(np.max(unit_reopenings)),
            }
        )
    all_changes = np.concatenate(changes)
    all_reopenings = np.concatenate(reopenings)
    positive = int(np.count_nonzero(all_changes > 0.0))
    return {
        "campaign_id": ENDPOINT_CAMPAIGN,
        "provenance": {
            "science_source_commit": provenance["science_source_commit"],
            "evidence_manifest_sha256": provenance["evidence_manifest_sha256"],
            "runtime_manifest_sha256": provenance["runtime_manifest_sha256"],
            "static_manifest_sha256": provenance["static_manifest_sha256"],
        },
        "record_inventory": inventory,
        "records_total": sum(inventory.values()),
        "physical_pair_records": len(rows),
        "records_without_successive_physical_maps": sum(inventory.values()) - len(rows),
        "successive_horizon_pairs": len(rows),
        "map_increments": int(all_changes.size),
        "positive_energy_increments": positive,
        "nonpositive_energy_increments": int(all_changes.size - positive),
        "positive_fraction": float(positive / all_changes.size),
        "energy_change": _summary(all_changes),
        "positive_reopening_charge": _summary(all_reopenings),
        "maximum_identity_residual": maximum_identity_residual,
        "maximum_independent_implementation_disagreement": (
            maximum_independent_disagreement
        ),
        "maximum_stored_row_mean_absolute_value": (
            maximum_stored_centering_residual
        ),
        "independent_values_compared": compared_values,
        "all_increments_within_fixed_float64_tolerance": True,
        "unit_rows": rows,
        "non_evaluable_families": {
            "prediction": "one-physical-endpoint-only",
            "validation": "response-secants-without-base-physical-endpoints",
            "qualification": "one-physical-endpoint-only",
            "stratum_control": "row-quotients-without-physical-map-arrays",
        },
        "gate_shape_status": "not-evaluable-from-retained-records",
        "gate_shape_reason": (
            "successive-physical-maps-are-retained-without-matching-"
            "final-gate-and-centered-normalized-shape-arrays"
        ),
    }


def x67_counts(endpoint_bundle: Path) -> dict[str, Any]:
    analysis = json.loads(
        (endpoint_bundle / "reports/final-analysis.json").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        row for row in analysis["heldout"]["rows"]
        if bool(row["cancellation_dominant"])
    ]
    by_trajectory: dict[str, int] = {}
    by_layer: dict[str, int] = {}
    for row in rows:
        trajectory = str(row["trajectory"])
        layer = str(int(row["layer"]))
        by_trajectory[trajectory] = by_trajectory.get(trajectory, 0) + 1
        by_layer[layer] = by_layer.get(layer, 0) + 1
    if (
        len(rows) != 6
        or sorted(by_trajectory.values()) != [1, 5]
        or by_layer != {"2": 4, "0": 1, "1": 1}
    ):
        raise ValueError("cancellation concentration counts changed")
    return {
        "cancellation_dominant_total": len(rows),
        "by_trajectory": by_trajectory,
        "by_layer": by_layer,
        "largest_trajectory_count": max(by_trajectory.values()),
        "layer_two_count": by_layer["2"],
        "interpretation": "descriptive-finite-concentration-only",
    }


def build_report(binding_root: Path) -> dict[str, Any]:
    dense_bundle = _bound_path(binding_root, DENSE_RELATIVE)
    endpoint_bundle = _bound_path(binding_root, ENDPOINT_RELATIVE)
    dense = audit_dense_archive(dense_bundle)
    endpoint = audit_endpoint_archive(endpoint_bundle)
    return {
        "schema_version": SCHEMA,
        "release_id": RELEASE_ID,
        "analysis_role": "cpu-only-retained-record-audit",
        "float64_tolerance": {
            "absolute": float(NUMERICAL_POLICY["identity_absolute_tolerance"]),
            "relative": float(NUMERICAL_POLICY["identity_relative_tolerance"]),
            "fixed_by": "direct-work-qualification-numerical-policy",
        },
        "dense_scalar_energy_archive": dense,
        "successive_physical_map_archive": endpoint,
        "cancellation_concentration": x67_counts(endpoint_bundle),
        "conclusions": {
            "dense_reopening_is_observed": dense["positive_energy_increments"] > 0,
            "full_map_work_charge_identity_replays": endpoint[
                "all_increments_within_fixed_float64_tolerance"
            ],
            "missing_arrays_are_not_reconstructed": True,
            "asymptotic_collapse_is_not_inferred": True,
        },
    }


def _tex_integer(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _tex_scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return rf"{mantissa:.3g}\times 10^{{{exponent}}}"








def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)




