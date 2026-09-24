#!/usr/bin/env python3
"""Independent aggregate replay for the radial--tangential analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


SCHEMA = "pldr-radial-tangential-independent-check-v1"
TERM_NAMES = (
    "radial_linear",
    "radial_charge",
    "tangent_interaction",
    "tangent_charge",
)
COMPONENT_NAMES = ("shape", "gate", "interaction", "implementation_defect")
STEP_PATTERN = re.compile(r'"step"\s*:\s*(\d+)')


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def scalar(record: Any, name: str, kind: type) -> Any:
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} is not scalar")
    return kind(value.item())


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def signs(values: np.ndarray, floor: float) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.int8)
    result[values > floor] = 1
    result[values < -floor] = -1
    return result


def radial(z: np.ndarray, d: np.ndarray, eligible: np.ndarray) -> dict[str, np.ndarray]:
    energy = np.sum(z * z, axis=1)
    alpha = np.zeros(len(z))
    alpha[eligible] = -np.sum(z[eligible] * d[eligible], axis=1) / energy[eligible]
    tangent = d + alpha[:, None] * z
    tau = np.zeros(len(z))
    tau[eligible] = np.sum(tangent[eligible] ** 2, axis=1) / energy[eligible]
    gain = (1.0 - alpha) ** 2 + tau
    endpoint = z + d
    direct = np.zeros(len(z))
    direct[eligible] = np.sum(endpoint[eligible] ** 2, axis=1) / energy[eligible]
    orthogonal = np.sum(z * tangent, axis=1)
    return {
        "energy": energy,
        "alpha": alpha,
        "tangent": tangent,
        "tau": tau,
        "gain": gain,
        "direct": direct,
        "orthogonal": orthogonal,
    }


def technical_mask(record: Any, arm: str, source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    increment = np.asarray(record[f"{arm}_increment"], dtype=np.float64).reshape(96, -1)
    shape = np.asarray(record[f"{arm}_shape_secant"], dtype=np.float64).reshape(96, -1)
    gate = np.asarray(record[f"{arm}_gate_secant"], dtype=np.float64).reshape(96, -1)
    interaction = np.asarray(record[f"{arm}_interaction_secant"], dtype=np.float64).reshape(96, -1)
    source_defect = np.asarray(record[f"{arm}_source_factorization_residual"], dtype=np.float64).reshape(96, -1)
    endpoint_defect = np.asarray(record[f"{arm}_endpoint_factorization_residual"], dtype=np.float64).reshape(96, -1)
    defect = endpoint_defect - source_defect
    components = np.stack((shape, gate, interaction, defect), axis=1)
    residual = increment - np.sum(components, axis=1)
    scale = np.maximum(
        np.max(np.abs(increment), axis=1),
        np.max(np.sum(np.abs(components), axis=1), axis=1),
    )
    coordinate_ok = np.max(np.abs(residual), axis=1) <= 1.0e-12 * np.maximum(scale, 1.0e-300)
    endpoint = np.asarray(record[f"{arm}_endpoint_z"], dtype=np.float64).reshape(96, -1)
    stored_change = np.asarray(record[f"{arm}_energy_change"], dtype=np.float64).reshape(-1)
    direct_change = np.sum(endpoint * endpoint, axis=1) - np.sum(source * source, axis=1)
    energy_ok = np.abs(stored_change - direct_change) <= 1.0e-10 * np.maximum.reduce((
        np.abs(stored_change), np.abs(direct_change), np.full(96, 1.0e-300)
    ))
    work_residual = np.asarray(record[f"{arm}_work_charge_residual"], dtype=np.float64).reshape(-1)
    work_ok = np.abs(work_residual) <= 1.0e-10 * np.maximum(
        np.abs(stored_change), 1.0e-300
    )
    return coordinate_ok & energy_ok & work_ok, increment, components


def scalar_invariants(record: Any) -> bool:
    return bool(
        scalar(record, "paired_source_bitwise", bool)
        and scalar(record, "incoming_adjoint_bitwise", bool)
        and scalar(record, "outside_gradients_bitwise", bool)
        and scalar(record, "natural_parameter_replay_max_abs", float) == 0.0
        and scalar(record, "control_parameter_replay_max_abs", float) == 0.0
        and scalar(record, "natural_observation_replay_max_abs", float) == 0.0
        and scalar(record, "control_observation_replay_max_abs", float) == 0.0
        and scalar(record, "control_outgoing_centered_adjoint_norm", float) == 0.0
        and not scalar(record, "measurement_registry_matches_checkpoint", bool)
    )


def dense_points(path: Path, expected: set[int], map_count: int) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if '"mixed_collapse_dense_timepoint"' not in line and '"mixed_collapse_full_timepoint"' not in line:
                continue
            match = STEP_PATTERN.search(line)
            if match is None or int(match.group(1)) not in expected:
                continue
            outer = json.loads(line)
            point = outer.get("mixed_collapse_dense_timepoint") or outer.get("mixed_collapse_full_timepoint")
            if not isinstance(point, dict) or int(point["step"]) not in expected:
                continue
            step = int(point["step"])
            if step in result:
                raise ValueError("duplicate dense point")
            energy = np.asarray(point["energy"], dtype=np.float64)
            if energy.shape != (map_count,):
                raise ValueError("dense map count changed")
            result[step] = energy
    if set(result) != expected:
        raise ValueError("dense schedule changed")
    return result


def product_contract(dense: dict[str, Any]) -> dict[str, Any]:
    spec = json.loads(Path(dense["specification"]["path"]).read_text(encoding="utf-8"))
    expected = {int(value) for value in spec["dense_snapshot_updates"]}
    half = int(spec["dense_half_width"])
    terminal = int(spec["terminal_update"])
    centers = [int(value) for value in spec["dense_centers"]]
    maps = int(spec["registry"]["map_count"])
    positive_errors = []
    affine_errors = []
    positive_windows = face_windows = positive_edges = face_edges = 0
    rows = []
    for log in dense["logs"]:
        points = dense_points(Path(log["path"]), expected, maps)
        for center in centers:
            left, right = max(0, center - half), min(terminal, center + half)
            block = np.asarray([points[step] for step in range(left, right + 1)])
            source, endpoint = block[:-1], block[1:]
            positive = np.logical_and.reduce(source > 0.0, axis=0)
            gain = np.zeros_like(source)
            np.divide(endpoint, source, out=gain, where=source > 0.0)
            reopening = np.zeros_like(source)
            reopening[source == 0.0] = endpoint[source == 0.0]
            product = np.multiply.reduce(gain[:, positive], axis=0) if np.any(positive) else np.empty(0)
            replay = block[0, positive] * product
            scale = np.maximum(np.maximum(np.abs(replay), np.abs(block[-1, positive])), np.finfo(float).tiny)
            product_error = np.abs(replay - block[-1, positive]) / scale
            affine = block[0].copy()
            for edge in range(len(source)):
                affine *= gain[edge]
                affine += reopening[edge]
            affine_scale = np.maximum(np.maximum(np.abs(affine), np.abs(block[-1])), np.finfo(float).tiny)
            affine_error = np.abs(affine - block[-1]) / affine_scale
            positive_errors.append(product_error)
            affine_errors.append(affine_error)
            positive_windows += int(np.sum(positive))
            face_windows += int(np.sum(~positive))
            positive_edges += int(np.sum(source > 0.0))
            face_edges += int(np.sum(source == 0.0))
            rows.append({
                "trajectory": log["trajectory"],
                "center": center,
                "edge_count": right - left,
                "positive_map_windows": int(np.sum(positive)),
                "face_map_windows": int(np.sum(~positive)),
                "positive_product_max_relative_residual": float(np.max(product_error)) if len(product_error) else 0.0,
                "affine_face_max_relative_residual": float(np.max(affine_error)),
            })
    all_positive = np.concatenate(positive_errors)
    all_affine = np.concatenate(affine_errors)
    return {
        "window_count": len(rows),
        "map_windows": len(rows) * maps,
        "positive_map_windows": positive_windows,
        "face_map_windows": face_windows,
        "positive_source_edges": positive_edges,
        "zero_face_edges": face_edges,
        "positive_product_max_relative_residual": float(np.max(all_positive)) if len(all_positive) else 0.0,
        "affine_face_max_relative_residual": float(np.max(all_affine)),
        "all_finite_products_reconstructed": bool(
            np.max(all_positive, initial=0.0) <= 1.0e-12
            and np.max(all_affine, initial=0.0) <= 1.0e-12
        ),
        "rows": rows,
        "scope": "retained-scalar-energy-only",
        "radial_components_available": False,
    }


def replay(inventory: dict[str, Any]) -> dict[str, Any]:
    floor = float(inventory["effect_floor"])
    counts = {name: 0 for name in ("eligible", "contract", "preserve", "reopen", "radial_failure", "tangent_excess")}
    alphas = []
    taus = []
    gains = []
    dominance = {name: 0 for name in TERM_NAMES}
    raw_comparisons = raw_agreements = 0
    floor_comparisons = floor_agreements = 0
    paired_eligible = 0
    maxima = {name: 0.0 for name in (
        "source_energy_disagreement", "orthogonality_residual", "gain_identity_residual",
        "source_alpha_recomposition", "source_tangent_recomposition",
        "source_gram_recomposition", "paired_identity_residual", "stored_contrast_disagreement",
    )}
    cells = []
    for row in inventory["science_records"]:
        if sha256_path(Path(row["path"])) != row["sha256"]:
            raise ValueError("science record digest changed")
        with np.load(row["path"], allow_pickle=False) as record:
            source = np.asarray(record["source_z"], dtype=np.float64).reshape(96, -1)
            stored_energy = np.asarray(record["source_energy"], dtype=np.float64).reshape(-1)
            computed_energy = np.sum(source * source, axis=1)
            maxima["source_energy_disagreement"] = max(
                maxima["source_energy_disagreement"], float(np.max(np.abs(stored_energy - computed_energy)))
            )
            scalar_ok = scalar_invariants(record)
            source_ok = stored_energy > floor
            arms = {}
            source_data = {}
            cell = {
                "trajectory": row["trajectory"], "step": row["step"], "layer": row["layer"],
                "planned_maps": 96, "source_eligible_maps": int(np.sum(source_ok)), "arms": {},
            }
            for arm in ("natural", "control"):
                technical, increment, components = technical_mask(record, arm, source)
                eligible = source_ok & technical & scalar_ok
                values = radial(source, increment, eligible)
                contract = eligible & (values["gain"] < 1.0)
                preserve = eligible & (values["gain"] == 1.0)
                reopen = eligible & (values["gain"] > 1.0)
                interval = eligible & (values["alpha"] > 0.0) & (values["alpha"] < 2.0)
                radial_failure = reopen & ~interval
                tangent_excess = reopen & interval & (
                    values["tau"] > values["alpha"] * (2.0 - values["alpha"])
                )
                masks = {
                    "contract": contract, "preserve": preserve, "reopen": reopen,
                    "radial_failure": radial_failure, "tangent_excess": tangent_excess,
                }
                counts["eligible"] += int(np.sum(eligible))
                for name, mask in masks.items():
                    counts[name] += int(np.sum(mask))
                alphas.append(values["alpha"][eligible])
                taus.append(values["tau"][eligible])
                gains.append(values["gain"][eligible])
                maxima["orthogonality_residual"] = max(
                    maxima["orthogonality_residual"], float(np.max(np.abs(values["orthogonal"][eligible]), initial=0.0))
                )
                maxima["gain_identity_residual"] = max(
                    maxima["gain_identity_residual"], float(np.max(np.abs(values["direct"][eligible] - values["gain"][eligible]), initial=0.0))
                )
                component_alpha = np.zeros((96, 4))
                component_alpha[eligible] = -np.sum(
                    source[eligible, None, :] * components[eligible], axis=2
                ) / stored_energy[eligible, None]
                component_tangent = components + component_alpha[:, :, None] * source[:, None, :]
                alpha_error = np.abs(np.sum(component_alpha[eligible], axis=1) - values["alpha"][eligible])
                tangent_total = np.sum(component_tangent, axis=1)
                tangent_error = np.linalg.norm(tangent_total[eligible] - values["tangent"][eligible], axis=1)
                gram = np.einsum(
                    "mai,mbi->mab", component_tangent[eligible], component_tangent[eligible]
                ) / stored_energy[eligible, None, None]
                gram_error = np.abs(np.sum(gram, axis=(1, 2)) - values["tau"][eligible])
                maxima["source_alpha_recomposition"] = max(
                    maxima["source_alpha_recomposition"], float(np.max(alpha_error, initial=0.0))
                )
                maxima["source_tangent_recomposition"] = max(
                    maxima["source_tangent_recomposition"], float(np.max(tangent_error, initial=0.0))
                )
                maxima["source_gram_recomposition"] = max(
                    maxima["source_gram_recomposition"], float(np.max(gram_error, initial=0.0))
                )
                arms[arm] = {"eligible": eligible, "increment": increment, "radial": values}
                source_data[arm] = (component_alpha, gram)
                cell["arms"][arm] = {
                    "eligible": int(np.sum(eligible)),
                    **{name: int(np.sum(mask)) for name, mask in masks.items()},
                }
            eligible = arms["natural"]["eligible"] & arms["control"]["eligible"]
            paired_eligible += int(np.sum(eligible))
            h = arms["control"]["increment"] - arms["natural"]["increment"]
            h_values = radial(source, h, eligible)
            n_values = arms["natural"]["radial"]
            beta = h_values["alpha"]
            term0 = -2.0 * (1.0 - n_values["alpha"]) * beta
            term1 = beta * beta
            term2 = np.zeros(96)
            term2[eligible] = 2.0 * np.sum(
                n_values["tangent"][eligible] * h_values["tangent"][eligible], axis=1
            ) / stored_energy[eligible]
            term3 = h_values["tau"]
            terms = np.stack((term0, term1, term2, term3), axis=1)
            stored = np.asarray(record["intervention_contrast"], dtype=np.float64).reshape(-1)
            natural_endpoint = source + arms["natural"]["increment"]
            control_endpoint = source + arms["control"]["increment"]
            direct = np.sum(control_endpoint**2, axis=1) - np.sum(natural_endpoint**2, axis=1)
            maxima["stored_contrast_disagreement"] = max(
                maxima["stored_contrast_disagreement"], float(np.max(np.abs(stored[eligible] - direct[eligible]), initial=0.0))
            )
            identity = stored[eligible] / stored_energy[eligible] - np.sum(terms[eligible], axis=1)
            maxima["paired_identity_residual"] = max(
                maxima["paired_identity_residual"], float(np.max(np.abs(identity), initial=0.0))
            )
            indices = np.argmax(np.abs(terms[eligible]), axis=1)
            local_dominance = {}
            for index, name in enumerate(TERM_NAMES):
                local = int(np.sum(indices == index))
                dominance[name] += local
                local_dominance[name] = local
            radial_energy = (term0[eligible] + term1[eligible]) * stored_energy[eligible]
            full_raw_sign = np.sign(stored[eligible])
            radial_raw_sign = np.sign(radial_energy)
            raw_comparable = (full_raw_sign != 0) & (radial_raw_sign != 0)
            raw_local_agree = raw_comparable & (full_raw_sign == radial_raw_sign)
            raw_comparisons += int(np.sum(raw_comparable))
            raw_agreements += int(np.sum(raw_local_agree))
            local_raw_comparisons = int(np.sum(raw_comparable))
            full_floor_sign = signs(stored[eligible], floor)
            radial_floor_sign = signs(radial_energy, floor)
            floor_comparable = (full_floor_sign != 0) & (radial_floor_sign != 0)
            floor_local_agree = floor_comparable & (full_floor_sign == radial_floor_sign)
            floor_comparisons += int(np.sum(floor_comparable))
            floor_agreements += int(np.sum(floor_local_agree))
            local_floor_comparisons = int(np.sum(floor_comparable))
            cell["paired"] = {
                "eligible": int(np.sum(eligible)),
                "raw_sign_comparisons": local_raw_comparisons,
                "raw_sign_agreements": int(np.sum(raw_local_agree)),
                "raw_agreement_fraction": float(np.sum(raw_local_agree) / local_raw_comparisons) if local_raw_comparisons else None,
                "floor_sign_comparisons": local_floor_comparisons,
                "floor_sign_agreements": int(np.sum(floor_local_agree)),
                "floor_agreement_fraction": float(np.sum(floor_local_agree) / local_floor_comparisons) if local_floor_comparisons else None,
                "dominant_terms": local_dominance,
            }
            cells.append(cell)
    product = product_contract(inventory["dense_archive"])
    return {
        "records": len(inventory["science_records"]),
        "planned_maps": len(inventory["science_records"]) * 96,
        "arm_counts": counts,
        "alpha_quantiles": quantiles(np.concatenate(alphas)),
        "tau_squared_quantiles": quantiles(np.concatenate(taus)),
        "gain_squared_quantiles": quantiles(np.concatenate(gains)),
        "paired": {
            "eligible": paired_eligible,
            "dominant_terms": dominance,
            "raw_sign_comparisons": raw_comparisons,
            "raw_sign_agreements": raw_agreements,
            "raw_sign_disagreements": raw_comparisons - raw_agreements,
            "raw_agreement_fraction": float(raw_agreements / raw_comparisons),
            "floor_sign_comparisons": floor_comparisons,
            "floor_sign_agreements": floor_agreements,
            "floor_sign_disagreements": floor_comparisons - floor_agreements,
            "floor_agreement_fraction": float(floor_agreements / floor_comparisons),
        },
        "maximum_absolute_residuals": maxima,
        "cells": cells,
        "positive_excursion_product": product,
    }


def compare(expected: Any, observed: Any, path: str = "") -> tuple[int, float, list[str]]:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            return 1, 0.0, [path or "<root>"]
        count = 0
        maximum = 0.0
        failures: list[str] = []
        for name in sorted(expected):
            local_count, local_max, local_failures = compare(
                expected[name], observed[name], f"{path}.{name}" if path else name
            )
            count += local_count
            maximum = max(maximum, local_max)
            failures.extend(local_failures)
        return count, maximum, failures
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            return 1, 0.0, [path]
        count = 0
        maximum = 0.0
        failures = []
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            local_count, local_max, local_failures = compare(left, right, f"{path}[{index}]")
            count += local_count
            maximum = max(maximum, local_max)
            failures.extend(local_failures)
        return count, maximum, failures
    if isinstance(expected, float) or isinstance(observed, float):
        difference = abs(float(expected) - float(observed))
        scale = max(abs(float(expected)), abs(float(observed)), 1.0e-300)
        passed = difference <= 5.0e-12 * scale + 1.0e-11
        return 1, difference, [] if passed else [path]
    return 1, 0.0, [] if expected == observed else [path]


def generated(inventory_path: Path, summary_path: Path) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    content_digest = summary.pop("content_sha256")
    if hashlib.sha256(canonical_bytes(summary)).hexdigest() != content_digest:
        raise ValueError("radial summary content digest changed")
    summary["content_sha256"] = content_digest
    observed = replay(inventory)
    count, maximum, failures = compare(summary["independent_contract"], observed)
    result = {
        "schema_version": SCHEMA,
        "inventory_sha256": sha256_path(inventory_path),
        "summary_sha256": sha256_path(summary_path),
        "fields_compared": count,
        "maximum_absolute_float_difference": maximum,
        "float_comparison_tolerance": {
            "absolute": 1.0e-11,
            "relative": 5.0e-12,
            "rationale": "independent-float64-reduction-order",
        },
        "mismatched_fields": failures,
        "all_aggregate_fields_match": not failures,
        "implementation_independence": {
            "imports_primary_analyzer": False,
            "imports_radial_identity_module": False,
            "uses_direct_coordinate_formulas": True,
        },
        "recomputed_contract_sha256": hashlib.sha256(canonical_bytes(observed)).hexdigest(),
    }
    if failures:
        raise ValueError("independent radial replay disagrees: " + ", ".join(failures[:8]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    payload = canonical_bytes(generated(arguments.inventory, arguments.summary))
    if arguments.check:
        if not arguments.output.is_file() or arguments.output.read_bytes() != payload:
            raise SystemExit("independent radial check is stale")
    else:
        write_bytes(arguments.output, payload)
    print("independent radial check: all aggregate fields match")


if __name__ == "__main__":
    main()
