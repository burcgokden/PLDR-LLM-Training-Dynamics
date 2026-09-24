#!/usr/bin/env python3
"""Reduce a source-only endpoint capture to per-map collapse certificates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.finite_increment import (  # noqa: E402
    absolute_diameter_enclosure,
    directed_state_margin_bounds,
    exact_diameter_slack,
    gate_shape_state_margin,
    write_json_atomic,
)
from confirm.finite_increment_live import SOURCE_SCHEMA  # noqa: E402
from confirm.finite_increment_specs import (  # noqa: E402
    ARCHITECTURE,
    OPTIMIZER,
    REGISTRY,
    REPORT_SCHEMA,
)


CERTIFICATE_SCHEMA = "pldr-finite-increment-certificate-v1"


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"source artifact omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def _array(record: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in record:
        raise ValueError(f"source artifact omits {name}")
    value = np.asarray(record[name], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return np.ascontiguousarray(value)


def produce(source_path: str | Path) -> dict[str, Any]:
    source = Path(source_path).resolve()
    expected_maps = (
        int(REGISTRY["context_count"]),
        int(ARCHITECTURE["layers"]),
        int(ARCHITECTURE["heads"]),
        int(ARCHITECTURE["rows_per_map"]),
        int(ARCHITECTURE["head_width"]),
    )
    expected_gates = (
        int(ARCHITECTURE["layers"]),
        int(ARCHITECTURE["head_width"]),
    )
    with np.load(source, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != SOURCE_SCHEMA:
            raise ValueError("unknown finite-increment source schema")
        metadata = json.loads(_scalar(record, "metadata_json", str))
        source_shape = _array(record, "source_shape", expected_maps)
        predicted_shape = _array(record, "predicted_shape", expected_maps)
        source_rows = _array(record, "source_rows", expected_maps)
        predicted_rows = _array(record, "predicted_rows", expected_maps)
        source_gate = _array(record, "source_gate", expected_gates)
        predicted_gate = _array(record, "predicted_gate", expected_gates)
    if metadata.get("successor_evaluated") is not False:
        raise ValueError("prediction source already contains successor evidence")
    if len(metadata.get("context_ids", [])) != expected_maps[0]:
        raise ValueError("source metadata has an incomplete context registry")

    decay = 1.0 - float(OPTIMIZER["learning_rate"]) * float(
        OPTIMIZER["weight_decay"]
    )
    map_reports = []
    tolerance_factor = 8192.0
    epsilon = np.finfo(np.float64).eps
    for context in range(expected_maps[0]):
        for layer in range(expected_maps[1]):
            for head in range(expected_maps[2]):
                x0 = source_rows[context, layer, head]
                x1 = predicted_rows[context, layer, head]
                s0 = source_shape[context, layer, head]
                s1 = predicted_shape[context, layer, head]
                exact = exact_diameter_slack(x0, x1)
                state = gate_shape_state_margin(
                    s0,
                    s1,
                    source_gate[layer],
                    predicted_gate[layer],
                    decay_multiplier=decay,
                )
                directed = directed_state_margin_bounds(
                    s0,
                    s1,
                    source_gate[layer],
                    predicted_gate[layer],
                    source_rows=x0,
                    endpoint_rows=x1,
                )
                scale = max(
                    float(exact["source_diameter_squared"]),
                    float(exact["endpoint_diameter_squared"]),
                    1.0,
                )
                source_enclosed = bool(
                    np.all(
                        np.asarray(exact["source_energy"])
                        >= np.asarray(directed["source_energy_lower"])
                    )
                    and np.all(
                        np.asarray(exact["source_energy"])
                        <= np.asarray(directed["source_energy_upper"])
                    )
                )
                endpoint_enclosed = bool(
                    np.all(
                        np.asarray(exact["endpoint_energy"])
                        >= np.asarray(directed["endpoint_energy_lower"])
                    )
                    and np.all(
                        np.asarray(exact["endpoint_energy"])
                        <= np.asarray(directed["endpoint_energy_upper"])
                    )
                )
                identity_valid = bool(
                    exact["identity_residual"] <= tolerance_factor * epsilon * scale
                    and state["gate_balance_max_abs_residual"]
                    <= tolerance_factor * epsilon * scale
                    and state["shape_bound_minimum_slack"]
                    >= -tolerance_factor * epsilon * scale
                    and state["state_bound_covers_endpoint"]
                    and source_enclosed
                    and endpoint_enclosed
                    and directed["state_bound_covers_endpoint"]
                )
                exact_slack_lower = float(np.nextafter(
                    directed["source_diameter_squared_lower"]
                    - directed["endpoint_diameter_squared_upper"],
                    -np.inf,
                ))
                absolute = absolute_diameter_enclosure(
                    directed["endpoint_energy_lower"],
                    directed["endpoint_energy_upper"],
                    source_lower=directed["source_diameter_squared_lower"],
                    source_upper=directed["source_diameter_squared_upper"],
                )
                gain_norm = float(np.max(np.abs(
                    predicted_gate[layer]
                )))
                gain_squared_upper = float(np.nextafter(
                    gain_norm * gain_norm, np.inf
                ))
                gain_bound_upper = float(np.nextafter(
                    4.0 * float(ARCHITECTURE["head_width"])
                    * gain_squared_upper,
                    np.inf,
                ))
                map_reports.append({
                    "context": context,
                    "context_id": metadata["context_ids"][context],
                    "layer": layer,
                    "head": head,
                    "pair_count": int(REGISTRY["pairs_per_map"]),
                    "source_diameter_squared": float(
                        exact["source_diameter_squared"]
                    ),
                    "source_diameter_squared_lower": float(
                        directed["source_diameter_squared_lower"]
                    ),
                    "source_diameter_squared_upper": float(
                        directed["source_diameter_squared_upper"]
                    ),
                    "predicted_diameter_squared": float(
                        exact["endpoint_diameter_squared"]
                    ),
                    "predicted_diameter_squared_lower": float(
                        directed["endpoint_diameter_squared_lower"]
                    ),
                    "predicted_diameter_squared_upper": float(
                        directed["endpoint_diameter_squared_upper"]
                    ),
                    "maximizing_pair_source": [
                        int(value) for value in exact["pairs"][int(
                            np.argmax(exact["source_energy"])
                        )]
                    ],
                    "maximizing_pair_endpoint": [
                        int(value) for value in exact["pairs"][int(
                            np.argmax(exact["endpoint_energy"])
                        )]
                    ],
                    "minimum_exact_slack": float(exact["minimum_slack"]),
                    "minimum_exact_slack_lower": exact_slack_lower,
                    "absolute_prediction": {
                        key: value for key, value in absolute.items()
                    },
                    "near_zero_gate_diameter_squared_upper": (
                        gain_bound_upper
                    ),
                    "near_zero_gate_bound_covers_prediction": bool(
                        directed["endpoint_diameter_squared_upper"]
                        <= gain_bound_upper
                    ),
                    "diameter_identity_residual": float(
                        exact["identity_residual"]
                    ),
                    "realized_prediction_contracts": bool(
                        directed["endpoint_diameter_squared_upper"]
                        < directed["source_diameter_squared_lower"]
                    ),
                    "nominal_prediction_contracts": bool(
                        exact["strict_contraction"]
                    ),
                    "minimum_state_margin": float(
                        directed["minimum_state_margin_lower"]
                    ),
                    "minimum_state_margin_nominal": float(
                        state["minimum_state_margin"]
                    ),
                    "alpha_star_zero_forcing": (
                        None
                        if directed["alpha_star_zero_forcing_lower"] is None
                        else float(directed["alpha_star_zero_forcing_lower"])
                    ),
                    "state_bound_endpoint_squared": float(
                        directed["state_bound_endpoint_squared_upper"]
                    ),
                    "strict_state_dominance": bool(
                        directed["strict_contraction_certified"]
                    ),
                    "state_bound_covers_prediction": bool(
                        directed["state_bound_covers_endpoint"]
                    ),
                    "gate_balance_max_abs_residual": float(
                        state["gate_balance_max_abs_residual"]
                    ),
                    "shape_bound_minimum_slack": float(
                        state["shape_bound_minimum_slack"]
                    ),
                    "maximum_energy_interval_width": float(max(
                        np.max(
                            directed["source_energy_upper"]
                            - directed["source_energy_lower"]
                        ),
                        np.max(
                            directed["endpoint_energy_upper"]
                            - directed["endpoint_energy_lower"]
                        ),
                    )),
                    "interval_encloses_float64_pair_energies": bool(
                        source_enclosed and endpoint_enclosed
                    ),
                    "identity_valid": identity_valid,
                })

    plga = metadata.get("plga")
    if not isinstance(plga, list) or len(plga) != expected_maps[1]:
        raise ValueError("source metadata omits an all-layer PLGA report")
    plga_valid = all(
        row["minimum_power_base"] > 0.0
        and row["power_base_first_max_abs_residual"]
        <= tolerance_factor * epsilon * row["plga_output_scale"]
        and row["power_exponent_first_max_abs_residual"]
        <= tolerance_factor * epsilon * row["plga_output_scale"]
        and row["preactivation_telescope_max_abs_residual"]
        <= tolerance_factor * epsilon * row["plga_output_scale"]
        and row["plga_output_telescope_max_abs_residual"]
        <= tolerance_factor * epsilon * row["plga_output_scale"]
        for row in plga
    )
    report: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA,
        "edge_report_schema": REPORT_SCHEMA,
        "source_path": str(source),
        "source_sha256": sha256_path(source),
        "role": metadata["role"],
        "seed": int(metadata["seed"]),
        "step": int(metadata["step"]),
        "registry_sha256": metadata["registry_sha256"],
        "source_parameter_sha256": metadata["source_parameter_sha256"],
        "predicted_parameter_sha256": metadata["predicted_parameter_sha256"],
        "successor_values_used": False,
        "map_order": metadata["map_order"],
        "map_count": len(map_reports),
        "pair_count": len(map_reports) * int(REGISTRY["pairs_per_map"]),
        "cross_map_pairs_formed": False,
        "decay_multiplier": decay,
        "rounding_policy": "float64-directed-nextafter-v1",
        "rounding_backend_obligations": [
            "IEEE-754 binary64 basic operations are correctly rounded",
            "numpy.nextafter returns the adjacent binary64 number",
            "numpy.sqrt is correctly rounded",
            "all inputs and intermediate interval endpoints are finite",
        ],
        "interval_reduction_order": (
            "pair-major-feature-major-serial-v1"
        ),
        "plga": plga,
        "plga_identity_valid": bool(plga_valid),
        "maps": map_reports,
        "all_identities_valid": bool(
            plga_valid and all(row["identity_valid"] for row in map_reports)
        ),
        "all_strict_state_dominance": bool(
            all(row["strict_state_dominance"] for row in map_reports)
        ),
        "all_predicted_contractions": bool(
            all(row["realized_prediction_contracts"] for row in map_reports)
        ),
        "minimum_state_margin": min(
            row["minimum_state_margin"] for row in map_reports
        ),
        "maximum_predicted_diameter_squared": max(
            row["predicted_diameter_squared"] for row in map_reports
        ),
        "maximum_predicted_diameter_squared_upper": max(
            row["predicted_diameter_squared_upper"]
            for row in map_reports
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = produce(arguments.source)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "maps": report["map_count"],
        "all_identities_valid": report["all_identities_valid"],
        "all_strict_state_dominance": report["all_strict_state_dominance"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
