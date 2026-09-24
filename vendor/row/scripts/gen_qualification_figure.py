#!/usr/bin/env python3
"""Generate the Q-stage Taylor-remainder figure from a sealed live record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from confirmation_artifacts import (  # noqa: E402
    verify_sealed_record,
    write_json_atomic,
)


DEFAULT_RECORD = (
    ROOT.parent / "experiment-data" / "manuscript-revisions" / "rev33"
    / "confirmation-program" / "q_qualification.json"
)
DEFAULT_FIGURE = (
    ROOT / "docs" / "figures" / "qualification_taylor_remainder.pdf"
)
DEFAULT_DATA = (
    ROOT / "docs" / "figures" / "qualification_taylor_remainder.json"
)


def generate(record_path, figure_path, data_path):
    with Path(record_path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    verify_sealed_record(record)
    if (
        record.get("schema_version") != "pldr-q-live-record-v2"
        or record.get("decision") != "QUALIFIED"
    ):
        raise ValueError("figure input is not a qualified Q record")
    diagnostics = record["jvp_and_taylor"]
    row_records = diagnostics["taylor_rows"]
    row_indices = [int(item["row_index"]) + 1 for item in row_records]
    observed_rows = [
        float(item["observed_remainder"]) for item in row_records]
    certified_rows = [
        float(item["certified_remainder_upper"]) for item in row_records]
    if (
        len(row_records) != diagnostics["row_count"]
        or row_indices != list(range(1, len(row_records) + 1))
        or any(
            not 0 <= observed <= certified
            for observed, certified
            in zip(observed_rows, certified_rows)
        )
    ):
        raise ValueError("Q per-row Taylor remainder is not enclosed")
    observed = float(diagnostics["observed_remainder"])
    certified = float(diagnostics["certified_remainder_upper"])
    if not 0 <= observed <= certified:
        raise ValueError("Q complete-stack Taylor remainder is not enclosed")
    layernorm_floor = float(
        record["measured_geometry"]["layernorm_centered_floor"])
    stack_factor = float(diagnostics["operator_to_frobenius_factor"])
    data = {
        "schema_version": "pldr-q-taylor-figure-data-v2",
        "source_scientific_record_sha256":
            record["scientific_record_sha256"],
        "row_count": diagnostics["row_count"],
        "row_input_dimension": diagnostics["row_input_dimension"],
        "operator_to_frobenius_factor": stack_factor,
        "measured_layernorm_centered_floor": layernorm_floor,
        "observed_remainder": observed,
        "certified_remainder_upper": certified,
        "enclosure_ratio": observed / certified if certified else 0.0,
        "rows": [
            {
                **item,
                "enclosure_ratio":
                    item["observed_remainder"]
                    / item["certified_remainder_upper"]
                    if item["certified_remainder_upper"] else 0.0,
            }
            for item in row_records
        ],
    }
    write_json_atomic(data_path, data)

    figure, axis = plt.subplots(figsize=(5.2, 3.15))
    axis.plot(
        row_indices, observed_rows, "o-",
        color="#2C7FB8", linewidth=1.0, markersize=4,
        label="observed remainder",
    )
    axis.plot(
        row_indices, certified_rows, "^-",
        color="#D95F0E", linewidth=1.0, markersize=4,
        label="certified upper",
    )
    axis.set_yscale("log")
    axis.set_xlabel("registered physical row")
    axis.set_ylabel("directional Taylor remainder")
    axis.set_title(
        rf"All-row qualification: $\sqrt{{d_r}}={stack_factor:.3f}$; "
        f"measured LN floor {layernorm_floor:.3f}",
        fontsize=9,
    )
    tick_stride = max(1, len(row_indices) // 5)
    axis.set_xticks(row_indices[::tick_stride])
    axis.grid(axis="both", alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.tight_layout()
    destination = Path(figure_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        destination,
        metadata={
            "Creator": "gen_qualification_figure.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default=str(DEFAULT_RECORD))
    parser.add_argument("--figure", default=str(DEFAULT_FIGURE))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    arguments = parser.parse_args()
    generate(arguments.record, arguments.figure, arguments.data)
    print(arguments.figure)


if __name__ == "__main__":
    main()
