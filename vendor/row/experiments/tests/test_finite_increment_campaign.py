import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENTS))

from analysis.analyze_finite_increment_confirmation import (  # noqa: E402
    PREDICTION_SCHEMA,
    predict,
)
from confirm.confirmation_artifacts import validate_measurement_registry  # noqa: E402
from confirm.finite_increment_live import SOURCE_SCHEMA, write_npz_atomic  # noqa: E402
from confirm.finite_increment import layernorm_shape  # noqa: E402
from confirm.produce_finite_increment_certificate import produce  # noqa: E402
from scripts.stage_finite_increment_confirmation import build_plan  # noqa: E402


def test_execution_plan_has_training_replay_retention_and_no_step_plus_one():
    plan = build_plan()
    identifiers = {node["id"] for node in plan["nodes"]}
    assert plan["node_count"] == 193
    assert {
        "train-construction-s7444",
        "train-heldout-s8444",
        "train-heldout-s9444",
        "source-construction-s7444-a1000",
        "successor-construction-s7444-a1000",
        "retain-construction-s7444-a1000",
        "aggregate-consecutive-all-layer",
        "construction-lock",
    }.issubset(identifiers)
    successor = next(
        node for node in plan["nodes"]
        if node["id"] == "successor-construction-s7444-a1000"
    )
    assert "work/finite-construction-s7444/ckpt_1000.pt" in successor["argv"]
    assert not any("ckpt_1001.pt" in value for value in successor["argv"])
    assert plan["source_only_prediction"] is True
    assert plan["successor_replays_same_complete_source"] is True


def test_finite_increment_registry_is_owned_by_trainer_validator(tmp_path):
    from scripts.stage_finite_increment_confirmation import create_registry

    tokens = tmp_path / "tokens.npy"
    tokenizer = tmp_path / "tokenizer.model"
    np.arange(256 * 5200, dtype=np.uint16).tofile(tokens)
    tokenizer.write_bytes(b"finite-increment-tokenizer")
    registry = create_registry(tokens, tokenizer)
    assert validate_measurement_registry(registry, require_nonempty=True)
    assert len(registry["construction"]) == 8
    assert len(registry["validation"]) == 16
    assert max(
        row["chunk_index"]
        for row in registry["construction"] + registry["validation"]
    ) >= registry["trainer_training_limit"]


def test_source_only_certificate_and_prediction_cover_all_maps(tmp_path):
    rng = np.random.default_rng(42042)
    shape = layernorm_shape(
        rng.normal(size=(24, 3, 4, 64, 64))
    )
    gate0 = np.ones((3, 64), dtype=np.float64)
    gate1 = np.full((3, 64), 0.99, dtype=np.float64)
    rows0 = shape * gate0[None, :, None, None, :]
    rows1 = shape * gate1[None, :, None, None, :]
    plga = [{
        "layer": layer,
        "strict_crossing_count": 1,
        "touching_zero_count": 0,
        "minimum_power_base": 1.0e-9,
        "maximum_abs_z_secant": 1.0,
        "maximum_abs_p_secant": 1.0,
        "maximum_abs_multiplied_contribution": 1.0,
        "power_base_first_max_abs_residual": 0.0,
        "power_exponent_first_max_abs_residual": 0.0,
        "preactivation_telescope_max_abs_residual": 0.0,
        "plga_output_telescope_max_abs_residual": 0.0,
        "plga_output_scale": 1.0,
    } for layer in range(3)]
    metadata = {
        "role": "construction",
        "seed": 7444,
        "step": 1000,
        "checkpoint_step": 1000,
        "checkpoint_sha256": "0" * 64,
        "registry_sha256": "1" * 64,
        "data_order_sha256": "2" * 64,
        "data_cursor_before": 32000,
        "update_chunks": list(range(32)),
        "source_loss": 1.0,
        "source_parameter_sha256": "3" * 64,
        "predicted_parameter_sha256": "4" * 64,
        "context_ids": [f"c{index:02d}" for index in range(24)],
        "context_chunks": list(range(24)),
        "map_order": "context-major-layer-major-head-major-v1",
        "successor_evaluated": False,
        "plga": plga,
        "resources": {},
    }
    source = tmp_path / "source.npz"
    write_npz_atomic(
        source,
        schema_version=np.asarray(SOURCE_SCHEMA),
        metadata_json=np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )),
        source_shape=shape,
        predicted_shape=shape,
        source_rows=rows0,
        predicted_rows=rows1,
        source_gate=gate0,
        predicted_gate=gate1,
    )
    certificate = produce(source)
    assert certificate["successor_values_used"] is False
    assert certificate["map_count"] == 288
    assert certificate["pair_count"] == 580_608
    assert certificate["cross_map_pairs_formed"] is False
    assert certificate["all_identities_valid"] is True
    assert certificate["all_strict_state_dominance"] is True
    assert certificate["rounding_policy"] == "float64-directed-nextafter-v1"
    assert all(
        row["absolute_prediction"]["absolute_upper"]
        == row["predicted_diameter_squared_upper"]
        and row["absolute_prediction"]["ratio_denominator_valid"]
        and row["near_zero_gate_bound_covers_prediction"]
        for row in certificate["maps"]
    )
    certificate_path = tmp_path / "certificate.json"
    certificate_path.write_text(
        json.dumps(certificate, sort_keys=True) + "\n", encoding="utf-8"
    )
    prediction_path = tmp_path / "prediction.json"
    status = predict(argparse.Namespace(
        certificate=str(certificate_path), output=str(prediction_path)
    ))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert status == 0
    assert prediction["schema_version"] == PREDICTION_SCHEMA
    assert prediction["successor_values_used"] is False
    assert prediction["valid"] is True
