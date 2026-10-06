import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from confirm.confirmation_artifacts import digest_object, validate_measurement_registry
from confirm.contrast_energy_specs import CAMPAIGN_ID, TAYLOR_POLICY
from confirm.contrast_energy import (
    aggregate_cocycle,
    cell_second_bounds,
    cell_weights,
    finite_registry_prediction,
    integrate_pair_energy,
)
from confirm.explicit_layernorm import affine_path_coefficients, explicit_layer_norm
from confirm.produce_validated_contrast_energy_jets import (
    INPUT_SCHEMA,
    SOURCE_SCHEMA,
    produce,
)
from analysis.aggregate_diameter_cocycle import aggregate


ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(filename, module_name):
    path = ROOT / "scripts" / filename
    specification = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _load_stage_module():
    return _load_script_module(
        "stage_contrast_energy_confirmation.py",
        "stage_contrast_energy_confirmation_test",
    )


def test_signed_cell_integration_recovers_exact_quadratic_energy():
    knots = np.asarray([0.0, 0.25, 0.5, 1.0])
    assert np.isclose(np.sum(cell_weights(knots)), 0.5)
    source = np.asarray([1.0, 4.0, 9.0])
    slope = -0.5 * source
    second = 0.125 * source
    endpoints = np.repeat(second[:, None], len(knots), axis=1)
    cells = cell_second_bounds(
        np.nextafter(endpoints, -np.inf),
        np.nextafter(endpoints, np.inf),
        np.full((3, 3), 1.0e-12),
        knots,
    )
    enclosure = integrate_pair_energy(
        source, slope, cells["cell_lower"], cells["cell_upper"], knots,
        source_charge=1.0e-14, arithmetic_charge=1.0e-13,
    )
    realized = 0.5625 * source
    assert np.all(enclosure["successor_lower"] <= realized)
    assert np.all(realized <= enclosure["successor_upper"])


def test_finite_maximum_certifies_both_signs_and_uses_source_only():
    pairs = np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    source = np.asarray([1.0, 4.0, 9.0])
    contraction = finite_registry_prediction(pairs, {
        "source_lower": source - 1.0e-12,
        "source_upper": source + 1.0e-12,
        "successor_lower": 0.5 * source - 1.0e-12,
        "successor_upper": 0.5 * source + 1.0e-12,
    })
    assert contraction["decision"] == "contraction"
    assert contraction["decision_uses_successor"] is False
    assert contraction["maxima"]["source_upper"]["pair"] == [1, 2]
    expansion = finite_registry_prediction(pairs, {
        "source_lower": source - 1.0e-12,
        "source_upper": source + 1.0e-12,
        "successor_lower": 1.5 * source - 1.0e-12,
        "successor_upper": 1.5 * source + 1.0e-12,
    }, successor_energy=1.5 * source)
    assert expansion["decision"] == "reexpansion"
    assert expansion["coverage"]["all_pairs_enclosed"]
    assert expansion["coverage"]["strict_sign_correct"]


def test_cocycle_allows_reexpansion_and_multiplies_in_order():
    result = aggregate_cocycle([
        {"lower_multiplier": 0.4, "upper_multiplier": 0.5},
        {"lower_multiplier": 1.1, "upper_multiplier": 1.2},
        {"lower_multiplier": 0.2, "upper_multiplier": 0.25},
    ])
    assert np.allclose(result["upper_products"], [0.5, 0.6, 0.15])
    assert result["temporary_reexpansion_count"] == 1
    assert result["certified_contraction_count"] == 2


def test_explicit_layernorm_coefficients_match_finite_differences():
    rng = np.random.default_rng(41002)
    value = rng.normal(size=(3, 7))
    direction = rng.normal(size=(3, 7)) / 4
    gain = 0.5 + rng.random(7)
    bias = rng.normal(size=7)
    coefficients = affine_path_coefficients(value, direction, gain, bias)
    step = 2.0e-5
    plus = explicit_layer_norm(value + step * direction, gain, bias)
    minus = explicit_layer_norm(value - step * direction, gain, bias)
    first = (plus - minus) / (2.0 * step)
    second = (plus - 2.0 * coefficients["value"] + minus) / (step * step)
    assert np.allclose(first, coefficients["first"], rtol=2e-8, atol=2e-8)
    assert np.allclose(second, coefficients["second"], rtol=2e-5, atol=2e-5)


def test_validated_jet_producer_binds_source_and_rejects_successor(tmp_path):
    source = tmp_path / "source.npz"
    physical = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    physical_jvp = -0.25 * physical
    with source.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema_version=np.asarray(SOURCE_SCHEMA),
            campaign_id=np.asarray(CAMPAIGN_ID),
            successor_evaluated=np.asarray(False),
            step_start=np.asarray(1000, dtype=np.int64),
            optimizer_step_after=np.asarray([1001], dtype=np.int64),
            normalized_rows=np.zeros((1, 2, 3)),
            gate=np.ones((1, 3)),
            physical_rows=physical,
            physical_row_jvp=physical_jvp,
            source_loss=np.asarray([1.0]),
            raw_gradient_sha256=np.asarray(["3" * 64]),
            clipped_gradient_sha256=np.asarray(["4" * 64]),
            optimizer_state_sha256=np.asarray(["5" * 64]),
            source_parameter_sha256=np.asarray(["6" * 64]),
            parameter_displacement_sha256=np.asarray(["7" * 64]),
            predicted_successor_parameter_sha256=np.asarray(["9" * 64]),
            successor_parameter_sha256=np.asarray([""]),
            full_parameter_update_bitwise_match=np.asarray(
                [], dtype=np.bool_),
            update_batch_sha256=np.asarray(["8" * 64]),
            data_cursor_before=np.asarray(7, dtype=np.int64),
            data_cursor_after=np.asarray(11, dtype=np.int64),
            update_batch_chunk_indices=np.asarray([[7, 8, 9, 10]]),
            checkpoint_sha256=np.asarray("0" * 64),
            registry_sha256=np.asarray("1" * 64),
            lock_sha256=np.asarray(""),
            data_order_sha256=np.asarray("2" * 64),
            role=np.asarray("development"),
            seed=np.asarray(7444, dtype=np.int64),
            layer=np.asarray(0, dtype=np.int64),
        )
    import hashlib
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    model = tmp_path / "model.npz"
    metadata = {
        "model_order": 3, "interval_backend": "test-exact",
        "explicit_layernorm": True, "rope_interval_enclosure": True,
        "adaptive_subdivision_rule": TAYLOR_POLICY["adaptive_rule"],
        "taylor_policy_sha256": digest_object(TAYLOR_POLICY),
        "backend_manifest_sha256": "b" * 64,
        "source_parameter_sha256": "6" * 64,
        "parameter_displacement_sha256": "7" * 64,
        "predicted_successor_parameter_sha256": "9" * 64,
    }
    common = {
        "schema_version": np.asarray(INPUT_SCHEMA),
        "source_sha256": np.asarray(source_digest),
        "pairs": np.asarray([[0, 1]], dtype=np.int64),
        "source_energy": np.asarray([1.0]),
        "source_slope": np.asarray([-0.5]),
        "knots": np.asarray([0.0, 0.5, 1.0]),
        "endpoint_second_lower": np.asarray([[0.125, 0.125, 0.125]]),
        "endpoint_second_upper": np.asarray([[0.125, 0.125, 0.125]]),
        "third_modulus": np.asarray([[1.0e-12, 1.0e-12]]),
        "source_charge": np.asarray([0.0]),
        "arithmetic_charge": np.asarray([0.0]),
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    with model.open("wb") as stream:
        np.savez_compressed(stream, successor_values_present=np.asarray(False), **common)
    assert produce(source, model)["successor_values_present"].item() is False
    with model.open("wb") as stream:
        np.savez_compressed(
            stream,
            successor_values_present=np.asarray(False),
            **{**common, "source_energy": np.asarray([1.25])},
        )
    with pytest.raises(ValueError, match="source energies disagree"):
        produce(source, model)
    with model.open("wb") as stream:
        np.savez_compressed(stream, successor_values_present=np.asarray(True), **common)
    with pytest.raises(ValueError, match="cannot contain successor"):
        produce(source, model)


def test_registry_and_plan_have_all_layers_and_no_interventions(tmp_path):
    registry = {
        "schema_version": "pldr-contrast-energy-registry-v1",
        "context_length": 256,
        "construction": [{"id": f"c{i}", "chunk_index": i} for i in range(8)],
        "validation": [{"id": f"v{i}", "chunk_index": i + 8} for i in range(16)],
        "anchors": list(range(1000, 9001, 500)),
        "context_head_block_rule": "head-equals-context-index-modulo-four-v1",
        "block_row_rule": "complete-ordered-generator-rows-0-through-63-v1",
        "history_length": 48,
        "pair_rule": "all-unordered-pairs-lexicographic-v1",
        "layers": [0, 1, 2],
        "dataset_sha256": "0" * 64,
        "tokenizer_sha256": "1" * 64,
    }
    registry["registry_sha256"] = digest_object(registry)
    assert validate_measurement_registry(registry, require_nonempty=True)
    stage = _load_stage_module()
    plan = stage.build_plan(
        tmp_path, tmp_path / "tokens.npy", tmp_path / "tokenizer.model",
        tmp_path / "protocol" / "registry.json")
    assert "intervention" not in json.dumps(plan).lower()
    assert len(plan["required_taylor_models"]) == 306
    jet_nodes = [node for node in plan["nodes"] if node["id"].startswith("validated-jets-")]
    assert len(jet_nodes) == 306 and all(node["argv"] for node in jet_nodes)
    assert all(
        ("--lock" in node["argv"]) == (node["role"] == "heldout")
        for node in jet_nodes
    )
    source_nodes = [
        node for node in plan["nodes"] if node["id"].startswith("source-")]
    assert len(source_nodes) == 306
    assert all("--data-order" in node["argv"] for node in source_nodes)
    retention_nodes = [
        node for node in plan["nodes"] if node["id"].startswith("retain-")]
    assert len(retention_nodes) == 102
    assert all(len(node["depends_on"]) == 3 for node in retention_nodes)
    assert all(
        node["argv"][1].endswith("retain_model_only_checkpoint.py")
        for node in retention_nodes
    )
    resource_nodes = [
        node for node in plan["nodes"]
        if node["id"] == "stage-b-resource-lock"]
    assert len(resource_nodes) == 1
    assert len(resource_nodes[0]["depends_on"]) == 6
    stage_b_sources = [
        node for node in source_nodes if node["stage"] == "B"]
    stage_c_sources = [
        node for node in source_nodes if node["stage"] == "C"]
    assert len(stage_b_sources) == 6
    assert len(stage_c_sources) == 96
    assert all(
        "stage-b-resource-lock" in node["depends_on"]
        for node in stage_c_sources
    )

def test_cocycle_aggregate_splits_nonconsecutive_anchor_edges(tmp_path):
    reports = []
    for step, lower, upper in (
        (1000, 0.8, 0.9),
        (1001, 1.05, 1.1),
        (1500, 0.7, 0.8),
    ):
        report = tmp_path / f"edge-{step}.json"
        report.write_text(json.dumps({
            "schema_version": "pldr-contrast-energy-analysis-v1",
            "valid": True,
            "metadata": {
                "trajectory": "development:seed7444",
                "role": "development",
                "seed": 7444,
                "layer": 0,
                "global_step_before": step,
                "global_step_after": step + 1,
                "registry_sha256": "1" * 64,
                "lock_sha256": "",
            },
            "prediction": {
                "decision_uses_successor": False,
                "lower_multiplier": lower,
                "upper_multiplier": upper,
                "decision": "unresolved",
            },
        }, sort_keys=True), encoding="utf-8")
        reports.append(report)
    result = aggregate(reports)
    group = result["groups"]["development:seed7444:layer0"]
    assert group["sampled_edge_count"] == 3
    assert group["contiguous_block_count"] == 2
    assert group["longest_contiguous_block"] == 2
    assert np.allclose(
        group["blocks"][0]["analysis"]["upper_products"], [0.9, 0.99])
    assert group["blocks"][1]["analysis"]["upper_products"] == [0.8]

def test_stage_b_resource_lock_replays_logs_and_drives_executor_cap(tmp_path):
    resource_module = _load_script_module(
        "finalize_contrast_energy_resources.py",
        "finalize_contrast_energy_resources_test",
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    node_ids = ["source-development", "source-construction"]
    for node_id, device, peak in (
        (node_ids[0], "cuda:0", 1000),
        (node_ids[1], "cuda:1", 2000),
    ):
        record = {
            "schema_version": "pldr-contrast-energy-node-log-v1",
            "node_id": node_id,
            "stage": "B",
            "role": "qualification",
            "device": device,
            "resolved_argv": ["python", "fixture.py"],
            "parent_records": [],
            "exit_code": 0,
            "missing_outputs": [],
            "output_records": [],
            "stdout_path": str(log_dir / f"{node_id}.stdout"),
            "stdout_sha256": "0" * 64,
            "resources": {
                "wall_seconds": 1.0,
                "peak_gpu_memory_mib": peak,
                "peak_host_rss_bytes": 4096,
            },
        }
        record["record_sha256"] = resource_module.digest_object(record)
        (log_dir / f"{node_id}.json").write_text(
            json.dumps(record, sort_keys=True), encoding="utf-8")

    lock, passed, failures = resource_module.build_lock(
        log_dir,
        node_ids,
        provisional_peak_mib=20 * 1024,
        enlargement=1.15,
    )
    assert passed and not failures
    assert lock["sealed_peak_cap_mib"] == {
        "cuda:0": 1150,
        "cuda:1": 2300,
    }
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    (protocol / "stage-b-resource-lock.json").write_text(
        json.dumps(lock, sort_keys=True), encoding="utf-8")
    executor = _load_script_module(
        "execute_contrast_energy_plan.py",
        "execute_contrast_energy_plan_test",
    )
    lock_node = {"depends_on": []}
    node = {
        "id": "source-stage-c",
        "device": "cuda:0",
        "depends_on": ["stage-b-resource-lock"],
    }
    cap, source, digest = executor._resource_cap(
        node, {"stage-b-resource-lock": lock_node}, tmp_path)
    assert (cap, source, digest) == (
        1150, "measured_stage_b", lock["resource_lock_sha256"])

    tampered_path = log_dir / f"{node_ids[0]}.json"
    tampered = json.loads(tampered_path.read_text(encoding="utf-8"))
    tampered["resources"]["peak_gpu_memory_mib"] = 999
    tampered_path.write_text(
        json.dumps(tampered, sort_keys=True), encoding="utf-8")
    _lock, passed, failures = resource_module.build_lock(
        log_dir,
        node_ids,
        provisional_peak_mib=20 * 1024,
        enlargement=1.15,
    )
    assert not passed
    assert f"invalid Stage B resource log {node_ids[0]}" in failures
