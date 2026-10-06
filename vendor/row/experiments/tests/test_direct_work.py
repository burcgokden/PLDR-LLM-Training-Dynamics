from companion_paths import acquisition_identity
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.direct_work import (  # noqa: E402
    center_rows,
    gate_shape_secants,
    physical_energy,
    precision_resolved_secants,
    sign_summary,
    work_charge,
)
from confirm.direct_work_specs import (  # noqa: E402
    DECISION_POLICY,
    RELEASE_ID,
    campaign_design,
    validate_design,
)
from confirm.resource_clocks import (  # noqa: E402
    CURRENT_SCHEMA,
    normalize_resource_clocks,
)


def test_exact_work_charge_and_zero_face():
    generator = np.random.default_rng(5101)
    source = generator.normal(size=(7, 5, 4))
    endpoint = source + generator.normal(scale=0.2, size=source.shape)
    ledger = work_charge(source, endpoint)
    np.testing.assert_allclose(
        ledger["energy_change"],
        ledger["signed_work"] + ledger["finite_step_charge"],
        rtol=2e-15,
        atol=2e-13,
    )
    zero_ledger = work_charge(np.zeros_like(source), endpoint)
    np.testing.assert_allclose(
        zero_ledger["endpoint_energy"],
        zero_ledger["finite_step_charge"],
        rtol=0.0,
        atol=0.0,
    )
    assert np.all(zero_ledger["signed_work"] == 0.0)


def test_gate_shape_secant_and_complete_gram_charge():
    generator = np.random.default_rng(5102)
    shape0 = generator.normal(size=(6, 9, 8))
    shape1 = shape0 + generator.normal(scale=0.1, size=shape0.shape)
    gate0 = generator.normal(size=8)
    gate1 = gate0 + generator.normal(scale=0.05, size=8)
    result = gate_shape_secants(shape0, shape1, gate0, gate1)
    np.testing.assert_allclose(
        result["increment_factorized"],
        result["shape_secant"]
        + result["gate_secant"]
        + result["interaction_secant"],
        rtol=2e-14,
        atol=2e-14,
    )
    component_charge = np.sum(result["component_gram"], axis=(-2, -1))
    direct_charge = physical_energy(result["increment_factorized"])
    np.testing.assert_allclose(component_charge, direct_charge, rtol=2e-14)
    np.testing.assert_allclose(np.mean(center_rows(shape0), axis=-2), 0.0, atol=1e-15)


def test_precision_resolved_native_secant_and_four_source_gram():
    generator = np.random.default_rng(5201)
    shape0 = generator.normal(size=(6, 9, 8))
    shape1 = shape0 + generator.normal(scale=0.1, size=shape0.shape)
    gate0 = generator.normal(size=8)
    gate1 = gate0 + generator.normal(scale=0.05, size=8)
    factorized = gate_shape_secants(shape0, shape1, gate0, gate1)
    defect0 = generator.normal(scale=1.0e-6, size=shape0.shape)
    defect1 = generator.normal(scale=1.0e-6, size=shape0.shape)
    result = precision_resolved_secants(
        factorized["source_factorized"] + defect0,
        factorized["endpoint_factorized"] + defect1,
        shape0,
        shape1,
        gate0,
        gate1,
    )
    np.testing.assert_allclose(
        result["native_increment"],
        result["shape_secant"]
        + result["gate_secant"]
        + result["interaction_secant"]
        + result["implementation_defect_secant"],
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        result["implementation_defect_secant"],
        defect1 - defect0,
        rtol=2e-10,
        atol=2e-15,
    )
    component_charge = np.sum(
        result["precision_component_gram"], axis=(-2, -1)
    )
    direct_charge = physical_energy(result["native_increment"])
    np.testing.assert_allclose(component_charge, direct_charge, rtol=2e-14)


def test_sign_summary_keeps_neutral_and_evaluability_separate():
    summary = sign_summary(
        np.array([3.0, 2.0, -4.0, 0.01, 8.0]),
        np.array([True, True, True, True, False]),
        effect_floor=0.1,
    )
    assert summary == {
        "planned": 5,
        "evaluable": 4,
        "positive": 2,
        "negative": 1,
        "neutral": 1,
        "majority_sign": "positive",
        "sign_status": "positive",
        "majority_fraction": 0.5,
    }


def test_design_has_no_unresolved_layer_pooling_and_uses_active_release_id():
    design = campaign_design()
    validate_design(design)
    assert design["release_id"] == RELEASE_ID == acquisition_identity('direct-work-acquisition-release')
    assert design["decision_policy"] == DECISION_POLICY
    assert design["decision_policy"]["unresolved_layers_have_no_pooled_fallback"]
    assert design["execution_policy"]["binding_root_is_required"]


def test_stage_cli_requires_explicit_binding_root():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/stage_direct_work_confirmation.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "--binding-root" in result.stderr
    assert "required" in result.stderr


def test_current_resource_clock_names_are_distinct():
    clocks = normalize_resource_clocks({
        "schema_version": CURRENT_SCHEMA,
        "attempt_elapsed_wall_seconds": 11.0,
        "accelerator_reservation_wall_seconds": 9.0,
    }, producer_process_elapsed_seconds=3.0)
    assert clocks.attempt_elapsed_wall_seconds == 11.0
    assert clocks.accelerator_reservation_wall_seconds == 9.0
    assert clocks.producer_process_elapsed_seconds == 3.0
    assert not clocks.legacy_schema


def test_legacy_clock_name_is_accepted_only_with_legacy_schema():
    legacy = normalize_resource_clocks({
        "schema_version": "pldr-observable-balance-attempt-v1",
        "elapsed_seconds": 7.5,
        "planned_device": "cuda:1",
    })
    assert legacy.legacy_schema
    assert legacy.accelerator_reservation_wall_seconds == 7.5
    with pytest.raises(ValueError, match="legacy clock name"):
        normalize_resource_clocks({
            "schema_version": CURRENT_SCHEMA,
            "elapsed_seconds": 7.5,
            "attempt_elapsed_wall_seconds": 7.5,
            "accelerator_reservation_wall_seconds": 7.5,
        })
    with pytest.raises(ValueError, match="disagree with schema"):
        normalize_resource_clocks({
            "schema_version": "pldr-observable-balance-attempt-v1",
            "attempt_elapsed_wall_seconds": 7.5,
            "accelerator_reservation_wall_seconds": 7.5,
        })
    with pytest.raises(ValueError, match="unknown resource clock schema"):
        normalize_resource_clocks({
            "schema_version": "pldr-unknown-clock-v1",
            "elapsed_seconds": 7.5,
        })
