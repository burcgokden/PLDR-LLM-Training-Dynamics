"""Execution-graph tests for the rev36 row-map confirmation campaign."""

from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from confirm.gate_shape_evidence import digest_object, write_json_atomic
from confirm.run_row_map_confirmation import (
    execution_plan,
    intervention_plan,
    select_interventions,
)
ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "execute_row_map_plan",
    ROOT / "scripts/execute_row_map_plan.py")
executor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(executor)


def _arguments(tmp_path):
    return SimpleNamespace(
        run_root=tmp_path / "runs",
        tokens=tmp_path / "tokens.bin",
        tokenizer=tmp_path / "tokenizer.model",
        output_root=tmp_path / "results",
        devices="cuda:0,cuda:1",
    )


def _signed(value, key):
    value[key] = digest_object(value)
    return value


def _geometry_record(plan, ratios):
    intervals = []
    for role, seed, values in (
        ("construction", 1234, ratios),
        ("validation", 2222, [0.01, 0.01, 0.01]),
    ):
        for layer, ratio in enumerate(values):
            intervals.append({
                "run_name": "construction" if role == "construction" else "heldout",
                "seed": seed,
                "role": role,
                "layer": layer,
                "source_step": 1000,
                "target_step": 2000,
                "energy_ratio": ratio,
            })
    value = {
        "schema_version": "pldr-row-map-analysis-v1",
        "kind": "geometry",
        "inputs": [],
        "checks": {
            "all_node_identities": True,
            "all_graph_bounds": True,
            "all_covered_domain_bounds": True,
        },
        "results": {"intervals": intervals},
        "all_checks_pass": True,
        "decision": "CONFIRMED",
    }
    return _signed(value, "analysis_sha256")


def test_plan_places_each_analyzer_after_its_own_gpu_stage(tmp_path):
    plan = execution_plan(_arguments(tmp_path))
    counts = Counter(job["stage"] for job in plan["jobs"])
    assert counts == {
        "setup": 2, "Q": 3, "G": 55, "H": 28, "D": 20, "A": 37,
    }
    assert plan["required_live_stage_sequence"] == [
        "setup", "Q", "G", "H", "D", "A"]
    for stage, final_job in (
        ("Q", "analyze-blocks"),
        ("G", "analyze-geometry"),
        ("H", "analyze-loss"),
        ("D", "analyze-dynamics"),
        ("A", "analyze-plga"),
    ):
        jobs = [job for job in plan["jobs"] if job["stage"] == stage]
        assert jobs[-1]["job_id"] == final_job
        assert jobs[-1]["device"] == "cpu"
    qualification = [
        job for job in plan["jobs"]
        if job["stage"] == "Q" and job["job_id"] == "analyze-blocks"][0]
    assert "--require-pass" in qualification["command"]


def test_construction_only_selection_ignores_smaller_validation_ratios(tmp_path):
    plan = execution_plan(_arguments(tmp_path))
    plan_path = tmp_path / "plan.json"
    geometry_path = tmp_path / "results/R/geometry-analysis.json"
    write_json_atomic(plan_path, plan)
    write_json_atomic(
        geometry_path, _geometry_record(plan, [0.5, 0.2, 0.4]))
    selection = select_interventions(SimpleNamespace(
        base_plan=plan_path,
        geometry_analysis=geometry_path,
    ))
    assert selection["layers"] == [1]
    assert not selection["fallback_minimum_ratio_rule_used"]
    unsigned = dict(selection)
    recorded = unsigned.pop("selection_sha256")
    assert recorded == digest_object(unsigned)


def test_selection_fallback_has_deterministic_lower_layer_tie_break(tmp_path):
    plan = execution_plan(_arguments(tmp_path))
    plan_path = tmp_path / "plan.json"
    geometry_path = tmp_path / "results/R/geometry-analysis.json"
    write_json_atomic(plan_path, plan)
    write_json_atomic(
        geometry_path, _geometry_record(plan, [0.5, 0.4, 0.4]))
    selection = select_interventions(SimpleNamespace(
        base_plan=plan_path,
        geometry_analysis=geometry_path,
    ))
    assert selection["layers"] == [1]
    assert selection["fallback_minimum_ratio_rule_used"]


def test_intervention_plan_carries_the_remaining_campaign_budget(tmp_path):
    args = _arguments(tmp_path)
    plan = execution_plan(args)
    plan_path = tmp_path / "plan.json"
    geometry_path = tmp_path / "results/R/geometry-analysis.json"
    selection_path = tmp_path / "selection.json"
    summary_path = tmp_path / "summary.json"
    write_json_atomic(plan_path, plan)
    geometry = _geometry_record(plan, [0.5, 0.2, 0.4])
    write_json_atomic(geometry_path, geometry)
    selection = select_interventions(SimpleNamespace(
        base_plan=plan_path,
        geometry_analysis=geometry_path,
    ))
    write_json_atomic(selection_path, selection)
    summary = {
        "schema_version": "pldr-row-map-execution-summary-v1",
        "plan_sha256": plan["plan_sha256"],
        "selected_stages": ["setup", "Q", "G", "H", "D", "A"],
        "dry_run": False,
        "resume": False,
        "aggregate_gpu_hours": 0.5,
        "wall_clock_hours": 0.4,
        "prior_verified_gpu_hours": 0.0,
        "prior_verified_wall_clock_hours": 0.0,
        "results": [{
            "job_id": job["job_id"],
            "stage": job["stage"],
            "device": job["device"],
            "status": "PASS",
            "elapsed_seconds": 0.0,
        } for job in plan["jobs"]],
    }
    _signed(summary, "summary_sha256")
    write_json_atomic(summary_path, summary)
    intervention = intervention_plan(SimpleNamespace(
        selection=selection_path,
        base_plan=plan_path,
        prior_summary=summary_path,
        geometry_analysis=geometry_path,
        run_root=args.run_root,
        tokens=args.tokens,
        registry=args.output_root / "registry.json",
        output_root=args.output_root / "I",
        analysis_root=args.output_root / "R",
        devices=args.devices,
    ))
    assert intervention["resource_caps"]["hard_gpu_hours"] == pytest.approx(1.9)
    assert intervention["resource_caps"][
        "hard_wall_clock_hours"] == pytest.approx(1.6)
    assert intervention["required_live_stage_sequence"] == ["I", "R"]
    assert intervention["stage_dependencies"] == {"I": [], "R": ["I"]}


def test_executor_runs_the_cpu_gate_after_both_device_lanes(
        tmp_path, monkeypatch):
    plan = {
        "schema_version": "pldr-row-map-execution-plan-v1",
        "plan_sha256": "f" * 64,
        "stage_order": ["Q"],
        "stage_dependencies": {"Q": []},
        "resource_caps": {
            "hard_gpu_hours": 2.4,
            "hard_wall_clock_hours": 2.0,
        },
        "jobs": [
            {"stage": "Q", "job_id": "q0", "device": "cuda:0",
             "output": str(tmp_path / "q0"), "command": "true"},
            {"stage": "Q", "job_id": "q1", "device": "cuda:1",
             "output": str(tmp_path / "q1"), "command": "true"},
            {"stage": "Q", "job_id": "gate", "device": "cpu",
             "output": str(tmp_path / "gate"), "command": "true"},
        ],
    }
    monkeypatch.setattr(executor, "_load_plan", lambda _path: plan)
    summary_path = tmp_path / "execution-summary.json"
    executor.execute(SimpleNamespace(
        plan=tmp_path / "ignored.json",
        stages="Q",
        record_root=tmp_path / "records",
        summary=summary_path,
        resume=False,
        dry_run=True,
    ))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["results"][-1]["job_id"] == "gate"
    assert {row["job_id"] for row in summary["results"][:2]} == {"q0", "q1"}
