#!/usr/bin/env python3
"""Execute selected stages of a row-map plan with two device lanes.

Archival execution interface: no validated interruption-safe cumulative campaign
budget guarantee. See the combined docs/RESOURCE_EXECUTION.md and
provenance/campaign-accounting.json for the inspected accounting boundary.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from gate_shape_evidence import digest_object, write_json_atomic  # noqa: E402


def _digest_path(path):
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        digest.update(target.read_bytes())
        return digest.hexdigest()
    if not target.is_dir():
        raise ValueError("completed job output does not exist")
    files = sorted(value for value in target.rglob("*") if value.is_file())
    if not files:
        raise ValueError("completed job output directory is empty")
    for value in files:
        relative = value.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(value.read_bytes())
    return digest.hexdigest()


def _load_plan(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("schema_version") not in {
        "pldr-row-map-execution-plan-v1",
        "pldr-row-map-intervention-plan-v1",
    }:
        raise ValueError("unknown row-map execution plan schema")
    unsigned = dict(plan)
    recorded = unsigned.pop("plan_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("execution plan digest does not replay")
    return plan


def _record_path(record_root, job):
    safe = job["job_id"].replace("/", "-")
    return record_root / job["stage"] / f"{safe}.json"


def _load_pass_record(path, job):
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    unsigned = dict(record)
    recorded = unsigned.pop("record_sha256", None)
    if recorded != digest_object(unsigned):
        return False
    if record.get("status") != "PASS" or record.get("job_id") != job["job_id"]:
        return False
    output = Path(job["output"])
    return output.exists() and record.get("output_sha256") == _digest_path(output)


def _execute_job(job, record_root, *, resume, dry_run):
    record_path = _record_path(record_root, job)
    if resume and _load_pass_record(record_path, job):
        return {
            "job_id": job["job_id"], "stage": job["stage"],
            "device": job["device"], "status": "SKIPPED_VERIFIED",
            "elapsed_seconds": 0.0,
        }
    output = Path(job["output"])
    if output.exists() and not dry_run:
        raise FileExistsError(
            f"job output already exists without a verified record: {output}")
    if dry_run:
        return {
            "job_id": job["job_id"], "stage": job["stage"],
            "device": job["device"], "status": "DRY_RUN",
            "elapsed_seconds": 0.0, "command": job["command"],
        }
    started = time.monotonic()
    process = subprocess.run(
        shlex.split(job["command"]), cwd=ROOT, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")
    elapsed = time.monotonic() - started
    status = "PASS" if process.returncode == 0 and output.exists() else "FAIL"
    record = {
        "schema_version": "pldr-row-map-job-record-v1",
        "job_id": job["job_id"],
        "stage": job["stage"],
        "device": job["device"],
        "command": job["command"],
        "status": status,
        "returncode": int(process.returncode),
        "elapsed_seconds": elapsed,
        "output": str(output),
        "output_sha256": _digest_path(output) if output.exists() else None,
        "combined_log": process.stdout,
    }
    record["record_sha256"] = digest_object(record)
    write_json_atomic(record_path, record)
    if status != "PASS":
        raise RuntimeError(
            f"{job['job_id']} failed with return code {process.returncode}")
    return record


def _verified_resource_usage(plan, record_root):
    gpu_seconds = 0.0
    wall_seconds = 0.0
    by_stage = {}
    for job in plan["jobs"]:
        path = _record_path(record_root, job)
        if not _load_pass_record(path, job):
            continue
        with path.open("r", encoding="utf-8") as stream:
            record = json.load(stream)
        elapsed = float(record["elapsed_seconds"])
        if elapsed < 0.0:
            raise ValueError("verified job has negative elapsed time")
        stage = by_stage.setdefault(job["stage"], {"cpu": 0.0, "lanes": {}})
        if job["device"] == "cpu":
            stage["cpu"] += elapsed
        else:
            gpu_seconds += elapsed
            stage["lanes"].setdefault(job["device"], 0.0)
            stage["lanes"][job["device"]] += elapsed
    stage_order = plan.get("stage_order", [])
    remaining = sorted(set(by_stage) - set(stage_order))
    for stage_name in [*stage_order, *remaining]:
        if stage_name not in by_stage:
            continue
        stage = by_stage[stage_name]
        wall_seconds += stage["cpu"] + max(stage["lanes"].values(), default=0.0)
    return gpu_seconds, wall_seconds


def _stage_verified(plan, stage, record_root):
    jobs = [job for job in plan["jobs"] if job["stage"] == stage]
    return bool(jobs) and all(
        _load_pass_record(_record_path(record_root, job), job)
        for job in jobs)


def execute(arguments):
    plan = _load_plan(arguments.plan)
    selected = [value for value in arguments.stages.split(",") if value]
    if not selected:
        raise ValueError("at least one stage must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError("a stage cannot be selected twice")
    known = {job["stage"] for job in plan["jobs"]}
    unknown = set(selected) - known
    if unknown:
        raise ValueError("unknown stages: " + ", ".join(sorted(unknown)))
    required_live = plan.get("required_live_stage_sequence")
    if not arguments.dry_run and required_live and selected != required_live:
        raise ValueError(
            "live execution must use the complete ordered stage sequence: "
            + ",".join(required_live))

    record_root = Path(arguments.record_root).resolve()
    start = time.monotonic()
    prior_gpu_seconds, prior_wall_seconds = _verified_resource_usage(
        plan, record_root)
    new_gpu_seconds = 0.0
    results = []
    completed_stages = set()
    dependencies = plan.get("stage_dependencies", {})
    caps = plan.get("resource_caps", {})
    hard_gpu = float(caps.get("hard_gpu_hours", 2.4))
    hard_wall = float(caps.get("hard_wall_clock_hours", 2.0))

    def enforce_caps():
        gpu_hours = (prior_gpu_seconds + new_gpu_seconds) / 3600.0
        wall_hours = (
            prior_wall_seconds + time.monotonic() - start) / 3600.0
        if gpu_hours > hard_gpu:
            raise RuntimeError("aggregate GPU-hour hard stop reached")
        if wall_hours > hard_wall:
            raise RuntimeError("wall-clock hard stop reached")

    for stage_name in selected:
        for dependency in dependencies.get(stage_name, []):
            if (
                dependency not in completed_stages
                and not _stage_verified(plan, dependency, record_root)
            ):
                raise RuntimeError(
                    f"stage {stage_name} requires verified stage {dependency}")
        stage_jobs = [
            job for job in plan["jobs"] if job["stage"] == stage_name]
        index = 0
        while index < len(stage_jobs):
            if stage_jobs[index]["device"] == "cpu":
                results.append(_execute_job(
                    stage_jobs[index], record_root, resume=arguments.resume,
                    dry_run=arguments.dry_run))
                index += 1
                enforce_caps()
                continue

            stop = index
            while (
                stop < len(stage_jobs)
                and stage_jobs[stop]["device"] != "cpu"
            ):
                stop += 1
            phase_jobs = stage_jobs[index:stop]
            lanes = {}
            for job in phase_jobs:
                lanes.setdefault(job["device"], []).append(job)

            def run_lane(jobs):
                return [
                    _execute_job(
                        job, record_root, resume=arguments.resume,
                        dry_run=arguments.dry_run)
                    for job in jobs
                ]

            with ThreadPoolExecutor(max_workers=max(len(lanes), 1)) as executor:
                futures = [
                    executor.submit(run_lane, jobs)
                    for jobs in lanes.values()
                ]
                for future in as_completed(futures):
                    lane_results = future.result()
                    results.extend(lane_results)
                    new_gpu_seconds += sum(
                        row["elapsed_seconds"] for row in lane_results
                        if row["status"] not in {
                            "DRY_RUN", "SKIPPED_VERIFIED",
                        })
            index = stop
            enforce_caps()
        completed_stages.add(stage_name)

    summary = {
        "schema_version": "pldr-row-map-execution-summary-v1",
        "plan_sha256": plan["plan_sha256"],
        "selected_stages": selected,
        "dry_run": bool(arguments.dry_run),
        "resume": bool(arguments.resume),
        "aggregate_gpu_hours": (
            prior_gpu_seconds + new_gpu_seconds) / 3600.0,
        "wall_clock_hours": (
            prior_wall_seconds + time.monotonic() - start) / 3600.0,
        "prior_verified_gpu_hours": prior_gpu_seconds / 3600.0,
        "prior_verified_wall_clock_hours": prior_wall_seconds / 3600.0,
        "results": [{
            key: value for key, value in row.items()
            if key in {
                "job_id", "stage", "device", "status", "elapsed_seconds",
            }
        } for row in results],
    }
    summary["summary_sha256"] = digest_object(summary)
    write_json_atomic(arguments.summary, summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stages", required=True)
    parser.add_argument("--record-root", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
