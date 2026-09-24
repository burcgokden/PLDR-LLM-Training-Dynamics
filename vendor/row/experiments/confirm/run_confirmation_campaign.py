#!/usr/bin/env python3
"""Resumable, dependency-gated executor for the frozen E0-E8 campaign."""

from __future__ import annotations

import argparse
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS / "analysis"))

from assemble_confirmation import assemble  # noqa: E402
import campaign_design  # noqa: E402
import campaign_record  # noqa: E402
from estimator_manifest import load_manifest  # noqa: E402
import launch_confirmation  # noqa: E402
import measure_direct_certificate as direct_measure  # noqa: E402
import measure_intervention as intervention_measure  # noqa: E402
import measure_optimizer_transport as transport_measure  # noqa: E402
import measure_row_map as row_measure  # noqa: E402
import measure_scale_transfer as scale_measure  # noqa: E402
from measurement_graph import build_measurement_graph  # noqa: E402


DTYPE = "float32_training_float64_certificate"


def _strict_json(path):
    return campaign_record.strict_load(path)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = campaign_record.strict_dumps(value)
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"cached JSON disagrees with recomputation: {path}")
        return path
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measurement_values(value):
    rows = value.get("measurements")
    if not isinstance(rows, list):
        raise ValueError("artifact has no measurement array")
    result = {row["name"]: float(row["value"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("artifact contains duplicate measurements")
    return result


def _run_log_rows(run_directory):
    with (Path(run_directory) / "log.jsonl").open(
        "r", encoding="utf-8",
    ) as stream:
        return [json.loads(line) for line in stream]


def _safe_slug(value):
    result = "".join(
        character if character.isalnum() or character in "-." else "-"
        for character in str(value)
    )
    if not result:
        raise ValueError("empty artifact slug")
    return result


def _protocols_through(through):
    if through not in campaign_record.PROTOCOLS:
        raise ValueError("through must be E0 through E8")
    return campaign_record.PROTOCOLS[
        :campaign_record.PROTOCOLS.index(through) + 1]


class CampaignRunner:
    def __init__(self, campaign_directory, devices):
        self.campaign = Path(campaign_directory).resolve()
        self.metadata = launch_confirmation._campaign_metadata(self.campaign)
        self.devices = tuple(devices)
        if not self.devices or len(self.devices) > 2:
            raise ValueError("one or two devices must be supplied")
        self.design = _strict_json(
            self.campaign / "frozen/protocols/campaign_design.json")
        if self.design != campaign_design.design_object():
            raise RuntimeError("frozen and executing campaign designs disagree")
        self.binding_root = Path(self.metadata["binding_root"]).resolve()
        self.source_root = Path(self.metadata["source_root"]).resolve()
        if self.source_root != ROOT.resolve():
            raise RuntimeError("campaign source root is not this repository")
        self.work = self.campaign / "work"
        self.runs = self.work / "runs"
        self.cache = self.work / "cache"
        self.failed = self.work / "failed"
        for path in (self.runs, self.cache, self.failed):
            path.mkdir(parents=True, exist_ok=True)
        frozen = self.campaign / "frozen/protocols"
        self.manifest = load_manifest(
            frozen / "estimator_manifest.json", ROOT)
        self.tokens = ROOT / self.design["data"]["tokens"]
        self.data_order = ROOT / self.design["data"]["tokens_manifest"]
        self.tokenizer = ROOT / self.design["data"]["tokenizer"]
        self.source_binding = ROOT / "MANIFEST.sha256"
        self._verify_resources()
        self._verify_inputs()

    def _layers(self, architecture_cell=None):
        if architecture_cell is None:
            architecture_cell = self.design["baseline"]["architecture_cell"]
        architecture = self.design["architectures"].get(architecture_cell)
        if not isinstance(architecture, dict):
            raise ValueError("unknown architecture cell")
        return tuple(range(int(architecture["model_depth"])))


    def _verify_resources(self):
        free = shutil.disk_usage("/pldr-work").free
        required = int(self.design["resources"][
            "minimum_free_workspace_bytes"])
        if free < required:
            raise RuntimeError(
                f"free workspace {free} is below frozen minimum {required}")
        if any(device.startswith("cuda") for device in self.devices):
            command = [
                "nvidia-smi",
                "--query-gpu=index,power.limit",
                "--format=csv,noheader,nounits",
            ]
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True)
            realized = {}
            for line in completed.stdout.splitlines():
                index, power = [item.strip() for item in line.split(",")]
                realized[int(index)] = float(power)
            limit = float(self.design["resources"][
                "maximum_power_watts_per_gpu"])
            for device in self.devices:
                index = int(device.split(":", 1)[1])
                if index not in realized or realized[index] > limit + 1e-9:
                    raise RuntimeError(
                        f"GPU {index} power cap is {realized.get(index)}, "
                        f"above {limit} W")

    def _verify_inputs(self):
        rows = (
            (self.tokens, self.design["data"]["tokens_sha256"]),
            (self.data_order,
             self.design["data"]["tokens_manifest_sha256"]),
            (self.tokenizer, self.design["data"]["tokenizer_sha256"]),
        )
        cache_path = self.work / "input-verification.json"
        cached = _strict_json(cache_path) if cache_path.is_file() else None
        realized = {}
        for path, expected in rows:
            stat = path.stat()
            key = str(path.resolve())
            fingerprint = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": expected,
            }
            if not (
                isinstance(cached, dict)
                and cached.get("files", {}).get(key) == fingerprint
            ):
                digest = _sha256(path)
                if digest != expected:
                    raise RuntimeError(
                        f"frozen input digest mismatch for {path}")
            realized[key] = fingerprint
        payload = {
            "schema_version": "pldr-campaign-input-verification-v1",
            "design_sha256": self.design["design_sha256"],
            "files": realized,
        }
        _write_json(cache_path, payload)
        if not self.source_binding.is_file():
            raise RuntimeError("source MANIFEST.sha256 is missing")

    def _batched(self, items, function):
        items = list(items)
        for start in range(0, len(items), len(self.devices)):
            batch = items[start:start + len(self.devices)]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = [
                    executor.submit(function, item, self.devices[index])
                    for index, item in enumerate(batch)
                ]
                for future in futures:
                    future.result()

    def _training_complete(self, run_directory, end_step):
        final = Path(run_directory) / "ckpt_final.pt"
        log = Path(run_directory) / "log.jsonl"
        if not final.is_file() or not log.is_file():
            return False
        payload = torch.load(final, map_location="cpu", weights_only=True)
        if int(payload.get("step", -1)) != int(end_step):
            return False
        rows = _run_log_rows(run_directory)
        if not rows or rows[-1].get("event") != "final":
            return False
        final_step = rows[-1].get("step")
        if final_step is not None:
            return int(final_step) == int(end_step)

        # Compatibility for the first two rev26 confirmation runs, whose
        # generation-complete record predated the terminal-step field.  Do
        # not infer completion from the checkpoint alone: the final event
        # must follow a loss-bearing training row at the same exact step.
        terminal_training_row = next(
            (row for row in reversed(rows[:-1]) if "loss" in row), None)
        return bool(
            terminal_training_row is not None
            and int(terminal_training_row.get("step", -1)) == int(end_step)
        )

    def _archive_incomplete(self, run_directory):
        run_directory = Path(run_directory)
        if not run_directory.exists():
            return
        destination = self.failed / (
            run_directory.name + f"-{time.time_ns()}")
        run_directory.rename(destination)
        print(f"archived incomplete run at {destination}", flush=True)

    def _common_train_command(self, *, name, outdir, seed, device,
                              architecture, schedule, steps,
                              checkpoints, optimizer_checkpoints,
                              snapshot_plan, init_from=None,
                              multiplier=1.0, terminal_validation=False,
                              skip_generation=False):
        common = self.design["common_training"]
        command = [
            sys.executable, "train_run.py",
            "--name", name,
            "--lr", str(schedule["maximum_learning_rate"]),
            "--learning_rate_multiplier", str(multiplier),
            "--warmup", str(schedule["warmup_steps"]),
            "--steps", str(steps),
            "--schedule_total_steps", str(schedule["total_steps"]),
            "--batch", str(common["batch"]),
            "--ctx", str(common["context"]),
            "--layers", str(architecture["layers"]),
            "--heads", str(architecture["heads"]),
            "--dk", str(architecture["dk"]),
            "--adff", str(architecture["adff"]),
            "--seed", str(seed),
            "--device", device,
            "--tokens", str(self.tokens),
            "--tok_model", str(self.tokenizer),
            "--outdir", str(outdir),
            "--probe_every", str(common["probe_every"]),
            "--sharp_every", str(common["sharp_every"]),
            "--sharp_pre_every", str(common["sharp_pre_every"]),
            "--sharp_block_every", str(common["sharp_block_every"]),
            "--sharp_full", str(common["sharp_full"]),
            "--ckpt_steps", ",".join(map(str, checkpoints)),
            "--optimizer_ckpt_steps",
            ",".join(map(str, optimizer_checkpoints)),
            "--confirmation_snapshot_plan", snapshot_plan,
            "--val_batches", "8",
            "--optimizer", common["optimizer"],
            "--wd", str(common["weight_decay"]),
            "--clip", str(common["gradient_value_clip"]),
            "--loss_mode", common["loss_mode"],
            "--accum", str(common["gradient_accumulation"]),
            "--probe_region", self.design["data"]["probe_region"],
            "--anneal_floor", str(schedule["anneal_floor"]),
            "--hold_until", str(schedule["hold_until"]),
            "--gen_prompts", str(common["generation_prompts"]),
            "--gen_cont", str(common["generation_continuations"]),
            "--fd_hs", ",".join(
                map(str, common["finite_difference_steps"])),
        ]
        if init_from is not None:
            command.extend(["--init_from", str(init_from)])
        if terminal_validation:
            command.append("--terminal_validation")
        if skip_generation:
            command.append("--skip_generation")
        return command

    def _run_training(self, *, run_directory, command, end_step):
        if self._training_complete(run_directory, end_step):
            print(f"training cache hit {run_directory}", flush=True)
            return
        self._archive_incomplete(run_directory)
        self._verify_resources()
        log_directory = self.work / "training-stdout"
        log_directory.mkdir(parents=True, exist_ok=True)
        output_path = log_directory / (Path(run_directory).name + ".log")
        print("launching " + " ".join(command), flush=True)
        with output_path.open("w", encoding="utf-8") as output:
            completed = subprocess.run(
                command, cwd=EXPERIMENTS, stdout=output,
                stderr=subprocess.STDOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"training failed ({completed.returncode}); see {output_path}")
        if not self._training_complete(run_directory, end_step):
            raise RuntimeError(f"training did not seal final state: {run_directory}")

    def _baseline_run(self, seed):
        return self.runs / "baseline" / f"baseline-s{seed:03d}"

    def _train_baselines(self, seeds):
        architecture = self.design["architectures"][
            self.design["baseline"]["architecture_cell"]]
        schedule = self.design["baseline"]["schedule"]
        plan = ";".join(
            f"{step}:{'+'.join(groups)}"
            for step, groups in self.design["baseline"][
                "certificate_snapshot_plan"].items()
        )

        def train(seed, device):
            name = f"baseline-s{seed:03d}"
            outdir = self.runs / "baseline"
            command = self._common_train_command(
                name=name, outdir=outdir, seed=seed, device=device,
                architecture=architecture, schedule=schedule,
                steps=schedule["total_steps"],
                checkpoints=self.design["baseline"][
                    "model_checkpoint_steps"],
                optimizer_checkpoints=self.design["baseline"][
                    "optimizer_checkpoint_steps"],
                snapshot_plan=plan,
            )
            self._run_training(
                run_directory=outdir / name, command=command,
                end_step=schedule["total_steps"])

        self._batched(seeds, train)

    def _checkpoint(self, run_directory, step, *, final=False):
        if final:
            path = Path(run_directory) / "ckpt_final.pt"
        else:
            path = Path(run_directory) / f"ckpt_{int(step)}.pt"
        if not path.is_file():
            raise RuntimeError(f"checkpoint is missing: {path}")
        return path

    def _snapshot(self, run_directory, step):
        path = Path(run_directory) / (
            f"confirmation_transport_{int(step):09d}.pt")
        if not path.is_file():
            raise RuntimeError(f"transport snapshot is missing: {path}")
        return path

    def _artifact_directory(self, protocol, run_id):
        path = self.campaign / "raw" / protocol / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _cache_directory(self, kind, run_key, suffix):
        path = self.cache / kind / _safe_slug(run_key) / _safe_slug(suffix)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _link(self, source, destination):
        source = Path(source)
        destination = Path(destination)
        if destination.exists():
            if os.stat(source).st_ino != os.stat(destination).st_ino:
                raise RuntimeError(f"artifact hardlink disagrees: {destination}")
            return destination
        os.link(source, destination)
        return destination

    def _tensor_groups_checkpoint(self, checkpoint):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        names = [
            name for name in payload["model"]
            if ".reslayerAs." in name or ".plgatt_layer." in name
        ]
        if not names:
            raise RuntimeError("checkpoint contains no deductive parameters")
        return [{"name": "deductive", "parameters": sorted(names)}]

    def _tensor_groups_snapshot(self, snapshot, group):
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
        rows = payload.get("blocks", {}).get(group, [])
        names = [row["name"] for row in rows]
        if not names:
            raise RuntimeError(f"snapshot contains no {group} tensor group")
        return [{"name": group, "parameters": names}]

    def _record(self, *, protocol, run_id, seed, step, measurement,
                measurement_path, model, tensor_groups, unit_key, device,
                extra_artifacts=(), segment_index=0, segment_step=None,
                segment_offset=0, dose=None, intervention_description=None):
        phase = self.campaign / "raw" / protocol
        record_path = phase / f"record-{run_id}-{int(step):09d}.json"
        if record_path.is_file():
            campaign_record.load_and_verify(
                record_path, binding_root=self.binding_root,
                artifact_root=phase)
            return record_path
        unit_directory = Path(measurement_path).parent
        graph, bindings = build_measurement_graph(
            protocol, measurement, self.manifest, root=ROOT)
        graph_path = _write_json(
            unit_directory / "certificate-graph.json", graph)
        assembled = assemble(protocol, graph, bindings, self.manifest)
        assembled_path = _write_json(
            unit_directory / "assembled.json", assembled)
        tensor_path = _write_json(
            unit_directory / "tensor-groups.json", tensor_groups)
        arguments = Namespace(
            campaign_dir=str(self.campaign), protocol=protocol,
            run_id=run_id, seed=int(seed), global_step=int(step),
            segment_index=int(segment_index),
            segment_step=int(step if segment_step is None else segment_step),
            segment_offset=int(segment_offset), checkpoint_step=int(step),
            source_binding=str(self.source_binding), model=str(model),
            data_order=str(self.data_order), tokenizer=str(self.tokenizer),
            tensor_groups=str(tensor_path), assembled=str(assembled_path),
            certificate_graph=str(graph_path),
            layer_bundle=unit_key.get("layer_bundle"),
            tensor_group=unit_key.get("tensor_group"),
            certificate_box=unit_key.get("certificate_box"),
            schedule_cell=unit_key.get("schedule_cell"),
            architecture_cell=unit_key.get("architecture_cell"),
            block_size=unit_key.get("block_size"),
            artifact=[
                str(measurement_path), str(tensor_path),
                *[str(path) for path in extra_artifacts],
            ],
            arm=unit_key.get("arm", "observational"), dose=dose,
            post_optimizer=False,
            intervention_description=(
                intervention_description or "no intervention"),
            device=device, dtype=DTYPE,
        )
        return launch_confirmation.create_record(arguments)

    def _e0(self, *, run_key, run_directory, checkpoint_step, device,
            successor_step=None, final_checkpoint=False,
            registered_seed=None):
        registered = (
            registered_seed is not None
            and 1 <= registered_seed <= 8
            and checkpoint_step in range(1000, 13000, 1000)
            and successor_step == checkpoint_step + 1
        )
        if registered:
            run_id = f"e0-s{registered_seed:03d}-t{checkpoint_step:06d}"
            directory = self._artifact_directory("E0", run_id)
        else:
            mode = "static" if successor_step == checkpoint_step else "joint"
            directory = self._cache_directory(
                "e0", run_key, f"t{checkpoint_step:06d}-{mode}")
            run_id = None
        measurement_path = directory / "measurement.json"
        rows_path = directory / "rows.npz"
        if measurement_path.is_file():
            if not rows_path.is_file():
                raise RuntimeError(f"E0 row artifact is missing: {rows_path}")
            measurement = _strict_json(measurement_path)
        else:
            checkpoint = self._checkpoint(
                run_directory, checkpoint_step, final=final_checkpoint)
            after_step = (
                checkpoint_step if successor_step is None else successor_step)
            successor = self._checkpoint(
                run_directory, after_step,
                final=final_checkpoint and after_step == checkpoint_step)
            cover = self.design["e0_cover"]
            measurement = row_measure.measure_checkpoint(
                run_directory=run_directory,
                checkpoint=checkpoint,
                successor_checkpoint=successor,
                token_path=self.tokens,
                device=device,
                batch_size=cover["probe_batch_size"],
                probe_offset=cover["probe_offset"],
                rows_per_layer=cover["center_rows_per_layer"],
                heldout_rows_per_layer=cover["heldout_rows_per_layer"],
                coordinate_padding=cover["coordinate_padding"],
                subdivisions_per_axis=cover["subdivisions_per_axis"],
                subdivisions_per_segment=cover[
                    "subdivisions_per_segment"],
                cover_strategy=cover["cover_strategy"],
                maximum_boxes=cover["maximum_boxes"],
                resource_action_limit=cover["resource_action_limit"],
                resource_storage_bytes_limit=cover[
                    "resource_storage_bytes_limit"],
                output_rows=rows_path,
                expected_segment_updates=(
                    0 if after_step == checkpoint_step else 1),
                physical_criterion=self.design["certificate_rules"][
                    "entry"]["criterion"],
                reserved_tail_budget=0.0,
            )
            _write_json(measurement_path, measurement)
        if registered:
            checkpoint = self._checkpoint(run_directory, checkpoint_step)
            self._record(
                protocol="E0", run_id=run_id, seed=registered_seed,
                step=checkpoint_step, measurement=measurement,
                measurement_path=measurement_path, model=checkpoint,
                tensor_groups=self._tensor_groups_checkpoint(checkpoint),
                unit_key={
                    "layer_bundle": "all_criterion_layers",
                    "arm": "observational",
                },
                device=device, extra_artifacts=(rows_path,),
            )
        return measurement, rows_path, measurement_path

    def _baseline_e0(self, seed, step, device, *, need_rows=True):
        run = self._baseline_run(seed)
        registered_path = self.campaign / "raw/E0" / (
            f"e0-s{seed:03d}-t{step:06d}") / "measurement.json"
        if seed <= 8 and step in range(1000, 13000, 1000):
            if registered_path.is_file():
                rows = registered_path.parent / "rows.npz"
                return _strict_json(registered_path), rows, registered_path
        successor = step + 1 if (
            step in self.design["baseline"]["model_checkpoint_steps"]
            and step + 1 in self.design["baseline"]["model_checkpoint_steps"]
        ) else step
        return self._e0(
            run_key=f"baseline-s{seed:03d}", run_directory=run,
            checkpoint_step=step, successor_step=successor,
            device=device, registered_seed=None)

    def _e1(self, *, run_key, run_directory, step, group, device,
            e0_rows, registered_seed=None, final_checkpoint=False):
        registered = (
            registered_seed is not None and registered_seed <= 8
            and step in (4000, 8000)
        )
        if registered:
            run_id = (
                f"e1-s{registered_seed:03d}-t{step:06d}-{group}")
            directory = self._artifact_directory("E1", run_id)
        else:
            run_id = None
            directory = self._cache_directory(
                "e1", run_key, f"t{step:06d}-{group}")
        measurement_path = directory / "measurement.json"
        certificate_path = directory / "lifted-certificate.json"
        snapshot = self._snapshot(run_directory, step)
        checkpoint = self._checkpoint(
            run_directory, step, final=final_checkpoint)
        if measurement_path.is_file() and certificate_path.is_file():
            measurement = _strict_json(measurement_path)
        else:
            measurement = transport_measure.measure_live_snapshot(
                snapshot_path=snapshot,
                model_checkpoint=checkpoint,
                run_directory=run_directory,
                row_artifact=e0_rows,
                tensor_group=group,
                device=device,
                certificate_output=certificate_path,
            )
            _write_json(measurement_path, measurement)
        if registered:
            transport_copy = self._link(snapshot, directory / "transport.pt")
            self._record(
                protocol="E1", run_id=run_id, seed=registered_seed,
                step=step, measurement=measurement,
                measurement_path=measurement_path, model=checkpoint,
                tensor_groups=self._tensor_groups_snapshot(snapshot, group),
                unit_key={"tensor_group": group, "arm": "observational"},
                device=device,
                extra_artifacts=(certificate_path, transport_copy),
            )
        return measurement, _strict_json(certificate_path), measurement_path

    def _e2_bundle(self, *, seed, run_key, run_directory, step,
                   layer, device, registered=False,
                   final_checkpoint=False):
        if registered:
            run_id = f"e2-s{seed:03d}-t{step:06d}-l{layer:02d}"
            directory = self._artifact_directory("E2", run_id)
        else:
            run_id = None
            directory = self._cache_directory(
                "e2", run_key, f"t{step:06d}-l{layer:02d}")
        measurement_path = directory / "measurement.json"
        e0, rows, _ = (
            self._baseline_e0(seed, step, device)
            if run_key.startswith("baseline-")
            else self._e0(
                run_key=run_key, run_directory=run_directory,
                checkpoint_step=step, successor_step=step, device=device,
                final_checkpoint=final_checkpoint)
        )
        e1, certificate, _ = self._e1(
            run_key=run_key, run_directory=run_directory, step=step,
            group="deductive", device=device, e0_rows=rows,
            final_checkpoint=final_checkpoint)
        if measurement_path.is_file():
            measurement = _strict_json(measurement_path)
        else:
            measurement = direct_measure.measure_live_e2(
                lifted_certificate=certificate,
                optimizer_measurement=e1,
                e0_measurement=e0,
                layer_index=layer,
            )
            _write_json(measurement_path, measurement)
        if registered:
            checkpoint = self._checkpoint(run_directory, step)
            self._record(
                protocol="E2", run_id=run_id, seed=seed, step=step,
                measurement=measurement, measurement_path=measurement_path,
                model=checkpoint,
                tensor_groups=self._tensor_groups_checkpoint(checkpoint),
                unit_key={
                    "layer_bundle": f"L{layer:02d}",
                    "arm": "observational",
                }, device=device,
            )
        return {
            "measurement": measurement,
            "optimizer": e1,
            "certificate": certificate,
            "e0": e0,
        }

    def _e3(self, *, seed, run_key, run_directory, layer,
            certificate_box, device, registered=False):
        if registered:
            run_id = (
                f"e3-s{seed:03d}-l{layer:02d}-{certificate_box}")
            directory = self._artifact_directory("E3", run_id)
        else:
            run_id = None
            directory = self._cache_directory(
                "e3", run_key, f"l{layer:02d}-{certificate_box}")
        measurement_path = directory / "measurement.json"
        family_steps = (
            self.design["scalar_diagnostic_rules"]["auxiliary_family"][
                "nominal_steps"]
            if certificate_box == "nominal"
            else self.design["scalar_diagnostic_rules"]["auxiliary_family"][
                "outer_steps"]
        )
        bundles = [
            self._e2_bundle(
                seed=seed, run_key=run_key, run_directory=run_directory,
                step=step, layer=layer, device=device,
                registered=(
                    registered and step in (6000, 10000)),
            )
            for step in family_steps
        ]
        if measurement_path.is_file():
            measurement = _strict_json(measurement_path)
        else:
            measurement = direct_measure.measure_live_e3(
                nominal_e2=bundles[0]["measurement"],
                family_e2=[row["measurement"] for row in bundles],
                certificate_box=certificate_box,
            )
            _write_json(measurement_path, measurement)
        if registered:
            checkpoint = self._checkpoint(run_directory, 10000)
            self._record(
                protocol="E3", run_id=run_id, seed=seed, step=10000,
                measurement=measurement, measurement_path=measurement_path,
                model=checkpoint,
                tensor_groups=self._tensor_groups_checkpoint(checkpoint),
                unit_key={
                    "layer_bundle": f"L{layer:02d}",
                    "certificate_box": certificate_box,
                    "arm": "observational",
                }, device=device,
            )
        return measurement

    def _e4(self, *, seed, run_key, run_directory, layer,
            selected_step, device, registered=False):
        if registered:
            run_id = f"e4-s{seed:03d}-t{selected_step:06d}-l{layer:02d}"
            directory = self._artifact_directory("E4", run_id)
        else:
            run_id = None
            directory = self._cache_directory(
                "e4", run_key, f"t{selected_step:06d}-l{layer:02d}")
        measurement_path = directory / "measurement.json"
        steps = self.design["scalar_diagnostic_rules"]["defect_envelope"]["steps"]
        bundles = [
            self._e2_bundle(
                seed=seed, run_key=run_key, run_directory=run_directory,
                step=step, layer=layer, device=device)
            for step in steps
        ]
        e3 = self._e3(
            seed=seed, run_key=run_key, run_directory=run_directory,
            layer=layer, certificate_box="outer", device=device,
            registered=False)
        if measurement_path.is_file():
            measurement = _strict_json(measurement_path)
        else:
            measurement = direct_measure.measure_live_e4(
                lifted_certificates=[row["certificate"] for row in bundles],
                optimizer_measurements=[row["optimizer"] for row in bundles],
                e2_measurements=[row["measurement"] for row in bundles],
                e3_measurement=e3,
                layer_index=layer,
                selected_step=selected_step,
            )
            _write_json(measurement_path, measurement)
        if registered:
            checkpoint = self._checkpoint(run_directory, selected_step)
            self._record(
                protocol="E4", run_id=run_id, seed=seed,
                step=selected_step, measurement=measurement,
                measurement_path=measurement_path, model=checkpoint,
                tensor_groups=self._tensor_groups_checkpoint(checkpoint),
                unit_key={
                    "layer_bundle": f"L{layer:02d}",
                    "arm": "observational",
                }, device=device,
            )
        return measurement

    def _e5(self, *, seed, layer, certificate_box, device,
            registered=False):
        run_key = f"baseline-s{seed:03d}"
        run = self._baseline_run(seed)
        if registered:
            run_id = f"e5-s{seed:03d}-l{layer:02d}-{certificate_box}"
            directory = self._artifact_directory("E5", run_id)
        else:
            run_id = None
            directory = self._cache_directory(
                "e5", run_key, f"l{layer:02d}-{certificate_box}")
        measurement_path = directory / "measurement.json"
        steps = [
            *self.design["scalar_diagnostic_rules"]["self_map"]["fit_steps"],
            self.design["scalar_diagnostic_rules"]["self_map"]["heldout_step"],
        ]
        bundles = [
            self._e2_bundle(
                seed=seed, run_key=run_key, run_directory=run,
                step=step, layer=layer, device=device)
            for step in steps
        ]
        entry = self._e3(
            seed=seed, run_key=run_key, run_directory=run,
            layer=layer, certificate_box="nominal", device=device,
            registered=False)
        successor = self._e3(
            seed=seed, run_key=run_key, run_directory=run,
            layer=layer, certificate_box="outer", device=device,
            registered=False)
        e4 = self._e4(
            seed=seed, run_key=run_key, run_directory=run,
            layer=layer, selected_step=10009, device=device,
            registered=False)
        if measurement_path.is_file():
            measurement = _strict_json(measurement_path)
        else:
            measurement = direct_measure.measure_live_e5(
                lifted_certificates=[row["certificate"] for row in bundles],
                optimizer_measurements=[row["optimizer"] for row in bundles],
                e2_measurements=[row["measurement"] for row in bundles],
                entry_e3_measurement=entry,
                successor_e3_measurement=successor,
                e4_measurement=e4,
                layer_index=layer,
                certificate_box=certificate_box,
            )
            _write_json(measurement_path, measurement)
        if registered:
            checkpoint = self._checkpoint(run, 10010)
            self._record(
                protocol="E5", run_id=run_id, seed=seed, step=10010,
                measurement=measurement, measurement_path=measurement_path,
                model=checkpoint,
                tensor_groups=self._tensor_groups_checkpoint(checkpoint),
                unit_key={
                    "layer_bundle": f"L{layer:02d}",
                    "certificate_box": certificate_box,
                    "arm": "observational",
                }, device=device,
            )
        return measurement

    def _e6(self, *, seed, run_key, run_directory, certificate_time,
            e0_trajectory, e3_by_layer, e4_by_layer, device,
            registered=False):
        if registered:
            run_id = f"e6-s{seed:03d}-t{certificate_time:06d}"
            directory = self._artifact_directory("E6", run_id)
        else:
            run_id = None
            directory = self._cache_directory(
                "e6", run_key, f"t{certificate_time:06d}")
        measurement_path = directory / "measurement.json"
        bundles = [
            self._e2_bundle(
                seed=seed, run_key=run_key, run_directory=run_directory,
                step=certificate_time, layer=layer, device=device)
            for layer in self._layers()
        ]
        if measurement_path.is_file():
            measurement = _strict_json(measurement_path)
        else:
            measurement = direct_measure.measure_live_e6(
                lifted_certificate=bundles[0]["certificate"],
                optimizer_measurement=bundles[0]["optimizer"],
                e2_measurements=[row["measurement"] for row in bundles],
                e3_measurements=e3_by_layer,
                e4_measurements=e4_by_layer,
                e0_trajectory=e0_trajectory,
                certificate_time=certificate_time,
            )
            _write_json(measurement_path, measurement)
        if registered:
            checkpoint = self._checkpoint(run_directory, certificate_time)
            self._record(
                protocol="E6", run_id=run_id, seed=seed,
                step=certificate_time, measurement=measurement,
                measurement_path=measurement_path, model=checkpoint,
                tensor_groups=self._tensor_groups_checkpoint(checkpoint),
                unit_key={"arm": "observational"}, device=device,
            )
        return measurement

    def phase_e0(self):
        self._train_baselines(range(1, 11))

        def measure(item, device):
            seed, step = item
            self._e0(
                run_key=f"baseline-s{seed:03d}",
                run_directory=self._baseline_run(seed),
                checkpoint_step=step, successor_step=step + 1,
                device=device, registered_seed=seed)

        self._batched(itertools.product(
            range(1, 9), range(1000, 13000, 1000)), measure)

    def phase_e1(self):
        def measure(item, device):
            seed, step = item
            _, rows, _ = self._baseline_e0(seed, step, device)
            for group in self.design["baseline"]["transport_tensor_groups"]:
                self._e1(
                    run_key=f"baseline-s{seed:03d}",
                    run_directory=self._baseline_run(seed), step=step,
                    group=group, device=device, e0_rows=rows,
                    registered_seed=seed)

        self._batched(itertools.product(range(1, 9), (4000, 8000)), measure)

    def phase_e2(self):
        def measure(item, device):
            seed, step = item
            for layer in self._layers():
                self._e2_bundle(
                    seed=seed, run_key=f"baseline-s{seed:03d}",
                    run_directory=self._baseline_run(seed), step=step,
                    layer=layer, device=device, registered=True)

        self._batched(itertools.product(range(1, 11), (6000, 10000)), measure)

    def phase_e3(self):
        def measure(seed, device):
            for layer in self._layers():
                for box in ("nominal", "outer"):
                    self._e3(
                        seed=seed, run_key=f"baseline-s{seed:03d}",
                        run_directory=self._baseline_run(seed), layer=layer,
                        certificate_box=box, device=device, registered=True)

        self._batched(range(1, 9), measure)

    def phase_e4(self):
        def measure(seed, device):
            for layer in self._layers():
                for step in (10001, 10009):
                    self._e4(
                        seed=seed, run_key=f"baseline-s{seed:03d}",
                        run_directory=self._baseline_run(seed), layer=layer,
                        selected_step=step, device=device, registered=True)

        self._batched(range(1, 11), measure)

    def phase_e5(self):
        def measure(seed, device):
            for layer in self._layers():
                for box in ("entry", "successor"):
                    self._e5(
                        seed=seed, layer=layer, certificate_box=box,
                        device=device, registered=True)

        self._batched(range(1, 9), measure)

    def phase_e6(self):
        def measure(seed, device):
            run_key = f"baseline-s{seed:03d}"
            run = self._baseline_run(seed)
            e3 = [
                self._e3(
                    seed=seed, run_key=run_key, run_directory=run,
                    layer=layer, certificate_box="outer", device=device)
                for layer in self._layers()
            ]
            e4 = [
                self._e4(
                    seed=seed, run_key=run_key, run_directory=run,
                    layer=layer, selected_step=10009, device=device)
                for layer in self._layers()
            ]
            for certificate_time in range(8000, 16000, 1000):
                trajectory = [
                    self._baseline_e0(seed, step, device)[0]
                    for step in range(certificate_time, 16001, 1000)
                ]
                self._e6(
                    seed=seed, run_key=run_key, run_directory=run,
                    certificate_time=certificate_time,
                    e0_trajectory=trajectory,
                    e3_by_layer=e3, e4_by_layer=e4,
                    device=device, registered=True)

        self._batched(range(1, 9), measure)

    def _branch_run(self, seed, source, arm):
        return self.runs / "e7" / (
            f"e7-s{seed:03d}-t{source:06d}-{arm}")

    def _train_e7_branches(self):
        self._train_baselines(range(11, 25))
        architecture = self.design["architectures"][
            self.design["baseline"]["architecture_cell"]]
        schedule = self.design["baseline"]["schedule"]
        jobs = list(itertools.product(
            range(1, 25),
            self.design["e7_interventions"]["source_checkpoints"],
            self.design["e7_interventions"]["arms"],
        ))

        def train(item, device):
            seed, source, arm = item
            outcome = source + self.design["e7_interventions"][
                "continuation_steps"]
            window_steps = [
                source + offset
                for offset in self.design["e7_interventions"][
                    "certificate_window_offsets"]
            ]
            if window_steps[-1] != outcome or len(window_steps) < 2:
                raise ValueError("E7 certificate window does not end at outcome")
            run = self._branch_run(seed, source, arm)
            command = self._common_train_command(
                name=run.name, outdir=run.parent, seed=seed, device=device,
                architecture=architecture, schedule=schedule,
                steps=self.design["e7_interventions"]["continuation_steps"],
                checkpoints=window_steps, optimizer_checkpoints=[],
                snapshot_plan=";".join(
                    f"{step}:deductive" for step in window_steps),
                init_from=self._checkpoint(self._baseline_run(seed), source),
                multiplier=self.design["e7_interventions"]["arms"][arm][
                    "learning_rate_multiplier"],
                terminal_validation=True, skip_generation=True,
            )
            self._run_training(
                run_directory=run, command=command, end_step=outcome)

        self._batched(jobs, train)

    def _branch_e6(self, *, certificate, optimizer, e2, e3, e4,
                   e0, certificate_time):
        layer_steps = [
            direct_measure._layer_lifted_step(
                certificate, optimizer, e2_row, layer)
            for e2_row, layer in zip(e2, self._layers())
        ]
        metrics = [
            np.asarray(row["construction"]["metric"], dtype=float)
            for row in e3
        ]
        family_data = [
            direct_measure._matrix_family_data(row) for row in e3
        ]
        q = max(value[2] for value in family_data)
        window_lengths = {value[3] for value in family_data}
        if len(window_lengths) != 1:
            raise ValueError("E7 branch layers use different path windows")
        window_length = window_lengths.pop()
        window_gain = max(value[4] for value in family_data)
        metric_lower = min(
            float(np.linalg.eigvalsh(metric)[0]) for metric in metrics)
        initial = max(
            direct_measure._metric_norm(row["state_before"], metric)
            for row, metric in zip(layer_steps, metrics)
        )
        persistent = max(
            float(row["closed_input"]["persistent_constant"])
            for row in e4)
        transient = max(
            float(row["closed_input"]["geometric_constant"])
            for row in e4)
        rate = max(
            float(row["closed_input"]["geometric_rate"])
            for row in e4)
        block_envelope = direct_measure._constant_edge_block_envelope(
            q=q,
            persistent=persistent,
            transient=transient,
            rate=rate,
            window_length=window_length,
            window_gain=window_gain,
        )
        horizon = int(self.design["e7_interventions"]["continuation_steps"])
        direct = _measurement_values(e0)["direct_row_map_upper"]
        criterion = float(
            self.design["certificate_rules"]["entry"]["criterion"])
        sustained = float(direct < criterion)
        input_value = {
            "certificate_time": int(certificate_time),
            "q": q,
            "r": rate,
            "initial_h_norm": initial,
            "forcing_constant": transient,
            "persistent_forcing_constant": persistent,
            "path_window_length": window_length,
            "path_window_gain": window_gain,
            "block_forcing_constant": block_envelope["forcing_constant"],
            "block_persistent_forcing_constant": block_envelope[
                "persistent_forcing_constant"],
            "block_rate": block_envelope["rate"],
            "metric_lower": metric_lower,
            "cover_floor": direct,
            "cover_transient_constant": 0.0,
            "cover_rate": self.design["scalar_diagnostic_rules"][
                "finite_entry"]["geometric_cover_rate"],
            "criterion": criterion,
            "observed_entry_step": certificate_time,
            "direct_row_map_upper": direct,
            "sustained_entry_indicator": sustained,
            "schedule_gains": [q] * horizon,
            "schedule_disturbance_bounds": [
                math.nextafter(
                    persistent + transient * rate ** index, math.inf)
                for index in range(horizon)
            ],
        }
        result = direct_measure.measure_e6(input_value)
        result.update({
            "schema_version": "pldr-live-e7-branch-horizon-v2",
            "time_semantics": "FINITE_IMPLEMENTED_SCHEDULE",
            "registered_branch_horizon_updates": horizon,
            "entry_right_censored_at_terminal": not bool(sustained),
            "infinite_continuation_claimed": False,
            "block_envelope": block_envelope,
            "closed_input": input_value,
        })
        return result

    def phase_e7(self):
        self._train_e7_branches()
        jobs = list(itertools.product(
            range(1, 25),
            self.design["e7_interventions"]["source_checkpoints"],
            self.design["e7_interventions"]["arms"],
        ))

        def measure(item, device):
            seed, source, arm = item
            outcome = source + self.design["e7_interventions"][
                "continuation_steps"]
            window_steps = [
                source + offset
                for offset in self.design["e7_interventions"][
                    "certificate_window_offsets"]
            ]
            run_key = f"e7-s{seed:03d}-t{source:06d}-{arm}"
            run = self._branch_run(seed, source, arm)
            e0, rows, _ = self._e0(
                run_key=run_key, run_directory=run,
                checkpoint_step=outcome, successor_step=outcome,
                final_checkpoint=True, device=device)
            e1, certificate, _ = self._e1(
                run_key=run_key, run_directory=run, step=outcome,
                group="deductive", device=device, e0_rows=rows,
                final_checkpoint=True)
            e2 = []
            e3 = []
            e4 = []
            for layer in self._layers():
                bundles = [
                    self._e2_bundle(
                        seed=seed, run_key=run_key, run_directory=run,
                        step=step, layer=layer, device=device,
                        final_checkpoint=(step == outcome))
                    for step in window_steps
                ]
                bundle = bundles[-1]
                e2.append(bundle["measurement"])
                family = direct_measure.measure_live_e3(
                    nominal_e2=bundle["measurement"],
                    family_e2=[row["measurement"] for row in bundles],
                    certificate_box="branch_outcome_registered_window")
                e3.append(family)
                e4.append(direct_measure.measure_live_e4(
                    lifted_certificates=[row["certificate"] for row in bundles],
                    optimizer_measurements=[row["optimizer"] for row in bundles],
                    e2_measurements=[row["measurement"] for row in bundles],
                    e3_measurement=family,
                    layer_index=layer,
                    selected_step=outcome,
                ))
            e6 = self._branch_e6(
                certificate=certificate, optimizer=e1, e2=e2, e3=e3,
                e4=e4, e0=e0, certificate_time=outcome)
            multiplier = self.design["e7_interventions"]["arms"][arm][
                "learning_rate_multiplier"]
            measurement = intervention_measure.measure_live_e7(
                e0_measurement=e0, e2_measurements=e2,
                e3_measurements=e3, e4_measurements=e4,
                e6_measurement=e6,
                source_run_directory=self._baseline_run(seed),
                branch_run_directory=run,
                source_step=source, outcome_step=outcome,
                intervention_dose=multiplier, arm=arm,
            )
            run_id = run_key
            directory = self._artifact_directory("E7", run_id)
            measurement_path = _write_json(
                directory / "measurement.json", measurement)
            checkpoint = self._checkpoint(run, outcome, final=True)
            snapshot = self._snapshot(run, outcome)
            transport_copy = self._link(snapshot, directory / "transport.pt")
            self._record(
                protocol="E7", run_id=run_id, seed=seed, step=outcome,
                measurement=measurement, measurement_path=measurement_path,
                model=checkpoint,
                tensor_groups=self._tensor_groups_snapshot(
                    snapshot, "deductive"),
                unit_key={
                    "arm": arm,
                    "schedule_cell": "paired_branch",
                },
                device=device, extra_artifacts=(transport_copy,),
                segment_index=1,
                segment_step=self.design["e7_interventions"][
                    "continuation_steps"],
                segment_offset=source, dose=multiplier,
                intervention_description=self.design[
                    "e7_interventions"]["arms"][arm]["description"],
            )

        self._batched(jobs, measure)

    def _e8_run(self, seed, architecture, schedule, partition):
        return self.runs / "e8" / (
            f"e8-{partition}-s{seed:03d}-{architecture}-{schedule}")

    def _train_e8_partition(self, seeds, partition):
        jobs = list(itertools.product(
            seeds,
            self.design["e8"]["split"]["architecture_cells"],
            self.design["e8"]["split"]["schedule_cells"],
        ))
        snapshots = self.design["e8"]["certificate_snapshot_steps"]
        snapshot_plan = ";".join(
            f"{step}:deductive" for step in snapshots)

        def train(item, device):
            seed, architecture_name, schedule_name = item
            run = self._e8_run(
                seed, architecture_name, schedule_name, partition)
            command = self._common_train_command(
                name=run.name, outdir=run.parent, seed=seed, device=device,
                architecture=self.design["architectures"][architecture_name],
                schedule=self.design["schedules"][schedule_name],
                steps=self.design["e8"]["checkpoint"],
                checkpoints=self.design["e8"][
                    "certificate_checkpoint_steps"],
                optimizer_checkpoints=[], snapshot_plan=snapshot_plan,
            )
            self._run_training(
                run_directory=run, command=command,
                end_step=self.design["e8"]["checkpoint"])

        self._batched(jobs, train)

    def _e8_physical(self, *, seed, architecture, schedule,
                     partition, device):
        run_key = f"e8-{partition}-s{seed:03d}-{architecture}-{schedule}"
        run = self._e8_run(seed, architecture, schedule, partition)
        cache_directory = self._cache_directory(
            "e8-physical", run_key, "certificate")
        e0_by_step = {
            step: self._e0(
                run_key=run_key, run_directory=run,
                checkpoint_step=step, successor_step=step, device=device)[0]
            for step in self.design["e8"]["certificate_checkpoint_steps"]
        }
        e2_by_step = {}
        for step in self.design["e8"]["certificate_snapshot_steps"]:
            e2_by_step[step] = [
                self._e2_bundle(
                    seed=seed, run_key=run_key, run_directory=run,
                    step=step, layer=layer, device=device)["measurement"]
                for layer in self._layers(architecture)
            ]
        nominal_step = self.design["e8"]["certificate_time"]
        e3 = []
        e4 = []
        for layer_index, layer in enumerate(self._layers(architecture)):
            family_rows = [
                e2_by_step[step][layer_index]
                for step in self.design["e8"]["certificate_snapshot_steps"]
            ]
            family = direct_measure.measure_live_e3(
                nominal_e2=e2_by_step[nominal_step][layer_index],
                family_e2=family_rows,
                certificate_box="e8_cell_trajectory")
            e3.append(family)
            bundles = [
                self._e2_bundle(
                    seed=seed, run_key=run_key, run_directory=run,
                    step=step, layer=layer, device=device)
                for step in self.design["e8"]["certificate_snapshot_steps"]
            ]
            e4.append(direct_measure.measure_live_e4(
                lifted_certificates=[row["certificate"] for row in bundles],
                optimizer_measurements=[row["optimizer"] for row in bundles],
                e2_measurements=[row["measurement"] for row in bundles],
                e3_measurement=family, layer_index=layer,
                selected_step=max(
                    self.design["e8"]["certificate_snapshot_steps"]),
            ))
        e6_path = cache_directory / "e6.json"
        if e6_path.is_file():
            e6 = _strict_json(e6_path)
        else:
            bundles = [
                self._e2_bundle(
                    seed=seed, run_key=run_key, run_directory=run,
                    step=nominal_step, layer=layer, device=device)
                for layer in self._layers(architecture)
            ]
            e6 = direct_measure.measure_live_e6(
                lifted_certificate=bundles[0]["certificate"],
                optimizer_measurement=bundles[0]["optimizer"],
                e2_measurements=[row["measurement"] for row in bundles],
                e3_measurements=e3, e4_measurements=e4,
                e0_trajectory=[
                    e0_by_step[step] for step in self.design["e8"][
                        "certificate_checkpoint_steps"]
                ],
                certificate_time=nominal_step,
            )
            _write_json(e6_path, e6)
        terminal_e0 = e0_by_step[self.design["e8"]["checkpoint"]]
        return terminal_e0, e6

    def phase_e8(self):
        training_seeds = self.design["e8"]["split"]["training_seeds"]
        heldout_seeds = self.design["e8"]["split"]["heldout_seeds"]
        self._train_e8_partition(training_seeds, "fit")
        training_rows = []
        for index, item in enumerate(itertools.product(
            training_seeds,
            self.design["e8"]["split"]["architecture_cells"],
            self.design["e8"]["split"]["schedule_cells"],
        )):
            seed, architecture, schedule = item
            device = self.devices[index % len(self.devices)]
            e0, e6 = self._e8_physical(
                seed=seed, architecture=architecture, schedule=schedule,
                partition="fit", device=device)
            training_rows.append(scale_measure.physical_summary(
                run_directory=self._e8_run(
                    seed, architecture, schedule, "fit"),
                e0_measurement=e0, e6_measurement=e6,
                architecture_cell=architecture, schedule_cell=schedule,
            ))
        fit = scale_measure.fit_baseline_models(training_rows)
        fit_path = _write_json(self.work / "e8-baseline-fit.json", fit)
        self._train_e8_partition(heldout_seeds, "heldout")
        jobs = list(itertools.product(
            heldout_seeds,
            self.design["e8"]["split"]["architecture_cells"],
            self.design["e8"]["split"]["schedule_cells"],
        ))

        def measure(item, device):
            seed, architecture, schedule = item
            run = self._e8_run(
                seed, architecture, schedule, "heldout")
            run_key = (
                f"e8-heldout-s{seed:03d}-{architecture}-{schedule}")
            e0, e6 = self._e8_physical(
                seed=seed, architecture=architecture, schedule=schedule,
                partition="heldout", device=device)
            checkpoint = self._checkpoint(
                run, self.design["e8"]["checkpoint"])
            for block_size in self.design["e8"]["block_sizes"]:
                tensor_directory = self._cache_directory(
                    "e8-tensors", run_key, f"b{block_size}")
                tensor_path = tensor_directory / "tensor.json"
                if tensor_path.is_file():
                    tensor = _strict_json(tensor_path)
                else:
                    tensor = scale_measure.collect_live_tensors(
                        run_directory=run, checkpoint=checkpoint,
                        token_path=self.tokens, device=device,
                        prompt_offset=self.design["e8"]["prompt_offset"],
                        block_size=block_size,
                        direct_row_map_upper=_measurement_values(e0)[
                            "direct_row_map_upper"],
                    )
                    _write_json(tensor_path, tensor)
                for arm in self.design["e8"]["model_arms"]:
                    measurement = scale_measure.measure_live_e8(
                        run_directory=run, checkpoint=checkpoint,
                        token_path=self.tokens, device=device,
                        e0_measurement=e0, e6_measurement=e6, fit=fit_path,
                        architecture_cell=architecture,
                        schedule_cell=schedule, block_size=block_size,
                        arm=arm, tensor_measurement=tensor,
                    )
                    run_id = (
                        f"e8-s{seed:03d}-{architecture}-{schedule}-"
                        f"b{block_size}-{arm}")
                    directory = self._artifact_directory("E8", run_id)
                    measurement_path = _write_json(
                        directory / "measurement.json", measurement)
                    self._record(
                        protocol="E8", run_id=run_id, seed=seed,
                        step=self.design["e8"]["checkpoint"],
                        measurement=measurement,
                        measurement_path=measurement_path, model=checkpoint,
                        tensor_groups=self._tensor_groups_checkpoint(checkpoint),
                        unit_key={
                            "arm": arm,
                            "architecture_cell": architecture,
                            "schedule_cell": schedule,
                            "block_size": block_size,
                        }, device=device,
                    )

        self._batched(jobs, measure)

    def execute(self, protocol):
        expected = os.environ.get("PLDR_CONFIRMATION_PROTOCOL")
        if expected != protocol:
            raise RuntimeError(
                "campaign phase must execute through the registered launcher")
        function = getattr(self, f"phase_{protocol.lower()}")
        function()

    def drive(self, through="E8"):
        for protocol in _protocols_through(through):
            analysis_path = self.campaign / "derived/analysis.json"
            if analysis_path.is_file():
                status = _strict_json(analysis_path)["experiments"][
                    protocol]["status"]
                if status in {"CONFIRMED", "NOT_CONFIRMED", "INFEASIBLE"}:
                    print(f"{protocol}: existing terminal status {status}",
                          flush=True)
                    continue
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--campaign-dir", str(self.campaign),
                "--devices", ",".join(self.devices),
                "--execute", protocol,
            ]
            try:
                returncode = launch_confirmation.launch(
                    self.campaign, protocol, command)
            except RuntimeError as error:
                print(f"{protocol}: BLOCKED: {error}", flush=True)
                continue
            if returncode != 0:
                raise RuntimeError(f"{protocol} execution failed")
            result = launch_confirmation.analyze_campaign(self.campaign)
            status = result["experiments"][protocol]["status"]
            print(f"{protocol}: {status}", flush=True)
        if through != campaign_record.PROTOCOLS[-1]:
            print(
                f"CAMPAIGN DRIVE: stopped after requested boundary {through}",
                flush=True,
            )
            return
        validation = launch_confirmation.validate_campaign(self.campaign)
        print(campaign_record.strict_dumps(validation), end="")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument(
        "--through", choices=campaign_record.PROTOCOLS,
        default=campaign_record.PROTOCOLS[-1],
        help=(
            "for --drive, stop after this protocol instead of entering "
            "later phases"
        ),
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--execute", choices=campaign_record.PROTOCOLS)
    actions.add_argument("--drive", action="store_true")
    arguments = parser.parse_args()
    devices = tuple(
        value for value in arguments.devices.split(",") if value)
    runner = CampaignRunner(arguments.campaign_dir, devices)
    if arguments.execute:
        runner.execute(arguments.execute)
    else:
        runner.drive(arguments.through)


if __name__ == "__main__":
    main()
