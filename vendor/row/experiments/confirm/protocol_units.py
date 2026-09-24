"""Deterministic exact unit grids for E0 through E8."""

from __future__ import annotations

import itertools

import campaign_design
from campaign_unit_key import UNIT_KEY_NAMES


def unit_key(
    *, seed, checkpoint, arm="observational", layer_bundle=None,
    tensor_group=None, certificate_box=None, schedule_cell=None,
    architecture_cell=None, block_size=None,
):
    value = {
        "seed": int(seed),
        "checkpoint": int(checkpoint),
        "layer_bundle": layer_bundle,
        "tensor_group": tensor_group,
        "certificate_box": certificate_box,
        "schedule_cell": schedule_cell,
        "architecture_cell": architecture_cell,
        "block_size": block_size,
        "arm": arm,
    }
    if set(value) != set(UNIT_KEY_NAMES):
        raise AssertionError("unit-key constructor is stale")
    return value


def expected_unit_keys(protocol_id):
    keys = []
    if protocol_id == "E0":
        for seed, checkpoint in itertools.product(
            range(1, 9), range(1000, 13000, 1000)
        ):
            keys.append(unit_key(
                seed=seed, checkpoint=checkpoint,
                layer_bundle="all_criterion_layers",
            ))
    elif protocol_id == "E1":
        for seed, group, checkpoint in itertools.product(
            range(1, 9),
            ("query", "key", "value", "deductive"),
            (4000, 8000),
        ):
            keys.append(unit_key(
                seed=seed, checkpoint=checkpoint, tensor_group=group,
            ))
    elif protocol_id == "E2":
        for seed, layer, checkpoint in itertools.product(
            range(1, 11), ("L00", "L03", "L07", "L11"),
            (6000, 10000),
        ):
            keys.append(unit_key(
                seed=seed, checkpoint=checkpoint, layer_bundle=layer,
            ))
    elif protocol_id == "E3":
        for seed, layer, box in itertools.product(
            range(1, 9), ("L00", "L03", "L07", "L11"),
            ("nominal", "outer"),
        ):
            keys.append(unit_key(
                seed=seed, checkpoint=10000, layer_bundle=layer,
                certificate_box=box,
            ))
    elif protocol_id == "E4":
        for seed, layer, checkpoint in itertools.product(
            range(1, 11), ("L00", "L03", "L07", "L11"),
            (10001, 10009),
        ):
            keys.append(unit_key(
                seed=seed, checkpoint=checkpoint, layer_bundle=layer,
            ))
    elif protocol_id == "E5":
        for seed, layer, box in itertools.product(
            range(1, 9), ("L00", "L03", "L07", "L11"),
            ("entry", "successor"),
        ):
            keys.append(unit_key(
                seed=seed, checkpoint=10010, layer_bundle=layer,
                certificate_box=box,
            ))
    elif protocol_id == "E6":
        for seed, checkpoint in itertools.product(
            range(1, 9), range(8000, 16000, 1000)
        ):
            keys.append(unit_key(seed=seed, checkpoint=checkpoint))
    elif protocol_id == "E7":
        for seed, checkpoint, arm in itertools.product(
            range(1, 25),
            tuple(campaign_design.E7_INTERVENTIONS[
                "outcome_checkpoints"
            ]),
            ("contraction_improving", "defect_increasing", "sham"),
        ):
            keys.append(unit_key(
                seed=seed, checkpoint=checkpoint, arm=arm,
                schedule_cell="paired_branch",
            ))
    elif protocol_id == "E8":
        for seed, architecture, schedule, block_size, arm in itertools.product(
            range(101, 109),
            ("w064_d04", "w128_d08", "w256_d12"),
            ("short_warmup", "long_warmup", "low_floor"),
            (2, 4, 8),
            ("theory", "constant_rate", "loss_only", "unconstrained_trend"),
        ):
            keys.append(unit_key(
                seed=seed, checkpoint=16000, arm=arm,
                architecture_cell=architecture, schedule_cell=schedule,
                block_size=block_size,
            ))
    else:
        raise ValueError("protocol must be E0 through E8")
    canonical = {
        tuple(value[name] for name in UNIT_KEY_NAMES) for value in keys
    }
    if len(canonical) != len(keys):
        raise AssertionError(f"duplicate registered unit key in {protocol_id}")
    return keys


EXPECTED_BASE_UNITS = {
    "E0": 96,
    "E1": 64,
    "E2": 80,
    "E3": 64,
    "E4": 80,
    "E5": 64,
    "E6": 64,
    "E7": 72,
    "E8": 72,
}


def validate_all_grids():
    arms = {"E7": 3, "E8": 4}
    blocks = {"E8": 3}
    for protocol_id, base_count in EXPECTED_BASE_UNITS.items():
        expected = base_count * arms.get(protocol_id, 1) * blocks.get(
            protocol_id, 1
        )
        realized = len(expected_unit_keys(protocol_id))
        if realized != expected:
            raise AssertionError(
                f"{protocol_id} grid has {realized} keys, expected {expected}"
            )
    return True
