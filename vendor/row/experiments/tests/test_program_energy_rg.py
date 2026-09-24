from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

from independent_rg_checker import check_rg_flow  # noqa: E402
from program_energy_rg import (  # noqa: E402
    construct_rg_flow,
    digest_object,
    q0_rg_fixture,
)


def test_q0_affine_rg_flow_and_independent_replay():
    flow = construct_rg_flow(q0_rg_fixture(), block_factor=2)
    replay = check_rg_flow(flow)
    assert flow["decision"] == "QUALIFIED"
    assert replay["decision"] == "REPLAYED"
    assert [level["block_size"] for level in flow["levels"]] == [1, 2, 4]
    assert [level["block_count"] for level in flow["levels"]] == [4, 2, 1]
    assert all(flow["conditions"].values())


def test_q0_witness_is_order_sensitive_and_gauge_covariant():
    flow = construct_rg_flow(q0_rg_fixture(), block_factor=2)
    diagnostics = flow["diagnostics"]
    assert (
        diagnostics["first_pair_order_sensitivity"]
        > diagnostics["semigroup_tolerance"]
    )
    assert (
        diagnostics["semigroup_residual"]
        <= diagnostics["semigroup_tolerance"]
    )
    assert (
        diagnostics["gauge_covariance_residual"]
        <= diagnostics["gauge_covariance_tolerance"]
    )


def test_rg_constructor_rejects_broken_owned_chain():
    edges = q0_rg_fixture()
    edges[1]["source_gauge"][0] *= 2.0
    unsigned = dict(edges[1])
    unsigned.pop("record_sha256")
    edges[1]["record_sha256"] = digest_object(unsigned)
    with pytest.raises(ValueError, match="gauge chain"):
        construct_rg_flow(edges, block_factor=2)


def test_rg_constructor_requires_complete_aligned_power():
    with pytest.raises(ValueError, match="exact power"):
        construct_rg_flow(q0_rg_fixture()[:3], block_factor=2)


def test_independent_replay_rejects_rehashed_block_tamper():
    flow = construct_rg_flow(q0_rg_fixture(), block_factor=2)
    tampered = copy.deepcopy(flow)
    tampered["levels"][1]["blocks"][0]["operator_upper"][0][0] += 1e-3
    unsigned = dict(tampered)
    unsigned.pop("record_sha256")
    tampered["record_sha256"] = digest_object(unsigned)
    with pytest.raises(ValueError, match="ordered_affine_blocks"):
        check_rg_flow(tampered)


def test_independent_replay_rejects_rehashed_tolerance_inflation():
    flow = construct_rg_flow(q0_rg_fixture(), block_factor=2)
    tampered = copy.deepcopy(flow)
    for level in tampered["levels"]:
        for block in level["blocks"]:
            block["gauge_covariance_tolerance"] = 1.0
    tampered["diagnostics"]["gauge_covariance_tolerance"] = 1.0
    tampered["diagnostics"]["semigroup_tolerance"] = 1.0
    unsigned = dict(tampered)
    unsigned.pop("record_sha256")
    tampered["record_sha256"] = digest_object(unsigned)
    with pytest.raises(ValueError, match="ordered_affine_blocks"):
        check_rg_flow(tampered)
