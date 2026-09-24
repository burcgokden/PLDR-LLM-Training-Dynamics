#!/usr/bin/env python3
"""Run content-opening producers for source-resolved row-map evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.source_resolved_live import (  # noqa: E402
    bind_complete_checkpoint,
    produce_edge,
    qualify,
    write_json_atomic,
)
from confirm.source_resolved_response import produce_mask_response  # noqa: E402
from confirm.source_resolved_block_response import (  # noqa: E402
    produce_block_response,
)
from confirm.source_resolved_rank_tube import produce_rank_tube  # noqa: E402
from confirm.source_resolved_policy import seal_construction_policy  # noqa: E402
from confirm.source_resolved_timecourse import produce_timecourse  # noqa: E402
from confirm.source_resolved_intervention import (  # noqa: E402
    produce_intervention_result,
    produce_prediction,
)


DEFAULT_PROTOCOL = (
    ROOT / "experiments" / "protocols" / "source_resolved_confirmation"
)


def _shared_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-order", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-dir", type=Path, default=DEFAULT_PROTOCOL
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bind = subparsers.add_parser(
        "bind-checkpoint", help="open and bind one complete checkpoint"
    )
    _shared_checkpoint_arguments(bind)
    bind.add_argument("--output", type=Path, required=True)

    edge = subparsers.add_parser(
        "edge", help="produce one exact all-map source edge"
    )
    edge.add_argument("--source-binding", type=Path, required=True)
    edge.add_argument("--endpoint-binding", type=Path, required=True)
    edge.add_argument("--registry", type=Path, required=True)
    edge.add_argument("--tokens", type=Path, required=True)
    edge.add_argument("--data-order", type=Path, required=True)
    edge.add_argument("--device", default="cuda:0")
    edge.add_argument("--output", type=Path, required=True)

    response = subparsers.add_parser(
        "mask-response", help="produce masked AdamW response evidence"
    )
    response.add_argument("--binding", type=Path, required=True)
    response.add_argument("--registry", type=Path, required=True)
    response.add_argument("--tokens", type=Path, required=True)
    response.add_argument("--data-order", type=Path, required=True)
    response.add_argument("--device", default="cuda:0")
    response.add_argument("--map-indices", default="all")
    response.add_argument("--random-directions", type=int, default=8)
    response.add_argument("--random-seed", type=int, default=45101)
    response.add_argument("--output", type=Path, required=True)

    block = subparsers.add_parser(
        "block-response", help="run the reduced lifted 16-update operator"
    )
    block.add_argument("--bindings", nargs="+", type=Path, required=True)
    block.add_argument("--edges", nargs="+", type=Path, required=True)
    block.add_argument("--mask-response", type=Path, required=True)
    block.add_argument("--registry", type=Path, required=True)
    block.add_argument("--tokens", type=Path, required=True)
    block.add_argument("--data-order", type=Path, required=True)
    block.add_argument("--device", default="cuda:0")
    block.add_argument("--random-seed", type=int, default=45301)
    block.add_argument("--output", type=Path, required=True)

    rank = subparsers.add_parser(
        "rank-tube-block", help="produce Q3 rank, tube, and block evidence"
    )
    rank.add_argument("--source-binding", type=Path, required=True)
    rank.add_argument("--endpoint-binding", type=Path, required=True)
    rank.add_argument(
        "--mask-responses", nargs="+", type=Path, required=True
    )
    rank.add_argument("--edges", nargs="+", type=Path, required=True)
    rank.add_argument("--block-response", type=Path, required=True)
    rank.add_argument("--registry", type=Path, required=True)
    rank.add_argument("--tokens", type=Path, required=True)
    rank.add_argument("--device", default="cuda:0")
    rank.add_argument("--output", type=Path, required=True)

    timecourse = subparsers.add_parser(
        "timecourse", help="assemble the native 564-point time course"
    )
    timecourse.add_argument("--log", type=Path, required=True)
    timecourse.add_argument("--registry", type=Path, required=True)
    timecourse.add_argument("--bindings", nargs="+", type=Path, required=True)
    timecourse.add_argument("--policy", type=Path, required=True)
    timecourse.add_argument(
        "--role", choices=("construction", "heldout"), required=True
    )
    timecourse.add_argument("--selected-plga-cap", type=float)
    timecourse.add_argument("--output", type=Path, required=True)

    seal = subparsers.add_parser(
        "seal-policy", help="seal construction choices before heldout launch"
    )
    seal.add_argument("--design", type=Path, required=True)
    seal.add_argument("--timecourse", type=Path, required=True)
    seal.add_argument("--edges", nargs="+", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)

    prediction = subparsers.add_parser(
        "intervention-predict", help="freeze a prediction before native launch"
    )
    prediction.add_argument("--source-binding", type=Path, required=True)
    prediction.add_argument("--registry", type=Path, required=True)
    prediction.add_argument("--tokens", type=Path, required=True)
    prediction.add_argument("--data-order", type=Path, required=True)
    prediction.add_argument("--intervention", required=True)
    prediction.add_argument(
        "--window", choices=("pre-onset", "onset", "late"), required=True
    )
    prediction.add_argument("--device", default="cuda:0")
    prediction.add_argument("--output", type=Path, required=True)

    intervention = subparsers.add_parser(
        "intervention", help="open paired native endpoints after prediction"
    )
    intervention.add_argument("--prediction", type=Path, required=True)
    intervention.add_argument("--control-binding", type=Path, required=True)
    intervention.add_argument("--arm-binding", type=Path, required=True)
    intervention.add_argument("--registry", type=Path, required=True)
    intervention.add_argument("--tokens", type=Path, required=True)
    intervention.add_argument("--policy", type=Path, required=True)
    intervention.add_argument("--device", default="cuda:0")
    intervention.add_argument("--output", type=Path, required=True)

    qualification = subparsers.add_parser(
        "qualify", help="exercise kernels and the selected device"
    )
    qualification.add_argument("--device", default="cpu")
    qualification.add_argument("--checkpoint", type=Path)
    qualification.add_argument("--tokens", type=Path)
    qualification.add_argument("--tokenizer", type=Path)
    qualification.add_argument("--data-order", type=Path)
    qualification.add_argument("--output", type=Path, required=True)
    qualification.add_argument("--require-pass", action="store_true")

    arguments = parser.parse_args()
    protocol = arguments.protocol_dir.resolve()
    if arguments.command == "bind-checkpoint":
        result = bind_complete_checkpoint(
            arguments.checkpoint,
            arguments.tokens,
            arguments.tokenizer,
            arguments.data_order,
            protocol_directory=protocol,
        )
    elif arguments.command == "edge":
        result = produce_edge(
            arguments.source_binding,
            arguments.endpoint_binding,
            arguments.registry,
            arguments.tokens,
            arguments.data_order,
            arguments.device,
            protocol_directory=protocol,
            producer_command=sys.argv,
        )
    elif arguments.command == "mask-response":
        indices = (
            None if arguments.map_indices == "all" else
            [int(value) for value in arguments.map_indices.split(",") if value]
        )
        result = produce_mask_response(
            arguments.binding,
            arguments.registry,
            arguments.tokens,
            arguments.data_order,
            arguments.device,
            protocol_directory=protocol,
            map_indices=indices,
            random_direction_count=arguments.random_directions,
            random_seed=arguments.random_seed,
        )
    elif arguments.command == "block-response":
        result = produce_block_response(
            arguments.bindings,
            arguments.edges,
            arguments.mask_response,
            arguments.registry,
            arguments.tokens,
            arguments.data_order,
            arguments.device,
            protocol_directory=protocol,
            random_seed=arguments.random_seed,
        )
    elif arguments.command == "rank-tube-block":
        result = produce_rank_tube(
            arguments.source_binding,
            arguments.endpoint_binding,
            arguments.mask_responses,
            arguments.edges,
            arguments.block_response,
            arguments.registry,
            arguments.tokens,
            arguments.device,
            protocol_directory=protocol,
        )
    elif arguments.command == "timecourse":
        result = produce_timecourse(
            arguments.log,
            arguments.registry,
            arguments.bindings,
            arguments.policy,
            arguments.role,
            protocol_directory=protocol,
            selected_plga_cap=arguments.selected_plga_cap,
        )
    elif arguments.command == "seal-policy":
        result = seal_construction_policy(
            arguments.design,
            arguments.timecourse,
            arguments.edges,
            protocol_directory=protocol,
        )
    elif arguments.command == "intervention-predict":
        result = produce_prediction(
            arguments.source_binding,
            arguments.registry,
            arguments.tokens,
            arguments.data_order,
            arguments.intervention,
            arguments.window,
            arguments.device,
            protocol_directory=protocol,
        )
    elif arguments.command == "intervention":
        result = produce_intervention_result(
            arguments.prediction,
            arguments.control_binding,
            arguments.arm_binding,
            arguments.registry,
            arguments.tokens,
            arguments.policy,
            arguments.device,
            protocol_directory=protocol,
        )
    elif arguments.command == "qualify":
        result = qualify(
            arguments.device,
            protocol_directory=protocol,
            checkpoint_path=arguments.checkpoint,
            tokens_path=arguments.tokens,
            tokenizer_path=arguments.tokenizer,
            order_path=arguments.data_order,
        )
        if arguments.require_pass and not result["technical_valid"]:
            raise SystemExit("source-resolved qualification did not pass")
    else:
        raise AssertionError("unhandled producer command")
    write_json_atomic(arguments.output, result)
    print(json.dumps({
        "command": arguments.command,
        "output": str(arguments.output.resolve()),
        "technical_valid": result["technical_valid"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
