#!/usr/bin/env python3
"""Two-step CPU mini-training through the real model/training path:
the one-command smoke test. No GPU and no dataset download; a tiny
random token file is synthesized in a temp directory. Exit code 0 on
success."""

import os
import subprocess
import sys
import tempfile

import numpy as np

EXP = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments")


def main():
    d = tempfile.mkdtemp(prefix="pldr-smoke-")
    tok = os.path.join(d, "toks.npy")
    rng = np.random.default_rng(0)
    rng.integers(1, 32000, size=5320 * 32, dtype=np.uint16).tofile(tok)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.call(
        [sys.executable, "train_run.py",
         "--name", "smoke", "--lr", "1e-3", "--warmup", "1",
         "--steps", "3", "--batch", "2", "--ctx", "32", "--layers", "1",
         "--heads", "2", "--dk", "8", "--adff", "12", "--device", "cpu",
         "--tokens", tok, "--outdir", os.path.join(d, "runs"),
         "--ckpt_steps", "", "--val_batches", "1",
         "--tok_model", "data/tokenizer.model",
         "--chronological_history_length", "2",
         "--confirmation_snapshot_steps", "2",
         "--confirmation_snapshot_blocks", "phi"],
        cwd=EXP, env=env)
    if result != 0:
        return result
    snapshot = os.path.join(
        d, "runs", "smoke", "confirmation_transport_000000002.pt")
    if not os.path.isfile(snapshot):
        print("smoke: confirmation transport snapshot is missing")
        return 1
    import torch
    payload = torch.load(snapshot, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "pldr-adamw-transport-v2":
        print("smoke: confirmation transport snapshot schema is stale")
        return 1
    if "learning_rate_prepared_next" not in payload:
        print("smoke: next scheduled learning rate is missing")
        return 1
    sys.path.insert(0, EXP)
    from confirm.confirmation_artifacts import load_complete_checkpoint
    complete_path = os.path.join(
        d, "runs", "smoke", "ckpt_final.pt")
    complete = load_complete_checkpoint(complete_path)
    if complete["step"] != 3 or complete["data_state"]["cursor_end"] != 6:
        print("smoke: complete checkpoint clock or cursor is stale")
        return 1
    from confirm.chronological_history import SCHEMA_VERSION, sidecar_path
    history_path = sidecar_path(complete_path)
    if not history_path.is_file():
        print("smoke: chronological gradient-history sidecar is missing")
        return 1
    history = torch.load(history_path, map_location="cpu", weights_only=False)
    if (
        history.get("schema_version") != SCHEMA_VERSION
        or history.get("checkpoint_step") != 3
        or history.get("history_length") != 2
        or not history.get("gates")
        or not all(
            gate.get("ready_for_next_update")
            for gate in history["gates"].values()
        )
    ):
        print("smoke: chronological gradient-history sidecar is invalid")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
