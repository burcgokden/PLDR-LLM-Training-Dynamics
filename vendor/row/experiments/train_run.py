"""Single-GPU instrumented pretraining run for the row-map-collapse study.

Model, loss, optimizer, LR schedule, gradient clipping and data packing match
the reference stack of the SOC study (PLDR-LLM-Self-Organized-Criticality,
pldr_model_v510), with these disclosed deviations: single GPU (no FSDP),
float32 throughout (no AMP), and a smaller model/context configured below.
Control-run flags (--optimizer, --clip, --wd, --loss_mode, --sdpa,
--freeze_g_at, --freeze_plga_at, --freeze_row_program_at, --freeze_attn_at, --freeze_ffn_at,
--freeze_v_at with --freeze_v_mode, --pulse_at/--pulse_len/--pulse_lr,
--accum, --mode_block/--mode_scale/--mode_from) intentionally deviate
from the reference stack and are recorded in the run config.

Wave-6 instrumentation (all off by default; enabled per run):
--sharp_block_every activates the per-fine-block raw and preconditioned
Lanczos probes (plga / phi / attn / ffn), the tracked signed modes with
per-step projection series, and the per-block clipping saturation;
--accum runs exact-gradient accumulation for matched-token batch
controls; --gen_prompts/--gen_cont widen the generation order parameter
with a prompt-level bootstrap interval; --fd_hs sets the
finite-difference step grid (float64 loss reduction).

Data protocol: training consumes chunks sequentially from --data_offset, or
through a frozen absolute chunk-index permutation supplied by --data_order.
The cursor is persisted as `data_offset_end`, so continuations resume the
stream instead of replaying it.  Probe/validation/prompt data come from a
globally reserved region at the tail of the token stream, excluded from
training in every phase (--probe_region global, the default).  The legacy
layout of waves 1-3 (probes placed right after this run's own training
range) remains available via --probe_region legacy for exact reproduction.

Along training we record: per-step loss and lr; every `probe_every` steps
the cross-input order-parameter proxies and row-map diagnostics; every
`sharp_every` steps raw-metric Lanczos extremal curvature (full and
block-restricted), the update-direction Rayleigh quotients, and the
collapse-coordinate directional curvature; every `sharp_pre_every` steps
the true Adam-preconditioned extremal curvature
lambda_{max,min}(D^{-1/2} H D^{-1/2}) with residuals, multiple starts, and
replicate batches.  At the end: checkpoints, stochastic-generation order
parameter in the style of the SOC paper, and sample continuations.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import resource
import random
import re
import sys
import time

import numpy as np

# torchao (imported through the model stack) calls the deprecated
# torch.utils._pytree.register_constant on an Enum subclass under
# torch 2.12, emitting a library-level deprecation log line the run
# does not control; silence that one logger (library issue, not a
# property of this training path)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pldr_model_v510 import PLDR_Model  # noqa: E402
import instrument  # noqa: E402
import optimizer_ledger  # noqa: E402
from confirm.confirmation_artifacts import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION,
    TRAIN_UPDATE_ORDER,
    build_model_only_state_manifest,
    build_state_manifest,
    capture_rng_states,
    digest_object,
    empty_measurement_registry,
    load_json_object,
    restore_rng_states,
    sha256_path,
    validate_complete_checkpoint,
    validate_measurement_registry,
    validate_model_only_checkpoint,
)
from confirm.chronological_history import (  # noqa: E402
    empty_history as empty_chronological_history,
    final_gate_parameters as chronological_gate_parameters,
    record_clipped_gate_gradients,
    restore_history as restore_chronological_history,
    save_sidecar as save_chronological_sidecar,
)
from confirm.finite_increment_specs import (  # noqa: E402
    REGISTRY_SCHEMA as FINITE_INCREMENT_REGISTRY_SCHEMA,
    TIME_COURSE_ANCHORS as FINITE_INCREMENT_TIME_COURSE_ANCHORS,
)
from confirm.finite_increment_timecourse import (  # noqa: E402
    record as finite_increment_timecourse_record,
    registered_token_rows as finite_increment_registered_rows,
)
from confirm.source_resolved_observer import (  # noqa: E402
    record_timepoint as source_resolved_timepoint_record,
)
from confirm.source_resolved_specs import (  # noqa: E402
    CAMPAIGN_ID as SOURCE_RESOLVED_CAMPAIGN_ID,
    snapshot_updates as source_resolved_snapshot_updates,
)
from confirm.orbitwise_specs import (  # noqa: E402
    CAMPAIGN_ID as ORBITWISE_CAMPAIGN_ID,
    DENSE_TIMEPOINT_SCHEMA as ORBITWISE_DENSE_TIMEPOINT_SCHEMA,
    FULL_TIMEPOINT_SCHEMA as ORBITWISE_FULL_TIMEPOINT_SCHEMA,
    REGISTRY as ORBITWISE_REGISTRY,
    dense_snapshot_updates as orbitwise_dense_snapshot_updates,
    full_snapshot_updates as orbitwise_full_snapshot_updates,
)
from confirm.mixed_collapse_specs import (  # noqa: E402
    CAMPAIGN_ID as MIXED_COLLAPSE_CAMPAIGN_ID,
    DENSE_TIMEPOINT_SCHEMA as MIXED_COLLAPSE_DENSE_TIMEPOINT_SCHEMA,
    FULL_TIMEPOINT_SCHEMA as MIXED_COLLAPSE_FULL_TIMEPOINT_SCHEMA,
    REGISTRY as MIXED_COLLAPSE_REGISTRY,
    campaign_design as mixed_collapse_campaign_design,
    dense_snapshot_updates as mixed_collapse_dense_snapshot_updates,
    full_snapshot_updates as mixed_collapse_full_snapshot_updates,
)


# --- reference loss / schedule (copied from pldr_run_model_v510.py) ---------

def masked_loss_function(y_true, y_pred):
    mask = torch.ne(y_true, 0)
    y_pred = torch.permute(y_pred, (0, 2, 1))
    loss_ = nn.CrossEntropyLoss(reduction="none")(y_pred, y_true)
    mask = mask.to(loss_.dtype)
    loss_ = loss_ * mask
    return torch.sum(loss_) / torch.sum(mask)


def masked_loss_function64(y_true, y_pred):
    """The masked CE with the log-softmax and reduction carried out in
    float64 on the float32 logits: the finite-difference curvature probe
    consumes this variant so its second differences are not quantized at
    the float32 ULP scale of the loss reduction."""
    mask = torch.ne(y_true, 0)
    y_pred = torch.permute(y_pred, (0, 2, 1)).double()
    loss_ = nn.CrossEntropyLoss(reduction="none")(y_pred, y_true)
    mask = mask.to(loss_.dtype)
    loss_ = loss_ * mask
    return torch.sum(loss_) / torch.sum(mask)


def masked_accuracy(y_true, y_pred):
    mask = torch.ne(y_true, 0)
    acc = torch.eq(y_true, torch.argmax(y_pred, dim=-1))
    acc = torch.logical_and(mask, acc)
    return acc.sum() / mask.sum()


def schedule_multiplier(
        step, total_steps, warmup_steps, *, alpha=0.0, const_lr=False,
        hold_until=-1):
    """Trainer-owned learning-rate multiplier at one scheduler epoch."""

    step = min(float(step), float(total_steps))
    total = float(total_steps)
    warmup = float(warmup_steps)
    floor = float(alpha)
    if const_lr:
        return min(1.0, (step + 1.0) / max(1.0, warmup))
    if hold_until >= 0:
        hold = float(hold_until)
        if step <= warmup and warmup > 1:
            return step / warmup
        if step <= hold:
            return 1.0
        decay = (step - hold) / (total - hold)
        return ((1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * decay))
                + floor)
    warmup_rise = step / warmup
    decay = (step - warmup) / (total - warmup)
    cosine = ((1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * decay))
              + floor)
    return warmup_rise if step <= warmup else cosine

def canonical_event_record(run_id, lineage_root, record, gpu_device_seconds):
    """Attach resolved identity fields after caller-owned event fields.

    Command-line configuration contains optional ``run_id`` and
    ``lineage_root`` arguments whose unresolved defaults are empty strings.
    The event ledger must carry the resolved values instead, including on its
    config row, so reserved identity fields are deliberately written last.
    """

    if not run_id or not lineage_root:
        raise ValueError("event records require resolved run and lineage ids")
    return {
        **record,
        "run_id": run_id,
        "lineage_root": lineage_root,
        "gpu_device_seconds": float(gpu_device_seconds),
    }


def LinearWarmupCosineLRSchedule(optimizer, total_steps, warmup_steps,
                                 alpha=0.0, last_epoch=-1):
    def lr_schedule_fun(step):
        return schedule_multiplier(
            step, total_steps, warmup_steps, alpha=alpha)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule_fun,
                                             last_epoch)


def HoldCosineLRSchedule(optimizer, total_steps, warmup_steps, hold_until,
                         alpha=0.1, last_epoch=-1):
    """Wave-8 plateau-then-anneal shape: linear warmup, hold at the peak
    rate until hold_until, cosine to alpha over the remaining steps.
    hold_until == 0 with warmup_steps == 1 is pure descent from the full
    rate at step 0.  The default cosine schedule is NOT routed through
    this function; its factor sequence is untouched."""
    def hold_schedule_fun(step):
        return schedule_multiplier(
            step, total_steps, warmup_steps, alpha=alpha,
            hold_until=hold_until)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, hold_schedule_fun,
                                             last_epoch)


def _schedule_phase(step, arguments):
    pulse_starts = [
        int(value) for value in arguments.pulse_train.split(",") if value
    ]
    in_pulse = (
        arguments.pulse_at >= 0
        and arguments.pulse_at <= step < arguments.pulse_at + arguments.pulse_len
    ) or any(
        start <= step < start + arguments.pulse_len for start in pulse_starts
    )
    if in_pulse:
        return "pulse"
    if arguments.const_lr:
        return "constant"
    if step <= arguments.warmup:
        return "warmup"
    if arguments.hold_until >= 0 and step <= arguments.hold_until:
        return "hold"
    return "cosine"


def create_masks(inp, device):
    size = inp.size()[1]
    look_ahead = 1 - torch.tril(torch.ones((size, size), device=device))
    pad = torch.eq(inp, 0).to(torch.float32)[:, None, None, :]
    return torch.maximum(pad, look_ahead.to(device))


# --- target-exposure ablation ----------------------------------------------

def last_only_targets(y):
    """Keep only the final-position label of each sequence (the structurally
    deployed conditional whose label is not among the supplied rows): the
    sequential-loss control for the target-exposure channel."""
    yz = torch.zeros_like(y)
    yz[:, -1] = y[:, -1]
    return yz


def adamw_fixed_step(opt, frozen_denom):
    """Manual AdamW step with a genuinely constant preconditioner
    (freeze_v_mode fixed): the first moment and its bias correction evolve
    normally and decoupled weight decay matches torch.optim.AdamW, but the
    denominator is the frozen sqrt(v_hat)+eps and exp_avg_sq is left
    untouched.  Call under torch.no_grad()."""
    for group in opt.param_groups:
        lr, wd = group["lr"], group["weight_decay"]
        b1 = group["betas"][0]
        for p in group["params"]:
            if p.grad is None or id(p) not in frozen_denom:
                continue
            st = opt.state[p]
            st["step"] += 1
            t = st["step"]
            t = t.item() if torch.is_tensor(t) else t
            m = st["exp_avg"]
            m.mul_(b1).add_(p.grad, alpha=1 - b1)
            p.mul_(1 - lr * wd)
            p.addcdiv_(m, frozen_denom[id(p)], value=-lr / (1 - b1 ** t))


def rand1_targets(y, seed):
    """Keep one uniformly drawn NON-final position per sequence: the
    supervised-position count and objective form match the last-only
    ablation, but the drawn label remains among the supplied rows, so the
    target-exposure channel stays active.  Deterministic in (seed, batch
    shape); the training loop passes the step index, probes pass 0 so the
    probe objective is fixed along training."""
    B, T = y.shape
    g = torch.Generator().manual_seed(int(seed) * 1000003 + 17)
    pos = torch.randint(0, T - 1, (B,), generator=g).to(y.device)
    rows = torch.arange(B, device=y.device)
    yz = torch.zeros_like(y)
    yz[rows, pos] = y[rows, pos]
    return yz


# --- global data reservation -----------------------------------------------

PROBE_RESERVE_CHUNKS = 5120  # tail chunks reserved for probes/val/prompts


def training_chunk_limit(total_chunks, reserve_chunks=PROBE_RESERVE_CHUNKS):
    """Return the first globally reserved chunk index used by training."""

    count = int(total_chunks)
    reserve = int(reserve_chunks)
    if count <= 0 or reserve <= 0 or reserve >= count:
        raise ValueError("the global probe reservation is invalid")
    return count - reserve


def validate_data_order(order, total_chunks,
                        reserve_chunks=PROBE_RESERVE_CHUNKS):
    """Validate an absolute chunk order with the trainer reserved region."""

    value = np.asarray(order)
    if value.ndim != 1 or value.dtype.kind not in "iu":
        raise ValueError("data order must be a one-dimensional integer array")
    value = np.asarray(value, dtype=np.int64)
    limit = training_chunk_limit(total_chunks, reserve_chunks)
    if (
        len(value) == 0
        or int(value.min()) < 0
        or int(value.max()) >= limit
    ):
        raise ValueError("data order enters the reserved probe region")
    if len(np.unique(value)) != len(value):
        raise ValueError("data order contains repeated training chunks")
    return value


def final_gate_registry_required(
    chronological_history_length, mixed_observer_active, final_gate_step_mode
):
    """Return whether final-gate tensors are required by any live consumer."""

    if chronological_history_length < 0:
        raise ValueError("chronological history length must be nonnegative")
    if final_gate_step_mode not in {
        "full", "frozen", "decay_only", "adaptive_only"
    }:
        raise ValueError("unknown final-gate step mode")
    return (
        chronological_history_length > 0
        or bool(mixed_observer_active)
        or final_gate_step_mode != "full"
    )


def final_gate_step_target(before, full_after, learning_rate, weight_decay, mode):
    """Return the post-step final-gate target for one factorial arm."""

    if mode not in {"full", "frozen", "decay_only", "adaptive_only"}:
        raise ValueError("unknown final-gate step mode")
    rate = float(learning_rate)
    decay = float(weight_decay)
    if not math.isfinite(rate) or not math.isfinite(decay) or rate < 0.0:
        raise ValueError("final-gate intervention coefficients are invalid")
    multiplier = 1.0 - rate * decay
    if mode == "full":
        return full_after
    if mode == "frozen":
        return before
    if mode == "decay_only":
        return before * multiplier
    return full_after + (1.0 - multiplier) * before

# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--run-id", default="",
                    help="launcher-created identity stored in every confirmation checkpoint")
    ap.add_argument("--lineage-root", default="",
                    help="fresh-run lineage root; defaults to --run-id")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument(
        "--learning_rate_multiplier", type=float, default=1.0,
        help="multiply every scheduled rate for a matched continuation arm",
    )
    ap.add_argument("--warmup", type=int, required=True)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dk", type=int, default=64)
    ap.add_argument("--adff", type=int, default=170)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tokens", default="data/refinedweb_tokens.npy")
    ap.add_argument("--tok_model", default="data/tokenizer.model")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--probe_every", type=int, default=100)
    ap.add_argument("--sharp_every", type=int, default=250)
    ap.add_argument("--sharp_pre_every", type=int, default=1000,
                    help="cadence of the preconditioned extremal-curvature "
                         "measurement (multi-start, replicate batches)")
    ap.add_argument("--ckpt_steps", type=str, default="1000,2000,4000,8000",
                    help="comma-separated global steps for model checkpoints")
    ap.add_argument("--optimizer_ckpt_steps", type=str, default="",
                    help="comma-separated global steps for checkpoints that "
                         "also retain optimizer state")
    ap.add_argument(
        "--chronological_history_length",
        type=int,
        default=0,
        help=(
            "retain this many signed post-clipping final-gate gradients in "
            "a digest-bound checkpoint sidecar; 0 disables capture"
        ),
    )
    ap.add_argument("--global_step_offset", type=int, default=-1,
                    help="global steps completed before this invocation; -1 "
                         "uses the initialized checkpoint step, or zero for "
                         "a fresh run")
    ap.add_argument("--schedule_total_steps", type=int, default=0,
                    help="global schedule horizon; 0 uses --steps and is "
                         "valid only for a fresh scheduled run")
    ap.add_argument("--val_batches", type=int, default=8)
    ap.add_argument("--dag", type=str, default="",
                    help="comma-separated lambda1,lambda2,lambda3 for "
                         "D_L(A_LM), D_L(A_P), D_L(G_LM); empty = off")
    ap.add_argument("--init_from", type=str, default="",
                    help="checkpoint to initialize model (and optimizer if "
                         "present) from, for continued-training runs")
    ap.add_argument("--reset_opt", action="store_true",
                    help="with --init_from: do not load the optimizer state")
    ap.add_argument("--confirmation_registry", default="",
                    help="frozen construction, validation, and application "
                         "chunk registry stored in every complete checkpoint")
    ap.add_argument("--campaign_spec", default="",
                    help="content-addressed prospective campaign specification "
                         "stored in every complete checkpoint")
    ap.add_argument("--confirmation_snapshot_steps", default="",
                    help="comma-separated optimizer steps at which to save "
                         "operation-ordered transport tensors; empty = off")
    ap.add_argument("--confirmation_snapshot_blocks", default="phi,plga",
                    help="comma-separated named tensor groups saved at the "
                         "registered confirmation steps; choices are phi, "
                         "plga, attn, ffn, query, key, value, and deductive")
    ap.add_argument("--confirmation_snapshot_plan", default="",
                    help="sparse STEP:GROUP+GROUP;... plan; mutually "
                         "exclusive with --confirmation_snapshot_steps")
    ap.add_argument("--const_lr", action="store_true",
                    help="use constant lr instead of warmup+cosine")
    ap.add_argument("--data_offset", type=int, default=-1,
                    help="training start offset in chunks; -1 = resume from "
                         "the checkpoint's data_offset_end (0 if none)")
    ap.add_argument(
        "--data_order",
        default="",
        help=(
            "optional .npy vector of absolute training-chunk indices; "
            "--data_offset is then a cursor into this frozen order"
        ),
    )
    ap.add_argument("--probe_region", choices=["global", "legacy"],
                    default="global",
                    help="global: probes/val/prompts from the reserved tail "
                         "region excluded from all training; legacy: the "
                         "wave-1..3 layout (probes just after this run's "
                         "training range)")
    ap.add_argument("--optimizer", choices=["adamw", "sgd", "sgdm"],
                    default="adamw")
    ap.add_argument("--wd", type=float, default=0.1,
                    help="weight decay (reference value 0.1 for AdamW)")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="gradient value-clipping level; <= 0 disables")
    ap.add_argument("--loss_mode", choices=["block", "last", "rand1"],
                    default="block",
                    help="block: reference blockwise loss at every aligned "
                         "position; last: final-position-only loss (the "
                         "target-exposure ablation); rand1: one random "
                         "non-final position per sequence (signal-count-"
                         "matched control with target exposure kept)")
    ap.add_argument("--sdpa", action="store_true",
                    help="SDPA control: G_LM = I at every layer via Gcache "
                         "(metric learner untrained and unused)")
    ap.add_argument("--freeze_g_at", type=int, default=-1,
                    help="freeze the metric-learner (phi) and PLGA-coupling "
                         "parameters at this step (0 = from start); -1 = "
                         "never")
    ap.add_argument("--freeze_v_at", type=int, default=-1,
                    help="freeze the Adam second-moment state at this step "
                         "(frozen-preconditioner control); -1 = never")
    ap.add_argument("--freeze_v_mode", choices=["restore", "fixed"],
                    default="restore",
                    help="restore: snapshot exp_avg_sq and copy it back "
                         "after each step (the accumulation ablation; each "
                         "update still carries a weight-(1-beta2) "
                         "contribution of the current squared gradient); "
                         "fixed: from the step after the trigger, take "
                         "manual AdamW steps whose denominator is the "
                         "frozen sqrt(v_hat)+eps (a genuinely constant "
                         "preconditioner)")
    ap.add_argument("--freeze_plga_at", type=int, default=-1,
                    help="freeze the PLGA coupling parameters (W, b_W, P, "
                         "a, b_a) at this step while the metric learner "
                         "stays trainable (0 = from start); -1 = never. "
                         "Like --freeze_g_at, a positive value takes "
                         "effect at step value+1")
    ap.add_argument("--freeze_final_gate_at", type=int, default=-1,
                    help="freeze only the final row-map LayerNorm scale in "
                         "every decoder layer at this step; 0 = from start")
    ap.add_argument(
        "--final_gate_step_mode",
        choices=["full", "frozen", "decay_only", "adaptive_only"],
        default="full",
        help=(
            "post-AdamW final-gate arm: retain the complete update, retain "
            "neither term, retain only decoupled decay, or retain only the "
            "adaptive innovation; optimizer moments continue on each arm-specific "
            "gradient path"
        ),
    )
    ap.add_argument("--freeze_row_program_at", type=int, default=-1,
                    help="freeze every reslayerAs parameter, but not PLGA, "
                         "at this step; 0 = from start")
    ap.add_argument("--freeze_upstream_shape_at", type=int, default=-1,
                    help="freeze every row-map parameter except the final "
                         "LayerNorm scale in every decoder layer at this "
                         "step; 0 = from start")
    ap.add_argument("--pulse_at", type=int, default=-1,
                    help="inject a large-step pulse: override the "
                         "scheduled lr with --pulse_lr for steps in "
                         "[pulse_at, pulse_at+pulse_len); -1 = off")
    ap.add_argument("--pulse_len", type=int, default=8)
    ap.add_argument("--pulse_lr", type=float, default=0.0)
    ap.add_argument("--pulse_train", type=str, default="",
                    help="comma-separated pulse start steps sharing "
                         "--pulse_len/--pulse_lr (wave-8 perturbation-"
                         "response protocol); mutually exclusive with "
                         "--pulse_at; empty = off")
    ap.add_argument("--anneal_floor", type=float, default=0.1,
                    help="cosine-anneal floor alpha (fraction of --lr "
                         "reached at the schedule end); the default 0.1 "
                         "reproduces the archived schedule factor "
                         "sequence exactly")
    ap.add_argument("--hold_until", type=int, default=-1,
                    help="hold the rate at --lr from the end of warmup "
                         "until this step, then cosine-anneal to "
                         "--anneal_floor over the remaining steps; 0 "
                         "with --warmup 1 gives pure descent from step "
                         "0; -1 = off (archived schedule shape)")
    ap.add_argument("--sharp_full", type=int, default=1,
                    help="1: include full-parameter-vector Lanczos in the "
                         "sharpness probes (default); 0: block statistics "
                         "only (the full reorthogonalization basis costs "
                         "n_iter parameter copies, prohibitive at large "
                         "model scale)")
    ap.add_argument("--sharp_block_every", type=int, default=0,
                    help="cadence of the per-fine-block raw and "
                         "preconditioned Lanczos probes and the tracked "
                         "signed modes (plga/phi/attn/ffn); 0 = off "
                         "(waves 1-5 behavior)")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient-accumulation micro-batches per optimizer "
                         "step; each step consumes accum sequential "
                         "micro-batches of --batch rows and reproduces the "
                         "exact batch*accum gradient of the masked mean "
                         "loss (matched-token batch controls)")
    ap.add_argument("--gen_prompts", type=int, default=8,
                    help="number of prompts for the generation order "
                         "parameter")
    ap.add_argument("--gen_cont", type=int, default=2,
                    help="stochastic continuations per prompt; m_gen "
                         "averages over all continuation pairs and a "
                         "bootstrap interval over prompts is reported")
    ap.add_argument("--skip_generation", action="store_true",
                    help="omit the terminal stochastic-generation diagnostic; "
                         "registered inference jobs run it separately")
    ap.add_argument(
        "--final_checkpoint",
        choices=["complete", "model", "none"],
        default="complete",
        help=(
            "terminal checkpoint retention; short intervention arms may "
            "select none after their terminal state has been logged"
        ),
    )
    ap.add_argument("--terminal_validation", action="store_true",
                    help="evaluate fixed validation batches at the final "
                         "step even when it is off the 1000-step grid")
    ap.add_argument("--fd_hs", type=str, default="0.01,0.02",
                    help="comma-separated step sizes for the "
                         "finite-difference collapse-curvature probe")
    ap.add_argument("--freeze_attn_at", type=int, default=-1,
                    help="freeze the attention projections (wq, wk, wv, "
                         "dense) at this step (0 = from start); -1 = "
                         "never; positive values take effect at step "
                         "value+1 like the other freeze flags")
    ap.add_argument("--freeze_ffn_at", type=int, default=-1,
                    help="freeze the pointwise feed-forward blocks at this "
                         "step (0 = from start); -1 = never")
    ap.add_argument("--mode_block", choices=["", "plga", "phi", "attn",
                                             "ffn"], default="",
                    help="apply the mode-amplitude intervention to this "
                         "tracked block (requires --sharp_block_every > 0)")
    ap.add_argument("--mode_scale", type=float, default=1.0,
                    help="after each optimizer step, rescale the component "
                         "of the realized block update along the tracked "
                         "mode by this factor (0.5 damps, 2.0 excites; "
                         "1.0 = off); the optimizer state is untouched")
    ap.add_argument("--mode_family", choices=["dyn", "gn", "live", "raw"],
                    default="dyn",
                    help="which tracked mode family the intervention "
                         "targets: dyn (streaming sign-corrected tracker "
                         "of the dominant coherent update direction; the "
                         "declared primary), live (top vector of the "
                         "live-restricted preconditioned block operator), "
                         "raw (top vector of the raw block Hessian); all "
                         "families are tracked and logged regardless")
    ap.add_argument("--mode_from", type=int, default=0,
                    help="first step at which the mode-amplitude "
                         "intervention applies")
    args = ap.parse_args()

    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if (
        not math.isfinite(args.learning_rate_multiplier)
        or args.learning_rate_multiplier <= 0.0
    ):
        raise ValueError("--learning_rate_multiplier must be finite and positive")
    effective_learning_rate = args.lr * args.learning_rate_multiplier
    if (
        not math.isfinite(effective_learning_rate)
        or effective_learning_rate <= 0.0
    ):
        raise ValueError("effective learning rate must be finite and positive")
    if args.global_step_offset < -1:
        raise ValueError("--global_step_offset must be -1 or nonnegative")
    if args.final_gate_step_mode != "full":
        if args.optimizer != "adamw":
            raise ValueError("final-gate factorial modes require AdamW")
        if args.freeze_final_gate_at >= 0:
            raise ValueError(
                "final-gate factorial mode cannot be combined with "
                "--freeze_final_gate_at"
            )
    if args.schedule_total_steps < 0:
        raise ValueError("--schedule_total_steps must be nonnegative")
    schedule_total_steps = args.schedule_total_steps or args.steps
    if args.chronological_history_length < 0:
        raise ValueError("--chronological_history_length must be nonnegative")
    if args.chronological_history_length > 0 and args.optimizer != "adamw":
        raise ValueError("chronological history capture requires AdamW")
    if not args.const_lr:
        assert 0 < args.warmup < schedule_total_steps, (
            "schedule requires 0 < warmup < steps, where steps is the "
            "global schedule horizon "
            f"(got warmup={args.warmup}, "
            f"schedule_total_steps={schedule_total_steps})")
    if args.pulse_at >= 0:
        assert args.pulse_lr > 0 and args.pulse_len > 0, (
            "--pulse_at requires --pulse_lr > 0 and --pulse_len > 0")
    if args.pulse_train:
        assert args.pulse_at < 0, (
            "--pulse_train and --pulse_at are mutually exclusive")
        assert args.pulse_lr > 0 and args.pulse_len > 0, (
            "--pulse_train requires --pulse_lr > 0 and --pulse_len > 0")
        _starts = [int(s) for s in args.pulse_train.split(",")]
        assert _starts == sorted(_starts), "--pulse_train must be sorted"
        assert all(b - a > args.pulse_len
                   for a, b in zip(_starts, _starts[1:])), (
            "--pulse_train pulses must be disjoint (gap > pulse_len)")
        assert _starts[0] > args.warmup, (
            "--pulse_train pulses must start after warmup")
    if args.hold_until >= 0:
        assert not args.const_lr, (
            "--hold_until is a scheduled-run flag; incompatible with "
            "--const_lr")
        assert args.hold_until < schedule_total_steps, (
            "--hold_until must lie strictly before the schedule horizon")
        assert args.warmup <= args.hold_until or args.hold_until == 0, (
            "--hold_until must not cut the warmup short (warmup <= "
            "hold_until, or hold_until == 0 for pure descent)")
    if args.accum > 1:
        assert args.probe_region == "global", (
            "--accum requires the global probe region")
        assert not args.dag and args.loss_mode == "block", (
            "--accum is defined for the blockwise CE objective only")
    if args.mode_block:
        assert args.sharp_block_every > 0, (
            "--mode_block requires --sharp_block_every > 0 (the tracked "
            "mode is refreshed at that cadence)")

    d_model = args.heads * args.dk
    dff = int(math.ceil(d_model * 4 * 2 / 3))
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_started = time.monotonic()
    rundir = os.path.join(args.outdir, args.name)
    os.makedirs(rundir, exist_ok=True)
    logf = open(os.path.join(rundir, "log.jsonl"), "a")

    def log(rec):
        rec = canonical_event_record(
            run_id, lineage_root, rec,
            time.monotonic() - run_started if device.type == "cuda" else 0.0,
        )
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    def terminal_resource_summary():
        peak_host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            peak_host *= 1024
        cuda = device.type == "cuda"
        return {
            "event": "resource_summary",
            "resolved_device": str(device),
            "elapsed_seconds": float(time.monotonic() - run_started),
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if cuda else 0),
            "peak_gpu_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device)) if cuda else 0),
            "peak_host_rss_bytes": peak_host,
        }

    # ---- init checkpoint (loaded once, reused below) ----------------------
    ck = None
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        if ck.get("schema_version") == CHECKPOINT_SCHEMA_VERSION:
            validate_complete_checkpoint(ck)
        elif ck.get("schema_version") == MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION:
            validate_model_only_checkpoint(ck)
            if not args.reset_opt:
                raise ValueError(
                    "model-only checkpoint initialization requires --reset_opt")
    checkpoint_step = int(ck.get("step", 0)) if ck is not None else 0
    run_id = args.run_id or f"{args.name}-seed{args.seed}-start{checkpoint_step}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id):
        raise ValueError("run identity must be a portable nonempty identifier")
    parent_identity = ck.get("run_identity") if ck is not None else None
    from_scratch = ck is None
    if from_scratch:
        lineage_root = args.lineage_root or run_id
    elif isinstance(parent_identity, dict):
        lineage_root = args.lineage_root or parent_identity.get(
            "lineage_root", parent_identity.get("run_id", ""))
    else:
        lineage_root = args.lineage_root
    if not lineage_root:
        raise ValueError("a continuation checkpoint lacks a lineage root")
    run_identity = {
        "schema_version": "pldr-run-identity-v1",
        "run_id": run_id,
        "lineage_root": lineage_root,
        "parent_run_id": (
            parent_identity.get("run_id")
            if isinstance(parent_identity, dict) else None),
        "from_scratch": from_scratch,
        "source_checkpoint_sha256": (
            (
                ck["state_manifest"].get("complete_state_sha256")
                or ck["state_manifest"].get("model_state_sha256")
            )
            if ck is not None and "state_manifest" in ck else None),
    }
    if args.global_step_offset < 0:
        global_step_offset = checkpoint_step
    else:
        global_step_offset = args.global_step_offset
    if ck is None and global_step_offset != 0:
        raise ValueError(
            "a nonzero --global_step_offset requires --init_from")
    if ck is not None and global_step_offset != checkpoint_step:
        raise ValueError(
            "global step offset disagrees with initialized checkpoint")
    global_step_end = global_step_offset + args.steps
    if not args.const_lr and global_step_end > schedule_total_steps:
        raise ValueError(
            "scheduled run extends beyond --schedule_total_steps")
    if (
        ck is not None
        and not args.const_lr
        and args.schedule_total_steps == 0
        and global_step_offset > 0
    ):
        raise ValueError(
            "scheduled continuation requires --schedule_total_steps")

    # ---- data -------------------------------------------------------------
    dataset_sha256 = sha256_path(args.tokens)
    tokenizer_sha256 = (
        sha256_path(args.tok_model)
        if args.tok_model and Path(args.tok_model).is_file()
        else None
    )
    campaign_binding = None
    if args.campaign_spec:
        campaign_path = Path(args.campaign_spec).resolve()
        frozen_campaign = load_json_object(campaign_path)
        recorded_digest = frozen_campaign.get("spec_sha256")
        unsigned_campaign = dict(frozen_campaign)
        unsigned_campaign.pop("spec_sha256", None)
        if (
            not isinstance(frozen_campaign.get("campaign_id"), str)
            or not frozen_campaign["campaign_id"]
            or not isinstance(recorded_digest, str)
            or recorded_digest != digest_object(unsigned_campaign)
        ):
            raise ValueError("campaign specification digest does not replay")
        if frozen_campaign["campaign_id"] == MIXED_COLLAPSE_CAMPAIGN_ID:
            if unsigned_campaign != mixed_collapse_campaign_design():
                raise ValueError("mixed-collapse campaign design drifted")
        campaign_binding = {
            "campaign_id": frozen_campaign["campaign_id"],
            "campaign_spec_sha256": recorded_digest,
            "campaign_spec_file_sha256": sha256_path(campaign_path),
            "campaign_spec_path": str(campaign_path),
        }
    elif ck is not None:
        campaign_binding = ck["data_state"].get("campaign_binding")
    if (
        ck is not None
        and ck["data_state"].get("campaign_binding") != campaign_binding
    ):
        raise ValueError("initialized checkpoint campaign binding disagrees")
    if args.confirmation_registry:
        measurement_registry = load_json_object(args.confirmation_registry)
        validate_measurement_registry(
            measurement_registry, require_nonempty=True)
    elif ck is not None:
        measurement_registry = ck["measurement_registry"]
        validate_measurement_registry(measurement_registry)
    else:
        measurement_registry = empty_measurement_registry(
            dataset_sha256=dataset_sha256,
            tokenizer_sha256=tokenizer_sha256,
            context_length=args.ctx,
        )
    if measurement_registry["dataset_sha256"] != dataset_sha256:
        raise ValueError("measurement registry dataset digest disagrees")
    if measurement_registry["context_length"] != args.ctx:
        raise ValueError("measurement registry context length disagrees")
    if measurement_registry["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError("measurement registry tokenizer digest disagrees")
    if ck is not None:
        if ck["data_state"]["dataset_sha256"] != dataset_sha256:
            raise ValueError("initialized checkpoint dataset digest disagrees")
        if ck["data_state"]["tokenizer_sha256"] != tokenizer_sha256:
            raise ValueError("initialized checkpoint tokenizer digest disagrees")
        if ck["measurement_registry"] != measurement_registry:
            raise ValueError("initialized checkpoint registry disagrees")
    toks = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    n_chunks = len(toks) // args.ctx
    chunks = toks[: n_chunks * args.ctx].reshape(n_chunks, args.ctx)
    finite_increment_rows = None
    if measurement_registry.get("schema_version") in {
        FINITE_INCREMENT_REGISTRY_SCHEMA,
        "pldr-source-restoring-registry-v1",
    }:
        finite_increment_rows = finite_increment_registered_rows(
            measurement_registry,
            chunks,
            device,
        )
    source_resolved_points = set()
    orbitwise_full_points = set()
    orbitwise_dense_points = set()
    mixed_collapse_full_points = set()
    mixed_collapse_dense_points = set()
    if (
        campaign_binding is not None
        and campaign_binding["campaign_id"] == SOURCE_RESOLVED_CAMPAIGN_ID
    ):
        if finite_increment_rows is None:
            raise ValueError(
                "source-resolved campaign needs the complete 24-context registry"
            )
        source_resolved_points = set(source_resolved_snapshot_updates())
    if (
        campaign_binding is not None
        and campaign_binding["campaign_id"] == ORBITWISE_CAMPAIGN_ID
    ):
        if finite_increment_rows is None:
            raise ValueError(
                "orbitwise campaign needs the complete 24-context registry"
            )
        orbitwise_full_points = set(orbitwise_full_snapshot_updates())
        orbitwise_dense_points = set(orbitwise_dense_snapshot_updates())
    if (
        campaign_binding is not None
        and campaign_binding["campaign_id"] == MIXED_COLLAPSE_CAMPAIGN_ID
    ):
        if finite_increment_rows is None:
            raise ValueError(
                "mixed-collapse campaign needs the complete 24-context registry"
            )
        mixed_collapse_full_points = set(
            mixed_collapse_full_snapshot_updates()
        )
        mixed_collapse_dense_points = set(
            mixed_collapse_dense_snapshot_updates()
        ) - mixed_collapse_full_points
    rows_per_step = args.batch * args.accum
    train_need = args.steps * rows_per_step

    offset_chunks = args.data_offset
    if offset_chunks < 0:
        offset_chunks = int(ck.get("data_offset_end", 0)) if ck else 0

    data_order = None
    data_order_sha256 = None
    if args.data_order:
        if args.probe_region != "global":
            raise ValueError("--data_order requires --probe_region global")
        order_path = Path(args.data_order)
        loaded_order = np.load(order_path, allow_pickle=False)
        data_order = validate_data_order(loaded_order, n_chunks)
        data_order_sha256 = sha256_path(order_path)

    checkpoint_order_sha256 = (
        ck["data_state"].get("data_order_sha256") if ck is not None else None
    )
    if ck is not None and checkpoint_order_sha256 != data_order_sha256:
        raise ValueError("initialized checkpoint data-order digest disagrees")
    available_rows = len(data_order) if data_order is not None else n_chunks
    if offset_chunks < 0 or offset_chunks + train_need > available_rows:
        raise ValueError("training cursor exceeds the available data order")

    def get_rows(c0, n):
        x = chunks[c0 : c0 + n]
        return torch.from_numpy(x.astype(np.int64)).to(device)

    def get_training_rows(c0, n):
        if data_order is None:
            return get_rows(c0, n)
        selected = data_order[c0 : c0 + n]
        x = chunks[selected]
        return torch.from_numpy(x.astype(np.int64)).to(device)

    def get_batch(i):
        return get_training_rows(offset_chunks + i * args.batch, args.batch)

    def get_step_rows(i):
        """All rows consumed by optimizer step i (accum micro-batches)."""
        return get_training_rows(offset_chunks + i * rows_per_step, rows_per_step)

    if args.probe_region == "global":
        pb = n_chunks - PROBE_RESERVE_CHUNKS
        assert data_order is not None or offset_chunks + train_need <= pb, (
            f"training range [{offset_chunks}, {offset_chunks + train_need})"
            f" overlaps the reserved probe region starting at {pb}")
        probe1 = get_rows(pb, 8)
        probe2 = get_rows(pb + 64, 8)
        sharp_batches_xy = [get_rows(pb + 128 + 32 * k, 8) for k in range(3)]
        val_starts = [pb + 256 + k * args.batch
                      for k in range(args.val_batches)]
        get_val = lambda s: get_rows(s, args.batch)  # noqa: E731
        # first 8 prompt positions match waves 1-5; additional prompts come
        # from a fresh sub-region (pb + 3200 + ...) so they cannot overlap
        # the tilt-measurement batches at pb + 2560 + ...
        prompt_rows = [chunks[pb + 2048 + 37 * k] if k < 8
                       else chunks[pb + 3200 + 37 * (k - 8)]
                       for k in range(args.gen_prompts)]
    else:
        assert offset_chunks == 0, "--probe_region legacy requires offset 0"
        assert n_chunks > train_need + 64 * args.batch, "not enough tokens"
        probe_base = train_need
        probe1 = get_batch(probe_base // args.batch + 1)[:8]
        probe2 = get_batch(probe_base // args.batch + 3)[:8]
        sharp_batches_xy = [get_batch(probe_base // args.batch + 5)[:8]]
        val_starts = [(probe_base // args.batch + 7 + k) * args.batch
                      for k in range(args.val_batches)]
        get_val = lambda s: get_rows(s, args.batch)  # noqa: E731
        prompt_rows = [chunks[probe_base // args.batch * args.batch
                              + 2000 + 37 * k]
                       for k in range(args.gen_prompts)]

    pm1 = create_masks(probe1[:, :-1], device)
    pm2 = create_masks(probe2[:, :-1], device)
    sharp_batches = []
    for sb in sharp_batches_xy:
        sx, sy = sb[:, :-1], sb[:, 1:]
        sharp_batches.append((sx, create_masks(sx, device), sy))
    sx, sm, sy = sharp_batches[0]

    # ---- model ------------------------------------------------------------
    model = PLDR_Model(
        num_layers=args.layers, d_model=d_model, num_heads=args.heads,
        dff=dff, input_vocab_size=32000, A_dff=args.adff,
        num_reslayerA=8, num_denseA=2, max_seq_len=4096,
        device=device,
    )
    nparams = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {nparams/1e6:.2f}M params, d_model={d_model}, "
          f"dff={dff}", flush=True)

    if ck is not None:
        model.load_state_dict(ck["model"])
        print(f"initialized from {args.init_from} (step {ck.get('step')}, "
              f"data_offset_end {ck.get('data_offset_end')})", flush=True)

    if args.sdpa:
        eye = torch.eye(args.dk, device=device)[None, None]  # [1,1,dk,dk]
        sdpa_gcache = [[eye, eye] for _ in range(args.layers)]
        orig_forward = model.forward

        def sdpa_forward(inputs, kvcachelst=None, Gcachelst=None, **kw):
            gc = sdpa_gcache if Gcachelst is None else Gcachelst
            return orig_forward(inputs, kvcachelst=kvcachelst,
                                Gcachelst=gc, **kw)

        model.forward = sdpa_forward

    def freeze_g(include_ln=False):
        n = 0
        for dec in model.decoder.dec_layers:
            for p in dec.mha1.reslayerAs.parameters():
                p.requires_grad_(False); n += 1
            for p in dec.mha1.plgatt_layer.parameters():
                p.requires_grad_(False); n += 1
            if include_ln:
                for p in dec.mha1.layernorm1.parameters():
                    p.requires_grad_(False); n += 1
        print(f"{args.name}: froze {n} metric-learner/PLGA tensors",
              flush=True)

    def freeze_plga():
        n = 0
        for dec in model.decoder.dec_layers:
            for p in dec.mha1.plgatt_layer.parameters():
                p.requires_grad_(False); n += 1
        print(f"{args.name}: froze {n} PLGA-coupling tensors "
              f"(metric learner stays trainable)", flush=True)

    def freeze_final_gate():
        gates = [
            dec.mha1.reslayerAs[-1].layernormA.weight
            for dec in model.decoder.dec_layers
        ]
        for parameter in gates:
            parameter.requires_grad_(False)
        print(f"{args.name}: froze {len(gates)} final LayerNorm gate tensors",
              flush=True)

    def freeze_row_program():
        count = 0
        for dec in model.decoder.dec_layers:
            for parameter in dec.mha1.reslayerAs.parameters():
                parameter.requires_grad_(False)
                count += 1
        print(f"{args.name}: froze {count} row-program tensors", flush=True)

    def freeze_upstream_shape():
        gates = {
            id(dec.mha1.reslayerAs[-1].layernormA.weight)
            for dec in model.decoder.dec_layers
        }
        count = 0
        for dec in model.decoder.dec_layers:
            for parameter in dec.mha1.reslayerAs.parameters():
                if id(parameter) not in gates:
                    parameter.requires_grad_(False)
                    count += 1
        print(f"{args.name}: froze {count} upstream row-shape tensors",
              flush=True)

    def freeze_attn():
        n = 0
        for dec in model.decoder.dec_layers:
            for mod in (dec.mha1.wq, dec.mha1.wk, dec.mha1.wv,
                        dec.mha1.dense):
                for p in mod.parameters():
                    p.requires_grad_(False); n += 1
        print(f"{args.name}: froze {n} attention-projection tensors "
              f"(carrier-search control)", flush=True)

    def freeze_ffn():
        n = 0
        for dec in model.decoder.dec_layers:
            for p in dec.ffn.parameters():
                p.requires_grad_(False); n += 1
        print(f"{args.name}: froze {n} feed-forward tensors "
              f"(carrier-search control)", flush=True)

    if args.sdpa:
        # SDPA control: the loss does not reach the metric-learner/PLGA
        # parameters, nor the LayerNorm feeding them, so they are frozen
        # (untrained, undecayed, excluded from curvature probes over
        # trainable blocks).
        freeze_g(include_ln=True)
    elif args.freeze_g_at == 0:
        freeze_g()
    if args.freeze_plga_at == 0:
        freeze_plga()
    if args.freeze_final_gate_at == 0:
        freeze_final_gate()
    if args.freeze_row_program_at == 0:
        freeze_row_program()
    if args.freeze_upstream_shape_at == 0:
        freeze_upstream_shape()
    if args.freeze_attn_at == 0:
        freeze_attn()
    if args.freeze_ffn_at == 0:
        freeze_ffn()

    # ---- optimizer / schedule ---------------------------------------------
    params = model.parameters()
    if args.optimizer == "adamw":
        opt = torch.optim.AdamW(
            params, lr=effective_learning_rate, betas=(0.9, 0.95),
                                eps=1e-5, weight_decay=args.wd)
    elif args.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=effective_learning_rate, momentum=0.0,
                              weight_decay=args.wd)
    else:  # sgdm
        opt = torch.optim.SGD(params, lr=effective_learning_rate, momentum=0.9,
                              weight_decay=args.wd)

    if ck is not None and "opt" in ck and not args.reset_opt:
        opt.load_state_dict(ck["opt"])
        for gview in opt.param_groups:
            gview["lr"] = effective_learning_rate
            # the loaded group carries the source run's initial_lr, which
            # any LR scheduler would silently adopt as its base
            gview["initial_lr"] = effective_learning_rate
    if global_step_offset > 0 and (ck is None or args.reset_opt or "opt" not in ck):
        for gview in opt.param_groups:
            gview["initial_lr"] = effective_learning_rate
    if args.const_lr:
        # constant rate with an optional declared safety ramp: factor
        # min(1, (s+1)/warmup).  warmup <= 1 gives a factor of exactly
        # 1.0 at every step, reproducing the archived const-lr runs
        # (they passed --warmup 1) bit for bit.
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda step: schedule_multiplier(
                step, schedule_total_steps, args.warmup,
                alpha=args.anneal_floor, const_lr=True),
            last_epoch=global_step_offset - 1)
    elif args.hold_until >= 0:
        sched = HoldCosineLRSchedule(
            opt, total_steps=schedule_total_steps,
            warmup_steps=args.warmup, hold_until=args.hold_until,
            alpha=args.anneal_floor,
            last_epoch=global_step_offset - 1,
        )
    else:
        sched = LinearWarmupCosineLRSchedule(
            opt, total_steps=schedule_total_steps,
            warmup_steps=args.warmup, alpha=args.anneal_floor,
            last_epoch=global_step_offset - 1,
        )
    if (
        ck is not None
        and ck.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
    ):
        source_config = ck["config"]
        schedule_matches = (
            float(source_config["lr"]) == float(args.lr)
            and float(source_config.get(
                "learning_rate_multiplier", 1.0))
                == float(args.learning_rate_multiplier)
            and int(source_config["warmup"]) == int(args.warmup)
            and bool(source_config.get("const_lr", False))
                == bool(args.const_lr)
            and int(source_config.get("hold_until", -1))
                == int(args.hold_until)
            and float(source_config.get("anneal_floor", 0.1))
                == float(args.anneal_floor)
            and int(ck["schedule_total_steps"])
                == int(schedule_total_steps)
        )
        if not args.reset_opt and schedule_matches:
            sched.load_state_dict(ck["scheduler"])
        restore_rng_states(ck["rng_states"])

    dag_coeffs = ([float(c) for c in args.dag.split(",")] if args.dag
                  else None)
    pulse_starts = ([int(s) for s in args.pulse_train.split(",")]
                    if args.pulse_train else None)

    def dag_losses(att_weights):
        """Reference-form DAG losses on (A_LM, A_P, G_LM), eq. (dagloss)."""
        alm = torch.stack([t[0] for t in att_weights])
        ap = torch.stack([torch.pow(t[0], t[1]) for t in att_weights])
        glm = torch.stack([t[4] for t in att_weights])
        dval = float(alm.shape[-1])
        out = []
        for M in (alm, ap, glm):
            tr = torch.einsum("...ii->...",
                              torch.linalg.matrix_exp(M * M))
            out.append(torch.abs(torch.log(tr / dval)).mean())
        return out

    rand1_state = {"seed": 0}

    def loss_fn_eff(y_true, y_pred):
        """The trained objective's CE part: blockwise, last-position, or
        random-single-position.  For rand1 the drawn positions depend on
        rand1_state["seed"]: the training loop sets it to the step index
        for the update and resets it to 0 before probes, so every probe
        evaluates one fixed objective along training."""
        if args.loss_mode == "last":
            return masked_loss_function(last_only_targets(y_true), y_pred)
        if args.loss_mode == "rand1":
            return masked_loss_function(
                rand1_targets(y_true, rand1_state["seed"]), y_pred)
        return masked_loss_function(y_true, y_pred)

    def loss_fn_eff64(y_true, y_pred):
        """float64-reduction variant of the effective CE objective, for
        the finite-difference curvature probe only."""
        if args.loss_mode == "last":
            return masked_loss_function64(last_only_targets(y_true), y_pred)
        if args.loss_mode == "rand1":
            return masked_loss_function64(
                rand1_targets(y_true, rand1_state["seed"]), y_pred)
        return masked_loss_function64(y_true, y_pred)

    # Full-training-objective loss for curvature probes on DAG cells: the
    # DAG penalties are recomputed from the attention weights of the most
    # recent forward pass (captured by a hook, so instrument's internal
    # forwards are covered; the tensors stay in the same autograd graph as
    # y_pred, so Hessian-vector products see the full objective).
    if dag_coeffs is not None:
        last_att = {}

        def _att_hook(module, hook_in, hook_out):
            last_att["att"] = hook_out[2]

        model.register_forward_hook(_att_hook)

        def probe_loss_full(y_true, y_pred):
            dA, dAp, dG = dag_losses(last_att["att"])
            return (loss_fn_eff(y_true, y_pred) + dag_coeffs[0] * dA
                    + dag_coeffs[1] * dAp + dag_coeffs[2] * dG)

        def probe_loss_full64(y_true, y_pred):
            dA, dAp, dG = dag_losses(last_att["att"])
            return (loss_fn_eff64(y_true, y_pred)
                    + dag_coeffs[0] * dA.double()
                    + dag_coeffs[1] * dAp.double()
                    + dag_coeffs[2] * dG.double())
    else:
        probe_loss_full = None
        probe_loss_full64 = None

    log({"event": "config", **vars(args), "n_params": nparams,
         "d_model": d_model, "dff": dff,
         "data_offset_chunks": offset_chunks,
         "data_offset_end": offset_chunks + train_need,
         "data_order_path": args.data_order or None,
         "data_order_sha256": data_order_sha256,
         "campaign_binding": campaign_binding,
         "global_step_offset_resolved": global_step_offset,
         "global_step_end": global_step_end,
         "schedule_total_steps_resolved": schedule_total_steps,
         "probe_reserve_chunks": (PROBE_RESERVE_CHUNKS
                                  if args.probe_region == "global" else 0),
         "n_chunks": int(n_chunks)})

    ckpt_steps = set(int(s) for s in args.ckpt_steps.split(",") if s)
    optimizer_ckpt_steps = set(
        int(s) for s in args.optimizer_ckpt_steps.split(",") if s)
    checkpoint_steps = ckpt_steps | optimizer_ckpt_steps
    if any(
        step <= global_step_offset or step > global_step_end
        for step in checkpoint_steps
    ):
        raise ValueError("checkpoint step is outside this invocation")
    frozen_v = None
    if args.confirmation_snapshot_plan and args.confirmation_snapshot_steps:
        raise ValueError(
            "confirmation snapshot plan and step list are mutually exclusive")
    confirmation_plan = {}
    if args.confirmation_snapshot_plan:
        for entry in args.confirmation_snapshot_plan.split(";"):
            if not entry or ":" not in entry:
                raise ValueError("malformed confirmation snapshot plan")
            step_text, block_text = entry.split(":", 1)
            step = int(step_text)
            blocks = tuple(name for name in block_text.split("+") if name)
            if step in confirmation_plan or not blocks:
                raise ValueError("duplicate or empty confirmation snapshot plan row")
            confirmation_plan[step] = blocks
    else:
        confirmation_blocks = tuple(
            s for s in args.confirmation_snapshot_blocks.split(",") if s)
        confirmation_plan = {
            int(step): confirmation_blocks
            for step in args.confirmation_snapshot_steps.split(",") if step
        }
    confirmation_steps = set(confirmation_plan)
    if any(
        step <= global_step_offset or step > global_step_end
        for step in confirmation_steps
    ):
        raise ValueError("confirmation snapshot step is outside this invocation")
    allowed_confirmation_blocks = {
        "phi", "plga", "attn", "ffn", "query", "key", "value",
        "deductive",
    }
    unknown_confirmation_blocks = set().union(
        *(set(blocks) for blocks in confirmation_plan.values())
    ) - allowed_confirmation_blocks if confirmation_plan else set()
    if unknown_confirmation_blocks:
        raise ValueError(
            "unknown confirmation snapshot blocks: "
            + ", ".join(sorted(unknown_confirmation_blocks)))
    if confirmation_steps and args.optimizer != "adamw":
        raise ValueError("confirmation transport snapshots require AdamW")
    frozen_denom = None  # freeze_v_mode fixed: id(p) -> sqrt(v_hat)+eps

    # parameter lists for per-step block velocities (all params of the
    # block, trainable or not, so frozen-block velocities read 0)
    plga_all = [p for dec in model.decoder.dec_layers
                for p in dec.mha1.plgatt_layer.parameters()]
    phi_all = [p for dec in model.decoder.dec_layers
               for p in dec.mha1.reslayerAs.parameters()]
    attn_all = [p for dec in model.decoder.dec_layers
                for mod in (dec.mha1.wq, dec.mha1.wk, dec.mha1.wv,
                            dec.mha1.dense)
                for p in mod.parameters()]
    ffn_all = [p for dec in model.decoder.dec_layers
               for p in dec.ffn.parameters()]
    query_all = [p for dec in model.decoder.dec_layers
                 for p in dec.mha1.wq.parameters()]
    key_all = [p for dec in model.decoder.dec_layers
               for p in dec.mha1.wk.parameters()]
    value_all = [p for dec in model.decoder.dec_layers
                 for p in dec.mha1.wv.parameters()]
    deductive_all = [*phi_all, *plga_all]
    all_fine = {
        "plga": plga_all,
        "phi": phi_all,
        "attn": attn_all,
        "ffn": ffn_all,
        "query": query_all,
        "key": key_all,
        "value": value_all,
        "deductive": deductive_all,
    }
    parameter_names = {id(p): name for name, p in model.named_parameters()}
    parameter_groups = {
        id(p): group for group in opt.param_groups for p in group["params"]
    }
    last_clip_mask = {
        name: torch.zeros_like(parameter, dtype=torch.bool, device="cpu")
        for name, parameter in model.named_parameters()
    }
    chronological_gates = (
        chronological_gate_parameters(model)
        if final_gate_registry_required(
            args.chronological_history_length,
            bool(mixed_collapse_full_points),
            args.final_gate_step_mode,
        )
        else {}
    )
    chronological_history = (
        restore_chronological_history(
            args.init_from,
            chronological_gates,
            checkpoint_step=checkpoint_step,
            history_length=args.chronological_history_length,
        )
        if args.chronological_history_length > 0 and args.init_from
        else empty_chronological_history(chronological_gates)
    )
    last_decay_mask = {
        name: torch.zeros_like(parameter, dtype=torch.bool, device="cpu")
        for name, parameter in model.named_parameters()
    }
    if any(not blocks for blocks in confirmation_plan.values()):
        raise ValueError("confirmation snapshot blocks are empty")


    # tracked signed modes (wave-6): parameter-space unit vectors per
    # (block, family), refreshed every sharp_block_every steps, held
    # between refreshes for the per-step signed projection series and
    # the mode-amplitude intervention.  Families: "live" = top vector of
    # the live-restricted preconditioned block operator; "raw" = top
    # vector of the raw block Hessian.
    mode_state = {}
    mode_q_sums = {}
    mode_q_counts = {}
    mode_blocks = []
    MODE_FAMS = (("gn", "pre_live", "gauss_newton"),
                 ("live", "pre_live", "hessian"),
                 ("raw", "raw", "hessian"))
    dyn_trackers = {}
    if args.sharp_block_every > 0:
        mode_blocks = ["plga", "attn"]
        if args.mode_block and args.mode_block not in mode_blocks:
            mode_blocks.append(args.mode_block)
        dyn_trackers = {blk: instrument.DynMode() for blk in mode_blocks}
    fd_hs = tuple(float(h) for h in args.fd_hs.split(",") if h)


    def save_ckpt(path, step, local_step, with_opt=False):
        cursor_end = offset_chunks + local_step * rows_per_step
        groups = []
        for group in opt.param_groups:
            groups.append({
                "learning_rate": float(group["lr"]),
                "initial_learning_rate": float(
                    group.get("initial_lr", effective_learning_rate)),
                "betas": (
                    [float(value) for value in group["betas"]]
                    if "betas" in group else None
                ),
                "epsilon": (
                    float(group["eps"]) if "eps" in group else None),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "momentum": float(group.get("momentum", 0.0)),
                "amsgrad": bool(group.get("amsgrad", False)),
                "maximize": bool(group.get("maximize", False)),
            })
        source_root = Path(__file__).resolve().parent
        code_paths = (
            source_root / "train_run.py",
            source_root / "pldr_model_v510.py",
            source_root / "optimizer_ledger.py",
            source_root / "confirm" / "confirmation_artifacts.py",
            source_root / "confirm" / "chronological_history.py",
        )
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "rng_states": capture_rng_states(),
            "data_state": {
                "dataset_sha256": dataset_sha256,
                "tokenizer_sha256": tokenizer_sha256,
                "tokens_path": str(Path(args.tokens)),
                "tokenizer_path": str(Path(args.tok_model)),
                "cursor_start": offset_chunks,
                "cursor_end": cursor_end,
                "rows_per_step": rows_per_step,
                "context_length": args.ctx,
                "probe_region": args.probe_region,
                "data_order_path": args.data_order or None,
                "data_order_sha256": data_order_sha256,
                "campaign_binding": campaign_binding,
            },
            "optimizer_config": {
                "name": args.optimizer,
                "gradient_clip_kind": (
                    "value" if args.clip > 0 else "disabled"),
                "gradient_clip_value": (
                    float(args.clip) if args.clip > 0 else None),
                "groups": groups,
            },
            "model_config": {
                "layers": args.layers,
                "heads": args.heads,
                "head_width": args.dk,
                "width": d_model,
                "feed_forward_width": dff,
                "row_hidden_width": args.adff,
                "row_residual_layers": 8,
                "row_dense_layers": 2,
                "vocabulary_size": 32000,
                "maximum_sequence_length": 4096,
                "dtype": str(next(model.parameters()).dtype),
            },
            "operation_order": list(TRAIN_UPDATE_ORDER),
            "branch_state": {
                "schema_version": "pldr-realized-branch-state-v3",
                "clip_mask": last_clip_mask,
                "decay_mask": last_decay_mask,
                "device": str(device),
                "dtype": str(next(model.parameters()).dtype),
                "scheduler_phase": _schedule_phase(step, args),
            },
            "measurement_registry": measurement_registry,
            "run_identity": run_identity,
            "intervention_state": {
                "freeze_g_at": args.freeze_g_at,
                "freeze_plga_at": args.freeze_plga_at,
                "freeze_final_gate_at": args.freeze_final_gate_at,
                "final_gate_step_mode": args.final_gate_step_mode,
                "freeze_row_program_at": args.freeze_row_program_at,
                "freeze_upstream_shape_at": args.freeze_upstream_shape_at,
                "freeze_attn_at": args.freeze_attn_at,
                "freeze_ffn_at": args.freeze_ffn_at,
                "freeze_v_at": args.freeze_v_at,
                "freeze_v_mode": args.freeze_v_mode,
                "pulse_at": args.pulse_at,
                "pulse_train": args.pulse_train,
                "pulse_len": args.pulse_len,
                "pulse_lr": args.pulse_lr,
                "mode_block": args.mode_block,
                "mode_family": args.mode_family,
                "mode_scale": args.mode_scale,
                "mode_from": args.mode_from,
            },
            "code_manifest": {
                str(item.relative_to(source_root)):
                    sha256_path(item)
                for item in code_paths
            },
            "state_manifest": {},
            "data_offset": offset_chunks,
            "data_offset_end": cursor_end,
            "segment_global_offset": global_step_offset,
            "segment_local_step": local_step,
            "schedule_total_steps": schedule_total_steps,
            "config": vars(args),
        }
        if with_opt:
            payload["state_manifest"] = build_state_manifest(payload)
            validate_complete_checkpoint(payload)
        else:
            retained = {
                name: payload[name]
                for name in (
                    "step", "model", "data_state", "optimizer_config",
                    "model_config", "operation_order", "intervention_state",
                    "code_manifest", "measurement_registry", "config",
                    "data_offset", "data_offset_end", "segment_global_offset",
                    "segment_local_step", "schedule_total_steps", "run_identity",
                )
            }
            retained["schema_version"] = MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION
            retained["state_manifest"] = build_model_only_state_manifest(retained)
            validate_model_only_checkpoint(retained)
            payload = retained
        torch.save(payload, path)
        if with_opt and args.chronological_history_length > 0:
            save_chronological_sidecar(
                path,
                step,
                chronological_history,
                chronological_gates,
                opt,
                history_length=args.chronological_history_length,
            )

    def begin_confirmation_snapshot(step, learning_rate):
        if step not in confirmation_steps:
            return None
        snapshot = {
            "schema_version": "pldr-adamw-transport-v2",
            "global_optimizer_step": step,
            "learning_rate_applied": float(learning_rate),
            "operation_order": [
                "averaged_raw_gradient",
                "value_clip",
                "first_moment",
                "second_moment",
                "bias_correction",
                "decoupled_decay_and_loss_update",
                "post_step_intervention",
            ],
            "blocks": {},
        }
        for block_name in confirmation_plan[step]:
            tensors = []
            for parameter in all_fine[block_name]:
                group = parameter_groups[id(parameter)]
                if group.get("amsgrad", False) or group.get("maximize", False):
                    raise ValueError(
                        "confirmation snapshot requires standard AdamW")
                state = opt.state.get(parameter, {})
                state_step = state.get("step", 0)
                if torch.is_tensor(state_step):
                    state_step = int(state_step.item())
                else:
                    state_step = int(state_step)
                tensors.append({
                    "name": parameter_names[id(parameter)],
                    "theta_before": parameter.detach().cpu().clone(),
                    "raw_gradient": (
                        None if parameter.grad is None
                        else parameter.grad.detach().cpu().clone()
                    ),
                    "first_moment_before": (
                        state["exp_avg"].detach().cpu().clone()
                        if "exp_avg" in state
                        else torch.zeros_like(parameter, device="cpu")
                    ),
                    "second_moment_before": (
                        state["exp_avg_sq"].detach().cpu().clone()
                        if "exp_avg_sq" in state
                        else torch.zeros_like(parameter, device="cpu")
                    ),
                    "state_step_before": state_step,
                    "beta1": float(group["betas"][0]),
                    "beta2": float(group["betas"][1]),
                    "epsilon": float(group["eps"]),
                    "weight_decay": float(group["weight_decay"]),
                    "clip_value": (
                        float(args.clip) if args.clip > 0 else None),
                })
            snapshot["blocks"][block_name] = tensors
        return snapshot

    def snapshot_after_clip(snapshot):
        if snapshot is None:
            return
        for block_name, tensors in snapshot["blocks"].items():
            for parameter, row in zip(all_fine[block_name], tensors):
                clipped = (
                    None if parameter.grad is None
                    else parameter.grad.detach().cpu().clone()
                )
                row["clipped_gradient"] = clipped
                if clipped is None:
                    row["clip_derivative"] = torch.zeros_like(
                        row["theta_before"], dtype=torch.bool)
                    row["decay_mask"] = torch.zeros_like(
                        row["theta_before"], dtype=torch.bool)
                else:
                    raw = row["raw_gradient"]
                    row["clip_derivative"] = (
                        torch.ones_like(raw, dtype=torch.bool)
                        if row["clip_value"] is None
                        else raw.abs() < row["clip_value"]
                    )
                    row["decay_mask"] = torch.full_like(
                        raw, row["weight_decay"] != 0.0,
                        dtype=torch.bool)

    def snapshot_after_optimizer(snapshot):
        if snapshot is None:
            return
        for block_name, tensors in snapshot["blocks"].items():
            for parameter, row in zip(all_fine[block_name], tensors):
                state = opt.state.get(parameter, {})
                state_step = state.get("step", 0)
                if torch.is_tensor(state_step):
                    state_step = int(state_step.item())
                else:
                    state_step = int(state_step)
                row["theta_after_optimizer"] = (
                    parameter.detach().cpu().clone())
                row["first_moment_after"] = (
                    state["exp_avg"].detach().cpu().clone())
                row["second_moment_after"] = (
                    state["exp_avg_sq"].detach().cpu().clone())
                row["state_step_after"] = state_step

    def finish_confirmation_snapshot(snapshot):
        if snapshot is None:
            return
        snapshot["learning_rate_prepared_next"] = float(
            opt.param_groups[0]["lr"])
        for block_name, tensors in snapshot["blocks"].items():
            for parameter, row in zip(all_fine[block_name], tensors):
                row["theta_after_intervention"] = (
                    parameter.detach().cpu().clone())
        path = os.path.join(
            rundir,
            "confirmation_transport_"
            f"{snapshot['global_optimizer_step']:09d}.pt",
        )
        torch.save(snapshot, path)


    # ---- training loop ----------------------------------------------------
    if global_step_offset in source_resolved_points:
        log({
            "event": "source_resolved_timepoint",
            "step": global_step_offset,
            "source_resolved_timepoint": source_resolved_timepoint_record(
                model,
                finite_increment_rows,
                step=global_step_offset,
                registry_sha256=measurement_registry["registry_sha256"],
            ),
        })
    if global_step_offset in orbitwise_full_points:
        log({
            "event": "orbitwise_full_timepoint",
            "step": global_step_offset,
            "orbitwise_full_timepoint": source_resolved_timepoint_record(
                model,
                finite_increment_rows,
                step=global_step_offset,
                registry_sha256=measurement_registry["registry_sha256"],
                campaign_id=ORBITWISE_CAMPAIGN_ID,
                point_schema=ORBITWISE_FULL_TIMEPOINT_SCHEMA,
                expected_map_count=ORBITWISE_REGISTRY["map_count"],
                include_absolute_plga=True,
            ),
        })
    if global_step_offset in orbitwise_dense_points:
        log({
            "event": "orbitwise_dense_timepoint",
            "step": global_step_offset,
            "orbitwise_dense_timepoint": source_resolved_timepoint_record(
                model,
                finite_increment_rows[
                    :len(ORBITWISE_REGISTRY["sentinel_context_ids"])
                ],
                step=global_step_offset,
                registry_sha256=measurement_registry["registry_sha256"],
                campaign_id=ORBITWISE_CAMPAIGN_ID,
                point_schema=ORBITWISE_DENSE_TIMEPOINT_SCHEMA,
                expected_map_count=ORBITWISE_REGISTRY["sentinel_map_count"],
                include_absolute_plga=True,
            ),
        })
    if global_step_offset in mixed_collapse_full_points:
        log({
            "event": "mixed_collapse_full_timepoint",
            "step": global_step_offset,
            "mixed_collapse_full_timepoint": source_resolved_timepoint_record(
                model,
                finite_increment_rows,
                step=global_step_offset,
                registry_sha256=measurement_registry["registry_sha256"],
                campaign_id=MIXED_COLLAPSE_CAMPAIGN_ID,
                point_schema=MIXED_COLLAPSE_FULL_TIMEPOINT_SCHEMA,
                expected_map_count=MIXED_COLLAPSE_REGISTRY["map_count"],
                include_absolute_plga=True,
            ),
        })
    if global_step_offset in mixed_collapse_dense_points:
        log({
            "event": "mixed_collapse_dense_timepoint",
            "step": global_step_offset,
            "mixed_collapse_dense_timepoint": source_resolved_timepoint_record(
                model,
                finite_increment_rows,
                step=global_step_offset,
                registry_sha256=measurement_registry["registry_sha256"],
                campaign_id=MIXED_COLLAPSE_CAMPAIGN_ID,
                point_schema=MIXED_COLLAPSE_DENSE_TIMEPOINT_SCHEMA,
                expected_map_count=MIXED_COLLAPSE_REGISTRY["map_count"],
                include_absolute_plga=True,
            ),
        })
    if mixed_collapse_full_points:
        log({
            "event": "mixed_collapse_gate_state",
            "step": global_step_offset,
            "final_gate_state": [
                parameter.detach().double().cpu().tolist()
                for parameter in chronological_gates.values()
            ],
            "final_gate_step_mode": args.final_gate_step_mode,
        })
    t0 = time.time()
    for local_step in range(1, args.steps + 1):
        step = global_step_offset + local_step
        if args.freeze_g_at > 0 and step - 1 == args.freeze_g_at:
            freeze_g()
        if args.freeze_plga_at > 0 and step - 1 == args.freeze_plga_at:
            freeze_plga()
        if args.freeze_final_gate_at > 0 and step - 1 == args.freeze_final_gate_at:
            freeze_final_gate()
        if args.freeze_row_program_at > 0 and step - 1 == args.freeze_row_program_at:
            freeze_row_program()
        if args.freeze_upstream_shape_at > 0 and step - 1 == args.freeze_upstream_shape_at:
            freeze_upstream_shape()
        if args.freeze_attn_at > 0 and step - 1 == args.freeze_attn_at:
            freeze_attn()
        if args.freeze_ffn_at > 0 and step - 1 == args.freeze_ffn_at:
            freeze_ffn()
        in_pulse = (args.pulse_at >= 0
                    and args.pulse_at <= step < args.pulse_at
                    + args.pulse_len)
        if pulse_starts is not None:
            in_pulse = any(t <= step < t + args.pulse_len
                           for t in pulse_starts)
        if in_pulse:
            # override the scheduled lr for this step only; the trailing
            # sched.step() recomputes the schedule value for the next step
            for gview in opt.param_groups:
                gview["lr"] = args.pulse_lr
        rand1_state["seed"] = step
        dag_vals = None
        if args.accum == 1:
            x = get_batch(local_step - 1)
            tar_inp = x[:, :-1]
            tar_real = x[:, 1:]
            mask = create_masks(tar_inp, device)
            preds, _, att_w, _ = model([tar_inp, mask])
            loss = loss_fn_eff(tar_real, preds)
            if dag_coeffs is not None:
                dA, dAp, dG = dag_losses(att_w)
                loss = (loss + dag_coeffs[0] * dA + dag_coeffs[1] * dAp
                        + dag_coeffs[2] * dG)
                dag_vals = (dA.item(), dAp.item(), dG.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
        else:
            # exact gradient accumulation for the blockwise CE: per micro-
            # batch, backpropagate the UNNORMALIZED masked sum; after all
            # micro-batches, divide every gradient by the total mask count.
            # This reproduces the batch*accum single-batch gradient of the
            # masked mean exactly (the masked mean is sum/count and the
            # gradient is linear in the sum), so accumulated small batches
            # are an identity control for the large batch.
            xall = get_step_rows(local_step - 1)
            opt.zero_grad(set_to_none=True)
            tot_sum, tot_cnt = 0.0, 0.0
            for a in range(args.accum):
                x = xall[a * args.batch:(a + 1) * args.batch]
                tar_inp = x[:, :-1]
                tar_real = x[:, 1:]
                mask = create_masks(tar_inp, device)
                preds, _, att_w, _ = model([tar_inp, mask])
                msk = torch.ne(tar_real, 0)
                yp = torch.permute(preds, (0, 2, 1))
                l_ = nn.CrossEntropyLoss(reduction="none")(yp, tar_real)
                lsum = (l_ * msk.to(l_.dtype)).sum()
                lsum.backward()
                tot_sum += lsum.item()
                tot_cnt += msk.sum().item()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(tot_cnt)
            loss = torch.tensor(tot_sum / tot_cnt)
        gnorm = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                               for p in model.parameters()
                               if p.grad is not None)).item()
        # per-step block gradient power (the carrier drive measurement)
        block_gnorms = {}
        for bname, plist in instrument.param_blocks(model).items():
            block_gnorms[f"gnorm_{bname}"] = torch.sqrt(
                sum((p.grad.detach() ** 2).sum() for p in plist
                    if p.grad is not None)).item() if plist else 0.0
        clip_fracs = {}
        if args.sharp_block_every > 0:
            for bname in ("attn", "ffn"):
                sq = sum((p.grad.detach() ** 2).sum()
                         for p in all_fine[bname] if p.grad is not None)
                block_gnorms[f"gnorm_{bname}"] = (
                    torch.sqrt(sq).item() if torch.is_tensor(sq) else 0.0)
            # clipping saturation per fine block, measured pre-clip: value
            # clipping breaks every linear-recurrence threshold, so its
            # per-block bite is logged next to the boundary statistics
            clip_fracs = instrument.clip_fractions(model, args.clip)
        lr_applied = opt.param_groups[0]["lr"]
        if (step in confirmation_steps
                and (frozen_denom is not None or frozen_v is not None)):
            raise ValueError(
                "confirmation snapshot requires ordinary AdamW state")
        confirmation_snapshot = begin_confirmation_snapshot(step, lr_applied)
        last_clip_mask = {}
        last_decay_mask = {}
        for parameter_name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None or args.clip <= 0:
                clipped = torch.zeros_like(
                    parameter, dtype=torch.bool, device="cpu")
            else:
                clipped = (
                    gradient.detach().abs() > args.clip
                ).to(device="cpu", dtype=torch.bool)
            group = parameter_groups[id(parameter)]
            decayed = (
                gradient is not None
                and float(group.get("weight_decay", 0.0)) != 0.0
            )
            last_clip_mask[parameter_name] = clipped
            last_decay_mask[parameter_name] = torch.full_like(
                clipped, decayed, dtype=torch.bool, device="cpu")
        if args.clip > 0:
            nn.utils.clip_grad_value_(model.parameters(),
                                      clip_value=args.clip)
        with torch.no_grad():
            prev_plga = [p.detach().clone() for p in plga_all]
            prev_phi = [p.detach().clone() for p in phi_all]
            factorial_gate_before = (
                [
                    parameter.detach().clone()
                    for parameter in chronological_gates.values()
                ]
                if args.final_gate_step_mode != "full" else None
            )
            prev_fine = None
            if args.sharp_block_every > 0:
                prev_fine = {n: [p.detach().clone() for p in all_fine[n]]
                             for n in ("attn", "ffn")}
        if args.chronological_history_length > 0:
            record_clipped_gate_gradients(
                chronological_history,
                chronological_gates,
                opt,
                global_step=step,
                history_length=args.chronological_history_length,
            )

        snapshot_after_clip(confirmation_snapshot)
        if frozen_denom is not None:
            with torch.no_grad():
                adamw_fixed_step(opt, frozen_denom)
        else:
            opt.step()
        snapshot_after_optimizer(confirmation_snapshot)
        if factorial_gate_before is not None:
            with torch.no_grad():
                for parameter, before in zip(
                    chronological_gates.values(),
                    factorial_gate_before,
                    strict=True,
                ):
                    group = parameter_groups[id(parameter)]
                    parameter.copy_(
                        final_gate_step_target(
                            before,
                            parameter,
                            group["lr"],
                            group.get("weight_decay", 0.0),
                            args.final_gate_step_mode,
                        )
                    )
        if args.freeze_v_at >= 0 and args.optimizer == "adamw":
            if (frozen_v is None and frozen_denom is None
                    and step >= args.freeze_v_at):
                if args.freeze_v_mode == "fixed":
                    b2 = opt.param_groups[0]["betas"][1]
                    eps = opt.param_groups[0]["eps"]
                    frozen_denom = {}
                    for p in model.parameters():
                        st = opt.state.get(p)
                        if st is not None and "exp_avg_sq" in st:
                            t = st["step"]
                            t = t.item() if torch.is_tensor(t) else t
                            frozen_denom[id(p)] = (
                                st["exp_avg_sq"] / (1 - b2 ** t)
                            ).sqrt().add_(eps)
                    print(f"{args.name}: froze Adam preconditioner "
                          f"(fixed denominator) at step {step}", flush=True)
                else:
                    frozen_v = {id(p): opt.state[p]["exp_avg_sq"].clone()
                                for p in model.parameters()
                                if p in opt.state
                                and "exp_avg_sq" in opt.state[p]}
                    print(f"{args.name}: froze Adam second moment at step "
                          f"{step} (restore mode: one step of adaptation "
                          f"per update remains)", flush=True)
            elif frozen_v is not None:
                for p in model.parameters():
                    st = opt.state.get(p)
                    if st is not None and id(p) in frozen_v:
                        st["exp_avg_sq"].copy_(frozen_v[id(p)])
        ledger_residuals = {}
        if (args.sharp_block_every > 0 and args.optimizer == "adamw"
                and frozen_denom is None and frozen_v is None):
            ledger_previous = {"plga": prev_plga, "phi": prev_phi}
            if prev_fine is not None:
                ledger_previous.update(prev_fine)
            for block_name, before in ledger_previous.items():
                residual = optimizer_ledger.adamw_step_ledger_residual(
                    opt, all_fine[block_name], before, lr_applied)
                if residual is not None:
                    ledger_residuals[
                        f"decay_ledger_residual_{block_name}"] = residual
        if (args.sharp_block_every > 0
                and args.optimizer == "adamw"):
            with torch.no_grad():
                for mblk in mode_blocks:
                    q_now = instrument.adam_inverse_precond(
                        opt, all_fine[mblk])
                    if q_now is None:
                        continue
                    if mblk not in mode_q_sums:
                        mode_q_sums[mblk] = [
                            torch.zeros_like(value) for value in q_now]
                        mode_q_counts[mblk] = 0
                    for acc, value in zip(mode_q_sums[mblk], q_now):
                        acc.add_(value)
                    mode_q_counts[mblk] += 1
        # per-step signed projections onto the tracked modes (measured on
        # the realized pre-intervention update), then the mode-amplitude
        # intervention: rescale the along-mode component of the realized
        # block update by mode_scale, leaving the optimizer state untouched
        mode_projs = {}
        mode_amp = None
        if mode_state or dyn_trackers:
            prev_map = {"plga": prev_plga, "phi": prev_phi}
            if prev_fine is not None:
                prev_map.update(prev_fine)
            with torch.no_grad():
                for blk, trk in dyn_trackers.items():
                    if blk in prev_map:
                        c = trk.update(all_fine[blk], prev_map[blk])
                        if c is not None:
                            mode_projs[f"{blk}_dyn"] = c
                for mname, vec in mode_state.items():
                    blk = mname.rsplit("_", 1)[0]
                    if (blk in prev_map
                            and len(vec) == len(all_fine[blk])):
                        mode_projs[mname] = instrument.signed_projection(
                            all_fine[blk], prev_map[blk], vec)
                target = (f"{args.mode_block}_{args.mode_family}"
                          if args.mode_block else "")
                if (target and args.mode_scale != 1.0
                        and step >= args.mode_from
                        and target in mode_projs):
                    c = mode_projs[target]
                    vec = (dyn_trackers[args.mode_block].v
                           if args.mode_family == "dyn"
                           else mode_state[target])
                    if vec is not None:
                        for p, v in zip(all_fine[args.mode_block], vec):
                            p.add_((args.mode_scale - 1.0) * c * v)
                        mode_amp = (args.mode_scale - 1.0) * c
        sched.step()
        finish_confirmation_snapshot(confirmation_snapshot)
        rand1_state["seed"] = 0  # probes evaluate the fixed objective

        # per-step block parameter velocities (carrier oscillation power)
        with torch.no_grad():
            dtheta_plga = torch.sqrt(
                sum(((p - q) ** 2).sum()
                    for p, q in zip(plga_all, prev_plga))).item()
            dtheta_phi = torch.sqrt(
                sum(((p - q) ** 2).sum()
                    for p, q in zip(phi_all, prev_phi))).item()
            dtheta_fine = {}
            if prev_fine is not None:
                for n in ("attn", "ffn"):
                    dtheta_fine[f"dtheta_{n}"] = torch.sqrt(
                        sum(((p - q) ** 2).sum()
                            for p, q in zip(all_fine[n],
                                            prev_fine[n]))).item()

        # LR record indexing (documented convention; pinned by
        # tests/test_lr_record_indexing.py): sched.step() has already
        # run above, so a NON-PULSE record at step t stores
        # sched.get_last_lr(), the rate prepared for the NEXT update
        # (the one step t+1 will apply); a PULSE record stores the
        # pulse rate just applied at step t.  Retained deliberately:
        # the probes below evaluate after the update, against the
        # state the recorded rate will act on.  Consumers aligning
        # applied rates to steps must shift non-pulse records by one.
        rec = {"step": step, "loss": loss.item(),
               "lr": args.pulse_lr if in_pulse else sched.get_last_lr()[0],
               "gnorm": gnorm, "dtheta_plga": dtheta_plga,
               "dtheta_phi": dtheta_phi, **block_gnorms, **dtheta_fine,
               **clip_fracs, **ledger_residuals}
        if mixed_collapse_full_points:
            rec["final_gate_state"] = [
                parameter.detach().double().cpu().tolist()
                for parameter in chronological_gates.values()
            ]
            rec["final_gate_step_mode"] = args.final_gate_step_mode
            rec["learning_rate_applied"] = float(lr_applied)
        for mname, c in mode_projs.items():
            rec[f"mode_proj_{mname}"] = c
        if mode_amp is not None:
            rec["mode_amp_applied"] = mode_amp
        if in_pulse:
            rec["pulse"] = True
        if dag_vals is not None and (step % 20 == 0 or step <= 10):
            rec["dag_A"], rec["dag_Ap"], rec["dag_G"] = dag_vals
        if step % 20 == 0 or step <= 10:
            rec["acc"] = masked_accuracy(tar_real, preds).item()
        if step % 500 == 0:
            el = time.time() - t0
            print(f"{args.name} step {step} loss {loss.item():.4f} "
                  f"lr {rec['lr']:.2e} {local_step/el:.2f} it/s", flush=True)

        if step % args.probe_every == 0 or local_step == 1:
            with torch.no_grad():
                op = instrument.order_params(model, probe1[:, :-1], pm1,
                                             probe2[:, :-1], pm2)
            rec.update(op)
            rd = instrument.rowmap_diagnostics(model, probe1[:, :-1], pm1)
            rec.update({f"rowmap_{k}": v for k, v in rd.items()})
            rec.update(instrument.coupling_norms(model))
            rec["att_entropy_layers"] = instrument.attention_entropy(
                model, probe1[:, :-1], pm1)
            rec["repr_effrank"] = instrument.repr_effrank(
                model, probe1[:, :-1], pm1)
            if (
                finite_increment_rows is not None
                and step in FINITE_INCREMENT_TIME_COURSE_ANCHORS
            ):
                rec["finite_increment_timecourse"] = (
                    finite_increment_timecourse_record(
                        model,
                        finite_increment_rows,
                        step=step,
                        registry_sha256=measurement_registry[
                            "registry_sha256"
                        ],
                    )
                )

        if step in source_resolved_points:
            rec["source_resolved_timepoint"] = source_resolved_timepoint_record(
                model,
                finite_increment_rows,
                step=step,
                registry_sha256=measurement_registry["registry_sha256"],
            )
        if step in orbitwise_full_points:
            rec["orbitwise_full_timepoint"] = source_resolved_timepoint_record(
                model,
                finite_increment_rows,
                step=step,
                registry_sha256=measurement_registry["registry_sha256"],
                campaign_id=ORBITWISE_CAMPAIGN_ID,
                point_schema=ORBITWISE_FULL_TIMEPOINT_SCHEMA,
                expected_map_count=ORBITWISE_REGISTRY["map_count"],
                include_absolute_plga=True,
            )
        if step in orbitwise_dense_points:
            rec["orbitwise_dense_timepoint"] = source_resolved_timepoint_record(
                model,
                finite_increment_rows[
                    :len(ORBITWISE_REGISTRY["sentinel_context_ids"])
                ],
                step=step,
                registry_sha256=measurement_registry["registry_sha256"],
                campaign_id=ORBITWISE_CAMPAIGN_ID,
                point_schema=ORBITWISE_DENSE_TIMEPOINT_SCHEMA,
                expected_map_count=ORBITWISE_REGISTRY["sentinel_map_count"],
                include_absolute_plga=True,
            )
        if step in mixed_collapse_full_points:
            rec["mixed_collapse_full_timepoint"] = (
                source_resolved_timepoint_record(
                    model,
                    finite_increment_rows,
                    step=step,
                    registry_sha256=measurement_registry["registry_sha256"],
                    campaign_id=MIXED_COLLAPSE_CAMPAIGN_ID,
                    point_schema=MIXED_COLLAPSE_FULL_TIMEPOINT_SCHEMA,
                    expected_map_count=MIXED_COLLAPSE_REGISTRY["map_count"],
                    include_absolute_plga=True,
                )
            )
        if step in mixed_collapse_dense_points:
            rec["mixed_collapse_dense_timepoint"] = (
                source_resolved_timepoint_record(
                    model,
                    finite_increment_rows,
                    step=step,
                    registry_sha256=measurement_registry["registry_sha256"],
                    campaign_id=MIXED_COLLAPSE_CAMPAIGN_ID,
                    point_schema=MIXED_COLLAPSE_DENSE_TIMEPOINT_SCHEMA,
                    expected_map_count=MIXED_COLLAPSE_REGISTRY["map_count"],
                    include_absolute_plga=True,
                )
            )

        if step % args.sharp_every == 0 or step == 50:
            sh = instrument.all_sharpness(
                model, loss_fn_eff, sx, sm, sy,
                optimizer=opt if args.optimizer == "adamw" else None,
                full=bool(args.sharp_full))
            rec.update(sh)
            cd = instrument.collapse_dir_curvature(
                model, loss_fn_eff, sx, sm, sy,
                opt if args.optimizer == "adamw" else None,
                probe1[:, :-1], pm1)
            rec.update(cd)
            # fd probe: float64 loss reduction so second differences are
            # not quantized at the float32 ULP scale; hs from --fd_hs
            fd = instrument.fd_collapse_curvature(
                model, loss_fn_eff64, sharp_batches, probe1[:, :-1], pm1,
                hs=fd_hs)
            rec.update(fd)
            if probe_loss_full is not None:
                shf = instrument.all_sharpness(
                    model, probe_loss_full, sx, sm, sy,
                    optimizer=opt if args.optimizer == "adamw" else None)
                rec.update({f"{k}_fullobj": v for k, v in shf.items()})
                cdf = instrument.collapse_dir_curvature(
                    model, probe_loss_full, sx, sm, sy,
                    opt if args.optimizer == "adamw" else None,
                    probe1[:, :-1], pm1)
                rec.update({f"{k}_fullobj": v for k, v in cdf.items()})
                fdf = instrument.fd_collapse_curvature(
                    model, probe_loss_full64, sharp_batches,
                    probe1[:, :-1], pm1, hs=fd_hs)
                rec.update({f"{k}_fullobj": v for k, v in fdf.items()})

        if args.sharp_block_every > 0 and step % args.sharp_block_every == 0:
            # dense-cadence carrier-search probes: raw + preconditioned
            # extremal curvature per fine block (CE objective, one batch,
            # one start) and the tracked signed modes with warm-started,
            # overlap-matched top vectors
            fine = instrument.param_blocks_fine(model)
            bs = instrument.block_sharpness(
                model, loss_fn_eff, sx, sm, sy,
                optimizer=opt if args.optimizer == "adamw" else None,
                blocks=fine)
            rec.update(bs)
            for mblk in mode_blocks:
                mparams = fine.get(mblk)
                q_count = mode_q_counts.get(mblk, 0)
                q_avg = (
                    [value / q_count for value in mode_q_sums[mblk]]
                    if q_count else None)
                if not (mparams and len(mparams) == len(all_fine[mblk])):
                    mode_q_sums.pop(mblk, None)
                    mode_q_counts.pop(mblk, None)
                    continue
                rec[f"mode_preconditioner_window_count_{mblk}"] = q_count
                rec[f"mode_preconditioner_average_{mblk}"] = "inverse_trailing"
                for fam, variant, operator in MODE_FAMS:
                    mname = f"{mblk}_{fam}"
                    mstat, mvec = instrument.mode_track(
                        model, loss_fn_eff, sx, sm, sy,
                        opt if args.optimizer == "adamw" else None,
                        mparams, prev_vec=mode_state.get(mname),
                        variant=variant, operator=operator,
                        inverse_preconditioner_avg=q_avg)
                    rec.update({f"{k}_{mname}": v
                                for k, v in mstat.items()})
                    if mvec is not None:
                        mode_state[mname] = mvec
                # alignment of the dynamical tracker with the eigen modes
                trk = dyn_trackers.get(mblk)
                if trk is not None and trk.v is not None:
                    with torch.no_grad():
                        for fam, _, _ in MODE_FAMS:
                            mv = mode_state.get(f"{mblk}_{fam}")
                            if mv is not None and len(mv) == len(trk.v):
                                al = abs(sum(
                                    (a * b).sum()
                                    for a, b in zip(trk.v, mv)).item())
                                rec[f"mode_dyn_align_{fam}_{mblk}"] = al
                mode_q_sums.pop(mblk, None)
                mode_q_counts.pop(mblk, None)

        if step % args.sharp_pre_every == 0:
            gr = instrument.gram_restricted(
                model, probe1[:, :-1], pm1,
                optimizer=opt if args.optimizer == "adamw" else None)
            rec.update(gr)
            if args.optimizer == "adamw":
                pre = instrument.precond_sharpness(
                    model, loss_fn_eff, sharp_batches[:2], opt)
                rec.update(pre)
                if probe_loss_full is not None:
                    pref = instrument.precond_sharpness(
                        model, probe_loss_full, sharp_batches[:2], opt)
                    rec.update({f"{k}_fullobj": v for k, v in pref.items()})

        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                vl, va, nb = 0.0, 0.0, 0
                vll = 0.0
                for vs in val_starts:
                    vx = get_val(vs)
                    vinp, vreal = vx[:, :-1], vx[:, 1:]
                    vm = create_masks(vinp, device)
                    vp, _, _, _ = model([vinp, vm])
                    vl += masked_loss_function(vreal, vp).item()
                    va += masked_accuracy(vreal, vp).item()
                    if args.loss_mode == "last":
                        vll += masked_loss_function(
                            last_only_targets(vreal), vp).item()
                    elif args.loss_mode == "rand1":
                        vll += masked_loss_function(
                            rand1_targets(vreal, 0), vp).item()
                    nb += 1
            model.train()
            rec["val_loss"] = vl / nb
            rec["val_acc"] = va / nb
            if args.loss_mode == "last":
                rec["val_loss_last"] = vll / nb
            elif args.loss_mode == "rand1":
                rec["val_loss_rand1"] = vll / nb

        log(rec)

        if step in checkpoint_steps:
            save_ckpt(
                os.path.join(rundir, f"ckpt_{step}.pt"),
                step,
                local_step,
                with_opt=step in optimizer_ckpt_steps,
            )

    if args.terminal_validation and global_step_end % 1000 != 0:
        model.eval()
        with torch.no_grad():
            validation_loss = 0.0
            validation_accuracy = 0.0
            validation_count = 0
            for validation_start in val_starts:
                validation_batch = get_val(validation_start)
                validation_input = validation_batch[:, :-1]
                validation_target = validation_batch[:, 1:]
                validation_mask = create_masks(validation_input, device)
                validation_prediction, _, _, _ = model(
                    [validation_input, validation_mask])
                validation_loss += masked_loss_function(
                    validation_target, validation_prediction).item()
                validation_accuracy += masked_accuracy(
                    validation_target, validation_prediction).item()
                validation_count += 1
        model.train()
        log({
            "event": "terminal_validation",
            "step": global_step_end,
            "val_loss": validation_loss / validation_count,
            "val_acc": validation_accuracy / validation_count,
            "val_batches": validation_count,
        })

    if args.final_checkpoint != "none":
        save_ckpt(
            os.path.join(rundir, "ckpt_final.pt"),
            global_step_end,
            args.steps,
            with_opt=args.final_checkpoint == "complete",
        )

    # ---- final: stochastic-generation order parameter + samples -----------
    if args.skip_generation:
        log(terminal_resource_summary())
        log({
            "event": "final",
            "step": global_step_end,
            "generation_skipped": True,
        })
        logf.close()
        print(f"{args.name} FINAL generation skipped", flush=True)
        return

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tok_model)

    def generate(prompt_ids, n_new=64, top_p=0.8, temp=1.0, gen=None):
        ids = list(prompt_ids)
        for _ in range(n_new):
            x = torch.tensor(ids, device=device)[None, :]
            m = create_masks(x, device)
            with torch.no_grad():
                logits, _, att, kv = model([x, m])
            logits = logits[0, -1] / temp
            probs = F.softmax(logits, dim=-1)
            sp_, si = torch.sort(probs, descending=True)
            cum = torch.cumsum(sp_, 0)
            k = int((cum < top_p).sum().item()) + 1
            pk = sp_[:k] / sp_[:k].sum()
            choice = si[torch.multinomial(pk, 1, generator=gen)]
            ids.append(choice.item())
            if choice.item() == sp.eos_id():
                break
        return ids, att, kv

    model.eval()
    gen = torch.Generator(device=device)
    prompts = [[int(t) for t in row[:96]] for row in prompt_rows]

    def ded_state(att, kv):
        A = torch.stack([kvi[2] for kvi in kv])
        G = torch.stack([a[4] for a in att])
        ALM = torch.stack([a[0] for a in att])
        AP = torch.stack([torch.pow(a[0], a[1]) for a in att])
        return {"A": A, "ALM": ALM, "AP": AP, "GLM": G}

    # per prompt: gen_cont stochastic continuations; the per-prompt value
    # is the mean of norm_rmse over all continuation pairs (gen_cont = 2
    # with seeds 1000+pi / 2000+pi reproduces the waves-1..5 statistic
    # exactly); a prompt-level bootstrap gives the 95% interval on m_gen
    m_runs = {"A": [], "ALM": [], "AP": [], "GLM": []}
    samples = []
    for pi, p in enumerate(prompts):
        states, id_lists = [], []
        for ci in range(args.gen_cont):
            gen.manual_seed(1000 * (ci + 1) + pi)
            ids_c, att_c, kv_c = generate(p, gen=gen)
            states.append(ded_state(att_c, kv_c))
            id_lists.append(ids_c)
        for k in m_runs:
            vals = [instrument.norm_rmse(states[i][k], states[j][k])
                    for i in range(len(states))
                    for j in range(i + 1, len(states))]
            m_runs[k].append(float(np.mean(vals)))
        if pi < 3:
            samples.append({"prompt": sp.decode(p),
                            "run1": sp.decode(id_lists[0][len(p):]),
                            "run2": sp.decode(id_lists[1][len(p):])})

    rng = np.random.default_rng(0)
    m_ci = {}
    for k, v in m_runs.items():
        arr = np.array(v)
        boots = [float(np.mean(arr[rng.integers(0, len(arr), len(arr))]))
                 for _ in range(1000)]
        m_ci[k] = [float(np.percentile(boots, 2.5)),
                   float(np.percentile(boots, 97.5))]

    final = {"event": "final", "step": global_step_end,
             "m_gen": {k: float(np.mean(v)) for k, v in m_runs.items()},
             "m_gen_all": m_runs, "m_gen_ci": m_ci, "samples": samples}
    log(terminal_resource_summary())
    log(final)
    print(f"{args.name} FINAL m_gen:", final["m_gen"], flush=True)
    logf.close()


if __name__ == "__main__":
    main()
