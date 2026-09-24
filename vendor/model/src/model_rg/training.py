"""Train the pinned PLDR architecture without changing its inference adapter.

The width family keeps five layers, 64-dimensional heads, and the complete
eight-unit metric generator. Only the number of heads and residual/FFN width
change. A separate module namespace prevents adapter state from being shared.
"""
from __future__ import annotations
import importlib
import math
import sys
import types
import weakref
import uuid
from pathlib import Path
import numpy as np
import torch
from model_rg.native import NativeModel, _AttentionState


class TrainingModel(NativeModel):
    def __init__(self, source, heads, seed, device="cuda:0"):
        self.path = Path(source).resolve()
        package_name = "training_pldr_" + uuid.uuid4().hex
        package = types.ModuleType(package_name)
        package.__path__ = [str(self.path)]
        sys.modules[package_name] = package
        config_module = importlib.import_module(package_name + ".configuration_pldrllm")
        self.module = importlib.import_module(package_name + ".modeling_pldrllm")
        config = config_module.PldrllmConfig(
            hidden_size=64*heads, intermediate_size=(64*heads*8)//3,
            num_attention_heads=heads, num_hidden_layers=5, head_dim=64,
            num_reslayerA=8, num_denseA=2, A_dff=170, use_cache=False,
            attention_dropout=0.0)
        config._attn_implementation = "eager"
        torch.manual_seed(seed)
        self.model = self.module.PldrllmForCausalLM(config).to(device)
        self.model.requires_grad_(True)
        self.device = torch.device(device)
        self.config = config
        self._attention_state = _AttentionState()
        self.model._rg_attention_state = self._attention_state
        self.eta = None
        self.capture = False
        self.head_outputs = {}
        self._attention = self.module.eager_attention_forward

        def mask_adapter(**kwargs):
            x = kwargs.get("input_embeds", kwargs.get("inputs_embeds"))
            if kwargs.get("past_key_values") is not None:
                raise ValueError("Training measurements use no cache")
            am = kwargs.get("attention_mask")
            if am is not None and not bool((am == 1).all()):
                raise ValueError("Training measurements use no padding")
            s = x.shape[1]
            mask = torch.full((s, s), torch.finfo(x.dtype).min, dtype=x.dtype, device=x.device)
            return torch.triu(mask, diagonal=1)[None, None]

        state_ref = weakref.ref(self._attention_state)

        def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
            state = state_ref()
            if state is None:
                raise RuntimeError("The attention model state has been released")
            if state.eta is not None:
                eta = state.eta
                factors = eta[module.layer_idx][None, :, None, None]
                query = query*(1+factors)
            out, weights = state._attention(module, query, key, value, attention_mask,
                                           scaling, dropout, **kwargs)
            if state.capture:
                state.head_outputs[module.layer_idx] = out[:, -1].square().mean(-1).sqrt()
            return out, weights

        self.module.create_causal_mask = mask_adapter
        self.module.eager_attention_forward = attention
        rng = np.random.default_rng(1907)
        self.hidden_projection = torch.tensor(rng.normal(size=(config.hidden_size, 4))/math.sqrt(config.hidden_size),
                                              dtype=torch.float32, device=device)
        self.logit_projection = torch.tensor(rng.normal(size=(config.vocab_size, 8))/math.sqrt(config.vocab_size),
                                             dtype=torch.float32, device=device)
        self.logit_projection -= self.logit_projection.mean(0, keepdim=True)


def mixture_covariance(fields):
    """Exact total covariance of a uniform empirical seed/context mixture.

    fields has shape (seeds, contexts, observations). Population denominators
    describe the declared finite empirical law, not unbiased estimators of an
    unknown population covariance.
    """
    x = np.asarray(fields, dtype=np.float64)
    if x.ndim != 3 or min(x.shape[:2]) < 1:
        raise ValueError("Expected nonempty seed, context, observation axes")
    means = x.mean(1)
    residual = x-means[:, None]
    within = np.einsum("sni,snj->ij", residual, residual)/(x.shape[0]*x.shape[1])
    seed_residual = means-means.mean(0)
    between = seed_residual.T@seed_residual/x.shape[0]
    flat = x.reshape(-1, x.shape[-1])
    centered = flat-flat.mean(0)
    total = centered.T@centered/len(flat)
    return within, between, total


def shared_checkpoint_susceptibility(within, between, block):
    if int(block) != block or block < 1:
        raise ValueError("Block size must be a positive integer")
    return np.asarray(within)+block*np.asarray(between)
