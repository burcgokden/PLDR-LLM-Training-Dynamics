"""Execute the pinned author's implementation with a narrow HF mask adapter.

The Transformers 5 API renamed input_embeds and changed mask/cache arguments.
For unpadded, uncached calls we supply the explicit triangular additive mask.
No downloaded source is edited. All PLGA, RoPE, normalization and FFN operations
are the author's modules. eta[l,h] multiplies the generated G by 1+eta[l,h],
implemented at QG before attention; downstream operators are recomputed.
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
from safetensors.torch import load_file


class _AttentionState:
    """Model-owned hook state, with no reference to the owning adapter."""


class NativeModel:
    @property
    def eta(self):
        return self._attention_state.eta

    @eta.setter
    def eta(self, value):
        self._attention_state.eta = value

    @property
    def capture(self):
        return self._attention_state.capture

    @capture.setter
    def capture(self, value):
        self._attention_state.capture = value

    @property
    def head_outputs(self):
        return self._attention_state.head_outputs

    @head_outputs.setter
    def head_outputs(self, value):
        self._attention_state.head_outputs = value

    @property
    def _attention(self):
        return self._attention_state._attention

    @_attention.setter
    def _attention(self, value):
        self._attention_state._attention = value

    def __init__(self, model_dir: str | Path, device="cuda:0"):
        self.path = Path(model_dir).resolve()
        pkgname = "pinned_pldr_" + uuid.uuid4().hex
        package = types.ModuleType(pkgname)
        package.__path__ = [str(self.path)]
        sys.modules[pkgname] = package
        config_module = importlib.import_module(pkgname + ".configuration_pldrllm")
        self.module = importlib.import_module(pkgname + ".modeling_pldrllm")
        config = config_module.PldrllmConfig.from_json_file(str(self.path / "config.json"))
        config._attn_implementation = "eager"
        self.model = self.module.PldrllmForCausalLM(config)
        self.model.load_state_dict(load_file(str(self.path / "model.safetensors")), strict=True)
        self.model.to(device).eval().requires_grad_(False)
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
                raise ValueError("This adapter only supports uncached calls")
            am = kwargs.get("attention_mask")
            if am is not None and not bool((am == 1).all()):
                raise ValueError("This adapter only supports unpadded calls")
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
                factors = eta[module.layer_idx][None, :, None, None] if eta.ndim == 2 else eta[:, module.layer_idx, :, None, None]
                query = query * (1 + factors)
            out, weights = state._attention(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            if state.capture:
                state.head_outputs[module.layer_idx] = out[:, -1].square().mean(-1).sqrt()
            return out, weights

        self.module.create_causal_mask = mask_adapter
        self.module.eager_attention_forward = attention
        rng = np.random.default_rng(1907)
        self.hidden_projection = torch.tensor(rng.normal(size=(config.hidden_size, 4)) / math.sqrt(config.hidden_size), dtype=torch.float32, device=device)
        self.logit_projection = torch.tensor(rng.normal(size=(config.vocab_size, 8)) / math.sqrt(config.vocab_size), dtype=torch.float32, device=device)
        self.logit_projection -= self.logit_projection.mean(0, keepdim=True)

    def forward(self, ids, eta=None, *, capture=False):
        self.eta = eta
        self.capture = capture
        self.head_outputs = {}
        return self.model(ids, use_cache=False, logits_to_keep=1,
                          output_pldr_attentions=capture, output_hidden_states=capture)

    def logits(self, ids, eta=None):
        return self.forward(ids, eta).logits[:, -1]

    def features(self, ids, targets):
        out = self.forward(ids, capture=True)
        logp = out.logits[:, -1].log_softmax(-1)
        p = logp.exp()
        entropies, headnorms, gnorms, acontrasts = [], [], [], []
        for l, att in enumerate(out.pldr_attentions):
            a, _, _, _, _, g, weights = att
            last = weights[:, :, -1]
            entropies.append(-(last * last.clamp_min(1e-38).log()).sum(-1))
            headnorms.append(self.head_outputs[l])
            gnorms.append(g.double().square().mean((-2, -1)).sqrt())
            centered = a.double() - a.double().mean(-2, keepdim=True)
            acontrasts.append(centered.square().mean((-2, -1)))
        hidden = [x[:, -1] @ self.hidden_projection for x in out.hidden_states]
        nll = -logp.gather(1, targets[:, None]).squeeze(1)
        entropy = -(p * logp).sum(-1)
        features = torch.cat([torch.cat(entropies, 1), torch.cat(headnorms, 1),
                              torch.cat(hidden, 1), out.logits[:, -1] @ self.logit_projection,
                              entropy[:, None], nll[:, None]], 1)
        diagnostics = {"g_rms": torch.stack(gnorms, 1), "a_row_energy": torch.stack(acontrasts, 1)}
        return features, diagnostics

    def feature_names(self):
        names = []
        for kind in ("attention_entropy", "head_output_rms"):
            names += [f"{kind}/layer{l}/head{h}" for l in range(self.config.num_hidden_layers) for h in range(self.config.num_attention_heads)]
        names += [f"hidden_projection/{stage}/p{k}" for stage in ["scaled_embedding"] + [f"layer{l}" for l in range(self.config.num_hidden_layers)] for k in range(4)]
        names += [f"logit_projection/p{k}" for k in range(8)]
        return names + ["predictive_entropy", "next_token_nll"]
