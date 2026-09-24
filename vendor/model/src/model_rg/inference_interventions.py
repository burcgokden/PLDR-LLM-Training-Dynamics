"""Reversible native operator interventions, without changing model weights."""
from contextlib import contextmanager

import torch


def operator_cache(output):
    """Native external-G layout: layer, (A, A_LM, G), batch, head, d, d."""
    return torch.stack([torch.stack([layer[j].detach().clone() for j in [0, 1, 5]])
                        for layer in output.pldr_attentions])


@contextmanager
def fixed_operators(model, cache):
    """Use the author's external-G branch with recomputed causal Q/K/V.

    The supplied operators may be individual-context tensors or a batch-one
    calibration reference. No KV cache is used. Restoring the exact original
    buffers and flags also restores ordinary full-generator inference.
    """
    decoder = model.model.decoder
    expected = (model.config.num_hidden_layers, 3, model.config.num_attention_heads,
                model.config.head_dim, model.config.head_dim)
    if cache.ndim != 6 or (cache.shape[0], cache.shape[1], *cache.shape[3:]) != expected:
        raise ValueError('Invalid native external-operator layout')
    parameter = next(model.model.parameters())
    if cache.dtype != parameter.dtype or cache.device != parameter.device:
        raise ValueError('The operator and native arithmetic conventions differ')
    if not torch.isfinite(cache).all():
        raise ValueError('Nonfinite external operator')
    changed = []

    def set_value(module, name, value):
        changed.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    try:
        set_value(decoder, 'custom_G_type', 'external')
        set_value(decoder, 'past_G_values', cache)
        set_value(decoder, 'past_G_values_status', torch.ones(cache.shape[0], dtype=torch.bool, device=cache.device))
        set_value(decoder, 'is_past_G_values_initialized', True)
        for layer in decoder.dec_layers:
            set_value(layer.mha1, 'custom_G_type', 'external')
            set_value(layer.mha1.plgatt_layer, 'custom_G_type', 'external')
        yield
    finally:
        for module, name, value in reversed(changed):
            setattr(module, name, value)


@contextmanager
def projected_rows(model):
    """Replace each emitted A by its row centroid before the native PLGA map."""
    handles = []

    def project(module, arguments):
        inputs, *rest = arguments
        query, key, value, matrix, mask = inputs
        reduced = matrix.mean(-2, keepdim=True).expand_as(matrix).contiguous()
        return ((query, key, value, reduced, mask), *rest)

    try:
        for layer in model.model.decoder.dec_layers:
            handles.append(layer.mha1.plgatt_layer.register_forward_pre_hook(project))
        yield
    finally:
        for handle in handles:
            handle.remove()
