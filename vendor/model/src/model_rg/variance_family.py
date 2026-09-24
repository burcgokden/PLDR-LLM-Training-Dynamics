"""Shape-aware initialization with a nonvanishing per-head reference law.

This is an initialization normalization, not a claimed infinite-width Adam law.
The native program initializes Linear weights with Xavier uniform and treats
the head axis of each rank-three PLGA parameter as a fan dimension.
"""
import math
import torch
from model_rg.criticality import generator_parameter


def linear_factor(fan_in, fan_out, reference_in, reference_out):
    """Map 2/(fan_in+fan_out) to a reference gain divided by fan_in."""
    if min(fan_in, fan_out, reference_in, reference_out) <= 0:
        raise ValueError('Fan dimensions must be positive')
    return math.sqrt(reference_in * (fan_in + fan_out) /
                     (fan_in * (reference_in + reference_out)))


def reference_dimensions(name, fan_in, fan_out, width, ffn, vocabulary):
    """Resolve architectural roles even when residual width equals vocabulary size."""
    if name == 'final_layer':
        if (fan_in, fan_out) != (width, vocabulary):
            raise ValueError('Unexpected vocabulary readout shape')
        return 128, vocabulary
    if (fan_in, fan_out) == (width, width):
        return 128, 128
    if (fan_in, fan_out) == (width, ffn):
        return 128, 341
    if (fan_in, fan_out) == (ffn, width):
        return 341, 128
    raise ValueError(f'Unrecognized wide linear shape: {name}')


def normalize_variance_initialization(model):
    width = model.config.hidden_size
    ffn = model.config.intermediate_size
    with torch.no_grad():
        for name, module in model.model.named_modules():
            if isinstance(module, torch.nn.Linear) and not generator_parameter(name):
                reference_in, reference_out = reference_dimensions(name, module.in_features, module.out_features,
                                                                    width, ffn, model.config.vocab_size)
                module.weight.mul_(linear_factor(module.in_features, module.out_features, reference_in, reference_out))
        for name, parameter in model.model.named_parameters():
            if 'plgatt_layer' in name and name.rsplit('.', 1)[-1] in ('Wlst', 'pwlst', 'alst'):
                heads, rows, cols = parameter.shape
                if rows != 64 or cols != 64 or heads != model.config.num_attention_heads:
                    raise ValueError(f'Unexpected PLGA tensor shape: {name}')
                parameter.mul_(math.sqrt((64 + heads) / 66))
