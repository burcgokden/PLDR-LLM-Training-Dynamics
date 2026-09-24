"""Dtype-preserving controls for the pinned native mathematical program."""
import torch


def preserve_response_dtype(model):
    def attention(module, query, key, value, attention_mask, scaling, dropout=0., **kwargs):
        weights = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = weights.softmax(-1)
        return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights
    model._attention = attention
    for module in model.model.modules():
        if module.__class__.__name__ == 'RotaryPositionalEmbeddings':
            def rotary(x, *, input_pos=None, mod=module):
                seq = x.size(1)
                cache = mod.cache[:seq] if input_pos is None else mod.cache[input_pos]
                shaped = x.reshape(*x.shape[:-1], -1, 2)
                cache = cache.view(-1, seq, 1, shaped.size(3), 2).to(x.dtype)
                value = torch.stack([shaped[..., 0] * cache[..., 0] - shaped[..., 1] * cache[..., 1],
                                     shaped[..., 1] * cache[..., 0] + shaped[..., 0] * cache[..., 1]], -1)
                return value.flatten(3)
            module.forward = rotary


def expand_smooth_response(model):
    model.module.iSwiGLU = lambda x: x * x * torch.sigmoid(x)
    for module in model.model.modules():
        if module.__class__.__name__ in ['SiLUActivation', 'SiLU']:
            module.forward = lambda x: x * torch.sigmoid(x)
        elif module.__class__.__name__ == 'GLUVariant':
            if module.activation is torch.nn.functional.silu:
                module.activation = lambda x: x * torch.sigmoid(x)
            elif module.activation.__class__.__name__ not in ['SiLUActivation', 'SiLU']:
                raise ValueError('The precision control requires the recorded SiLU activation')
            # A registered activation module is expanded by its own module visit.
        elif isinstance(module, torch.nn.LayerNorm):
            def layernorm(x, mod=module):
                axes = tuple(range(x.ndim - len(mod.normalized_shape), x.ndim))
                centered = x - x.mean(axes, keepdim=True)
                result = centered * torch.rsqrt(centered.square().mean(axes, keepdim=True) + mod.eps)
                if mod.weight is not None:
                    result = result * mod.weight
                if mod.bias is not None:
                    result = result + mod.bias
                return result
            module.forward = layernorm
