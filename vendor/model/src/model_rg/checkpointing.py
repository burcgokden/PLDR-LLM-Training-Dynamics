"""Full-batch decoder recomputation for the pinned native PLDR graph."""
import torch
from torch.utils.checkpoint import checkpoint


def enable_decoder_checkpointing(model):
    """Recompute complete decoder layers without splitting the native batch.

    Non-reentrant checkpointing preserves keyword arguments and structured
    outputs. Inference and all no-gradient observations use the original call.
    No module parameters, optimizer groups, or state-dict keys are replaced.
    """
    count = 0
    for module in model.model.modules():
        if not isinstance(module, model.module.PLDR_DecoderLayer):
            continue
        if hasattr(module, '_modelrg_original_forward'):
            raise ValueError('Decoder checkpointing was already enabled')
        original = module.forward
        module._modelrg_original_forward = original

        def forward(*args, _original=original, _module=module, **kwargs):
            if _module.training and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        module.forward = forward
        count += 1
    if count != model.config.num_hidden_layers:
        raise AssertionError('Native decoder inventory changed')
    return count
