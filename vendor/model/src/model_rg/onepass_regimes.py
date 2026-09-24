"""Explicit drives and a prefix-preserving, nonrepeated RefinedWeb block stream."""
import numpy as np
import torch

from model_rg.criticality import generator_parameter
from model_rg.scheduled_regimes import recipe as reference_recipe, RECIPE_NAMES
from model_rg.schedules import multiplier as cosine_multiplier


DOCUMENTS = 524288
BLOCKS_PER_DOCUMENT = 8
MAXIMUM_UPDATES = DOCUMENTS*BLOCKS_PER_DOCUMENT//32


def recipe(name, heads, total_steps, warmup_override=-1):
    profile = reference_recipe(name, heads)
    if total_steps == 0:
        if name != 'controlled' or warmup_override != -1:
            raise ValueError('Only the selected controlled constant-rate law is supported')
        profile.update(total_steps=0, warmup_steps=0, floor_fraction=1.)
    elif total_steps == 768 and warmup_override == 128:
        profile.update(total_steps=768, warmup_steps=128)
    elif total_steps in (32768, 250000) and warmup_override == -1:
        profile['total_steps'] = total_steps
    else:
        raise ValueError('An explicitly selected scientific or qualification drive is required')
    return profile


def multiplier(step, warmup_steps, total_steps, floor_fraction=.1):
    if total_steps == 0 and warmup_steps == 0 and floor_fraction == 1. and step >= 0:
        return 1.
    return cosine_multiplier(step, warmup_steps, total_steps, floor_fraction)


def optimizer_and_scheduler(model, profile):
    named = list(model.model.named_parameters())
    groups = [dict(params=[p for n,p in named if generator_parameter(n)], lr=profile['generator_peak']),
              dict(params=[p for n,p in named if not generator_parameter(n)], lr=profile['body_peak'])]
    optimizer = torch.optim.AdamW(groups, betas=tuple(profile['betas']), eps=profile['epsilon'],
                                  weight_decay=profile['weight_decay'], foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: multiplier(
        step, profile['warmup_steps'], profile['total_steps'], profile['floor_fraction']))
    return optimizer, scheduler


def sample_batches(tokens, seed, steps):
    if tokens.shape != (DOCUMENTS,513) or tokens.dtype != np.int32:
        raise ValueError('The verified 524288-document token corpus is required')
    if not 1 <= steps <= MAXIMUM_UPDATES:
        raise ValueError('The requested horizon exceeds the nonrepeated stream; prepare new data')
    # Permute the whole fixed block population before selecting a horizon prefix.
    blocks = np.random.default_rng(seed+1000).permutation(DOCUMENTS*BLOCKS_PER_DOCUMENT)[:steps*32]
    rows = (blocks//BLOCKS_PER_DOCUMENT).reshape(steps,32)
    offsets = (64*(blocks%BLOCKS_PER_DOCUMENT)).reshape(steps,32)
    batches = tokens[rows[:,:,None], offsets[:,:,None]+np.arange(65)]
    return rows, offsets, batches
