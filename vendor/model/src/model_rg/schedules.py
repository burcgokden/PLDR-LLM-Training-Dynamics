"""Explicit reference warmup/cosine drive and controlled recipe adaptations.

The scalar schedule follows LinearWarmupCosineLRSchedule in Burc Gokden's
Apache-2.0 PLDR-LLM-Self-Organized-Criticality implementation, commit
6c9ad835f23f8557942de2c00dfe33ec8262da81. Preserved sources and the two
released-model configurations are in internal/reference-schedule.
"""
import math

import torch

from model_rg.criticality import generator_parameter


def recipe(name, heads):
    if name not in ['controlled', 'reference1', 'reference2']:
        raise ValueError('An explicitly selected schedule recipe is required')
    reference = name != 'controlled'
    peak = .0012 if name == 'reference1' else .001 if name == 'reference2' else .0003
    return dict(name=name, heads=heads, total_steps=250000,
        warmup_steps=1000 if name == 'reference2' else 2000,
        floor_fraction=.1, generator_peak=peak,
        body_peak=peak if reference else .0006/heads,
        betas=[.9,.95], epsilon=1e-5 if reference else 1e-8,
        weight_decay=.1 if reference else .01,
        clipping='value' if reference else 'norm',
        objective='all_nonpadding_targets' if reference else 'last_external_target',
        arithmetic='float32 full batch32; no TF32; per-tensor AdamW')


def multiplier(step, warmup_steps, total_steps, floor_fraction=.1):
    if not 0 < warmup_steps < total_steps or not 0 <= floor_fraction <= 1 or step < 0:
        raise ValueError('Invalid schedule domain')
    total_steps=float(total_steps);warmup_steps=float(warmup_steps)
    alpha=float(floor_fraction);step=min(float(step),total_steps)
    warmup_rise=(1/warmup_steps)*step
    decay_step=step-warmup_steps;decay_rate=total_steps-warmup_steps
    cosine_decay=(1-alpha)*.5*(1+math.cos(math.pi*(decay_step/decay_rate)))+alpha
    return warmup_rise if step <= warmup_steps else cosine_decay


def optimizer_and_scheduler(model, profile):
    named=list(model.model.named_parameters())
    groups=[dict(params=[p for n,p in named if generator_parameter(n)],lr=profile['generator_peak']),
            dict(params=[p for n,p in named if not generator_parameter(n)],lr=profile['body_peak'])]
    optimizer=torch.optim.AdamW(groups,betas=tuple(profile['betas']),eps=profile['epsilon'],
        weight_decay=profile['weight_decay'],foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:multiplier(step,
        profile['warmup_steps'],profile['total_steps'],profile['floor_fraction']))
    return optimizer,scheduler


def loss(model, crops, profile):
    if profile['objective']=='last_external_target':
        return torch.nn.functional.cross_entropy(model.logits(crops[:,:-1]),crops[:,-1])
    model.eta=None;model.capture=False;model.head_outputs={}
    logits=model.model(crops[:,:-1],use_cache=False,logits_to_keep=0).logits
    targets=crops[:,1:];mask=targets.ne(0)
    values=torch.nn.functional.cross_entropy(logits.transpose(1,2),targets,reduction='none')
    if not mask.any():raise ValueError('No nonpadding target in the selected batch')
    return (values*mask.to(values.dtype)).sum()/mask.sum()


def clip(model, profile):
    if profile['clipping']=='value':
        torch.nn.utils.clip_grad_value_(model.model.parameters(),1.,foreach=False)
    else:
        torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
