"""Paper-specified subcritical recipes alongside the two released near-critical recipes.

The labels describe Table1 of arXiv:2603.23539v1. They are hypotheses about
the controlled adaptation, not classifications inferred from its outcomes.
"""
from model_rg.schedules import recipe as released_recipe


RECIPE_NAMES=['controlled','reference1','reference2','subcritical1','subcritical2']


def recipe(name,heads):
    if name in ['controlled','reference1','reference2']:
        return released_recipe(name,heads)
    if name not in ['subcritical1','subcritical2']:
        raise ValueError('An explicitly selected reference regime recipe is required')
    profile=released_recipe('reference1',heads)
    peak=.0006 if name=='subcritical1' else .0003
    profile.update(name=name,warmup_steps=6000 if name=='subcritical1' else 2000,
                   generator_peak=peak,body_peak=peak)
    return profile
