"""Independent reconstruction of the physical study's declared numerical design.

No producer validator is imported here. Source identity and native execution
are separate from this scientific-design comparison.
"""
import json
import math


def check(spec, archived=False):
    def demand(c, m):
        if not c: raise ValueError('Independent physical design: ' + m)
    stage = spec['stage']
    demand(stage in ['qualification', 'development', 'confirmation'], 'stage')
    steps = {'qualification': 12, 'development': 2048, 'confirmation': 16384}[stage]
    checkpoints = {'qualification': [0, 12], 'development': [0, 512, 2048],
                   'confirmation': [0, 512, 2048, 16384]}[stage]
    if stage == 'confirmation':
        grid = [(h, s, 'full') for h in [4, 8] for s in range(640101, 640104)]
        grid += [(4, 640101, a) for a in ['frozen-generator', 'shuffled']]
    elif stage == 'qualification':
        grid = [(h, 640101, a) for h in [4, 8] for a in ['full', 'frozen-generator', 'shuffled']]
    else:
        grid = [(h, 640101, 'full') for h in [4, 8]]
    demand(type(spec['steps']) is int and spec['steps'] == steps, 'steps')
    demand(spec['checkpoints'] == checkpoints, 'checkpoints')
    demand(type(spec['batch_size']) is int and spec['batch_size'] == 32, 'batch')
    demand([(c['heads'], c['seed'], c['arm']) for c in spec['cases']] == grid, 'case grid')
    for c in spec['cases']:
        demand(c['name'] == f"h{c['heads']}-s{c['seed']}-{c['arm']}", 'name')
    demand(spec['learning_rate'] == .0003, 'rate')
    o = spec['optimizer']
    demand(o['name'] == 'AdamW' and o['betas'] == [.9, .95] and o['epsilon'] == 1e-8
           and o['weight_decay'] == .01 and o['gradient_clip_norm'] == 1., 'AdamW')
    rt = spec['runtime']
    demand(rt['dtype'] == 'float32' and rt['tf32'] is False and rt['threads'] == 4, 'arithmetic')
    if archived:
        demand(spec['schema'] == 'physical-native-study-v1', 'archive schema')
        demand(spec['schedule'] == '128-update linear warmup, cosine decay to one fifth of peak; physical-update clock', 'archive schedule')
    else:
        demand(spec['schema'] == 'physical-native-study-v2', 'schema')
        demand(spec['schedule'] == dict(name='linear-cosine',warmup=128,floor=.2,
               clock='zero-based update before optimizer.step'), 'schedule')
        demand(set(o) == {'name', 'betas', 'epsilon', 'weight_decay', 'gradient_clip_norm', 'memory'}, 'optimizer fields')
        demand(o['memory'] == 'zero at physical origin', 'moment origin')
    return steps


def expected_rate(index, horizon):
    return .0003 * ((index + 1) / 128 if index < 128 else
                   .2 + .4 * (1 + math.cos(math.pi * (index - 128) / max(1, horizon - 128))))
