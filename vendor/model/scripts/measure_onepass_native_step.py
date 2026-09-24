#!/usr/bin/env python
"""Measure complete native steps on unrevealed single-pass source blocks."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.criticality import generator_parameter, predictive_kl
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.schedules import loss as recipe_loss, clip as recipe_clip


FIELD_NAMES = ['row_energy', 'matrix_energy', 'row_ratio', 'centroid_squared_rms',
               'operator_rms', 'attention_entropy', 'head_rms']


def observe_chunk(model, crops):
    """Native float32 forward; fixed-unit observations accumulated in float64."""
    model.model.eval()
    batch = torch.as_tensor(crops, dtype=torch.long)
    with torch.no_grad():
        output = model.forward(batch[:, :64], capture=True)
        logits = output.logits[:, -1].double()
        logp = logits.log_softmax(-1)
        matrices = torch.stack([x[0].double() for x in output.pldr_attentions], 1)
        operators = torch.stack([x[5].double() for x in output.pldr_attentions], 1)
        attention = torch.stack([x[6][:, :, -1].double() for x in output.pldr_attentions], 1)
        centroid = matrices.mean(-2)
        energy = (matrices-centroid[..., None, :]).square().mean((-2, -1))
        total = matrices.square().mean((-2, -1))
        fields = torch.stack([energy, total, energy/total.clamp_min(1e-30),
            centroid.square().mean(-1), operators.square().mean((-2, -1)).sqrt(),
            -(attention*attention.clamp_min(1e-300).log()).sum(-1),
            torch.stack([model.head_outputs[k].double() for k in range(5)], 1)], -1)
        hidden = torch.stack([x[:, -1].double().square().mean(-1).sqrt()
                              for x in output.hidden_states], 1)
        result = dict(fields=fields, centroids=centroid, hidden_rms=hidden, logits=logits,
            nll=-logp.gather(1, batch[:, 64, None]).squeeze(1),
            prediction_entropy=-(logp.exp()*logp).sum(-1))
    return {name:value.numpy().copy() for name, value in result.items()}


def observe(model, crops):
    chunks=[observe_chunk(model,crops[begin:begin+32]) for begin in range(0,len(crops),32)]
    return {name:np.concatenate([value[name] for value in chunks]) for name in chunks[0]}


def state_digests(model, optimizer):
    """Every named parameter and Adam tensor, with explicit scalar/group records."""
    parameters, states = {}, {}
    for name, parameter in model.model.named_parameters():
        value = parameter.detach().cpu().contiguous().numpy()
        parameters[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(value.tobytes()).hexdigest())
        states[name] = {}
        for key, entry in optimizer.state[parameter].items():
            if isinstance(entry, torch.Tensor):
                value = entry.detach().cpu().contiguous().numpy()
                states[name][key] = dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
            else:
                states[name][key] = entry
    groups = [{k:v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups]
    return dict(parameters=parameters, optimizer_states=states, optimizer_groups=groups)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-feasible-20260908')
    p.add_argument('--protocol', default='onepass-native-step-selection.json')
    p.add_argument('--case', required=True)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    root = Path(a.root);study = root/a.study
    protocol = study/'protocols'/a.protocol;spec = json.loads(protocol.read_text())
    case = next(c for c in spec['cases'] if c['name'] == a.case)
    repo=Path(__file__).resolve().parents[1]
    for name,digest in spec['producer_sources'].items():
        if sha256(repo/name)!=digest:raise AssertionError('A selected conditional-step producer changed')
    for name,digest in spec['inputs_sha256'].items():
        if sha256(name)!=digest:raise AssertionError('A selected conditional-step input changed')
    out = study/'measurements'/case['name'];out.mkdir(parents=True, exist_ok=False)
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    data = root/'data/refinedweb-onepass-524288';probe = root/'controlled-study-20260905/data/short'
    if sha256(case['parent']) != case['parent_sha256']:
        raise AssertionError('Incoming complete training state changed')
    bind_run(out, [protocol, Path(case['parent']), Path(case['parent_manifest']),
        Path(case['training_verification']),data/'manifest.json',data/'tokens.npy', probe/'tokens.npy', probe/'offsets.npy',
        source/'modeling_pldrllm.py', source/'configuration_pldrllm.py'], vars(a))
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    parent = torch.load(case['parent'], map_location='cpu', mmap=True, weights_only=True)
    condition = parent['arguments']
    if (condition['heads'], condition['seed'], parent['step']) != (case['heads'], case['seed'], case['step']):
        raise AssertionError('Incoming state identity changed')
    proof=json.loads(Path(case['training_verification']).read_text())
    if proof['status']!='complete' or proof['verified_files'][str(Path(case['parent']))]!=sha256(case['parent']):
        raise AssertionError('The incoming state lacks complete independent native reconstruction')
    profile=parent['recipe']
    model = TrainingModel(source, case['heads'], case['seed'], 'cpu')
    model.model.load_state_dict(parent['model'])
    optimizer,scheduler=optimizer_and_scheduler(model,profile)
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    scheduler.load_state_dict(copy.deepcopy(parent['scheduler']))
    names = dict(model.model.named_parameters())
    generator_names = [name for name in names if generator_parameter(name)]
    body_names = [name for name in names if not generator_parameter(name)]
    if not generator_names or not body_names:raise AssertionError('Both weight partitions must be present')
    total_blocks=4194304;consumed=32*case['step']
    if consumed>=total_blocks:raise AssertionError('No remaining source blocks at the selected state')
    order=np.random.default_rng(condition['stream_seed']+1000).permutation(total_blocks)
    remaining=order[consumed:];rng=np.random.default_rng(spec['batch_seed'])
    block_ids=np.stack([remaining[rng.choice(len(remaining),32,replace=False)] for _ in range(spec['batch_replicas'])])
    rows=block_ids//8;offsets=64*(block_ids%8)
    source_tokens=np.load(data/'tokens.npy',mmap_mode='r')
    batches=source_tokens[rows[...,None],offsets[...,None]+np.arange(65)]
    tokens, probe_offsets = np.load(probe/'tokens.npy'), np.load(probe/'offsets.npy')
    cohort = np.arange(512, 1024)
    crops = tokens[cohort[:, None], probe_offsets[cohort, None]+np.arange(65)]
    base = observe(model, crops)
    raw = dict(rows=rows, offsets=offsets, cohort=cohort,block_ids=block_ids,training_crops=batches)
    raw.update({'base_'+key:value for key, value in base.items()})
    branch_values = {name:[] for name in ['generator', 'body', 'full']}
    losses, norms, displacement_norms, target_counts, all_digests = [], [], [], [], []
    for k, batch_array in enumerate(batches):
        # Fresh batches are conditional draws at exactly the same complete state.
        model.model.load_state_dict(parent['model'])
        optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        scheduler.load_state_dict(copy.deepcopy(parent['scheduler']))
        model.model.train();optimizer.zero_grad(set_to_none=True)
        batch = torch.as_tensor(batch_array, dtype=torch.long)
        loss=recipe_loss(model,batch,profile)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native training loss')
        loss.backward()
        group_norms=[torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in g['params'] if p.grad is not None])) for g in optimizer.param_groups]
        norm=torch.linalg.vector_norm(torch.stack(group_norms))
        recipe_clip(model,profile)
        optimizer.step();scheduler.step()
        target_counts.append(32 if profile['objective']=='last_external_target' else int(batch[:,1:].ne(0).sum()))
        losses.append(float(loss.detach()));norms.append(float(norm))
        squared = [sum(float((names[name].detach().double()-parent['model'][name].double()).square().sum())
                       for name in group) for group in [generator_names, body_names]]
        displacement_norms.append(squared)
        digests=state_digests(model,optimizer);digests['scheduler']=copy.deepcopy(scheduler.state_dict())
        all_digests.append(digests)
        if k == 0:write_json(out/'first-step-digests.json',digests)
        branch_values['full'].append(observe(model, crops))
        # The two partial corners use the displacements from this same globally
        # clipped step. They are interventions on weights, not separate optimizers.
        generator_after = {name:names[name].detach().clone() for name in generator_names}
        with torch.no_grad():
            for name in generator_names:names[name].copy_(parent['model'][name])
        branch_values['body'].append(observe(model, crops))
        with torch.no_grad():
            for name in body_names:names[name].copy_(parent['model'][name])
            for name in generator_names:names[name].copy_(generator_after[name])
        branch_values['generator'].append(observe(model, crops))
        print(case['name'], 'native conditional batch', k+1, 'of', len(batches),
              'seconds', round(time.time()-started, 1), flush=True)
    for branch, values in branch_values.items():
        for key in base:raw[branch+'_'+key] = np.stack([value[key] for value in values])
        raw[branch+'_predictive_kl'] = np.stack([
            predictive_kl(torch.from_numpy(base['logits']), torch.from_numpy(value['logits'])).numpy()
            for value in values])
    raw.update(losses=np.array(losses), gradient_norms=np.array(norms),
               displacement_squared_norms=np.array(displacement_norms),supervised_targets=np.array(target_counts))
    if any(not np.isfinite(value).all() for value in raw.values()):
        raise FloatingPointError('Nonfinite native step observation')
    # A repeat forward at the restored parent checks intervention cleanup.
    model.model.load_state_dict(parent['model'])
    restored = observe(model, crops)
    for key in base:
        np.testing.assert_array_equal(base[key], restored[key])
        if (base[key].shape != restored[key].shape or base[key].dtype != restored[key].dtype
                or base[key].tobytes() != restored[key].tobytes()):
            raise AssertionError('Restored parent emission bytes differ: '+key)
    write_json(out/'all-step-digests.json',all_digests)
    np.savez_compressed(out/'measurements.npz', **raw)
    risk_drift={}
    for branch in ['generator','body','full']:
        changes=(raw[branch+'_nll']-raw['base_nll'][None,:]).mean(1)
        kls=raw[branch+'_predictive_kl'].mean(1)
        risk_drift[branch]=dict(risk_changes=changes.tolist(),mean_risk_change=float(changes.mean()),
            conditional_standard_error=float(changes.std(ddof=1)/np.sqrt(len(changes))) if len(changes)>1 else None,
            predictive_kl_means=kls.tolist(),mean_predictive_kl=float(kls.mean()))
    changes=np.array(risk_drift['full']['risk_changes'])-np.array(risk_drift['generator']['risk_changes'])-np.array(risk_drift['body']['risk_changes'])
    risk_drift['interaction']=dict(risk_changes=changes.tolist(),mean_risk_change=float(changes.mean()),
        conditional_standard_error=float(changes.std(ddof=1)/np.sqrt(len(changes))) if len(changes)>1 else None)
    result = dict(schema='onepass-native-conditional-step-corners-v1', status='complete', case=case,
        batch_seed=spec['batch_seed'], batch_replicas=spec['batch_replicas'],
        remaining_blocks=total_blocks-consumed,profile=profile,probe_contexts=512,risk_drift=risk_drift,
        conditional_law=spec['conditional_law'],
        branches=['generator', 'body', 'full'], field_names=FIELD_NAMES,
        generator_parameter_names=generator_names, body_parameter_names=body_names,
        generator_dimension=sum(names[name].numel() for name in generator_names),
        body_dimension=sum(names[name].numel() for name in body_names),
        restored_parent_observations_bitwise_equal=True,
        started_at=datetime.fromtimestamp(started, timezone.utc).isoformat(),
        seconds=time.time()-started,
        scope='Fresh full batch32 CPU float32 gradients from the unconsumed source-block population, with the complete saved AdamW state and actual next schedule rate. Every conditional branch uses distinct source positions and restores the incoming state; the independent branches do not form a repeated training trajectory. The two partial weight corners use displacements from the same fully clipped native step. All512heldoutcontexts use native batch32 forwards. This conditional unrevealed-data law differs from conditioning on the complete realized future stream. These are neither additional training initializations nor bytewise GPU-step reconstructions.')
    write_json(out/'results.json', result)
    write_json(out/'manifest.json', dict(schema=result['schema'], status='complete', case=case,
        binding_sha256=sha256(out/'binding.json'), raw_sha256=sha256(out/'measurements.npz'),
        first_step_digests_sha256=sha256(out/'first-step-digests.json'),
        all_step_digests_sha256=sha256(out/'all-step-digests.json'),
        results_sha256=sha256(out/'results.json')))


if __name__ == '__main__':main()
