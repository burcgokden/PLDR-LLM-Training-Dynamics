"""Read-only admission for source-disjoint single-pass native interventions.

Archived reconstruction does not grant permission to execute current workers.
"""
import json
from pathlib import Path
import numpy as np
import torch
from model_rg.provenance import sha256

NAMES = {'reference1': 14, 'subcritical1': 14, 'early-h2': 2, 'early-h8': 8}
POLICY = dict(batch=32, native_dtype='float32', tf32=False, cpu_threads=2, memory_ceiling_bytes=20*1024**3)
BRANCHES = [dict(name=n, offset=e, reset_first_moment=r) for n, e, r in [
    ('native_keep', 1e-9, False), ('raised_keep', 1e-6, False),
    ('native_reset', 1e-9, True), ('raised_reset', 1e-6, True), ('replay', 1e-9, False)]]

def require(condition, message):
    if not condition:
        raise ValueError(message)

def producer_inventory(repo):
    return ['scripts/run_potential_factorial.py', 'scripts/train_potential_avalanches.py',
            'scripts/factorial_admission.py'] + [
        str(p.relative_to(repo)) for p in sorted((repo / 'src/model_rg').glob('*.py'))]

def freeze_corpus(root):
    path = root / 'data/refinedweb-onepass-524288/tokens.npy'
    manifest_path = path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    digest = sha256(path)
    require(manifest['status'] == 'complete', 'Incomplete corpus manifest')
    require(list(array.shape) == manifest['shape'] == [524288, 513], 'Corpus shape')
    require(str(array.dtype) == manifest['dtype'] == 'int32', 'Corpus dtype')
    require(digest == manifest['tokens_sha256'], 'Corpus digest disagrees with manifest')
    return dict(path=str(path.resolve()), sha256=digest, shape=list(array.shape),
                dtype=str(array.dtype), bytes=path.stat().st_size,
                manifest=str(manifest_path.resolve()), manifest_sha256=sha256(manifest_path))

def check_corpus(spec, root):
    frozen = spec['corpus']
    path = root / 'data/refinedweb-onepass-524288/tokens.npy'
    manifest_path = path.parent / 'manifest.json'
    require(frozen['path'] == str(path.resolve()), 'Unexpected corpus path')
    require(frozen['manifest'] == str(manifest_path.resolve()), 'Unexpected corpus manifest path')
    require(sha256(manifest_path) == frozen['manifest_sha256'], 'Corpus manifest changed')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['status'] == 'complete', 'Incomplete corpus manifest')
    require(path.stat().st_size == frozen['bytes'], 'Corpus byte length changed')
    digest = sha256(path)
    require(digest == frozen['sha256'] == spec['token_sha256'] == manifest['tokens_sha256'],
            'Frozen corpus digest changed')
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    require(list(array.shape) == frozen['shape'] == manifest['shape'] == [524288, 513], 'Corpus shape changed')
    require(str(array.dtype) == frozen['dtype'] == manifest['dtype'] == 'int32', 'Corpus dtype changed')
    return array, {str(path): digest, str(manifest_path): frozen['manifest_sha256']}

def check_design(spec, case_name, profile):
    require(spec['schema'] == 'potential-factorial-v2', 'Unsupported design schema')
    require(spec.get('role') in ['scientific', 'qualification'], 'Invalid output role')
    require(spec.get('runtime_policy') == POLICY, 'Changed numerical or memory policy')
    require(spec.get('reserved_updates') == 128, 'Changed reserved source suffix')
    require(spec.get('source_policy') == 'permutation-prefix-reserved-suffix-v1', 'Changed source policy')
    require(type(spec['steps']) is int and spec['steps'] == 128, 'Scientific length must be 128')
    require(type(spec['profile_steps']) is int and spec['profile_steps'] == 64, 'Profile length must be 64')
    require(spec['branches'] == BRANCHES, 'Changed intervention or branch family')
    require(len(spec['cases']) == 4 and {c['name'] for c in spec['cases']} == set(NAMES), 'Changed case family')
    require(case_name in NAMES, 'Unknown case')
    for case in spec['cases']:
        require(case['kind'] == ('early' if case['name'].startswith('early') else 'continuation_parent'), 'Changed case kind')
    require(not profile or case_name == 'reference1', 'Profile must use reference1 at 14 heads')

def check_profile(study, spec, selected, protocol_hash):
    path = study / 'profile/reference1/manifest.json'
    require(path.is_file(), 'Completed native profile required')
    profile = json.loads(path.read_text())
    case = next(c for c in spec['cases'] if c['name'] == 'reference1')
    require(profile['status'] == 'complete' and profile['profile'] is True, 'Invalid profile role or status')
    require(profile.get('role') == 'qualification', 'Invalid profile output role')
    require(profile['steps_per_branch'] == 64, 'Noncanonical profile length')
    require(profile['case'] == case, 'Profile incoming identity changed')
    require(profile['protocol_sha256'] == protocol_hash, 'Stale profile protocol')
    require(profile['producer_sources'] == spec['producer_sources'], 'Stale profile source')
    require(0 < profile['max_cuda_memory_bytes'] <= 20 * 1024**3, 'Profile memory ceiling')
    require(profile['qualification'].get('unchanged_offset_gradient_bitwise') is True, 'Unqualified native gradient')
    require(profile['qualification'].get('unchanged_offset_forward_bitwise') is True, 'Unqualified native forward')
    require(set(profile['branches']) == {'native_keep', 'replay'}, 'Noncanonical profile branches')
    arrays = {}
    bound = {str(path): sha256(path)}
    for name, branch in profile['branches'].items():
        require(branch['steps'] == 64 and branch['offset'] == 1e-9 and branch['reset_first_moment'] is False,
                'Noncanonical profile branch')
        raw = path.parent / (name + '.npz')
        bound[str(raw)] = sha256(raw)
        require(bound[str(raw)] == branch['artifact_sha256'], 'Profile arrays changed')
        with np.load(raw, allow_pickle=False) as z:
            arrays[name] = {k: z[k] for k in ['rows', 'offsets', 'lr', 'loss', 'initial_native_logits',
                                            'initial_intervened_logits', 'final_logits', 'targets']}
        z = arrays[name]
        expected_shapes = dict(rows=(64,32), offsets=(64,32), lr=(64,2), loss=(64,),
            initial_native_logits=(64,32000), initial_intervened_logits=(64,32000),
            final_logits=(64,32000), targets=(64,))
        require(all(z[k].shape == shape for k,shape in expected_shapes.items()), 'Invalid profile array shape')
        require(all(np.isfinite(v).all() for v in z.values()), 'Invalid profile arrays')
        require(np.array_equal(z['targets'], selected['probes'][:,64]), 'Profile target cohort')
        require(np.array_equal(z['rows'], selected['reference1_rows'][:64]) and
                np.array_equal(z['offsets'], selected['reference1_offsets'][:64]), 'Profile source prefix')
        require(np.array_equal(z['initial_native_logits'], z['initial_intervened_logits']), 'Profile forward disagreement')
    for name in arrays['native_keep']:
        require(np.array_equal(arrays['native_keep'][name], arrays['replay'][name]), 'Profile replay disagreement')
    keep, replay = profile['branches']['native_keep'], profile['branches']['replay']
    require(replay.get('complete_state_replay_bitwise') is True and
            keep['final_state_sha256'] == replay['final_state_sha256'], 'Profile state replay disagreement')
    return bound

def preflight(study, case_name, profile, repo, root):
    """Return checked read-only inputs before any CUDA, model, or output construction."""
    study, repo, root = Path(study), Path(repo), Path(root)
    protocol = study / 'protocol.json'
    spec = json.loads(protocol.read_text())
    check_design(spec, case_name, profile)
    require(set(spec['producer_sources']) == set(producer_inventory(repo)), 'Incomplete producer inventory')
    bound = {str(protocol): sha256(protocol)}
    for source, digest in spec['producer_sources'].items():
        require(sha256(repo / source) == digest, 'Producer changed ' + source)
        bound[str(repo / source)] = digest
    assets = root / 'assets/PLDR-LLM-v51-SOC-110M-1'
    native = {str(assets / name) for name in ['modeling_pldrllm.py', 'configuration_pldrllm.py']}
    require(set(spec['native_sources']) == native, 'Incomplete native inventory')
    for source, digest in spec['native_sources'].items():
        require(sha256(source) == digest, 'Native source changed ' + source)
        bound[source] = digest
    selected_path = study / 'selection.npz'
    require(sha256(selected_path) == spec['selection_sha256'], 'Selection changed')
    bound[str(selected_path)] = spec['selection_sha256']
    with np.load(selected_path, allow_pickle=False) as archive:
        selected = {name: archive[name] for name in archive.files}
    expected_keys = {'probes', 'coordinates'} | {n + suffix for n in NAMES for suffix in ['_rows', '_offsets']}
    require(set(selected) == expected_keys, 'Incomplete selected input inventory')
    require(selected['probes'].shape == (64, 65), 'Changed evaluation panel')
    require(np.issubdtype(selected['probes'].dtype, np.integer) and
            np.all((selected['probes'] >= 0) & (selected['probes'] < 32000)), 'Probe token domain')
    require(selected['coordinates'].shape == (128,) and
            np.issubdtype(selected['coordinates'].dtype, np.integer) and
            len(np.unique(selected['coordinates'])) == 128 and
            np.all((selected['coordinates'] >= 0) & (selected['coordinates'] < 4096)), 'Coordinate panel domain')
    for name in NAMES:
        rows, offsets = selected[name + '_rows'], selected[name + '_offsets']
        require(rows.shape == offsets.shape == (128, 32), 'Changed batch or source length')
        require(np.issubdtype(rows.dtype, np.integer) and np.issubdtype(offsets.dtype, np.integer), 'Source index dtype')
        require(np.all((rows >= 0) & (rows < 524288)) and
                np.all((offsets >= 0) & (offsets <= 448) & (offsets % 64 == 0)), 'Source index domain')
    if not profile:
        bound.update(check_profile(study, spec, selected, bound[str(protocol)]))
    token, corpus_bound = check_corpus(spec, root)
    bound.update(corpus_bound)
    case = next(c for c in spec['cases'] if c['name'] == case_name)
    for key in ['checkpoint', 'parent_manifest', 'parent_history']:
        require(sha256(case[key]) == case[key + '_sha256'], 'Incoming state changed')
        bound[case[key]] = case[key + '_sha256']
    bound.update(check_source(case, selected, spec))
    saved = torch.load(case['checkpoint'], map_location='cpu', weights_only=True)
    job = saved['arguments'] if case['kind'] == 'continuation_parent' else saved['job']
    require(job['heads'] == NAMES[case_name], 'Noncanonical incoming width')
    role_file = study / 'ROLE.json'
    if role_file.exists():
        require(json.loads(role_file.read_text()).get('role') == spec['role'], 'Destination role conflict')
        bound[str(role_file)] = sha256(role_file)
    return spec, case, saved, selected, token, bound


def check_source(case, selected, spec):
    """Check the ordered consumed prefix and the reserved, disjoint suffix."""
    history_path = Path(case['parent_history'])
    require(sha256(history_path) == case['parent_history_sha256'], 'Parent source history changed')
    with np.load(history_path, allow_pickle=False) as old:
        prefix = (old['rows']*8 + old['offsets']//64).reshape(-1)
    start = case['start_step']
    require(prefix.size == start*32, 'Parent clock and source prefix disagree')
    name = case['name']
    suffix = (selected[name+'_rows']*8 + selected[name+'_offsets']//64).reshape(-1)
    count = (start + spec['reserved_updates'] + spec['steps'])*32
    order = np.random.default_rng(case['stream_seed']+1000).permutation(524288*8)[:count]
    require(np.array_equal(prefix, order[:start*32]), 'Parent source order changed')
    require(np.array_equal(suffix, order[(start+128)*32:]), 'Changed disjoint source suffix')
    combined = np.concatenate([prefix, suffix])
    require(np.unique(combined).size == combined.size == case['unique_combined_blocks'], 'Repeated source block')
    require(case['suffix_start_step'] == start+128, 'Source reservation clock changed')
    return {str(history_path): case['parent_history_sha256']}
