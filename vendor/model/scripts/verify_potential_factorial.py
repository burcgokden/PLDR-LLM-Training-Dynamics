#!/usr/bin/env python3
"""Independent raw reduction of the completed paired native mechanism assay.

Imports no producer observer or analysis reducer. Full-state bitwise equality
is a source-bound producer digest check; saved arrays are compared directly.
"""
from companion_paths import configured_path
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from factorial_admission import check_design,check_corpus,check_profile,check_source,producer_inventory,freeze_corpus
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(configured_path('data:model'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def main(study,out,executed_source=None):
    study,out=Path(study),Path(out)
    if out.exists():raise FileExistsError('Use a fresh verification destination')
    protocol=study/'protocol.json';s=read(protocol)
    legacy=s['schema']=='potential-factorial-v1'
    source=Path(executed_source) if executed_source else (REPO/'evidence/executed/potential-factorial' if legacy else REPO)
    if legacy:
        if sha(source/'protocol.json')!=sha(protocol):raise ValueError('Unbound archived producer protocol')
        corpus=freeze_corpus(ROOT)
        if corpus['sha256']!=s['token_sha256']:raise ValueError('Frozen corpus digest changed')
        corpus_bound={corpus['path']:corpus['sha256'],corpus['manifest']:corpus['manifest_sha256']}
        corpus_scope='Token SHA-256 was frozen at execution. Shape, dtype, size and manifest digest are additional presently verified inputs; no retrospective manifest freezing is asserted.'
        inventory=['scripts/run_potential_factorial.py','scripts/train_potential_avalanches.py']+[str(p.relative_to(source)) for p in sorted((source/'src/model_rg').glob('*.py'))]
    else:
        check_design(s,'reference1',False)
        _,corpus_bound=check_corpus(s,ROOT);corpus=s['corpus']
        corpus_scope='Token and manifest bytes, array layout and selection were frozen before execution and independently checked again.'
        inventory=producer_inventory(source)
    if set(s['producer_sources'])!=set(inventory):raise ValueError('Incomplete executing-source inventory')
    summary=read(study/'analysis/factorial-summary.json')
    names={'reference1':14,'subcritical1':14,'early-h2':2,'early-h8':8}
    if s['steps']!=128 or s['profile_steps']!=64 or len(s['cases'])!=4 or {c['name'] for c in s['cases']}!=set(names):raise ValueError('Changed fixed family')
    bound={str(protocol):sha(protocol),**corpus_bound,str(study/'selection.npz'):sha(study/'selection.npz')}
    if legacy:bound[str(source/'protocol.json')]=sha(source/'protocol.json')
    for f,h in s['producer_sources'].items():
        if sha(source/f)!=h:raise ValueError('Changed executing source '+f)
        bound[str(source/f)]=h
    for f,h in s['native_sources'].items():
        if sha(f)!=h:raise ValueError('Changed native source')
        bound[f]=h
    if sha(study/'selection.npz')!=s['selection_sha256']:raise ValueError('Changed selected source')
    selected=np.load(study/'selection.npz');claims={c['case']['name']:c for c in summary['cases']};checks=[]
    if set(claims)!=set(names):raise ValueError('Incomplete summary family')
    if not legacy:bound.update(check_profile(study,s,selected,sha(protocol)))
    vocabulary_coordinates=0;largest_error=0.
    for case in s['cases']:
        name=case['name'];h=names[name];path=study/'runs'/name;m=read(path/'manifest.json');rows=selected[name+'_rows'];offsets=selected[name+'_offsets']
        parent=Path(case['checkpoint']).parent
        if sha(case['checkpoint'])!=case['checkpoint_sha256'] or sha(case['parent_manifest'])!=case['parent_manifest_sha256']:raise ValueError('Changed complete incoming state')
        history=parent/('activity.npz' if case['kind']=='early' else 'sampling.npz')
        old=np.load(history)
        for input_path in [Path(case['checkpoint']),Path(case['parent_manifest']),history]:bound[str(input_path)]=sha(input_path)
        combined=np.concatenate([(old['rows']*8+old['offsets']//64).reshape(-1),(rows*8+offsets//64).reshape(-1)])
        if np.unique(combined).size!=combined.size or combined.size!=case['unique_combined_blocks']:raise ValueError('Repeated block across parent and continuation')
        if legacy:
            order=np.random.default_rng(case['stream_seed']+1000).permutation(524288*8)[:combined.size]
            if not np.array_equal(order,combined):raise ValueError('Incorrect consuming-source law')
        else:
            bound.update(check_source(case,selected,s))
            if m.get('role')!=s['role']:raise ValueError('Result role differs from frozen protocol')
            if m['producer_sources']!=s['producer_sources']:raise ValueError('Result source differs from frozen protocol')
        if m['status']!='complete' or m['profile'] is not False or m['case']!=case or m['producer_sources']!=s['producer_sources'] or m['protocol_sha256']!=sha(protocol) or m['steps_per_branch']!=128:raise ValueError('Incomplete or unbound case')
        if set(m['branches'])!={'native_keep','native_reset','raised_keep','raised_reset','replay'}:raise ValueError('Incomplete factorial')
        raw={}
        definitions={b['name']:b for b in s['branches']}
        for branch,b in m['branches'].items():
            if b['steps']!=128 or any(b[k]!=definitions[branch][k] for k in ['offset','reset_first_moment']):raise ValueError('Changed executed branch')
            p=path/(branch+'.npz');bound[str(p)]=sha(p)
            if bound[str(p)]!=b['artifact_sha256']:raise ValueError('Changed branch')
            z=np.load(p);raw[branch]=z
            if not np.array_equal(rows,z['rows']) or not np.array_equal(offsets,z['offsets']):raise ValueError('Unmatched source suffix')
            if len(z['loss'])!=128 or not np.isfinite(z['loss']).all():raise ValueError('Invalid gradient-forced training record')
            if not np.array_equal(z['targets'],selected['probes'][:,64]) or z['final_logits'].shape!=(64,32000):raise ValueError('Changed full-vocabulary evaluation cohort')
            if branch=='replay':continue
            if z['fields'].shape!=(129,2,5,h,14) or z['step_logits'].shape!=(129,2,32000):raise ValueError('Observation axes changed')
            logits=z['step_logits'].astype(float);logp=logits-logsumexp(logits,axis=-1,keepdims=True)
            kk=np.sum(np.exp(logp[:-1])*(logp[:-1]-logp[1:]),axis=-1);vocabulary_coordinates+=int(logits.size)
            np.testing.assert_allclose(kk,z['predictive_step_kl'][1:],rtol=2e-7,atol=3e-13)
            q=np.sqrt(np.mean(z['fields'][1:,0,...,0]**2,axis=(1,2))).sum()
            g=np.sqrt(np.mean(z['tensor_statistics'][1:,0,...,4,1]**2,axis=(1,2))).sum()
            final=z['final_logits'].astype(float);nn=logsumexp(final,axis=-1)-final[np.arange(64),z['targets']]
            c=claims[name]['branches'][branch]
            for key,value in [('integrated_activity',q),('integrated_curvature',g),('summed_step_kl',kk[:,0].sum()),('nll_final',nn.mean())]:
                largest_error=max(largest_error,abs(float(value)-c[key]))
                if not np.isclose(value,c[key],rtol=2e-11,atol=2e-13):raise ValueError('Independent metric mismatch '+key)
            sym=z['symmetric'][1:];energy=np.mean(z['fields'][1:,...,0]**2,axis=(2,3))
            recovered=np.mean(sym[...,0]+sym[...,1]+sym[...,2],axis=(2,3))
            np.testing.assert_allclose(energy,recovered,atol=2e-14,rtol=2e-11)
            v=sym[:,0,...,1].sum();u=sym[:,0,...,0].sum()
            if not np.isclose(c['symmetric_base_fraction'],v/(u+v),rtol=2e-13):raise ValueError('Incorrect base denominator')
        for branch,z in raw.items():
            if not np.array_equal(z['initial_native_logits'],raw['native_keep']['initial_native_logits']):raise ValueError('Different incoming forward state')
            if not np.array_equal(z['lr'],raw['native_keep']['lr']):raise ValueError('Unmatched applied rates')
        for suffix in ['native','raised']:
            if not np.array_equal(raw[suffix+'_keep']['initial_intervened_logits'],raw[suffix+'_reset']['initial_intervened_logits']):raise ValueError('Moment reset changed initial emission')
        for field in ['rows','offsets','lr','loss','final_logits']:
            if not np.array_equal(raw['native_keep'][field],raw['replay'][field]):raise ValueError('Replay array mismatch')
        if m['branches']['native_keep']['final_state_sha256']!=m['branches']['replay']['final_state_sha256']:raise ValueError('Complete-state replay mismatch')
        for metric,entry in claims[name]['effects'].items():
            y={n:c[metric] for n,c in claims[name]['branches'].items()}
            independent=y['raised_reset']-y['raised_keep']-y['native_reset']+y['native_keep']
            if not np.isclose(entry['interaction'],independent,rtol=1e-13,atol=1e-14):raise ValueError('Factorial interaction mismatch')
        checks.append(dict(case=name,source_prefix_verified=True,unique_combined_blocks=int(combined.size),raw_replay_bitwise=True,source_bound_state_digest_equal=True))
        bound[str(path/'manifest.json')]=sha(path/'manifest.json')
    profile_path=study/'profile/reference1/manifest.json';profile=read(profile_path)
    if not profile['profile'] or profile['steps_per_branch']!=64 or not profile['qualification']['unchanged_offset_gradient_bitwise'] or not profile['branches']['replay']['complete_state_replay_bitwise']:
        raise ValueError('Missing real native profile')
    if profile['protocol_sha256']!=sha(protocol) or profile['producer_sources']!=s['producer_sources'] or profile['case']!=next(c for c in s['cases'] if c['name']=='reference1'):
        raise ValueError('Unbound native profile')
    if set(profile['branches'])!={'native_keep','replay'} or not 0<profile['max_cuda_memory_bytes']<=20*1024**3:raise ValueError('Invalid native profile design')
    profile_arrays={}
    for name,b in profile['branches'].items():
        pp=profile_path.parent/(name+'.npz');bound[str(pp)]=sha(pp)
        if bound[str(pp)]!=b['artifact_sha256'] or b['steps']!=64 or b['offset']!=1e-9 or b['reset_first_moment'] is not False:raise ValueError('Changed native profile branch')
        with np.load(pp,allow_pickle=False) as z:profile_arrays[name]={k:z[k] for k in ['rows','offsets','lr','loss','initial_native_logits','initial_intervened_logits','final_logits','targets']}
        z=profile_arrays[name]
        if z['rows'].shape!=(64,32) or z['loss'].shape!=(64,) or z['final_logits'].shape!=(64,32000):raise ValueError('Invalid native profile shape')
        if not all(np.isfinite(a).all() for a in z.values()) or not np.array_equal(z['rows'],selected['reference1_rows'][:64]) or not np.array_equal(z['offsets'],selected['reference1_offsets'][:64]):raise ValueError('Invalid native profile source')
        if not np.array_equal(z['initial_native_logits'],z['initial_intervened_logits']):raise ValueError('Native profile forward mismatch')
    if any(not np.array_equal(profile_arrays['native_keep'][k],profile_arrays['replay'][k]) for k in profile_arrays['native_keep']):raise ValueError('Native profile saved replay mismatch')
    if profile['branches']['native_keep']['final_state_sha256']!=profile['branches']['replay']['final_state_sha256']:raise ValueError('Native profile state replay mismatch')
    role='scientific' if legacy else s['role']
    scientific_updates=2048 if role=='scientific' else 0
    if summary['scientific_updates']!=scientific_updates or summary['replay_updates']!=(512 if role=='scientific' else 0) or summary['profile_updates']!=128:raise ValueError('Role counts changed')
    for p in (study/'analysis').glob('*.json'):bound[str(p)]=sha(p)
    bound[str(study/'profile/reference1/manifest.json')]=sha(study/'profile/reference1/manifest.json')
    for p in [REPO/'scripts/analyze_potential_factorial.py',REPO/'scripts/analyze_potential_block_area.py',REPO/'scripts/factorial_admission.py',Path(__file__).resolve()]:bound[str(p)]=sha(p)
    for c in s['cases']:
        for key in ['checkpoint','parent_manifest']:bound[c[key]]=sha(c[key])
    bound.update(s['native_sources'])
    bound[str(REPO/'scripts/factorial_admission.py')]=sha(REPO/'scripts/factorial_admission.py')
    result=dict(schema='potential-factorial-verification-v2',status='passed',checks=checks,scientific_updates=scientific_updates,
        reconstruction_mode='retained-executed-source' if legacy else 'current-schema',
        executing_source_root=str(source.resolve()),execution_permission=False,role=role,
        verified_corpus=corpus,corpus_binding_scope=corpus_scope,
        reconstructed_step_logit_coordinates=vocabulary_coordinates,maximum_primary_reduction_error=largest_error,
        verifier_sha256=sha(__file__),checked_sha256=bound,
        scope='Independent raw arrays, nonrepeated complete source prefix, finite energy, paired effects and predictive logit reductions. Full-state replay is additionally checked by source-bound producer digests; full native tensors are not independently reconstructed.')
    # Every emitted path identifies bytes checked now, including archived producers.
    for path,digest in bound.items():
        if sha(path)!=digest:raise ValueError('Verification input changed before final binding: '+path)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','scientific_updates','reconstructed_step_logit_coordinates','maximum_primary_reduction_error']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,default=ROOT/'potential-factorial-20260913');p.add_argument('--output',type=Path,required=True);p.add_argument('--executed-source',type=Path);a=p.parse_args();main(a.study,a.output,a.executed_source)
