#!/usr/bin/env python3
"""Independently reconstruct the native single-pass criticality evidence."""
import argparse
from collections import defaultdict
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict, discrepancy, array_discrepancy, finite_array, finite_scalar
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.provenance import sha256, write_json
from model_rg.replay_equality import logical_equal as exact_equal, require_replay, require_observations, contract
from model_rg.run_outcome import TERMINAL

REPO=Path(__file__).resolve().parents[1]


def source_population(protocol, environment):
    """Reconstruct the declared resource law without using the producer selector."""
    source=protocol['source'];design=protocol['design']
    halves=np.random.default_rng(source['partition_seed']).permutation(524288).reshape(2,-1)
    policy=source.get('population_policy','full-master-half-v1')
    if policy=='full-master-half-v1':
        documents=262144
    elif policy=='nested-width-prefix-v1':
        spec=protocol['joint_family']['specification']
        if len(design['heads'])!=1:raise ValueError('A clock-family member has one width')
        n=design['heads'][0];documents=n*spec['corpus_documents_per_head']
        if n not in spec['heads'] or design['steps']!=n*spec['steps_per_head']:
            raise ValueError('Joint-family time or width mismatch')
        if design['controls']!=[2*r/n for r in spec['rate_ratios']]:
            raise ValueError('Joint-family rate ratio mismatch')
        for name in ['seeds','environments','shared_seed']:
            if design[name]!=spec[name]:raise ValueError('Joint-family design changed')
        fraction=32*design['steps']/(8*documents)
        if fraction!=protocol['joint_family']['consumed_fraction']:
            raise ValueError('Joint-family consumed fraction differs')
    else:
        raise ValueError('Unknown source population law')
    if not 1<=documents<=262144 or source['population_documents']!=documents or source['population_blocks']!=8*documents:
        raise ValueError('Source population size differs from the declared law')
    if 32*design['steps']>8*documents:
        raise ValueError('Source population is exhausted')
    return halves[environment,:documents]


def verify(study,analysis_path,output,source_root=None,observations_only=False):
    started=time.monotonic()
    if output.exists(): raise FileExistsError(output)
    protocol_path=study/'protocol.json'
    protocol=load_json_strict(protocol_path.read_text())
    ph=sha256(protocol_path)
    analysis=load_json_strict(analysis_path.read_text())
    if protocol['schema']!='critical-onepass-v1' or protocol['role']!='scientific':
        raise ValueError('Wrong execution role')
    if analysis['status']!='complete' or analysis['protocol_sha256']!=ph:
        raise ValueError('Missing complete analysis')
    retained_checkpoints={};retained_auxiliary={}
    checked={str(protocol_path):ph,str(analysis_path):sha256(analysis_path)}
    for record in [protocol['source_sha256'],analysis['source_sha256']]:
        for name,digest in record.items():
            path=(source_root or REPO)/name
            if sha256(path)!=digest: raise ValueError('Changed producer '+name)
            checked[str(path)]=digest
    for name,digest in protocol['input_sha256'].items():
        if sha256(name)!=digest: raise ValueError('Changed input '+name)
        checked[name]=digest
    selection_path=study/'selection.npz'
    if sha256(selection_path)!=protocol['selection_sha256']:
        raise ValueError('Changed selection')
    checked[str(selection_path)]=sha256(selection_path)
    with np.load(selection_path) as saved:
        for e in protocol['design']['environments']:
            population=source_population(protocol,e)
            np.testing.assert_array_equal(saved[f'population_{e}'],population)
            local=np.random.default_rng(9152502+e).permutation(len(population)*8)[:protocol['design']['steps']*32]
            blocks=(8*population[local//8]+local%8).reshape(-1,32)
            np.testing.assert_array_equal(saved[f'blocks_{e}'],blocks)
            if len(np.unique(blocks))!=blocks.size: raise ValueError('Repeated supervised token block')
        input_paths=list(protocol['input_sha256'])
        cp=Path(next(n for n in input_paths if '/refinedweb-onepass-524288/tokens.npy' in n)).parent
        pp=Path(next(n for n in input_paths if '/data/short/tokens.npy' in n)).parent
        cm=load_json_strict((cp/'manifest.json').read_text())
        if cm['tokens_sha256']!=sha256(cp/'tokens.npy') or cm['records_sha256']!=sha256(cp/'records.json'):
            raise ValueError('Corpus manifest differs from actual data')
        records=load_json_strict((cp/'records.json').read_text())
        probes=load_json_strict((pp/'records.json').read_text())
        training_hashes={r['content_sha256'] for r in records}
        evaluation_hashes={probes[i]['content_sha256'] for i in saved['probe_rows']}
        if len(training_hashes)!=524288 or training_hashes&evaluation_hashes:
            raise ValueError('Document identity failure')
        pt=np.load(pp/'tokens.npy',mmap_mode='r');po=np.load(pp/'offsets.npy')
        pr=saved['probe_rows']
        np.testing.assert_array_equal(saved['probes'],pt[pr[:,None],po[pr,None]+np.arange(65)])
        block_selections={e:saved[f'blocks_{e}'].copy() for e in protocol['design']['environments']}
    qpath=study/'qualification/verification.json'
    q=load_json_strict(qpath.read_text())
    if q['status']!='passed' or q['protocol_sha256']!=ph or q['scientific_updates']!=0:
        raise ValueError('Missing qualification')
    for name,digest in q['checked_sha256'].items():
        if sha256(name)!=digest: raise ValueError('Changed qualification artifact')
        checked[name]=digest
    checked[str(qpath)]=sha256(qpath)
    states=[torch.load(study/'qualification'/name/'final-state.pt',map_location='cpu',weights_only=False)
            for name in ['native','replay']]
    require_replay(*states)
    with np.load(study/'qualification/native/observations.npz') as a, np.load(study/'qualification/replay/observations.npz') as b:
        require_observations(a,b)
    del states
    groups=defaultdict(list);initials={};shared={};updates=0;failed=[]
    jobs=protocol['jobs']
    expected={(e,h,float(g),s) for e in protocol['design']['environments'] for h in protocol['design']['heads']
              for g in protocol['design']['controls'] for s in protocol['design']['seeds']}
    if len(jobs)!=len(expected) or {(j['environment'],j['heads'],float(j['control']),j['seed']) for j in jobs}!=expected:
        raise ValueError('Jobs do not cover the full frozen design')
    for job in jobs:
        root=study/'runs'/job['run_id'];mp=root/'manifest.json'
        m=load_json_strict(mp.read_text())
        if m['job']!=job or m['role']!='scientific' or m['protocol_sha256']!=ph:
            raise ValueError('Native manifest mismatch')
        if m['scientific_updates']!=m['completed_steps'] or m['qualification_updates']!=0:
            raise ValueError('Update-role accounting mismatch')
        updates+=m['scientific_updates'];checked[str(mp)]=sha256(mp)
        for name,digest in m['artifacts'].items():
            if observations_only and name!='observations.npz':
                (retained_checkpoints if name=='final-state.pt' else retained_auxiliary)[str(root/name)]=digest
                continue
            if sha256(root/name)!=digest: raise ValueError('Artifact changed '+str(root/name))
            checked[str(root/name)]=digest
        if m['status'] not in TERMINAL: raise ValueError('Nonterminal run')
        if m['status']!='complete': failed.append(job['run_id']);continue
        init_key=(job['environment'],job['heads'],job['seed'])
        previous=initials.setdefault(init_key,m['initial_parameter_sha256'])
        if previous!=m['initial_parameter_sha256']: raise ValueError('Control changes initialization')
        previous=shared.setdefault(job['shared_seed'],m['initial_shared_sha256'])
        if previous!=m['initial_shared_sha256']: raise ValueError('Shared initial generator changed')
        with np.load(root/'observations.npz') as a:
            np.testing.assert_array_equal(a['blocks'],block_selections[job['environment']][:m['completed_steps']])
            if len(a['training_loss'])!=m['completed_steps']: raise ValueError('Loss count mismatch')
            for name in a.files:
                if not np.isfinite(a[name]).all(): raise ValueError('Nonfinite raw observation')
        if m['status']!='complete': failed.append(job['run_id']);continue
        if m['completed_steps']!=job['steps']: raise ValueError('False completed status')
        groups[(job['environment'],job['heads'],job['control'])].append((job['seed'],root))
    index={(r['environment'],r['heads'],r['control'],r['step'],r['field']):r for r in analysis['cells']}
    count=0;maximum_error=0.
    for (e,h,g),rows in groups.items():
        if len(rows)!=len(protocol['design']['seeds']):continue
        rows.sort()
        arrays=[]
        for _,root in rows:
            with np.load(root/'observations.npz') as raw:
                arrays.append({'steps':raw['steps'], 'heads':raw['heads']})
        for k,t in enumerate(arrays[0]['steps']):
            for field,coordinate in [('row',0),('attention',1),('operator',2),('common_amplitude',None)]:
                if coordinate is None:
                    energy=np.stack([a['heads'][k,...,3] for a in arrays])
                    ratio=np.stack([a['heads'][k,...,0] for a in arrays])
                    x=np.sqrt(np.maximum(energy-ratio*np.maximum(energy,1e-30),0.))
                else:
                    x=np.stack([a['heads'][k,...,coordinate] for a in arrays])
                centered=x-x.mean(axis=0,keepdims=True)
                head_mean=centered.sum(-1)/h
                variance_sum=np.sum(head_mean**2)
                chi=h*variance_sum/((len(rows)-1)*64*5)
                v=np.sum(centered**2)/((len(rows)-1)*64*5*h)
                record=index[(e,h,g,int(t),field)]
                expected_values={'mean':float(np.sum(x)/x.size),'susceptibility':float(chi),
                    'intensive_variance':float(chi/h),'one_head_variance':float(v),
                    'cross_head_covariance':float((chi-v)/(h-1))}
                for name,value in expected_values.items():
                    error=discrepancy(value,record[name],name,atol=2e-12,rtol=2e-12)
                    maximum_error=max(maximum_error,error)
                if chi < -1e-12 or finite_greater(chi, h*v+2e-12*(1+abs(h*v)), 'scripts/verify_critical_onepass.py:169'):
                    raise ValueError('Exact head covariance bound violated')
                count+=1
    if count!=len(index) or updates!=analysis['scientific_updates']:
        raise ValueError('Analysis coverage mismatch')
    if sorted(failed)!=sorted(r['run_id'] for r in analysis.get('failed_paths', analysis.get('numerical_failures',[]))):
        raise ValueError('Numerical failures were omitted')
    result=dict(schema='critical-onepass-verification-v1',status='passed',study=str(study),
        analysis=str(analysis_path),protocol_sha256=ph,scientific_paths=len(jobs),
        scientific_updates=updates,qualification_updates=q['qualification_updates'],
        cells_independently_reconstructed=count,maximum_absolute_reduction_error=maximum_error,
        evaluation_training_document_overlap=0,distinct_blocks_per_complete_path=protocol['design']['steps']*32,
        numerical_failures=failed,full_qualification_state_replay=True,checked_sha256=checked,
        observations_only=observations_only, retained_checkpoint_identities=retained_checkpoints, retained_auxiliary_artifact_identities=retained_auxiliary, checkpoint_scope='Scientific checkpoint and shared-history identities are retained metadata in observations-only mode; every scalar observation and the full qualification replay are checked.', source_root=str((source_root or REPO).resolve()), replay_contract=contract(), verifier_sha256=sha256(__file__),runtime_seconds=time.monotonic()-started)
    write_json(output,result)
    print(json.dumps({k:result[k] for k in ['status','scientific_paths','scientific_updates','cells_independently_reconstructed']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path);p.add_argument('--observations-only',action='store_true');a=p.parse_args()
    verify(a.study.resolve(),a.analysis.resolve(),a.output.resolve(),a.source_root,a.observations_only)

if __name__=='__main__':main()
