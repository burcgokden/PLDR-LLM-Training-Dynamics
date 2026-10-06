"""Freeze disjoint language documents and physical chains for released-model adaptation."""
from companion_paths import configured_path
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json,re,sys
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.provenance import sha256,write_json
from prepare_lattice_data import compile_sampler,generate
ROOT=Path(configured_path('data:model'))

def prepare(study):
    study=Path(study).resolve()
    if not study.is_relative_to(ROOT): raise ValueError('Unauthorized experiment destination')
    out=study/'data';out.mkdir(exist_ok=False)
    corpus=ROOT/'data/refinedweb-onepass-524288'
    prior=ROOT/'finetuning-sectors-20260912/data'
    records=json.loads((corpus/'records.json').read_text())
    info=Path(configured_path('assets:refinedweb/datasets/huggingface_datasets/tiiuae___falcon-refinedweb/default/0.0.0/c735840575b629292b41da8dde11dcd523d4f91c/dataset_info.json'))
    offsets=np.r_[0,np.cumsum(json.loads(info.read_text())['splits']['train']['shard_lengths'])]
    positions=np.array([offsets[int(re.search(r'train-(\d+)-of-',r['shard']).group(1))]+r['row'] for r in records],dtype=np.int64)
    with np.load(prior/'lexical-labels.npz') as z: labels=z['labels']
    with np.load(prior/'evaluation.npz') as z: excluded=z['document_ids']
    tokens=np.load(corpus/'tokens.npy',mmap_mode='r')
    rng=np.random.default_rng(219001)
    arrays={};counts={}
    for label,name in [(1,'technical'),(0,'narrative'),(-1,'unassigned')]:
        candidates=np.flatnonzero((labels==label)&(positions>=160000000)&(~np.isin(np.arange(len(labels)),excluded)))
        order=rng.permutation(candidates)
        if len(order)<4864:raise ValueError('Insufficient separated documents')
        for split,ids in [('validation',order[:256]),('test',order[256:768]),('train',order[768:4864])]:
            arrays[f'{name}-{split}-ids']=ids
            arrays[f'{name}-{split}-tokens']=np.asarray(tokens[ids],np.int32)
            arrays[f'{name}-{split}-positions']=positions[ids]
            counts[f'{name}-{split}']=len(ids)
    np.savez_compressed(out/'language.npz',**arrays)
    del arrays
    write_json(out/'language.json',dict(status='complete',counts=counts,seed=219001,
        train_blocks_per_document=8,block_length=64,target_position=64,
        validation_and_test_prefixes=[32,64,128],validation_and_test_offsets=[0,192],
        source_policy='The existing lexical classifier defines domains. Documents are disjoint across train, validation and test. Earlier evaluation documents excluded. Global positions >=160 million under the cached ordering. Original released pretraining manifests are unavailable, so this is documented position separation, not a claim of independently verified text deduplication against pretraining.',
        reported_selection='Model 5 is the primary candidate from the reported longest pretraining and benchmark averages; models 1 and 4 are evaluated on validation only before selection.',
        inputs={str(corpus/n):sha256(corpus/n) for n in ['tokens.npy','records.json','manifest.json']},
        labels_sha256=sha256(prior/'lexical-labels.npz'),data_sha256=sha256(out/'language.npz'),producer_sha256=sha256(__file__)))
    binary=compile_sampler(study)
    # Validation and test use wholly distinct chain seed ranges. Complete configurations,
    # rather than individual target spins, are the split units.
    jobs=[]
    for split,sizes,ratios,chains,samples,origin in [
        ('train',[4,8,16],[.85,.925,1.,1.075,1.15],16,128,2191000),
        ('validation',[4,8,16],[.85,1.,1.15],8,64,2192000),
        ('test',[4,6,8,12,16,20,24],[.94,.97,1.,1.03,1.06],16,128,2193000)]:
        for q in [2,3]:
            for L in sizes:
                for ratio in ratios:
                    idx=sum(j[0]==split for j in jobs)
                    jobs.append((split,q,L,ratio,chains,samples,origin+idx))
    specs=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(generate,binary,out/'physical'/split/f'q{q}-L{L}-r{ratio:.3f}',q,L,ratio,seed,chains,samples,burn=4096,thin=96):(split,seed) for split,q,L,ratio,chains,samples,seed in jobs}
        for fut in as_completed(futures):
            spec=fut.result();spec['split']=futures[fut][0];specs.append(spec)
            print('physical source',spec['split'],spec['q'],spec['L'],spec['temperature_ratio'],flush=True)
    specs.sort(key=lambda c:(c['split'],c['q'],c['L'],c['temperature_ratio']))
    for i,c in enumerate(specs):c['id']=i
    write_json(out/'physical.json',dict(status='complete',cells=specs,
        producer_sha256=sha256(__file__),sampler_sha256=sha256(binary),sampler_source_sha256=sha256(REPO/'scripts/potts_mc.cpp'),
        source_law='Periodic square-lattice q-state Potts equilibrium. q=2 is Ising in the stated temperature units. Independent chains; half ordered, half random starts; fixed Wolff burn-in and thinning.',
        adaptation_law='Uniform physical cell, uniformly selected target site, uniform color permutation and square-lattice symmetry. Random permutation of the finite training configurations in each epoch. Every prefix strictly excludes its target and all later spins. Repeat epochs are explicitly permitted for this adaptation dataset.',
        test_policy='Test chains are not evaluated during hyperparameter or epoch selection. Unseen sizes 6,12,20,24 remain test-only.'))
    print(out,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();prepare(a.study)
