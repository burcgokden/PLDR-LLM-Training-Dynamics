"""Admission of released-model computations before filesystem or native effects.

Archived scientific producers are verified separately against their snapshots.
This contract authorizes only current execution and never upgrades old records.
"""
from pathlib import Path
import ast
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import numpy as np
import torch
from model_rg.provenance import sha256

REPO=Path(__file__).resolve().parents[2]
from companion_paths import data_root
ROOT=data_root('model')
SCHEMA='released-execution-qualification-v2'
PINS={1:'7a34e2ca9aa78038683677cfda17fe3a9fe6da8a',4:'c377be06f2294aeccb56d19ea5f52640b1850daf',5:'de8e539c0ba1829072f4b8c2c5fae3bde0a3a2d2'}
ENTRY={'full':['train_released_adaptation','assess_released_adaptation','observe_released_geometry'],
       'factor':['train_released_lowrank']}
REQUIRED_ASSETS={'config.json','configuration_pldrllm.py','modeling_pldrllm.py','model.safetensors','tokenizer.model','tokenizer_config.json','special_tokens_map.json','generation_config.json'}


def need(value,message):
    if not value:raise ValueError(message)


def read(path):
    try:value=json.loads(Path(path).read_text())
    except (OSError,json.JSONDecodeError) as exc:raise ValueError('Missing or malformed JSON: '+str(path)) from exc
    need(isinstance(value,dict),'Expected JSON object: '+str(path))
    return value


def destination(study,name=None,kind='runs'):
    study=Path(study).resolve()
    need(study!=ROOT and study.is_relative_to(ROOT),'Unauthorized study destination')
    if name is None:return study
    need(isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}',name) is not None,'Invalid run name')
    base=(study/kind).resolve();out=(base/name).resolve()
    need(base.is_relative_to(study) and out.parent==base,'Run escapes study directory')
    return out


def configure():
    # These are the actual semantics used after admission by both workers.
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def runtime():
    packages={n:importlib.metadata.version(n) for n in ['numpy','torch','transformers','safetensors','sentencepiece','huggingface-hub','pyarrow']}
    return dict(python=platform.python_version(),packages=packages,cuda=torch.version.cuda,
        dtype=str(torch.get_default_dtype()),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn=torch.backends.cudnn.version(),
        optimizer=dict(name='AdamW',betas=[.9,.95],eps=1e-8,weight_decay=.01,clip_norm=1.),
        inference=dict(cache=False,temperature=1.,mask='un-padded causal',attention='eager'))


def source_inventory(repo=REPO):
    """Resolve literal project imports, including relative imports, to a fixed point.

    The downloaded native package is inventoried in full separately because its
    import names are constructed dynamically by NativeModel.
    """
    repo=Path(repo)
    pending=[repo/'scripts'/n for n in ['qualify_released_execution.py','qualify_released_base.py',
        'train_released_adaptation.py','train_released_lowrank.py','assess_released_adaptation.py',
        'observe_released_geometry.py','select_released_base.py','initialize_released_study.py','run_released_study.py',
        'run_released_lowrank_queue.py','run_released_assessment_queue.py','prepare_released_adaptation.py']]
    pending += [repo/'src/model_rg/released_qualification.py']
    found={}
    while pending:
        p=pending.pop().resolve();n=str(p.relative_to(repo))
        if n in found:continue
        need(p.is_file(),'Missing qualification dependency: '+n);found[n]=sha256(p)
        for node in ast.walk(ast.parse(p.read_text())):
            names=([node.module] if isinstance(node,ast.ImportFrom) and node.module else
                [a.name for a in node.names] if isinstance(node,ast.Import) else [])
            for mod in names:
                rel=Path(*mod.split('.'))
                bases=[repo,repo/'src',repo/'scripts']
                if isinstance(node,ast.ImportFrom) and node.level:
                    parent=p.parent
                    for _ in range(node.level-1):parent=parent.parent
                    bases=[parent]
                for b in bases:
                    for candidate in [b/rel.with_suffix('.py'),b/rel/'__init__.py']:
                        if candidate.is_file():pending.append(candidate)
    for n in ['assessment-plan.json','language-adaptation-extension.json','released-pinned-assets.json']:
        p=repo/'docs/templates'/n;found[str(p.relative_to(repo))]=sha256(p)
    return dict(sorted(found.items()))


def assets_inventory(study,index):
    entries=read(Path(study)/'downloaded-models.json');name=f'PLDR-LLM-v51-SOC-110M-{index}'
    entry=entries.get(name,{})
    need(entry.get('repository')=='fromthesky/'+name and entry.get('revision')==PINS.get(index),'Unpinned model identity')
    files=entry.get('files',{});need(REQUIRED_ASSETS<=set(files),'Incomplete pinned model assets')
    folder=(ROOT/'assets'/name).resolve();result={}
    for n,h in files.items():
        p=(folder/n).resolve();need(p.is_relative_to(folder) and p.is_file(),'Invalid asset path')
        need(sha256(p)==h,'Changed pinned asset: '+n);result[str(p)]=h
    for p in folder.rglob('*'):
        if p.is_file() and '.cache' not in p.parts and p.suffix in ['.py','.json','.model','.safetensors']:
            need(str(p.resolve()) in result,'Unbound native source/asset: '+str(p))
    return result


def data_inventory(study):
    study=Path(study);result={}
    def bind(p,h=None):
        p=Path(p).resolve();actual=sha256(p)
        need(h is None or h==actual,'Changed data: '+str(p));result[str(p)]=actual
    lang=read(study/'data/language.json');phys=read(study/'data/physical.json')
    need(lang.get('status')=='complete' and phys.get('status')=='complete','Incomplete source data')
    for n in ['data/language.json','data/physical.json','assessment-plan.json','language-adaptation-extension.json','study-design.json','downloaded-models.json']:bind(study/n)
    bind(study/'data/language.npz',lang['data_sha256'])
    for p,h in lang['inputs'].items():bind(p,h)
    bind(ROOT/'finetuning-sectors-20260912/data/lexical-labels.npz',lang['labels_sha256'])
    bind(ROOT/'finetuning-sectors-20260912/data/evaluation.npz')
    if (study/'singlepass-comparison-plan.json').exists():bind(study/'singlepass-comparison-plan.json')
    expected=[]
    for split,sizes,ratios,chains,samples in [('train',[4,8,16],[.85,.925,1.,1.075,1.15],16,128),('validation',[4,8,16],[.85,1.,1.15],8,64),('test',[4,6,8,12,16,20,24],[.94,.97,1.,1.03,1.06],16,128)]:
        expected += [(split,q,L,r,chains,samples) for q in [2,3] for L in sizes for r in ratios]
    observed=[(c['split'],c['q'],c['L'],c['temperature_ratio'],c['chains'],c['samples_per_chain']) for c in phys['cells']]
    need(sorted(observed)==sorted(expected),'Physical source design differs from the implemented worker')
    for cell in phys['cells']:bind(cell['path'],cell['sha256'])
    with np.load(study/'data/language.npz',allow_pickle=False) as z:
        used=[]
        for domain in ['technical','narrative','unassigned']:
            for split,count in [('train',4096),('validation',256),('test',512)]:
                ids=z[f'{domain}-{split}-ids'];need(len(ids)==count and len(np.unique(ids))==count,'Invalid split identity')
                used.extend(ids.tolist())
        need(len(set(used))==len(used),'Train/validation/test document overlap')
    return dict(sorted(result.items()))


def validate_design(study):
    study=Path(study);d=read(study/'study-design.json')
    need(d.get('schema')=='released-study-design-v2','Missing frozen study design')
    need(d.get('mode') in ['single-pass','replay-retained'],'Unsupported study mode')
    need(d.get('selected_model')==5,'Current released workers require selected model 5')
    expected=1 if d['mode']=='single-pass' else 3
    need(d.get('epochs')==expected,'Corpus-resource law mismatch')
    for n,key in [('assessment-plan.json','assessment_sha256'),('language-adaptation-extension.json','extension_sha256')]:
        need(sha256(study/n)==d.get(key),'Changed frozen stage plan: '+n)
    plan=read(study/'assessment-plan.json');ext=read(study/'language-adaptation-extension.json')
    need(plan.get('schema')=='released-adaptation-assessment-plan-v1' and ext.get('schema')=='released-language-constraint-extension-v1','Unsupported stage schema')
    need(plan.get('selected_base')==5 and ext.get('base')==5 and ext.get('rank')==4,'Incompatible stage identity')
    need(plan.get('language_training_epochs')==expected and plan.get('physical_training_epochs')==expected and ext.get('epochs')==expected,'Inconsistent stage epochs')
    need(ext.get('trainable_parameters')==203264 and ext.get('training_examples_per_epoch')==32768,'Changed factor design')
    templates=REPO/'docs/templates'
    canonical=read(templates/'assessment-plan.json');canonical.update(language_training_epochs=expected,physical_training_epochs=expected)
    need(plan==canonical,'Assessment plan differs from implemented fixed design')
    canonical=read(templates/'language-adaptation-extension.json');canonical['epochs']=expected
    need({k:v for k,v in ext.items() if k!='motivation'}=={k:v for k,v in canonical.items() if k!='motivation'},'Factor plan differs from implemented fixed design')
    return d


def contract(study,index,branch):
    need(branch in ENTRY,'Unknown qualification branch');validate_design(study)
    return dict(schema=SCHEMA,model=index,branch=branch,entrypoints=ENTRY[branch],
        sources=source_inventory(),assets=assets_inventory(study,index),data=data_inventory(study),runtime=runtime(),
        scope='Finite native computation and conservative execution correspondence; not a validation of population laws or arbitrary hyperparameters.')


def qualify_path(study,index,branch):
    signature=hashlib.sha256(json.dumps(source_inventory(),sort_keys=True).encode()).hexdigest()[:16]
    return Path(study)/'execution-qualification'/f'model{index}-{branch}'/signature/'verification.json'


def validate(study,index,branch,entrypoint):
    study=destination(study);p=qualify_path(study,index,branch);q=read(p)
    need(q.get('status')=='passed' and q.get('schema')==SCHEMA,'Missing completed current execution qualification')
    need(entrypoint in ENTRY.get(branch,[]),'Unsupported qualified entry point')
    need(q.get('contract')==contract(study,index,branch),'Stale source, asset, data, design, branch or numerical policy')
    files=q.get('artifacts');need(isinstance(files,dict) and set(files)=={'measurements.json','trace.npz'},'Incomplete qualification artifacts')
    for n,h in files.items():need(sha256(p.parent/n)==h,'Changed qualification result: '+n)
    with np.load(p.parent/'trace.npz',allow_pickle=False) as z:
        need(z['trace'].shape==(3,3) and np.isfinite(z['trace']).all(),'Invalid native qualification trace')
        need(len(z['document_ids'])==96 and len(np.unique(z['document_ids']))==96 and np.array_equal(z['block_numbers'],np.zeros(96)),'Invalid distinct-block qualification trace')
    m=read(p.parent/'measurements.json')
    need(m.get('status')=='passed' and m.get('branch')==branch and m.get('model')==index,'Incompatible native result')
    need(m.get('qualification_updates')==3 and m.get('scientific_updates')==0,'Incomplete branch qualification')
    need(m.get('distinct_training_blocks')==96 and m.get('prefix_exclusion') is True,'Incomplete source/prefix test')
    need(0<=m.get('selected_gradient_error',math.inf)<2e-5 and 0<=m.get('selected_logit_error',math.inf)<1e-3,'Selected output check failed')
    if branch=='factor':need(m.get('trainable_parameters')==203264 and len(m.get('targets',[]))==11 and 0<=m.get('merge_error',math.inf)<1e-3,'Factor export qualification failed')
    return q,p


def admit_training(a,branch):
    study=destination(a.study);out=destination(study,a.name);d=validate_design(study)
    index=getattr(a,'model',5);need(index==5,'Training design requires selected base 5')
    need(a.device in ['cuda:0','cuda:1'],'Unsupported device')
    need(type(a.epochs) is int and a.epochs==d['epochs'],'Unfrozen epoch/resource law')
    need(type(a.seed) is int and 0<=a.seed<2**63,'Invalid source seed')
    need(type(a.limit_steps) is int and 0<=a.limit_steps<=256,'Invalid development step budget')
    tasks=['technical','narrative','mixture','general']+(['physical'] if branch=='full' else [])
    need(a.task in tasks and math.isfinite(a.lr),'Unsupported task/rate')
    rates=([1e-4,3e-4] if branch=='factor' else [3e-5,1e-4] if a.task=='physical' else [1e-5,3e-5])
    need(a.lr in rates,'Learning rate is outside the frozen pilot/main design')
    if not a.limit_steps and branch=='full':need(a.lr==(3e-5 if a.task=='physical' else 1e-5),'Main rate differs from design')
    configure();q,p=validate(study,index,branch,'train_released_lowrank' if branch=='factor' else 'train_released_adaptation')
    return study,out,q,p


def assessment_identity(study,name):
    study=destination(study);destination(study,name,'assessment');selection=read(study/'base-selection.json')
    need(selection.get('selected')==5,'Assessment supports selected model 5')
    candidates=selection.get('candidates',[])
    need(sorted(c.get('model') for c in candidates)==[1,4,5],'Incomplete validation-only base selection')
    for row in candidates:
        path=Path(study)/f"base-qualification/model{row['model']}/verification.json"
        need(sha256(path)==row.get('qualification_sha256'),'Changed base validation result')
        record=read(path);need(record.get('status')=='passed','Unpassed base validation')
        need(abs(float(np.mean([v['nll'] for v in record['language']]))-row['language_nll'])<1e-12,'Invalid base validation reduction')
    need(min(candidates,key=lambda c:c['language_nll'])['model']==5,'Selection differs from validation')
    identity=dict(name=name,model=5,base_selection_sha256=sha256(study/'base-selection.json'))
    if name!='base5':
        root=destination(study,name);r=read(root/'result.json');p=read(root/'protocol.json')
        need(r.get('status')=='complete' and r.get('name')==name and r.get('model')==5,'Incomplete or unrelated selected checkpoint')
        need(sha256(root/'protocol.json')==r.get('protocol_sha256') and sha256(root/'best.pt')==r.get('best_sha256'),'Changed assessment checkpoint')
        need(p.get('model')==5 and min(r['observations'],key=lambda row:row['nll'])['step']==r['selected_step'],'Invalid validation selection')
        identity.update(checkpoint_sha256=r['best_sha256'],result_sha256=sha256(root/'result.json'),protocol_sha256=r['protocol_sha256'],step=r['selected_step'],epoch=r['selected_epoch'],task=p['task'])
    return identity


def admit_assessment(study,name,entrypoint='assess_released_adaptation'):
    configure();identity=assessment_identity(study,name);q,p=validate(study,5,'full',entrypoint)
    return q,p,identity
