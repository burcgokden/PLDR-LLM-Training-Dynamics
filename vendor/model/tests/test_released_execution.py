"""Admission must fail before output, data-draw, checkpoint or native effects."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from model_rg import released_qualification as q
from model_rg.provenance import sha256


def put(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))

@pytest.fixture
def study(tmp_path,monkeypatch):
    monkeypatch.setattr(q,'ROOT',tmp_path)
    s=tmp_path/'study';s.mkdir()
    plan=q.read(q.REPO/'docs/templates/assessment-plan.json');plan.update(language_training_epochs=1,physical_training_epochs=1)
    ext=q.read(q.REPO/'docs/templates/language-adaptation-extension.json');ext['epochs']=1
    put(s/'assessment-plan.json',plan);put(s/'language-adaptation-extension.json',ext)
    put(s/'study-design.json',dict(schema='released-study-design-v2',mode='single-pass',selected_model=5,epochs=1,assessment_sha256=sha256(s/'assessment-plan.json'),extension_sha256=sha256(s/'language-adaptation-extension.json')))
    def contract(study,index,branch):return dict(schema=q.SCHEMA,model=index,branch=branch,entrypoints=q.ENTRY[branch],sources={'producer.py':'source'},assets={'weights':'asset'},data={'split':'data'},runtime={'dtype':'float32'})
    monkeypatch.setattr(q,'contract',contract)
    for branch in ['full','factor']:
        root=q.qualify_path(s,5,branch).parent;root.mkdir(parents=True)
        put(root/'measurements.json',dict(status='passed',model=5,branch=branch,qualification_updates=3,scientific_updates=0,distinct_training_blocks=96,prefix_exclusion=True,selected_logit_error=0.,selected_gradient_error=0.,trainable_parameters=203264,targets=list(range(11)),merge_error=0.))
        np.savez(root/'trace.npz',trace=np.ones((3,3)),document_ids=np.arange(96),block_numbers=np.zeros(96))
        put(root/'verification.json',dict(schema=q.SCHEMA,status='passed',contract=contract(s,5,branch),artifacts={n:sha256(root/n) for n in ['measurements.json','trace.npz']}))
    candidates=[]
    for index in [1,4,5]:
        path=s/f'base-qualification/model{index}/verification.json';put(path,dict(status='passed',language=[dict(nll=6-index)]))
        candidates.append(dict(model=index,language_nll=6-index,qualification_sha256=sha256(path)))
    put(s/'base-selection.json',dict(selected=5,candidates=candidates))
    return s


def args(study,**changes):
    d=dict(study=str(study),name='fresh-run',model=5,task='technical',device='cuda:0',epochs=1,seed=1,limit_steps=0,lr=1e-5);d.update(changes);return SimpleNamespace(**d)

@pytest.mark.parametrize('branch',['full','factor'])
def test_current_contract_admits_without_creating_outputs(study,branch):
    q.admit_training(args(study,lr=1e-4 if branch=='factor' else 1e-5),branch)
    assert not (study/'runs').exists()
    q.admit_assessment(study,'base5')
    assert not (study/'assessment').exists()

@pytest.mark.parametrize('branch',['full','factor'])
@pytest.mark.parametrize('field',['schema','status','model','branch','entrypoints','sources','assets','data','runtime','artifacts'])
def test_every_bound_contract_field_rejects_before_worker_effects(study,branch,field):
    p=q.qualify_path(study,5,branch);d=q.read(p)
    if field in ['schema','status','artifacts']:d[field]='changed'
    else:d['contract'][field]='changed'
    put(p,d)
    if branch=='full':from scripts.train_released_adaptation import main
    else:from scripts.train_released_lowrank import main
    with pytest.raises(ValueError):main(args(study,lr=1e-4 if branch=='factor' else 1e-5))
    assert not (study/'runs').exists()

@pytest.mark.parametrize('field',['schema','status','model','sources','assets','data','runtime','artifacts'])
def test_assessment_rejection_precedes_directory_and_checkpoint(study,field):
    p=q.qualify_path(study,5,'full');d=q.read(p)
    if field in ['schema','status','artifacts']:d[field]='changed'
    else:d['contract'][field]='changed'
    put(p,d)
    from scripts.assess_released_adaptation import main
    with pytest.raises(ValueError):main(SimpleNamespace(study=str(study),name='base5',device='cuda:0',kind='language'))
    assert not (study/'assessment').exists()

@pytest.mark.parametrize('name',['../escape','x/y','..','/tmp/escape','','a.b'])
@pytest.mark.parametrize('branch',['full','factor'])
def test_worker_rejects_escaping_names(study,name,branch):
    with pytest.raises(ValueError):q.admit_training(args(study,name=name,lr=1e-4 if branch=='factor' else 1e-5),branch)
    assert not (study/'runs').exists()

@pytest.mark.parametrize('changes',[dict(epochs=3),dict(epochs=0),dict(seed=-1),dict(limit_steps=-1),dict(limit_steps=257),dict(lr=float('nan')),dict(lr=.1),dict(model=4),dict(device='cpu'),dict(task='arbitrary')])
def test_unfrozen_training_design_rejected(study,changes):
    with pytest.raises(ValueError):q.admit_training(args(study,**changes),'full')
    assert not (study/'runs').exists()

@pytest.mark.parametrize('branch',['full','factor'])
def test_result_artifacts_cannot_be_changed_or_omitted(study,branch):
    root=q.qualify_path(study,5,branch).parent
    (root/'trace.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='qualification result'):q.validate(study,5,branch,q.ENTRY[branch][0])


def test_assessment_checkpoint_has_own_binding_before_loading(study):
    root=study/'runs'/'fitted';root.mkdir(parents=True);(root/'best.pt').write_bytes(b'weights')
    put(root/'protocol.json',dict(model=5,task='technical'))
    put(root/'result.json',dict(status='complete',name='fitted',model=5,protocol_sha256=sha256(root/'protocol.json'),best_sha256=sha256(root/'best.pt'),observations=[dict(step=0,nll=3.),dict(step=1,nll=2.)],selected_step=1,selected_epoch=1.))
    q.admit_assessment(study,'fitted')
    (root/'best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='assessment checkpoint'):q.admit_assessment(study,'fitted')


def test_plan_tamper_rejected_before_data_loading(study):
    p=study/'assessment-plan.json';d=q.read(p);d['language_training_epochs']=3;put(p,d)
    with pytest.raises(ValueError,match='frozen stage plan'):q.admit_training(args(study),'full')
