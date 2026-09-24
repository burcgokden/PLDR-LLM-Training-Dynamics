"""Initialize a frozen single-pass design, or explicitly replay a retained design.

Copies only design manifests and, when requested, source-data metadata/arrays.
It never copies model selections, qualifications, trained outcomes or scores.
"""
import argparse,json,shutil,sys,tempfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.provenance import sha256,write_json
from model_rg.released_qualification import destination,assets_inventory,validate_design,data_inventory,need,ROOT


def initialize(study,mode='single-pass',reuse_data=None):
    study=destination(study);need(not study.exists(),'Study already exists')
    templates=REPO/'docs/templates';payload={n:json.loads((templates/n).read_text()) for n in ['assessment-plan.json','language-adaptation-extension.json']}
    epochs=1 if mode=='single-pass' else 3
    payload['assessment-plan.json'].update(language_training_epochs=epochs,physical_training_epochs=epochs)
    payload['language-adaptation-extension.json'].update(epochs=epochs,motivation='Fixed factor-coordinate comparison; no test-based fitting.')
    need(mode in ['single-pass','replay-retained'],'Unsupported design mode')
    if reuse_data:
        reuse_data=Path(reuse_data).resolve()
        for n in ['language.json','language.npz','physical.json']:need((reuse_data/'data'/n).is_file(),'Missing retained source data: '+n)
    need(shutil.which('g++') is not None,'The physical sampler requires g++')
    need(shutil.disk_usage(study.parent).free>8*2**30,'At least 8 GiB free space required for initialization')
    final=study
    study=Path(tempfile.mkdtemp(prefix='.initializing-',dir=study.parent))
    shutil.copy2(templates/'released-pinned-assets.json',study/'downloaded-models.json')
    for n,obj in payload.items():write_json(study/n,obj)
    write_json(study/'study-design.json',dict(schema='released-study-design-v2',mode=mode,selected_model=5,epochs=epochs,
        assessment_sha256=sha256(study/'assessment-plan.json'),extension_sha256=sha256(study/'language-adaptation-extension.json'),
        provenance='Fixed design replay from published source templates; a new single-pass resource law when mode is single-pass. Neither mode is a new preregistration.',
        training_law='Each fixed proper-prefix example once per path' if epochs==1 else 'Three visits to each finite-corpus example',
        test_policy='Validation selects rates and checkpoints; test outcomes cannot modify the frozen comparison.',
        templates={n:sha256(templates/n) for n in payload}))
    for index in [1,4,5]:assets_inventory(study,index)
    validate_design(study)
    if reuse_data:
        (study/'data').mkdir()
        for n in ['language.json','language.npz','physical.json']:shutil.copy2(reuse_data/'data'/n,study/'data'/n)
        data_inventory(study)
    write_json(study/'initialization.json',dict(status='passed',stage='ready-for-qualification' if reuse_data else 'ready-for-data',mode=mode,
        reused_data=str(reuse_data) if reuse_data else None,producer_sha256=sha256(__file__),outcomes_copied=False,qualification_copied=False))
    study.rename(final)
    return final


def preflight(study,stage,device=None):
    study=destination(study);validate_design(study)
    for index in [1,4,5]:assets_inventory(study,index)
    if stage!='data':data_inventory(study)
    if device:
        import torch
        need(device in ['cuda:0','cuda:1'] and torch.cuda.is_available(),'Unavailable CUDA device')
        free,total=torch.cuda.mem_get_info(device);need(free>6*2**30,'Insufficient free GPU memory')
    return dict(status='passed',stage=stage,study=str(study),device=device)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--mode',choices=['single-pass','replay-retained'],default='single-pass');p.add_argument('--reuse-data');p.add_argument('--preflight',choices=['data','qualification']);p.add_argument('--device');a=p.parse_args()
    print(preflight(a.study,a.preflight,a.device) if a.preflight else initialize(a.study,a.mode,a.reuse_data))
