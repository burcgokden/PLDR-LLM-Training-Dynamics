"""Checked stage orchestration for the released-model fixed study design."""
import argparse,json,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from initialize_released_study import preflight
from model_rg.released_qualification import validate_design,validate,configure


def main(a):
    study=Path(a.study).resolve();d=validate_design(study)
    preflight(study,'qualification' if a.stage!='prepare' else 'data',a.device if a.stage not in ['prepare','analyze'] else None)
    def run(script,*args):subprocess.run([sys.executable,str(REPO/'scripts'/script),'--study',str(study),*args],cwd=REPO,check=True)
    if a.stage=='prepare':run('prepare_released_adaptation.py')
    elif a.stage=='qualify':
        for index in [1,4,5]:run('qualify_released_base.py','--model',str(index),'--device',a.device)
        for branch in ['full','factor']:run('qualify_released_execution.py','--model','5','--branch',branch,'--device',a.device)
        run('select_released_base.py')
    elif a.stage in ['language','physical']:
        configure();validate(study,5,'full','train_released_adaptation')
        pilots=[('technical-low','technical',1e-5,219020),('technical-high','technical',3e-5,219020)] if a.stage=='language' else [('physical-low','physical',3e-5,219021),('physical-high','physical',1e-4,219021)]
        for name,task,lr,seed in pilots:run('train_released_adaptation.py','--name','pilot-'+name,'--task',task,'--model','5','--device',a.device,'--lr',str(lr),'--epochs',str(d['epochs']),'--seed',str(seed),'--limit-steps','256')
        jobs=[(task+'-s0',task,1e-5,219100) for task in ['technical','narrative','mixture','general']] if a.stage=='language' else [('physical-s0','physical',3e-5,219110),('physical-s1','physical',3e-5,219111)]
        for name,task,lr,seed in jobs:run('train_released_adaptation.py','--name',name,'--task',task,'--model','5','--device',a.device,'--lr',str(lr),'--epochs',str(d['epochs']),'--seed',str(seed))
    elif a.stage=='factor':run('run_released_lowrank_queue.py')
    elif a.stage=='assessment':run('run_released_assessment_queue.py','--worker',a.worker)
    elif a.stage=='analyze':
        for kind in ['spin','language','generation']:run('analyze_released_adaptation.py','--kind',kind)
        run('check_released_source_chains.py');run('analyze_released_spatial_centering.py')
        run('reconstruct_released_independent.py','--output',str(study/'analysis/independent.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--stage',choices=['prepare','qualify','language','physical','factor','assessment','analyze'],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--worker',choices=['language','physical'],default='language');main(p.parse_args())
