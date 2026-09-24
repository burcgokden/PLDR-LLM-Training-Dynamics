"""One-GPU queue with validation-only learning-rate choice and retained candidates."""
import argparse,json,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
def main(a):
    s=Path(a.study)
    terminal=s/'runs/general-s0/result.json'
    if not terminal.is_file() or json.loads(terminal.read_text()).get('status')!='complete':raise ValueError('Prerequisite has not completed successfully: '+str(terminal))
    epochs=json.loads((s/'study-design.json').read_text())['epochs']
    candidates=[]
    for tag,lr in [('low',1e-4),('high',3e-4)]:
        name='pilot-lowrank-'+tag
        subprocess.run([sys.executable,str(REPO/'scripts/train_released_lowrank.py'),'--study',str(s),'--name',name,'--task','technical','--device','cuda:0','--lr',str(lr),'--epochs',str(epochs),'--seed','219100','--limit-steps','256'],cwd=REPO,check=True)
        result=json.loads((s/'runs'/name/'result.json').read_text());candidates.append(dict(name=name,lr=lr,validation_nll=result['observations'][-1]['nll']))
    selected=min(candidates,key=lambda r:r['validation_nll']);(s/'lowrank-rate-selection.json').write_text(json.dumps(dict(selected=selected,candidates=candidates,criterion='Lowest technical proper-prefix validation NLL after the fixed 256-update development run; no test access.'),indent=2)+'\n')
    for task in ['technical','narrative','mixture','general']:
        subprocess.run([sys.executable,str(REPO/'scripts/train_released_lowrank.py'),'--study',str(s),'--name',task+'-lowrank','--task',task,'--device','cuda:0','--lr',str(selected['lr']),'--epochs',str(epochs),'--seed','219100'],cwd=REPO,check=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);main(p.parse_args())
