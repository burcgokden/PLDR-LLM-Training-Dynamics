"""Coordinate two local GPUs after their separate training queues complete."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]

def execute(script,study,name,device,kind=None):
    args=[sys.executable,str(REPO/'scripts'/script),'--study',str(study),'--name',name,'--device',device]
    if kind:args+=['--kind',kind]
    subprocess.run(args,cwd=REPO,check=True)

def main(a):
    s=Path(a.study);device='cuda:0' if a.worker=='language' else 'cuda:1';prerequisite='general-lowrank' if a.worker=='language' else 'physical-s1'
    terminal=s/'runs'/prerequisite/'result.json'
    if not terminal.is_file() or json.loads(terminal.read_text()).get('status')!='complete':raise ValueError('Prerequisite has not completed successfully: '+str(terminal))
    if a.worker=='language':
        for name in ['base5']+[task+suffix for suffix in ['-s0','-lowrank'] for task in ['technical','narrative','mixture','general']]:execute('assess_released_adaptation.py',s,name,device,'language')
    else:
        for name in ['base5','physical-s0','physical-s1']:
            for kind in ['spin','exact']:execute('assess_released_adaptation.py',s,name,device,kind)
            execute('observe_released_geometry.py',s,name,device)
    state=s/'generation-queue.json';lock=s/'generation-queue.lock'
    while True:
        with lock.open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX)
            if state.exists():jobs=json.loads(state.read_text())
            else:jobs=[dict(name=name,kind=kind,status='pending') for kind in ['critical','thermal'] for name in ['physical-s0','physical-s1','base5']]
            available=[j for j in jobs if j['status']=='pending']
            if not available:break
            job=available[0];job.update(status='running',worker=a.worker,pid=os.getpid(),start=time.time());state.write_text(json.dumps(jobs,indent=2)+'\n');chosen=(job['name'],job['kind'])
        try:execute('assess_released_adaptation.py',s,chosen[0],device,chosen[1]);status='complete'
        except Exception:status='failed';raise
        finally:
            with lock.open('a') as stream:
                fcntl.flock(stream,fcntl.LOCK_EX);jobs=json.loads(state.read_text())
                job=next(j for j in jobs if (j['name'],j['kind'])==chosen);job.update(status=status,end=time.time());state.write_text(json.dumps(jobs,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--worker',choices=['language','physical'],required=True);main(p.parse_args())
