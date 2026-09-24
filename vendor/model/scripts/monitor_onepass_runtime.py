#!/usr/bin/env python
"""Record controller liveness, progress and failures without changing experiments."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256, write_json

CONTROLLERS={
    'launcher-onepass.json':{'run_onepass_study.py'},
    'launcher-onepass-observations.json':{'run_onepass_observations.py','resume_onepass_observations.py'},
    'launcher-onepass-analyses.json':{'run_onepass_analyses.py'},
    'launcher-repetition-analysis.json':{'run_repetition_analysis.py','resume_onepass_repetition_analysis.py'},
    'launcher-prefix-risk-comparison.json':{'run_prefix_risk_comparison.py'},
    'launcher-onepass-schedule-comparison.json':{'run_onepass_schedule_comparison.py'},
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--output',required=True);p.add_argument('--interval',type=int,default=60)
    a=p.parse_args()
    if a.interval<15:raise ValueError('Avoid excessive polling')
    repo=Path(__file__).resolve().parents[1];root=Path(a.root).resolve();study=root/a.study
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    write_json(output/'configuration.json',dict(at=datetime.now(timezone.utc).isoformat(),
        arguments=vars(a),source_sha256=sha256(__file__),status_source_sha256=sha256(repo/'scripts/onepass_status.py'),
        scope='Read-only operational monitoring. Record failures and missing controller processes; never alter scientific results, waive checks or silently restart failed analyses. Local records do not imply that a human or assistant has read each sample.'))
    os.nice(10);missing_counts={};previous_alerts=None
    wanted=set().union(*CONTROLLERS.values(),{'train_onepass_regimes.py'})
    while True:
        processes=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                args=(proc/'cmdline').read_bytes().decode().split('\0')
                matches=[Path(v).name for v in args if Path(v).name in wanted]
                if not matches or '--root' not in args or Path(args[args.index('--root')+1]).resolve()!=root:continue
                if '--study' in args and args[args.index('--study')+1]!=a.study:continue
                values={k.removeprefix('--'):args[args.index(k)+1] for k in ['--run-id','--device'] if k in args}
                processes.append(dict(pid=int(proc.name),script=matches[0],**values))
            except (OSError,UnicodeError,ValueError,IndexError):continue
        live={p['script'] for p in processes};alerts=[];states={}
        for filename,scripts in CONTROLLERS.items():
            path=study/filename
            try:state=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError) as exc:
                alerts.append(dict(controller=filename,kind='unreadable_record',error=str(exc)));continue
            status=state['status'];states[filename]=dict(status=status,records=len(state.get('records',[])))
            if 'fail' in status:alerts.append(dict(controller=filename,kind='reported_failure',status=status))
            missing=status!='complete' and not scripts.intersection(live)
            missing_counts[filename]=missing_counts.get(filename,0)+1 if missing else 0
            if missing_counts[filename]>=2:
                alerts.append(dict(controller=filename,kind='controller_process_missing',status=status))
        command=[sys.executable,str(repo/'scripts/onepass_status.py'),'--root',str(root),'--study',a.study,'--json']
        result=subprocess.run(command,cwd=repo,capture_output=True,text=True)
        progress=json.loads(result.stdout) if result.returncode==0 else {'error':result.stderr[-2000:]}
        snapshot=dict(at=datetime.now(timezone.utc).isoformat(),controllers=states,
            processes=processes,alerts=alerts,progress=progress)
        write_json(output/'latest.json',snapshot)
        with (output/'history.jsonl').open('a') as log:log.write(json.dumps(snapshot,sort_keys=True)+'\n')
        if alerts!=previous_alerts:
            event=dict(at=snapshot['at'],alerts=alerts)
            with (output/'alerts.jsonl').open('a') as log:log.write(json.dumps(event,sort_keys=True)+'\n')
            print(json.dumps(event),flush=True);previous_alerts=alerts
        if len(states)==len(CONTROLLERS) and all(s['status']=='complete' for s in states.values()):
            write_json(output/'complete.json',dict(status='complete',at=snapshot['at'],scope='All six monitored experiment and analysis controllers report completion. Manuscript interpretation and release verification are separate.'))
            return
        time.sleep(a.interval)


if __name__=='__main__':main()
