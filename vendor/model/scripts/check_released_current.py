"""Read-only current execution preflight, without native models or updates."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.released_qualification import configure,validate,assessment_identity
from model_rg.provenance import sha256,write_json

def main(a):
    configure();s=Path(a.study);records=[]
    for branch,entry in [('full','train_released_adaptation'),('factor','train_released_lowrank')]:
        q,p=validate(s,5,branch,entry);records.append(dict(branch=branch,qualification=str(p),sha256=sha256(p)))
    assessment_identity(s,'base5')
    result=dict(status='passed',scientific_updates=0,native_loads=0,contracts=records,checker_sha256=sha256(__file__))
    if a.output:write_json(a.output,result)
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output');main(p.parse_args())
