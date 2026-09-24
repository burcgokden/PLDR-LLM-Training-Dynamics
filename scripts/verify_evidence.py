#!/usr/bin/env python3
"""Check every numerical record and the display/claim coverage index."""
import argparse,json
from pathlib import Path
from evidence_store import Evidence
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-repo',type=Path,required=True);p.add_argument('--extract',type=Path);p.add_argument('--prefix',default='');a=p.parse_args()
    print(json.dumps(Evidence(a.data_repo).verify(a.extract,a.prefix),indent=2))
