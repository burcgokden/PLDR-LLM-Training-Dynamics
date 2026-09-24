#!/usr/bin/env python3
"""List logical identities or write a decompressed numerical record to stdout."""
import argparse,sys
from pathlib import Path
from evidence_store import Evidence
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-repo',type=Path,required=True);p.add_argument('--list',action='store_true');p.add_argument('--prefix',default='');p.add_argument('identity',nargs='?');a=p.parse_args();e=Evidence(a.data_repo)
    if a.list:print('\n'.join(k for k in sorted(e.records) if k.startswith(a.prefix)))
    elif a.identity:sys.stdout.buffer.write(e.read(a.identity))
    else:p.error('Supply --list or a logical record identity')
