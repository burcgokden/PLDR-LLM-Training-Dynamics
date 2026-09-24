#!/usr/bin/env python
"""Analyze the complete recorded dynamics study; every required input must exist."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output-root');args=parser.parse_args()
    study=Path(args.root)/'criticality-dynamics-20260906'
    for name in ['launcher-pilot','launcher-replication','launcher-long-horizon','collective-return','tangent-replication','transport-arithmetic','tangent-resolution','row-transport','row-transport-transfer','row-projection','row-adjoint','row-adjoint-source-control','row-gradient-projection']:
        if json.loads((study/(name+'.json')).read_text())['status']!='complete':raise AssertionError('Incomplete required evidence: '+name)
    destination=Path(args.output_root) if args.output_root else study/'analysis'
    commands=[('analyze_dynamics.py','pilot',['--pattern','pilot-*']),
              ('analyze_dynamics_replication.py','replication',[]),
              ('analyze_dynamics.py','long',['--pattern','long-*']),
              ('analyze_dynamics_controls.py','controls',[]),
              ('analyze_row_transport.py','row-transport-summary',[]),
              ('analyze_row_projection.py','row-projection-summary',[]),
              ('analyze_optimizer_scales.py','optimizer-scales',[]),
              ('analyze_row_adjoint.py','row-adjoint-summary',['--include-source-control']),
              ('analyze_row_gradient_projection.py','row-gradient-projection-summary',[]),
              ('analyze_metric_collectives.py','metric-collectives',[])]
    for script,label,extra in commands:
        output=destination/label
        subprocess.run([sys.executable,'scripts/'+script,'--root',args.root,'--output',str(output),*extra],check=True)


if __name__=='__main__':main()
