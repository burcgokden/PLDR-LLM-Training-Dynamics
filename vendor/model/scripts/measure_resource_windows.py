#!/usr/bin/env python
"""Exact finite remaining-index law distances, not measured output-law distances."""
import argparse
import math
from pathlib import Path
from model_rg.provenance import sha256,write_json


def index_tv(remaining,draws):
    if remaining<1 or not 0<=draws<=remaining:raise ValueError('Invalid remaining population or draws')
    return -math.expm1(math.fsum(math.log1p(-j/remaining) for j in range(draws)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    rows=[]
    for population,t in [(4194304,8192),(4194304,32768),(524288,2048)]:
        remaining=population-32*t
        for h in [1,4,16,64]:
            draws=32*h;tv=index_tv(remaining,draws);bound=draws*(draws-1)/(2*remaining)
            assert 0<=tv<=min(1,bound)+1e-12
            rows.append(dict(population=population,incoming_update=t,remaining=remaining,horizon=h,
                             draws=draws,exact_index_tv=tv,pair_bound=bound))
    write_json(out,dict(status='passed',rows=rows,producer_sha256=sha256(__file__),
        scope='Exact ordered index-law TV. A common native observation obeys an upper bound by data processing; its actual law distance is not measured.'))


if __name__=='__main__':main()
