"""All-runs summary table (table_runs.tex + runs_summary.csv),
separated from the wave-1 figure generation so that no invocation
overwrites another target's artifact.
Usage: python3 analysis/make_table.py <run> [<run> ...]
(defaults to every directory under runs/ that contains a log.jsonl,
which excludes the minimal-instance sweep directory)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import RUNS

