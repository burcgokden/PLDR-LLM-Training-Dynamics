#!/usr/bin/env python3
"""Analyze every frozen paired corner on the disjoint single-pass suffix."""
from companion_paths import configured_path
import argparse,json,shutil
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from analyze_potential_factorial import factorial,LABELS
from analyze_potential_block_area import reduce_path
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(configured_path('data:model'))


