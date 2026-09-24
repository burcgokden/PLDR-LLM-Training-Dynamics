#!/usr/bin/env python3
"""Exercise actual outcome and accounting CLIs in ordinary and optimized Python."""
from companion_paths import child_pythonpath
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
from numerical_validation import load_json_strict
from model_rg.provenance import sha256,write_json
REPO=Path(__file__).resolve().parents[1]


