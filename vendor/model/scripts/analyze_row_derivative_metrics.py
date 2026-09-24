#!/usr/bin/env python3
"""Complete descriptive matrix-derivative errors for the eight retained pairs."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict
from analyze_row_path_confirmation import transport




