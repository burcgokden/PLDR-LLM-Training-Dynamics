#!/usr/bin/env python3
"""Verify complete scoped inventory, manifest bindings and replay interval unions."""
import argparse
from pathlib import Path
from numerical_validation import load_json_strict
from model_rg.provenance import sha256,write_json
