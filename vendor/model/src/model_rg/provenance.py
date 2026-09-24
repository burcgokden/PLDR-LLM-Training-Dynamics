"""Small, explicit provenance records. No credentials or source text are logged."""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, value: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(p)


def environment() -> dict:
    import numpy as np
    import torch
    import transformers
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}


def source_manifest() -> dict:
    root = Path(__file__).resolve().parents[2]
    files = sorted(p for d in (root / "src", root / "scripts") for p in d.rglob("*.py"))
    return {str(p.relative_to(root)): sha256(p) for p in files}
