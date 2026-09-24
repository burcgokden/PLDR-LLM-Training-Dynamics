#!/usr/bin/env python
"""Collect every head and final-position predictions, with external next tokens."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.native import NativeModel
from model_rg.provenance import environment, sha256, source_manifest, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--length", type=int, default=128)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = NativeModel(args.model, args.device)
    tokens = np.load(Path(args.data) / "tokens.npy")[:args.limit]
    feats, g, a = [], [], []
    start = time.time()
    with torch.no_grad():
        for k in range(0, len(tokens), args.batch_size):
            ids = torch.tensor(tokens[k:k + args.batch_size, :args.length], dtype=torch.long, device=args.device)
            target = torch.tensor(tokens[k:k + args.batch_size, args.length], dtype=torch.long, device=args.device)
            f, diag = model.features(ids, target)
            if not torch.isfinite(f).all():
                raise FloatingPointError("Nonfinite features")
            feats.append(f.double().cpu().numpy())
            g.append(diag["g_rms"].cpu().numpy())
            a.append(diag["a_row_energy"].cpu().numpy())
            if k % (args.batch_size * 16) == 0:
                print(f"{k}/{len(tokens)} length={args.length} seconds={time.time()-start:.1f}", flush=True)
    np.savez_compressed(root / "features.npz", features=np.concatenate(feats), g_rms=np.concatenate(g), a_row_energy=np.concatenate(a))
    write_json(root / "manifest.json", {"schema": "model-rg-features-v1", "arguments": vars(args),
        "environment": environment(), "source_files": source_manifest(), "feature_names": model.feature_names(),
        "model_sha256": sha256(Path(args.model) / "model.safetensors"),
        "native_code_sha256": sha256(Path(args.model) / "modeling_pldrllm.py"),
        "data_manifest_sha256": sha256(Path(args.data) / "manifest.json"),
        "features_sha256": sha256(root / "features.npz"), "seconds": time.time()-start,
        "documents": len(tokens), "parameters": sum(p.numel() for p in model.model.parameters()),
        "max_gpu_memory_bytes": torch.cuda.max_memory_allocated(args.device)})


if __name__ == "__main__":
    main()
