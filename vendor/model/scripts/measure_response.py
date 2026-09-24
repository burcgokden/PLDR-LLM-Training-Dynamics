#!/usr/bin/env python
"""All-head, full-decoder Fisher response and held-out finite perturbations."""
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.native import NativeModel
from model_rg.provenance import environment, sha256, source_manifest, write_json


def fisher(logits, jac):
    p = logits.double().softmax(-1)
    j = jac.double()
    mean = p @ j
    centered = j - mean
    return centered.T @ (p[:, None] * centered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--documents", type=int, default=32)
    ap.add_argument("--offset", type=int, default=512)
    ap.add_argument("--step", type=float, default=.01)
    ap.add_argument("--length", type=int, default=128)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = NativeModel(args.model, args.device)
    tokens = np.load(Path(args.data) / "tokens.npy")[args.offset:args.offset+args.documents, :args.length]
    shape = (m.config.num_hidden_layers, m.config.num_attention_heads)
    n = int(np.prod(shape))
    eye = torch.eye(n, dtype=torch.float32, device=args.device).reshape(n, *shape)
    rng = np.random.default_rng(117)
    directions = rng.normal(size=(8,n))
    directions /= np.sqrt(np.mean(directions**2, axis=1, keepdims=True))
    dirs = torch.tensor(directions, dtype=torch.float32, device=args.device).reshape(8,*shape)
    amplitudes = [.003,.01,.03,.1]
    metrics, jacobians, baselines, actual, jvp_checks, fd_checks = [], [], [], [], [], []
    start = time.time()
    for doc, row in enumerate(tokens):
        ids = torch.tensor(row, dtype=torch.long, device=args.device)[None]
        with torch.no_grad():
            baseline = m.logits(ids)[0].double()
            jac = torch.empty((m.config.vocab_size, n), dtype=torch.float32, device=args.device)
            for k in range(0,n,14):
                e = eye[k:k+14]
                eta = torch.cat([args.step*e,-args.step*e])
                yp = m.logits(ids.repeat(len(eta),1),eta)
                size = len(e)
                jac[:,k:k+size] = ((yp[:size]-yp[size:])/(2*args.step)).T
            h = fisher(baseline, jac)
            metrics.append(h.cpu().numpy())
            jacobians.append(jac.cpu().numpy())
            baselines.append(baseline.cpu().numpy())
            this = []
            p = baseline.softmax(-1)
            logp = baseline.log_softmax(-1)
            for amp in amplitudes:
                logits = m.logits(ids.repeat(8,1),amp*dirs).double()
                kl = (p[None] * (logp[None]-logits.log_softmax(-1))).sum(-1)
                this.append(kl.cpu().numpy())
            actual.append(this)
            eta = torch.cat([.005*dirs, -.005*dirs])
            val = m.logits(ids.repeat(16,1),eta).double()
            jdir = (val[:8]-val[8:])/.01
            pred = torch.tensor(directions, device=args.device) @ jac.double().T
            # Remove the softmax gauge and weight in the predictive Fisher norm.
            jdir -= (jdir @ p)[:,None]
            pred -= (pred @ p)[:,None]
            fd_checks.append((((jdir-pred).square()@p)/(jdir.square()@p)).sqrt().cpu().numpy())
        if doc < 4:
            for d in range(2):
                zero = torch.zeros(shape, dtype=torch.float32, device=args.device)
                _, tangent = torch.autograd.functional.jvp(lambda eta: m.logits(ids,eta),zero,dirs[d],strict=False)
                tj = tangent[0].double()
                pj = jac.double() @ dirs[d].double().flatten()
                tj -= p @ tj
                pj -= p @ pj
                jvp_checks.append({"document":doc,"direction":d,
                    "relative_fisher_error": float((((tj-pj).square()@p)/(tj.square()@p)).sqrt().item())})
        print(f"document {doc+1}/{len(tokens)} elapsed={time.time()-start:.1f}s",flush=True)
    np.savez_compressed(out/"response.npz", fisher=np.asarray(metrics), logits=np.asarray(baselines),
                        jacobian=np.asarray(jacobians), directions=directions,
                        amplitudes=np.asarray(amplitudes), kl=np.asarray(actual),
                        half_step_direction_error=np.asarray(fd_checks))
    write_json(out/"manifest.json", {"schema":"model-rg-response-v1","arguments":vars(args),
        "environment":environment(),"source_files":source_manifest(),"autodiff_checks":jvp_checks,
        "model_sha256":sha256(Path(args.model)/"model.safetensors"),
        "data_manifest_sha256":sha256(Path(args.data)/"manifest.json"),
        "response_sha256":sha256(out/"response.npz"),"seconds":time.time()-start,
        "max_gpu_memory_bytes":torch.cuda.max_memory_allocated(args.device)})


if __name__ == "__main__":
    main()
