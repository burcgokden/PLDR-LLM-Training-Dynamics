"""The minimal genuine PLDR chain (d_k = 1, one head): the end-to-end
instance of the carrier-modulator analysis, with every constant computable.

Chain (per sample; sequence length T, vocabulary V, scalar head):
    r      = mean_t q_t^2                       (query Gram, fixed scale;
                                                 at d_k = 1 LayerNorm is
                                                 degenerate and is replaced
                                                 by this fixed normalization)
    A      = alpha + J (r - rbar)               (one-unit metric learner:
                                                 the jet chart is exact)
    A_LM   = iswiglu(w A + b) + eps             (adjacency activation)
    A_p    = A_LM ** p                          (learned power)
    G      = a A_p + b_a                        (energy-curvature scalar)
    s_ij   = q_i G k_j                          (power-law attention scores)
    E      = causal softmax(s)                  (attention)
    h_i    = sum_j E_ij v_j                     (linear value path)
    logits = h_i * W_o + c_o                    (readout over V classes)

Loss: blockwise CE at every aligned position (labels in-context: the
target-exposure channel is structurally on) or last-position-only CE
(exposure structurally off).  Token embeddings and q = k = v = embedding
are fixed, so the trainable parameters split exactly into the modulator
(alpha, J) and the carrier block (w, b, p, a, b_a) plus the readout
(W_o, c_o); rho (input spread of r) is set by the embedding gap.

The script trains the chain with full-batch GD (optionally minibatched)
over a declared (eta, rho, loss mode) grid, records the modulator and
carrier trajectories, the EXACT carrier-block sharpness (dense Hessian of
the loss in the five carrier coordinates, no Lanczos), and the collapse
classification, and writes one JSON per cell plus a phase-diagram summary
with the zero-fitted-parameter theory overlay: the predicted boundary is
eta_c = 2 / s_carrier(tilted state), with s_carrier measured as the exact
block sharpness at the converged sub-critical state of the same rho (a
computed constant of the chain, not a regression).

Usage:
    python3 toy_chain.py                 # full declared sweep -> runs/toy/
    python3 toy_chain.py --quick         # coarse grid for smoke testing
"""

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

EPS_ADJ = 1e-9


def iswiglu(x):
    return x * F.silu(x)


class ToyChain:
    def __init__(self, V=2, T=8, N=512, rho=0.25, seed=0,
                 device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.V, self.T, self.N = V, T, N
        self.device = torch.device(device)
        # fixed scalar embeddings: c +/- d with gap d chosen so that the
        # sample-to-sample spread of r matches rho
        c = 1.0
        d = rho * math.sqrt(self.T) / (2 * c)
        self.emb = torch.tensor([c - d, c + d], device=self.device)
        # Markov token stream: P(next == current) = 0.7 (learnable task)
        toks = torch.empty(N, T + 1, dtype=torch.long)
        toks[:, 0] = torch.randint(0, V, (N,), generator=g)
        stay = torch.rand(N, T, generator=g) < 0.7
        flip = torch.randint(1, V, (N, T), generator=g)
        for t in range(1, T + 1):
            toks[:, t] = torch.where(stay[:, t - 1], toks[:, t - 1],
                                     (toks[:, t - 1] + flip[:, t - 1]) % V)
        self.toks = toks.to(self.device)
        self.x = self.toks[:, :-1]              # [N, T] inputs
        self.y = self.toks[:, 1:]               # [N, T] labels
        q = self.emb[self.x]                    # [N, T]
        self.q = q
        self.r = (q ** 2).mean(dim=1)           # [N]
        self.rbar = self.r.mean().item()
        self.rho_emp = self.r.std().item()
        mask = torch.triu(torch.ones(T, T, device=self.device), 1)
        self.neg = mask * -1e9

    def init_params(self, seed=0):
        g = torch.Generator().manual_seed(seed + 1)

        def p(*shape, scale=1.0):
            t = scale * torch.randn(*shape, generator=g)
            return t.to(self.device).requires_grad_(True)

        return {
            "alpha": p(1, scale=0.3), "J": p(1, scale=0.5),
            "w": p(1, scale=0.7), "b": p(1, scale=0.3),
            "pp": p(1, scale=0.3), "a": p(1, scale=0.7),
            "ba": p(1, scale=0.3),
            "Wo": p(self.V, scale=0.5), "co": p(self.V, scale=0.1),
        }

    def forward(self, P, idx=None):
        x = self.x if idx is None else self.x[idx]
        y = self.y if idx is None else self.y[idx]
        r = self.r if idx is None else self.r[idx]
        q = self.q if idx is None else self.q[idx]
        A = P["alpha"] + P["J"] * (r - self.rbar)             # [n]
        ALM = iswiglu(P["w"] * A + P["b"]) + EPS_ADJ          # [n]
        Ap = ALM ** P["pp"]
        G = P["a"] * Ap + P["ba"]                             # [n]
        s = G[:, None, None] * q[:, :, None] * q[:, None, :]  # [n, Tq, Tk]
        E = F.softmax(s + self.neg, dim=-1)
        h = (E * q[:, None, :]).sum(-1)                       # [n, T]
        logits = h[:, :, None] * P["Wo"] + P["co"]            # [n, T, V]
        return logits, y, A

    def loss(self, P, mode="block", idx=None):
        logits, y, A = self.forward(P, idx)
        if mode == "last":
            return F.cross_entropy(logits[:, -1], y[:, -1]), A
        return F.cross_entropy(logits.reshape(-1, self.V),
                               y.reshape(-1)), A


CARRIER = ["w", "b", "pp", "a", "ba"]
MODULATOR = ["alpha", "J"]


def carrier_sharpness(chain, P, mode):
    """EXACT largest eigenvalue of the carrier-block Hessian (dense, 5x5)."""
    names = CARRIER

    def f(v):
        Q = dict(P)
        for i, n in enumerate(names):
            Q[n] = v[i:i + 1]
        return chain.loss(Q, mode)[0]

    v0 = torch.cat([P[n].detach() for n in names]).clone().requires_grad_(True)
    H = torch.autograd.functional.hessian(f, v0)
    return torch.linalg.eigvalsh(H)[-1].item()


def run_cell(chain, eta, mode, steps, seed, batch=None, log_every=25,
             record_sharp=True, params=None, clip=None):
    """Train the chain at step size eta.  With params given, continue from
    that state (the hot protocol: pretrain sub-critically to the tilted
    sharpened state, then switch eta; the crossing event of the mechanism).
    clip: optional gradient value-clipping level (the reference stack's
    entrywise clip), needed above the boundary so the supercritical
    oscillation is bounded instead of diverging in one step."""
    P = params if params is not None else chain.init_params(seed)
    allp = list(P.values())
    g = torch.Generator().manual_seed(seed + 9)
    hist = {"step": [], "loss": [], "J": [], "alpha": [], "a": [],
            "w": [], "pp": [], "s_carrier": [], "dw2": []}
    prev_c = torch.cat([P[n].detach() for n in CARRIER]).clone()
    diverged = False
    for t in range(1, steps + 1):
        idx = (torch.randperm(chain.N, generator=g)[:batch]
               if batch else None)
        L, _ = chain.loss(P, mode, idx)
        if not torch.isfinite(L):
            diverged = True
            break
        for p_ in allp:
            if p_.grad is not None:
                p_.grad = None
        L.backward()
        with torch.no_grad():
            for p_ in allp:
                if p_.grad is not None:
                    gstep = (p_.grad.clamp(-clip, clip) if clip
                             else p_.grad)
                    p_ -= eta * gstep
        if t % log_every == 0 or t == 1:
            cur_c = torch.cat([P[n].detach() for n in CARRIER]).clone()
            hist["step"].append(t)
            hist["loss"].append(L.item())
            hist["J"].append(P["J"].item())
            hist["alpha"].append(P["alpha"].item())
            hist["a"].append(P["a"].item())
            hist["w"].append(P["w"].item())
            hist["pp"].append(P["pp"].item())
            hist["dw2"].append(((cur_c - prev_c) ** 2).sum().item())
            if record_sharp:
                hist["s_carrier"].append(carrier_sharpness(chain, P, mode))
            prev_c = cur_c
    out = {"eta": eta, "mode": mode, "steps": steps, "seed": seed,
           "batch": batch, "clip": clip, "rho_emp": chain.rho_emp,
           "diverged": diverged, "hist": hist}
    if not diverged:
        tail = max(1, len(hist["J"]) // 4)

        def med(vals):
            v = sorted(vals)
            return float(v[len(v) // 2])

        out["J_end"] = med([abs(j) for j in hist["J"][-tail:]])
        out["u_end"] = out["J_end"] * chain.rho_emp
        out["a_end"] = med([abs(v) for v in hist["a"][-tail:]])
        out["s_end"] = (med(hist["s_carrier"][-tail:])
                        if hist["s_carrier"] else None)
    out["params_end"] = ({k: v.detach().tolist() for k, v in P.items()}
                         if not diverged else None)
    return out, P


def clone_params(P):
    return {k: v.detach().clone().requires_grad_(True)
            for k, v in P.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default="runs/toy")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rhos = [0.1, 0.25, 0.5] if not args.quick else [0.25]
    etas = ([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
            if not args.quick else [0.05, 0.8])
    eta0 = etas[0]
    modes = ["block", "last"] if not args.quick else ["block"]

    summary = []
    for rho in rhos:
        chain = ToyChain(rho=rho, seed=args.seed)
        # cold sub-critical reference: converges to the tilted sharpened
        # state; its exact carrier sharpness gives the predicted boundary
        # (computed constants of the chain, nothing fitted)
        ref, Pref = run_cell(chain, eta0, "block", args.steps, args.seed,
                             clip=1.0)
        J_ref, s_ref = ref.get("J_end"), ref.get("s_end")
        a_ref = ref.get("a_end")
        eta_c_pred = (2.0 / s_ref) if s_ref else None
        with open(os.path.join(args.outdir,
                               f"toy_ref_rho{rho}.json"), "w") as f:
            json.dump(ref, f)
        # cold cells: GD from initialization (no crossing event; the
        # self-regulated regime).  hot cells: continue from the tilted
        # sharpened state at the new step size (the crossing event; the
        # protocol of the mechanism claim).
        for protocol in ["cold", "hot"]:
            for mode in modes:
                for eta in etas:
                    start = (clone_params(Pref) if protocol == "hot"
                             else None)
                    cell, _ = run_cell(chain, eta, mode, args.steps,
                                       args.seed, params=start, clip=1.0)
                    cell["protocol"] = protocol
                    cell["rho"] = rho
                    cell["J_ref"] = J_ref
                    cell["a_ref"] = a_ref
                    cell["s_ref"] = s_ref
                    cell["eta_c_pred"] = eta_c_pred
                    if not cell["diverged"] and J_ref:
                        cell["collapsed"] = cell["J_end"] < 0.1 * J_ref
                        cell["J_ratio"] = cell["J_end"] / J_ref
                        cell["a_ratio"] = (cell["a_end"] / a_ref
                                           if a_ref else None)
                    name = f"toy_{protocol}_rho{rho}_{mode}_eta{eta}.json"
                    with open(os.path.join(args.outdir, name), "w") as f:
                        json.dump(cell, f)
                    summary.append({k: cell.get(k) for k in
                                    ["protocol", "rho", "mode", "eta",
                                     "diverged", "collapsed", "J_end",
                                     "u_end", "a_end", "s_end", "J_ratio",
                                     "a_ratio", "eta_c_pred", "J_ref"]})
                    print(f"{protocol} rho={rho} mode={mode} eta={eta}: "
                          f"div={cell['diverged']} "
                          f"J_ratio={cell.get('J_ratio')} "
                          f"a_ratio={cell.get('a_ratio')} "
                          f"s_end={cell.get('s_end')} "
                          f"(pred eta_c={eta_c_pred})", flush=True)
    with open(os.path.join(args.outdir, "phase.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
