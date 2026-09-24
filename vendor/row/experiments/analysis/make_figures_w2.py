"""Wave-2 figures: warm-up scan (P4), regrowth (P5), tilt decay (P6),
seed replicates."""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import (RUNS, OUT, load_run, series, min_layer, smooth,
                          BLUES, ORANGES)

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.4, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.1,
    "figure.dpi": 150, "savefig.bbox": "tight",
})


def fig_warmup(names):
    """Warm-up scan at eta=1e-3: loss spikes, sharpness crashes, collapse."""
    runs = [(n, *load_run(n)) for n in names]
    runs.sort(key=lambda r: r[1]["warmup"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, (n, c, s, f) in enumerate(runs):
        frac = 0.3 + 0.6 * i / max(len(runs) - 1, 1)
        col = plt.cm.viridis(frac)
        lab = f"$T_w$={c['warmup']}"
        x, y = series(s, "loss")
        axes[0].plot(*smooth(x, np.array(y), 51), color=col, label=lab,
                     lw=0.9)
        if any("lam_upd" in r for r in s):  # Lanczos-instrumented runs only
            x, y = series(s, "lam_full")
            axes[1].plot(x, np.maximum(np.array(y, dtype=float), 1e-1),
                         color=col, lw=0.9)
        x, y = min_layer(s, "rowmap_sigma_med")
        axes[2].plot(x, np.maximum(y, 1e-8), color=col, lw=0.9)
    axes[0].set(xlabel="step", ylabel="train loss (smoothed)",
                xlim=(0, 6000), ylim=(4.5, 8))
    axes[1].set(xlabel="step", ylabel="$\\lambda_{\\max}$ (Lanczos)",
                yscale="log", xlim=(0, 6000))
    axes[2].set(xlabel="step",
                ylabel="min-layer median $\\sigma(J_\\varphi)$",
                yscale="log")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_w2_warmup.pdf"))
    fig.savefig(os.path.join(OUT, "fig_w2_warmup.png"), dpi=110)
    plt.close(fig)


def fig_regrow(name, source_name):
    """P5: continued training at constant small lr from collapsed ckpt."""
    c, s, f = load_run(name)
    cs, ss, fs = load_run(source_name)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    x0, y0 = min_layer(ss, "rowmap_sigma_med")
    x1, y1 = min_layer(s, "rowmap_sigma_med")
    axes[0].plot(x0, np.maximum(y0, 1e-8), color="#d95f02",
                 label="source run (warm-up + cosine)")
    axes[0].plot(x1 + x0[-1], np.maximum(y1, 1e-8), color="#1f78b4",
                 label=f"continued at constant $\\eta$={c['lr']:g}")
    axes[0].axvline(x0[-1], color="k", lw=0.6, ls=":")
    axes[0].set(xlabel="step (concatenated)",
                ylabel="min-layer median $\\sigma(J_\\varphi)$",
                yscale="log")
    axes[0].legend(fontsize=7, frameon=False)
    x0, y0 = min_layer(ss, "m_A_layers")
    x1, y1 = min_layer(s, "m_A_layers")
    axes[1].plot(x0, np.maximum(y0, 1e-8), color="#d95f02")
    axes[1].plot(x1 + x0[-1], np.maximum(y1, 1e-8), color="#1f78b4")
    axes[1].axvline(x0[-1], color="k", lw=0.6, ls=":")
    axes[1].set(xlabel="step (concatenated)",
                ylabel="min-layer $\\hat m_A$", yscale="log")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_w2_regrow.pdf"))
    fig.savefig(os.path.join(OUT, "fig_w2_regrow.png"), dpi=110)
    plt.close(fig)


def fig_tilt(names_labels):
    """P6: per-layer tilt along training, collapsing vs non-collapsing."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    for name, lab, col in names_labels:
        p = os.path.join(RUNS, name, "tilt.json")
        if not os.path.exists(p):
            continue
        data = json.load(open(p))
        pts = []
        for ck, v in data.items():
            step = (int(ck.split("_")[1].split(".")[0])
                    if "final" not in ck else None)
            cs = [l["C_fro"] for l in v["layers"]]
            pts.append((step, cs[0], cs[-1]))
        laststep = max(s for s, _, _ in pts if s is not None) * 1.5
        pts = [(s if s is not None else laststep, a, b) for s, a, b in pts]
        pts.sort()
        axes[0].plot([q[0] for q in pts], [q[1] for q in pts], "o-", ms=3,
                     color=col, label=lab)
        axes[1].plot([q[0] for q in pts], [q[2] for q in pts], "o-", ms=3,
                     color=col)
    axes[0].set(xlabel="step (final at 1.5$\\times$)",
                ylabel="$\\|C\\|_F$, layer 1", yscale="log")
    axes[1].set(xlabel="step (final at 1.5$\\times$)",
                ylabel="$\\|C\\|_F$, layer 3", yscale="log")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_w2_tilt.pdf"))
    fig.savefig(os.path.join(OUT, "fig_w2_tilt.png"), dpi=110)
    plt.close(fig)


def fig_eos(pairs):
    """P3: Lanczos sharpness vs the stability curves 2/eta and 38/eta,
    collapsing vs non-collapsing (runs with Lanczos instrumentation)."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.4 * len(pairs), 3.2),
                             squeeze=False)
    for k, (name, lab) in enumerate(pairs):
        ax = axes[0, k]
        c, s, f = load_run(name)
        x, lam = series(s, "lam_full")
        lam = np.array(lam, dtype=float)
        xs, lrs = series(s, "lr")
        lr = np.interp(x, xs, np.array(lrs, dtype=float))
        ax.plot(x, np.maximum(lam, 1e-1), ".-", ms=2.5, lw=0.7,
                color="#333333", label="$\\lambda_{\\max}$ (Lanczos)")
        ax.plot(x, 2 / lr, lw=0.9, color="#1f78b4", label="$2/\\eta_t$")
        ax.plot(x, 38 / lr, lw=0.9, ls="--", color="#d95f02",
                label="$38/\\eta_t$")
        ax.set(xlabel="step", yscale="log", title=lab)
        ax.title.set_size(8)
        if k == 0:
            ax.set_ylabel("curvature scale")
            ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_w2_eos.pdf"))
    fig.savefig(os.path.join(OUT, "fig_w2_eos.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    warm = [n for n in os.listdir(RUNS)
            if n.startswith("w2-lr1e-3-w") and "s2" not in n
            and "s3" not in n] + ["w1-lr1e-3-w1000"]
    warm = [n for n in warm if os.path.exists(
        os.path.join(RUNS, n, "log.jsonl"))]
    fig_warmup(warm)
    if os.path.exists(os.path.join(RUNS, "w3-regrow-lr1e-4", "log.jsonl")):
        fig_regrow("w3-regrow-lr1e-4", "w1-lr1e-3-w1000")
    fig_tilt([
        ("w1-lr1e-4-w1000", "$\\eta_{\\max}$=1e-4 (non-coll.)", "#a6cee3"),
        ("w1-lr3e-4-w1000", "$\\eta_{\\max}$=3e-4 (non-coll.)", "#1f78b4"),
        ("w1-lr1e-3-w1000", "$\\eta_{\\max}$=1e-3 (coll.)", "#fdbf6f"),
        ("w1-lr1.5e-3-w1000", "$\\eta_{\\max}$=1.5e-3 (coll.)", "#ff7f00"),
        ("w1-lr2e-3-w1000", "$\\eta_{\\max}$=2e-3 (coll.)", "#d95f02"),
    ])
    eos_pairs = []
    for n, lab in [("w2-lr3e-4-w1000-s2",
                    "$\\eta_{\\max}$=3e-4 (non-collapsing)"),
                   ("w2-lr1e-3-w1000-s2",
                    "$\\eta_{\\max}$=1e-3 (collapsing)")]:
        if os.path.exists(os.path.join(RUNS, n, "log.jsonl")):
            eos_pairs.append((n, lab))
    if eos_pairs:
        fig_eos(eos_pairs)
    print("wave-2 figures written")
