"""Figures and tables for the training-dynamics paper, from run logs.

Conventions: non-collapsing runs in blues (light->dark with increasing
lr), collapsing runs in oranges/reds (light->dark), colorblind-safe
pairing; one axis per panel; thin lines; recessive grid.

Phase criterion (fixed for this scale, disclosed in the paper): a run is
"collapsing" iff the median row-map Jacobian singular value of its most
collapsed layer, averaged over the last quarter of training, is < 0.1.
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "docs", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

BLUES = plt.cm.Blues
ORANGES = plt.cm.Oranges


def load_run(name):
    recs = []
    with open(os.path.join(RUNS, name, "log.jsonl")) as f:
        for line in f:
            recs.append(json.loads(line))
    config = next(r for r in recs if r.get("event") == "config")
    final = next((r for r in recs if r.get("event") == "final"), None)
    steps = [r for r in recs if "step" in r]
    return config, steps, final


def series(steps, key):
    rs = [r for r in steps if key in r and r[key] is not None]
    return (np.array([r["step"] for r in rs]),
            [r[key] for r in rs])


def smooth(x, y, w=101):
    if len(y) < 2 * w:
        return x, np.asarray(y)
    k = np.ones(w) / w
    ys = np.convolve(y, k, mode="valid")
    xs = x[w // 2 : w // 2 + len(ys)]
    return xs, ys


def min_layer(steps, key):
    """min over layers of a per-layer list-valued diagnostic."""
    xs, ys = series(steps, key)
    return xs, np.array([min(v) for v in ys])


def collapse_stat(steps, total_steps):
    xs, ys = min_layer(steps, "rowmap_sigma_med")
    if len(xs) == 0:
        return np.nan
    lastq = ys[xs > 0.75 * total_steps]
    return float(np.mean(lastq)) if len(lastq) else np.nan


def classify(config, steps):
    s = collapse_stat(steps, config["steps"])
    return "flat ($\\ge$1 layer)" if s < 0.1 else "sub-critical"


def fig_overview(run_names, tag="wave1", detail_run=None):
    runs = []
    for n in run_names:
        c, s, f = load_run(n)
        runs.append((n, c, s, f))
    runs.sort(key=lambda r: r[1]["lr"])

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))

    for i, (n, c, s, f) in enumerate(runs):
        phase = classify(c, s)
        frac = 0.35 + 0.55 * i / max(len(runs) - 1, 1)
        col = BLUES(frac) if phase == "sub-critical" else ORANGES(frac)
        lab = f"$\\eta_{{\\max}}$={c['lr']:g}"
        x, y = series(s, "loss")
        axes[0, 0].plot(*smooth(x, y), color=col, label=lab, lw=1.0)
        x, y = min_layer(s, "m_A_layers")
        axes[0, 1].plot(x, np.maximum(y, 1e-8), color=col, lw=1.0)
        x, y = min_layer(s, "rowmap_sigma_med")
        axes[1, 0].plot(x, np.maximum(y, 1e-8), color=col, lw=1.0)

    if detail_run is not None:
        c, s, f = load_run(detail_run)
        xs, ys = series(s, "rowmap_sigma_med")
        ys = np.array(ys)
        for li in range(ys.shape[1]):
            axes[1, 1].plot(xs, np.maximum(ys[:, li], 1e-8),
                            lw=1.0, label=f"layer {li+1}",
                            color=ORANGES(0.35 + 0.3 * li))
        axes[1, 1].legend(fontsize=7, frameon=False)
        axes[1, 1].set_title(f"per layer, $\\eta_{{\\max}}$="
                             f"{c['lr']:g}", fontsize=8)

    axes[0, 0].set(xlabel="step", ylabel="train loss (smoothed)")
    axes[0, 0].set_ylim(4.4, 7.5)
    axes[0, 1].set(xlabel="step",
                   ylabel="min-layer $\\hat m_A$", yscale="log")
    axes[1, 0].set(xlabel="step",
                   ylabel="min-layer median $\\sigma(J_\\varphi)$",
                   yscale="log")
    axes[1, 1].set(xlabel="step",
                   ylabel="median $\\sigma(J_\\varphi)$", yscale="log")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_overview.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_overview.png"), dpi=110)
    plt.close(fig)


def fig_transition(run_names, tag="wave1"):
    rows = []
    for n in run_names:
        c, s, f = load_run(n)
        if f is None:
            continue
        val = [r["val_loss"] for r in s if "val_loss" in r]
        xs, mmin = min_layer(s, "m_A_layers")
        lastq = mmin[xs > 0.75 * c["steps"]]
        rows.append((c["lr"], f["m_gen"]["A"], f["m_gen"]["GLM"],
                     val[-1] if val else np.nan,
                     float(np.mean(lastq)) if len(lastq) else np.nan))
    rows.sort()
    lr = [r[0] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.0))
    ax[0].plot(lr, [max(r[1], 1e-12) for r in rows], "o-", color="#333333",
               ms=4, label="$m^{\\rm gen}_A$ (model)")
    ax[0].plot(lr, [max(r[2], 1e-12) for r in rows], "s--", color="#888888",
               ms=4, label="$m^{\\rm gen}_{G_{\\rm LM}}$ (model)")
    ax[0].plot(lr, [max(r[4], 1e-12) for r in rows], "^:", color="#d95f02",
               ms=4, label="$\\hat m_A$ (min layer)")
    ax[0].set(xlabel="$\\eta_{\\max}$", ylabel="final order parameter",
              xscale="log", yscale="log")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(lr, [r[3] for r in rows], "o-", color="#333333", ms=4)
    ax[1].set(xlabel="$\\eta_{\\max}$", ylabel="final val loss", xscale="log")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_transition.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_transition.png"), dpi=110)
    plt.close(fig)


def fig_tracking(run_names, tag="wave1"):
    """Sanity-check scatter: per-layer order parameter vs per-layer
    Jacobian scale, annotated with RUN-LEVEL correlation statistics
    (one correlation per run; pooled point counts are pseudoreplicated
    and are not reported)."""
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    run_rs = []
    for n in run_names:
        c, s, f = load_run(n)
        J, M = [], []
        for r in s:
            if "m_A_layers" in r and "rowmap_sigma_med" in r:
                for m, j in zip(r["m_A_layers"], r["rowmap_sigma_med"]):
                    J.append(max(j, 1e-8))
                    M.append(max(m, 1e-8))
        J, M = np.array(J), np.array(M)
        ax.plot(J, M, ".", ms=2, alpha=0.25, color="#1f78b4",
                rasterized=True)
        keep = (J > 1e-6) & (M > 1e-6)
        if keep.sum() > 10:
            run_rs.append(np.corrcoef(np.log10(J[keep]),
                                      np.log10(M[keep]))[0, 1])
    run_rs = np.array(run_rs)
    ax.text(0.05, 0.92,
            f"run-level $r$: median {np.median(run_rs):.2f}, "
            f"range [{run_rs.min():.2f}, {run_rs.max():.2f}]",
            transform=ax.transAxes, fontsize=8)
    ax.set(xlabel="per-layer median $\\sigma(J_\\varphi)$", xscale="log",
           ylabel="per-layer $\\hat m_A$", yscale="log")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_tracking.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_tracking.png"), dpi=110)
    plt.close(fig)
    return float(np.median(run_rs)), len(run_rs)




