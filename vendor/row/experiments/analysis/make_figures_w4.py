"""Wave-4 figures: the preregistered boundary test in the declared
metric, the mechanism ablations, and the fixed-pipeline regrowth
comparison.  See wave4_prereg.md for the declared statistics.

Usage: python3 analysis/make_figures_w4.py        (all wave-4 figures)
Figures are written to docs/figures with the w4 tag; no invocation of
this script overwrites a wave-1/2 artifact.
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


def load_run(name):
    recs = []
    with open(os.path.join(RUNS, name, "log.jsonl")) as f:
        for line in f:
            recs.append(json.loads(line))
    config = next(r for r in recs if r.get("event") == "config")
    steps = [r for r in recs if "step" in r]
    return config, steps


def series(steps, key):
    rs = [r for r in steps if key in r and r[key] is not None]
    return (np.array([r["step"] for r in rs]), [r[key] for r in rs])


def min_layer(steps, key):
    xs, ys = series(steps, key)
    return xs, np.array([min(v) for v in ys])


def s_pre(steps, group="nonemb"):
    """Preregistered stability statistic: eta_t * median over replicate
    batches of lam_pre_max_<group>, with min/max batch spread."""
    xs, med, lo, hi, resid = [], [], [], [], []
    for r in steps:
        k = f"lam_pre_max_{group}"
        if k in r and "lr" in r:
            v = np.array(r[k]) * r["lr"]
            xs.append(r["step"])
            med.append(np.median(v))
            lo.append(v.min())
            hi.append(v.max())
            rr = np.array(r.get(f"lam_pre_resid_{group}", [np.nan]))
            resid.append(np.max(rr) * r["lr"])
    return (np.array(xs), np.array(med), np.array(lo), np.array(hi),
            np.array(resid))


def s_raw(steps):
    xs, ys = [], []
    for r in steps:
        if "lam_full" in r and "lr" in r:
            xs.append(r["step"])
            ys.append(r["lam_full"] * r["lr"])
    return np.array(xs), np.array(ys)


def fig_precond(adamw_runs, sgd_runs, tag="w4"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = {"w4-base-lr1e-3": "#d95f02", "w4-base-lr1e-3-s2": "#fdae61",
              "w4-base-lr3e-4": "#1f78b4", "w4-base-lr3e-4-s2": "#a6cee3"}
    for n in adamw_runs:
        c, s = load_run(n)
        xs, med, lo, hi, _ = s_pre(s, "nonemb")
        col = colors.get(n, "#666666")
        ax[0].plot(xs, med, "-o", ms=2.5, lw=1.0, color=col,
                   label=n.replace("w4-base-", ""))
        ax[0].fill_between(xs, lo, hi, alpha=0.2, color=col, lw=0)
        xsf, medf, lof, hif, _ = s_pre(s, "full")
        ax[0].plot(xsf, medf, ":", lw=0.7, color=col)
    ax[0].axhline(38, color="k", lw=0.8, ls="--")
    ax[0].text(0.02, 38 * 1.15, "$2(1{+}\\beta_1)/(1{-}\\beta_1) = 38$",
               fontsize=7, transform=ax[0].get_yaxis_transform())
    ax[0].set(xlabel="step", yscale="log",
              ylabel="$\\eta_t\\,\\lambda^{\\rm pre}_{\\max}$"
                     " (nonemb; dotted: full)")
    ax[0].legend(fontsize=7, frameon=False)

    for n in sgd_runs:
        c, s = load_run(n)
        xs, ys = s_raw(s)
        style = "-" if "sgdm" in n else "--"
        ax[1].plot(xs, ys, style, lw=1.0,
                   label=n.replace("w4-", ""))
    ax[1].axhline(2, color="k", lw=0.8, ls=":")
    ax[1].axhline(38, color="k", lw=0.8, ls="--")
    ax[1].text(0.02, 2 * 1.2, "$2$ (SGD)", fontsize=7,
               transform=ax[1].get_yaxis_transform())
    ax[1].text(0.02, 38 * 1.2, "$38$ (SGD+momentum)", fontsize=7,
               transform=ax[1].get_yaxis_transform())
    ax[1].set(xlabel="step", yscale="log",
              ylabel="$\\eta_t\\,\\lambda_{\\rm full}$ (raw)")
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_precond.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_precond.png"), dpi=110)
    plt.close(fig)


def fig_ablate(tag="w4"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    left = [("w4-base-lr1e-3", "#d95f02", "baseline"),
            ("w4-noclip-lr1e-3", "#7570b3", "clip off"),
            ("w4-nowd-lr1e-3", "#1b9e77", "wd 0"),
            ("w4-frzv-lr1e-3", "#e7298a", "frozen $\\hat v$@2k"),
            ("w4-frzg500-lr1e-3", "#66a61e", "frozen $G$@500"),
            ("w4-sdpa-lr1e-3", "#666666", "SDPA ($G{=}I$)"),
            ("w4-dag005-lr1e-3", "#a6761d", "DAG 0.005"),
            ("w4-dag015-lr1e-3", "#e6ab02", "DAG 0.015")]
    for n, col, lab in left:
        try:
            c, s = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        ax[0].plot(xs, np.maximum(ys, 1e-8), lw=1.0, color=col, label=lab)
    ax[0].axhline(0.1, color="k", lw=0.6, ls=":")
    ax[0].set(xlabel="step", yscale="log",
              ylabel="min-layer median $\\sigma(J_\\varphi)$")
    ax[0].legend(fontsize=6.5, frameon=False, ncol=2)

    right = [("w4-base-lr1e-3", "#d95f02", "block loss, $10^{-3}$"),
             ("w4-last-lr1e-3", "#fdae61", "last-only loss, $10^{-3}$"),
             ("w4-base-lr3e-4", "#1f78b4", "block loss, $3{\\times}10^{-4}$"),
             ("w4-last-lr3e-4", "#a6cee3",
              "last-only loss, $3{\\times}10^{-4}$")]
    for n, col, lab in right:
        try:
            c, s = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        ax[1].plot(xs, np.maximum(ys, 1e-8), lw=1.0, color=col, label=lab)
    ax[1].axhline(0.1, color="k", lw=0.6, ls=":")
    ax[1].set(xlabel="step", yscale="log",
              ylabel="min-layer median $\\sigma(J_\\varphi)$")
    ax[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_ablate.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_ablate.png"), dpi=110)
    plt.close(fig)


def fig_regrow(tag="w4"):
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.2))
    runs = [("w4-base-lr1e-3", "#d95f02", "source run ($10^{-3}$)", 0),
            ("w4-regrow2-lr1e-4", "#1f78b4",
             "continuation, const $10^{-4}$", 12000),
            ("w4-regrow2-reset-lr1e-4", "#a6cee3",
             "continuation, optimizer reset", 12000)]
    for n, col, lab, off in runs:
        try:
            c, s = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        ax[0].plot(xs + off, np.maximum(ys, 1e-8), lw=1.0, color=col,
                   label=lab)
        xs, ys = min_layer(s, "m_A_layers")
        ax[1].plot(xs + off, np.maximum(ys, 1e-8), lw=1.0, color=col)
    ax[0].axhline(0.1, color="k", lw=0.6, ls=":")
    ax[0].set(xlabel="step (continuation offset by source length)",
              yscale="log",
              ylabel="min-layer median $\\sigma(J_\\varphi)$")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].set(xlabel="step (continuation offset by source length)",
              yscale="log", ylabel="min-layer $\\hat m_A$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_regrow.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_regrow.png"), dpi=110)
    plt.close(fig)


def onset_and_boundary_report(adamw_runs, sgd_runs):
    """Preregistered test: statistic value at collapse onset (first probe
    with min-layer median sigma_max < 0.1) and max/trajectory summary."""
    out = {}
    for n in adamw_runs + sgd_runs:
        try:
            c, s = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        below = xs[ys < 0.1]
        onset = int(below[0]) if len(below) else None
        entry = {"onset": onset}
        if n in adamw_runs:
            px, pm, plo, phi, presid = s_pre(s, "nonemb")
            if len(px):
                entry["S_pre_med_post2k"] = float(
                    np.median(pm[px >= 2000])) if (px >= 2000).any() else None
                entry["S_pre_max"] = float(pm.max())
                entry["S_pre_resid_max"] = float(np.nanmax(presid))
                if onset is not None:
                    i = int(np.argmin(np.abs(px - onset)))
                    entry["S_pre_at_onset"] = float(pm[i])
                    entry["S_pre_at_onset_step"] = int(px[i])
        rx, ry = s_raw(s)
        if len(rx):
            entry["S_raw_med_post2k"] = float(
                np.median(ry[rx >= 2000])) if (rx >= 2000).any() else None
            if onset is not None:
                i = int(np.argmin(np.abs(rx - onset)))
                entry["S_raw_at_onset"] = float(ry[i])
        out[n] = entry
    with open(os.path.join(OUT, "w4_boundary_report.json"), "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(k, json.dumps(v))
    return out


ADAMW = ["w4-base-lr1e-3", "w4-base-lr1e-3-s2",
         "w4-base-lr3e-4", "w4-base-lr3e-4-s2"]
SGD = ["w4-sgd-lr1e-1", "w4-sgd-lr3e-1",
       "w4-sgdm-lr3e-2", "w4-sgdm-lr1e-1", "w4-sgdm-lr3e-1"]

if __name__ == "__main__":
    have = set(os.listdir(RUNS))
    adamw = [n for n in ADAMW if n in have]
    sgd = [n for n in SGD if n in have]
    fig_precond(adamw, sgd)
    fig_ablate()
    fig_regrow()
    onset_and_boundary_report(adamw, sgd)
    print("wave-4 figures written to", OUT)
