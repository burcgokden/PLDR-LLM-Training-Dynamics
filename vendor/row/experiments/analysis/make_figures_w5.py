"""Wave-5 figures and the preregistered gate report: the coupling-block
boundary and self-quench test, the windowed force law (decision gate),
pulse triggers, noise scaling, route selection (decision gate), the
endpoint reworking (all-layer + operator primary), the comparative
entropy/rank diagnostics, and the minimal-instance panels.
See wave5_prereg.md for the declared statistics and falsifiers.

Usage: python3 analysis/make_figures_w5.py     (all wave-5 artifacts)
Figures are written to docs/figures with the w5 tag; no invocation of
this script overwrites another wave's artifact.
"""

import json
import os

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

BOUNDARY = {"adamw": 38.0, "sgdm": 38.0, "sgd": 2.0}

BASE_SUB = ["w5-base-lr3e-4", "w5-base-lr3e-4-s2", "w5-base-lr3e-4-s3"]
BASE_COL = ["w5-base-lr1e-3", "w5-base-lr1e-3-s2", "w5-base-lr1e-3-s3"]
SGDM = ["w5-sgdm-lr3e-2", "w5-sgdm-lr1e-1"]
UNCLIP = ["w5-sgd-noclip-lr3e-5", "w5-sgdm-noclip-lr3e-4"]
PULSE = ["w5-pulse2k-lr3e-4", "w5-pulse6k-lr3e-4", "w5-pulse10k-lr3e-4",
         "w5-pulse2k-hi-lr3e-4"]
NOISE = ["w5-b8-lr1e-3", "w5-base-lr1e-3", "w5-b64-lr1e-3"]
ROUTES = ["w5-frzplga-lr1e-3", "w5-frzplga-lr2e-3", "w5-rand1-lr1e-3",
          "w5-last-nowd-lr1e-3", "w5-frzvfix-lr1e-3", "w5-last-b8-lr1e-3"]


def load_run(name):
    recs = []
    with open(os.path.join(RUNS, name, "log.jsonl")) as f:
        for line in f:
            recs.append(json.loads(line))
    config = next(r for r in recs if r.get("event") == "config")
    steps = [r for r in recs if "step" in r]
    final = next((r for r in recs if r.get("event") == "final"), None)
    return config, steps, final


def series(steps, key):
    rs = [r for r in steps if key in r and r[key] is not None]
    return (np.array([r["step"] for r in rs]),
            [r[key] for r in rs])


def min_layer(steps, key):
    xs, ys = series(steps, key)
    return xs, np.array([min(v) for v in ys])


def max_layer(steps, key):
    xs, ys = series(steps, key)
    return xs, np.array([max(v) for v in ys])


def s_blk(steps):
    """Declared block statistic S_blk(t) = eta_t * lam_plga."""
    xs, ys = [], []
    for r in steps:
        if r.get("lam_plga") is not None and "lr" in r:
            xs.append(r["step"])
            ys.append(r["lam_plga"] * r["lr"])
    return np.array(xs), np.array(ys)


def last_quarter(xs, ys):
    if len(xs) == 0:
        return np.array([])
    t = xs.max()
    return np.asarray(ys)[xs > 0.75 * t]


def block_bootstrap_ci(vals, blocklen=10, n=1000, alpha=0.05, seed=0):
    """Declared uncertainty: time-block bootstrap of the median."""
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    blocks = [vals[i:i + blocklen] for i in range(0, len(vals), blocklen)]
    meds = []
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        meds.append(np.median(np.concatenate([blocks[j] for j in pick])))
    return [float(np.quantile(meds, alpha / 2)),
            float(np.quantile(meds, 1 - alpha / 2))]


def spearman_perm(x, y, n_perm=2000, seed=0):
    """Spearman rho with a permutation p-value (no scipy)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)

    def pear(a, b):
        a = a - a.mean()
        b = b - b.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    rho = pear(rx, ry)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(pear(rx, rng.permutation(ry))) >= abs(rho):
            cnt += 1
    return rho, (cnt + 1) / (n_perm + 1)


def onset_step(steps, thresh=0.1):
    xs, ys = min_layer(steps, "rowmap_sigma_med")
    below = xs[ys < thresh]
    return int(below[0]) if len(below) else None


def force_windows(steps, wlen=250):
    """Per-window force-law records: u at window start, du across the
    window, oscillation power V_w (mean dtheta_plga^2), drive power
    P_w (mean gnorm_plga^2), and S_blk at the window start."""
    ux, uy = min_layer(steps, "rowmap_sigma_med")
    bx, by = s_blk(steps)
    dt = {r["step"]: r.get("dtheta_plga") for r in steps
          if r.get("dtheta_plga") is not None}
    gp = {r["step"]: r.get("gnorm_plga") for r in steps
          if r.get("gnorm_plga") is not None}
    if len(ux) == 0:
        return []
    out = []
    t0, tmax = int(ux.min()), int(ux.max())
    for w0 in range(t0, tmax - wlen + 1, wlen):
        w1 = w0 + wlen
        iu0 = np.searchsorted(ux, w0, side="right") - 1
        iu1 = np.searchsorted(ux, w1, side="right") - 1
        if iu0 < 0 or iu1 <= iu0:
            continue
        u0, u1 = float(uy[iu0]), float(uy[iu1])
        vs = [dt[s] ** 2 for s in range(w0, w1) if s in dt]
        ps = [gp[s] ** 2 for s in range(w0, w1) if s in gp]
        if not vs:
            continue
        ib = np.argmin(np.abs(bx - w0)) if len(bx) else None
        sb = (float(by[ib]) if ib is not None
              and abs(bx[ib] - w0) <= wlen / 2 else None)
        out.append({"w0": w0, "u": u0, "du": u1 - u0,
                    "V": float(np.mean(vs)), "P": float(np.mean(ps)),
                    "S_blk": sb})
    return out


# ---------------------------------------------------------------------------
# figures


def fig_boundary(tag="w5"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    fam = {"w5-base-lr3e-4": "#1f78b4", "w5-base-lr3e-4-s2": "#7bb3d9",
           "w5-base-lr3e-4-s3": "#a6cee3",
           "w5-base-lr1e-3": "#d95f02", "w5-base-lr1e-3-s2": "#f0904a",
           "w5-base-lr1e-3-s3": "#fdae61"}
    for n, col in fam.items():
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = s_blk(s)
        ax[0].plot(xs, np.maximum(ys, 1e-6), lw=1.0, color=col,
                   label=n.replace("w5-base-", ""))
    ax[0].axhline(38, color="k", lw=0.8, ls="--")
    ax[0].text(0.02, 38 * 1.2, "$38$", fontsize=7,
               transform=ax[0].get_yaxis_transform())
    ax[0].set(xlabel="step", yscale="log",
              ylabel="$S_{\\rm blk} = \\eta_t\\,\\lambda_{\\rm plga}$"
                     " (CE objective)")
    ax[0].legend(fontsize=6.5, frameon=False, ncol=2)

    for n, col, lab in [("w5-sgdm-lr3e-2", "#1b9e77", "sgdm 3e-2 (clip)"),
                        ("w5-sgdm-lr1e-1", "#d95f02", "sgdm 1e-1 (clip)"),
                        ("w5-sgd-noclip-lr3e-5", "#7570b3",
                         "sgd 3e-5 unclipped"),
                        ("w5-sgdm-noclip-lr3e-4", "#e7298a",
                         "sgdm 3e-4 unclipped")]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = s_blk(s)
        ax[1].plot(xs, np.maximum(ys, 1e-6), lw=1.0, color=col, label=lab)
        rx = np.array([r["step"] for r in s
                       if r.get("lam_full") is not None])
        ry = np.array([r["lam_full"] * r["lr"] for r in s
                       if r.get("lam_full") is not None])
        ax[1].plot(rx, np.maximum(ry, 1e-6), ":", lw=0.7, color=col)
    ax[1].axhline(38, color="k", lw=0.8, ls="--")
    ax[1].axhline(2, color="k", lw=0.8, ls=":")
    ax[1].set(xlabel="step", yscale="log",
              ylabel="$S_{\\rm blk}$ (solid); "
                     "$\\eta_t\\lambda_{\\rm full}$ (dotted)")
    ax[1].legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_boundary.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_boundary.png"), dpi=110)
    plt.close(fig)


def fig_force(pooled, tag="w5"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    if pooled:
        u = np.array([w["u"] for w in pooled])
        du = np.array([w["du"] for w in pooled])
        uv = np.array([w["u"] * w["V"] for w in pooled])
        ax[0].scatter(np.maximum(uv, 1e-16), du, s=8, alpha=0.6,
                      color="#d95f02", edgecolors="none")
        ax[0].set(xscale="log", xlabel="$u \\cdot V_w$ "
                  "(jet $\\times$ window oscillation power)",
                  ylabel="window jet change $\\Delta u$")
        ax[0].axhline(0, color="k", lw=0.6)
        rho, p = spearman_perm(du, uv)
        ax[0].set_title(f"Spearman $\\rho = {rho:.2f}$ "
                        f"(perm.\\ $p = {p:.3f}$)", fontsize=8)
        # tertile panel: mean decrement across V tertiles within u bins
        uq = np.quantile(u, [1 / 3, 2 / 3])
        vq = np.quantile(uv / np.maximum(u, 1e-16), [1 / 3, 2 / 3])
        V = uv / np.maximum(u, 1e-16)
        width = 0.25
        for bi, (lo, hi, lab) in enumerate(
                [(-np.inf, uq[0], "low $u$"), (uq[0], uq[1], "mid $u$"),
                 (uq[1], np.inf, "high $u$")]):
            sel = (u > lo) & (u <= hi) if np.isfinite(hi) else (u > lo)
            means = []
            for vlo, vhi in [(-np.inf, vq[0]), (vq[0], vq[1]),
                             (vq[1], np.inf)]:
                s2 = sel & (V > vlo) & (V <= vhi if np.isfinite(vhi)
                                        else V > vlo)
                means.append(float(np.mean(-du[s2])) if s2.any()
                             else np.nan)
            ax[1].bar(np.arange(3) + (bi - 1) * width, means, width,
                      label=lab)
        ax[1].set(xticks=range(3),
                  xticklabels=["low $V_w$", "mid $V_w$", "high $V_w$"],
                  ylabel="mean decrement $-\\Delta u$")
        ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_force.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_force.png"), dpi=110)
    plt.close(fig)


def fig_pulse(tag="w5"):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.2))
    cells = [("w5-base-lr3e-4", "#1f78b4", "base $3{\\times}10^{-4}$",
              None, 0),
             ("w5-pulse2k-lr3e-4", "#d95f02", "pulse@2k", 2000, 8),
             ("w5-pulse6k-lr3e-4", "#1b9e77", "pulse@6k", 6000, 8),
             ("w5-pulse10k-lr3e-4", "#7570b3", "pulse@10k", 10000, 8),
             ("w5-pulse2k-hi-lr3e-4", "#e7298a",
              "pulse@2k, $2{\\times}10^{-3}\\times16$", 2000, 16)]
    for n, col, lab, at, ln in cells:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        ax.plot(xs, np.maximum(ys, 1e-8), lw=1.0, color=col, label=lab)
        if at:
            ax.axvspan(at, at + ln, color=col, alpha=0.25, lw=0)
    ax.axhline(0.1, color="k", lw=0.6, ls=":")
    ax.set(xlabel="step", yscale="log",
           ylabel="min-layer median $\\sigma(J_\\varphi)$")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_pulse.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_pulse.png"), dpi=110)
    plt.close(fig)


def fig_noise(tag="w5"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for n, col, lab in [("w5-b8-lr1e-3", "#a6cee3", "batch 8"),
                        ("w5-base-lr1e-3", "#1f78b4", "batch 32"),
                        ("w5-b64-lr1e-3", "#08306b", "batch 64"),
                        ("w5-last-b8-lr1e-3", "#e7298a",
                         "last loss, batch 8")]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        ax[0].plot(xs, np.maximum(ys, 1e-8), lw=1.0, color=col, label=lab)
    ax[0].axhline(0.1, color="k", lw=0.6, ls=":")
    ax[0].set(xlabel="step", yscale="log",
              ylabel="min-layer median $\\sigma(J_\\varphi)$")
    ax[0].legend(fontsize=7, frameon=False)
    bs, depth = [], []
    for n, b in [("w5-b8-lr1e-3", 8), ("w5-base-lr1e-3", 32),
                 ("w5-b64-lr1e-3", 64)]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        lq = last_quarter(xs, ys)
        if len(lq):
            bs.append(b)
            depth.append(float(np.median(lq)))
    ax[1].plot(bs, depth, "o-", color="#1f78b4")
    ax[1].set(xlabel="batch size", xscale="log", yscale="log",
              ylabel="last-quarter median (min-layer)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_noise.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_noise.png"), dpi=110)
    plt.close(fig)


def fig_routes(tag="w5"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for n, col, lab in [("w5-base-lr1e-3", "#d95f02", "base $10^{-3}$"),
                        ("w5-frzplga-lr1e-3", "#1b9e77",
                         "frozen coupling, $10^{-3}$"),
                        ("w5-frzplga-lr2e-3", "#66a61e",
                         "frozen coupling, $2{\\times}10^{-3}$"),
                        ("w5-rand1-lr1e-3", "#7570b3",
                         "rand1 loss, $10^{-3}$"),
                        ("w5-frzvfix-lr1e-3", "#e7298a",
                         "fixed $\\sqrt{\\hat v}$@2k")]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        ax[0].plot(xs, np.maximum(ys, 1e-8), lw=1.0, color=col, label=lab)
    ax[0].axhline(0.1, color="k", lw=0.6, ls=":")
    ax[0].set(xlabel="step", yscale="log",
              ylabel="min-layer median $\\sigma(J_\\varphi)$")
    ax[0].legend(fontsize=6.5, frameon=False)

    for n, col, lab in [("w5-base-lr1e-3", "#d95f02", "base"),
                        ("w5-last-nowd-lr1e-3", "#1f78b4",
                         "last loss, wd 0"),
                        ("w5-last-b8-lr1e-3", "#a6cee3",
                         "last loss, b8 (wd 0.1)")]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = series(s, "a_norm_layers")
        if len(xs):
            ax[1].plot(xs, [np.mean(v) for v in ys], lw=1.0, color=col,
                       label=lab)
    ax[1].set(xlabel="step", yscale="log",
              ylabel="mean coupling amplitude "
                     "$\\overline{\\|a\\|}_F$")
    ax[1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_routes.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_routes.png"), dpi=110)
    plt.close(fig)


def fig_endpoints(cells, tag="w5"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    names, l1, l2, l3, mg = [], [], [], [], []
    for n in cells:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = series(s, "rowmap_sigma_med")
        if not len(xs):
            continue
        arr = np.array(ys)
        lq = arr[xs > 0.75 * xs.max()]
        med = np.median(lq, axis=0)
        names.append(n.replace("w5-", ""))
        l1.append(med[0]); l2.append(med[1]); l3.append(med[2])
        mg.append(f["m_gen"]["GLM"] if f else np.nan)
    x = np.arange(len(names))
    for i, (vals, lab) in enumerate([(l1, "layer 1"), (l2, "layer 2"),
                                     (l3, "layer 3")]):
        ax[0].bar(x + (i - 1) * 0.27, np.maximum(vals, 1e-8), 0.27,
                  label=lab)
    ax[0].axhline(0.1, color="k", lw=0.6, ls=":")
    ax[0].set(xticks=x, yscale="log",
              ylabel="last-quarter median $\\sigma(J_\\varphi)$")
    ax[0].set_xticklabels(names, rotation=40, ha="right", fontsize=6.5)
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].bar(x, np.maximum(mg, 1e-8), 0.5, color="#7570b3")
    ax[1].axhline(0.01, color="k", lw=0.6, ls=":")
    ax[1].set(xticks=x, yscale="log",
              ylabel="operator endpoint $m^{\\rm gen}(G_{\\rm LM})$")
    ax[1].set_xticklabels(names, rotation=40, ha="right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_endpoints.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_endpoints.png"), dpi=110)
    plt.close(fig)


def fig_comparative(tag="w5"):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for n, col, lab in [("w5-base-lr3e-4", "#1f78b4",
                         "$3{\\times}10^{-4}$ (sub-critical)"),
                        ("w5-base-lr1e-3", "#d95f02",
                         "$10^{-3}$ (flattening)")]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = series(s, "att_entropy_layers")
        if len(xs):
            ax[0].plot(xs, [np.mean(v) for v in ys], lw=1.0, color=col,
                       label=lab)
        xs, ys = series(s, "repr_effrank")
        if len(xs):
            ax[1].plot(xs, ys, lw=1.0, color=col, label=lab)
    ax[0].set(xlabel="step", ylabel="mean attention entropy (nats)")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].set(xlabel="step", ylabel="representation effective rank")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_comparative.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_comparative.png"), dpi=110)
    plt.close(fig)


def fig_toy(tag="w5"):
    phase_path = os.path.join(RUNS, "toy", "phase.json")
    if not os.path.exists(phase_path):
        return
    cells = json.load(open(phase_path))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    colors = {0.1: "#a6cee3", 0.25: "#1f78b4", 0.5: "#08306b"}
    for rho, col in colors.items():
        pts = [(2 / c["eta"], c["s_end"]) for c in cells
               if c["protocol"] == "hot" and c["mode"] == "block"
               and c["rho"] == rho and not c["diverged"]
               and c.get("s_end")]
        if pts:
            x, y = zip(*pts)
            ax[0].plot(x, y, "o", ms=4, color=col,
                       label=f"$\\rho = {rho}$")
    lim = [3e-1, 3e3]
    ax[0].plot(lim, lim, "k--", lw=0.7, label="$s = 2/\\eta$")
    ax[0].set(xscale="log", yscale="log", xlim=lim,
              xlabel="stability boundary $2/\\eta$",
              ylabel="exact carrier sharpness (last-quarter med.)")
    ax[0].legend(fontsize=7, frameon=False)
    for rho, col in colors.items():
        pts = sorted((c["eta"], c.get("J_ratio"), c.get("a_ratio"))
                     for c in cells
                     if c["protocol"] == "hot" and c["mode"] == "block"
                     and c["rho"] == rho and c.get("J_ratio"))
        if pts:
            e, jr, ar = zip(*pts)
            ax[1].plot(e, jr, "o-", ms=3, lw=0.9, color=col)
            ax[1].plot(e, ar, "s--", ms=3, lw=0.9, color=col, alpha=0.6)
    ax[1].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[1].set(xscale="log", yscale="log", xlabel="$\\eta$ (hot cells)",
              ylabel="end/reference ratio: $J$ (circles), $a$ (squares)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_{tag}_toy.pdf"))
    fig.savefig(os.path.join(OUT, f"fig_{tag}_toy.png"), dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# preregistered gate report


def gate_report():
    rep = {}

    def lq_med_sblk(name):
        c, s, f = load_run(name)
        xs, ys = s_blk(s)
        lq = last_quarter(xs, ys)
        return (float(np.median(lq)) if len(lq) else None,
                block_bootstrap_ci(lq) if len(lq) else None)

    # P1': sustain vs self-quench
    p1 = {}
    for n in BASE_SUB:
        try:
            m, ci = lq_med_sblk(n)
        except FileNotFoundError:
            continue
        p1[n] = {"lq_med": m, "ci": ci, "pass": m is not None
                 and 2 <= m <= 40}
    for n in BASE_COL:
        try:
            m, ci = lq_med_sblk(n)
        except FileNotFoundError:
            continue
        p1[n] = {"lq_med": m, "ci": ci, "pass": m is not None and m < 1}
    try:
        msub, _ = lq_med_sblk("w5-sgdm-lr3e-2")
        mcol, _ = lq_med_sblk("w5-sgdm-lr1e-1")
        p1["sgdm"] = {"sub": msub, "col": mcol,
                      "pass": msub is not None and mcol is not None
                      and 9.5 <= msub <= 152 and mcol < msub / 2}
    except FileNotFoundError:
        pass
    cells1 = [v for v in p1.values() if isinstance(v, dict)]
    p1["verdict"] = (all(v.get("pass") for v in cells1)
                     if cells1 else None)
    rep["P1_sustain_quench"] = p1

    # P2': onset excursion
    p2 = {}
    for n, bnd in ([(x, 38.0) for x in BASE_COL]
                   + [("w5-sgdm-lr1e-1", 38.0)]):
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        on = onset_step(s)
        entry = {"onset": on}
        if on is not None:
            bx, by = s_blk(s)
            near = np.abs(bx - on) <= 250
            entry["S_blk_near_onset"] = ([float(v) for v in by[near]]
                                         if near.any() else [])
            around = (bx >= on - 500) & (bx <= on + 500)
            entry["excursion"] = bool((by[near] > bnd).any()) if \
                near.any() else False
            entry["quench_transit"] = bool(
                around.any() and (by[around] > bnd).any()
                and (by[(bx > on)][:4] < 1).any() if (bx > on).any()
                else False)
            entry["pass"] = entry["excursion"] or entry["quench_transit"]
        p2[n] = entry
    cells2 = [v for v in p2.values() if isinstance(v, dict)]
    p2["verdict"] = (all(v.get("pass") for v in cells2)
                     if cells2 else None)
    rep["P2_onset_excursion"] = p2

    # P3' (GATE): force law over pre-onset supercritical windows
    pooled, per_cell = [], {}
    for n in BASE_COL:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        on = onset_step(s) or 10 ** 9
        ws = [w for w in force_windows(s)
              if w["w0"] < on and w["S_blk"] is not None
              and w["S_blk"] > 38.0]
        per_cell[n] = len(ws)
        pooled.extend(ws)
    p3 = {"n_windows": len(pooled), "per_cell": per_cell}
    if pooled:
        du = np.array([w["du"] for w in pooled])
        uv = np.array([w["u"] * w["V"] for w in pooled])
        rho, p = spearman_perm(du, uv)
        rho_p, p_p = spearman_perm(
            du, np.array([w["u"] * w["P"] for w in pooled]))
        p3.update({"spearman_uV": rho, "p_uV": p,
                   "spearman_uP": rho_p, "p_uP": p_p,
                   "pass": rho < -0.3 and p < 0.01,
                   "kill": abs(rho) < 0.15})
    rep["P3_force_law_GATE"] = p3

    # P4': pulses
    p4 = {}
    for n, at, ln in [("w5-pulse2k-lr3e-4", 2000, 8),
                      ("w5-pulse6k-lr3e-4", 6000, 8),
                      ("w5-pulse10k-lr3e-4", 10000, 8),
                      ("w5-pulse2k-hi-lr3e-4", 2000, 16)]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        pre = ys[(xs >= at - 500) & (xs < at)]
        post = ys[(xs >= at) & (xs <= at + 500)]
        pre_v = float(np.median(pre)) if len(pre) else None
        drop = (float(np.min(post)) if len(post) else None)
        transient = (pre_v is not None and drop is not None
                     and drop <= pre_v / 5)
        rec_win = ys[(xs > at + ln) & (xs <= at + ln + 2000)]
        recovered = bool((rec_win > 0.1).any()) if len(rec_win) else None
        p4[n] = {"pre": pre_v, "min_post": drop, "transient": transient,
                 "recovered": recovered,
                 "pass": bool(transient) and bool(recovered)}
    cells4 = [v for v in p4.values() if isinstance(v, dict)]
    p4["verdict"] = (all(v.get("pass") for v in cells4)
                     if cells4 else None)
    rep["P4_pulse"] = p4

    # P5': noise scaling
    p5 = {}
    depths = {}
    for n, b in [("w5-b8-lr1e-3", 8), ("w5-base-lr1e-3", 32),
                 ("w5-b64-lr1e-3", 64)]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = min_layer(s, "rowmap_sigma_med")
        lq = last_quarter(xs, ys)
        depths[b] = float(np.median(lq)) if len(lq) else None
    p5["depths"] = depths
    ds = [depths.get(b) for b in (8, 32, 64)]
    p5["monotone"] = (None not in ds and ds[0] > ds[1] > ds[2])
    try:
        c, s, f = load_run("w5-last-b8-lr1e-3")
        on = onset_step(s)
        xs, ys = min_layer(s, "rowmap_sigma_med")
        esc = None
        if on is not None:
            after = (xs > on) & (xs <= on + 2400)
            esc = bool((ys[after] > 0.1).any())
        p5["last_b8"] = {"onset": on, "escaped_within_2400": esc}
    except FileNotFoundError:
        pass
    p5["verdict"] = bool(p5.get("monotone"))
    rep["P5_noise"] = p5

    # P6' (GATE): frozen coupling block
    p6 = {}
    for n in ["w5-frzplga-lr1e-3", "w5-frzplga-lr2e-3"]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = series(s, "rowmap_sigma_med")
        arr = np.array(ys)
        lq = arr[xs > 0.75 * xs.max()]
        per_layer = np.median(lq, axis=0).tolist()
        flat_any = bool(min(per_layer) < 0.1)
        mg = f["m_gen"]["GLM"] if f else None
        p6[n] = {"per_layer_lq_med": per_layer,
                 "any_layer_flat": flat_any,
                 "m_gen_GLM": mg,
                 "pass": (not flat_any)
                 and (mg is None or mg >= 0.01)}
    cells6 = [v for v in p6.values() if isinstance(v, dict)]
    p6["verdict"] = (all(v.get("pass") for v in cells6)
                     if cells6 else None)
    rep["P6_frozen_coupling_GATE"] = p6

    # P7': last + no weight decay
    try:
        c, s, f = load_run("w5-last-nowd-lr1e-3")
        xs, ys = series(s, "a_norm_layers")
        a0 = float(np.mean(ys[0])) if len(xs) else None
        a1 = float(np.mean(ys[-1])) if len(xs) else None
        mgA = f["m_gen"]["ALM"] if f else None
        mgG = f["m_gen"]["GLM"] if f else None
        rep["P7_last_nowd"] = {
            "a_norm_first": a0, "a_norm_last": a1,
            "m_gen_ALM": mgA, "m_gen_GLM": mgG,
            "pass": (mgA is not None and mgA > 0.05 and mgG > 0.05
                     and a1 is not None and a0 is not None
                     and a1 >= a0 / 2 and a1 <= 2 * a0)}
    except FileNotFoundError:
        pass

    # P8': rand1 endpoint match (tilt half is measure_tilt, post hoc)
    try:
        c, s, f = load_run("w5-rand1-lr1e-3")
        cb, sb_, fb = load_run("w5-base-lr1e-3")

        def endpoints(steps, fin):
            xs, ys = min_layer(steps, "rowmap_sigma_med")
            lq = last_quarter(xs, ys)
            xs2, ys2 = max_layer(steps, "rowmap_sigma_med")
            lq2 = last_quarter(xs2, ys2)
            return {"min_layer": float(np.mean(lq)) if len(lq) else None,
                    "max_layer": float(np.mean(lq2)) if len(lq2)
                    else None,
                    "m_gen_GLM": fin["m_gen"]["GLM"] if fin else None}

        er, eb = endpoints(s, f), endpoints(sb_, fb)
        match = (er["min_layer"] is not None and eb["min_layer"] is not None
                 and (er["min_layer"] < 0.1) == (eb["min_layer"] < 0.1))
        rep["P8_rand1"] = {"rand1": er, "base": eb,
                           "endpoint_match": match,
                           "tilt": "run measure_tilt post hoc"}
    except FileNotFoundError:
        pass

    # P9': genuinely fixed preconditioner
    try:
        c, s, f = load_run("w5-frzvfix-lr1e-3")
        loss = np.array([r["loss"] for r in s])
        xs = np.array([r["step"] for r in s])
        pre = loss[(xs > 1500) & (xs <= 2000)]
        post = loss[(xs > 2000) & (xs <= 6000)]
        nonfinite = bool(~np.isfinite(loss).all()) or xs.max() < 12000
        blowup = bool(len(post) and len(pre)
                      and np.max(post) > 2 * np.median(pre))
        rep["P9_fixed_preconditioner"] = {
            "last_step": int(xs.max()), "nonfinite_or_early_stop":
            nonfinite, "loss_blowup": blowup,
            "pass": nonfinite or blowup}
    except FileNotFoundError:
        pass

    # P10': unclipped classical cells
    p10 = {}
    for n, bnd in [("w5-sgd-noclip-lr3e-5", 2.0),
                   ("w5-sgdm-noclip-lr3e-4", 38.0)]:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        xs = np.array([r["step"] for r in s])
        loss = np.array([r["loss"] for r in s])
        rx = np.array([r["step"] for r in s
                       if r.get("lam_full") is not None])
        ry = np.array([r["lam_full"] * r["lr"] for r in s
                       if r.get("lam_full") is not None])
        lq = last_quarter(rx, ry)
        med = float(np.median(lq)) if len(lq) else None
        mxs, mys = min_layer(s, "rowmap_sigma_med")
        mlq = last_quarter(mxs, mys)
        flat = bool(len(mlq) and np.mean(mlq) < 0.1)
        p10[n] = {"finite": bool(np.isfinite(loss).all()),
                  "lq_med_eta_lam_full": med,
                  "in_band": med is not None
                  and bnd / 4 <= med <= 1.5 * bnd,
                  "any_flattening": flat,
                  "pass": bool(np.isfinite(loss).all()) and med is not None
                  and bnd / 4 <= med <= 1.5 * bnd and not flat}
    cells10 = [v for v in p10.values() if isinstance(v, dict)]
    p10["verdict"] = (all(v.get("pass") for v in cells10)
                      if cells10 else None)
    rep["P10_unclipped"] = p10

    # decision gates
    rep["GATES"] = {
        "force_law_pass": rep.get("P3_force_law_GATE", {}).get("pass"),
        "force_law_kill": rep.get("P3_force_law_GATE", {}).get("kill"),
        "frozen_coupling_pass":
            rep.get("P6_frozen_coupling_GATE", {}).get("verdict"),
    }
    with open(os.path.join(OUT, "w5_gate_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep["GATES"], indent=1))
    return rep


if __name__ == "__main__":
    fig_boundary()
    pooled = []
    for n in BASE_COL:
        try:
            c, s, f = load_run(n)
        except FileNotFoundError:
            continue
        on = onset_step(s) or 10 ** 9
        pooled += [w for w in force_windows(s)
                   if w["w0"] < on and w["S_blk"] is not None
                   and w["S_blk"] > 38.0]
    fig_force(pooled)
    fig_pulse()
    fig_noise()
    fig_routes()
    fig_endpoints(BASE_SUB[:1] + BASE_COL[:1] + SGDM + UNCLIP + ROUTES)
    fig_comparative()
    fig_toy()
    gate_report()
    print("wave-5 figures written to", OUT)
