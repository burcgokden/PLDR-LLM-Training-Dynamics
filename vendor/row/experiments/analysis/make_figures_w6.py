"""Wave-6 figures and reports: the metric-correct per-block
statistics, the tracked-mode coherence panels, the redesigned force
law, the causal damp comparison, the carrier-search freeze panel, the
matched-token noise panel, and the reference-endpoint panel; plus the
two JSON reports: w6_gate_report.json (the wave-6 verdicts of
wave6_prereg.md W1-W8, computed by analysis/gates.py) and
w5_gate_report_corrected.json (the corrected wave-5 re-audit applying
every declared clause; the archived w5_gate_report.json is never
rewritten).

Usage: python3 analysis/make_figures_w6.py
Figures are written to docs/figures with the w6 tag; no invocation of
this script overwrites another wave's artifact.
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gates  # noqa: E402
import thresholds as T  # noqa: E402

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

SUB = gates.W6_SUB
COL = gates.W6_COL
REF = gates.W6_REF


def _save(fig, tag):
    fig.savefig(os.path.join(OUT, f"{tag}.pdf"))
    fig.savefig(os.path.join(OUT, f"{tag}.png"), dpi=110)
    plt.close(fig)
    print(f"  {tag}")


def fig_metric():
    """Raw vs preconditioned vs live-restricted block statistics for
    the base cells: the metric-correct panel."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True)
    specs = [("raw $\\eta_t\\lambda_{\\mathrm{plga}}$", gates.s_blk),
             ("pre $\\eta_t\\lambda^{\\mathrm{pre}}_{\\mathrm{plga}}$",
              lambda s: _s_key(s, "lam_pre_blk_plga")),
             ("live $\\eta_t\\lambda^{\\mathrm{pre,live}}_"
              "{\\mathrm{plga}}$",
              lambda s: _s_key(s, "lam_pre_live_plga"))]
    for ax, (ttl, fn) in zip(axes, specs):
        for i, n in enumerate(SUB):
            try:
                _, s, _ = gates.load_run(n)
            except FileNotFoundError:
                continue
            xs, ys = fn(s)
            ax.plot(xs, ys, color=plt.cm.Blues(0.4 + 0.2 * i), lw=0.9)
        for i, n in enumerate(COL):
            try:
                _, s, _ = gates.load_run(n)
            except FileNotFoundError:
                continue
            xs, ys = fn(s)
            ax.plot(xs, ys, color=plt.cm.Oranges(0.4 + 0.2 * i), lw=0.9)
        ax.set_yscale("log")
        ax.set_title(ttl)
        ax.set_xlabel("step")
    axes[0].set_ylabel("$\\eta_t \\lambda$")
    fig.suptitle("Block statistics in three metrics (blues: sub-"
                 "critical seeds; oranges: flattening seeds)", y=1.04)
    _save(fig, "fig_w6_metric")


def _s_key(steps, key):
    xs, ys = [], []
    for r in steps:
        v = r.get(key)
        if v is not None and "lr" in r and np.isfinite(v):
            xs.append(r["step"])
            ys.append(v * r["lr"])
    return np.array(xs), np.array(ys)


def fig_mode():
    """Coherence panel: dyn-mode captured power and windowed flip rate
    for the flattening cells, with onsets marked."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for i, n in enumerate(COL):
        try:
            _, s, _ = gates.load_run(n)
        except FileNotFoundError:
            continue
        ws, _ = gates.dyn_windows(s)
        if not ws:
            continue
        xs = [w["w0"] for w in ws]
        axes[0].plot(xs, [w["captured"] for w in ws],
                     color=plt.cm.Oranges(0.4 + 0.2 * i), lw=1.0)
        axes[1].plot(xs, [w["flip"] for w in ws],
                     color=plt.cm.Oranges(0.4 + 0.2 * i), lw=1.0)
        on = gates.onset_step(s)
        for ax in axes:
            if on is not None:
                ax.axvline(on, color=plt.cm.Oranges(0.4 + 0.2 * i),
                           ls=":", lw=0.8)
    axes[1].axhline(T.FLIP_OSC, color="k", ls="--", lw=0.8,
                    label=f"oscillatory marker {T.FLIP_OSC}")
    axes[1].axhline(T.FLIP_DEAD, color="k", ls=":", lw=0.8,
                    label=f"kill level {T.FLIP_DEAD}")
    axes[0].set_ylabel("captured power fraction")
    axes[1].set_ylabel("window flip rate")
    for ax in axes:
        ax.set_xlabel("window start step")
    axes[1].legend(frameon=False, fontsize=7)
    axes[0].set_yscale("log")
    fig.suptitle("Tracked dyn-mode coherence (plga block; onsets "
                 "dotted)", y=1.02)
    _save(fig, "fig_w6_mode")


def fig_force(report):
    """Redesigned force-law panel from the W4 pooled windows."""
    pooled = []
    for n in COL:
        try:
            _, s, _ = gates.load_run(n)
        except FileNotFoundError:
            continue
        on = gates.onset_step(s)
        ws, _ = gates.dyn_windows(s)
        pooled += [w for w in ws if on is not None and w["w0"] < on
                   and w["flip"] > T.FLIP_OSC and w["P"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    if pooled:
        uv = [w["u"] * w["V"] for w in pooled]
        du = [w["du"] for w in pooled]
        axes[0].scatter(uv, du, s=14, alpha=0.7)
        axes[0].set_xscale("log")
    axes[0].set_xlabel("$u \\cdot V_w$ (dyn-mode oscillation power)")
    axes[0].set_ylabel("$\\Delta u$ per window")
    w4 = report.get("W4_force_law", {})
    axes[0].set_title(
        f"n={w4.get('n_windows', 0)}, "
        f"$\\rho_{{uV}}$={w4.get('spearman_uV', float('nan')):.2f}, "
        f"$\\rho_{{uP}}$={w4.get('spearman_uP', float('nan')):.2f}")
    table = w4.get("tertile_table", {})
    if table:
        for i, (k, row) in enumerate(sorted(table.items())):
            axes[1].bar(np.arange(3) + 0.28 * i, row, width=0.26,
                        label=k)
        axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("$V_w$ tertile")
    axes[1].set_ylabel("mean $-\\Delta u$")
    axes[1].set_title("tertile pattern within $u$ tertiles")
    _save(fig, "fig_w6_force")


def fig_causal():
    """Damp/excite comparison: min-layer trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=False)
    pairs = [("w6-damp-plga-lr1e-3", "w6-base-lr1e-3"),
             ("w6-damp-plga-lr1e-3-s2", "w6-base-lr1e-3-s2"),
             ("w6-damp-attn-lr1e-3", "w6-base-lr1e-3")]
    for i, (damp, base) in enumerate(pairs):
        try:
            _, sd, _ = gates.load_run(damp)
            _, sb, _ = gates.load_run(base)
        except FileNotFoundError:
            continue
        xs, ys = gates.layer_agg(sd, "rowmap_sigma_med", "min")
        axes[0].plot(xs, ys, lw=1.0,
                     label=damp.replace("w6-", ""))
        xs, ys = gates.layer_agg(sb, "rowmap_sigma_med", "min")
        axes[0].plot(xs, ys, lw=0.8, ls="--", color="gray",
                     alpha=0.6)
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("damp cells (solid) vs seed-matched base "
                      "(dashed)")
    for n, lbl in [("w6-excite-plga-lr3e-4", "excite 2.0 @3e-4"),
                   ("w6-base-lr3e-4", "base 3e-4")]:
        try:
            _, s, _ = gates.load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = gates.layer_agg(s, "rowmap_sigma_med", "min")
        axes[1].plot(xs, ys, lw=1.0, label=lbl)
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_title("excitation at the sub-critical rate")
    for ax in axes:
        ax.set_xlabel("step")
        ax.set_ylabel("min-layer median $\\sigma(J_\\varphi)$")
    _save(fig, "fig_w6_causal")


def fig_carrier():
    """Carrier-search freeze panel."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, lr, base in zip(axes, ("lr1e-3", "lr2e-3"),
                            ("w6-base-lr1e-3", "w6-ref-lr2e-3")):
        cells = [(base, "base", "k"),
                 (f"w6-frzattn-{lr}", "frozen attn", "C0"),
                 (f"w6-frzffn-{lr}", "frozen ffn", "C1"),
                 (f"w5-frzplga-{lr}", "frozen plga (archived)", "C2")]
        for n, lbl, c in cells:
            try:
                _, s, _ = gates.load_run(n)
            except FileNotFoundError:
                continue
            xs, ys = gates.layer_agg(s, "rowmap_sigma_med", "min")
            ax.plot(xs, ys, lw=1.0, color=c, label=lbl)
        ax.set_yscale("log")
        ax.set_title(lr)
        ax.set_xlabel("step")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("min-layer median $\\sigma(J_\\varphi)$")
    fig.suptitle("Carrier search by block freezing", y=1.02)
    _save(fig, "fig_w6_carrier")


def fig_noise():
    """Matched-token batch panel."""
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    cells = [("w6-b8tok-lr1e-3", "b8, token-matched"),
             ("w6-base-lr1e-3", "b32 (base)"),
             ("w6-b64tok-lr1e-3", "b64, token-matched"),
             ("w6-b8acc4-lr1e-3", "b8 x accum 4 (identity)")]
    for n, lbl in cells:
        try:
            _, s, _ = gates.load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = gates.layer_agg(s, "rowmap_sigma_med", "min")
        # token axis so different step counts align
        cfg = gates.load_run(n)[0]
        toks = xs * cfg["batch"] * cfg.get("accum", 1) * cfg["ctx"]
        ax.plot(toks / 1e6, ys, lw=1.0, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("training tokens (M)")
    ax.set_ylabel("min-layer median $\\sigma(J_\\varphi)$")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("Noise against token exposure (matched tokens)")
    _save(fig, "fig_w6_noise")


def fig_reference():
    """Reference-endpoint panel: per-layer end medians + m_gen."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    names = REF + ["w6-ref-lr1.5e-3", "w6-ref-lr2e-3-24k"] + COL[:1]
    labels, mgens = [], []
    for i, n in enumerate(names):
        try:
            _, s, fin = gates.load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = gates.series(s, "rowmap_sigma_med")
        nl = len(ys[-1])
        med = [float(np.median([v[li] for x, v in zip(xs, ys)
                                if x > 0.75 * xs.max()]))
               for li in range(nl)]
        axes[0].scatter([i] * nl, med, s=18)
        labels.append(n.replace("w6-", ""))
        mgens.append(fin["m_gen"]["GLM"] if fin else np.nan)
    axes[0].axhline(0.1, color="k", ls="--", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=45, ha="right",
                            fontsize=6)
    axes[0].set_ylabel("per-layer last-quarter median $\\sigma$")
    axes[1].bar(range(len(labels)), mgens)
    axes[1].axhline(T.MGEN_THR, color="k", ls="--", lw=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right",
                            fontsize=6)
    axes[1].set_ylabel("$m^{\\mathrm{gen}}(G_{LM})$")
    fig.suptitle("Reference-endpoint repair", y=1.02)
    _save(fig, "fig_w6_reference")


def main():
    print("wave-6 verdicts:")
    rep = gates.wave6_report()
    with open(os.path.join(OUT, "w6_gate_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("  w6_gate_report.json:", json.dumps(rep.get("GATES", {})))

    print("corrected wave-5 re-audit:")
    rep5 = gates.corrected_wave5_report()
    with open(os.path.join(OUT, "w5_gate_report_corrected.json"),
              "w") as f:
        json.dump(rep5, f, indent=1, default=float)
    print("  w5_gate_report_corrected.json written")

    fig_metric()
    fig_mode()
    fig_force(rep)
    fig_causal()
    fig_carrier()
    fig_noise()
    fig_reference()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
