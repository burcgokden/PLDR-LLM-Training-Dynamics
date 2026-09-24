"""Wave-7 figures and report: avalanche event-size statistics with
CSN fits and the threshold sweep, constant-eta stationarity and the
marginal-stability margins, cross-block propagation matrices under
block freezes, size-duration and aftershock panels, and the
noise-dose / finite-size panels; plus w7_gate_report.json (the wave-7
verdicts of wave7_prereg.md W7.1-W7.8, computed by analysis/gates.py).

Usage: python3 analysis/make_figures_w7.py
Figures are written to docs/figures with the w7 tag; no invocation of
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
import avalanche as av  # noqa: E402
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

COL = gates.W6_COL
SUB = gates.W6_SUB
C_COL = ["#b2182b", "#d6604d", "#f4a582"]
C_SUB = ["#2166ac", "#4393c3", "#92c5de"]


def _save(fig, tag):
    fig.savefig(os.path.join(OUT, f"{tag}.pdf"))
    fig.savefig(os.path.join(OUT, f"{tag}.png"), dpi=110)
    plt.close(fig)
    print(f"  {tag}")


def _ccdf(ax, sizes, **kw):
    s = np.sort(np.asarray(sizes, float))
    if not len(s):
        return
    ax.loglog(s, 1.0 - np.arange(len(s)) / len(s), drawstyle="steps-post",
              **kw)


def fig_ccdf(rep):
    """(a) excluded event-size CCDFs with the MLE fits; (b) exponent
    stability under the declared threshold sweep; (c) interevent CV
    against the surrogate envelopes; (d) duration CCDFs at the lowered
    threshold."""
    r1 = rep["W7_1_tail"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    ax = axes[0, 0]
    for i, (col, sub) in enumerate(zip(COL, SUB)):
        ex = gates.excluded_pair(col, sub)
        sizes = [e["size"] for e in ex["events"]]
        _ccdf(ax, sizes, color=C_COL[i], label=f"{col} (n={len(sizes)})")
        exs = gates.excluded_pair(sub, col)
        _ccdf(ax, [e["size"] for e in exs["events"]], color=C_SUB[i],
              alpha=0.7, label=f"{sub} (n={len(exs['events'])})")
        fit = r1["cells"][col]["a"].get("fit")
        if fit:
            xs = np.geomspace(fit["xmin"], max(sizes), 40)
            frac = fit["n_tail"] / len(sizes)
            ax.loglog(xs, frac * (xs / fit["xmin"]) **
                      (1 - fit["alpha"]), color=C_COL[i], ls="--",
                      lw=0.8)
            ax.axvline(fit["xmin"], color=C_COL[i], ls=":", lw=0.6)
    ax.set_xlabel("event size $S$ (integrated excess)")
    ax.set_ylabel("CCDF")
    ax.legend(fontsize=5.5, loc="lower left")
    ax.set_title("event sizes, 3x threshold, data events excluded",
                 fontsize=8)

    ax = axes[0, 1]
    for i, col in enumerate(COL):
        sw = r1["cells"][col]["c_sweep"]
        keys = sorted(sw)
        vals = [sw[k]["alpha"] for k in keys]
        xs = np.arange(len(keys))
        ax.plot(xs, [v if v else np.nan for v in vals], "o-",
                color=C_COL[i], ms=3, label=col)
        prim = r1["cells"][col]["a"].get("fit")
        if prim:
            ax.axhline(prim["alpha"], color=C_COL[i], ls=":", lw=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=6)
    ax.axhline(1.5, color="k", ls="--", lw=0.8,
               label="mean-field $\\tau=3/2$")
    ax.set_ylabel("MLE tail exponent $\\hat\\tau$")
    ax.legend(fontsize=6)
    ax.set_title("exponent under the threshold sweep", fontsize=8)

    ax = axes[1, 0]
    for i, col in enumerate(COL):
        d = r1["cells"][col]["d"]
        ax.errorbar([i], [d["cv"]],
                    yerr=[[d["cv"] - d["ci"][0]], [d["ci"][1] - d["cv"]]]
                    if d["ci"] else None,
                    fmt="o", color=C_COL[i], capsize=3)
        if np.isfinite(d.get("surr_cv_q95_max", np.nan)):
            ax.plot([i - 0.2, i + 0.2], [d["surr_cv_q95_max"]] * 2,
                    color="k", lw=1.0)
            ax.plot([i - 0.2, i + 0.2], [d["surr_cv_q95_min"]] * 2,
                    color="k", lw=1.0, ls=":")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="Poisson")
    ax.axhline(T.AV_CV_MIN, color="gray", ls=":", lw=0.8)
    ax.set_xticks(range(len(COL)))
    ax.set_xticklabels([c.split("-")[-1] if "-s" in c else "s1"
                        for c in COL])
    ax.set_ylabel("interevent CV")
    ax.legend(fontsize=6)
    ax.set_title("clustering vs the surrogate envelopes "
                 "(solid ar1/phase max, dotted min)", fontsize=8)

    ax = axes[1, 1]
    for i, (col, sub) in enumerate(zip(COL, SUB)):
        ex = gates.excluded_pair(col, sub, mult=T.AV_DUR_MULT)
        dur = [e["dur"] for e in ex["events"]]
        if dur:
            vals, cnts = np.unique(dur, return_counts=True)
            ccdf = 1.0 - np.cumsum(cnts) / len(dur)
            ax.loglog(vals, np.maximum(ccdf, 1e-4), "o-",
                      color=C_COL[i], ms=3,
                      label=f"{col} (n={len(dur)})")
    ax.set_xlabel("event duration $D$ (steps), 2x threshold")
    ax.set_ylabel("CCDF")
    ax.legend(fontsize=5.5)
    ax.set_title("durations at the lowered threshold", fontsize=8)
    fig.suptitle("Wave 7: avalanche event statistics (W7.1)", y=1.005)
    _save(fig, "fig_w7_ccdf")


def fig_stationarity(rep):
    """Constant-eta cells: event rate, per-third exponents, the margin
    series, and per-third margin distributions."""
    r2 = rep["W7_2_stationarity"]
    r3 = rep["W7_3_margin"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    ax = axes[0, 0]
    for i, n in enumerate(gates.W7_CONST):
        try:
            ce = gates.cell_events(n, warmup=T.W7_CONST_STAT_START)
        except FileNotFoundError:
            continue
        xs, rs = av.rate_series(ce["events"], ce["T"], T.AV_RATE_WIN,
                                lo=T.W7_CONST_STAT_START)
        ax.plot(xs + T.AV_RATE_WIN / 2, rs, color=C_COL[i], label=n)
    try:
        cs = gates.cell_events(gates.W7_CONST_SUB,
                               warmup=T.W7_CONST_STAT_START)
        xs, rs = av.rate_series(cs["events"], cs["T"], T.AV_RATE_WIN,
                                lo=T.W7_CONST_STAT_START)
        ax.plot(xs + T.AV_RATE_WIN / 2, rs, color=C_SUB[0], ls="--",
                label=gates.W7_CONST_SUB)
    except FileNotFoundError:
        pass
    ax.set_xlabel("step")
    ax.set_ylabel(f"events / {T.AV_RATE_WIN} steps")
    ax.legend(fontsize=6)
    ax.set_title("event rate at constant $\\eta$ (no sweep)", fontsize=8)

    ax = axes[0, 1]
    for i, n in enumerate(gates.W7_CONST):
        cell = r2["cells"].get(n, {})
        if "thirds" not in cell:
            continue
        al = [t["alpha"] if isinstance(t["alpha"], float) else np.nan
              for t in cell["thirds"]]
        ax.plot([1, 2, 3], al, "o-", color=C_COL[i], label=n)
        if cell.get("alpha_full"):
            ax.axhline(cell["alpha_full"], color=C_COL[i], ls=":",
                       lw=0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("third of the stationary window")
    ax.set_ylabel("$\\hat\\tau$ per third")
    ax.legend(fontsize=6)
    ax.set_title("exponent stationarity (W7.2)", fontsize=8)

    ax = axes[1, 0]
    for i, n in enumerate(gates.W7_CONST):
        try:
            _, st, _ = gates.load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = gates.s_pre_live(st)
        m = np.asarray(xs) > T.W7_CONST_STAT_START
        ax.semilogy(np.asarray(xs)[m], np.asarray(ys)[m],
                    color=C_COL[i], lw=0.7, alpha=0.8, label=n)
        w = np.asarray(ys)[m]
        if len(w):
            for q, ls in ((0.5, "--"), (0.95, ":")):
                ax.axhline(np.quantile(w, q), color=C_COL[i], ls=ls,
                           lw=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("$\\eta_t\\,\\lambda^{\\rm pre,live}_{\\rm plga}$")
    ax.legend(fontsize=6)
    ax.set_title("margin series with q50/q95 (W7.3)", fontsize=8)

    ax = axes[1, 1]
    for i, n in enumerate(gates.W7_CONST):
        try:
            _, st, _ = gates.load_run(n)
        except FileNotFoundError:
            continue
        xs, ys = gates.s_pre_live(st)
        w = np.asarray(ys)[np.asarray(xs) > T.W7_CONST_STAT_START]
        w = w[np.isfinite(w) & (w > 0)]
        if len(w):
            ax.hist(np.log(w), bins=24, histtype="step",
                    color=C_COL[i], density=True,
                    label=f"{n} (A={r3['cells'][n]['skew_A']:.2f})"
                    if n in r3.get("cells", {})
                    and r3["cells"][n]["skew_A"] is not None else n)
    ax.set_xlabel("$\\ln(\\eta_t\\,\\lambda^{\\rm pre,live}_{\\rm plga})$")
    ax.set_ylabel("density")
    ax.legend(fontsize=6)
    ax.set_title("margin distribution: upper-edge compression",
                 fontsize=8)
    fig.suptitle("Wave 7: constant-$\\eta$ stationarity and marginal "
                 "stability (W7.2, W7.3)", y=1.005)
    _save(fig, "fig_w7_stationarity")


def _matrix_panel(ax, mat, blocks, title):
    n = len(blocks)
    grid = np.full((n, n), np.nan)
    for i, a in enumerate(blocks):
        for j, b in enumerate(blocks):
            m = mat.get(f"{a}->{b}")
            if m:
                grid[i, j] = m["ratio"]
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=5)
    for i in range(n):
        for j in range(n):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center",
                        va="center", fontsize=6,
                        color="w" if grid[i, j] < 3.5 else "k")
    ax.set_xticks(range(n))
    ax.set_xticklabels(blocks, fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(blocks, fontsize=6)
    ax.set_title(title, fontsize=7)
    ax.grid(False)
    return im


def fig_propagation(rep):
    """Lag-resolved coincidence-ratio matrices: base against frozen."""
    r5 = rep["W7_5_propagation"]
    panels = [("w6-base-lr1e-3", r5["base"].get("w6-base-lr1e-3")),
              ("w6-ref-lr2e-3", r5["base"].get("w6-ref-lr2e-3"))]
    for n in gates.W7_FROZEN:
        c = r5["frozen"].get(n)
        if c and "verdict" not in c:
            pm = gates._prop_matrix(n)
            panels.append((n, pm))
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(2.6 * ncol, 2.6 * nrow))
    axes = np.atleast_2d(axes)
    blocks = ["plga", "phi", "attn", "ffn", "rest"]
    im = None
    for k, (name, pm) in enumerate(panels):
        ax = axes[k // ncol, k % ncol]
        if pm is None:
            ax.axis("off")
            continue
        im = _matrix_panel(ax, pm["matrix"], blocks,
                           f"{name}\n({pm['total_rate_per_1k']:.0f}"
                           " ev/1k, all blocks)")
    for k in range(len(panels), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
                     label="coincidence ratio over chance")
    fig.suptitle("Wave 7: cross-block burst propagation and rerouting "
                 "under freezes (W7.5)", y=1.02)
    _save(fig, "fig_w7_propagation")


def fig_aftershock(rep):
    """Size-duration scaling and post-mainshock rate decay."""
    r4 = rep["W7_4_scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    for i, (col, sub) in enumerate(zip(COL, SUB)):
        ex = gates.excluded_pair(col, sub, mult=T.AV_DUR_MULT)
        ev = [e for e in ex["events"] if e["dur"] >= 2]
        if not ev:
            continue
        ax.loglog([e["dur"] for e in ev], [e["size"] for e in ev], "o",
                  ms=3, alpha=0.55, color=C_COL[i], label=col)
        sd = r4["cells"][col].get("size_duration")
        if sd:
            ds = np.linspace(2, max(e["dur"] for e in ev), 20)
            s0 = np.exp(np.mean(np.log([e["size"] for e in ev]))
                        - sd["gamma"]
                        * np.mean(np.log([e["dur"] for e in ev])))
            ax.loglog(ds, s0 * ds ** sd["gamma"], ls="--",
                      color=C_COL[i], lw=0.8,
                      label=f"$\\gamma={sd['gamma']:.2f}$")
    ax.set_xlabel("duration $D$ (steps), 2x threshold")
    ax.set_ylabel("size $S$")
    ax.legend(fontsize=5.5)
    ax.set_title("size-duration scaling (W7.4)", fontsize=8)

    ax = axes[1]
    lags = np.arange(0, T.AV_OMORI_LATE + 5, 5)
    for i, (col, sub) in enumerate(zip(COL, SUB)):
        om = r4["cells"][col].get("omori")
        if not om:
            continue
        ex = gates.excluded_pair(col, sub, mult=T.AV_DUR_MULT)
        starts = np.array([e["start"] for e in ex["events"]])
        counts = np.zeros(len(lags) - 1)
        for m in om["mains"]:
            d = starts - m
            for k in range(len(lags) - 1):
                counts[k] += np.sum((d > lags[k]) & (d <= lags[k + 1]))
        rate = counts / (len(om["mains"]) * np.diff(lags))
        ax.plot(lags[1:], rate, "o-", ms=3, color=C_COL[i],
                label=f"{col} (ratio {om['ratio']:.1f}, "
                      f"p={om['p']:.2f})")
    ax.axvline(T.AV_OMORI_EARLY, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("steps after mainshock")
    ax.set_ylabel("event rate (events/step)")
    ax.legend(fontsize=5.5)
    ax.set_title("aftershock decay after the largest events", fontsize=8)
    fig.suptitle("Wave 7: collective-relaxation discriminators", y=1.03)
    _save(fig, "fig_w7_aftershock")


def fig_dose(rep):
    """Noise dose at matched tokens; finite-size cutoff comparison."""
    r7 = rep["W7_7_dose"]
    r6 = rep["W7_6_finite_size"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    ax = axes[0]
    order = ["b8tok", "base", "b64tok"]
    xs = [8, 32, 64]
    if all(k in r7.get("cells", {}) for k in order):
        rates = [r7["cells"][k]["rate_per_1k"] for k in order]
        cis = [r7["cells"][k]["rate_ci"] for k in order]
        yerr = np.array(
            [[r - c[0] for r, c in zip(rates, cis)],
             [c[1] - r for r, c in zip(rates, cis)]])
        ax.errorbar(xs, rates, yerr=yerr, fmt="o-", capsize=3,
                    color="#b2182b", label="events/1k steps")
        acc = r7["cells"].get("b8acc4")
        if acc:
            ax.plot([8], [acc["rate_per_1k"]], "s", color="#2166ac",
                    label="batch 8 x accum 4 (identity)")
        ax2 = ax.twinx()
        ax2.plot(xs, [r7["cells"][k]["rate_per_Mtok"] for k in order],
                 "^--", color="gray", ms=4, lw=0.8,
                 label="events/M tokens")
        ax2.set_ylabel("events per M tokens", fontsize=7)
        ax2.grid(False)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.set_xlabel("batch size (matched tokens)")
    ax.set_ylabel(f"events / {T.AV_RATE_WIN} steps")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title(f"noise dose (W7.7: {r7.get('verdict')})", fontsize=8)

    ax = axes[1]
    styles = {"20.6M": "-", "110M": "--"}
    cols = {"w6-base-lr1e-3|w5e7-base-lr1e-3": "#b2182b",
            "w6-base-lr3e-4|w5e7-base-lr3e-4": "#2166ac",
            "w6-frzattn-lr1e-3|w6e8-frzattn-lr1e-3": "#4daf4a"}
    for pair, ent in r6.get("pairs", {}).items():
        for tag in ("20.6M", "110M"):
            b = (ent or {}).get(tag)
            if not b or not b.get("plga"):
                continue
            name = pair.split("|")[0 if tag == "20.6M" else 1]
            try:
                ce = gates.cell_events(name)
            except FileNotFoundError:
                continue
            _ccdf(ax, [e["size"] for e in ce["events"]],
                  color=cols.get(pair, "k"), ls=styles[tag],
                  alpha=0.85,
                  label=f"{name} ({tag}, q99={b['plga']['q99']:.2g})")
    ax.set_xlabel("event size $S$ (plga)")
    ax.set_ylabel("CCDF")
    ax.legend(fontsize=5)
    ax.set_title("finite-size comparison (W7.6, descriptive; "
                 "batch/steps not matched)", fontsize=8)
    fig.suptitle("Wave 7: trigger dose and scale", y=1.03)
    _save(fig, "fig_w7_dose")


def main():
    print("wave-7 verdicts:")
    rep = gates.wave7_report()
    with open(os.path.join(OUT, "w7_gate_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("  w7_gate_report.json:", json.dumps(rep.get("GATES", {})))
    fig_ccdf(rep)
    fig_stationarity(rep)
    fig_propagation(rep)
    fig_aftershock(rep)
    fig_dose(rep)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
