"""Wave-8 figures and report: size-duration scaling on the constant
cells, the anneal-trajectory (schedule-shape) scan, perturbation-
response (susceptibility), temporal branching, and the width scan; plus
the swept-threshold margin overview built from archived logs, the
w8_macros.tex inline-number macros, and w8_gate_report_current.json
(the wave-8 verdicts of wave8_prereg.md W8.1-W8.6, computed by
analysis/gates.py under the current tie-aware statistics; the
historical w8_gate_report.json is byte-frozen and never written).

Usage: python3 analysis/make_figures_w8.py
Figures are written to docs/figures with the w8 tag (plus
fig_sweep_margins); missing cells are skipped gracefully so the script
runs before and after the wave: pre-launch it emits the archived-log
figure and a report with missing verdicts, post-launch the full set.
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
import sandpile as sp  # noqa: E402
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

C_A = ["#b2182b", "#d6604d", "#f4a582"]   # archived const 1e-3 trio
C_N = ["#2166ac", "#4393c3", "#92c5de"]   # new const 7.5e-4 trio
C_F = {0.1: "#1b7837", 0.3: "#5aae61", 0.55: "#a6dba0", 1.0: "#762a83"}
BLK_C = {"plga": "#b2182b", "phi": "#2166ac", "attn": "#1b7837",
         "ffn": "#e08214"}


def _save(fig, tag):
    fig.savefig(os.path.join(OUT, f"{tag}.pdf"))
    fig.savefig(os.path.join(OUT, f"{tag}.png"), dpi=110)
    plt.close(fig)
    print(f"  {tag}")


def _have(name):
    return os.path.exists(os.path.join(gates.RUNS, name, "log.jsonl"))


def fig_gamma(rep):
    """(a,b) ln S vs ln D with OLS fits for both constant trios at the
    lowered threshold; (c) gamma with bootstrap CIs against the floor
    and the surrogate envelopes."""
    r = rep["W8_1_gamma"]
    if not r.get("cells"):
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    panels = [(axes[0], list(zip(gates.W7_CONST, gates.W8_CONST75)),
               C_A, r"constant $10^{-3}$ (archived)"),
              (axes[1], list(zip(gates.W8_CONST75, gates.W7_CONST)),
               C_N, r"constant $7.5\times10^{-4}$ (new)")]
    labels, gams, cis, envs = [], [], [], []
    for ax, pairs, colors, title in panels:
        for i, (name, partner) in enumerate(pairs):
            if not _have(name):
                continue
            ex = gates._dur_events_excluded(name, partner)
            ev = [e for e in ex["events"] if e["dur"] >= 2
                  and e["size"] > 0]
            if not ev:
                continue
            d = [e["dur"] for e in ev]
            s = [e["size"] for e in ev]
            ax.loglog(d, s, ".", ms=3.5, alpha=0.6, color=colors[i],
                      label=name.replace("-lr", " "))
            cell = r["cells"].get(name, {})
            sd = cell.get("size_duration")
            if sd:
                dd = np.array([min(d), max(d)], float)
                ax.loglog(dd, np.exp(sd["gamma"] * np.log(dd)
                                     + np.log(np.median(s))
                                     - sd["gamma"]
                                     * np.log(np.median(d))),
                          "-", color=colors[i], lw=1.0)
                labels.append(name)
                gams.append(sd["gamma"])
                cis.append(sd.get("ci", (np.nan, np.nan)))
                envs.append(cell.get("gamma_surr_q95", {}))
        ax.set_xlabel("duration D (steps)")
        ax.set_ylabel("size S")
        ax.set_title(title)
        ax.legend(fontsize=6, frameon=False)
    ax = axes[2]
    if gams:
        xs = np.arange(len(gams))
        lo = [g - c[0] for g, c in zip(gams, cis)]
        hi = [c[1] - g for g, c in zip(gams, cis)]
        ax.errorbar(xs, gams, yerr=[lo, hi], fmt="o", ms=4, capsize=3,
                    color="k")
        for x, e in zip(xs, envs):
            for k, c in (("ar1", "#e08214"), ("phase", "#8073ac")):
                if k in e and np.isfinite(e[k]):
                    ax.plot([x - 0.2, x + 0.2], [e[k], e[k]], "-",
                            color=c, lw=1.4)
        ax.axhline(T.AV_GAMMA_FLOOR, color="r", ls="--", lw=0.9,
                   label=r"$\gamma = 1$ floor")
        ax.axhspan(*T.W8_GAMMA_BAND, color="0.9", zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([n.replace("-const", "\nconst")
                            for n in labels], fontsize=5.5, rotation=30)
        ax.set_ylabel(r"size-duration exponent $\gamma$")
        ax.set_title("estimates, CIs, surrogate q95 bars")
        ax.legend(fontsize=6, frameon=False)
    _save(fig, "fig_w8_gamma")


def fig_anneal(rep):
    """(a) declared schedule shapes; (b) min-layer row-map trajectories
    for the schedule-shape cells; (c) late-window event rate against
    the anneal floor; (d) the late-anneal descent cell."""
    r = rep["W8_2_anneal"]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    ax = axes[0, 0]
    xs = np.arange(12000)
    for floor in (0.1, 0.3, 0.55):
        ax.plot(xs, sp.schedule_factors("cosine", 12000, 1000,
                                        alpha=floor),
                color=C_F[floor], label=f"cosine, floor {floor}")
    ax.plot(xs, sp.schedule_factors("const", 12000, 250),
            color=C_F[1.0], label="constant (ramp 250)")
    ax.plot(xs, sp.schedule_factors("hold", 12000, 1000, alpha=0.1,
                                    hold=T.W8_HOLD),
            color="#e08214", ls="--", label=f"hold to {T.W8_HOLD}")
    xl = np.arange(T.W8_LATE_STEPS)
    ax.plot(xl, sp.schedule_factors("hold", T.W8_LATE_STEPS, 1,
                                    alpha=0.1, hold=0),
            color="k", ls=":", label="late-anneal (descent)")
    ax.axvspan(T.W8_LATE_LO, 12000, color="0.92", zorder=0)
    ax.set_xlabel("step")
    ax.set_ylabel(r"schedule factor $\eta_t/\eta_{\max}$")
    ax.set_title("declared shapes (late window shaded)")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[0, 1]
    for floor, name in gates.W8_FLOOR_CELLS:
        if not _have(name):
            continue
        _, st, _ = gates.load_run(name)
        sx, sy = gates.layer_agg(st, "rowmap_sigma_med", "min")
        ax.semilogy(sx, sy, color=C_F[floor],
                    label=f"floor {floor} ({name})")
    for name, c, l in ((gates.W8_HOLD_CELL, "#e08214", "hold6k"),
                       (gates.W8_SHORT_CELL, "#8073ac", "short6k")):
        if _have(name):
            _, st, _ = gates.load_run(name)
            sx, sy = gates.layer_agg(st, "rowmap_sigma_med", "min")
            ax.semilogy(sx, sy, color=c, ls="--", label=l)
    ax.axhline(T.FLATTEN_THR, color="r", ls="--", lw=0.9)
    ax.set_xlabel("step")
    ax.set_ylabel(r"min-layer median $\sigma_{\max}(J_\varphi)$")
    ax.set_title("row-map trajectories")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1, 0]
    chain = [c for c in r.get("floor_chain", []) if "rate_per_1k" in c]
    if chain:
        f = [c["floor"] for c in chain]
        v = [c["rate_per_1k"] for c in chain]
        err = np.array([[c["rate_per_1k"] - c["rate_ci"][0],
                         c["rate_ci"][1] - c["rate_per_1k"]]
                        if c.get("rate_ci") else [0, 0]
                        for c in chain]).T
        ax.errorbar(f, v, yerr=err, fmt="o-", capsize=3, color="k")
        ax.set_xlabel("anneal floor")
        ax.set_ylabel("late-window events / 1k steps")
        ax.set_title(f"floor chain (verdict: "
                     f"{r.get('chain_verdict', 'n/a')})")
    ax = axes[1, 1]
    if _have(gates.W8_LATE_CELL):
        _, st, _ = gates.load_run(gates.W8_LATE_CELL)
        sx, sy = gates.layer_agg(st, "rowmap_sigma_med", "min")
        ax.semilogy(sx, sy, color="k", label="min-layer median")
        lx, ly = gates.series(st, "lr")
        ax2 = ax.twinx()
        ax2.plot(lx, ly, color="#762a83", lw=0.9, alpha=0.7)
        ax2.set_ylabel(r"$\eta_t$", color="#762a83")
        ax2.grid(False)
        ax.axhline(T.FLATTEN_THR, color="r", ls="--", lw=0.9)
        on = r.get("cells", {}).get("lateann", {}).get("onset")
        if on:
            ax.axvline(on, color="0.4", ls=":")
        ax.set_xlabel("step (continuation)")
        ax.set_ylabel(r"min-layer median $\sigma_{\max}$")
        ax.set_title("late-anneal descent from the non-collapsed "
                     "constant checkpoint")
        ax.legend(fontsize=6, frameon=False)
    _save(fig, "fig_w8_anneal")


def fig_susceptibility(rep):
    """(a) per-pulse response against the placement envelope; (b) chi
    against the base rate; (c) triggered fraction against its null."""
    r = rep["W8_3_susceptibility"]
    cells = [(n, c) for n, c in r.get("cells", {}).items()
             if "chi" in c]
    if not cells:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    ax = axes[0]
    width = 0.8 / len(cells)
    for i, (n, c) in enumerate(cells):
        pr_lr = c["base_lr"]
        _, st, _ = gates.load_run(n)
        g = gates.gnorm_series(n, "plga")
        ev, _ = av.extract_events(g[2], T.AV_THR_MULT, T.AV_MED_WIN,
                                  T.W7_CONST_STAT_START)
        pr = sp.pulse_response(ev, list(T.W8_PULSE_STARTS),
                               T.W8_PULSE_LEN, T.W8_RESP_WIN, g[0],
                               T.W7_CONST_STAT_START)
        xs = np.arange(len(pr["R"])) + i * width
        ax.bar(xs, pr["R"], width=width * 0.9,
               label=f"{pr_lr:g}", alpha=0.8)
        ax.axhline(c["envelope_q95"], color=f"C{i}", ls="--", lw=0.9)
    ax.set_xlabel("pulse index")
    ax.set_ylabel("response R (summed sizes)")
    ax.set_title("per-pulse response; dashed = placement q95")
    ax.legend(fontsize=6, frameon=False, title="base rate")
    ax = axes[1]
    lrs = [c["base_lr"] for _, c in cells]
    chi = [c["chi"]["chi"] for _, c in cells]
    err = np.array([[c["chi"]["chi"] - c["chi"]["ci"][0],
                     c["chi"]["ci"][1] - c["chi"]["chi"]]
                    for _, c in cells]).T
    ax.errorbar(lrs, chi, yerr=err, fmt="o-", capsize=3, color="k")
    ax.set_xscale("log")
    ax.set_xlabel(r"base constant rate $\eta$")
    ax.set_ylabel(r"susceptibility $\chi$ (response per unit dose)")
    ax.set_title(f"verdict: {r.get('verdict', 'n/a')}")
    ax = axes[2]
    tf = [c["triggered_frac"] for _, c in cells]
    te = [c["triggered_env_q95"] for _, c in cells]
    xs = np.arange(len(cells))
    ax.bar(xs - 0.15, tf, width=0.3, label="observed")
    ax.bar(xs + 0.15, te, width=0.3, label="placement q95",
           color="0.6")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lr:g}" for lr in lrs])
    ax.set_xlabel("base rate")
    ax.set_ylabel("triggered-pulse fraction")
    ax.legend(fontsize=6, frameon=False)
    _save(fig, "fig_w8_susceptibility")


def fig_branching(rep):
    """(a) sigma_hat per constant cell against the Poisson placement
    and surrogate envelopes; (b) cross-block offspring excess for the
    two lead cells."""
    r = rep["W8_4_branching"]
    cells = [(n, c) for n, c in r.get("cells", {}).items()
             if "sigma" in c]
    if not cells:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    xs = np.arange(len(cells))
    sig = [c["sigma"]["sigma"] for _, c in cells]
    for i, (n, c) in enumerate(cells):
        ci = c.get("sigma_ci")
        if ci:
            ax.plot([i, i], ci, "-", color="k", lw=1.0)
        ax.plot([i - 0.25, i + 0.25],
                [c["poisson_q95"], c["poisson_q95"]], "-",
                color="#762a83", lw=1.4)
        for k, col in (("ar1", "#e08214"), ("phase", "#8073ac")):
            v = c["surr_q95"].get(k)
            if v is not None and np.isfinite(v):
                ax.plot([i - 0.25, i + 0.25], [v, v], ":", color=col,
                        lw=1.2)
    ax.plot(xs, sig, "o", ms=5, color="k", zorder=5)
    ax.axhline(1.0, color="r", ls="--", lw=0.9,
               label=r"$\sigma = 1$ (critical)")
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("-const", "\nconst")
                        for n, _ in cells], fontsize=5.5, rotation=30)
    ax.set_ylabel(r"branching proxy $\hat\sigma$")
    ax.set_title("clusters vs Poisson (solid) and surrogate (dotted) "
                 "q95")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1]
    lead = [n for n in ("w7-const-lr1e-3", "w8-const-lr7.5e-4")
            if any(n == m for m, _ in cells)]
    w = 0.35
    for j, n in enumerate(lead):
        c = dict(cells)[n]
        cs = c.get("cross_sigma") or {}
        blks = [b for b in ("plga", "phi", "attn", "ffn") if cs.get(b)]
        ex = [cs[b]["excess"] for b in blks]
        ax.bar(np.arange(len(blks)) + (j - 0.5) * w, ex, width=w,
               label=n)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(("plga", "phi", "attn", "ffn"))
    ax.set_ylabel("cross-block offspring excess")
    ax.set_title("source-block excess over chance")
    ax.legend(fontsize=6, frameon=False)
    _save(fig, "fig_w8_branching")


def fig_fss(rep):
    """(a) pre-onset event-size CCDFs by width; (b) q99 and S_max
    against d_model, the 110M cell annotated separately."""
    r = rep["W8_5_fss"]
    if not r.get("widths"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    WC = {128: "#2166ac", 256: "#1b7837", 512: "#b2182b"}
    for width, cells in gates.W8_WIDTH:
        for name in cells:
            if not _have(name):
                continue
            config, st, _ = gates.load_run(name)
            ce = gates.cell_events(name)
            if ce is None:
                continue
            onset = gates.onset_step(st)
            pre = av.events_in(ce["events"],
                               config.get("warmup", 0) + 1,
                               onset if onset else ce["T"])
            s = np.sort([e["size"] for e in pre])
            if len(s):
                ax.loglog(s, 1.0 - np.arange(len(s)) / len(s),
                          drawstyle="steps-post", color=WC[width],
                          alpha=0.7, lw=1.0)
    for width, c in WC.items():
        ax.plot([], [], color=c, label=f"d_model {width}")
    ax.set_xlabel("pre-onset event size S")
    ax.set_ylabel("CCDF")
    ax.set_title("event sizes by width")
    ax.legend(fontsize=6, frameon=False)
    ax = axes[1]
    for width, cells in gates.W8_WIDTH:
        for r_c in r["widths"].get(width, []):
            cut = r_c.get("pre_onset_cutoffs")
            if not cut:
                continue
            mfc = "k" if r_c.get("collapsing") else "none"
            ax.errorbar([width], [cut["q"]],
                        yerr=[[cut["q"] - cut["q_ci"][0]],
                              [cut["q_ci"][1] - cut["q"]]],
                        fmt="o", ms=5, capsize=3, color="k", mfc=mfc)
            ax.plot([width], [cut["smax"]], "^", ms=4, color="0.5")
    s110 = r.get("scale_110m_descriptive")
    if s110 and s110.get("cutoffs"):
        ax.plot([896], [s110["cutoffs"]["q"]], "s", ms=5,
                color="#762a83", label="110M (descriptive)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$d_{\mathrm{model}}$")
    ax.set_ylabel(f"q{int(T.W8_FSS_Q * 100)} (o) and max (^) size")
    ax.set_title(f"cutoff growth (verdict: {r.get('verdict', 'n/a')}; "
                 "filled = collapsing)")
    ax.legend(fontsize=6, frameon=False)
    _save(fig, "fig_w8_fss")


def fig_sweep_margins():
    """The swept-threshold overview from archived logs: per-block
    margin statistic eta_t * lam_pre_live against the schedule, with
    onsets marked, for the three archived collapsing seeds.  The
    dotted reference is the frozen-preconditioner EMA-momentum
    boundary (never a verdict criterion for clipped AdamW)."""
    cells = [n for n in gates.W6_COL if _have(n)]
    if not cells:
        return
    fig, axes = plt.subplots(1, len(cells), figsize=(11.5, 3.4),
                             sharey=True)
    if len(cells) == 1:
        axes = [axes]
    for ax, name in zip(axes, cells):
        _, st, _ = gates.load_run(name)
        for blk, c in BLK_C.items():
            xs, ys = gates.s_pre_live(st, blk)
            if len(xs):
                ax.semilogy(xs, ys, color=c, lw=0.9, alpha=0.85,
                            label=blk)
        ax.axhline(38.0, color="0.3", ls=":", lw=1.0)
        on = gates.W7_ONSETS.get(name)
        if on:
            ax.axvline(on, color="r", ls="--", lw=0.9)
        lx, ly = gates.series(st, "lr")
        ax2 = ax.twinx()
        ax2.plot(lx, ly, color="0.55", lw=0.8, alpha=0.8)
        ax2.grid(False)
        ax2.set_yticks([])
        if ax is axes[-1]:
            ax2.set_ylabel(r"$\eta_t$ (gray)", color="0.4")
        ax.set_xlabel("step")
        ax.set_title(name, fontsize=8)
    axes[0].set_ylabel(r"$\eta_t\,\lambda^{\mathrm{pre,live}}_{"
                       r"\mathrm{blk}}$")
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    _save(fig, "fig_sweep_margins")


def _macro(fh, name, val, fmt="{:.3g}"):
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        fh.write(f"\\gdef\\{name}{{--}}\n")
    elif isinstance(val, str):
        fh.write(f"\\gdef\\{name}{{{val}}}\n")
    else:
        fh.write(f"\\gdef\\{name}{{{fmt.format(val)}}}\n")


def write_macros(rep):
    """w8_macros.tex: the inline numbers the manuscript quotes; every
    macro degrades to -- when its cell is missing."""
    path = os.path.join(OUT, "w8_macros.tex")
    g = rep["W8_1_gamma"]
    s = rep["W8_3_susceptibility"]
    b = rep["W8_4_branching"]
    a = rep["W8_2_anneal"]
    with open(path, "w") as fh:
        fh.write("% generated by analysis/make_figures_w8.py; "
                 "do not edit\n")
        names = {"w7-const-lr1e-3": "GamAOne",
                 "w7-const-lr1e-3-s2": "GamATwo",
                 "w7-const-lr1e-3-s3": "GamAThree",
                 "w8-const-lr7.5e-4": "GamNOne",
                 "w8-const-lr7.5e-4-s2": "GamNTwo",
                 "w8-const-lr7.5e-4-s3": "GamNThree"}
        for cell, tag in names.items():
            sd = g.get("cells", {}).get(cell, {}).get("size_duration")
            _macro(fh, f"wEight{tag}", sd["gamma"] if sd else None)
            _macro(fh, f"wEight{tag}Lo",
                   sd["ci"][0] if sd and "ci" in sd else None)
            _macro(fh, f"wEight{tag}Hi",
                   sd["ci"][1] if sd and "ci" in sd else None)
        _macro(fh, "wEightGammaVerdict", g.get("verdict"))
        for cell, tag in (("w8-pulse-const3e-4", "ChiLo"),
                          ("w8-pulse-const7.5e-4", "ChiMid"),
                          ("w8-pulse-const1e-3", "ChiHi")):
            c = s.get("cells", {}).get(cell, {})
            _macro(fh, f"wEight{tag}",
                   c.get("chi", {}).get("chi") if c.get("chi")
                   else None)
        _macro(fh, "wEightSuscVerdict", s.get("verdict"))
        for cell, tag in (("w7-const-lr1e-3", "SigA"),
                          ("w8-const-lr7.5e-4", "SigN"),
                          ("w7-const-lr3e-4", "SigSub")):
            c = b.get("cells", {}).get(cell, {})
            _macro(fh, f"wEight{tag}",
                   c.get("sigma", {}).get("sigma") if c.get("sigma")
                   else None)
        _macro(fh, "wEightBranchVerdict", b.get("verdict"))
        _macro(fh, "wEightChainVerdict", a.get("chain_verdict"))
        late = a.get("cells", {}).get("lateann", {})
        _macro(fh, "wEightLateOnset", late.get("onset"), "{:d}")
        _macro(fh, "wEightLateVerdict", late.get("verdict"))
        _macro(fh, "wEightAnnealVerdict", a.get("verdict"))
        _macro(fh, "wEightFssVerdict",
               rep["W8_5_fss"].get("verdict"))
        _macro(fh, "wEightConstVerdict",
               rep["W8_6_const_fallback"].get("verdict"))
    print("  w8_macros.tex")


def main():
    rep = gates.wave8_report()
    # The historical report at w8_gate_report.json is byte-frozen and
    # is NEVER written by any target (test_frozen_reports.py pins its
    # digest).  The current regeneration, which uses the tie-aware
    # Spearman implementation and adds distinct-duration counts, is
    # written under a distinct name so regenerating current analyses
    # cannot touch the frozen historical bytes.
    frozen = os.path.join(OUT, "w8_gate_report.json")
    if not os.path.exists(frozen):
        raise SystemExit(
            "make_figures_w8: frozen historical report missing at "
            + frozen)
    with open(os.path.join(OUT, "w8_gate_report_current.json"),
              "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print("  w8_gate_report_current.json (frozen report untouched)")
    fig_sweep_margins()
    fig_gamma(rep)
    fig_anneal(rep)
    fig_susceptibility(rep)
    fig_branching(rep)
    fig_fss(rep)
    write_macros(rep)
    print("wave-8 verdicts:", rep["GATES"])


if __name__ == "__main__":
    main()
