"""make_figures.py -- publication figures/tables, generated ONLY from result CSVs.

Nothing is hand-entered: every value is read from experiments/*/ or csvs/, so a figure
can never drift from the run that produced it. Re-run after any new sweep.

    pip install matplotlib
    python make_figures.py

PLOTTING CONVENTIONS (why the plots look the way they do)
---------------------------------------------------------
1. EMPTY-SET vs GENUINE ZERO. When CARE certifies nothing, the auto-applied set S is
   empty and the selective error is 0 by the convention R:=0. That is NOT a good
   result, and plotting it as "zero error" next to a real zero would be misleading.
   Every figure therefore marks empty-S points with hollow markers and excludes them
   from error axes; the Pareto figure places them on an explicit "nothing applied" band.
2. VARIANCE IS SHOWN. Pareto points carry the bootstrap CI upper bound (err_hi), which
   is exactly the quantity the `covered` flag tests against alpha -- not a decorative
   error bar.
3. ZERO VARIANCE IS EXPLAINED, NOT HIDDEN. On tax the error is 0 in all 10 seeds
   because CARE certifies precisely the columns where the proposer is 100% accurate
   (fig5 shows this decomposition). The figure exists so the reader can verify the
   mechanism rather than wonder.
4. NO SMOOTHING, NO INTERPOLATION. Only measured alphas are drawn as markers; lines
   connect measurements to aid reading and are not model fits.

OUTPUT MAP (file -> paper numbering): fig5_mechanism = Fig. 2, fig9_error_rate = Fig. 3,
fig2_pareto = Fig. 4, fig8_heldout = Fig. 5; tab1_audit / tab2_theorem2 / tab3_nonoracle
= Tables 1-3. fig7_label_budget, tab4_calibration and tab6_sweep are artifact-only (the
paper quotes their numbers in the text). tab5_error_drop.tex is written by
audit_error_drop.py --tex, not by this script.
"""
from __future__ import annotations

import csv
import os

OUT_FIG, OUT_TAB, EXP = "paper/figures", "paper/tables", "experiments"

# consistent identity for every proposer across every figure
STYLE = {
    "Baran":       dict(color="#1b4f72", marker="o", ls="-"),
    "BClean":      dict(color="#7d6608", marker="s", ls="--"),
    "Jellyfish":   dict(color="#78281f", marker="D", ls="-."),
    "GPT-4o-mini": dict(color="#186a3b", marker="v", ls=":"),
    "RetClean":    dict(color="#5b2c6f", marker="^", ls=(0, (4, 1, 1, 1))),
    "HoloClean":   dict(color="#117864", marker="P", ls=(0, (1, 1))),
}
CLEANERS = [("Baran", "results_baran_errors"), ("BClean", "results_bclean"),
            ("Jellyfish", "results_detected"), ("RetClean", "results_retclean"),
            ("GPT-4o-mini", "results_gpt4o"), ("HoloClean", "results_holoclean")]
DATASETS = ["hospital", "beers", "flights", "rayyan", "tax"]
EXCLUDE = {("Baran", "flights"), ("BClean", "rayyan"), ("BClean", "tax"),
           # RetClean on hospital reproduces Jellyfish exactly (its lake retrieved
           # nothing usable, so every repair fell through to the same LLM). Reporting
           # it would double-count one proposer. See REPRODUCE.md D29.
           ("RetClean", "hospital")}
# HoloClean on beers is degenerate in a different way: 711 of its 878 changes are null
# imputations on cells the gold standard does not label as errors, leaving 121
# constraint-driven repairs, all correct. The paper excludes the pair from Table 1 only,
# so its harm is not quoted there (the figures keep the points they had).
TAB1_EXCLUDE = EXCLUDE | {("HoloClean", "beers")}

# Per-pair results-directory overrides, for sweeps that could not share a directory
# with the rest of their proposer's runs. Jellyfish/tax was run on a 25,000-row shard
# with --row-range, so it lives in its own directory and must not be confused with a
# full-table run; audit_proposers.SHARDS records the matching row bounds.
RESULT_OVERRIDE = {("Jellyfish", "tax"): "results_jelly_tax25k"}
EMPTY = 0.999          # human_cost at/above this  <=>  auto-applied set is empty


def _read(p):
    return list(csv.DictReader(open(p, newline=""))) if os.path.exists(p) else []


def _baran_medians():
    """Baran's row per dataset as a median over draws (audit_baran_median.py).

    Baran's active-learning sample is unseeded, so one run is a draw rather than a
    measurement -- and the log that happens to sit in csvs/ is the MAXIMUM of six
    draws on hospital. Table 1 therefore reports the median of every column for the
    datasets where draws exist, so accuracy, pi_0 and automation all describe the
    same central run instead of mixing one lucky draw's accuracy with another's
    automation. Datasets without draws (tax) fall through to the single-run audit.
    """
    return {r["dataset"]: r for r in _read(f"{EXP}/baran_median.csv")}


# Baran's active-learning sample is unseeded; these directories hold one result file per
# independent draw (audit_baran_median.py). Table 1 reports medians over them, and the
# figures below draw the same population: every draw as a small marker, the median as the
# full-size one, so no figure shows the lucky shipped draw as if it were the measurement.
BARAN_DRAW_DIRS = {"hospital": "results_baran_var_hospital",
                   "beers": "results_baran_var_beers",
                   "rayyan": "results_baran_var_rayyan",
                   "tax": "results_baran_var_tax"}


def _baran_draw_rows(ds):
    """Pareto rows for every Baran draw on ``ds`` (shipped log + var directories)."""
    import glob as _glob
    if ds not in BARAN_DRAW_DIRS:
        return []
    paths = [os.path.join(EXP, "results_baran_errors", f"{ds}_pareto.csv")] + sorted(
        _glob.glob(os.path.join(EXP, BARAN_DRAW_DIRS[ds], "*", f"{ds}_pareto.csv")))
    return [_read(p) for p in paths if os.path.exists(p)]


def pareto(rd, ds, cleaner=None):
    """Result rows for one (cleaner, dataset). RESULT_OVERRIDE wins when the pair was
    run into its own directory (row-sharded sweeps)."""
    rd = RESULT_OVERRIDE.get((cleaner, ds), rd) if cleaner else rd
    return _read(os.path.join(EXP, rd, f"{ds}_pareto.csv"))


def care(rows, strata="mondrian"):
    """[(alpha, human_cost, err, err_hi, covered)] sorted by alpha.

    ``strata="best"`` picks column x source control when the sweep produced it (i.e.
    the log mixed decision types) and falls back to column-only otherwise. That is the
    right default for reporting a proposer's operating point, because for a mixed log
    the column-only number is depressed by an artifact -- two incompatible score scales
    sharing one threshold -- rather than by the proposer's quality.
    """
    if strata == "best":
        for s in ("mondrian_src", "mondrian"):
            got = care(rows, s)
            if got:
                return got
        return []
    out = [(float(r["alpha"]), float(r["human_cost"]), float(r["realized_error"]),
            float(r.get("err_hi") or 0.0), r.get("covered") == "True")
           for r in rows if r.get("baseline") == "CARE" and r.get("strata") == strata]
    return sorted(out)


def applyall(rows):
    for r in rows:
        if r.get("baseline") == "LLMOnly":
            return float(r["realized_error"])
    return None


def _style_axes(ax, fs=9):
    ax.grid(alpha=0.25, lw=0.5, ls=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=fs - 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ figures

def fig2(plt):
    """Operating points at alpha=0.2 with CI upper bounds. Empty-S points are shown
    on a separate band so a 'zero error' that means 'nothing applied' cannot be
    mistaken for a genuine zero."""
    fig, ax = plt.subplots(figsize=(3.34, 2.3))
    import statistics as _st
    for c, rd in CLEANERS:
        st = STYLE[c]
        xs, ys, es, hollow = [], [], [], []
        for d in DATASETS:
            if (c, d) in EXCLUDE:
                continue
            draws = _baran_draw_rows(d) if c == "Baran" else []
            if len(draws) >= 2:
                # every draw small and faint; the median of the draws full-size
                pts = []
                for rows in draws:
                    for a, h, e, ehi, _cov in care(rows):
                        if abs(a - 0.2) < 1e-9:
                            pts.append((h, e))
                for h, e in pts:
                    ax.plot(h, e, marker=st["marker"], ms=3, ls="none",
                            color=st["color"], alpha=0.35, zorder=2)
                xs.append(_st.median(p[0] for p in pts))
                ys.append(_st.median(p[1] for p in pts))
                es.append(0.0); hollow.append(False)
                continue
            for a, h, e, ehi, _cov in care(pareto(rd, d, c)):
                if abs(a - 0.2) > 1e-9:
                    continue
                xs.append(h); ys.append(e); es.append(max(0.0, ehi - e))
                hollow.append(h >= EMPTY)
        if not xs:
            continue
        for x, y, er, ho in zip(xs, ys, es, hollow):
            ax.errorbar(x, y, yerr=[[0], [er]], fmt=st["marker"], ms=5, lw=0,
                        elinewidth=0.9, capsize=2, ecolor=st["color"],
                        color="white" if ho else st["color"],
                        markeredgecolor=st["color"], markeredgewidth=1.1, zorder=3)
        ax.plot([], [], marker=st["marker"], ls="none", color=st["color"],
                ms=5, label=f"CARE + {c}")
    aa = [applyall(pareto(rd, d, c)) for c, rd in CLEANERS for d in DATASETS
          if (c, d) not in EXCLUDE and applyall(pareto(rd, d, c)) is not None]
    if aa:
        ax.plot([0.0] * len(aa), aa, "x", c="#909497", ms=5, mew=1.2,
                label="ungoverned apply-all", zorder=2)
    ax.axhline(0.2, ls="--", c="#c0392b", lw=1, zorder=1)
    ax.text(0.02, 0.212, r"budget $\alpha=0.2$", fontsize=7, color="#c0392b")
    ax.axvspan(EMPTY, 1.02, color="#eaeded", zorder=0)
    ax.text(0.995, 0.55, "nothing\napplied", fontsize=7, ha="right", color="#566573",
            rotation=90, va="center")
    ax.set_xlabel("human cost (fraction of repairs escalated)", fontsize=9)
    ax.set_ylabel("error on applied repairs", fontsize=9)
    ax.set_xlim(-0.04, 1.02)
    _style_axes(ax)
    # legend inside the axes: the region x in [0.15, 0.95], y > 0.3 holds no point
    # (governed points sit below the budget line, apply-all crosses sit at x=0)
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2,
              bbox_to_anchor=(0.55, 1.02), columnspacing=0.8, handletextpad=0.3,
              labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig2_pareto.pdf", bbox_inches="tight")
    print("  fig2_pareto.pdf")


def _poison_series():
    out = []
    # Theorem 2 exhibits. Configurations are listed explicitly rather than globbed so
    # that a vacuous run cannot silently join the table: if the gate certifies too
    # little for any poisoned cell to reach the applied set, eps_S stays 0 at every
    # level and the "agreement" is between two zeros.
    #
    # The two Jellyfish configurations were exactly that and have been REPLACED by
    # Baran/beers and Baran/rayyan. The tuned LLM certifies 0.00 at every budget on
    # hospital, flights and rayyan and 0.04 on beers, so no poisoned cell can reach an
    # applied set that is empty. That is the gate refusing an unreliable proposer --
    # correct behaviour, and the reason the LLM paradigm cannot carry a Theorem 2
    # exhibit. Section 7.6 states this rather than showing two rows of zeros.
    #
    # A KEEP LOG CANNOT CARRY ONE EITHER, which is why HoloClean is absent despite
    # being the only remaining third paradigm. Multi-source logs are stratified by
    # column x source (deviation D23). Trusted poison is attributed to its own source,
    # so it forms its own stratum, is judged on its own error alone, and is refused --
    # eps_S stays 0.000 at every poison level even though the auto-applied set is
    # non-empty. Measured on both HoloClean/hospital and BClean(+keep)/hospital. The
    # Theorem 1 reading of that is favourable (the gate catches all of it); the
    # Theorem 2 exhibit needs eps_S to RISE, so these configurations cannot supply one.
    # Consequence: Table 2 spans five configurations and TWO paradigms, not three.
    for lab, path, a in [
            ("Baran / beers", f"{EXP}/results_poison_baran_beers/beers_trustedpoison_poison.csv", 0.2),
            ("Baran / rayyan", f"{EXP}/results_poison_baran_rayyan/rayyan_trustedpoison_poison.csv", 0.2),
            ("BClean / flights", f"{EXP}/results_poison_bclean_flights/flights_trustedpoison_poison.csv", 0.2),
            ("Baran / hospital", f"{EXP}/results_poison_baran_hospital/hospital_trustedpoison_poison.csv", 0.2),
            ("BClean / hospital", f"{EXP}/results_poison_bclean_hospital/hospital_trustedpoison_poison.csv", 0.2)]:
        rs = [r for r in _read(path) if r.get("baseline") == "CARE"]
        if not rs:
            print(f"  !! tab2: no CARE rows in {path} -- run the poison sweeps in reproduce.sh")
            continue
        pts = sorted((float(r["applied_contamination"]), float(r["realized_error"]),
                      float(r.get("err_lo") or 0), float(r.get("err_hi") or 0)) for r in rs)
        base = next((e for eps, e, _l, _h in pts if eps < 1e-9), 0.0)
        out.append((lab, a, base, pts))
    return out


PROPOSER_LABEL = {"baran": "Baran", "bclean": "BClean", "jellyfish": "Jellyfish",
                  "gpt4o": "GPT-4o-mini", "holoclean": "HoloClean",
                  "retclean": "RetClean"}


def fig9(plt, alpha=0.2, partition="mondrian"):
    """A(alpha) against the pre-registered ceiling, as the injected error rate rises.

    One panel per base table. The x-axis is the MEASURED cell error rate, never the
    level in Ni et al.'s file name -- beers inner-30 is 0.239 and hospital inner-30 is
    0.185, so plotting the nominal level would put unlike points on the same abscissa.

    Solid line with filled markers: A(alpha). Dotted grey line with square markers:
    the proposer's accuracy on error cells, the quantity a mean-based reading would
    extrapolate from. Dashed line: the floor-adjusted ceiling
    pi_alpha, computed and sealed before any calibration was spent. A hollow marker
    marks a run that certified nothing, which is the outcome the ceiling is supposed to
    predict -- the dashed line should reach zero at the same x. Triangles are the
    `outer` (typo-like) contrast points at their own measured rate; they belong to the
    same proposer and colour but a different error kind, so they are never joined into
    the inner-error line.
    """
    # Prefer the median-over-draws file when it exists: Baran is unseeded, so a curve
    # of single draws would contradict Table 1, which already reports medians.
    src = f"{EXP}/sweep_median.csv"
    if not os.path.exists(src):
        src = f"{EXP}/sweep.csv"
    rows = [r for r in _read(src)
            if r.get("partition") == partition
            and abs(float(r["alpha"]) - alpha) < 1e-9
            and r.get("cell_error_rate") not in ("", None)]
    if not rows:
        print("  fig9 skipped (no experiments/sweep.csv; run audit_sweep.py)")
        return
    draws = max((int(r.get("draws", 1) or 1) for r in rows), default=1)

    bases = sorted({r["base"] for r in rows})
    ncol = 2 if len(bases) > 2 else len(bases)
    nrow = (len(bases) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.34, 1.15 * nrow + 0.4),
                             sharey=True, squeeze=False)
    seen = {}
    for i, base in enumerate(bases):
        ax = axes[i // ncol][i % ncol]
        for tag in sorted({r["proposer"] for r in rows if r["base"] == base}):
            name = PROPOSER_LABEL.get(tag, tag)
            st = STYLE.get(name, dict(color="#444444", marker="o", ls="-"))
            sub = sorted((r for r in rows if r["base"] == base and r["proposer"] == tag),
                         key=lambda r: float(r["cell_error_rate"]))
            inner = [r for r in sub if r["kind"] == "inner"]
            outer = [r for r in sub if r["kind"] == "outer"]
            if inner:
                x = [float(r["cell_error_rate"]) for r in inner]
                a = [float(r["A"]) for r in inner]
                c = [float(r["pi_alpha_floor"]) for r in inner]
                if all(r.get("acc") not in ("", None) for r in inner):
                    ax.plot(x, [float(r["acc"]) for r in inner], ls=":", lw=0.9,
                            color="#909497", marker="s", ms=2.2, zorder=1)
                ax.plot(x, c, ls="--", lw=0.8, color=st["color"], alpha=0.55, zorder=2)
                # Range bar over draws, where there is more than one. Baran's spread
                # across draws is the widest uncertainty in the paper, so hiding it
                # behind a median line would misrepresent the curve.
                if all("A_min" in r for r in inner):
                    for r, xi in zip(inner, x):
                        lo, hi = float(r["A_min"]), float(r["A_max"])
                        if hi - lo > 1e-9:
                            ax.plot([xi, xi], [lo, hi], color=st["color"], lw=0.7,
                                    alpha=0.45, zorder=2, solid_capstyle="butt")
                ax.plot(x, a, ls=st["ls"], lw=1.0, color=st["color"], zorder=3)
                # Hollow = certified nothing. Filled and hollow must be visually
                # distinct: an empty set at A=0 and a genuine A=0 look identical on a
                # line, and only the first is the predicted collapse.
                for xi, ai in zip(x, a):
                    ax.plot([xi], [ai], marker=st["marker"], ms=3.0, ls="none",
                            color=st["color"], zorder=4,
                            mfc=("none" if ai <= 1e-9 else st["color"]),
                            mew=0.8)
                seen[name] = st
            for r in outer:
                ax.plot([float(r["cell_error_rate"])], [float(r["A"])], marker="^",
                        ms=3.4, ls="none", color=st["color"], mfc="none", mew=0.9,
                        zorder=5)
        ax.set_title(base, fontsize=8, pad=2)
        ax.set_xlim(0, max(float(r["cell_error_rate"]) for r in rows) * 1.05)
        ax.set_ylim(-0.03, 1.03)
        _style_axes(ax, fs=8)
        if i % ncol == 0:
            ax.set_ylabel("fraction", fontsize=8)
        if i // ncol == nrow - 1:
            ax.set_xlabel("cell error rate", fontsize=8)
    for j in range(len(bases), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if seen:
        handles = [plt.Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"],
                              ms=3, lw=1.0, label=rf"$A({alpha:g})$, {n}")
                   for n, s in sorted(seen.items())]
        handles.append(plt.Line2D([], [], color="#909497", ls=":", lw=0.9, marker="s",
                                  ms=2.2, label="accuracy"))
        handles.append(plt.Line2D([], [], color="#666666", ls="--", lw=0.8,
                                  label=r"ceiling $\pi_\alpha$"))
        fig.legend(handles=handles, fontsize=7, frameon=False, ncol=3,
                   loc="lower center", bbox_to_anchor=(0.5, -0.06),
                   handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig9_error_rate.pdf", bbox_inches="tight")
    print(f"  fig9_error_rate.pdf ({len(rows)} points, {len(bases)} tables, "
          f"alpha={alpha}, {partition}, "
          f"{'median of %d draws' % draws if draws > 1 else 'SINGLE DRAW'})")


def tab6():
    """Every swept variant: measured rate, accuracy, sealed ceiling, A, R, coverage.

    Artifact and extended-version material -- far too long for the body, and the point
    of the figure is that it does not need reading. Sorted so each proposer's series
    reads down the page in increasing error rate.
    """
    rows = [r for r in _read(f"{EXP}/sweep.csv") if r.get("partition") == "mondrian"]
    if not rows:
        print("  tab6 skipped (no experiments/sweep.csv)")
        return
    rows.sort(key=lambda r: (r["proposer"], r["base"], r["kind"],
                             float(r["cell_error_rate"] or 0), float(r["alpha"])))
    out = ["% generated by make_figures.py -- do not edit",
           r"\begin{tabular}{lllrrrrrrrc}", r"\toprule",
           r"proposer & table & kind & rate & $n$ & acc & $\pi_0$ & "
           r"$\pi_\alpha^{\mathrm{fl}}$ & $\alpha$ & $A$ & cov \\", r"\midrule"]
    last = None
    for r in rows:
        key = (r["proposer"], r["base"], r["kind"])
        if last and key != last:
            out.append(r"\addlinespace[2pt]")
        last = key
        cov = r"\checkmark" if str(r["covered"]).lower() == "true" else r"\textbf{!}"
        out.append(
            f"{PROPOSER_LABEL.get(r['proposer'], r['proposer'])} & {r['base']} & "
            f"{r['kind']} & {float(r['cell_error_rate']):.3f} & {r['n']} & "
            f"{float(r['acc']):.3f} & {float(r['pi0']):.3f} & "
            f"{float(r['pi_alpha_floor']):.3f} & {r['alpha']} & "
            f"{float(r['A']):.2f} & {cov} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{OUT_TAB}/tab6_sweep.tex", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"  tab6_sweep.tex ({len(rows)} rows)")


def fig8(plt):
    """The naive held-out threshold against CARE, as a function of label budget.

    The naive rule accepts a stratum when its OBSERVED calibration error is <= alpha;
    CARE requires a 1-delta upper bound. The two converge where labels are plentiful
    and diverge where they are scarce -- which Section 7.3 shows is the regime that
    decides certifiability. Violation rate over seeds is the y-axis; delta is the
    rate a valid method is allowed.
    """
    rows = _read(f"{EXP}/heldout_budget.csv")
    if not rows:
        print("  fig8: skipped (run audit_heldout_budget.py first)"); return
    COL = {("Baran", "beers"): "#7d6608", ("BClean", "flights"): "#78281f",
           ("Jellyfish", "beers"): "#1b4f72"}
    fig, ax = plt.subplots(figsize=(3.34, 2.05))
    delta = None
    for (prop, ds), col in COL.items():
        pts = sorted((int(r["cal_size"]), int(r["seeds"]),
                      int(r["heldout_violations"]), int(r["care_violations"]))
                     for r in rows if r["proposer"] == prop and r["dataset"] == ds)
        if not pts:
            continue
        delta = float([r for r in rows if r["proposer"] == prop][0]["delta"])
        ax.plot([p[0] for p in pts], [p[2] / p[1] for p in pts], ls="-", marker="o",
                ms=3, lw=1.3, color=col, label=f"{prop}/{ds}")
        ax.plot([p[0] for p in pts], [p[3] / p[1] for p in pts], ls="--", marker="s",
                ms=2.5, lw=1.0, color=col, alpha=0.8)
    if delta is not None:
        ax.axhline(delta, ls=":", lw=0.9, color="#c0392b")
        ax.text(26, delta + 0.015, rf"permitted rate $\delta={delta:g}$", fontsize=7,
                color="#c0392b")
    ax.set_xscale("log")
    _cs = sorted({int(r["cal_size"]) for r in rows})
    ax.set_xticks(_cs); ax.set_xticklabels([str(c) for c in _cs])
    ax.minorticks_off()
    ax.set_xlabel("labelled calibration cells", fontsize=9)
    ax.set_ylabel("violations / seeds", fontsize=9)
    ax.set_ylim(-0.03, 0.85)
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="#566573", ls="-", marker="o", ms=3, lw=1.2),
                plt.Line2D([], [], color="#566573", ls="--", marker="s", ms=2.5, lw=1.0)]
    labels += ["held-out threshold", "CARE"]
    ax.legend(handles, labels, fontsize=7, frameon=False, loc="upper right",
              handlelength=2.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig8_heldout.pdf", bbox_inches="tight")
    print("  fig8_heldout.pdf")


def fig7(plt):
    """What does certification cost in labelled cells?

    A1 asks for a clean calibration set, and what one costs to label is the practical
    question. This answers with a purchase curve rather than an inequality: how much
    certifiable automation a fixed number of labelled cells buys. The shape is the
    result -- each curve saturates, and where it saturates is set by the proposer's
    pi_0 rather than by the budget, so an operator can tell when to stop labelling.
    A curve flat at zero says labelling will never help, which is also worth knowing
    before spending the afternoon.
    """
    rows = _read(f"{EXP}/label_budget.csv")
    if not rows:
        print("  fig7: skipped (run audit_label_budget.py first)"); return
    COL = {("Baran", "hospital"): "#1b4f72", ("Baran", "beers"): "#7d6608",
           ("BClean", "flights"): "#78281f"}
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    drew = False
    for (prop, ds), col in COL.items():
        for mode, ls, mk in (("mondrian", "-", "o"), ("marginal", ":", "s")):
            pts = sorted((int(r["labels"]), float(r["automation"]))
                         for r in rows
                         if r["proposer"] == prop and r["dataset"] == ds
                         and r["strata"] == mode)
            if len(pts) < 2:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, color=col,
                    marker=mk, ms=3, lw=1.3,
                    label=f"{prop}/{ds}" if mode == "mondrian" else None)
            drew = True
    if not drew:
        print("  fig7: no rows"); return
    ax.set_xscale("log")
    ax.set_xlabel("labelled calibration cells", fontsize=8)
    ax.set_ylabel(r"certifiable automation $A(0.2)$", fontsize=8)
    ax.set_ylim(-0.03, 1.05)
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="#566573", ls="-", marker="o", ms=3, lw=1.2),
                plt.Line2D([], [], color="#566573", ls=":", marker="s", ms=3, lw=1.2)]
    labels += ["column-conditional", "marginal"]
    ax.legend(handles, labels, fontsize=5.8, frameon=False, loc="center right",
              handlelength=2.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig7_label_budget.pdf", bbox_inches="tight")
    print("  fig7_label_budget.pdf")


def tab4():
    """Calibration table: the per-run calibration numbers behind Section 8.3."""
    rows = _read(f"{EXP}/calibration.csv")
    if not rows:
        print("  tab4: skipped (run audit_calibration.py first)"); return
    rows.sort(key=lambda r: -float(r["gap"]))
    out = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"proposer & dataset & $n$ & conf. & acc. & ECE \\", r"\midrule"]
    for r in rows:
        out.append(f"{r['proposer']} & {r['dataset']} & {int(r['n']):,} & "
                   f"{float(r['mean_conf']):.3f} & {float(r['accuracy']):.3f} & "
                   f"{float(r['ece']):.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{OUT_TAB}/tab4_calibration.tex", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("  tab4_calibration.tex")


def _cp_upper(k, n, delta):
    """Exact one-sided Clopper-Pearson upper bound on a binomial rate, by bisection on
    the binomial CDF. Same bound the certifier uses, recomputed here independently so
    the figure is not simply echoing the implementation it is meant to explain."""
    import math
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(120):
        m = (lo + hi) / 2
        s = sum(math.comb(n, i) * m ** i * (1 - m) ** (n - i) for i in range(k + 1))
        lo, hi = (m, hi) if s > delta else (lo, m)
    return hi


# (dx pt, dy pt, ha) for the dataset labels next to Baran's points in fig5(b)
_BARAN_LABEL_OFFSET = {"beers": (-5, 3, "right"), "tax": (5, 5, "left"),
                       "rayyan": (5, 0, "left"), "hospital": (5, 0, "left")}


def fig5(plt):
    """Two panels that together answer 'why is this exactly 0, and why is it flat?'

    (a) The certification spectrum on tax: the per-column error bound the certifier
        actually thresholds. There is a wide empty gap between the fifth and sixth
        column, so every budget inside the gap selects the identical column set --
        which is why A(alpha) is flat and the error is exactly 0 in every seed.
    (b) The response curve: CARE's automation against how accurate the proposer was.
        This places the degenerate (100% accurate) and abstaining (<5% accurate) pairs
        at the ends of one measured curve rather than treating them as anomalies."""
    import collections
    acc_rows = _read("experiments/proposer_accuracy.csv")
    try:
        from bench.datasets import load
    except Exception as e:
        print(f"  fig5: skipped ({e})"); return
    path = "csvs/baran_tax_mapped.csv"
    if not (os.path.exists(path) and acc_rows):
        print("  fig5: need csvs/baran_tax_mapped.csv + audit_proposers.py output")
        return

    _art, defects, _ = load("tax")
    gold = {k: (str(v).lower().strip() if v is not None else v)
            for k, v in defects.gold.items()}
    per, bad = collections.Counter(), collections.Counter()
    for r in _read(path):
        ref = f"{r['row_id']}::{r['column']}"
        if ref in gold:
            per[r["column"]] += 1
            bad[r["column"]] += ((r.get("value") or "").strip().lower()
                                 != str(gold[ref]).lower().strip())
    L = len(per)
    # delta MUST match the sweeps (bench.run_study --delta, default 0.1). Computing the
    # figure at a different delta would show bounds the certifier never actually used.
    DELTA = 0.10
    bounds = {c: _cp_upper(bad[c], min(per[c], 400), DELTA / L) for c in per}
    cols = sorted(per, key=lambda c: bounds[c])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    ys = list(range(len(cols)))
    vals = [bounds[c] for c in cols]
    n_cert = sum(v <= 0.2 for v in vals)
    ax.barh(ys, vals, height=0.62, edgecolor="black", linewidth=0.4, zorder=2,
            color=["#1b4f72" if v <= 0.2 else "#aab7b8" for v in vals])
    for i, c in enumerate(cols):                       # counts in their own column
        ax.text(1.04, i, f"{per[c]:,}", va="center", ha="left", fontsize=7,
                color="#212f3d")
    ax.text(1.04, -0.95, "cells", va="center", ha="left", fontsize=7,
            color="#566573", style="italic")
    for a, ls, dy, ha in [(0.05, ":", -1.62, "right"), (0.1, "--", -1.62, "left"),
                          (0.2, "-", -0.95, "left")]:
        ax.axvline(a, color="#c0392b", lw=0.9, ls=ls, zorder=4, ymin=0.0, ymax=0.86)
        ax.text(a + (0.012 if ha == "left" else -0.012 if ha == "right" else 0), dy,
                rf"{a}", fontsize=7, color="#c0392b", ha=ha, va="center")
    ax.text(0.125, -2.55, r"budget $\alpha$", fontsize=7, color="#c0392b",
            ha="center", va="center")
    gap_y = n_cert - 0.5                               # between last certified and next
    ax.annotate("", xy=(vals[n_cert - 1], gap_y), xytext=(vals[n_cert], gap_y),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="#212f3d"),
                zorder=6)
    ax.annotate(f"gap: no column has a\nbound between {vals[n_cert - 1]:.3f}\n"
                f"and {vals[n_cert]:.3f}, so every "
                r"$\alpha$"
                "\nin that range certifies\nthe same five columns",
                xy=((vals[n_cert - 1] + vals[n_cert]) / 2, gap_y),
                xytext=(0.42, gap_y - 1.05), fontsize=7, ha="left", va="center",
                color="#212f3d", zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#212f3d"))
    ax.set_yticks(ys); ax.set_yticklabels(cols, fontsize=7)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.invert_yaxis(); ax.set_xlim(0, 1.02); ax.set_ylim(len(cols) - 0.35, -3.0)
    ax.set_xlabel("certified upper bound on column error rate", fontsize=9)
    ax.set_title("(a) certification spectrum, tax / Baran", fontsize=8.5, pad=2)
    _style_axes(ax)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#1b4f72", ec="black", lw=0.4,
                                     label=r"certified at $\alpha=0.2$"),
                       plt.Rectangle((0, 0), 1, 1, fc="#aab7b8", ec="black", lw=0.4,
                                     label="escalated")],
              fontsize=7, frameon=False, loc="upper right",
              bbox_to_anchor=(0.97, 1.0))

    _bmed = _baran_medians()
    for c, rd in CLEANERS:
        st = STYLE[c]
        xs, ys2, mk = [], [], []
        for d in DATASETS:
            if (c, d) in EXCLUDE:
                continue
            med = _bmed.get(d) if c == "Baran" else None
            if med:
                # median over draws, with the min-max range of A(0.2) as a bar
                x, y = float(med["acc"]), float(med["A2"])
                ax2.plot([x, x], [float(med["A2_min"]), float(med["A2_max"])],
                         color=st["color"], lw=1.0, alpha=0.5, zorder=2)
                # name the Baran points: several share x~0.86-0.91 and their range
                # bars overlap; per-dataset offsets keep the labels apart
                dx, dy, ha = _BARAN_LABEL_OFFSET.get(d, (-4, 0, "right"))
                ax2.annotate(d, (x, y), xytext=(dx, dy), textcoords="offset points",
                             fontsize=6.5, color="#566573", ha=ha, va="center",
                             zorder=4)
                xs.append(x); ys2.append(y); mk.append(False)
                continue
            ar = [r for r in acc_rows if r["proposer"] == c and r["dataset"] == d]
            rows = [x for x in care(pareto(rd, d, c)) if abs(x[0] - 0.2) < 1e-9]
            if not ar or not rows:
                continue
            xs.append(float(ar[0]["acc_on_errors"]))
            ys2.append(1 - rows[0][1])
            mk.append(float(ar[0]["acc_on_errors"]) >= 0.9999)
            if c == "Baran":               # single-draw Baran pair: same labelling
                dx, dy, ha = _BARAN_LABEL_OFFSET.get(d, (-4, 0, "right"))
                ax2.annotate(d, (xs[-1], ys2[-1]), xytext=(dx, dy),
                             textcoords="offset points", fontsize=6.5,
                             color="#566573", ha=ha, va="center", zorder=4)
        if not xs:
            continue                       # never leave a legend entry with no points
        for x, y, deg in zip(xs, ys2, mk):
            ax2.plot(x, y, marker=st["marker"], ms=5.5, ls="none",
                     color="white" if deg else st["color"],
                     markeredgecolor=st["color"], markeredgewidth=1.2, zorder=3)
        ax2.plot([], [], marker=st["marker"], ls="none", color=st["color"], ms=5,
                 label=c)
    ax2.plot([0, 1], [0, 1], ls="--", lw=0.9, color="#909497", zorder=1)
    # label the diagonal in the empty band above the abstaining cluster, not through
    # the Baran points / legend
    ax2.text(0.28, 0.34, "automation = accuracy", fontsize=7, color="#909497",
             rotation=41, ha="center", va="center")
    ax2.plot([], [], marker="o", ls="none", mfc="white", markeredgecolor="grey",
             ms=5, label="exact proposer (hollow)")
    ax2.set_xlabel("proposer accuracy on error cells", fontsize=9)
    ax2.set_ylabel(r"automation $A(0.2)$", fontsize=9)
    ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("(b) response to proposer quality", fontsize=8.5)
    _style_axes(ax2)
    # two-column legend keeps the box in the upper-left corner (x<0.6, y>0.75), a
    # region no point can occupy (A(0.2) <= accuracy there), clear of the diagonal
    _h, _l = ax2.get_legend_handles_labels()
    fig.legend(_h, _l, fontsize=7, frameon=False, loc="lower center", ncol=7,
               columnspacing=1.2, handletextpad=0.3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT_FIG}/fig5_mechanism.pdf", bbox_inches="tight")
    print("  fig5_mechanism.pdf")


# ------------------------------------------------------------------ tables

def tab1():
    """Master table. Each CARE number sits next to the proposer statistic that makes it
    interpretable, so a reader never has to guess why a value is 0 or 1."""
    _BARAN_MED = _baran_medians()
    audit = {(r["proposer"], r["dataset"]): r
             for r in _read("experiments/proposer_accuracy.csv")}
    L = [r"\footnotesize", r"\setlength{\tabcolsep}{4.5pt}",
         r"\renewcommand{\arraystretch}{0.92}",
         r"\begin{tabular}{llrrrrrrrl}", r"\toprule",
         r"& & \multicolumn{4}{c}{proposer (no \textsc{Care})} & "
         r"\multicolumn{4}{c}{with \textsc{Care}} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"Proposer & Dataset & $n$ & acc. & $\pi_0$ & harm & $A(0.05)$ & $A(0.1)$ & $A(0.2)$ "
         r"& $R(0.2)$ \\", r"\midrule"]
    for c, rd in CLEANERS:
        for d in DATASETS:
            a = audit.get((c, d))
            rows = care(pareto(rd, d, c))
            if not a:
                continue
            acc = float(a["acc_on_errors"])
            # "harm": share of ALREADY-CORRECT cells the proposer rewrote. Invisible to
            # repair-F1, which scores only error cells, and the dominant deployment risk
            # for proposers that run without a detector.
            _ct = int(a.get("clean_touched") or 0)
            _cb = int(a.get("clean_broken") or 0)
            # Only quote a rate when the denominator can support one. BClean touches
            # 17 clean cells on beers and 156 on flights; "1.00" there would sit beside
            # Jellyfish's 0.40-over-359,372 and invite a comparison the data cannot bear.
            harm = (f"${_cb/_ct:.2f}$" if _ct >= 500 else r"\textemdash")
            med = _BARAN_MED.get(d) if c == "Baran" else None
            if med:
                acc = float(med["acc"])
                a = dict(a, frac_in_perfect_cols=med["pi0"])
                n = int(med["n"])
            else:
                n = int(a["n_proposed_on_errors"])
            nstr = f"{n:,}".replace(",", "{,}")
            if (c, d) in TAB1_EXCLUDE or not rows:
                if (c, d) == ("HoloClean", "beers"):
                    harm = r"\textemdash"
                L.append(f"{c} & \\textsc{{{d}}} & {nstr} & {acc:.3f} & "
                         f"{float(a['frac_in_perfect_cols']):.3f} & {harm} & "
                         r"\multicolumn{4}{c}{\emph{excluded}} \\")
                continue
            by = {round(x[0], 3): x for x in rows}
            cells = []
            for al in (0.05, 0.1, 0.2):
                if med:
                    v = float(med["A" + str(al).replace("0.", "")])
                    cells.append(r"$0$" if v <= 1 - EMPTY else f"${v:.2f}$")
                    continue
                x = by.get(al)
                cells.append("--" if x is None else
                             (r"$0$" if x[1] >= EMPTY else f"${1 - x[1]:.2f}$"))
            if med:
                rstr = (r"$\varnothing$" if float(med["A2"]) <= 1 - EMPTY
                        else f"${float(med['R02']):.3f}$")
            else:
                x2 = by.get(0.2)
                if x2 is None:
                    rstr = "--"
                elif x2[1] >= EMPTY:
                    rstr = r"$\varnothing$"
                else:
                    rstr = f"${x2[2]:.3f}$"
            mark = r"$^{\dagger}$" if acc >= 0.9999 else ""
            # A refusal at every tested budget is one verdict, not three measurements
            # of zero, so it is printed once, in words, like "excluded" above. A row
            # that certifies something at any budget keeps its per-budget numbers.
            if all(x == r"$0$" for x in cells):
                tail = r"\multicolumn{4}{c}{\emph{refused at every $\alpha$}}"
            else:
                tail = " & ".join(cells) + f" & {rstr}"
            L.append(f"{c} & \\textsc{{{d}}}{mark} & {nstr} & {acc:.3f} & "
                     f"{float(a['frac_in_perfect_cols']):.3f} & {harm} & "
                     + tail + " \\\\")
        L.append(r"\addlinespace[1.5pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab1_audit.tex", "w").write("\n".join(L) + "\n")
    print("  tab1_audit.tex")


def tab2():
    L = [r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\renewcommand{\arraystretch}{0.9}",
         r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Proposer / data & $\alpha$ & $\varepsilon_S$ & pred. & obs."
         r" & bound \\", r"\midrule"]
    for lab, a, base, pts in _poison_series():
        prop, ds = [s.strip() for s in lab.split("/")]
        first = True
        for eps, obs, _lo, _hi in pts:
            if eps < 1e-9:
                continue
            pred = eps + (1 - eps) * base
            name = f"{prop} / \\textsc{{{ds}}}" if first else ""
            L.append(f"{name} & {a if first else ''} & {eps:.3f} & {pred:.3f} & "
                     f"{obs:.3f} & {a + eps * (1 - a):.3f} \\\\")
            first = False
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab2_theorem2.tex", "w").write("\n".join(L) + "\n")
    print("  tab2_theorem2.tex")


def tab3():
    """Oracle vs real (Raha) detection, per proposer, from logs that can bear the
    comparison: a full-coverage log (Jellyfish proposes on every cell; BClean and
    HoloClean via their keep-logs) or a re-run under the detector's own queue
    (Baran, ``baran_raha``). A change-only oracle log has no row for a detector
    false positive, so it is never re-sliced onto a detector queue here."""
    # (proposer, oracle dir, non-oracle dir, marker). Rows appear only when BOTH
    # sweeps exist, so the table grows as runs land without ever showing a
    # half-populated comparison. The marker flags a proposer whose
    # rows here are NOT the same measurement as Table 1's. BClean is the only such case:
    # Table 1 uses the change-only log it natively emits, scored on error cells, while
    # this table needs the full-coverage keep-log scored on every proposed cell (a
    # change-only log has no row for a detector false positive). Both are correct; they
    # answer different questions. Within a row, oracle and non-oracle share one log and
    # one scoring, so that comparison is exact.
    PAIRS = [("Jellyfish", "results_detected", "results_raha_jelly", ""),
             ("Baran", "results_baran_detected", "results_raha_baran", ""),
             ("BClean", "results_bclean_keep_oracle", "results_raha_bclean",
              r"$^{\ddagger}$"),
             ("HoloClean", "results_holoclean_keep_oracle", "results_raha_holoclean",
              r"$^{\ddagger}$")]
    # Proposer is a spanning row header rather than a column: it keeps the table
    # inside one narrow column as more proposers are added.
    L = [r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\renewcommand{\arraystretch}{0.9}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{detector} & \multicolumn{2}{c}{oracle}"
         r" & \multicolumn{2}{c}{Raha} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Dataset & prec. & rec. & $A(0.2)$ & $R(0.2)$ & $A(0.2)$ "
         r"& $R(0.2)$ \\", r"\midrule"]
    dq = {r["dataset"]: r for r in _read("experiments/detector_quality.csv")}

    def fmt(x):
        # An empty auto-applied set is a refusal; print the verdict once across the
        # A/R pair instead of a 0 beside an empty-set symbol.
        _a, h, e, _hi, _c = x
        return (r"\multicolumn{2}{c}{\emph{refused}}", None) if h >= EMPTY else \
               (f"${1-h:.2f}$", f"${e:.3f}$")

    any_row = False
    for prop, odir, rdir, mark in PAIRS:
        wrote = False
        for d in DATASETS:
            o = [x for x in care(pareto(odir, d), "best") if abs(x[0] - 0.2) < 1e-9]
            r = [x for x in care(pareto(rdir, d), "best") if abs(x[0] - 0.2) < 1e-9]
            if not o or not r:
                continue
            q = dq.get(d, {})
            oa, oe = fmt(o[0]); ra, re_ = fmt(r[0])
            if not wrote:
                L.append(r"\multicolumn{7}{@{}l}{\emph{" + prop + r"}" + mark + r"} \\")
            ocell = oa if oe is None else f"{oa} & {oe}"
            rcell = ra if re_ is None else f"{ra} & {re_}"
            L.append(f"\\quad\\textsc{{{d}}} & {float(q.get('precision', 0)):.2f} & "
                     f"{float(q.get('recall', 0)):.2f} & {ocell} & {rcell} \\\\")
            wrote = any_row = True
        if wrote:
            L.append(r"\addlinespace[2pt]")
    if not any_row:
        print("  tab3: no oracle/non-oracle pair available"); return
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab3_nonoracle.tex", "w").write("\n".join(L) + "\n")
    print("  tab3_nonoracle.tex")


def main():
    os.makedirs(OUT_FIG, exist_ok=True); os.makedirs(OUT_TAB, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif", "axes.linewidth": 0.6,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        })
    except ImportError:
        print("matplotlib missing -> tables only"); plt = None
    print("generating ->")
    if plt:
        fig2(plt); fig5(plt); fig7(plt); fig8(plt); fig9(plt)
    tab1(); tab2(); tab3(); tab4(); tab6()
    print("done -> paper/figures/*.pdf, paper/tables/*.tex")


if __name__ == "__main__":
    main()
