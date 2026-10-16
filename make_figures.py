"""Build the paper's figures and tables from the result CSVs in experiments/ and csvs/.

    python make_figures.py  -> paper/figures, paper/tables

Hollow markers denote runs whose auto-applied set is empty.
"""
from __future__ import annotations

import csv
import os

OUT_FIG, OUT_TAB, EXP = "paper/figures", "paper/tables", "experiments"

# one style per cleaner, shared by all figures
STYLE = {
    "Baran":       dict(color="#1b4f72", marker="o", ls="-"),
    "BClean":      dict(color="#7d6608", marker="s", ls="--"),
    "BClean+":     dict(color="#b9770e", marker="X", ls="-."),
    "GIDCL":       dict(color="#78281f", marker="D", ls=":"),
    "HoloClean":   dict(color="#117864", marker="P", ls=(0, (1, 1))),
    "ZeroEC":      dict(color="#6c3483", marker="v", ls=(0, (3, 1, 1, 1))),
}
# GIDCL is scored over its own detector's queue (--scoring detected).
CLEANERS = [("Baran", "results_baran_errors"), ("BClean", "results_bclean"),
            ("BClean+", "results_bclean_plus"), ("GIDCL", "results_gidcl_owndet"),
            ("HoloClean", "results_holoclean"), ("ZeroEC", "results_zeroec")]
DATASETS = ["hospital", "beers", "flights", "rayyan", "tax"]
# BClean+ runs only on hospital, beers and flights, so it needs no entry here.
EXCLUDE = {("Baran", "flights"), ("BClean", "rayyan"), ("BClean", "tax")}
# HoloClean/beers: most changes are null imputations on unlabelled cells; Table 1 only.
TAB1_EXCLUDE = EXCLUDE | {("HoloClean", "beers")}

# per-(cleaner, dataset) results directory overrides
RESULT_OVERRIDE = {}
EMPTY = 0.999          # human_cost at/above this  <=>  auto-applied set is empty


def _read(p):
    return list(csv.DictReader(open(p, newline=""))) if os.path.exists(p) else []


def _baran_medians():
    """Baran's per-dataset medians over unseeded draws (from audit_baran_median.py)."""
    return {r["dataset"]: r for r in _read(f"{EXP}/baran_median.csv")}


# one result file per independent Baran draw
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
    """Result rows for one (cleaner, dataset), honouring RESULT_OVERRIDE."""
    rd = RESULT_OVERRIDE.get((cleaner, ds), rd) if cleaner else rd
    return _read(os.path.join(EXP, rd, f"{ds}_pareto.csv"))


def care(rows, strata="mondrian"):
    """CARE rows as [(alpha, human_cost, err, err_hi, covered)] sorted by alpha.

    ``strata="best"`` uses column x source strata when present, else column-only.
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
    """Realised error of the apply-all baseline, or None."""
    for r in rows:
        if r.get("baseline") == "ApplyAll":
            return float(r["realized_error"])
    return None


def _style_axes(ax, fs=9):
    ax.grid(alpha=0.25, lw=0.5, ls=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=fs - 1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ figures

CAL_FRAC = 0.4        # calibration split (bench.study)
RECERT_K = 30         # audit labels per written stratum per batch


def _batch_cost(c, d, h):
    """Human cost per repair: calibration labels + re-certification audit + escalations.

    ``h`` is the escalated fraction of the non-calibration part.
    """
    import csv as _csv
    n = strata = None
    for r in _read("experiments/proposer_accuracy.csv"):
        if r["proposer"] == c and r["dataset"] == d:
            n = int(r["n_proposed_on_errors"])
    if os.path.exists("experiments/procedures.csv"):
        for r in _read("experiments/procedures.csv"):
            if (r["cleaner"] == c and r["dataset"] == d and r["procedure"] == "bonferroni_cp"
                    and abs(float(r["alpha"]) - 0.2) < 1e-9):
                strata = int(r["written_strata"]) / max(1, int(r["seeds"]))
    recert = (RECERT_K * strata / n) if (n and strata is not None) else 0.0
    return min(1.0, CAL_FRAC + (1 - CAL_FRAC) * h + recert)


def fig2(plt):
    """fig2_pareto.pdf: human cost per batch against realised error at alpha=0.2."""
    fig, ax = plt.subplots(figsize=(3.34, 2.35))
    import statistics as _st
    for c, rd in CLEANERS:
        st = STYLE[c]
        xs, ys, es, hollow = [], [], [], []
        for d in DATASETS:
            if (c, d) in EXCLUDE:
                continue
            draws = _baran_draw_rows(d) if c == "Baran" else []
            if len(draws) >= 2:
                # each draw small and faint; the median full-size
                pts = []
                for rows in draws:
                    for a, h, e, ehi, _cov in care(rows):
                        if abs(a - 0.2) < 1e-9:
                            pts.append((h, e))
                pts = [(_batch_cost(c, d, h), e) for h, e in pts]
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
                xs.append(_batch_cost(c, d, h)); ys.append(e); es.append(max(0.0, ehi - e))
                hollow.append(h >= EMPTY)
        if not xs:
            continue
        for x, y, er, ho in zip(xs, ys, es, hollow):
            ax.errorbar(x, y, yerr=[[0], [er]], fmt=st["marker"], ms=5.5, lw=0,
                        elinewidth=0.9, capsize=2, ecolor=st["color"],
                        color="white" if ho else st["color"],
                        markeredgecolor=st["color"], markeredgewidth=1.1, zorder=3)
        ax.plot([], [], marker=st["marker"], ls="none", color=st["color"],
                ms=5, label=f"CARE + {c}")
    aa = [applyall(pareto(rd, d, c)) for c, rd in CLEANERS for d in DATASETS
          if (c, d) not in EXCLUDE and applyall(pareto(rd, d, c)) is not None]
    YMAX = 0.32                                   # apply-all points above it are clipped
    if aa:
        inside = [v for v in aa if v <= YMAX - 0.02]
        above = sorted(v for v in aa if v > YMAX - 0.02)
        ax.plot([0.0] * len(inside), inside, "x", c="#909497", ms=5.5, mew=1.2,
                label="ungoverned apply-all", zorder=2)
        if above:
            ax.plot([0.0] * len(above), [YMAX - 0.012] * len(above), "x", c="#909497",
                    ms=5.5, mew=1.2, zorder=2)
            ax.text(0.03, YMAX - 0.012, f"{len(above)} more, up to {max(above):.2f}",
                    fontsize=6.5, color="#566573", va="center")
    ax.axhline(0.2, ls="--", c="#c0392b", lw=1, zorder=1)
    ax.text(0.09, 0.207, r"budget $\alpha=0.2$", fontsize=7.5, color="#c0392b")
    ax.axvspan(0.985, 1.02, color="#eaeded", zorder=0)
    ax.text(0.975, 0.30, "nothing\napplied", fontsize=7, ha="right", va="top",
            color="#566573")
    ax.axvline(CAL_FRAC, ls=":", c="#566573", lw=0.8, zorder=1)
    ax.text(CAL_FRAC + 0.015, 0.30, "40% calibration\nsplit", fontsize=7,
            color="#566573", ha="left", va="top")
    ax.set_xlabel("human cost per batch (labels + escalations) / repairs", fontsize=8.5)
    ax.set_ylabel("error on applied repairs", fontsize=9)
    ax.set_xlim(-0.04, 1.02); ax.set_ylim(-0.015, YMAX)
    _style_axes(ax)
    fig.legend(fontsize=7, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.54, -0.03), columnspacing=0.9, handletextpad=0.3,
               labelspacing=0.3)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(f"{OUT_FIG}/fig2_pareto.pdf", bbox_inches="tight")
    print("  fig2_pareto.pdf")


def _poison_series():
    """Trusted-poison sweeps for Table 2: [(label, alpha, base error, points)]."""
    out = []
    # configurations listed explicitly so a run with an empty applied set cannot join
    for lab, path, a in [
            ("Baran / beers", f"{EXP}/results_poison_baran_beers/beers_trustedpoison_poison.csv", 0.2),
            ("Baran / rayyan", f"{EXP}/results_poison_baran_rayyan/rayyan_trustedpoison_poison.csv", 0.2),
            ("BClean / flights", f"{EXP}/results_poison_bclean_flights/flights_trustedpoison_poison.csv", 0.2),
            ("Baran / hospital", f"{EXP}/results_poison_baran_hospital/hospital_trustedpoison_poison.csv", 0.2),
            ("BClean / hospital", f"{EXP}/results_poison_bclean_hospital/hospital_trustedpoison_poison.csv", 0.2)]:
        rs = [r for r in _read(path) if r.get("baseline") == "CARE"]
        if not rs:
            print(f"  warning: tab2: no CARE rows in {path} -- run the poison sweeps in reproduce.sh")
            continue
        pts = sorted((float(r["applied_contamination"]), float(r["realized_error"]),
                      float(r.get("err_lo") or 0), float(r.get("err_hi") or 0)) for r in rs)
        base = next((e for eps, e, _l, _h in pts if eps < 1e-9), 0.0)
        out.append((lab, a, base, pts))
    return out


PROPOSER_LABEL = {"baran": "Baran", "bclean": "BClean", "bclean_plus": "BClean+",
                  "gidcl": "GIDCL", "holoclean": "HoloClean", "zeroec": "ZeroEC"}


def fig9(plt, alpha=0.2, partition="mondrian"):
    """fig9_error_rate.pdf: A(alpha), accuracy and ceiling pi_alpha against measured cell error rate."""
    # prefer medians over Baran draws when available
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
    ncol = len(bases)
    nrow = 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.5),
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
                    ax.plot(x, [float(r["acc"]) for r in inner], ls=":", lw=1.0,
                            color="#909497", marker="s", ms=2.8, zorder=1)
                ax.plot(x, c, ls="--", lw=1.2, color="#5dade2", zorder=2)
                # min-max range over draws
                if all("A_min" in r for r in inner):
                    for r, xi in zip(inner, x):
                        lo, hi = float(r["A_min"]), float(r["A_max"])
                        if hi - lo > 1e-9:
                            ax.plot([xi, xi], [lo, hi], color=st["color"], lw=0.7,
                                    alpha=0.45, zorder=2, solid_capstyle="butt")
                ax.plot(x, a, ls=st["ls"], lw=1.2, color=st["color"], zorder=3)
                # hollow = certified nothing
                for xi, ai in zip(x, a):
                    ax.plot([xi], [ai], marker=st["marker"], ms=3.8, ls="none",
                            color=st["color"], zorder=4,
                            mfc=("none" if ai <= 1e-9 else st["color"]),
                            mew=0.8)
                seen[name] = st
            for r in outer:
                ax.plot([float(r["cell_error_rate"])], [float(r["A"])], marker="^",
                        ms=3.4, ls="none", color=st["color"], mfc="none", mew=0.9,
                        zorder=5)
        ax.set_title(base, fontsize=9, pad=2)
        ax.set_xlim(0, max(float(r["cell_error_rate"]) for r in rows) * 1.05)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        _style_axes(ax, fs=9)
        if i % ncol == 0:
            ax.set_ylabel("fraction", fontsize=9)
        if i // ncol == nrow - 1:
            ax.set_xlabel("cell error rate", fontsize=9)
    for j in range(len(bases), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if seen:
        handles = [plt.Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"],
                              ms=3, lw=1.0, label=rf"$A({alpha:g})$, {n}")
                   for n, s in sorted(seen.items())]
        handles.append(plt.Line2D([], [], color="#909497", ls=":", lw=0.9, marker="s",
                                  ms=2.2, label="accuracy"))
        handles.append(plt.Line2D([], [], color="#5dade2", ls="--", lw=1.2,
                                  label=r"ceiling $\pi_\alpha$ (sealed before calibration)"))
        fig.legend(handles=handles, fontsize=8, frameon=False, ncol=3,
                   loc="lower center", bbox_to_anchor=(0.5, -0.1),
                   handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig9_error_rate.pdf", bbox_inches="tight")
    print(f"  fig9_error_rate.pdf ({len(rows)} points, {len(bases)} tables, "
          f"alpha={alpha}, {partition}, "
          f"{'median of %d draws' % draws if draws > 1 else 'SINGLE DRAW'})")


def tab6():
    """tab6_sweep.tex: every swept variant with error rate, accuracy, ceiling, A and coverage."""
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
    """fig8_heldout.pdf: violation rate of a held-out threshold vs CARE against label budget."""
    rows = _read(f"{EXP}/heldout_budget.csv")
    if not rows:
        print("  fig8: skipped (run audit_heldout_budget.py first)"); return
    COL = {("Baran", "beers"): "#7d6608", ("BClean", "flights"): "#78281f",
           ("BClean+", "beers"): "#1b4f72"}
    fig, ax = plt.subplots(figsize=(3.34, 1.42))
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
    ax.legend(handles, labels, fontsize=6.5, frameon=False, loc="upper right",
              handlelength=2.2, borderaxespad=0.2, labelspacing=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/fig8_heldout.pdf", bbox_inches="tight")
    print("  fig8_heldout.pdf")


def fig7(plt):
    """fig7_label_budget.pdf: certifiable automation A(0.2) against labelled calibration cells."""
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
    """tab4_calibration.tex: mean confidence, accuracy and ECE per cleaner and dataset."""
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
    """One-sided Clopper-Pearson upper bound for k/n, by bisection on the binomial CDF."""
    import math
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(120):
        m = (lo + hi) / 2
        s = sum(math.comb(n, i) * m ** i * (1 - m) ** (n - i) for i in range(k + 1))
        lo, hi = (m, hi) if s > delta else (lo, m)
    return hi


def fig5(plt):
    """fig5_mechanism.pdf: (a) per-column error bounds on tax/Baran, (b) A(0.2) against accuracy per pair."""
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
    # same delta as the sweeps (bench.run_study --delta)
    DELTA = 0.10
    bounds = {c: _cp_upper(bad[c], min(per[c], 400), DELTA / L) for c in per}
    cols = sorted(per, key=lambda c: bounds[c])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    fig.subplots_adjust(left=0.135, right=0.995, bottom=0.25, top=0.935, wspace=0.58)

    # (a) one bar per column, repair counts in the tick labels
    ys = list(range(len(cols)))
    vals = [bounds[c] for c in cols]
    n_cert = sum(v <= 0.2 for v in vals)
    ax.barh(ys, vals, height=0.62, edgecolor="black", linewidth=0.4, zorder=2,
            color=["#1b4f72" if v <= 0.2 else "#aab7b8" for v in vals])
    for a, ls, dy, ha in [(0.05, ":", -1.55, "right"), (0.1, "--", -1.55, "left"),
                          (0.2, "-", -0.85, "left")]:
        ax.axvline(a, color="#c0392b", lw=0.9, ls=ls, zorder=4, ymin=0.0, ymax=0.87)
        ax.text(a + (0.012 if ha == "left" else -0.012), dy, rf"{a}", fontsize=7.5,
                color="#c0392b", ha=ha, va="center")
    ax.text(0.125, -2.4, r"budget $\alpha$", fontsize=7.5, color="#c0392b",
            ha="center", va="center")
    gap_y = n_cert - 0.5                               # between last certified and next
    ax.annotate("", xy=(vals[n_cert - 1], gap_y), xytext=(vals[n_cert], gap_y),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="#212f3d"), zorder=6)
    ax.annotate(f"no bound in ({vals[n_cert - 1]:.3f}, {vals[n_cert]:.3f})",
                xy=((vals[n_cert - 1] + vals[n_cert]) / 2, gap_y),
                xytext=(0.42, gap_y - 1.3), fontsize=7.5, ha="left", va="center",
                color="#212f3d", zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#212f3d"))
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c}  ({per[c]:,})" for c in cols], fontsize=7.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.invert_yaxis(); ax.set_xlim(0, 1.0); ax.set_ylim(len(cols) - 0.35, -2.9)
    ax.set_xlabel("certified upper bound on column error rate", fontsize=9)
    ax.set_title("(a) per-column bounds, Baran on tax (repairs in parentheses)",
                 fontsize=8.5, pad=3)
    _style_axes(ax)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#1b4f72", ec="black", lw=0.4,
                                     label=r"certified at $\alpha=0.2$"),
                       plt.Rectangle((0, 0), 1, 1, fc="#aab7b8", ec="black", lw=0.4,
                                     label="escalated")],
              fontsize=7.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))

    # (b) one row per pair, sorted by accuracy: accuracy (grey) against A(0.2) (colour)
    _bmed = _baran_medians()
    rows_b = []                                        # (label, cleaner, acc, A, lo, hi, exact)
    for c, rd in CLEANERS:
        for d in DATASETS:
            if (c, d) in TAB1_EXCLUDE:
                continue
            med = _bmed.get(d) if c == "Baran" else None
            if med:
                rows_b.append((f"{c} / {d}", c, float(med["acc"]), float(med["A2"]),
                               float(med["A2_min"]), float(med["A2_max"]), False))
                continue
            ar = [r for r in acc_rows if r["proposer"] == c and r["dataset"] == d]
            rows = [x for x in care(pareto(rd, d, c)) if abs(x[0] - 0.2) < 1e-9]
            if not ar or not rows:
                continue
            acc = float(ar[0]["acc_on_errors"])
            rows_b.append((f"{c} / {d}", c, acc, 1 - rows[0][1], None, None, acc >= 0.9999))
    rows_b.sort(key=lambda t: -t[2])
    yy = list(range(len(rows_b)))
    for y, (lab, c, acc, A, lo, hi, exact) in zip(yy, rows_b):
        st = STYLE[c]
        ax2.plot([A, acc], [y, y], color="#b0b7bb", lw=1.0, zorder=1)
        if lo is not None and hi - lo > 1e-9:
            ax2.plot([lo, hi], [y, y], color=st["color"], lw=3.0, alpha=0.25, zorder=2,
                     solid_capstyle="butt")
        ax2.plot(acc, y, marker="o", ms=3.6, ls="none", color="#7b7d7d", zorder=3)
        ax2.plot(A, y, marker=st["marker"], ms=5.5, ls="none",
                 color="white" if exact else st["color"],
                 markeredgecolor=st["color"], markeredgewidth=1.2, zorder=4)
    ax2.set_yticks(yy); ax2.set_yticklabels([t[0] for t in rows_b], fontsize=6.2)
    ax2.invert_yaxis(); ax2.set_ylim(len(rows_b) - 0.45, -0.55)
    ax2.set_xlim(-0.03, 1.03); ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_xlabel(r"accuracy on error cells ($\bullet$) and automation $A(0.2)$",
                   fontsize=9)
    ax2.set_title("(b) every scored pair, sorted by accuracy", fontsize=8.5, pad=2)
    _style_axes(ax2)
    ax2.plot([], [], marker="o", ls="none", color="#7b7d7d", ms=3.6, label="accuracy")
    for c, _rd in CLEANERS:
        st = STYLE[c]
        ax2.plot([], [], marker=st["marker"], ls="none", color=st["color"], ms=5,
                 label=f"$A$, {c}")
    ax2.plot([], [], marker="o", ls="none", mfc="white", markeredgecolor="grey", ms=5,
             label="exact cleaner (hollow)")
    ax2.plot([], [], color="#1b4f72", lw=3.0, alpha=0.25, label="Baran min\u2013max over draws")
    _h, _l = ax2.get_legend_handles_labels()
    fig.legend(_h, _l, fontsize=7, frameon=False, loc="upper center", ncol=5,
               columnspacing=1.0, handletextpad=0.3, bbox_to_anchor=(0.5, 0.105))
    fig.savefig(f"{OUT_FIG}/fig5_mechanism.pdf", bbox_inches="tight")
    print("  fig5_mechanism.pdf")


def tab1():
    """tab1_audit.tex: cleaner statistics alongside CARE's A(alpha) and R(0.2)."""
    _BARAN_MED = _baran_medians()
    audit = {(r["proposer"], r["dataset"]): r
             for r in _read("experiments/proposer_accuracy.csv")}
    fmt_only = {(r["cleaner"], r["dataset"]): int(r["format_only"])
                for r in _read("experiments/overwrite_formats.csv")}
    L = [r"\footnotesize", r"\setlength{\tabcolsep}{4.5pt}",
         r"\renewcommand{\arraystretch}{0.9}",
         r"\begin{tabular}{llrrrrrrrl}", r"\toprule",
         r"& & \multicolumn{4}{c}{cleaner alone (no \textsc{Care})} & "
         r"\multicolumn{4}{c}{with \textsc{Care}} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"Cleaner & Dataset & $n$ & acc. & $\pi_0$ & overwr. & $A(0.05)$ & $A(0.1)$ & $A(0.2)$ "
         r"& $R(0.2)$ \\", r"\midrule"]
    for c, rd in CLEANERS:
        for d in DATASETS:
            a = audit.get((c, d))
            rows = care(pareto(rd, d, c))
            if not a:
                continue
            acc = float(a["acc_on_errors"])
            # "overwr.": number of already-correct cells the cleaner rewrote
            harm = f"{int(a.get('clean_broken') or 0):,}".replace(",", "{,}")
            if fmt_only.get((c, d)):   # format-only rewrites in parentheses
                harm += f"\\,({fmt_only[(c, d)]})"
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
            # refusal at every alpha is printed once as text
            if all(x == r"$0$" for x in cells):
                tail = r"\multicolumn{4}{c}{\emph{refused at every $\alpha$}}"
            else:
                tail = " & ".join(cells) + f" & {rstr}"
            L.append(f"{c} & \\textsc{{{d}}}{mark} & {nstr} & {acc:.3f} & "
                     f"{float(a['frac_in_perfect_cols']):.3f} & {harm} & "
                     + tail + " \\\\")
        L.append(r"\addlinespace[1.5pt]")
    while L and L[-1].startswith(r"\addlinespace"):   # a cleaner with no rows yet
        L.pop()
    L.append(r"\addlinespace[1.5pt]")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab1_audit.tex", "w").write("\n".join(L) + "\n")
    print("  tab1_audit.tex")


def tab2():
    """tab2_poison.tex: predicted vs observed error under trusted poison."""
    L =[r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\renewcommand{\arraystretch}{0.9}",
         r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Cleaner / data & $\alpha$ & $\varepsilon_S$ & pred. & obs."
         r" & bound \\", r"\midrule"]
    for lab, a, base, pts in _poison_series():   # highest contamination level per configuration
        prop, ds = [s.strip() for s in lab.split("/")]
        eps, obs, _lo, _hi = max(pts, key=lambda p: p[0])
        pred = eps + (1 - eps) * base
        L.append(f"{prop} / \\textsc{{{ds}}} & {a} & {eps:.3f} & {pred:.3f} & "
                 f"{obs:.3f} & {a + eps * (1 - a):.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab2_poison.tex", "w").write("\n".join(L) + "\n")
    print("  tab2_poison.tex")


def tab3():
    """tab3_nonoracle.tex: A(0.2) and R(0.2) under oracle vs real (Raha or own) detection."""
    # (cleaner, oracle dir, real-detector dir, footnote marker); rows need both sweeps
    PAIRS = [("Baran", "results_baran_detected", "results_raha_baran", ""),
             ("BClean", "results_bclean_keep_oracle", "results_raha_bclean",
              r"$^{\ddagger}$"),
             ("HoloClean", "results_holoclean_keep_oracle", "results_raha_holoclean",
              r"$^{\ddagger}$"),
             # GIDCL: oracle queue vs its own detector's queue
             ("GIDCL", "results_gidcl", "results_gidcl_owndet", r"$^{\S}$")]
    # cleaner as a spanning row header keeps the table one column wide
    L = [r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\renewcommand{\arraystretch}{0.9}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{detector} & \multicolumn{2}{c}{oracle}"
         r" & \multicolumn{2}{c}{real detector} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Dataset & prec. & rec. & $A(0.2)$ & $R(0.2)$ & $A(0.2)$ "
         r"& $R(0.2)$ \\", r"\midrule"]
    dq = {r["dataset"]: r for r in _read("experiments/detector_quality.csv")}
    dq_own = _own_detector_quality("gidcl")

    def fmt(x):
        # empty applied set: one "refused" cell spanning A and R
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
            q = dq_own.get(d, {}) if prop == "GIDCL" else dq.get(d, {})
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


def _own_detector_quality(tool):
    """Precision/recall of a cleaner's own detector (csvs/<tool>_<ds>_detected.csv)."""
    out = {}
    try:
        from bench.datasets import load
    except Exception:
        return out
    for d in DATASETS:
        p = f"csvs/{tool}_{d}_detected.csv"
        if not os.path.exists(p):
            continue
        _art, defects, _ = load(d)
        gold = set(defects.gold)
        det = {f"{r['row_id']}::{r['column']}" for r in _read(p)}
        tp = len(det & gold)
        out[d] = {"precision": tp / len(det) if det else 0.0,
                  "recall": tp / len(gold) if gold else 0.0}
    return out


def tab5():
    """tab5_error_drop.tex: error drop rate of apply-all vs CARE at alpha=0.2 (oracle detection)."""
    rows = _read(f"{EXP}/error_drop.csv")
    if not rows:
        print("  tab5: skipped (run audit_error_drop.py first)"); return
    sel = [r for r in rows if abs(float(r["alpha"]) - 0.2) < 1e-9
           and r["detection"] == "oracle"]

    def f(x):
        return "---" if x in ("", None) else f"${float(x):+.2f}$"
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Proposer & Dataset & errors & EDR apply-all & EDR \textsc{Care} & "
         r"changes/fixes/breaks \\", r"\midrule"]
    for r in sel:
        L.append(f"{r['proposer']} & \\textsc{{{r['dataset']}}} & "
                 f"{float(r['errors_before']):.0f} & {f(r['edr_apply_all'])} & "
                 f"{f(r['edr_care'])} & {float(r['n_changes']):.0f}/"
                 f"{float(r['fixes']):.0f}/{float(r['breaks']):.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab5_error_drop.tex", "w").write("\n".join(L) + "\n")
    print(f"  tab5_error_drop.tex ({len(sel)} rows)")


def tab8():
    """tab8_gidcl_confidence.tex: AUROC of GIDCL's repair-selection score per seed."""
    rows = _read(f"{EXP}/gidcl_confidence.csv")
    if not rows:
        print("  tab8: skipped (run audit_gidcl_confidence.py first)"); return
    seeds = sorted({int(r["seed"]) for r in rows})
    by = {(r["dataset"], int(r["seed"])): r for r in rows}
    L = [r"\begin{tabular}{l" + "r" * len(seeds) + "}", r"\toprule",
         "Dataset & " + " & ".join(f"seed {k}" for k in seeds) + r" \\", r"\midrule"]
    for d in DATASETS:
        if not any((d, k) in by for k in seeds):
            continue
        cells = [f"{float(by[(d, k)]['auroc']):.3f}" if (d, k) in by else "--"
                 for k in seeds]
        L.append(f"\\textsc{{{d}}} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab8_gidcl_confidence.tex", "w").write("\n".join(L) + "\n")
    print("  tab8_gidcl_confidence.tex")


def fig10(plt):
    """fig10_planner.pdf: label planner, (a) break-even stratum size, (b) reviews saved vs column count."""
    rows = _read(f"{EXP}/planner_breakeven.csv")
    if not rows:
        print("  fig10: skipped (run tools/audit_planner.py first)"); return
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(3.34, 1.62))
    COL = {0.05: "#78281f", 0.1: "#1b4f72", 0.2: "#7d6608"}
    for a, col in COL.items():
        pts = sorted((float(r["p"]), float(r["value"])) for r in rows
                     if r["panel"] == "a" and r["series"] == "breakeven"
                     and abs(float(r["alpha"]) - a) < 1e-9 and r["value"])
        ax.plot([p / a for p, _ in pts], [v for _, v in pts], marker="o", ms=2.2, lw=1.1,
                color=col, label=rf"$\alpha={a:g}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"error rate $p/\alpha$", fontsize=8)
    ax.set_ylabel("break-even size", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=1.4,
              borderaxespad=0.1)
    _style_axes(ax, fs=8)
    SER = [("per_column_40", "40%/column", "#909497", "--"),
           ("per_column_best_n", r"best $n$/column", "#1b4f72", "-."),
           ("planner", "planner", "#c0392b", "-")]
    for key, lab, col, ls in SER:
        pts = sorted((int(r["x"]), float(r["value"])) for r in rows
                     if r["panel"] == "b" and r["series"] == key)
        bx.plot([x for x, _ in pts], [v for _, v in pts], ls=ls, marker="o", ms=2.2,
                lw=1.1, color=col, label=lab)
    bx.axhline(0, lw=0.6, color="#566573")
    bx.set_xscale("log")
    bx.set_xlabel("columns $W$", fontsize=8)
    bx.set_ylabel("reviews saved / queue", fontsize=8)
    bx.set_ylim(-0.5, 1.55); bx.set_yticks([-0.5, 0, 0.5, 1.0])
    bx.legend(fontsize=6, frameon=False, loc="upper right", bbox_to_anchor=(1.03, 1.02),
              handlelength=1.3, borderaxespad=0.1, labelspacing=0.25)
    for a_, t in ((ax, "(a)"), (bx, "(b)")):
        a_.text(0.98 if a_ is ax else 0.04, 0.04 if a_ is ax else 0.96, t,
                transform=a_.transAxes, fontsize=7, ha="right" if a_ is ax else "left",
                va="bottom" if a_ is ax else "top")
    _style_axes(bx, fs=8)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(f"{OUT_FIG}/fig10_planner.pdf", bbox_inches="tight")
    print("  fig10_planner.pdf")


def tab9():
    """tab9_scaling.tex: index build, memory, verification and calibration cost on the first N rows of tax."""
    rows = _read(f"{EXP}/scaling.csv")
    if not rows:
        print("  tab9: skipped (run tools/audit_scaling.py first)"); return
    L = [r"\begin{tabular}{@{}rr|rrr|r|rrr|rrr@{}}", r"\toprule",
         r"& & \multicolumn{3}{c|}{build (s), FDs} & index & \multicolumn{3}{c|}{verify ($\mu$s), FDs} & \multicolumn{3}{c}{calibrate (ms), strata} \\",
         r"rows & repairs & 1 & 2 & 3 & (MB) & 1 & 2 & 3 & 1 & 10 & 100 \\", r"\midrule"]
    for r in rows:
        n = int(r["rows"])
        L.append(f"{n // 1000}K & {int(r['candidates']):,} & "
                 + " & ".join(f"{float(r[f'fd{k}_build_s']):.1f}" for k in (1, 2, 3)) + " & "
                 + f"{float(r['fd3_index_mb']):.0f} & "
                 + " & ".join(f"{float(r[f'fd{k}_verify_us']):.0f}" for k in (1, 2, 3)) + " & "
                 + " & ".join(f"{1e3 * float(r[f'strata{m}_calibrate_s']):.0f}" for m in (1, 10, 100))
                 + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab9_scaling.tex", "w").write("\n".join(L) + "\n")
    print("  tab9_scaling.tex")


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
        fig2(plt); fig5(plt); fig7(plt); fig8(plt); fig9(plt); fig10(plt)
    tab1(); tab2(); tab3(); tab4(); tab5(); tab6(); tab8(); tab9()
    print("done -> paper/figures/*.pdf, paper/tables/*.tex")


if __name__ == "__main__":
    main()
