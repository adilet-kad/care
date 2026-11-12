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
# HoloClean/beers: most changes are null imputations on unlabelled cells; main table (Table 3) only.
TAB_MAIN_EXCLUDE = EXCLUDE | {("HoloClean", "beers")}

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
TAX_RUN = "csvs/baran_var_tax/2/baran_tax_mapped.csv"   # the Baran run of Figs. 1 and 3(a)
RECERT_K = 30         # audit labels per written stratum per batch


def _batch_cost(c, d, h):
    """Human cost per repair: calibration labels + re-validation audit + escalations.

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


def _batch_cost_parts(c, d, h):
    """(checked sample, audit, escalations) shares of ``_batch_cost``, clipped to sum <= 1."""
    total = _batch_cost(c, d, h)
    esc = (1 - CAL_FRAC) * h
    audit = max(0.0, total - CAL_FRAC - esc)
    return CAL_FRAC, audit, esc


def fig_budget(plt):
    """fig_budget.pdf: per pair, error of the written set at alpha=0.2 against apply-all (top)
    and the share CARE writes at alpha=0.05 and 0.2 (bottom)."""
    import statistics as _st
    from matplotlib.patches import Patch

    def at(rows, alpha):
        for a, h, e, ehi, _cov in care(rows):
            if abs(a - alpha) < 1e-9:
                return h, e, ehi
        return None

    pts = {}   # (c, d) -> dict(aa, e, ehi, refused, a05, a2, er)
    for c, rd in CLEANERS:
        for d in DATASETS:
            if (c, d) in EXCLUDE:
                continue
            rows_all = pareto(rd, d, c)
            aa = applyall(rows_all)
            draws = _baran_draw_rows(d) if c == "Baran" else []
            if len(draws) >= 2:
                x2 = [at(r, 0.2) for r in draws]; x5 = [at(r, 0.05) for r in draws]
                x2 = [x for x in x2 if x]; x5 = [x for x in x5 if x]
                es = [x[1] for x in x2]
                pts[(c, d)] = dict(aa=aa, e=_st.median(es), ehi=max(x[2] for x in x2),
                                   refused=False, er=(min(es), max(es)),
                                   a2=_st.median([1 - x[0] if x[0] < EMPTY else 0.0 for x in x2]),
                                   a05=_st.median([1 - x[0] if x[0] < EMPTY else 0.0 for x in x5]))
                continue
            x2, x5 = at(rows_all, 0.2), at(rows_all, 0.05)
            if x2 is None:
                continue
            pts[(c, d)] = dict(aa=aa, e=x2[1], ehi=x2[2], refused=x2[0] >= EMPTY, er=None,
                               a2=0.0 if x2[0] >= EMPTY else 1 - x2[0],
                               a05=0.0 if (x5 is None or x5[0] >= EMPTY) else 1 - x5[0])

    ABBR = {"hospital": "hos", "beers": "bee", "flights": "fli", "rayyan": "ray", "tax": "tax"}
    names = [c for c, _ in CLEANERS]
    order = [(c, d) for c in names for d in DATASETS if (c, d) in pts]
    xs, gap, w = [], 0.9, 0.38
    x = 0.0
    centres = {}
    for c in names:
        for d in [d for cc, d in order if cc == c]:
            xs.append(x); centres[(c, d)] = x; x += 1.0
        x += gap
    YMAX = 0.5
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.34, 2.8), sharex=True,
                                   gridspec_kw={"hspace": 0.12, "height_ratios": [1.25, 1]})
    GREY, BLUE, LIGHT, RED = "#b3b6b7", "#1b4f72", "#9dbcd4", "#c0392b"
    for (c, d), xc in centres.items():
        q = pts[(c, d)]
        aa = q["aa"]
        if aa is not None:
            over = aa > 0.2
            ax1.bar(xc - w / 2, min(aa, YMAX), w, color="#8e9496" if over else GREY, zorder=2)
            if aa > YMAX:
                ax1.text(xc - w / 2, YMAX + 0.01, f"{aa:.2f}", fontsize=4.8, ha="center",
                         va="bottom", color="#566573", rotation=90)
        if q["refused"]:
            ax1.text(xc + w / 2, 0.012, "ref.", fontsize=5, ha="center", va="bottom",
                     color=BLUE, rotation=90)
            ax2.text(xc, 0.02, "ref.", fontsize=5, ha="center", va="bottom", color=BLUE,
                     rotation=90)
            continue
        ax1.bar(xc + w / 2, q["e"], w, color=BLUE, zorder=3)
        ax1.plot([xc + w / 2] * 2, [q["e"], q["ehi"]], color="#154360", lw=0.8, zorder=4)
        ax1.plot([xc + w / 2 - 0.12, xc + w / 2 + 0.12], [q["ehi"]] * 2, color="#154360",
                 lw=0.8, zorder=4)
        if q["er"]:
            ax1.plot([xc + w / 2] * 2, list(q["er"]), color="#f39c12", lw=1.6, zorder=5)
        ax2.bar(xc - w / 2, q["a05"], w, color=LIGHT, zorder=2)
        ax2.bar(xc + w / 2, q["a2"], w, color=BLUE, zorder=2)
    # budgets: alpha=0.2 is the one the text uses, so it is the heavy line
    ax1.axhline(0.2, ls="--", c=RED, lw=1.4, zorder=1)
    xb = centres.get(("GIDCL", "hospital"), xs[0]) - 0.45
    ax1.text(xb, 0.21, r"budget $\alpha=0.2$", fontsize=6.5, color=RED,
             ha="left", va="bottom", fontweight="bold")
    ax1.axhline(0.05, ls=":", c=RED, lw=0.7, zorder=1)
    xl = centres.get(("GIDCL", "beers"), xs[len(xs) // 2]) - 0.45
    ax1.text(xl, 0.058, r"$\alpha=0.05$", fontsize=5.5, color=RED, ha="left", va="bottom")
    ax1.set_ylim(0, YMAX + 0.12); ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax1.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.4", "0.5"])
    ax2.set_ylim(0, 1.08); ax2.set_yticks([0, 0.5, 1.0]); ax2.set_yticklabels(["0", "0.5", "1"])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([ABBR[d] for c, d in order], fontsize=6, rotation=90)
    for c in names:
        cx = [centres[(cc, d)] for cc, d in order if cc == c]
        if cx:
            ax1.text(sum(cx) / len(cx), YMAX + 0.13, c, fontsize=6.5, ha="center", va="bottom")
    ax1.set_ylabel("written error", fontsize=7)
    ax2.set_ylabel("share written", fontsize=7)
    ax2.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
    for ax in (ax1, ax2):
        _style_axes(ax, fs=7)
    handles = [Patch(color=GREY), Patch(color=BLUE),
               plt.Line2D([], [], color="#154360", lw=0.8),
               plt.Line2D([], [], color="#f39c12", lw=1.6),
               Patch(color=LIGHT), Patch(color=BLUE)]
    labels = ["apply-all error", r"CARE error, $\alpha=0.2$", "upper bound",
              "Baran min–max", r"written at $\alpha=0.05$", r"written at $\alpha=0.2$"]
    fig.legend(handles, labels, fontsize=6.0, frameon=False, loc="lower center", ncol=3,
               columnspacing=1.0, handletextpad=0.4, handlelength=1.3,
               bbox_to_anchor=(0.56, -0.005))
    fig.subplots_adjust(left=0.14, right=0.99, top=0.93, bottom=0.22)
    fig.savefig(f"{OUT_FIG}/fig_budget.pdf", bbox_inches="tight")
    print("  fig_budget.pdf")


def _poison_series():
    """Trusted-poison sweeps (extra_poison.tex): [(label, alpha, base error, points)]."""
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
            print(f"  warning: extra_poison: no CARE rows in {path} -- run the poison sweeps in reproduce.sh")
            continue
        pts = sorted((float(r["applied_contamination"]), float(r["realized_error"]),
                      float(r.get("err_lo") or 0), float(r.get("err_hi") or 0)) for r in rs)
        base = next((e for eps, e, _l, _h in pts if eps < 1e-9), 0.0)
        out.append((lab, a, base, pts))
    return out


PROPOSER_LABEL = {"baran": "Baran", "bclean": "BClean", "bclean_plus": "BClean+",
                  "gidcl": "GIDCL", "holoclean": "HoloClean", "zeroec": "ZeroEC"}


def fig_corruption(plt, alpha=0.2, partition="mondrian"):
    """fig_corruption.pdf: controlled corruption, one panel per table. The error of writing
    everything (apply-all) against the error of what CARE writes, as the injected error
    rate rises, with the budget line."""
    src = f"{EXP}/sweep_median.csv"
    if not os.path.exists(src):
        src = f"{EXP}/sweep.csv"
    rows = [r for r in _read(src)
            if r.get("partition") == partition
            and abs(float(r["alpha"]) - alpha) < 1e-9
            and r.get("kind") == "inner"
            and r.get("cell_error_rate") not in ("", None)]
    if not rows:
        print("  fig_corruption skipped (no experiments/sweep.csv; run audit_sweep.py)")
        return
    draws = max((int(r.get("draws", 1) or 1) for r in rows), default=1)
    bases = sorted({r["base"] for r in rows})
    GREY, BLUE, RED = "#8e9496", "#1b4f72", "#c0392b"
    ncol = 2
    nrow = (len(bases) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4, 1.2 * nrow + 0.45), sharex=True,
                             sharey=True, squeeze=False)
    xmax = max(float(r["cell_error_rate"]) for r in rows) * 1.08
    for i, base in enumerate(bases):
        ax = axes[i // ncol][i % ncol]
        sub = sorted((r for r in rows if r["base"] == base),
                     key=lambda r: float(r["cell_error_rate"]))
        x = [float(r["cell_error_rate"]) for r in sub]
        if all(r.get("acc") not in ("", None) for r in sub):
            ax.plot(x, [1 - float(r["acc"]) for r in sub], ls="-", lw=1.3, color=GREY,
                    marker="s", ms=3.4, zorder=2)
        ax.plot(x, [float(r["R"]) for r in sub], ls="-", lw=1.5, color=BLUE, marker="o",
                ms=3.6, zorder=3)
        ax.axhline(alpha, ls="--", lw=1.1, color=RED, zorder=1)
        ax.set_title(base, fontsize=8.5, pad=2)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.04, 1.04)
        ax.set_xticks([0, 0.2, 0.4]); ax.set_yticks([0, 0.2, 0.5, 1])
        ax.set_yticklabels(["0", "0.2", "0.5", "1"])
        _style_axes(ax, fs=8.5)
    for k in range(len(bases), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    axes[0][0].text(0.02, alpha + 0.035, r"budget $\alpha=0.2$", fontsize=7.5, color=RED,
                    fontweight="bold", va="bottom")
    fig.supxlabel("injected cell error rate", fontsize=8.5, y=0.015)
    fig.supylabel("error of the repairs written", fontsize=8.5, x=0.015)
    handles = [plt.Line2D([], [], color=GREY, ls="-", marker="s", ms=3, lw=1.2,
                          label="writing everything"),
               plt.Line2D([], [], color=BLUE, ls="-", marker="o", ms=3, lw=1.4,
                          label="writing what CARE admits")]
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.54, 1.0), handletextpad=0.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.14, top=0.86, wspace=0.08,
                        hspace=0.28)
    fig.savefig(f"{OUT_FIG}/fig_corruption.pdf", bbox_inches="tight")
    print(f"  fig_corruption.pdf ({len(rows)} points, {len(bases)} tables, alpha={alpha}, "
          f"{partition}, {'median of %d draws' % draws if draws > 1 else 'SINGLE DRAW'})")


def tab_sweep():
    """extra_sweep.tex: every swept variant with error rate, accuracy, ceiling, A and coverage."""
    rows = [r for r in _read(f"{EXP}/sweep.csv") if r.get("partition") == "mondrian"]
    if not rows:
        print("  extra_sweep skipped (no experiments/sweep.csv)")
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
    with open(f"{OUT_TAB}/extra_sweep.tex", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"  extra_sweep.tex ({len(rows)} rows)")


def fig_naive(plt):
    """fig_naive_threshold.pdf: violation rate of a held-out threshold vs CARE against label budget."""
    rows = _read(f"{EXP}/heldout_budget.csv")
    if not rows:
        print("  fig_naive_threshold: skipped (run audit_heldout_budget.py first)"); return
    COL = {("Baran", "beers"): "#7d6608", ("BClean", "flights"): "#78281f",
           ("BClean+", "beers"): "#1b4f72"}
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    delta, care_max = None, {}
    for (prop, ds), col in COL.items():
        pts = sorted((int(r["cal_size"]), int(r["seeds"]),
                      int(r["heldout_violations"]), int(r["care_violations"]))
                     for r in rows if r["proposer"] == prop and r["dataset"] == ds)
        if not pts:
            continue
        delta = float([r for r in rows if r["proposer"] == prop][0]["delta"])
        ax.plot([p[0] for p in pts], [p[2] / p[1] for p in pts], ls="-", marker="o",
                ms=3.8, lw=1.5, color=col, label=f"no bound, {prop}/{ds}")
        for p in pts:
            care_max[p[0]] = max(care_max.get(p[0], 0.0), p[3] / p[1])
    xs = sorted(care_max)
    ax.plot(xs, [care_max[x] for x in xs], ls="--", marker="s", ms=3.6, lw=1.5,
            color="#17202a", label="CARE, all three pairs", zorder=5)
    if delta is not None:
        ax.axhline(delta, ls=":", lw=1.0, color="#c0392b")
        ax.text(xs[0] * 1.04, delta + 0.02, rf"$\delta={delta:g}$",
                fontsize=8, color="#c0392b", ha="left")
    ax.set_xscale("log")
    _cs = sorted({int(r["cal_size"]) for r in rows})
    ax.set_xticks(_cs); ax.set_xticklabels([str(c) for c in _cs])
    ax.minorticks_off()
    ax.set_xlabel("labelled cells per stratum", fontsize=8.5)
    ax.set_ylabel("share of seeds over budget", fontsize=8.5)
    ax.set_ylim(-0.03, 0.85)
    _style_axes(ax, fs=8.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right", handlelength=2.2,
              labelspacing=0.3, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{OUT_FIG}/fig_naive_threshold.pdf", bbox_inches="tight")
    print("  fig_naive_threshold.pdf")


def fig_label_budget(plt):
    """extra_label_budget.pdf: error-bounded automation A(0.2) against labelled calibration cells."""
    rows = _read(f"{EXP}/label_budget.csv")
    if not rows:
        print("  extra_label_budget: skipped (run audit_label_budget.py first)"); return
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
        print("  extra_label_budget: no rows"); return
    ax.set_xscale("log")
    ax.set_xlabel("labelled calibration cells", fontsize=8)
    ax.set_ylabel(r"error-bounded automation $A(0.2)$", fontsize=8)
    ax.set_ylim(-0.03, 1.05)
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="#566573", ls="-", marker="o", ms=3, lw=1.2),
                plt.Line2D([], [], color="#566573", ls=":", marker="s", ms=3, lw=1.2)]
    labels += ["column-conditional", "marginal"]
    ax.legend(handles, labels, fontsize=5.8, frameon=False, loc="center right",
              handlelength=2.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(f"{OUT_FIG}/extra_label_budget.pdf", bbox_inches="tight")
    print("  extra_label_budget.pdf")


def tab_calibration():
    """extra_calibration.tex: mean confidence, accuracy and ECE per cleaner and dataset."""
    rows = _read(f"{EXP}/calibration.csv")
    if not rows:
        print("  extra_calibration: skipped (run audit_calibration.py first)"); return
    rows.sort(key=lambda r: -float(r["gap"]))
    out = [r"\begin{tabular}{llrrrr}", r"\toprule",
           r"proposer & dataset & $n$ & conf. & acc. & ECE \\", r"\midrule"]
    for r in rows:
        out.append(f"{r['proposer']} & {r['dataset']} & {int(r['n']):,} & "
                   f"{float(r['mean_conf']):.3f} & {float(r['accuracy']):.3f} & "
                   f"{float(r['ece']):.3f} \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{OUT_TAB}/extra_calibration.tex", "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("  extra_calibration.tex")


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


def fig_mechanism(plt):
    """fig_mechanism.pdf: (a) per-column error bounds on tax/Baran, (b) A(0.2) against accuracy per pair."""
    import collections
    acc_rows = _read("experiments/proposer_accuracy.csv")
    try:
        from bench.datasets import load
    except Exception as e:
        print(f"  fig_mechanism: skipped ({e})"); return
    # the run shown in the paper's problem figure (Baran's draw 2 on tax)
    path = TAX_RUN
    if not (os.path.exists(path) and acc_rows):
        print(f"  fig_mechanism: need {path} + audit_proposers.py output")
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
    # bound from a checked sample of at most 400 cells with the column's error rate
    def _bound(c):
        n = min(per[c], 400)
        return _cp_upper(round(bad[c] * n / per[c]), n, DELTA / L)
    bounds = {c: _bound(c) for c in per}
    cols = sorted(per, key=lambda c: bounds[c])

    from matplotlib.colors import LinearSegmentedColormap
    fig = plt.figure(figsize=(7.0, 2.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.85], wspace=0.30,
                             left=0.13, right=0.995, bottom=0.2, top=0.86)
    inner = outer[1].subgridspec(1, 2, wspace=0.07)
    ax = fig.add_subplot(outer[0])
    axb, axc = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])

    # (a) one bar per column. The message is the split: columns whose bound is under
    # every budget are written, the others go to a person.
    BLUE, GREY, RED = "#1b4f72", "#7f8c8d", "#c0392b"
    ys = list(range(len(cols)))
    vals = [bounds[c] for c in cols]
    n_adm = sum(v <= 0.2 for v in vals)
    share = sum(per[c] for c in cols[:n_adm]) / sum(per.values())
    ax.barh(ys, [max(v, 0.012) for v in vals], height=0.66, edgecolor="black",
            linewidth=0.4, zorder=2, color=[BLUE if v <= 0.2 else GREY for v in vals])
    ax.axvline(0.2, color=RED, lw=1.1, zorder=3)
    ax.text(0.215, -1.05, r"error budget $\alpha$ = 0.2", fontsize=8, color=RED,
            ha="left", va="center")
    ax.text(0.30, (n_adm - 1) / 2, f"written: {n_adm} columns,\n{share:.0%} of the repairs",
            fontsize=8.5, color=BLUE, fontweight="bold", ha="left", va="center",
            linespacing=1.15)
    ax.text(vals[n_adm] + 0.03, n_adm, f"escalated: {len(cols) - n_adm} columns",
            fontsize=8.5, color="#424949", fontweight="bold", ha="left", va="center",
            zorder=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c}  ({per[c]:,})" for c in cols], fontsize=7.5)
    for lab, v in zip(ax.get_yticklabels(), vals):
        lab.set_color(BLUE if v <= 0.2 else "#424949")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.invert_yaxis(); ax.set_xlim(0, 1.0); ax.set_ylim(len(cols) - 0.4, -1.7)
    ax.set_xlabel("upper bound on the column's error rate", fontsize=8.5)
    ax.set_title("(a) Baran on tax, column by column", fontsize=8.5, pad=3)
    _style_axes(ax)

    # (b), (c) one cell per cleaner-table pair: accuracy beside A(0.2)
    _bmed = _baran_medians()
    ACC, AUT, RNG = {}, {}, {}
    for c, rd in CLEANERS:
        for d in DATASETS:
            if (c, d) in TAB_MAIN_EXCLUDE:
                continue
            med = _bmed.get(d) if c == "Baran" else None
            if med:
                ACC[c, d], AUT[c, d] = float(med["acc"]), float(med["A2"])
                RNG[c, d] = (float(med["A2_min"]), float(med["A2_max"]))
                continue
            ar = [r for r in acc_rows if r["proposer"] == c and r["dataset"] == d]
            rows = [x for x in care(pareto(rd, d, c)) if abs(x[0] - 0.2) < 1e-9]
            if ar and rows:
                ACC[c, d], AUT[c, d] = float(ar[0]["acc_on_errors"]), 1 - rows[0][1]
    names = [c for c, _ in CLEANERS]
    short = {"hospital": "hos", "beers": "bee", "flights": "fli",
             "rayyan": "ray", "tax": "tax"}
    cmap = LinearSegmentedColormap.from_list("care", ["#f4f6f7", "#aed6f1", "#2e86c1", "#1b4f72"])
    for axm, title, V in ((axb, "(b) accuracy on error cells", ACC),
                          (axc, "(c) automation $A(0.2)$", AUT)):
        for i, c in enumerate(names):
            for j, d in enumerate(DATASETS):
                if (c, d) not in V:
                    axm.add_patch(plt.Rectangle((j, i), 1, 1, fc="white", ec="#d5d8dc",
                                                lw=0.6, hatch="////", alpha=0.5))
                    continue
                v = V[c, d]
                axm.add_patch(plt.Rectangle((j, i), 1, 1, fc=cmap(v), ec="white", lw=1.5))
                col = "white" if v > 0.62 else "#1b2631"
                txt = "ref." if (V is AUT and v == 0) else f"{v:.2f}"
                if V is AUT and ACC[c, d] >= 0.9999:
                    txt += r"$^\dagger$"
                axm.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=7.2,
                         color=col, fontweight="bold" if V is AUT else None)
        axm.set_xlim(0, len(DATASETS)); axm.set_ylim(len(names), 0)
        axm.set_xticks([j + 0.5 for j in range(len(DATASETS))])
        axm.set_xticklabels([short[d] for d in DATASETS], fontsize=7.5)
        axm.xaxis.tick_top()
        axm.set_yticks([i + 0.5 for i in range(len(names))])
        axm.set_yticklabels(names if axm is axb else [], fontsize=7.2)
        axm.tick_params(length=0, pad=1.5)
        for sp in axm.spines.values():
            sp.set_visible(False)
        axm.set_title(title, fontsize=8, pad=13)
    fig.savefig(f"{OUT_FIG}/fig_mechanism.pdf", bbox_inches="tight")
    print("  fig_mechanism.pdf")


def tab_main():
    """tab_main.tex: what each cleaner does alone, what writing everything unreviewed
    costs against two budgets, and what CARE writes while keeping them."""
    import statistics as _st
    _BARAN_MED = _baran_medians()
    audit = {(r["proposer"], r["dataset"]): r
             for r in _read("experiments/proposer_accuracy.csv")}
    OK = r"\cellcolor{okgreen}\checkmark"
    BAD = r"\cellcolor{badred}$\boldsymbol{\times}$"
    BUDGETS = (0.05, 0.2)
    L = [r"\normalsize", r"\setlength{\tabcolsep}{3.2pt}",
         r"\renewcommand{\arraystretch}{0.95}",
         r"\begin{tabular}{llrrrrrccrrrrc}", r"\toprule",
         r"& & \multicolumn{4}{c}{the cleaner's own repairs} & "
         r"\multicolumn{3}{c}{\begin{tabular}[b]{@{}c@{}}if every repair\\[-2pt]is written\end{tabular}} & "
         r"\multicolumn{5}{c}{if only what \textsc{Care} admits is written} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}\cmidrule(lr){10-14}",
         r"& & & & & & & \multicolumn{2}{c}{budget kept?} & "
         r"\multicolumn{3}{c}{share written, $A(\alpha)$} & & budget \\",
         r"\cmidrule(lr){8-9}\cmidrule(lr){10-12}",
         r"Cleaner & Table & $n$ & acc. & $\pi_0$ & overwr. & error & "
         r"$5\%$ & $20\%$ & $\alpha{=}0.05$ & $0.1$ & $0.2$ & error at $0.2$ & kept? \\",
         r"\midrule"]
    broken = {b: 0 for b in BUDGETS}
    written = {0.05: [], 0.1: [], 0.2: []}
    scored = 0
    for c, rd in CLEANERS:
        for d in DATASETS:
            a = audit.get((c, d))
            rows_all = pareto(rd, d, c)
            rows = care(rows_all)
            if not a:
                continue
            acc = float(a["acc_on_errors"])
            harm = f"{int(a.get('clean_broken') or 0):,}".replace(",", "{,}")
            med = _BARAN_MED.get(d) if c == "Baran" else None
            if med:
                acc = float(med["acc"])
                a = dict(a, frac_in_perfect_cols=med["pi0"])
                n = int(med["n"])
                draw_rows = _baran_draw_rows(d)
                draws = [applyall(r) for r in draw_rows]
                draws = [v for v in draws if v is not None]
                aa = _st.median(draws) if draws else applyall(rows_all)
            else:
                n = int(a["n_proposed_on_errors"])
                aa = applyall(rows_all)
            nstr = f"{n:,}".replace(",", "{,}")
            pre = (f"{c} & \\textsc{{{d}}}" + (r"$^{\dagger}$" if acc >= 0.9999 else "")
                   + f" & {nstr} & {acc:.3f} & {float(a['frac_in_perfect_cols']):.3f} & {harm}")
            if (c, d) in TAB_MAIN_EXCLUDE or not rows:
                if (c, d) == ("HoloClean", "beers"):
                    pre = pre.replace(harm, r"\textemdash")
                L.append(pre + r" & \multicolumn{8}{c}{\emph{excluded (nothing to govern)}} \\")
                continue
            scored += 1
            # the cleaner alone: error of writing everything, verdict at each budget
            verdict = []
            for b in BUDGETS:
                bad = aa is not None and aa > b
                broken[b] += bad
                verdict.append(BAD if bad else OK)
            broke2 = aa is not None and aa > 0.2
            astr = "--" if aa is None else (f"$\\mathbf{{{aa:.3f}}}$" if broke2 else f"${aa:.3f}$")
            # CARE: share written at each budget, realised error at 0.2
            by = {round(x[0], 3): x for x in rows}
            cells = []
            for al in (0.05, 0.1, 0.2):
                if med:
                    v = float(med["A" + str(al).replace("0.", "")])
                    v = 0.0 if v <= 1 - EMPTY else v
                else:
                    x = by.get(al)
                    v = None if x is None else (0.0 if x[1] >= EMPTY else 1 - x[1])
                if v is None:
                    cells.append("--"); continue
                written[al].append(v)
                cells.append(r"$0$" if v == 0.0 else f"${v:.2f}$")
            if med:
                refused2 = float(med["A2"]) <= 1 - EMPTY
                r2 = None if refused2 else float(med["R02"])
                # CARE kept the budget on every draw at every alpha (max realised error)
                kept = all(x[2] <= x[0] + 1e-12 for rr in draw_rows for x in care(rr))
            else:
                x2 = by.get(0.2)
                refused2 = x2 is None or x2[1] >= EMPTY
                r2 = None if refused2 else x2[2]
                kept = all(x[2] <= x[0] + 1e-12 for x in rows)
            rstr = r"$\varnothing$" if r2 is None else (f"$\\mathbf{{{r2:.3f}}}$" if broke2 else f"${r2:.3f}$")
            if all(x == r"$0$" for x in cells):
                tail = r"\multicolumn{4}{c}{\emph{refused at every $\alpha$}}"
            else:
                tail = " & ".join(cells) + f" & {rstr}"
            L.append(pre + f" & {astr} & " + " & ".join(verdict) + " & " + tail
                     + f" & {OK if kept else BAD} \\\\")
        L.append(r"\addlinespace[1.5pt]")
    while L and L[-1].startswith(r"\addlinespace"):   # a cleaner with no rows yet
        L.pop()
    mean_w = {al: sum(v) / len(v) for al, v in written.items() if v}
    L += [r"\midrule",
          r"\multicolumn{7}{l}{pairs that break the budget (of $" + str(scored) + r"$)} & "
          + " & ".join(f"$\\mathbf{{{broken[b]}}}$" for b in BUDGETS)
          + r" & & & & & $\mathbf{0}$ \\",
          r"\multicolumn{9}{l}{average share written, $A(\alpha)$, over the $" + str(scored) + r"$ pairs} & "
          + " & ".join(f"$\\mathbf{{{mean_w[al]:.2f}}}$" for al in (0.05, 0.1, 0.2)) + r" & & \\",
          r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab_main.tex", "w").write("\n".join(L) + "\n")
    print("  tab_main.tex")


def tab_poison():
    """extra_poison.tex: predicted vs observed error under trusted poison."""
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
    open(f"{OUT_TAB}/extra_poison.tex", "w").write("\n".join(L) + "\n")
    print("  extra_poison.tex")


def tab_detector():
    """tab_detector.tex: A(0.2) and R(0.2) under oracle vs real (Raha or own) detection."""
    # (cleaner, oracle dir, real-detector dir, footnote marker); rows need both sweeps
    PAIRS = [("Baran", "results_baran_detected", "results_raha_baran", ""),
             ("BClean", "results_bclean_keep_oracle", "results_raha_bclean",
              r"$^{\ddagger}$"),
             ("HoloClean", "results_holoclean_keep_oracle", "results_raha_holoclean",
              r"$^{\ddagger}$"),
             # GIDCL: oracle queue vs its own detector's queue
             ("GIDCL", "results_gidcl", "results_gidcl_owndet", r"$^{\S}$")]
    # cleaner as a spanning row header keeps the table one column wide
    L = [r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{0.9}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"& \multicolumn{2}{c}{detector} & \multicolumn{2}{c}{oracle}"
         r" & \multicolumn{2}{c}{real detector} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Dataset & prec. & rec. & share & error & share & error \\", r"\midrule"]
    dq = {r["dataset"]: r for r in _read("experiments/detector_quality.csv")}
    dq_own = _own_detector_quality("gidcl")

    def fmt(x):
        # empty applied set: "refused" in the share column and a dash for the error, so
        # every row keeps one value per column
        _a, h, e, _hi, _c = x
        return (r"\emph{refused}", None) if h >= EMPTY else \
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
            # bold: yield did not fall under the real detector (A real >= A oracle, both written)
            if re_ is not None and oe is not None and (1 - r[0][1]) >= (1 - o[0][1]) - 1e-9:
                ra = r"$\mathbf{" + ra.strip("$") + "}$"
            if not wrote:
                L.append(r"\multicolumn{7}{@{}l}{\emph{" + prop + r"}" + mark + r"} \\")
            ocell = f"{oa} & --" if oe is None else f"{oa} & {oe}"
            rcell = f"{ra} & --" if re_ is None else f"{ra} & {re_}"
            L.append(f"\\quad\\textsc{{{d}}} & {float(q.get('precision', 0)):.2f} & "
                     f"{float(q.get('recall', 0)):.2f} & {ocell} & {rcell} \\\\")
            wrote = any_row = True
        if wrote:
            L.append(r"\addlinespace[2pt]")
    if not any_row:
        print("  tab_detector: no oracle/non-oracle pair available"); return
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT_TAB}/tab_detector.tex", "w").write("\n".join(L) + "\n")
    print("  tab_detector.tex")


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


def tab_error_drop():
    """extra_error_drop.tex: error drop rate of apply-all vs CARE at alpha=0.2 (oracle detection)."""
    rows = _read(f"{EXP}/error_drop.csv")
    if not rows:
        print("  extra_error_drop: skipped (run audit_error_drop.py first)"); return
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
    open(f"{OUT_TAB}/extra_error_drop.tex", "w").write("\n".join(L) + "\n")
    print(f"  extra_error_drop.tex ({len(sel)} rows)")


def tab_gidcl_confidence():
    """extra_gidcl_confidence.tex: AUROC of GIDCL's repair-selection score per seed."""
    rows = _read(f"{EXP}/gidcl_confidence.csv")
    if not rows:
        print("  extra_gidcl_confidence: skipped (run audit_gidcl_confidence.py first)"); return
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
    open(f"{OUT_TAB}/extra_gidcl_confidence.tex", "w").write("\n".join(L) + "\n")
    print("  extra_gidcl_confidence.tex")


def fig_planner(plt):
    """extra_planner.pdf: label planner, (a) break-even stratum size, (b) reviews saved vs column count."""
    rows = _read(f"{EXP}/planner_breakeven.csv")
    if not rows:
        print("  extra_planner: skipped (run tools/audit_planner.py first)"); return
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
    fig.savefig(f"{OUT_FIG}/extra_planner.pdf", bbox_inches="tight")
    print("  extra_planner.pdf")


def tab_scaling():
    """extra_scaling.tex (numbers quoted in Section 4.2): index build, memory, verification and calibration cost on the first N rows of tax."""
    rows = _read(f"{EXP}/scaling.csv")
    if not rows:
        print("  extra_scaling: skipped (run tools/audit_scaling.py first)"); return
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
    open(f"{OUT_TAB}/extra_scaling.tex", "w").write("\n".join(L) + "\n")
    print("  extra_scaling.tex")


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
        fig_budget(plt); fig_mechanism(plt); fig_label_budget(plt); fig_naive(plt); fig_corruption(plt); fig_planner(plt)
    tab_main(); tab_poison(); tab_detector(); tab_calibration(); tab_error_drop(); tab_sweep(); tab_gidcl_confidence(); tab_scaling()
    print("done -> paper/figures/*.pdf, paper/tables/*.tex")


if __name__ == "__main__":
    main()
