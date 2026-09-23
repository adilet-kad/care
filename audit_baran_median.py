"""audit_baran_median.py -- Baran's Table 1 row, as a median over draws.

Baran selects its labelled tuples by unseeded active learning, so a single run is one
draw from a distribution, not a measurement. That matters here beyond tidiness: the
log shipped in `csvs/` reports A(0.2)=0.86 on hospital, the MAXIMUM of six independent
draws whose median is 0.58, and 0.77 on rayyan against a median of 0.69; on tax the
shipped log is the LEAST accurate draw (0.732 against a median of 0.914). Reporting a
single draw is not wrong, but anyone re-running the cleaner will rarely reproduce it.

So Table 1's Baran rows report the MEDIAN over every draw available -- the shipped log
plus the five in `csvs/baran_var_*/` -- for every column of the table, not only for
automation. Reporting a median automation beside a single draw's accuracy would mix
two different runs in one row.

The median rather than the mean: the marginal control column is bimodal (a proposer
emitting constant confidence certifies all or none), and a mean between two modes
describes no run that happened.

    python audit_baran_median.py
    -> experiments/baran_median.csv

flights is absent by design: Baran is exactly correct there, every draw certifies
everything, and the pair is excluded from the paper's claims as degenerate.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# dataset -> (results dirs holding one subdirectory per draw, mapped-log dirs)
CONFIGS = {
    "hospital": ("experiments/results_baran_var_hospital", "csvs/baran_var_hospital"),
    "beers": ("experiments/results_baran_var_beers", "csvs/baran_var_beers"),
    "rayyan": ("experiments/results_baran_var_rayyan", "csvs/baran_var_rayyan"),
    "tax": ("experiments/results_baran_var_tax", "csvs/baran_var_tax"),
}
ALPHAS = (0.05, 0.1, 0.2)


def _norm(v):
    return str(v).strip().lower() if v is not None else ""


def proposer_stats(ds, log, gold):
    """n on error cells, accuracy, pi_0 -- the three proposer columns of Table 1."""
    per, ok = collections.Counter(), collections.Counter()
    n = 0
    with open(log, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            ref = f"{r['row_id']}::{r['column']}"
            if ref in gold:
                n += 1
                per[r["column"]] += 1
                ok[r["column"]] += _norm(r.get("value")) == gold[ref]
    if not n:
        return None
    acc = sum(ok.values()) / n
    pi0 = sum(per[c] for c in per if ok[c] == per[c]) / n
    return n, acc, pi0


def care(path, alpha, strata="mondrian"):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if (r["baseline"] == "CARE" and r["strata"] == strata
                    and abs(float(r["alpha"]) - alpha) < 1e-9):
                return 1.0 - float(r["human_cost"]), float(r["realized_error"])
    return None


def main() -> int:
    from bench.datasets import load

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="experiments/baran_median.csv")
    args = ap.parse_args()
    os.chdir(ROOT)

    rows = []
    for ds, (rdir, ldir) in CONFIGS.items():
        gold = {k: _norm(v) for k, v in load(ds)[1].gold.items()}
        logs = [f"csvs/baran_{ds}_mapped.csv"] + sorted(
            glob.glob(os.path.join(ldir, "*", f"baran_{ds}_mapped.csv")))
        res = [f"experiments/results_baran_errors/{ds}_pareto.csv"] + sorted(
            glob.glob(os.path.join(rdir, "*", f"{ds}_pareto.csv")))
        logs = [p for p in logs if os.path.exists(p)]
        res = [p for p in res if os.path.exists(p)]
        if len(logs) != len(res):
            print(f"  [skip] {ds}: {len(logs)} logs but {len(res)} result files -- "
                  f"each draw needs both, or the median mixes runs")
            continue
        if len(logs) < 2:
            print(f"  [skip] {ds}: only {len(logs)} draw(s); see docs/REPRODUCE.md section 3")
            continue

        S = [proposer_stats(ds, p, gold) for p in logs]
        S = [s for s in S if s]
        rec = {"dataset": ds, "draws": len(S),
               "n": round(statistics.median(s[0] for s in S)),
               "acc": round(statistics.median(s[1] for s in S), 4),
               "pi0": round(statistics.median(s[2] for s in S), 4),
               "n_min": min(s[0] for s in S), "n_max": max(s[0] for s in S),
               "acc_min": round(min(s[1] for s in S), 4),
               "acc_max": round(max(s[1] for s in S), 4),
               "pi0_min": round(min(s[2] for s in S), 4),
               "pi0_max": round(max(s[2] for s in S), 4)}
        for a in ALPHAS:
            vals = [care(p, a) for p in res]
            vals = [v for v in vals if v]
            au = sorted(v[0] for v in vals)
            er = [v[1] for v in vals]
            tag = str(a).replace("0.", "")
            rec[f"A{tag}"] = round(statistics.median(au), 4)
            rec[f"A{tag}_min"] = round(au[0], 4)
            rec[f"A{tag}_max"] = round(au[-1], 4)
            if a == 0.2:
                rec["R02"] = round(statistics.median(er), 4)
                rec["R02_max"] = round(max(er), 4)
        rows.append(rec)

    if not rows:
        print("no datasets had draws; nothing written")
        return 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'dataset':<10}{'draws':>6}{'n':>7}{'acc':>7}{'pi0':>7}"
          f"{'A(.05)':>8}{'A(.1)':>7}{'A(.2)':>7}{'R(.2)':>8}")
    for r in rows:
        print(f"{r['dataset']:<10}{r['draws']:>6}{r['n']:>7}{r['acc']:>7.3f}"
              f"{r['pi0']:>7.3f}{r['A05']:>8.2f}{r['A1']:>7.2f}{r['A2']:>7.2f}"
              f"{r['R02']:>8.3f}")
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    print("A(0.2) spread per dataset (min -> max over draws):")
    for r in rows:
        print(f"  {r['dataset']:<10} {r['A2_min']:.2f} -> {r['A2_max']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
