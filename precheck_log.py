"""precheck_log.py -- FREE certifiability pre-check for any cleaner's repair log.

Run AFTER the (expensive) generation step and BEFORE the pareto sweep, to predict
whether CARE will certify any automation -- so you never spend a sweep on a config
that will just escalate everything.

TWO SCOPES (the distinction matters, and it's a real methodological choice):

  --scope errors   Score only the dataset's TRUE ERROR cells. This is what CARE
                   evaluates under ORACLE detection: the hardest possible subset,
                   and the standard repair-F1 population.

  --scope touched  Score EVERY cell the cleaner proposed on (default). Repairing a
                   clean cell to itself counts as correct; changing it counts as an
                   error. This is what CARE evaluates when the cleaner's own log is
                   the work queue (`--detection log:<path>`), and it is the more
                   deployment-realistic picture: a cleaner proposes on many cells and
                   the governance layer must judge all of them.

The two can disagree sharply. On hospital/Jellyfish, ProviderNumber is 2.8% error
over touched cells (it preserves clean values well) but 0% successful on true error
cells. Report whichever you use, explicitly.

It also reports whether the CONFIDENCE column actually discriminates -- if
high-confidence repairs are not more accurate than low-confidence ones, the conformal
controller has nothing to rank on and can only take a stratum all-or-nothing.

Usage (from the CARE repo root):
    python precheck_log.py <dataset> <log.csv> [--scope touched|errors] [--alpha 0.1]
"""

import argparse
import collections
import csv
import statistics as stats

try:
    from prepare_log import COL_MAPS
except Exception:
    COL_MAPS = {}

# A per-column stratum needs enough calibration mass AND low enough error: ~60
# attempted cells gives ~24 calibration after a 40/60 split, roughly the floor at
# which an exact Clopper-Pearson + Bonferroni bound can certify error <= 0.2 at
# delta=0.1. Heuristic, but it matched the real sweep on hospital and flights.
_MIN_MASS = 60


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("log")
    ap.add_argument("--scope", default="touched", choices=["touched", "errors"])
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    from bench.datasets import load
    art, defects, _dcols = load(args.dataset)
    gold = {k: (str(v).lower().strip() if v is not None else v)
            for k, v in defects.gold.items()}
    cur = {f"{c.row_id}::{c.col}": (str(c.value).lower().strip() if c.value is not None else "")
           for c in art.iter_cells()}
    cmap = COL_MAPS.get(args.dataset, {})
    mapped = lambda c: cmap.get(c, c)

    per = collections.defaultdict(lambda: [0, 0])      # col -> [n, correct]
    conf_recs = []                                     # (confidence, correct)
    with open(args.log, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            col = mapped(r["column"])
            ref = f"{r['row_id']}::{col}"
            v = (r.get("value") or "").strip().lower()
            if not v:
                continue
            if ref in gold:
                ok = (v == str(gold[ref]).lower().strip())
            elif args.scope == "touched" and ref in cur:
                ok = (v == cur[ref])                   # preserved a clean cell
            else:
                continue
            per[col][0] += 1
            per[col][1] += ok
            try:
                conf_recs.append((float(r["confidence"]), ok))
            except (KeyError, TypeError, ValueError):
                pass

    print(f"\n========== PRE-CHECK: {args.dataset}  (scope={args.scope}, "
          f"alpha={args.alpha}) ==========")
    print(f"{'column':18}{'n':>7}{'acc':>9}{'err':>9}   verdict")
    any_cert = False
    tot = ok_tot = 0
    for col, (n, k) in sorted(per.items(), key=lambda x: -x[1][0]):
        acc = k / n if n else 0.0
        err = 1 - acc
        tot += n; ok_tot += k
        cert = (n >= _MIN_MASS and err <= args.alpha)
        any_cert = any_cert or cert
        print(f"{col:18}{n:7}{acc:9.1%}{err:9.1%}   {'<== likely CERTIFIES' if cert else ''}")
    if tot:
        print(f"\noverall: n={tot}  acc={ok_tot/tot:.1%}  err={1-ok_tot/tot:.1%}")

    if conf_recs:
        # Break ties RANDOMLY, not by correctness. `sort(reverse=True)` on
        # (confidence, correct) puts correct answers first inside a tie group, so
        # "top 10%" reports the best 10% of a tie rather than a random 10%. These
        # signals are coarse -- self-consistency over 5 samples takes 6 values, and
        # 65% of tax cells sit at the maximum -- so that bias is large: it reported
        # top-10% accuracy of 1.000 on three logs whose fair values are 0.51-0.75.
        import random as _rnd
        _rnd.Random(0).shuffle(conf_recs)
        conf_recs.sort(key=lambda r: -r[0])
        ties = sum(1 for c, _o in conf_recs if c == conf_recs[0][0]) / len(conf_recs)
        print("\nconfidence discrimination (top-k by confidence, ties broken at random):")
        for f in (0.1, 0.25, 0.5, 1.0):
            k = max(1, int(len(conf_recs) * f))
            sub = conf_recs[:k]
            acc = sum(1 for _c, o in sub if o) / len(sub)
            print(f"  top {f*100:4.0f}%: acc={acc:6.1%}  err={1-acc:6.1%}")
        if ties > 0.15:
            print(f"  NOTE: {ties:.0%} of cells share the maximum confidence, so any "
                  f"top-k% below that is an arbitrary slice of one tie group.\n"
                  f"        Prefer the per-level accuracies over top-k% here.")
        cok = [c for c, o in conf_recs if o]
        cno = [c for c, o in conf_recs if not o]
        if cok and cno:
            mo, mn = stats.mean(cok), stats.mean(cno)
            verdict = "USABLE (ranks correctly)" if mo > mn else \
                      "ANTI-CORRELATED -- CARE would auto-apply the WORST repairs first"
            print(f"  mean conf | correct={mo:.3f}  incorrect={mn:.3f}  -> {verdict}")

    print()
    if any_cert:
        print("VERDICT: at least one column has the mass + accuracy to certify -> the "
              "sweep should show REAL graded automation. PROCEED.")
    else:
        print("VERDICT: no column clears the mass+accuracy bar at this alpha -> the "
              "sweep will likely escalate ~100%. Reconsider (or try --scope touched / "
              "a looser alpha) before spending it.")
    print("=" * (30 + len(args.dataset)) + "\n")


if __name__ == "__main__":
    main()
