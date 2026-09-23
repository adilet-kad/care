"""audit_recalibration.py -- would recalibrating the confidence be enough?

Section 8.3 shows proposer confidence is not a probability. The natural reply is
that this is a solved problem: fit Platt scaling or isotonic regression on held-out
data, then auto-apply everything scoring above 1-alpha. This script tests that
reply directly, on the same splits, budgets and seeds CARE is evaluated on.

The reply fails, and for a reason worth stating precisely. Recalibration targets a
MARGINAL property: after fitting, cells scoring 0.9 are right about 90% of the time
*on average over the whole pool*. Auto-applying is a SELECTION, and the selected
subset is exactly where the recalibrated score is most optimistic, because
selection keeps the cells whose scores landed high -- including the ones that
landed high by estimation error. Nothing in Platt or isotonic bounds the error rate
*conditional on having been selected*, and neither carries a finite-sample
correction: both are asymptotic fits, so on a few dozen calibration cells the fitted
map is itself noisy in a direction selection then exploits.

What we report is therefore not calibration quality but budget compliance: over
`seeds` random splits, how often does each policy's realised error on the cells it
auto-applied exceed alpha? CARE's guarantee says this should happen at most a
delta fraction of the time. Nothing constrains the recalibrated thresholds.

Both fits are stdlib-only, matching the rest of the library: Platt is a 1-D logistic
regression by gradient descent, isotonic is the pool-adjacent-violators algorithm.

    python audit_recalibration.py
    -> experiments/recalibration.csv
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import statistics
import sys

CONFIGS = [
    ("Baran", "hospital", "csvs/baran_hospital_mapped.csv"),
    ("Baran", "beers", "csvs/baran_beers_mapped.csv"),
    ("BClean", "flights", "csvs/bclean_flights_mapped.csv"),
    ("Jellyfish", "beers", "csvs/jellyfish_beers_mapped.csv"),
]


# --------------------------------------------------------------------------
def platt(scores, labels, *, epochs=400, lr=0.5):
    """1-D logistic recalibration: P(correct) = sigmoid(a*s + b)."""
    a, b, n = 1.0, 0.0, max(len(scores), 1)
    for _ in range(epochs):
        ga = gb = 0.0
        for s, y in zip(scores, labels):
            p = 1 / (1 + math.exp(-max(-30, min(30, a * s + b))))
            e = p - y
            ga += e * s
            gb += e
        a -= lr * ga / n
        b -= lr * gb / n
    return lambda s: 1 / (1 + math.exp(-max(-30, min(30, a * s + b))))


def isotonic(scores, labels):
    """Pool-adjacent-violators: the standard non-parametric calibration map."""
    pts = sorted(zip(scores, labels))
    if not pts:
        return lambda s: 0.0
    xs = [p[0] for p in pts]
    blocks = [[p[1], 1.0] for p in pts]          # [sum, count]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] / blocks[i][1] <= blocks[i + 1][0] / blocks[i + 1][1] + 1e-12:
            i += 1
            continue
        blocks[i][0] += blocks[i + 1][0]
        blocks[i][1] += blocks[i + 1][1]
        del blocks[i + 1]
        del xs[i + 1]
        if i:
            i -= 1
    vals = [b[0] / b[1] for b in blocks]

    def f(s):
        lo, hi = 0, len(xs) - 1
        while lo < hi:                            # last knot with x <= s
            mid = (lo + hi + 1) // 2
            if xs[mid] <= s:
                lo = mid
            else:
                hi = mid - 1
        return vals[lo] if xs[lo] <= s else vals[0]
    return f


def heldout_thresholds(cal_scores, cal_ok, cal_stratum, *, alpha, grid):
    """Per-stratum thresholds chosen by EMPIRICAL calibration error alone.

    This is CARE's own procedure with exactly one thing removed: the
    Clopper-Pearson upper bound and its Bonferroni correction. Same fixed grid,
    same per-stratum selection, same "smallest threshold wins" tie-break -- the
    only difference is that a stratum is accepted when its OBSERVED error on the
    calibration cells is at or below alpha, rather than when the upper end of a
    1-delta confidence interval is.

    It is the baseline a practitioner reaches for first: why pay for a bound
    when the calibration set already tells you the error rate? The answer is
    that k/n is an estimate, selection runs
    over the whole grid, and the threshold that looks best is disproportionately
    the one whose estimate was luckiest. Returns {stratum: threshold}; a stratum
    with no acceptable threshold escalates entirely, as it does under CARE.
    """
    by_stratum = {}
    for s, ok, st in zip(cal_scores, cal_ok, cal_stratum):
        by_stratum.setdefault(st, []).append((s, ok))
    out = {}
    for st, pairs in by_stratum.items():
        pairs.sort(key=lambda t: t[0])
        scores = [s for s, _ in pairs]
        total = len(pairs)
        suffix_wrong = [0] * (total + 1)
        for i in range(total - 1, -1, -1):
            suffix_wrong[i] = suffix_wrong[i + 1] + (0.0 if pairs[i][1] else 1.0)
        chosen = math.inf
        for lam in grid:
            i = bisect.bisect_left(scores, lam)
            n = total - i
            if n == 0:
                continue
            if suffix_wrong[i] / n <= alpha:      # empirical rate, no bound
                chosen = lam
                break
        out[st] = chosen
    return out


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench.study import evaluate_split, load_and_propose
    from care.conformal import by_cardinality, by_column
    from care.core.artifact import CellKey as _CK

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="experiments/recalibration.csv")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    import collections as _c

    rows = []
    for tag, ds, log in CONFIGS:
        if args.only and ds not in args.only:
            continue
        if not os.path.exists(log):
            print(f"  [skip] {tag}/{ds}: missing {log}")
            continue
        print(f"\n===== {tag} / {ds} =====", flush=True)
        art, defects, repairs, _ = load_and_propose(
            ds, f"{tag.lower()}:{log}", detection="oracle", fast_verify=True)
        gold = {k: (str(v).lower().strip() if v is not None else v)
                for k, v in defects.gold.items()}
        counts = _c.Counter(_CK.parse(r).col for r in gold)
        median = sorted(counts.values())[len(counts) // 2] if counts else 0
        sf = by_column() if median >= 60 else by_cardinality(art)

        from care.conformal.rcps import _DEFAULT_GRID, _candidate_grid
        grid = _candidate_grid(_DEFAULT_GRID)      # the same grid CARE searches

        acc = {k: {"viol": 0, "err": [], "auto": []}
               for k in ("CARE", "HeldOut", "Platt", "Isotonic", "Raw")}
        for s in range(args.seeds):
            scores, dec = evaluate_split(repairs, gold, alpha=args.alpha,
                                         delta=args.delta, seed=s, strata_fn=sf,
                                         return_decisions=True)
            cal, test = dec["cal_refs"], dec["test_refs"]
            ok = lambda r: 1.0 if (str(repairs[r].proposed_value).lower().strip()
                                   == gold[r]) else 0.0          # noqa: E731
            cs = [repairs[r].s_hat for r in cal]
            cy = [ok(r) for r in cal]
            maps = {"Platt": platt(cs, cy), "Isotonic": isotonic(cs, cy),
                    "Raw": lambda x: x}
            # HeldOut: per-stratum, empirical-error selection on the SAME grid.
            lam = heldout_thresholds([repairs[r].s_hat for r in cal],
                                     [ok(r) for r in cal], [sf(repairs[r]) for r in cal],
                                     alpha=args.alpha, grid=grid)
            sel = [r for r in test
                   if repairs[r].s_hat >= lam.get(sf(repairs[r]), math.inf)]
            if sel:
                e = 1.0 - sum(ok(r) for r in sel) / len(sel)
                acc["HeldOut"]["err"].append(e)
                acc["HeldOut"]["viol"] += (e > args.alpha + 1e-12)
            acc["HeldOut"]["auto"].append(len(sel) / max(len(test), 1))

            for name, f in maps.items():
                sel = [r for r in test if f(repairs[r].s_hat) >= 1 - args.alpha]
                if sel:
                    e = 1.0 - sum(ok(r) for r in sel) / len(sel)
                    acc[name]["err"].append(e)
                    acc[name]["viol"] += (e > args.alpha + 1e-12)
                acc[name]["auto"].append(len(sel) / max(len(test), 1))
            c = scores["CARE"]
            if 1.0 - c.human_cost > 0:
                acc["CARE"]["err"].append(c.realized_error)
                acc["CARE"]["viol"] += (c.realized_error > args.alpha + 1e-12)
            acc["CARE"]["auto"].append(1.0 - c.human_cost)

        for name, d in acc.items():
            rows.append({
                "policy": name, "proposer": tag, "dataset": ds,
                "alpha": args.alpha, "seeds": args.seeds,
                "budget_violations": d["viol"],
                "violation_rate": round(d["viol"] / args.seeds, 3),
                "mean_error": round(statistics.mean(d["err"]), 4) if d["err"] else None,
                "max_error": round(max(d["err"]), 4) if d["err"] else None,
                "mean_automation": round(statistics.mean(d["auto"]), 4),
            })
            r = rows[-1]
            print(f"  {name:<9} viol {r['budget_violations']:>2}/{args.seeds}  "
                  f"err mean {r['mean_error']}  max {r['max_error']}  "
                  f"A {r['mean_automation']:.3f}", flush=True)

    if not rows:
        return 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    existing = args.append and os.path.exists(args.out)
    with open(args.out, "a" if existing else "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if not existing:
            wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
