"""audit_heldout_budget.py -- does the finite-sample correction earn its cost?

The first question about CARE is why the Clopper-Pearson bound and the Bonferroni
correction are needed at all: the calibration set already reports an error
rate, so why not take the most permissive threshold whose OBSERVED error is at or
below alpha? `audit_recalibration.py` answers that at the natural calibration size and
the answer there is uncomfortable -- the naive rule never exceeds the budget and
automates more than CARE does.

That comparison is run in the wrong regime, and this script says so with data. The
bound is a finite-sample correction; where n is large it converges to the empirical
rate and costs automation for nothing. It earns its keep only where a stratum holds
few labelled cells, which Section 7.3 identifies as the operating point that decides
certifiability in the first place. So the honest experiment sweeps the label budget
and asks where the two rules separate.

    python audit_heldout_budget.py
    -> experiments/heldout_budget.csv

Reported per (proposer, dataset, budget): how often each rule's realised error on the
cells it auto-applied exceeded alpha, over `seeds` random splits, and how much it
automated. A valid rule may exceed the budget at most a delta fraction of the time;
nothing constrains the naive one. Read the violation columns together with the
automation columns -- a rule that certifies nothing trivially never violates.
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
    ("Baran", "beers", "csvs/baran_beers_mapped.csv"),
    ("BClean", "flights", "csvs/bclean_flights_mapped.csv"),
    ("Jellyfish", "beers", "csvs/jellyfish_beers_mapped.csv"),
]
BUDGETS = [25, 50, 100, 200, 400, 800]


def heldout_thresholds(scores, oks, strata, *, alpha, grid):
    """Per-stratum thresholds selected on EMPIRICAL calibration error.

    CARE's own procedure with the bound removed: same fixed grid, same per-stratum
    search, same smallest-threshold-wins rule. A stratum is accepted when its observed
    error is at or below alpha rather than when a 1-delta upper bound is.
    """
    by = {}
    for s, ok, st in zip(scores, oks, strata):
        by.setdefault(st, []).append((s, ok))
    out = {}
    for st, pairs in by.items():
        pairs.sort(key=lambda t: t[0])
        ss = [s for s, _ in pairs]
        total = len(pairs)
        suffix_wrong = [0.0] * (total + 1)
        for i in range(total - 1, -1, -1):
            suffix_wrong[i] = suffix_wrong[i + 1] + (0.0 if pairs[i][1] else 1.0)
        chosen = math.inf
        for lam in grid:
            i = bisect.bisect_left(ss, lam)
            n = total - i
            if n and suffix_wrong[i] / n <= alpha:
                chosen = lam
                break
        out[st] = chosen
    return out


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench.study import evaluate_split, load_and_propose
    from care.conformal import by_cardinality, by_column
    from care.conformal.rcps import _DEFAULT_GRID, _candidate_grid
    from care.core.artifact import CellKey as _CK

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budgets", nargs="*", type=int, default=BUDGETS)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="experiments/heldout_budget.csv")
    args = ap.parse_args()

    import collections as _c

    grid = _candidate_grid(_DEFAULT_GRID)
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

        def ok(r):
            return 1.0 if (str(repairs[r].proposed_value).lower().strip()
                           == gold[r]) else 0.0

        print(f"  {'budget':>7} | {'CARE viol':>10} {'CARE A':>7} | "
              f"{'HeldOut viol':>13} {'HeldOut A':>10} {'HeldOut worst':>14}")
        for budget in args.budgets:
            care_v = held_v = 0
            care_a, held_a, held_e = [], [], []
            for s in range(args.seeds):
                scores, dec = evaluate_split(
                    repairs, gold, alpha=args.alpha, delta=args.delta, seed=s,
                    strata_fn=sf, return_decisions=True, cal_size=budget)
                cal, test = dec["cal_refs"], dec["test_refs"]
                c = scores["CARE"]
                care_a.append(1.0 - c.human_cost)
                if 1.0 - c.human_cost > 0:
                    care_v += c.realized_error > args.alpha + 1e-12

                lam = heldout_thresholds(
                    [repairs[r].s_hat for r in cal], [ok(r) for r in cal],
                    [sf(repairs[r]) for r in cal], alpha=args.alpha, grid=grid)
                sel = [r for r in test
                       if repairs[r].s_hat >= lam.get(sf(repairs[r]), math.inf)]
                held_a.append(len(sel) / max(len(test), 1))
                if sel:
                    e = 1.0 - sum(ok(r) for r in sel) / len(sel)
                    held_e.append(e)
                    held_v += e > args.alpha + 1e-12

            worst = max(held_e) if held_e else None
            print(f"  {budget:>7} | {care_v:>4}/{args.seeds:<5} "
                  f"{statistics.mean(care_a):>7.3f} | {held_v:>6}/{args.seeds:<6} "
                  f"{statistics.mean(held_a):>10.3f} "
                  f"{(f'{worst:.3f}' if worst is not None else '-'):>14}")
            rows.append({
                "proposer": tag, "dataset": ds, "cal_size": budget,
                "alpha": args.alpha, "delta": args.delta, "seeds": args.seeds,
                "care_violations": care_v,
                "care_automation": round(statistics.mean(care_a), 4),
                "heldout_violations": held_v,
                "heldout_automation": round(statistics.mean(held_a), 4),
                "heldout_worst_error": round(worst, 4) if worst is not None else None,
            })

    if not rows:
        print("no configurations ran")
        return 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tot_h = sum(r["heldout_violations"] for r in rows)
    tot_c = sum(r["care_violations"] for r in rows)
    n = sum(r["seeds"] for r in rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    print(f"over all budgets: CARE {tot_c}/{n} violations, HeldOut {tot_h}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
