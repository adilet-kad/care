"""audit_drift_probe.py -- does the guarantee survive a non-exchangeable split?

Assumption A1 says calibration and deployment cells are exchangeable. Section 9 names
non-stationarity as the sharper threat and does nothing about it. This is the cheapest
experiment that finds out whether that admission is a formality or a real hole.

The manipulation is the split, not the data. CARE normally calibrates on a uniform
random subset, which makes A1 true by construction. Here calibration and test are
divided by a REAL key instead -- flights has 38 distinct `src` values of roughly 100
cells each -- so the certifier calibrates on some sources and is deployed on others it
has never seen. Nothing else changes: same log, same proposer, same budget.

Two arms:

  random   the paper's protocol. A1 holds. Coverage should hold.
  drift    split by the key column (default `src`), so calibration and deployment
           come from disjoint sources. A1 is violated and nothing in the certifier
           can see it. Coverage MAY fail.

Read the arms together. `drift` failing while `random` holds is the informative
outcome: it locates the failure in the assumption rather than the method. `drift`
NOT failing is also worth knowing -- it would mean this dataset's sources are close
enough to exchangeable that the split does not test anything, and a stronger key is
needed before the experiment is reportable.

    python audit_drift_probe.py
    -> experiments/drift_probe.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import random
import statistics
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _norm(v):
    return str(v).strip().lower() if v is not None else ""


def main() -> int:
    from bench.datasets import load
    from bench.study import load_and_propose
    from care.conformal import by_cardinality, by_column
    from care.conformal.rcps import rcps_threshold
    from care.core.artifact import CellKey as _CK

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="flights")
    ap.add_argument("--log", default="csvs/bclean_flights_mapped.csv")
    ap.add_argument("--tag", default=None,
                    help="backend tag; inferred from the log name if omitted")
    ap.add_argument("--key", default="src", help="column whose value groups the rows")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.1, 0.2])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="experiments/drift_probe.csv")
    args = ap.parse_args()
    os.chdir(ROOT)

    art, defects, _ = load(args.dataset)
    gold = {k: _norm(v) for k, v in defects.gold.items()}
    # A degenerate proposer cannot show a coverage failure: Baran is exactly correct
    # on flights, so every arm reports zero error whatever the split does. The probe
    # needs a proposer that errs, or it measures nothing.
    tag = args.tag or os.path.basename(args.log).split("_")[0]
    _art, _d, repairs, _r = load_and_propose(
        args.dataset, f"{tag}:{args.log}", detection="oracle", fast_verify=True)

    # row -> group, from the key column's value in the dirty table
    grp = {}
    for c in art.iter_cells():
        if c.col == args.key:
            grp[str(c.row_id)] = _norm(c.value)
    groups = sorted(set(grp.values()))
    if len(groups) < 4:
        raise SystemExit(f"{args.key} has only {len(groups)} distinct values; a split "
                         f"by it cannot separate calibration from deployment")

    refs = [r for r in repairs if r in gold]
    ok = {r: (_norm(repairs[r].proposed_value) == gold[r]) for r in refs}
    counts = collections.Counter(_CK.parse(r).col for r in gold)
    median = sorted(counts.values())[len(counts) // 2] if counts else 0
    sf = by_column() if median >= 60 else by_cardinality(art)

    def evaluate(cal, test, alpha):
        """Calibrate per stratum on `cal`, decide `test`, return realised error."""
        by = collections.defaultdict(list)
        for r in cal:
            by[sf(r)].append(r)
        thr = {}
        for st, rs in by.items():
            lam, _ = rcps_threshold([repairs[r].s_hat for r in rs],
                                    [ok[r] for r in rs],
                                    alpha=alpha, delta=args.delta, bound="exact")
            thr[st] = lam
        sel = [r for r in test if repairs[r].s_hat >= thr.get(sf(r), float("inf"))]
        if not sel:
            return None, 0.0
        return 1.0 - sum(ok[r] for r in sel) / len(sel), len(sel) / max(len(test), 1)

    rows = []
    for alpha in args.alphas:
        for arm in ("random", "drift"):
            errs, autos, viol = [], [], 0
            for s in range(args.seeds):
                rng = random.Random(s)
                if arm == "random":
                    sh = refs[:]
                    rng.shuffle(sh)
                    k = int(0.4 * len(sh))
                    cal, test = sh[:k], sh[k:]
                else:
                    g = groups[:]
                    rng.shuffle(g)
                    cut = max(1, int(0.4 * len(g)))
                    cal_g = set(g[:cut])
                    cal = [r for r in refs if grp.get(_CK.parse(r).row_id) in cal_g]
                    test = [r for r in refs if grp.get(_CK.parse(r).row_id) not in cal_g]
                if not cal or not test:
                    continue
                e, a = evaluate(cal, test, alpha)
                autos.append(a)
                if e is not None:
                    errs.append(e)
                    viol += e > alpha + 1e-12
            rows.append({
                "dataset": args.dataset, "key": args.key, "arm": arm,
                "alpha": alpha, "delta": args.delta, "seeds": len(autos),
                "violations": viol,
                "mean_error": round(statistics.mean(errs), 4) if errs else None,
                "max_error": round(max(errs), 4) if errs else None,
                "mean_automation": round(statistics.mean(autos), 4) if autos else 0.0,
            })

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{args.dataset}, split by {args.key} ({len(groups)} groups), "
          f"{args.seeds} seeds\n")
    print(f"{'alpha':>6} {'arm':<8} {'violations':>11} {'mean err':>9} "
          f"{'max err':>8} {'automation':>11}")
    for r in rows:
        me = f"{r['mean_error']:.3f}" if r["mean_error"] is not None else "-"
        xe = f"{r['max_error']:.3f}" if r["max_error"] is not None else "-"
        print(f"{r['alpha']:>6} {r['arm']:<8} {r['violations']:>4}/{r['seeds']:<6} "
              f"{me:>9} {xe:>8} {r['mean_automation']:>11.3f}")
    print(f"\nwrote {args.out}")
    rv = sum(r["violations"] for r in rows if r["arm"] == "random")
    dv = sum(r["violations"] for r in rows if r["arm"] == "drift")
    print(f"\nrandom split: {rv} violations   drift split: {dv} violations")
    if dv > rv:
        print("The non-exchangeable split degrades coverage: A1 is load-bearing, and "
              "the\nexperiment is worth reporting.")
    else:
        print("The split did not degrade coverage. Either these sources are close to\n"
              "exchangeable, or the automation is too low for a violation to be "
              "possible.\nCheck the automation column before concluding anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
