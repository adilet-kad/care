"""audit_label_budget.py -- how many labelled cells does certification actually cost?

Assumption A1 asks for a clean calibration set, and whether labelling one is
affordable on a dirty enterprise table is the practical question. Section 5 answers
that with an inequality, n >~ ln(|Lambda|/delta)/alpha labels per stratum. This script
answers it with a measurement: hold everything else fixed, vary only the number of
labelled cells an operator is willing to pay for, and plot what certifiable
automation that buys.

The model is the operator's, not the statistician's. Given a budget of n labels,
we draw n cells uniformly at random from the work queue, calibrate on those, and
decide every remaining cell. The draw is deliberately not stratified-proportional:
somebody labelling 50 cells across a 17-column table cannot guarantee each column
its quota, and columns that draw too few labels should fail to certify. That
failure is the quantity of interest.

Two things the curve shows that the inequality alone does not. Where it saturates
-- the label count beyond which more labelling buys nothing, because automation has
reached the pi_0 ceiling set by the proposer rather than by the calibration data.
And how steeply it climbs before then, which is what tells an operator whether the
first afternoon of labelling is worth spending.

    python audit_label_budget.py
    -> experiments/label_budget.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

DEFAULT_BUDGETS = (25, 50, 100, 200, 400, 800, 1600)

# (proposer tag, dataset, log). Chosen to span the two regimes the inequality
# predicts: a table whose strata are large enough to calibrate cheaply, and one
# whose strata are thin enough that the multiplicity penalty bites.
CONFIGS = [
    ("Baran", "hospital", "csvs/baran_hospital_mapped.csv"),
    ("Baran", "beers", "csvs/baran_beers_mapped.csv"),
    ("BClean", "flights", "csvs/bclean_flights_mapped.csv"),
]


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench.study import evaluate_split, load_and_propose
    from care.conformal import by_cardinality, by_column, marginal_strata
    from care.core.artifact import CellKey as _CK

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    ap.add_argument("--only", nargs="*", default=None, help="restrict to dataset names")
    ap.add_argument("--out", default="experiments/label_budget.csv")
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
        truth = defects.gold
        counts = _c.Counter(_CK.parse(r).col for r in truth)
        median = sorted(counts.values())[len(counts) // 2] if counts else 0
        mondrian = by_column() if median >= 60 else by_cardinality(art)

        for mode, sf in (("mondrian", mondrian), ("marginal", marginal_strata)):
            for n in args.budgets:
                if n >= len([r for r in truth if r in repairs]):
                    continue
                autos, errs = [], []
                for s in range(args.seeds):
                    sc = evaluate_split(repairs, truth, alpha=args.alpha,
                                        delta=args.delta, seed=s, strata_fn=sf,
                                        cal_size=n)
                    autos.append(1.0 - sc["CARE"].human_cost)
                    errs.append(sc["CARE"].realized_error)
                rows.append({
                    "proposer": tag, "dataset": ds, "strata": mode,
                    "alpha": args.alpha, "labels": n,
                    "automation": round(statistics.mean(autos), 4),
                    "automation_sd": round(statistics.stdev(autos), 4) if len(autos) > 1 else 0.0,
                    "realized_error": round(statistics.mean(errs), 4),
                    "seeds": args.seeds,
                })
                print(f"  {mode:<9} labels={n:<5} A={rows[-1]['automation']:.3f} "
                      f"+-{rows[-1]['automation_sd']:.3f}  R={rows[-1]['realized_error']:.3f}",
                      flush=True)

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
