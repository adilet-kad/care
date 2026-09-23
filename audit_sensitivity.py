"""audit_sensitivity.py -- is the guarantee sensitive to |Lambda| and delta?

Two knobs in CARE are not derived from the data: the size of the fixed candidate grid
|Lambda|, and the confidence level delta. If the reported automation were an artifact
of tuning them the result would not mean much, so this sweeps both over the existing
logs and reports what moves.

What should happen, if the construction is sound:

  * COVERAGE is insensitive. The guarantee is one-sided and holds for any grid and any
    delta; changing them must not produce a budget violation. If it does, the
    Bonferroni accounting is wrong.
  * AUTOMATION moves, mildly and in a predictable direction. A larger grid searches
    more thresholds and pays a larger delta/|Lambda| penalty per test; a larger delta
    buys a looser bound and more automation. Neither should swing the qualitative
    picture -- a proposer that certifies nothing should not start certifying half its
    repairs because the grid grew.

So the result to report is the RANGE of automation across the sweep next to a zero
violation count, not a single number. That answers "did you tune this?" with evidence
rather than assurance.

    python audit_sensitivity.py
    -> experiments/sensitivity.csv

Note on |Lambda|: `_candidate_grid(G)` for an integer G returns G+1 thresholds, so we
pass G = |Lambda| - 1 to land on the requested cardinality exactly.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

CONFIGS = [
    ("Baran", "hospital", "csvs/baran_hospital_mapped.csv"),
    ("Baran", "beers", "csvs/baran_beers_mapped.csv"),
    ("BClean", "flights", "csvs/bclean_flights_mapped.csv"),
    ("Jellyfish", "beers", "csvs/jellyfish_beers_mapped.csv"),
]
LAMBDAS = [5, 17, 50]
DELTAS = [0.05, 0.1, 0.2]


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench.study import evaluate_split, load_and_propose
    from care.conformal import by_cardinality, by_column
    from care.core.artifact import CellKey as _CK

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--lambdas", nargs="*", type=int, default=LAMBDAS)
    ap.add_argument("--deltas", nargs="*", type=float, default=DELTAS)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="experiments/sensitivity.csv")
    args = ap.parse_args()

    import collections as _c

    rows = []
    for tag, ds, log in CONFIGS:
        if args.only and ds not in args.only:
            continue
        if not os.path.exists(log):
            print(f"  [skip] {tag}/{ds}: missing {log}")
            continue
        print(f"\n===== {tag} / {ds}  (alpha={args.alpha}) =====", flush=True)
        art, defects, repairs, _ = load_and_propose(
            ds, f"{tag.lower()}:{log}", detection="oracle", fast_verify=True)
        gold = {k: (str(v).lower().strip() if v is not None else v)
                for k, v in defects.gold.items()}
        counts = _c.Counter(_CK.parse(r).col for r in gold)
        median = sorted(counts.values())[len(counts) // 2] if counts else 0
        sf = by_column() if median >= 60 else by_cardinality(art)

        print(f"  {'|Lambda|':>9} {'delta':>6} {'automation':>11} {'realised err':>13} "
              f"{'violations':>11}")
        for L in args.lambdas:
            for d in args.deltas:
                autos, errs, viol = [], [], 0
                for s in range(args.seeds):
                    sc = evaluate_split(repairs, gold, alpha=args.alpha, delta=d,
                                        seed=s, strata_fn=sf, grid=L - 1)["CARE"]
                    a = 1.0 - sc.human_cost
                    autos.append(a)
                    if a > 0:
                        errs.append(sc.realized_error)
                        viol += sc.realized_error > args.alpha + 1e-12
                ma = statistics.mean(autos)
                me = statistics.mean(errs) if errs else None
                print(f"  {L:>9} {d:>6} {ma:>11.3f} "
                      f"{(f'{me:.3f}' if me is not None else '-'):>13} "
                      f"{viol:>4}/{args.seeds:<6}")
                rows.append({
                    "proposer": tag, "dataset": ds, "alpha": args.alpha,
                    "n_lambda": L, "delta": d, "seeds": args.seeds,
                    "automation": round(ma, 4),
                    "realized_error": round(me, 4) if me is not None else None,
                    "violations": viol,
                })

    if not rows:
        print("no configurations ran")
        return 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tv = sum(r["violations"] for r in rows)
    n = sum(r["seeds"] for r in rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    print(f"violations across the whole sweep: {tv}/{n}")
    by_pair = {}
    for r in rows:
        by_pair.setdefault((r["proposer"], r["dataset"]), []).append(r["automation"])
    print("automation range per configuration (min -> max over the 9 settings):")
    for (p, d), v in by_pair.items():
        print(f"  {p+'/'+d:<20} {min(v):.3f} -> {max(v):.3f}   spread {max(v)-min(v):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
