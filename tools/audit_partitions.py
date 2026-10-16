"""Strata partitions (column, type, cardinality, planner): automation, labels and validity.

Run on hospital and rayyan for Baran and GIDCL over 50 splits. The planner pools columns
using a 20-label pilot per column; pilot cells count as labels and as reviewed repairs.

    python tools/audit_partitions.py   -> experiments/partitions.csv
"""
from __future__ import annotations

import csv
import os
import random
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
os.environ.setdefault("CARE_PROGRESS", "0")

CONFIGS = [("baran", "hospital"), ("baran", "rayyan"), ("gidcl", "hospital"),
           ("gidcl", "rayyan")]
ALPHAS = [0.05, 0.1, 0.2]
DELTA, SEEDS, PILOT = 0.1, 50, 20
OUT = "experiments/partitions.csv"


def _norm(v):
    return str(v).lower().strip() if v is not None else None


def main():
    from bench.study import evaluate_split, load_and_propose
    from care.conformal import by_cardinality, by_column, by_type
    from care.conformal.planner import plan
    from care.core.artifact import CellKey

    rows = []
    for tag, ds in CONFIGS:
        art, defects, repairs, _ = load_and_propose(
            ds, f"{tag}:csvs/{tag}_{ds}_mapped.csv", detection="oracle", fast_verify=True)
        gold = defects.gold
        refs = [r for r in gold if r in repairs]
        total = len(refs)
        col = by_column()
        fixed = {"column": col, "type": by_type(art), "cardinality": by_cardinality(art)}
        print(f"\n== {tag}/{ds}: {total} repairs", flush=True)
        for a in ALPHAS:
            for name in ["column", "type", "cardinality", "planner"]:
                auto, R, over, labels, units = [], [], 0, [], []
                for s in range(SEEDS):
                    reps, sf, pilot_n = repairs, fixed.get(name), 0
                    if name == "planner":
                        rng = random.Random(1_000_003 + s)
                        by_col = {}
                        for r in refs:
                            by_col.setdefault(col(r), []).append(r)
                        beliefs, pilot = {}, set()
                        for c, members in sorted(by_col.items()):
                            k = min(PILOT, len(members))
                            draw = rng.sample(members, k)
                            pilot.update(draw)
                            wrong = sum(_norm(repairs[r].proposed_value) != _norm(gold[r])
                                        for r in draw)
                            beliefs[c] = (len(members) - k, (1 + wrong, 1 + k - wrong))
                        p = plan(beliefs, alpha=a, delta=DELTA, kappa=1.0)
                        sf = p.strata_fn(col)
                        units.append(sum(1 for u in p.units if len(u.strata) > 1))
                        reps = {r: v for r, v in repairs.items() if r not in pilot}
                        pilot_n = len(pilot)
                    sc, dec = evaluate_split(reps, gold, alpha=a, delta=DELTA, seed=s,
                                             strata_fn=sf, return_decisions=True)
                    c = sc["CARE"]
                    auto.append(c.n_auto / total)
                    R.append(c.realized_error)
                    labels.append(len(dec["cal_refs"]) + pilot_n)
                    if c.n_auto:
                        over += c.realized_error > a + 1e-12
                ngroups = len({sf(r) for r in refs}) if name != "planner" else None
                row = {"cleaner": tag, "dataset": ds, "alpha": a, "partition": name,
                       "groups": ngroups, "written_share": round(statistics.mean(auto), 4),
                       "labels": round(statistics.mean(labels)),
                       "realized_error": round(statistics.mean(R), 4),
                       "over_alpha": over, "seeds": SEEDS,
                       "pooled_units": round(statistics.mean(units), 2) if units else None}
                rows.append(row)
                print(f"  alpha={a:<4} {name:>11}: written {row['written_share']:.3f} "
                      f"labels {row['labels']:>4}  R {row['realized_error']:.3f}  "
                      f"over-alpha {over}/{SEEDS}"
                      + (f"  groups {ngroups}" if ngroups else ""), flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
