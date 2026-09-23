"""Per-column accuracy of an LLM proposer at each self-consistency level, on error cells.

Section 7.2 ("What would lift the LLM rows") reads three numbers off the shipped
Jellyfish logs: on beers the `state` column is 0.991 accurate at full agreement over
115 error cells while `abv` is 0.744 over 387, and on hospital four columns are
0.885-1.000 accurate at full agreement over 25-32 cells each. This prints those
tallies from the log and the gold labels alone, so the sentence can be checked
without re-running the certifier.

    python tools/agreement_strata.py beers
    python tools/agreement_strata.py hospital --min-cells 20

Only error cells are counted (the Table 1 population). `--level` restricts the
report to one agreement value (default: the maximum in the log).
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _n(v):
    return str(v).strip().lower() if v is not None else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--log", default=None, help="default csvs/jellyfish_<dataset>_mapped.csv")
    ap.add_argument("--level", type=float, default=None, help="agreement level (default: max)")
    ap.add_argument("--min-cells", type=int, default=20)
    a = ap.parse_args()
    from bench.datasets import load
    _art, defects, _ = load(a.dataset)
    gold = {k: _n(v) for k, v in defects.gold.items()}
    path = a.log or f"csvs/jellyfish_{a.dataset}_mapped.csv"
    tally = collections.defaultdict(lambda: [0, 0])      # (column, level) -> [n, correct]
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            ref = f"{r['row_id']}::{r['column']}"
            if ref not in gold:
                continue
            try:
                lvl = round(float(r["confidence"]), 4)
            except (KeyError, TypeError, ValueError):
                continue
            t = tally[(r["column"], lvl)]
            t[0] += 1
            t[1] += (_n(r.get("value")) == gold[ref])
    levels = sorted({k[1] for k in tally})
    level = a.level if a.level is not None else levels[-1]
    print(f"{a.dataset}: {path}; agreement levels {levels}; reporting level {level:g}, "
          f"columns with >= {a.min_cells} error cells at that level")
    rows = [(col, n, k / n) for (col, lvl), (n, k) in tally.items()
            if abs(lvl - level) < 1e-9 and n >= a.min_cells]
    for col, n, acc in sorted(rows, key=lambda x: -x[2]):
        print(f"  {col:<20s} n={n:<6d} acc={acc:.3f}")
    if not rows:
        print("  (none)")


if __name__ == "__main__":
    main()
