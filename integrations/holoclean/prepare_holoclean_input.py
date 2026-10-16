"""Write a HoloClean-loadable copy of a CARE dirty table.

Drops the index column and checks it is 1..N in order, since export_holoclean.py
maps row_id = _tid_ + 1. Column names are kept as in dirty.csv. Writes
<dataset>_holoclean.csv next to this script (or --out).

    python prepare_holoclean_input.py beers   --care-root ../CARE
    python prepare_holoclean_input.py flights --care-root ../CARE
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

INDEX_COLS = ("index", "tuple_id", "tid", "id_")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset")
    ap.add_argument("--care-root", default="../CARE",
                    help="path to the CARE repo holding data/<dataset>/dirty.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-unordered", action="store_true",
                    help="write even if the index is not 1..N in order (UNSAFE: the "
                         "row mapping in export_holoclean.py assumes it is)")
    args = ap.parse_args()

    src = os.path.join(args.care_root, "data", args.dataset, "dirty.csv")
    if not os.path.exists(src):
        raise SystemExit(f"not found: {src}\nPass --care-root pointing at the CARE repo.")

    with open(src, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{src} is empty")

    header = list(rows[0])
    idx = next((c for c in header if c.lower() in INDEX_COLS), None)
    if idx is None:
        print(f"  no index column found in {header[:3]}...; writing as-is")
    else:
        # the row offset needs a contiguous, ordered index
        bad = None
        for i, r in enumerate(rows, 1):
            try:
                if int(r[idx]) != i:
                    bad = (i, r[idx])
                    break
            except (TypeError, ValueError):
                bad = (i, r[idx])
                break
        if bad and not args.allow_unordered:
            raise SystemExit(
                f"row {bad[0]} has {idx}={bad[1]!r}, so the index is not 1..N in order.\n"
                f"export_holoclean.py maps _tid_ + 1 -> row_id, which would misalign "
                f"every\nrepair. Fix the source or pass --allow-unordered and set "
                f"--row-offset yourself.")
        if bad:
            print(f"  warning: index is not 1..N (row {bad[0]} = {bad[1]!r}); writing anyway "
                  f"because --allow-unordered was given. Check --row-offset.")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"{args.dataset}_holoclean.csv")
    cols = [c for c in header if c != idx]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out}")
    print(f"  {len(rows)} rows, {len(cols)} columns"
          + (f" (dropped index column {idx!r})" if idx else ""))
    print(f"  columns: {cols}")
    print("\nNext, from the HoloClean repo root:")
    print(f"  python export_holoclean.py {args.dataset} \\")
    print(f"    --data {os.path.basename(out)} \\")
    print(f"    --dcs constraints/{args.dataset}_constraints.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
