"""Import Ni et al. (PVLDB 2024) controlled-noise variants as CARE datasets.

A synthetic 1-based `index` column is prepended so the loader does not take the first
attribute as the row key. Pairs whose clean and dirty files differ on more than
--max-rate of cells (Ni's flights outer files) are refused. The level in a variant's
name is not its cell error rate; the manifest records the measured rate.

    python tools/import_ni_variants.py --adr /tmp/adr [--raha-datasets <raha>/datasets]
    python tools/import_ni_variants.py --verify
    -> data/<ds>_ni_<variant>/{dirty,clean}.csv, data/ni_variants.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same as bench.datasets.NULL_TOKENS, so error counts match defects.gold.
NULL_TOKENS = {"", "empty", "nan", "null", "n/a", "na", "?"}

DATASETS = ("beers", "hospital", "rayyan", "flights")


def _norm(v):
    s = str(v).strip()
    return None if s.lower() in NULL_TOKENS else s


def _read(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise SystemExit(f"empty file: {path}")
    return rows[0], rows[1:]


def _write_indexed(path, header, rows):
    """Write with a synthetic 1-based `index` column prepended."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index"] + header)
        for i, r in enumerate(rows, 1):
            w.writerow([i] + r)


def _diff(drows, crows, ncols):
    n = 0
    for a, b in zip(drows, crows):
        for j in range(ncols):
            x = _norm(a[j]) if j < len(a) else None
            y = _norm(b[j]) if j < len(b) else None
            if x != y:
                n += 1
    return n


def variant_files(adr, ds, kinds, levels):
    """(variant_name, path) for every requested file that exists on disk."""
    base = os.path.join(adr, "data_with_rules", ds)
    out = []
    for kind in kinds:
        if kind == "orig":
            # inner_outer-01 matches the original dirty table; used as the anchor
            out.append(("orig", os.path.join(base, "noise", f"{ds}-inner_outer_error-01.csv")))
            continue
        for lv in levels:
            out.append((f"{kind}{lv}",
                        os.path.join(base, "noise", f"{ds}-{kind}_error-{lv}.csv")))
    return [(v, p) for v, p in out if os.path.exists(p)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--adr", default="/tmp/adr",
                    help="clone of github.com/WelkinNi/Automatic-Data-Repair")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--raha-datasets", default=None,
                    help="also mirror each variant into Raha's datasets/ tree, which "
                         "export_baran_raha.py and export_detection.py read")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--kinds", nargs="*", default=["inner", "outer", "orig"])
    ap.add_argument("--levels", nargs="*", type=int, default=[10, 20, 30, 50, 70])
    ap.add_argument("--max-rate", type=float, default=0.5,
                    help="refuse a clean/dirty pair differing on more than this "
                         "fraction of cells (catches Ni's flights outer files)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--force", action="store_true", help="overwrite existing variants")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.adr, "data_with_rules")):
        raise SystemExit(
            f"{args.adr}/data_with_rules not found. Clone it first:\n"
            f"  git clone --depth 1 https://github.com/WelkinNi/Automatic-Data-Repair.git {args.adr}")

    manifest = args.manifest or os.path.join(args.data_dir, "ni_variants.csv")
    rows, refused = [], []

    for ds in args.datasets:
        cpath = os.path.join(args.adr, "data_with_rules", ds, "clean.csv")
        if not os.path.exists(cpath):
            print(f"[skip] {ds}: no clean.csv in the clone")
            continue
        chead, crows = _read(cpath)

        for var, dpath in variant_files(args.adr, ds, args.kinds, args.levels):
            name = f"{ds}_ni_{var}"
            dhead, drows = _read(dpath)

            # the pair must be aligned before cells are compared
            if dhead != chead:
                refused.append((name, "header differs from clean.csv"))
                continue
            if len(drows) != len(crows):
                refused.append((name, f"{len(drows)} rows vs clean's {len(crows)}"))
                continue

            ncols = len(chead)
            d = _diff(drows, crows, ncols)
            cells = len(crows) * ncols
            rate = d / cells
            if rate > args.max_rate:
                refused.append((name, f"{rate:.3f} of cells differ from clean.csv "
                                      f"-- broken pair, not a noisy one"))
                continue

            folder = os.path.join(args.data_dir, name)
            if os.path.exists(os.path.join(folder, "dirty.csv")) and not args.force:
                print(f"[keep] {name} already imported (--force to overwrite)")
            else:
                _write_indexed(os.path.join(folder, "clean.csv"), chead, crows)
                _write_indexed(os.path.join(folder, "dirty.csv"), chead, drows)
            if args.raha_datasets:
                rf = os.path.join(args.raha_datasets, name)
                if not os.path.exists(os.path.join(rf, "dirty.csv")) or args.force:
                    _write_indexed(os.path.join(rf, "clean.csv"), chead, crows)
                    _write_indexed(os.path.join(rf, "dirty.csv"), chead, drows)

            kind = "orig" if var == "orig" else "".join(c for c in var if c.isalpha())
            level = "".join(c for c in var if c.isdigit()) or "0"
            rows.append({"dataset": name, "base": ds, "kind": kind, "level": int(level),
                         "rows": len(crows), "cols": ncols - 0, "cells": cells,
                         "error_cells": d, "cell_error_rate": round(rate, 5),
                         "source": os.path.relpath(dpath, args.adr)})

    if not rows:
        print("nothing imported")
        return 1

    rows.sort(key=lambda r: (r["base"], r["kind"], r["level"]))
    os.makedirs(os.path.dirname(manifest) or ".", exist_ok=True)
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'variant':<26}{'rows':>7}{'cols':>6}{'err cells':>11}{'rate':>8}")
    for r in rows:
        print(f"{r['dataset']:<26}{r['rows']:>7}{r['cols']:>6}"
              f"{r['error_cells']:>11}{r['cell_error_rate']:>8.3f}")
    print(f"\nwrote {manifest} ({len(rows)} variants)")
    if refused:
        print("\nREFUSED (deliberately -- see the module docstring):")
        for name, why in refused:
            print(f"  {name:<26} {why}")
    print("\nNow verify the loader agrees with these counts:")
    print("  python tools/import_ni_variants.py --verify")
    return 0


def verify(manifest=None) -> int:
    """Load every imported variant and check its gold count, rows and columns against the manifest."""
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from bench.datasets import load

    manifest = manifest or os.path.join("data", "ni_variants.csv")
    if not os.path.exists(manifest):
        raise SystemExit(f"no manifest at {manifest}; run the import first")
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh))

    bad = 0
    print(f"{'variant':<26}{'expected':>10}{'gold':>8}{'cols':>6}  verdict")
    for r in rows:
        art, defects, cols = load(r["dataset"])
        keys = {c.row_id for c in art.iter_cells()}
        exp, got = int(r["error_cells"]), len(defects.gold)
        ok = exp == got and len(keys) == int(r["rows"]) and len(cols) == int(r["cols"])
        bad += not ok
        note = "ok" if ok else (
            f"MISMATCH (rows keyed {len(keys)}/{r['rows']}, cols {len(cols)}/{r['cols']})")
        print(f"{r['dataset']:<26}{exp:>10}{got:>8}{len(cols):>6}  {note}")
    print(f"\n{len(rows) - bad}/{len(rows)} variants load exactly as imported")
    if bad:
        print("A mismatch is an import bug. Do not run sweeps until it is zero.")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        raise SystemExit(verify())
    raise SystemExit(main())
