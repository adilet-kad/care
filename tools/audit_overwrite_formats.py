"""Overwrites of correct cells by a cleaner: format-only variants vs semantic changes.

An overwrite changes a cell whose dirty value already equals the clean value. It is
format-only when it is numerically equal to the clean value or dateutil parses both
to the same date and time.

    python tools/audit_overwrite_formats.py rayyan csvs/gidcl_rayyan_mapped.csv
    python tools/audit_overwrite_formats.py --all   -> experiments/overwrite_formats.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

csv.field_size_limit(sys.maxsize)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rows(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        r = list(csv.reader(fh))
    return r[0], r[1:]


def _n(v):
    return (v or "").strip().lower()


PAIRS = [("BClean", "bclean", "beers"), ("BClean", "bclean", "flights"),
         ("BClean+", "bclean_plus", "beers"), ("BClean+", "bclean_plus", "flights"),
         ("GIDCL", "gidcl", "beers"), ("GIDCL", "gidcl", "flights"),
         ("GIDCL", "gidcl", "rayyan"), ("GIDCL", "gidcl", "tax"),
         ("HoloClean", "holoclean", "flights")]


def _overwrites(dataset, log):
    dh, dr = _rows(os.path.join(_ROOT, "data", dataset, "dirty.csv"))
    ch, cr = _rows(os.path.join(_ROOT, "data", dataset, "clean.csv"))
    first = dh[0].strip().lower()
    idx = first in {"index", "tid", "id", "row_id", "key", ""} or \
        all(r[0].strip().lstrip("-").isdigit() for r in cr[:200])
    dirty, clean = {}, {}
    for i, (r, c) in enumerate(zip(dr, cr)):
        key = r[0] if idx else str(i)
        for j, h in enumerate(dh):
            dirty[(key, h)] = r[j] if j < len(r) else ""
            clean[(key, h)] = c[j] if j < len(c) else ""
    ow = []
    with open(log, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            k = (r["row_id"], r["column"])
            if k in dirty and _n(dirty[k]) == _n(clean[k]) and _n(r["value"]) != _n(clean[k]):
                ow.append((k, r["value"], clean[k]))
    return ow


def _format_only(v, c):
    import warnings
    from dateutil import parser as du
    try:
        if float(_n(v)) == float(_n(c)):
            return True
    except ValueError:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return du.parse(v) == du.parse(c)
        except Exception:
            return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", nargs="?")
    ap.add_argument("log", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="experiments/overwrite_formats.csv")
    a = ap.parse_args()
    if a.all:
        rows = []
        for name, tag, ds in PAIRS:
            log = os.path.join(_ROOT, "csvs", f"{tag}_{ds}_mapped.csv")
            if not os.path.exists(log):
                continue
            ow = _overwrites(ds, log)
            f = sum(_format_only(v, c) for _, v, c in ow)
            rows.append({"cleaner": name, "dataset": ds, "overwrites": len(ow),
                         "format_only": f, "semantic": len(ow) - f})
            print(f"  {name:9s} {ds:8s} overwrites {len(ow):4d}  format-only {f:3d}")
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}")
        return
    import pandas as pd
    from dateutil import parser as du

    dh, dr = _rows(os.path.join(_ROOT, "data", a.dataset, "dirty.csv"))
    ch, cr = _rows(os.path.join(_ROOT, "data", a.dataset, "clean.csv"))
    dirty = {(r[0], h): r[j] for r in dr for j, h in enumerate(dh)}
    clean = {(r[0], h): r[j] for r in cr for j, h in enumerate(ch)}
    ow = []
    with open(a.log, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            k = (r["row_id"], r["column"])
            if k in dirty and _n(dirty[k]) == _n(clean[k]) and _n(r["value"]) != _n(clean[k]):
                ow.append((k, r["value"], clean[k]))
    print(f"{len(ow)} overwrites of correct cells:",
          dict(collections.Counter(k[1] for k, _, _ in ow)))

    def parse(fn, v):
        try:
            return fn(v).date()
        except Exception:
            return None

    by_col = collections.defaultdict(list)
    for k, v, c in ow:
        by_col[k[1]].append((v, c))
    for col, pairs in by_col.items():
        for name, fn in (("dateutil", du.parse), ("pandas", pd.to_datetime)):
            res = collections.Counter()
            for v, c in pairs:
                pv, pc = parse(fn, v), parse(fn, c)
                res["unparseable" if pv is None else
                    ("same date" if pv == pc else "different date")] += 1
            print(f"  {col:24s} {name:8s} {dict(res)}")
        print(f"  {col:24s} e.g. " + "; ".join(f"{c} -> {v}" for v, c in pairs[:3]))


if __name__ == "__main__":
    main()
